#include "sketch_policy.h"

#include <tvm/runtime/registry.h>
#include <tvm/support/parallel_for.h>

#include <algorithm>
#include <iomanip>
#include <limits>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "sketch_policy_rules.h"

namespace tvm {
namespace auto_scheduler {

/********** Sketch generation rules **********/
static RuleSkipStage rule_skip_stage;
static RuleAlwaysInline rule_always_inline;
static RuleMultiLevelTiling rule_multi_level_tiling;
static RuleMultiLevelTilingWithFusion rule_multi_level_tiling_with_fusion;
static RuleAddCacheRead rule_add_cache_read_stage;
static RuleAddCacheWrite rule_add_cache_write_stage;
static RuleAddRfactor rule_add_rfactor;
static RuleCrossThreadReduction rule_cross_thread_reduction;
static RuleSimplifyComputeWithConstTensor rule_simplify_compute_with_const_tensor;
static RuleSpecialComputeLocationGPU rule_special_compute_location_gpu;

/********** Init population rules **********/
static InitFillTileSize init_fill_tile_size;
static InitChangeComputeLocation init_change_compute_location;
static InitParallel init_parallel;
static InitUnroll init_unroll;
static InitVectorization init_vectorization;
static InitThreadBind init_thread_bind;

/********** Group Sketch policy **********/
TVM_REGISTER_NODE_TYPE(GroupSketchPolicyNode);

GroupSketchPolicy::GroupSketchPolicy(SearchTaskGroup task_group, GroupCostModel proxy_model, 
                                    CostModel xgb, Map<String, ObjectRef> params, 
                                    int seed, int verbose) {
    auto node = make_object<GroupSketchPolicyNode>();
    node->task_group = std::move(task_group);
    node->proxy_model = std::move(proxy_model);
    node->xgb = std::move(xgb);
    node->params = std::move(params);
    node->rand_gen = std::mt19937(seed);
    node->sample_init_min_pop_ =
        GetIntParam(node->params, SketchParamKey::SampleInitPopulation::min_population);
    
    // TODO init search callbacks(Preload measure states)

    // Sketch Generation Rules
    if (node->task_group->tasks[0]->target->GetAttr<String>("device", "") == "mali") {
        node->sketch_rules.push_back(&rule_always_inline);
        node->sketch_rules.push_back(&rule_simplify_compute_with_const_tensor);
        node->sketch_rules.push_back(&rule_add_rfactor);
        node->sketch_rules.push_back(&rule_add_cache_write_stage);
        node->sketch_rules.push_back(&rule_multi_level_tiling_with_fusion);
        node->sketch_rules.push_back(&rule_multi_level_tiling);
        node->sketch_rules.push_back(&rule_skip_stage);
    } else {
        node->sketch_rules.push_back(&rule_add_cache_read_stage);
        node->sketch_rules.push_back(&rule_special_compute_location_gpu);
        node->sketch_rules.push_back(&rule_always_inline);
        node->sketch_rules.push_back(&rule_simplify_compute_with_const_tensor);
        node->sketch_rules.push_back(&rule_cross_thread_reduction);
        node->sketch_rules.push_back(&rule_add_cache_write_stage);
        node->sketch_rules.push_back(&rule_multi_level_tiling_with_fusion);
        node->sketch_rules.push_back(&rule_multi_level_tiling);
        node->sketch_rules.push_back(&rule_skip_stage);
    }
    // Initial Population Generation Rules
    node->init_rules.push_back(&init_fill_tile_size);
    node->init_rules.push_back(&init_thread_bind);
    node->init_rules.push_back(&init_unroll);
    if (node->task_group->tasks[0]->target->GetAttr<String>("device", "") == "mali") {
        node->init_rules.push_back(&init_vectorization);
    }
    // Mutation Rules for Evolutionary Search
    node->mutation_rules.push_back(std::make_shared<MutateTileSize>(0.90));
    node->mutation_rules.push_back(std::make_shared<MutateAutoUnroll>(0.10));

    data_ = std::move(node);
}

Array<State> GroupSketchPolicyNode::Search(int n_trials, int early_stopping, 
                                int num_measure_per_iter, GroupMeasurer measurer) {
    num_measure_per_iter_ = num_measure_per_iter;

    if (n_trials <= 1) {
        const Array<Array<State> >& best_group_states = SearchOneRound(0);
        ICHECK_GT(best_group_states.size(), 0);
        return best_group_states[0];
    } else {
        // get group workload key
        String group_workload_key;
        for(const auto& task: task_group->tasks){
            group_workload_key = group_workload_key+task->workload_key+"_";
        }
        int num_random = 
            static_cast<int>(GetDoubleParam(params, SketchParamKey::eps_greedy)*num_measure_per_iter);
        early_stopping = early_stopping < 0 ? std::numeric_limits<int>::max() >> 1 : early_stopping;
        measurer->Reset();

        int ct = 0;
        int empty_retry_count = GetIntParam(params, SketchParamKey::empty_retry_count);

        Array<Array<State> > best_group_states, random_group_states;
        Array<Array<MeasureInput> > group_measure_inputs;
        Array<MeasureResult> group_measure_results;
        while (ct<n_trials) {
            if (!group_measure_inputs.empty()) {
                auto t_begin = std::chrono::high_resolution_clock::now();
                // Retrain the cost model before the next search round
                PrintTitle("Train cost model", verbose);
                proxy_model->Update(group_measure_inputs, group_measure_results);
                PrintTimeElapsed(t_begin, "training", verbose);
            }
            PrintTitle("Search", verbose);
            best_group_states = SearchOneRound(num_random*3, &random_group_states);

            // Infer bound. This is necessary for computing the correct ToStr() for redundancy check
            // best states infer bound
            Array<Array<State> > best_group_states_tmp, random_group_states_tmp;
            for(size_t item_id=0; item_id<best_group_states.size(); item_id++) {
                Array<State> group_state_infer;
                for(size_t task_id=0; task_id<task_group->tasks.size(); task_id++) {
                    group_state_infer.push_back(
                        task_group->tasks[task_id]->compute_dag.InferBound(
                                                    best_group_states[item_id][task_id])
                    );
                }
                best_group_states_tmp.push_back(group_state_infer);
            }
            // random states infer bound
            for(size_t item_id=0; item_id<random_group_states.size(); item_id++) {
                Array<State> group_state_infer;
                for(size_t task_id=0; task_id<task_group->tasks.size(); task_id++) {
                    group_state_infer.push_back(
                        task_group->tasks[task_id]->compute_dag.InferBound(
                                                    random_group_states[item_id][task_id])
                    );
                }
                random_group_states_tmp.push_back(group_state_infer);
            }
            best_group_states = best_group_states_tmp;
            random_group_states = random_group_states_tmp;

            group_measure_inputs = PickStatesWithEpsGreedy(best_group_states, random_group_states, n_trials-ct);

            if (group_measure_inputs.empty()) {
                if (empty_retry_count-- > 0) { 
                    continue;
                } else {
                    StdCout(verbose) << "It seems all candidates in the search space have been measured."
                           << std::endl;
                    break;
                }
            } else {
                empty_retry_count = GetIntParam(params, SketchParamKey::empty_retry_count);
            }

            PrintTitle("Measure", verbose);
            group_measure_results = measurer->Measure(task_group, GetRef<GroupSketchPolicy>(this), group_measure_inputs);
            ct += group_measure_inputs.size();

            if (ct-(measurer->best_ct[group_workload_key])>early_stopping 
            && measurer->has_valid.count(group_workload_key)) {
                StdCout(verbose) << "Stop early since no performance improvement in the last "
                        << early_stopping << " measurements trials.\n";
                break;
            }

            for (const auto& res : group_measure_results) {
                measured_states_throughputs_.push_back(1.0 / FloatArrayMean(res->costs));
            }
        }
        PrintTitle("Done", verbose);

        return measurer->best_state[group_workload_key];
    }
}

Array<Array<State> > GroupSketchPolicyNode::SearchOneRound(int num_random_states, Array<Array<State> >* random_states){
    int population = GetIntParam(params, SketchParamKey::EvolutionarySearch::population);
    int num_use_measured = std::min(
        static_cast<int>(measured_states_vector_.size()),
        static_cast<int>(
          GetDoubleParam(params, SketchParamKey::SampleInitPopulation::use_measured_ratio) *
          population));
    
    if (cache_empty_) {
        for(size_t task_id=0; task_id<task_group->tasks.size(); task_id++){
            sketch_cache_.push_back(GenerateSketchesForTask(task_id));
        }
        cache_empty_=false;
    }

    Array<Array<State> > init_population = SampleInitPopulationForGroup();

    std::vector<int> indices = Argsort(measured_states_throughputs_);
    for (int i = 0; i < num_use_measured; i++) {
        init_population.push_back(measured_states_vector_[indices[i]]);
    }
    if (num_random_states > 0 && random_states != nullptr) {
        *random_states = RandomSampleStatesForGroup(init_population, &rand_gen, num_random_states);
    }
    return EvolutionarySearch(init_population, num_measure_per_iter_ * 2);
}

Array<State> GroupSketchPolicyNode::GenerateSketchesForTask(int task_id) {
    const State& init_state = task_group->tasks[task_id]->compute_dag->init_state;

    Array<State> states_buf1{init_state}, states_buf2;
    Array<State>* pnow = &states_buf1;
    Array<State>* pnext = &states_buf2;

    std::unordered_map<State, int, ObjectHash, ObjectEqual> cur_stage_id_map;
    cur_stage_id_map[init_state] = static_cast<int>(init_state->stages.size()) - 1;

    Array<State> out_states;
    while (!pnow->empty()) {
        pnext->clear();
        for (const State& state : *pnow) {
            int stage_id = cur_stage_id_map[state];

            if (stage_id < 0) {
                out_states.push_back(state);
                continue;
            }

            for (const auto& rule : sketch_rules) {
                auto cond = rule->MeetCondition(*this, task_id, state, stage_id);
                if (cond != SketchGenerationRule::ConditionKind::kSkip) {
                    for (const auto& pair : rule->Apply(*this, task_id, state, stage_id)) {
                        cur_stage_id_map[pair.first] = pair.second;
                        pnext->push_back(pair.first);
                    }
                    if (cond == SketchGenerationRule::ConditionKind::kApplyAndSkipRest) {
                        break;
                    }
                }
            }
        }
        std::swap(pnow, pnext);
    }

    for (size_t i = 0; i < out_states.size(); ++i) {
        auto state = out_states[i];
        auto pstate = state.CopyOnWrite();
        for (size_t step_id = 0; step_id < pstate->transform_steps.size(); ++step_id) {
            if (pstate->transform_steps[step_id]->IsInstance<RfactorStepNode>()) {
                ICHECK_GE(step_id, 1);
                int split_step_id = static_cast<int>(step_id - 1);
                auto step = pstate->transform_steps[split_step_id].as<SplitStepNode>();
                ICHECK(step != nullptr);
                pstate->transform_steps.Set(split_step_id, 
                    SplitStep(step->stage_id, step->iter_id, step->extent, 
                            {NullOpt},step->inner_to_outer));
            }
        }
        out_states.Set(i, std::move(state));
    }
    StdCout(verbose) << "Generate Sketches\t\t#s: " << out_states.size() << std::endl;
    return out_states;
}

Array<State> GroupSketchPolicyNode::SampleInitPopulationForTask(int task_id) {
    const Array<State>& sketches = sketch_cache_[task_id];
    int population = GetIntParam(params, SketchParamKey::EvolutionarySearch::population);

    auto tic_begin = std::chrono::high_resolution_clock::now();

    int fail_ct = 0;
    Array<State> out_states;
    std::vector<std::mt19937> rand_gens;
    rand_gens.reserve(population);
    for (int i = 0; i < population; i++) {
        rand_gens.push_back(std::mt19937(rand_gen()));
    }

    std::unordered_set<std::string> explored_state_strs;
    size_t iter = 1;
    size_t unchange_cnt = 0;
    while (static_cast<int>(out_states.size()) < sample_init_min_pop_)  {
        std::vector<State> temp_states(population);

        support::parallel_for(0, population, [this, &temp_states, &sketches, &rand_gens, &task_id](int index) {
            State tmp_s = sketches[(rand_gens[index])() % sketches.size()];
            bool valid = true;
            for (const auto& rule : init_rules) {
                if (rule->Apply(this, task_id, &tmp_s, &rand_gens[index]) == 
                    PopulationGenerationRule::ResultKind::kInvalid) {
                    valid = false;
                    break;
                }
            }
            if (valid) { temp_states[index] = std::move(tmp_s); }
        });

        Array<State> cand_states;
        for (auto tmp_s : temp_states) {
            if (tmp_s.defined()) {cand_states.push_back(std::move(tmp_s));}
            else {fail_ct++;}
        }
        unchange_cnt++;
        if (!cand_states.empty()) {
            std::vector<float> pop_scores;
            pop_scores.reserve(cand_states.size());
            cand_states = task_group->tasks[task_id]->compute_dag.InferBound(cand_states);
            PruneInvalidState(task_group->tasks[task_id], &cand_states);
            xgb->Predict(task_group->tasks[task_id], cand_states, &pop_scores);
            for (size_t i = 0; i < cand_states.size(); i++) {
                const auto state_str = cand_states[i].ToStr();
                if (pop_scores[i] > -1e10 && explored_state_strs.count(state_str) == 0) {
                    explored_state_strs.insert(state_str);
                    out_states.push_back(std::move(cand_states[i]));
                    unchange_cnt = 0;
                } else {
                    fail_ct++;
                }
            }
        }
        if (iter % 5 == 0) {
            double duration = std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::high_resolution_clock::now() - tic_begin).count();
            StdCout(verbose) << "Sample Iter: " << iter << std::fixed << std::setprecision(4)
                        << "\t#Pop: " << out_states.size() << "\t#Target: " << sample_init_min_pop_
                        << "\tfail_ct: " << fail_ct << "\tTime elapsed: " << std::fixed
                        << std::setprecision(2) << duration << std::endl;
        }

        if (unchange_cnt == 5) {
            if (sample_init_min_pop_ > 1) {
                sample_init_min_pop_ /= 2;
                StdCout(verbose) << "#Target has been reduced to " << sample_init_min_pop_
                            << " due to too many failures or duplications" << std::endl;
            }
            unchange_cnt = 0;
        }
        iter++;
    }
    double duration = std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::high_resolution_clock::now() - tic_begin).count();
    StdCout(verbose) << "Sample Initial Population\t#s: " << out_states.size()
                    << "\tfail_ct: " << fail_ct << "\tTime elapsed: " << std::fixed
                    << std::setprecision(2) << duration << std::endl;
    return out_states;
}

Array<Array<State> > GroupSketchPolicyNode::SampleInitPopulationForGroup() {
    int population = GetIntParam(params, SketchParamKey::EvolutionarySearch::population);

    auto tic_begin = std::chrono::high_resolution_clock::now();

    int fail_ct = 0;

    Array<Array<State> > out_states;
    // rand generator for group 
    std::vector<std::vector<std::mt19937> > rand_gens;
    rand_gens.reserve(population);
    for (int pop_id = 0; pop_id < population; pop_id++) {
        std::vector<std::mt19937> group_rand_gens;
        for(size_t task_id=0; task_id<task_group->tasks.size(); task_id++) {
            group_rand_gens.push_back(std::mt19937(rand_gen()));
        }
        rand_gens.push_back(group_rand_gens);
    }

    std::unordered_set<std::string> explored_state_strs;
    size_t iter = 1;
    while (static_cast<int>(out_states.size()) < sample_init_min_pop_) {
        Array<Array<State> > tasks_init_pop;
        for(size_t task_id=0; task_id<task_group->tasks.size(); task_id++) {
            tasks_init_pop.push_back(SampleInitPopulationForTask(task_id));
        }

        // std::vector<std::vector<State> > temp_states(population);
        
        // support::parallel_for(0, population, [this, &temp_states, &tasks_init_pop, &rand_gens](int index){
        //     std::vector<State> tmp_group_s;
        //     for(size_t task_id=0; task_id<task_group->tasks.size(); task_id++) {
        //         State tmp_s = tasks_init_pop[task_id][(rand_gens[index][task_id])() % (tasks_init_pop[task_id].size())];
        //         tmp_group_s.push_back(tmp_s);
        //     }
        //     temp_states[index] = std::move(tmp_group_s);
        // });
        Array<Array<State> > temp_states;
        for(int popuid=0;popuid<population;popuid++){
            Array<State> tmp_group_s;
            for(size_t task_id=0; task_id<task_group->tasks.size(); task_id++) {
                State tmp_s = tasks_init_pop[task_id][(rand_gens[popuid][task_id])() % (tasks_init_pop[task_id].size())];
                tmp_group_s.push_back(tmp_s);
            }
            temp_states.push_back(tmp_group_s);
        }

        std::vector<float> pop_scores;
        pop_scores.reserve(temp_states.size());
        proxy_model->Predict(task_group, temp_states, &pop_scores);

        for(size_t i=0; i<temp_states.size(); i++) {
            std::string  group_state_str = "";
            for (size_t task_id=0; task_id<task_group->tasks.size(); task_id++) {
                group_state_str += temp_states[i][task_id].ToStr() + "_";
            }
            // if (explored_state_strs.count(group_state_str) == 0) {
            if (pop_scores[i] > -1e10 && explored_state_strs.count(group_state_str) == 0) {
                explored_state_strs.insert(group_state_str);
                out_states.push_back(std::move(temp_states[i]));
            }
        }
        iter++;
    }
    double duration = std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::high_resolution_clock::now() - tic_begin).count();
    StdCout(verbose) << "Sample Initial Population\t#s: " << out_states.size()
                    << "\tfail_ct: " << fail_ct << "\tTime elapsed: " << std::fixed
                    << std::setprecision(2) << duration << std::endl;
    return out_states;
}

Array<Array<State> > GroupSketchPolicyNode::EvolutionarySearch(
                                        const Array<Array<State> >& init_population, 
                                        int out_size) {
    Array<Array<State> > best_states;
    auto tic_begin = std::chrono::high_resolution_clock::now();

    size_t population = GetIntParam(params, SketchParamKey::EvolutionarySearch::population);
    double mutation_prob = GetDoubleParam(params, SketchParamKey::EvolutionarySearch::mutation_prob);
    int num_iters = GetIntParam(params, SketchParamKey::EvolutionarySearch::num_iters);

    bool is_cost_model_reasonable = !proxy_model->IsInstance<GroupRandomModelNode>();

    if (!is_cost_model_reasonable && num_iters > 2) {
        num_iters = 2;
        StdCout(1) << "GA iteration number has been adjusted to " << num_iters
                        << " due to random cost model" << std::endl;
        // StdCout(verbose) << "GA iteration number has been adjusted to " << num_iters
        //                 << " due to random cost model" << std::endl;
    }

    Array<Array<State> > group_states_buf1{init_population}, group_states_buf2;
    group_states_buf1.reserve(population);
    group_states_buf2.reserve(population);
    Array<Array<State> >* pnow = &group_states_buf1;
    Array<Array<State> >* pnext = &group_states_buf2;

    using StateHeapItem = std::pair<Array<State>, float>;
    auto cmp = [](const StateHeapItem& left, const StateHeapItem& right) {
        return left.second > right.second;
    };
    std::vector<StateHeapItem> heap;
    std::unordered_set<std::string> in_heap(measured_states_set_);
    heap.reserve(out_size);

    std::vector<float> pop_scores;
    std::vector<double> pop_selection_probs;
    float max_score = -1e-10f;
    pop_scores.reserve(population);
    pop_selection_probs.reserve(population);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    int mutation_success_ct, mutation_fail_ct;
    mutation_success_ct = mutation_fail_ct = 0;
    std::vector<float> rule_weights;
    std::vector<double> rule_selection_probs;
    for (const auto& rule : mutation_rules) {
        rule_weights.push_back(rule->weight);
    }
    ComputePrefixSumProb(rule_weights, &rule_selection_probs);

    // // rand generator for group 
    // std::vector<std::vector<std::mt19937> > rand_gens;
    // rand_gens.reserve(population);
    // for (size_t pop_id = 0; pop_id < population; pop_id++) {
    //     std::vector<std::mt19937> group_rand_gens;
    //     for(size_t task_id=0; task_id<task_group->tasks.size(); task_id++) {
    //         group_rand_gens.push_back(std::mt19937(rand_gen()));
    //     }
    //     rand_gens.push_back(group_rand_gens);
    // }

    // Genetic Algorithm
    for (int k = 0; k < num_iters + 1; ++k) {
        // TODO model_predict
        proxy_model->Predict(task_group, *pnow, &pop_scores);

        for (size_t i=0; i<pnow->size(); ++i) {
            const Array<State>& group_state = (*pnow)[i];
            std::string group_state_str = "";
            for(size_t task_id=0; task_id<task_group->tasks.size(); task_id++) {
                group_state_str += group_state[task_id].ToStr() + "_" ;
            }

            if (in_heap.count(group_state_str) == 0) {
                if (static_cast<int>(heap.size()) < out_size) {
                    heap.emplace_back((*pnow)[i], pop_scores[i]);
                    std::push_heap(heap.begin(), heap.end(), cmp);
                    in_heap.insert(group_state_str);
                } else if (pop_scores[i] > heap.front().second) {
                    std::string old_state_str = "";
                    for(size_t task_id=0; task_id<task_group->tasks.size(); task_id++) {
                        old_state_str += (heap.front().first[task_id]).ToStr() + "_";
                    }
                    in_heap.erase(old_state_str);
                    in_heap.insert(group_state_str);

                    std::pop_heap(heap.begin(), heap.end(), cmp);
                    heap.back() = StateHeapItem(group_state, pop_scores[i]);
                    std::push_heap(heap.begin(), heap.end(), cmp);
                }
                if (pop_scores[i] > max_score) {
                    max_score = pop_scores[i];
                }
            }
        }

        if (k % 5 == 0 || k == num_iters) {
            StdCout(verbose) << "GA Iter: " << k;
            if (!heap.empty()) {
                StdCout(verbose) << std::fixed << std::setprecision(4) << "\tMax score: " << max_score
                            << std::fixed << std::setprecision(4)
                            << "\tMin score: " << heap.front().second;
            } else {
                StdCout(verbose) << "\tMax score: N/A\tMin score: N/A";
            }
            StdCout(verbose) << "\t#Pop: " << heap.size() << "\t#M+: " << mutation_success_ct / (k + 1)
                        << "\t#M-: " << mutation_fail_ct / (k + 1) << std::endl;
        }
        if (k == num_iters) {
            break;
        }
        ComputePrefixSumProb(pop_scores, &pop_selection_probs);

        // Do mutation
        while (pnext->size() < population) {
            // 按概率随机选取一个group state
            Array<State> tmp_gs = (*pnow)[RandomChoose(pop_selection_probs, &rand_gen)];
            // 判断该group state是否需要突变
            if (dis(rand_gen) < mutation_prob) {
                // 进行突变
                bool do_mutation = false;
                // 突变后新的group state
                Array<State> new_tmp_gs;
                // 遍历原来的group state，为每个state做mutation
                for(size_t task_id=0; task_id<task_group->tasks.size();task_id++) {
                    State tmp_s = tmp_gs[task_id];
                    // 判断该state是否需要mutation
                    if (dis(rand_gen) < mutation_prob) {
                        //为需要mutation的state随机选择一个rule
                        const auto& rule = mutation_rules[RandomChoose(rule_selection_probs, &rand_gen)];
                        if (rule->Apply(this, task_id, &tmp_s, &rand_gen) == 
                                PopulationGenerationRule::ResultKind::kValid) {
                            // 若mutation应用成果则加入到突变后的group state中
                            new_tmp_gs.push_back(tmp_s);
                            do_mutation = true;
                        } else {
                            // 突变失败则不进行后续突变
                            break;
                        }
                    } else {
                        // 不需要mutation则直接保留
                        new_tmp_gs.push_back(std::move(tmp_s));
                    }
                }
                // 去除突变后的state不合理的值
                Array<State> final_mutate_state;
                if (new_tmp_gs.size() == task_group->tasks.size() && do_mutation) {
                    for(size_t task_id=0; task_id<task_group->tasks.size();task_id++) {
                        Array<State> valid_new_tmp_s;
                        valid_new_tmp_s.push_back(
                            task_group->tasks[task_id]->compute_dag.InferBound(new_tmp_gs[task_id])
                        );
                        PruneInvalidState(task_group->tasks[task_id], &valid_new_tmp_s);
                        if (valid_new_tmp_s.size() != 0) {
                            final_mutate_state.push_back(std::move(valid_new_tmp_s[0]));
                        } else {
                            break;
                        }
                    }
                    if (final_mutate_state.size() == task_group->tasks.size()) {
                        pnext->push_back(std::move(final_mutate_state));
                        mutation_success_ct++;
                    } else {
                        mutation_fail_ct++;
                    }
                }
            }else {
                // 不突变的直接进入下一轮
                pnext->push_back(std::move(tmp_gs));
            }
        }

        std::swap(pnext, pnow);
        pnext->clear();
    }

    std::sort(heap.begin(), heap.end(), cmp);
    for (auto& item : heap) {
        best_states.push_back(std::move(item.first));
    }
    double duration = std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::high_resolution_clock::now() - tic_begin).count();
    StdCout(verbose) << "EvolutionarySearch\t\t#s: " << best_states.size()
                   << "\tTime elapsed: " << std::fixed << std::setprecision(2) << duration
                   << std::endl;
    return best_states;
}

Array<Array<MeasureInput> > GroupSketchPolicyNode::PickStatesWithEpsGreedy(const Array<Array<State> >& best_states, 
                                                const Array<Array<State> > random_states, 
                                                int remaining_n_trials){
    int num_random =
        static_cast<int>(GetDoubleParam(params, SketchParamKey::eps_greedy) * num_measure_per_iter_);
    int num_good = num_measure_per_iter_ - num_random;
    Array<Array<MeasureInput> > group_measure_inputs;
    size_t offset_best = 0, offset_random = 0;

    while (static_cast<int>(group_measure_inputs.size()) < std::min(num_measure_per_iter_, remaining_n_trials)) {
        Array<State> group_state;

        bool has_best = offset_best < best_states.size();
        bool has_random = offset_random < random_states.size();

        if (static_cast<int>(group_measure_inputs.size()) < num_good) {
            if (has_best) {
                group_state = best_states[offset_best++];
            } else if (has_random) {
                group_state = random_states[offset_random++];
            }else {
                break;
            }
        } else {
            if (has_random) {
                group_state = random_states[offset_random++];
            } else if (has_best) {
                group_state = best_states[offset_best++];
            } else {
                break;
            }
        }

        std::string group_state_str = "";
        for(size_t task_id=0; task_id<task_group->tasks.size(); task_id++) {
            group_state_str += group_state[task_id].ToStr() + "_";
        }
        if (!measured_states_set_.count(group_state_str)) {
            measured_states_set_.insert(group_state_str);
            measured_states_vector_.push_back(group_state);
            Array<MeasureInput> input_temp;
            for(size_t task_id=0; task_id<task_group->tasks.size(); task_id++) {
                input_temp.push_back(MeasureInput(task_group->tasks[task_id], group_state[task_id]));
            }
            group_measure_inputs.push_back(input_temp);
        }
    }
    return group_measure_inputs;
}

/* register */
TVM_REGISTER_GLOBAL("auto_scheduler.GroupSketchPolicy")
    .set_body_typed([](SearchTaskGroup task_group, GroupCostModel proxy_model, 
                        CostModel xgb, Map<String, ObjectRef> params, int seed) {
        return GroupSketchPolicy(task_group, proxy_model, xgb, params, seed);
    });
}
}