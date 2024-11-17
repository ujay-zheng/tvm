#include "sketch_policy_rules.h"

#include <set>
#include <string>
#include <utility>
#include <vector>

#include "sketch_policy.h"

namespace tvm {
namespace auto_scheduler {

static std::vector<int> auto_unroll_configs_cpu = {0, 16, 64, 512};
static std::vector<int> auto_unroll_configs_gpu = {0, 16, 64, 512, 1024};

/****************************** Sketch Generation Rule ******************************/

/********** RuleSkipStage **********/
SketchGenerationRule::ConditionKind RuleSkipStage::MeetCondition(
    const GroupSketchPolicyNode& policy, int task_id, const State& state, int stage_id) const {

    return ConditionKind::kApply;
}

std::vector<std::pair<State, int> > RuleSkipStage::Apply(
    const GroupSketchPolicyNode& policy, int task_id, const State& state, int stage_id) const {

    return {std::make_pair(state, stage_id - 1)};
}

/********** RuleAlwaysInline **********/
inline bool ShouldAlwaysBeInlined(
    const GroupSketchPolicyNode& policy, int task_id, const State& state, int stage_id){

    const SearchTask& task = policy.task_group->tasks[task_id];
    const Stage& stage = state->stages[stage_id];

    if (stage->op_type == StageKind::kPlaceholder || 
        IsOutputOp(task, state, stage_id) ||
        HasReduceIter(stage)) {
            return false;
    } 
    if (IsGPUTask(task)) { return true; } 
    else { return IsStrictlyInlineable(task, state, stage_id); }
}

SketchGenerationRule::ConditionKind RuleAlwaysInline::MeetCondition(
    const GroupSketchPolicyNode& policy, int task_id, const State& state, int stage_id) const {

    return ShouldAlwaysBeInlined(policy, task_id, state, stage_id) 
            ? ConditionKind::kApplyAndSkipRest
            : ConditionKind::kSkip;
}

std::vector<std::pair<State, int> > RuleAlwaysInline::Apply(
    const GroupSketchPolicyNode& policy, int task_id, const State& state, int stage_id) const {

    State tmp_s = state;
    tmp_s.compute_inline(stage_id);
    return {std::make_pair(std::move(tmp_s), stage_id-1)};
}

/********** RuleMultiLevelTiling **********/
SketchGenerationRule::ConditionKind RuleMultiLevelTiling::MeetCondition(
    const GroupSketchPolicyNode& policy, int task_id, const State& state, int stage_id) const {

    return NeedsMultilevelTiling(policy.task_group->tasks[task_id], state, stage_id)
            ? ConditionKind::kApplyAndSkipRest
            : ConditionKind::kSkip;
}

std::vector<std::pair<State, int> > RuleMultiLevelTiling::Apply(
    const GroupSketchPolicyNode& policy, int task_id, const State& state, int stage_id) const {

    const std::string& multi_level_tiling_structure = IsGPUTask(policy.task_group->tasks[task_id])
        ? GetStringParam(policy.params, SketchParamKey::MultiLevelTiling::gpu_structure)
        : GetStringParam(policy.params, SketchParamKey::MultiLevelTiling::cpu_structure);
    State tmp_s = DoMultiLevelTiling(state, stage_id, multi_level_tiling_structure);
    return {std::make_pair(std::move(tmp_s), stage_id - 1)};
}

/********** RuleMultiLevelTilingWithFusion **********/
SketchGenerationRule::ConditionKind RuleMultiLevelTilingWithFusion::MeetCondition(
    const GroupSketchPolicyNode& policy, int task_id, const State& state, int stage_id) const {

    if(NeedsMultilevelTiling(policy.task_group->tasks[task_id], state, stage_id) &&
    HasSingleElementwiseMatchedConsumer(policy.task_group->tasks[task_id], state, stage_id)) {
        return HasCacheWriteStage(state, stage_id) || IsGPUTask(policy.task_group->tasks[task_id])
                    ? ConditionKind::kApplyAndSkipRest
                    : ConditionKind::kApply;
    }
    return ConditionKind::kSkip;
}

std::vector<std::pair<State, int> > RuleMultiLevelTilingWithFusion::Apply(
    const GroupSketchPolicyNode& policy, int task_id, const State& state, int stage_id) const {

    int target_stage_id;
    ICHECK(
        HasSingleElementwiseMatchedConsumer(
            policy.task_group->tasks[task_id], 
            state, stage_id, &target_stage_id
    ));
    const std::string& multi_level_tiling_structure = IsGPUTask(policy.task_group->tasks[task_id]) 
        ? GetStringParam(policy.params, SketchParamKey::MultiLevelTiling::gpu_structure)
        : GetStringParam(policy.params, SketchParamKey::MultiLevelTiling::cpu_structure);
    std::vector<int> spatial_split_step_ids;
    State base_state =
        DoMultiLevelTiling(state, stage_id, multi_level_tiling_structure, &spatial_split_step_ids);
    
    std::vector<std::pair<State, int> > ret;
    std::vector<int> follow_tiling_levels = IsGPUTask(policy.task_group->tasks[task_id]) 
                                            ? std::vector<int>{3}
                                            : std::vector<int>{1, 2};
    for(int level: follow_tiling_levels) {
        if(tolower(multi_level_tiling_structure[level-1]) != 's') { continue;}
        State tmp_s = base_state;
        tmp_s = FollowTiling(tmp_s, target_stage_id, spatial_split_step_ids, level);
        const Iterator& target_iter = 
            tmp_s->stages[target_stage_id]->iters[level*spatial_split_step_ids.size()-1];
        tmp_s.compute_at(stage_id, target_stage_id, target_iter);
        ret.emplace_back(std::move(tmp_s), stage_id - 1);
    }
    return ret;
}

/********** RuleAddCacheRead **********/
SketchGenerationRule::ConditionKind RuleAddCacheRead::MeetCondition(
    const GroupSketchPolicyNode& policy, int task_id, const State& state, int stage_id) const {

    const SearchTask& task = policy.task_group->tasks[task_id];

    const std::set<int>& consumers = GetConsumers(task, state, stage_id);
    
    if(consumers.size() == 0) { return ConditionKind::kSkip;}
    
    int target_stage_id = *consumers.begin();
    if(!NeedsMultilevelTiling(task, state, target_stage_id)) {return ConditionKind::kSkip;}
    
    if(HasCrossThreadReduction(state, target_stage_id)) { return ConditionKind::kSkip; }
    
    const std::set<int>& producers = GetDirectProducers(task, state, target_stage_id);
    if(producers.find(stage_id) == producers.end()) { return ConditionKind::kSkip;}
    
    return ConditionKind::kApplyAndSkipRest;
}

std::vector<std::pair<State, int> > RuleAddCacheRead::Apply(
    const GroupSketchPolicyNode& policy, int task_id, const State& state, int stage_id) const {

    const SearchTask& task = policy.task_group->tasks[task_id];
    const std::set<int>& comsumers = GetConsumers(task, state, stage_id);
    State tmp_s = state;

    int target_stage_id_offset = 0;
    for(int orig_target_stage_id : comsumers) {
        int target_stage_id = orig_target_stage_id + target_stage_id_offset;
        
        int added_stage_id = tmp_s.cache_read(stage_id, "shared", {target_stage_id}, task->compute_dag);
        target_stage_id_offset++;
        target_stage_id++;
        
        const auto& share_read_pos = 
            GetLastReduceIteratorInOutermostReduceTile(tmp_s->stages[target_stage_id]);
        tmp_s.compute_at(added_stage_id, target_stage_id, share_read_pos);
    }
    return {std::make_pair(tmp_s, stage_id)};
}

/********** RuleAddCacheWrite **********/
SketchGenerationRule::ConditionKind RuleAddCacheWrite::MeetCondition(
    const GroupSketchPolicyNode& policy, int task_id, const State& state, int stage_id) const {

    if(NeedsMultilevelTiling(policy.task_group->tasks[task_id], state, stage_id) &&
        !HasSingleElementwiseMatchedConsumer(policy.task_group->tasks[task_id], state, stage_id)){
        return IsGPUTask(policy.task_group->tasks[task_id]) 
                ? ConditionKind::kApplyAndSkipRest
                : ConditionKind::kApply;
    }
    return ConditionKind::kSkip;
}

std::vector<std::pair<State, int> > RuleAddCacheWrite::Apply(
    const GroupSketchPolicyNode& policy, int task_id, const State& state, int stage_id) const {

    State tmp_s = state;
    tmp_s.cache_write(stage_id, "local", policy.task_group->tasks[task_id]->compute_dag);
    return {std::make_pair(std::move(tmp_s), stage_id)};
}

/********** RuleAddRfactor **********/
SketchGenerationRule::ConditionKind RuleAddRfactor::MeetCondition(
    const GroupSketchPolicyNode& policy, int task_id, const State& state, int stage_id) const {

    return (NeedsRfactor(policy.task_group->tasks[task_id], state, stage_id) &&
            !HasCacheWriteStage(state, stage_id))
            ? ConditionKind::kApply
            : ConditionKind::kSkip;
}

std::vector<std::pair<State, int> > RuleAddRfactor::Apply(
    const GroupSketchPolicyNode& policy, int task_id, const State& state, int stage_id) const {

    Array<Iterator> space_iters, reduce_iters;
    Iterator fused_reduce_iter;
    State base_state =
        FuseAllReductionIterators(state, stage_id, &fused_reduce_iter, &space_iters, &reduce_iters);
    
    const auto& split_res = base_state.split(stage_id, fused_reduce_iter, {Integer(1)});
    int factor_axis_id = static_cast<int>(space_iters.size());
    std::vector<std::pair<State, int> > ret;
    for(const auto& split_iter : split_res) {
        State tmp_s = base_state;
        int rstage_id = 
            tmp_s.rfactor(stage_id, split_iter, factor_axis_id, policy.task_group->tasks[task_id]->compute_dag);
        if (split_iter == split_res[1]) {
            Array<Iterator> new_order;
            for(size_t i=0; i<tmp_s->stages[rstage_id]->iters.size(); ++i) {
                if (i != space_iters.size()) {
                    new_order.push_back(tmp_s->stages[rstage_id]->iters[i]);
                }
            }
            new_order.push_back(tmp_s->stages[rstage_id]->iters[space_iters.size()]);
            tmp_s.reorder(rstage_id, new_order);
        }
        ret.emplace_back(std::move(tmp_s), rstage_id - 1);
    }

    return ret;
}

/********** RuleSimplifyComputeWithConstTensor **********/
SketchGenerationRule::ConditionKind RuleSimplifyComputeWithConstTensor::MeetCondition(
    const GroupSketchPolicyNode& policy, int task_id, const State& state, int stage_id) const {

    return state->stages[stage_id]->op->attrs.count(SearchPolicyKey::simplify_const_tensor_indices)
            ? ConditionKind::kApplyAndSkipRest
            : ConditionKind::kSkip;
}

std::vector<std::pair<State, int> > RuleSimplifyComputeWithConstTensor::Apply(
    const GroupSketchPolicyNode& policy, int task_id, const State& state, int stage_id) const {

    std::set<std::string> const_tensor_indices = 
        GetIterNameSetParam(state->stages[stage_id]->op->attrs, SearchPolicyKey::simplify_const_tensor_indices);
    
    State tmp_s = state;
    Array<Array<Iterator> > tiled_outer_iters;
    Array<Iterator> unrolled_inner_iters;

    size_t tile_level = 2;

    for(const auto& iter : state->stages[stage_id]->iters) {
        if(const_tensor_indices.count(iter->name)){
            unrolled_inner_iters.push_back(tmp_s.unroll(stage_id, iter));
        } else {
            ICHECK(iter->iter_kind == IteratorKind::kSpatial);
            tiled_outer_iters.push_back(
                tmp_s.split(stage_id, iter, Array<Optional<Integer> >(tile_level-1, NullOpt))
            );
        }
    }

    Array<Iterator> new_order;
    for (size_t i=0; i<tile_level; ++i){
        for(size_t j=0; j<tiled_outer_iters.size(); ++j){
            new_order.push_back(tiled_outer_iters[j][i]);
        }
    }
    new_order.insert(new_order.end(), unrolled_inner_iters.begin(), unrolled_inner_iters.end());
    tmp_s.reorder(stage_id, new_order);

    return {std::make_pair(tmp_s, stage_id-1)};
}

/********** RuleCrossThreadReduction **********/
SketchGenerationRule::ConditionKind RuleCrossThreadReduction::MeetCondition(
    const GroupSketchPolicyNode& policy, int task_id, const State& state, int stage_id) const {

    ICHECK(IsGPUTask(policy.task_group->tasks[task_id]));

    if (HasCacheWriteStage(state, stage_id)) { return ConditionKind::kSkip; }
    
    const auto& op = state->stages[stage_id]->op;
    if (op->IsInstance<te::ComputeOpNode>()) {
        auto [cum_space_len, cum_reduce_len] = 
            GetCumulativeSpaceAndReductionLength(state->stages[stage_id]);
        if (NeedsMultilevelTiling(policy.task_group->tasks[task_id], state, stage_id)) {
            if (cum_space_len > policy.task_group->tasks[task_id]->hardware_params->max_local_memory_per_block) {
                return ConditionKind::kSkip;
            }
            return cum_space_len < cum_reduce_len ? ConditionKind::kApply : ConditionKind::kSkip;
        } else if (cum_reduce_len > 1) {
            return cum_reduce_len > policy.task_group->tasks[task_id]->hardware_params->warp_size 
                    ? ConditionKind::kApply
                    : ConditionKind::kSkip;
        }
    }
    return ConditionKind::kSkip;
}

std::vector<std::pair<State, int>> RuleCrossThreadReduction::Apply(
    const GroupSketchPolicyNode& policy, int task_id, const State& state, int stage_id) const {

    const SearchTask& task = policy.task_group->tasks[task_id];
    State tmp_s = state;

    Array<Iterator> space_iters, reduce_iters;
    Iterator fused_reduce_iter;
    tmp_s =
        FuseAllReductionIterators(tmp_s, stage_id, &fused_reduce_iter, &space_iters, &reduce_iters);
    
    bool fusible = false;
    int target_stage_id = GetSingleConsumerId(policy.task_group->tasks[task_id], tmp_s, stage_id);
    int num_common_outer = -1;
    if  (target_stage_id >= 0) {
        num_common_outer =
            GetNumCommonOuterIterator(policy.task_group->tasks[task_id], tmp_s, stage_id, target_stage_id);
        if (num_common_outer > 0 && 
            !NeedsMultilevelTiling(policy.task_group->tasks[task_id], state, target_stage_id)) {
            fusible = true;
        }
    }
    
    if (fusible) {
        const Stage& target_stage = state->stages[target_stage_id];
        std::vector<int> split_step_ids;
    
        GetSplitStepIds(tmp_s, target_stage_id, &split_step_ids);
    
        if(split_step_ids.size() == 0) {
            ICHECK(!HasReduceIter(target_stage));
            const auto& split_res = tmp_s.split(target_stage_id, target_stage->iters.back(),
                                                {Integer(task->hardware_params->warp_size)});
            tmp_s.bind(target_stage_id, split_res[1], IteratorAnnotation::kThreadX);
            split_step_ids.push_back(tmp_s->transform_steps.size() - 2);
        }
    
        ICHECK_EQ(split_step_ids.size(), 1);
    
        const Iterator& target_iter = tmp_s->stages[target_stage_id]->iters[num_common_outer-1];
        const auto& split_res = tmp_s.follow_split(stage_id, fused_reduce_iter, split_step_ids[0], 1);
        tmp_s.bind(stage_id, split_res[1], IteratorAnnotation::kThreadX);
        tmp_s.compute_at(stage_id, target_stage_id, target_iter);
    } else {
        const auto& split_res = 
            tmp_s.split(stage_id, fused_reduce_iter, {Integer(task->hardware_params->warp_size)});
        tmp_s.bind(stage_id, split_res[1], IteratorAnnotation::kThreadX);
    }

    return {std::make_pair(std::move(tmp_s), stage_id - 1)};
}

/********** RuleSpecialComputeLocationGPU **********/
SketchGenerationRule::ConditionKind RuleSpecialComputeLocationGPU::MeetCondition(
    const GroupSketchPolicyNode& policy, int task_id, const State& state, int stage_id) const {

    if(GetProducers(policy.task_group->tasks[task_id], state, stage_id).empty()) {
        return ConditionKind::kSkip;
    }

    if(!ShouldAlwaysBeInlined(policy, task_id, state, stage_id)) {
        return ConditionKind::kSkip;
    }

    const std::set<int>& consumers = GetConsumers(policy.task_group->tasks[task_id], state, stage_id);
    if(consumers.size() == 1 && state->stages[*consumers.begin()]->op->attrs.count(
                                SearchPolicyKey::simplify_const_tensor_indices)) {
        return ConditionKind::kApplyAndSkipRest;
    }
    return ConditionKind::kSkip;
}

std::vector<std::pair<State, int>> RuleSpecialComputeLocationGPU::Apply(
    const GroupSketchPolicyNode& policy, int task_id, const State& state, int stage_id) const {

    State tmp_s = state;
    const std::set<int>& consumers = GetConsumers(policy.task_group->tasks[task_id], state, stage_id);
    ICHECK_EQ(consumers.size(), 1);

    const Stage& target_stage = state->stages[*consumers.begin()];
    for(size_t i=0; i<target_stage->iters.size(); ++i) {
        if(target_stage->iters[i]->annotation == IteratorAnnotation::kUnroll) {
            ICHECK_GT(i, 0);

            tmp_s.compute_at(stage_id, *consumers.begin(), target_stage->iters[i-1]);
            break;
        }
    }
    return {std::make_pair(std::move(tmp_s), stage_id - 1)};
}

/********** RuleCustomSketch **********/
SketchGenerationRule::ConditionKind RuleCustomSketch::MeetCondition(
    const GroupSketchPolicyNode& policy, int task_id, const State& state, int stage_id) const {
    return ConditionKind::kSkip;
}

std::vector<std::pair<State, int>> RuleCustomSketch::Apply(
    const GroupSketchPolicyNode& policy, int task_id, const State& state, int stage_id) const {
    return std::vector<std::pair<State, int>>();
}

/****************************** Init Population******************************/
PopulationGenerationRule::ResultKind InitFillTileSize::Apply(
    GroupSketchPolicyNode* policy, int task_id, State* state, std::mt19937* rand_gen) const {
    
    SplitFactorizationMemo split_memo;
    int max_innermost_split_factor = 
        GetIntParam(policy->params, SketchParamKey::max_innermost_split_factor);

    StateNode* pstate = state->CopyOnWrite();
    for(size_t step_id=0; step_id<(*state)->transform_steps.size(); ++step_id) {
        if (auto ps = (*state)->transform_steps[step_id].as<SplitStepNode>()) {
            bool all_defined = true;
            for (const auto& len : ps->lengths){
                if (!len) {
                    all_defined = false;
                    break;
                }
            }
            if (all_defined) { continue; }

            ICHECK(ps->extent);
            int extent = GetIntImm(ps->extent.value());
            const auto& candidate_lens = split_memo.GetFactorizationSchemes(extent, ps->lengths.size(),
                                                                      max_innermost_split_factor);
            ICHECK(!candidate_lens.empty());
            const auto& candidate_lengths = candidate_lens[(*rand_gen)() % candidate_lens.size()];

            pstate->transform_steps.Set(
                step_id,
                SplitStep(ps->stage_id, ps->iter_id, ps->extent,
                            Array<Optional<Integer>>(candidate_lengths.begin(), candidate_lengths.end()),
                            ps->inner_to_outer));
        }
    }
    pstate->concrete = true;
    
    return ResultKind::kValid;
}

PopulationGenerationRule::ResultKind InitChangeComputeLocation::Apply(
    GroupSketchPolicyNode* policy, int task_id, State* state, std::mt19937* rand_gen) const {
    
    if (GetIntParam(policy->params, SketchParamKey::disable_change_compute_location)) {
        return ResultKind::kValid;
    }

    for (int stage_id = static_cast<int>((*state)->stages.size())-1; stage_id>=0; stage_id--) {
        const Stage& stage = (*state)->stages[stage_id];
        if (stage->op_type == StageKind::kPlaceholder || stage->compute_at == ComputeAtKind::kInlined) {
            continue;
        }
        if (IsTiled(stage) || NeedsMultilevelTiling(policy->task_group->tasks[task_id], *state, stage_id)) {
            continue;
        }
    
        std::vector<std::pair<int, int> > candidates = 
            GetComputeLocationCandidates(policy->task_group->tasks[task_id], *state, stage_id);
    
        int choice = (*rand_gen)() % (candidates.size() + 2);

        if (choice == 0) {
            if (!HasReduceIter(stage)) {
                const auto& stage_to_attach_iter = (*state)->attach_map->stage_to_attach_iter;
                if (stage_to_attach_iter.find(stage_id) != stage_to_attach_iter.end()) {
                    state->compute_inline(stage_id);
                }
            }
        }else if (choice == 1) {
            state->compute_root(stage_id);
        } else {
            choice = choice - 2;
            const Stage& stage = (*state)->stages[candidates[choice].first];
            state->compute_at(stage_id, candidates[choice].first, 
                                stage->iters[candidates[choice].second]);
        }
    }
    try { *state = policy->task_group->tasks[task_id]->compute_dag.InferBound(*state); } 
    catch (std::exception& e) { return ResultKind::kInvalid; }

    return ResultKind::kValid;
}

PopulationGenerationRule::ResultKind InitParallel::Apply(
    GroupSketchPolicyNode* policy, int task_id, State* state, std::mt19937* rand_gen) const {
    
    std::function<void(const GroupSketchPolicyNode&, int task_id, State*, int stage_id, int iter_offset)>
        annotate_parallel;
    annotate_parallel = [&annotate_parallel](const GroupSketchPolicyNode& policy, int task_id,
                                            State* state, int stage_id, int iter_offset) {
        const Stage& stage = (*state)->stages[stage_id];
        
        Array<Iterator> to_fuse;
        int64_t parallel_degree = 1;
        
        size_t iter_id = iter_offset;
        for (; iter_id < stage->iters.size(); ++iter_id) {
            const Iterator& it = stage->iters[iter_id];
            if (it->iter_kind == IteratorKind::kReduction ||
                it->annotation != IteratorAnnotation::kNone) {
                break;
            }
            to_fuse.push_back(it);
            parallel_degree *= GetExtent(it);

            if (parallel_degree > policy.task_group->tasks[task_id]->hardware_params->num_cores*16){
                break;
            }

            if ((*state)->attach_map->iter_to_attached_stages.count(std::make_pair(stage_id, iter_id))) {
                break;
            }
        }

        if (parallel_degree == 1) {
            auto res =
                (*state)->attach_map->iter_to_attached_stages.find(std::make_pair(stage_id, iter_id));
            if (res != (*state)->attach_map->iter_to_attached_stages.end()) {
                for (int attached_stage_id : res->second) {
                    annotate_parallel(policy, task_id, state, attached_stage_id, 0);
                }
                annotate_parallel(policy, task_id, state, stage_id, iter_id + 1);
            }
        }

        if (!to_fuse.empty()) {
            if (to_fuse.size() == 1) {
                state->parallel(stage_id, to_fuse[0]);
            } else {
                Iterator fused_iter = state->fuse(stage_id, to_fuse);
                state->parallel(stage_id, fused_iter);
            }
        }
    };

    for (size_t stage_id=0; stage_id<(*state)->stages.size(); ++stage_id) {
        const Stage& stage = (*state)->stages[stage_id];
        if (stage->compute_at != ComputeAtKind::kRoot || stage->op_type == StageKind::kPlaceholder) {
            continue;
        }
        annotate_parallel(*policy, task_id, state, stage_id, 0);
    }

    return ResultKind::kValid;
}

PopulationGenerationRule::ResultKind InitUnroll::Apply(
    GroupSketchPolicyNode* policy, int task_id, State* state, std::mt19937* rand_gen) const {
    
    std::vector<int>& auto_unroll_configs = IsGPUTask(policy->task_group->tasks[task_id])
                                            ? auto_unroll_configs_gpu
                                            : auto_unroll_configs_cpu;

    for (size_t stage_id=0; stage_id<(*state)->stages.size(); ++stage_id) {
        const Stage& stage = (*state)->stages[stage_id];
        if (stage->compute_at == ComputeAtKind::kInlined || stage->op_type == StageKind::kPlaceholder) {
            continue;
        }

        if (stage->op->attrs.count(SearchPolicyKey::always_unroll_inner)) {
            const auto& to_unroll_name_set =
                GetIterNameSetParam(stage->op->attrs, SearchPolicyKey::always_unroll_inner);
            
            std::set<std::string> visited_names;
            for (int n=static_cast<int>(stage->iters.size())-1; n>=0; n--) {
                const Iterator& it = stage->iters[n];

                size_t size_before = visited_names.size();
                ExtractOriginalIterators(it->name, &visited_names);
                if (size_before == visited_names.size()) { break; }

                std::set<std::string> name;
                ExtractOriginalIterators(it->name, &name);
                if (name.size() == 1 && to_unroll_name_set.count(*name.begin())) {
                    if (it->annotation == IteratorAnnotation::kNone) { 
                        state->unroll(stage_id, it);
                    }
                }
            }
        }

        if (HasReduceIter(stage)) {
            int value = auto_unroll_configs[(*rand_gen)() % auto_unroll_configs.size()];
            state->pragma(stage_id, (*state)->stages[stage_id]->iters[0],
                    std::string("auto_unroll_max_step") + "$" + std::to_string(value));
        }
    }
    return ResultKind::kValid;
}

PopulationGenerationRule::ResultKind InitVectorization::Apply(
    GroupSketchPolicyNode* policy, int task_id, State* state, std::mt19937* rand_gen) const {
    
    for (size_t stage_id=0; stage_id<(*state)->stages.size(); ++stage_id){
        const Stage& stage = (*state)->stages[stage_id];

        if (stage->compute_at == ComputeAtKind::kInlined || 
            stage->op_type == StageKind::kPlaceholder) {
            continue;
        }
        
        int64_t cum_length_prod = 1;
        
        int num_fusible = 0;
        while (num_fusible < static_cast<int>(stage->iters.size())) {
            int iter_id = static_cast<int>(stage->iters.size()) - 1 - num_fusible;
        
            if ((*state)->attach_map->iter_to_attached_stages.count(std::make_pair(stage_id, iter_id))) {
                break;
            }

            const Iterator& it = stage->iters[iter_id];
        
            if (it->iter_kind == IteratorKind::kReduction ||
                it->annotation != IteratorAnnotation::kNone) {
                break;
            }

            if (IsTiled(stage) && num_fusible != 0) { break; }

            cum_length_prod *= GetExtent(it);
            if (cum_length_prod > GetIntParam(policy->params, SketchParamKey::max_vectorize_size)) {
                break;
            }

            num_fusible++;
        }

        if (num_fusible > 1) { num_fusible = 1 + (*rand_gen)() % (num_fusible - 1); }
        
        if (num_fusible == 1) {
            state->vectorize(stage_id, stage->iters.back());
        } else if (num_fusible > 1) {
            Array<Iterator> to_fuse(stage->iters.end() + (-num_fusible), stage->iters.end());
            state->vectorize(stage_id, state->fuse(stage_id, to_fuse));
        }
    }
    return ResultKind::kValid;
}

PopulationGenerationRule::ResultKind InitThreadBind::Apply(
    GroupSketchPolicyNode* policy, int task_id, State* state, std::mt19937* rand_gen) const {
    
    std::set<int> multi_level_tiling_root_set;
    for (size_t stage_id = 0; stage_id < (*state)->stages.size(); ++stage_id) {
        if (NeedsMultilevelTiling(policy->task_group->tasks[task_id], *state, stage_id)) {
            const Stage& stage = (*state)->stages[stage_id];
            if (stage->compute_at == ComputeAtKind::kInlined) {
                continue;
            } else if (stage->compute_at != ComputeAtKind::kIter) {
                ICHECK(HasCrossThreadReduction(*state, stage_id));
            } else {
                const auto res = (*state)->attach_map->stage_to_attach_iter.find(stage_id);
                ICHECK(res != (*state)->attach_map->stage_to_attach_iter.end());
                multi_level_tiling_root_set.insert(res->second.first);
            }
        }
    }

    *state = policy->task_group->tasks[task_id]->compute_dag.InferBound(*state);
    
    for(int stage_id=(*state)->stages.size()-1; stage_id>=0;--stage_id) {
        const Stage& stage = (*state)->stages[stage_id];

        if (stage->compute_at == ComputeAtKind::kInlined ||
            stage->op_type == StageKind::kPlaceholder) {
            continue;
        }
        
        if (HasCrossThreadReduction(*state, stage_id)) {
            if (stage->compute_at != ComputeAtKind::kRoot) { continue; }
        
            Iterator fused_it;
            *state = std::move(FuseAllOuterSpaceIterators(*state, stage_id, &fused_it));
            state->bind(stage_id, fused_it, IteratorAnnotation::kBlockX);
            continue;
        }
        
        if (HasAnnotatedIter(stage, IteratorAnnotation::kThreadX)) { continue; }

        if (stage->compute_at == ComputeAtKind::kRoot) {
            if (!multi_level_tiling_root_set.count(stage_id)) {
                Iterator fused_it;
                *state = FuseAllOuterSpaceIterators(*state, stage_id, &fused_it);
        
                if (GetExtent(fused_it) <= policy->task_group->tasks[task_id]->hardware_params->warp_size) {
                    state->bind(stage_id, fused_it, IteratorAnnotation::kThreadX);
                } else {
                    const auto& split_its = state->split(
                        stage_id, fused_it, {Integer(policy->task_group->tasks[task_id]->hardware_params->warp_size)});
                    state->bind(stage_id, split_its[0], IteratorAnnotation::kBlockX);
                    state->bind(stage_id, split_its[1], IteratorAnnotation::kThreadX);
                }
                continue;
            }

            auto pop = stage->op.as<te::ComputeOpNode>();
            std::vector<Iterator> to_fuse;
            int total_space_extent = 1;
            for (const auto& i : pop->root_iter_vars()) {
                ICHECK(i->dom.defined());
                const auto& pint = i->dom->extent.as<IntImmNode>();
                ICHECK(pint);
                total_space_extent *= pint->value;
            }

            bool check_min_thread_extent = true;
            
            if(total_space_extent <= policy->task_group->tasks[task_id]->hardware_params->warp_size * 2) {
                check_min_thread_extent = false;
            }

            for (size_t i = 0; i < pop->axis.size(); i++) {
                const auto& it = (*state)->stages[stage_id]->iters[i];
                if (!StrEndsWith(it->name, ".0")) { break; }
                to_fuse.push_back(it);
            }
            const auto& blockidx_it = state->fuse(stage_id, to_fuse);
            state->bind(stage_id, blockidx_it, IteratorAnnotation::kBlockX);

            to_fuse.clear();
            for (size_t i = 1; i < pop->axis.size() + 1; i++) {
                const auto& it = (*state)->stages[stage_id]->iters[i];
                if (!StrEndsWith(it->name, ".1")) { break; }
                to_fuse.push_back((*state)->stages[stage_id]->iters[i]);
            }
            const auto& vthread_it = state->fuse(stage_id, to_fuse);
            if (GetExtent(vthread_it) > policy->task_group->tasks[task_id]->hardware_params->max_vthread_extent) {
                return ResultKind::kInvalid;
            }
            state->bind(stage_id, vthread_it, IteratorAnnotation::kVThread);
            
            to_fuse.clear();
            for (size_t i = 2; i < pop->axis.size() + 2; i++) {
                const auto& it = (*state)->stages[stage_id]->iters[i];
                if (!StrEndsWith(it->name, ".2")) { break; }
                to_fuse.push_back((*state)->stages[stage_id]->iters[i]);
            }
            const auto& threadidx_it = state->fuse(stage_id, to_fuse);
            if (check_min_thread_extent &&
                GetExtent(threadidx_it) < policy->task_group->tasks[task_id]->hardware_params->warp_size) {
                return ResultKind::kInvalid;
            }
            state->bind(stage_id, threadidx_it, IteratorAnnotation::kThreadX);
        } else if (stage->compute_at == ComputeAtKind::kIter &&
                    StrEndsWith(stage->op->name, ".shared")) {
            const auto& it = (*state)->attach_map->stage_to_attach_iter.find(stage_id);
            ICHECK(it != (*state)->attach_map->stage_to_attach_iter.end());
            Array<Integer> spatial_split_step_ids = GetSpatialSplitStepIds(*state, it->second.first);
            
            Iterator fused = state->fuse(stage_id, (*state)->stages[stage_id]->iters);
            
            const auto& iters0 = state->split(stage_id, fused, {Integer(1)});
            state->vectorize(stage_id, iters0[1]);
            
            const auto& iters1 =
                state->follow_fused_split(stage_id, iters0[0], spatial_split_step_ids, 1, true);
            state->bind(stage_id, iters1[1], IteratorAnnotation::kThreadX);
        }
    }
    return ResultKind::kValid;
}

PopulationGenerationRule::ResultKind MutateTileSize::Apply(
    GroupSketchPolicyNode* policy, int task_id, State* state, std::mt19937* rand_gen) const {
    
    int max_innermost_split_factor = 
        GetIntParam(policy->params, SketchParamKey::max_innermost_split_factor);

    std::vector<size_t> split_step_ids;
    for (size_t i = 0; i < (*state)->transform_steps.size(); ++i) {
        if (auto ps = (*state)->transform_steps[i].as<SplitStepNode>()) {
            if (!ps->extent.defined() || !ps->extent.value()->IsInstance<IntImmNode>()) {
                continue;
            }
            auto innermost_factor = ps->lengths.back().value_or(max_innermost_split_factor + 1);
            if (GetIntImm(innermost_factor) <= max_innermost_split_factor) {
                split_step_ids.push_back(i);
            }
        }
    }
    
    if (split_step_ids.empty()) { return ResultKind::kInvalid; }

    int retry_ct = 0;
    int64_t extent = 1;
    int step_id;
    const SplitStepNode* ps;

    do {
        step_id = split_step_ids[(*rand_gen)() % split_step_ids.size()];
        ps = (*state)->transform_steps[step_id].as<SplitStepNode>();
        ICHECK(ps != nullptr);
        extent = GetIntImm(ps->extent.value());
        retry_ct += 1;
    } while (retry_ct < static_cast<int>(split_step_ids.size()) << 2 && (extent == 1 || extent == 0));

    if (extent <= 1) { return ResultKind::kInvalid; }

    std::vector<int> lengths(ps->lengths.size() + 1, 1);
    for (int i = 0; i < static_cast<int>(ps->lengths.size()); ++i) {
        lengths[i + 1] = GetIntImm(ps->lengths[i].value());
    }
    lengths[0] = extent / ElementProduct(lengths);

    std::vector<int> random_perm;
    RandomPermutation(lengths.size(), &random_perm, rand_gen);

    for (size_t i = 0; i < random_perm.size(); ++i) {
        size_t src_idx = random_perm[i];
        int length = lengths[src_idx];
        if (length <= 1) { continue; }

        size_t dst_idx = random_perm[(i + 1) % random_perm.size()];
        const std::vector<int>& factors = policy->split_memo.GetFactors(length);
        ICHECK_GE(factors.size(), 1);

        int divide_factor;
        if (dst_idx == lengths.size() - 1) {
            int max_factor_index = static_cast<int>(factors.size()) - 1;
            for (; max_factor_index >= 1; max_factor_index--) {
                if (factors[max_factor_index] * lengths[dst_idx] <= max_innermost_split_factor) {
                    break;
                }
            }
            if (max_factor_index == 0) { continue; }
            divide_factor = factors[1 + (*rand_gen)() % (max_factor_index)];
        } else {
            divide_factor = factors[1 + (*rand_gen)() % (factors.size() - 1)];
        }

        Array<Integer> new_lengths;
        for (size_t j = 1; j < lengths.size(); ++j) {
            if (j == src_idx) { new_lengths.push_back(Integer(lengths[j] / divide_factor)); }
            else if (j == dst_idx) { new_lengths.push_back(Integer(lengths[j] * divide_factor)); }
            else { new_lengths.push_back(Integer(lengths[j])); }
        }

        ICHECK_LE(GetIntImm(new_lengths.back()), max_innermost_split_factor);
    
        StateNode* pstate = state->CopyOnWrite();
        pstate->transform_steps.Set(
            step_id, SplitStep(ps->stage_id, ps->iter_id, ps->extent,
                                Array<Optional<Integer>>(new_lengths.begin(), new_lengths.end()),
                                ps->inner_to_outer));
        return ResultKind::kValid;
    }
    return ResultKind::kInvalid;
}

PopulationGenerationRule::ResultKind MutateAutoUnroll::Apply(
    GroupSketchPolicyNode* policy, int task_id, State* state, std::mt19937* rand_gen) const {
    
    std::vector<int> pragma_steps;
    for (size_t i = 0; i < (*state)->transform_steps.size(); ++i) {
        if (auto ps = (*state)->transform_steps[i].as<PragmaStepNode>()) {
            if (StrStartsWith(ps->pragma_type, "auto_unroll_max_step")) {
                pragma_steps.push_back(i);
            }
        }
    }
    if (pragma_steps.empty()) { return ResultKind::kInvalid; }

    std::vector<int>& auto_unroll_configs = IsGPUTask(policy->task_group->tasks[task_id])
                                            ? auto_unroll_configs_gpu
                                            : auto_unroll_configs_cpu;

    auto step_id = pragma_steps[(*rand_gen)() % pragma_steps.size()];
    auto ps = (*state)->transform_steps[step_id].as<PragmaStepNode>();
    ICHECK(ps);

    int val = auto_unroll_configs[(*rand_gen)() % auto_unroll_configs.size()];
    StateNode* pstate = state->CopyOnWrite();
    pstate->transform_steps.Set(
        step_id, PragmaStep(ps->stage_id, ps->iter_id,
                            std::string("auto_unroll_max_step") + "$" + std::to_string(val)));
    Stage new_stage = pstate->stages[ps->stage_id];
    new_stage.CopyOnWrite()->attrs.auto_unroll_max_step = val;
    pstate->stages.Set(ps->stage_id, new_stage);
    return ResultKind::kValid;
}

PopulationGenerationRule::ResultKind MutateComputeLocation::Apply(
    GroupSketchPolicyNode* policy, int task_id, State* state, std::mt19937* rand_gen) const {
    
    if (GetIntParam(policy->params, SketchParamKey::disable_change_compute_location)) {
        return ResultKind::kInvalid;
    }

    std::vector<int> compute_at_steps;
    for (size_t s = 0; s < (*state)->transform_steps.size(); ++s) {
        if (auto ps = (*state)->transform_steps[s].as<ComputeAtStepNode>()) {
            int stage_inc = GetTargetStageIDInState(*state, s) - ps->stage_id;
            
            if (IsTiled((*state)->stages[ps->stage_id + stage_inc])) { continue; }
            
            if (NeedsMultilevelTiling(policy->task_group->tasks[task_id], *state, ps->stage_id + stage_inc)) {
                continue;
            }
            compute_at_steps.push_back(s);
        }
    }

    if (compute_at_steps.empty()) { return ResultKind::kInvalid; }

    size_t step_id = compute_at_steps[(*rand_gen)() % compute_at_steps.size()];
    auto ps = (*state)->transform_steps[step_id].as<ComputeAtStepNode>();
    int stage_inc = GetTargetStageIDInState(*state, step_id) - ps->stage_id;
    ICHECK(ps != nullptr);

    std::vector<std::pair<int, int>> candidates =
        GetComputeLocationCandidates(policy->task_group->tasks[task_id], *state, ps->stage_id + stage_inc);
    if (candidates.empty()) { return ResultKind::kInvalid; }
    int choice = (*rand_gen)() % (candidates.size());
    int new_compute_at_stage_id = candidates[choice].first;
    int new_compute_at_iter_id = candidates[choice].second;

    State tmp_s = policy->task_group->tasks[task_id]->compute_dag->init_state;
    for (size_t s = 0; s < (*state)->transform_steps.size(); ++s) {
        if (s == step_id) {
            tmp_s.CopyOnWrite()->transform_steps.push_back(
                ComputeAtStep(ps->stage_id, new_compute_at_stage_id - stage_inc, new_compute_at_iter_id));
        } else {
            tmp_s.CopyOnWrite()->transform_steps.push_back((*state)->transform_steps[s]);
        }
        try {
            StepApplyToState(tmp_s->transform_steps.back(), &tmp_s, policy->task_group->tasks[task_id]->compute_dag);
        } catch (Error& e) {
            return ResultKind::kInvalid;
        }
    }
    *state = tmp_s;
    return ResultKind::kValid;
}

PopulationGenerationRule::ResultKind MutateParallel::Apply(
    GroupSketchPolicyNode* policy, int task_id, State* state, std::mt19937* rand_gen) const {
    
    std::vector<int> parallel_steps;
    for (size_t s = 0; s < (*state)->transform_steps.size(); ++s) {
        auto ps = (*state)->transform_steps[s].as<AnnotationStepNode>();
        if (!ps || ps->annotation != IteratorAnnotation::kParallel) { continue; }

        if (ps->iter_id != 0 || s == 0 || !(*state)->transform_steps[s - 1].as<FuseStepNode>()) {
            continue;
        }
        auto fuse_step = (*state)->transform_steps[s - 1].as<FuseStepNode>();
        if (fuse_step->fused_ids[0] != 0) { continue; }
        parallel_steps.push_back(s);
    }
    if (parallel_steps.empty()) { return ResultKind::kInvalid; }

    size_t step_id = parallel_steps[(*rand_gen)() % parallel_steps.size()];

    State tmp_s = policy->task_group->tasks[task_id]->compute_dag->init_state;
    for (size_t s = 0; s < step_id - 1; ++s) {
        const auto& step = (*state)->transform_steps[s];
        tmp_s.CopyOnWrite()->transform_steps.push_back(step);
        StepApplyToState(step, &tmp_s, policy->task_group->tasks[task_id]->compute_dag);
    }

    auto fuse_step = (*state)->transform_steps[step_id - 1].as<FuseStepNode>();
    int stage_id = fuse_step->stage_id;
    const Stage& stage = tmp_s->stages[stage_id];
    size_t max_fusable_iter_id;
    for (max_fusable_iter_id = 0; max_fusable_iter_id < stage->iters.size(); ++max_fusable_iter_id) {
        const Iterator& it = stage->iters[max_fusable_iter_id];
        if (it->iter_kind == IteratorKind::kReduction || it->annotation != IteratorAnnotation::kNone) {
            break;
        }
        
        if (tmp_s->attach_map->iter_to_attached_stages.count(
            std::make_pair(stage_id, max_fusable_iter_id))) {
            break;
        }
    }
    if (max_fusable_iter_id == 0) { return ResultKind::kInvalid; }

    int fuse_to_iter_id = (*rand_gen)() % max_fusable_iter_id + 1;
    Array<Integer> fused_ids;
    for (int i = 0; i < fuse_to_iter_id; ++i) { fused_ids.push_back(i); }
    int iter_offset = fuse_step->fused_ids.back()->value - fused_ids.back()->value;
    if (iter_offset == 0) { return ResultKind::kInvalid; }
        
    auto new_fuse_step = FuseStep(stage_id, fused_ids);
    tmp_s.CopyOnWrite()->transform_steps.push_back(new_fuse_step);
    StepApplyToState(new_fuse_step, &tmp_s, policy->task_group->tasks[task_id]->compute_dag);
    tmp_s.CopyOnWrite()->transform_steps.push_back((*state)->transform_steps[step_id]);
    StepApplyToState((*state)->transform_steps[step_id], &tmp_s, policy->task_group->tasks[task_id]->compute_dag);

    for (size_t s = step_id + 1; s < (*state)->transform_steps.size(); ++s) {
        auto step = (*state)->transform_steps[s];
        if (step->stage_id == stage_id) {
            if (auto ps = step.as<AnnotationStepNode>()) { 
                if (ps->iter_id == 0) {
                    step = AnnotationStep(ps->stage_id, 0, ps->annotation);
                } else {
                    ICHECK_LE(ps->iter_id + iter_offset, tmp_s->stages[stage_id]->iters.size());
                    step = AnnotationStep(ps->stage_id, ps->iter_id + iter_offset, ps->annotation);
                }
            } else if (auto ps = step.as<PragmaStepNode>()) {
                if (ps->iter_id == 0) {
                    step = PragmaStep(ps->stage_id, 0, ps->pragma_type);
                } else {
                    ICHECK_LE(ps->iter_id + iter_offset, tmp_s->stages[stage_id]->iters.size());
                    step = PragmaStep(ps->stage_id, ps->iter_id + iter_offset, ps->pragma_type);
                }
            }else {
                return ResultKind::kInvalid;
            }
        }
        if (IsStageNumberChangingStep(step)) { return ResultKind::kInvalid; }
        tmp_s.CopyOnWrite()->transform_steps.push_back(step);
        try {
            StepApplyToState(tmp_s->transform_steps.back(), &tmp_s, policy->task_group->tasks[task_id]->compute_dag);
        } catch (Error& e) {
            return ResultKind::kInvalid;
        }
    }
    
    *state = tmp_s;
    return ResultKind::kValid;
}

}
}