#include <tvm/auto_scheduler/measure.h>
#include <tvm/runtime/registry.h>
#include <algorithm>

#include "search_policy/empty_policy.h"
#include "search_policy/sketch_policy.h"
#include "utils.h"

namespace tvm {
namespace auto_scheduler {

/********** Group Measure **********/
TVM_REGISTER_OBJECT_TYPE(GroupMeasurerNode);
TVM_REGISTER_OBJECT_TYPE(GroupMeasureCallbackNode);

GroupMeasurer::GroupMeasurer(Optional<Array<GroupMeasureCallback> > callbacks, int run_number, int measure_loop_repeat, int n_parallel){
    auto node = make_object<GroupMeasurerNode>();
    node->callbacks=std::move(callbacks);
    node->n_parallel=n_parallel;
    node->run_number=run_number;
    node->measure_loop_repeat=measure_loop_repeat;
    node->max_continuous_error = ProgramMeasurerNode::DEFAULT_MAX_CONTINUOUS_ERROR;
    data_ = std::move(node);
}

void GroupMeasurerNode::Reset() {
  ct = error_ct = 0;
  best_flops.clear();
  best_ct.clear();
  best_state.clear();
  has_valid.clear();
}

Array<MeasureResult> GroupMeasurerNode::Measure(const SearchTaskGroup& task_group, 
                                                const GroupSketchPolicy policy,
                                                const Array<Array<MeasureInput> >& inputs){
    // auto t_begin = std::chrono::high_resolution_clock::now();
    Array<MeasureResult> results;
    results.reserve(inputs.size());

    double totoal_flop_ct = 0.0;
    for(const auto& task : task_group->tasks){
        totoal_flop_ct += task->compute_dag->flop_ct;
    } 
    
    for (size_t i = 0; i < inputs.size(); i += n_parallel) {
        Array<Array<MeasureInput> > group_measure_inputs(
            inputs.begin() + i, inputs.begin() + std::min(i + n_parallel, inputs.size()));

        // build and run
        Array<MeasureResult> group_measure_results = BuildAndRun(group_measure_inputs, task_group->launch_id_list);
        // update current best state according to the new measure result
        for (size_t j = 0; j < group_measure_inputs.size(); ++j) {
            String workload_key;
            for(const auto& measure_info : group_measure_inputs[j]){
                workload_key = workload_key+measure_info->task->workload_key+"_";
            }
            double flops;
            if (group_measure_results[j]->error_no == 0) {
                flops = totoal_flop_ct / FloatArrayMean(group_measure_results[j]->costs);
                error_ct = 0;
                has_valid.insert(workload_key);
            }else {
                flops = 0.0;
                error_ct++;
            }
            if (flops > best_flops[workload_key]) {
                best_flops[workload_key] = flops;
                Array<State> group_best_state;
                for(const auto& measure_info : group_measure_inputs[j]){
                    group_best_state.push_back(measure_info->state);
                }
                best_state[workload_key] = group_best_state;
                best_ct[workload_key] = ct;
            }
            ct++;
        }

        if(callbacks) {
            for (const auto& callback : callbacks.value()) {
                callback->Callback(group_measure_inputs, group_measure_results);
            }
        }
        
        for (auto& res : group_measure_results) {
            results.push_back(res);
        }
    }
    return results;
}

Array<MeasureResult> GroupMeasurerNode::BuildAndRun(const Array<Array<MeasureInput> >& group_measure_inputs, const Array<Array<Integer> > launch_id_list){
    if (const auto* f = runtime::Registry::Get("auto_scheduler.group_measurer.build_and_run")) {
        Array<MeasureResult> results = (*f)(group_measure_inputs, launch_id_list, measure_loop_repeat);
std::cout << "================================BuildAndRun==============================" << std::endl;
        return results;
    }
    LOG(FATAL)  << "auto_scheduler.group_builder_and_runner.group_build_and_run is not registered. "
                << "This is a function registered in Python, "
                << "make sure the TVM Python runtime has been loaded successfully.";
    throw;
}

TVM_REGISTER_GLOBAL("auto_scheduler.GroupMeasurer")
    .set_body_typed([](Array<GroupMeasureCallback> callbacks, int run_number, int measure_loop_repeat, int n_parallel) {
      return GroupMeasurer(callbacks, run_number, measure_loop_repeat, n_parallel);
    });

}
}