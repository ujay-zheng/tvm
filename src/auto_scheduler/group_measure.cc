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

GroupMeasurer::GroupMeasurer(Optional<Array<GroupMeasureCallback> > callbacks, int measure_loop_repeat){
    auto node = make_object<GroupMeasurerNode>();
    node->callbacks=std::move(callbacks);
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
                                                const Array<Array<MeasureInput> >& measure_inputs){
    // auto t_begin = std::chrono::high_resolution_clock::now();
    Array<MeasureResult> results;
    results.reserve(measure_inputs.size());

    double totoal_flop_ct = 0.0;
    for(const auto& task : task_group->tasks){
        totoal_flop_ct += task->compute_dag->flop_ct;
    } 

    // build and run all measure inputs
    Array<MeasureResult> measure_results = BuildAndRun(measure_inputs, task_group->topological_seq, task_group->streams_num, task_group->events_num);
    // update current best state according to the new measure result
    for (size_t i = 0; i < measure_results.size(); ++i) {
        String workload_key;
        for(const auto& measure_info : measure_inputs[i]){
            workload_key = workload_key+measure_info->task->workload_key+"_";
        }
        double flops;
        if (measure_results[i]->error_no == 0) {
            flops = totoal_flop_ct / FloatArrayMean(measure_results[i]->costs);
            error_ct = 0;
            has_valid.insert(workload_key);
        }else {
            flops = 0.0;
            error_ct++;
        }
        if (flops > best_flops[workload_key]) {
            best_flops[workload_key] = flops;
            Array<State> group_best_state;
            for(const auto& measure_info : measure_inputs[i]){
                group_best_state.push_back(measure_info->state);
            }
            best_state[workload_key] = group_best_state;
            best_ct[workload_key] = ct;
        }
        ct++;
    }

    if(callbacks) {
        for (const auto& callback : callbacks.value()) {
            callback->Callback(measure_inputs, measure_results);
        }
    }
        
    for (auto& res : measure_results) {
        results.push_back(res);
    }
    return results;
}

Array<MeasureResult> GroupMeasurerNode::BuildAndRun(const Array<Array<MeasureInput> >& group_measure_inputs, const Array<Array<Integer> > topological_seq, int streams_num, int events_num){
    if (const auto* f = runtime::Registry::Get("auto_scheduler.group_measurer.build_and_run")) {
        Array<MeasureResult> results = (*f)(group_measure_inputs, topological_seq, streams_num, events_num, measure_loop_repeat);
        return results;
    }
    LOG(FATAL)  << "auto_scheduler.group_builder_and_runner.group_build_and_run is not registered. "
                << "This is a function registered in Python, "
                << "make sure the TVM Python runtime has been loaded successfully.";
    throw;
}

TVM_REGISTER_GLOBAL("auto_scheduler.GroupMeasurer")
    .set_body_typed([](Array<GroupMeasureCallback> callbacks, int measure_loop_repeat) {
      return GroupMeasurer(callbacks, measure_loop_repeat);
    });

}
}