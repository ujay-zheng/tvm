/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file auto_scheduler/auto_schedule.cc
 * \brief The user interface and tuning options of the TVM auto-scheduler.
 */

#include <tvm/auto_scheduler/auto_schedule.h>
#include <tvm/runtime/registry.h>

#include "utils.h"

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_NODE_TYPE(TuningOptionsNode);

TuningOptions::TuningOptions(int num_measure_trials, int early_stopping, int num_measures_per_round,
                             int verbose, ProgramBuilder builder, ProgramRunner runner,
                             Optional<Array<MeasureCallback>> measure_callbacks) {
  auto node = make_object<TuningOptionsNode>();
  node->num_measure_trials = num_measure_trials;
  node->early_stopping = early_stopping;
  node->num_measures_per_round = num_measures_per_round;
  node->verbose = verbose;
  node->builder = std::move(builder);
  node->runner = std::move(runner);
  node->measure_callbacks = std::move(measure_callbacks);
  data_ = std::move(node);
}

std::pair<te::Schedule, Array<te::Tensor>> AutoSchedule(SearchPolicy search_policy,
                                                        TuningOptions tuning_options) {
  // Create a ProgramMeasurer to handle the schedule build and performance measure
  ProgramMeasurer measurer =
      ProgramMeasurer(tuning_options->builder, tuning_options->runner,
                      tuning_options->measure_callbacks, tuning_options->verbose);
  // Search for the best schedule
  State state =
      search_policy->Search(tuning_options->num_measure_trials, tuning_options->early_stopping,
                            tuning_options->num_measures_per_round, measurer);
  if (state.defined()) {
    return search_policy->search_task->compute_dag.ApplySteps(state->transform_steps);
  } else {
    StdCout(tuning_options->verbose)
        << "No valid state found in this search round. Check if it has traversed all of the "
        << "search space." << std::endl;
    // Return the default schedule
    return {te::Schedule(search_policy->search_task->compute_dag->ops),
            search_policy->search_task->compute_dag->tensors};
  }
}

std::vector<std::pair<te::Schedule, Array<te::Tensor> > > AutoScheduleForGroup(
                                                    GroupSearchPolicy search_policy,
                                                    TuningOptions tuning_options,
                                                    Optional<Array<GroupMeasureCallback> > measure_callbacks,
                                                    int run_number,
                                                    int measure_loop_repeat, 
                                                    int n_parallel) {
  GroupMeasurer measurer = GroupMeasurer(measure_callbacks, run_number, measure_loop_repeat, n_parallel);
  Array<State> group_state = search_policy->Search(tuning_options->num_measure_trials, tuning_options->early_stopping,
                            tuning_options->num_measures_per_round, measurer);
  
  std::vector<std::pair<te::Schedule, Array<te::Tensor> > > ret;

  if(group_state.size() == 0) {
    StdCout(tuning_options->verbose)
        << "No valid group state found in this search round. Check if it has traversed all of the "
        << "search space." << std::endl;
    for(size_t task_id=0; task_id<search_policy->task_group->tasks.size(); task_id++) {
      ret.push_back({te::Schedule(search_policy->task_group->tasks[task_id]->compute_dag->ops),
                      search_policy->task_group->tasks[task_id]->compute_dag->tensors});
    }
  } else {
    for(size_t task_id=0; task_id<search_policy->task_group->tasks.size(); task_id++) {
      if (group_state[task_id].defined()) {
        ret.push_back(search_policy->task_group->tasks[task_id]->compute_dag.ApplySteps(group_state[task_id]->transform_steps));
      } else {
        StdCout(1)
          << task_id 
          << " failed "
          << "No valid state found in this search round. Check if it has traversed all of the "
          << "search space." << std::endl;
          ret.push_back({te::Schedule(search_policy->task_group->tasks[task_id]->compute_dag->ops),
                        search_policy->task_group->tasks[task_id]->compute_dag->tensors});
      }
    }
  }

  return ret;
}

TVM_REGISTER_GLOBAL("auto_scheduler.TuningOptions")
    .set_body_typed([](int num_measure_trials, int early_stopping, int num_measures_per_round,
                       int verbose, ProgramBuilder builder, ProgramRunner runner,
                       Optional<Array<MeasureCallback>> measure_callbacks) {
      return TuningOptions(num_measure_trials, early_stopping, num_measures_per_round, verbose,
                           builder, runner, measure_callbacks);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.AutoSchedule")
    .set_body_typed([](SearchPolicy search_policy, TuningOptions tuning_options) {
      auto [sch, return_tensors] = AutoSchedule(search_policy, tuning_options);
      return Array<ObjectRef>{sch, return_tensors};
    });

TVM_REGISTER_GLOBAL("auto_scheduler.AutoScheduleForGroup")
    .set_body_typed([](GroupSearchPolicy search_policy, TuningOptions tuning_options,
                      Optional<Array<GroupMeasureCallback>> measure_callbacks,
                      int run_number, int measure_loop_repeat, int n_parallel) {
      std::vector<std::pair<te::Schedule, Array<te::Tensor> > > imp = AutoScheduleForGroup(
                                                              search_policy, tuning_options,
                                                              measure_callbacks, run_number,
                                                              measure_loop_repeat, n_parallel);
      Array<Array<ObjectRef> > ret;
      for(size_t item_id=0; item_id<imp.size(); item_id++) {
        auto [sch, return_tensors] = imp[item_id];
        ret.push_back(Array<ObjectRef>{sch, return_tensors});
      }
      return ret;
    });
}  // namespace auto_scheduler
}  // namespace tvm
