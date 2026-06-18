/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/distributed_runtime/partial_run_mgr.h"


namespace tensorflow {

bool PartialRunMgr::FindOrCreate(int step_id,
                                 CancellationManager** cancellation_manager) {
  mutex_lock l(mu_);
  auto it = step_id_to_partial_run_.find(step_id);
  if (it != step_id_to_partial_run_.end()) {
    *cancellation_manager = it->second->cancellation_manager.get();
    return false;
  }

  std::unique_ptr<PartialRunState> partial_run =
      std::make_unique<PartialRunState>();
  partial_run->cancellation_manager = std::make_unique<CancellationManager>();
  *cancellation_manager = partial_run->cancellation_manager.get();
  step_id_to_partial_run_[step_id] = std::move(partial_run);
  return true;
}

void PartialRunMgr::ExecutorDone(int step_id,
                                 const absl::Status& executor_status) {
  StatusCallback done;
  absl::Status callback_status;
  {
    mutex_lock l(mu_);
    auto run_it = step_id_to_partial_run_.find(step_id);
    if (run_it == step_id_to_partial_run_.end()) {
      return;
    }
    // If we found the partial_run, we call the final callback, if it
    // exists.
    // It is guaranteed that run_it->second->final_callback is left empty
    // after the std::move call.
    done = std::move(run_it->second->final_callback);
    if (!executor_status.ok()) {
      run_it->second->final_status = executor_status;
    }
    callback_status = run_it->second->final_status;
    run_it->second->executor_done = true;
  }
  if (done != nullptr) {
    done(callback_status);
    mutex_lock l(mu_);
    step_id_to_partial_run_.erase(step_id);
  }
}

void PartialRunMgr::PartialRunDone(int step_id, StatusCallback done,
                                   const absl::Status& status) {
  absl::Status callback_status;
  {
    mutex_lock l(mu_);
    auto run_it = step_id_to_partial_run_.find(step_id);
    if (run_it == step_id_to_partial_run_.end()) {
      return;
    }
    run_it->second->final_status.Update(status);
    if (!run_it->second->executor_done) {
      // If we found the partial_run, we set the final callback to call only
      // when the executor is completely done.
      run_it->second->final_callback = std::move(done);
      return;
    }
    callback_status = run_it->second->final_status;
  }
  // Otherwise we call the callback immediately.
  done(callback_status);
  mutex_lock l(mu_);
  step_id_to_partial_run_.erase(step_id);
}

}  // namespace tensorflow
