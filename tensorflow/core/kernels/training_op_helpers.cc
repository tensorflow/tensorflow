/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/training_op_helpers.h"

namespace tensorflow {

mutex* GetTrainingVariableMutex(OpKernelContext* ctx, int input) {
  if (ctx->input_dtype(input) == DT_RESOURCE) {
    Var* var;
    if (LookupResource(ctx, HandleFromInput(ctx, input), &var).ok()) {
      return var->mu();
    } else {
      ctx->CtxFailureWithWarning(
          errors::Internal("Invalid variable reference."));
      return nullptr;
    }
  }
  return ctx->input_ref_mutex(input);
}

// MaybeLockVariableInputMutexesInOrder is a helper function to acquire mutexes
// in address order to mitigate deadlock.  Returns a vector of acquired mutexes.
// Safe to pass duplicates - will only lock each distinct mutex once.  If
// do_lock is false, returns immediately.  Note that this silently doesn't lock
// mutexes for invalid variable references; in all usages this is followed by
// GetInputTensor which will signal a failure.
std::vector<mutex_lock> MaybeLockVariableInputMutexesInOrder(
    OpKernelContext* ctx, bool do_lock, const std::vector<int>& input_ids) {
  bool any_resource = false;
  for (auto i : input_ids) {
    if (ctx->input_dtype(i) == DT_RESOURCE) {
      any_resource = true;
      break;
    }
  }
  std::vector<mutex_lock> locks;
  if (!do_lock && !any_resource) {
    return locks;
  }
  std::vector<mutex*> mutexes;
  std::vector<int> acquire_order;
  for (auto input : input_ids) {
    mutex* mutex = GetTrainingVariableMutex(ctx, input);
    // Only lock each mutex once if duplicates exist (n^2 but n is 2 or 3).
    if (std::find(mutexes.begin(), mutexes.end(), mutex) == mutexes.end()) {
      acquire_order.push_back(mutexes.size());
      mutexes.push_back(mutex);
    }
  }
  std::sort(acquire_order.begin(), acquire_order.end(),
            [&mutexes](int a, int b) { return mutexes[a] < mutexes[b]; });

  for (auto input : acquire_order) {
    mutex* mu = GetTrainingVariableMutex(ctx, input);
    if (mu != nullptr) {
      locks.emplace_back(*mu);
    }
  }
  return locks;
}

void MaybeForwardRefInputToRefOutput(OpKernelContext* ctx, int input,
                                     int output) {
  if (ctx->input_dtype(input) != DT_RESOURCE) {
    ctx->forward_ref_input_to_ref_output(input, output);
  }
}

}  // end namespace tensorflow
