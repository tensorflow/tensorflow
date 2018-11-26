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

#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {

mutex* GetTrainingVariableMutex(OpKernelContext* ctx, int input,
                                Var** maybe_resource) {
  *maybe_resource = nullptr;
  if (ctx->input_dtype(input) == DT_RESOURCE) {
    if (LookupResource(ctx, HandleFromInput(ctx, input), maybe_resource).ok()) {
      return (*maybe_resource)->mu();
    } else {
      ctx->CtxFailureWithWarning(
          errors::Internal("Invalid variable reference."));
      return nullptr;
    }
  }
  return ctx->input_ref_mutex(input);
}

// MaybeLockVariableInputMutexesInOrder is a helper function to acquire mutexes
// in address order to mitigate deadlock.  Returns a structure that, when
// deleted, will release the acquired mutexes. Safe to pass duplicates - will
// only lock each distinct mutex once.  If do_lock is false, returns
// immediately.  Note that this silently doesn't lock mutexes for invalid
// variable references; in all usages this is followed by GetInputTensor which
// will signal a failure.
VariableInputLockHolder MaybeLockVariableInputMutexesInOrder(
    OpKernelContext* ctx, bool do_lock, const std::vector<int>& input_ids) {
  bool any_resource = false;
  for (auto i : input_ids) {
    if (ctx->input_dtype(i) == DT_RESOURCE) {
      any_resource = true;
      break;
    }
  }
  if (!do_lock && !any_resource) {
    return VariableInputLockHolder({}, {});
  }
  std::vector<Var*> vars;
  std::vector<mutex*> mutexes;
  std::vector<int> acquire_order;
  for (auto input : input_ids) {
    Var* var;
    mutex* mutex = GetTrainingVariableMutex(ctx, input, &var);
    if (var) vars.push_back(var);
    // Only lock each mutex once if duplicates exist (n^2 but n is 2 or 3).
    if (std::find(mutexes.begin(), mutexes.end(), mutex) == mutexes.end()) {
      acquire_order.push_back(mutexes.size());
      mutexes.push_back(mutex);
    }
  }
  std::sort(acquire_order.begin(), acquire_order.end(),
            [&mutexes](int a, int b) { return mutexes[a] < mutexes[b]; });

  std::unique_ptr<std::vector<mutex_lock>> locks =
      MakeUnique<std::vector<mutex_lock>>();
  locks->reserve(acquire_order.size());

  for (auto input : acquire_order) {
    Var* var;
    mutex* mu = GetTrainingVariableMutex(ctx, input, &var);
    core::ScopedUnref scoped_unref(var);
    if (mu != nullptr) {
      locks->emplace_back(*mu);
    }
  }
  return VariableInputLockHolder(std::move(vars), std::move(locks));
}

void MaybeForwardRefInputToRefOutput(OpKernelContext* ctx, int input,
                                     int output) {
  if (ctx->input_dtype(input) != DT_RESOURCE) {
    ctx->forward_ref_input_to_ref_output(input, output);
  }
}

}  // end namespace tensorflow
