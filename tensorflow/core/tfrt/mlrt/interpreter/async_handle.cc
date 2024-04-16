/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/mlrt/interpreter/async_handle.h"

#include <memory>
#include <utility>

#include "tensorflow/core/tfrt/mlrt/interpreter/context.h"
#include "tsl/concurrency/async_value_ref.h"
#include "tsl/concurrency/chain.h"

namespace mlrt {

std::pair<AsyncHandle::Promise, AsyncHandle> AsyncHandle::Allocate(
    const ExecutionContext& current) {
  auto user_contexts = current.CopyUserContexts();

  auto new_context = std::make_unique<ExecutionContext>(
      &current.loaded_executable(), std::move(user_contexts));
  new_context->set_work_queue(current.work_queue());

  auto shared_state = tsl::MakeConstructedAsyncValueRef<tsl::Chain>();

  Promise promise(shared_state);
  AsyncHandle handle(std::move(new_context), std::move(shared_state));
  return {std::move(promise), std::move(handle)};
}

}  // namespace mlrt
