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
#ifndef TENSORFLOW_C_EAGER_TFE_EXECUTOR_INTERNAL_H_
#define TENSORFLOW_C_EAGER_TFE_EXECUTOR_INTERNAL_H_

#include <memory>

#include "tensorflow/core/common_runtime/eager/eager_executor.h"

struct TFE_Executor {
  explicit TFE_Executor(bool async)
      : owned_executor(new tensorflow::EagerExecutor(async)) {}

  explicit TFE_Executor(tensorflow::EagerExecutor* executor)
      : owned_executor(nullptr), unowned_executor(executor) {}

  tensorflow::EagerExecutor* executor() {
    return owned_executor == nullptr ? unowned_executor : owned_executor.get();
  }

  std::unique_ptr<tensorflow::EagerExecutor> owned_executor;
  tensorflow::EagerExecutor* unowned_executor;
};

#endif  // TENSORFLOW_C_EAGER_TFE_EXECUTOR_INTERNAL_H_
