/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_FACTORY_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_FACTORY_H_

#include <string>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Executor;
class Graph;
struct LocalExecutorParams;

class ExecutorFactory {
 public:
  virtual Status NewExecutor(const LocalExecutorParams& params,
                             std::unique_ptr<const Graph> graph,
                             std::unique_ptr<Executor>* out_executor) = 0;
  virtual ~ExecutorFactory() {}

  static void Register(const string& executor_type, ExecutorFactory* factory);
  static Status GetFactory(const string& executor_type,
                           ExecutorFactory** out_factory);
};

Status NewExecutor(const string& executor_type,
                   const LocalExecutorParams& params,
                   std::unique_ptr<const Graph> graph,
                   std::unique_ptr<Executor>* out_executor);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_FACTORY_H_
