/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TOOLS_EVALUATION_TASKS_TASK_EXECUTOR_H_
#define TENSORFLOW_LITE_TOOLS_EVALUATION_TASKS_TASK_EXECUTOR_H_

#include "absl/types/optional.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"

namespace tflite {
namespace evaluation {
// A common task execution API to avoid boilerpolate code in defining the main
// function.
class TaskExecutor {
 public:
  virtual ~TaskExecutor() {}
  // If the run is successful, the latest metrics will be returned.
  virtual absl::optional<EvaluationStageMetrics> Run() = 0;
};

// Just a declaration. In order to avoid the boilerpolate main-function code,
// every evaluation task should define this function.
std::unique_ptr<TaskExecutor> CreateTaskExecutor(int* argc, char* argv[]);
}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_TASKS_TASK_EXECUTOR_H_
