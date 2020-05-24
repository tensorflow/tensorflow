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
#include "absl/types/optional.h"
#include "tensorflow/lite/tools/evaluation/tasks/task_executor.h"
#include "tensorflow/lite/tools/logging.h"

// This could serve as the main function for all eval tools.
int main(int argc, char* argv[]) {
  auto task_executor = tflite::evaluation::CreateTaskExecutor(&argc, argv);
  if (task_executor == nullptr) {
    TFLITE_LOG(ERROR) << "Could not create the task evaluation!";
    return EXIT_FAILURE;
  }
  const auto metrics = task_executor->Run();
  if (!metrics.has_value()) {
    TFLITE_LOG(ERROR) << "Could not run the task evaluation!";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
