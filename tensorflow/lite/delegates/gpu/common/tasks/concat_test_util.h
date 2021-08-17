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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONCAT_TEST_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONCAT_TEST_UTIL_H_

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"

namespace tflite {
namespace gpu {

absl::Status ConcatWidthTest(TestExecutionEnvironment* env);
absl::Status ConcatHeightTest(TestExecutionEnvironment* env);
absl::Status ConcatChannelsTest(TestExecutionEnvironment* env);
absl::Status ConcatChannelsAlignedx4Test(TestExecutionEnvironment* env);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONCAT_TEST_UTIL_H_
