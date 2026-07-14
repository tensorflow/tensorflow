/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_SELECT_V2_TEST_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_SELECT_V2_TEST_UTIL_H_

#include "absl/status/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"

namespace tflite {
namespace gpu {

absl::Status SelectV2Test(TestExecutionEnvironment* env);

absl::Status SelectV2BatchTest(TestExecutionEnvironment* env);

absl::Status SelectV2ChannelsTest(TestExecutionEnvironment* env);

absl::Status SelectV2ChannelsBatchTest(TestExecutionEnvironment* env);

absl::Status SelectV2BroadcastTrueTest(TestExecutionEnvironment* env);

absl::Status SelectV2BroadcastFalseTest(TestExecutionEnvironment* env);

absl::Status SelectV2BroadcastBothTest(TestExecutionEnvironment* env);

absl::Status SelectV2ChannelsBroadcastFalseTest(TestExecutionEnvironment* env);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_SELECT_V2_TEST_UTIL_H_
