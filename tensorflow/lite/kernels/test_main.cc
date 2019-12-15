/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/tools/command_line_flags.h"

namespace {

void InitKernelTest(int* argc, char** argv) {
  bool use_nnapi = false;
  std::vector<tflite::Flag> flags = {
      tflite::Flag::CreateFlag("use_nnapi", &use_nnapi, "Use NNAPI"),
  };
  tflite::Flags::Parse(argc, const_cast<const char**>(argv), flags);

  if (use_nnapi) {
    tflite::SingleOpModel::SetForceUseNnapi(true);
  }
}

}  // namespace

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  InitKernelTest(&argc, argv);
  return RUN_ALL_TESTS();
}
