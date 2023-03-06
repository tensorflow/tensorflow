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
#include "tensorflow/lite/delegates/flex/allowlisted_flex_ops.h"

#include <set>
#include <string>

#include <gtest/gtest.h>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/lite/delegates/flex/allowlisted_flex_ops_internal.h"

namespace tflite {
namespace flex {

// Get all cpu kernels registered in Tensorflow.
std::set<std::string> GetAllCpuKernels() {
  auto is_cpu_kernel = [](const tensorflow::KernelDef& def) {
    return (def.device_type() == "CPU" || def.device_type() == "DEFAULT");
  };

  tensorflow::KernelList kernel_list =
      tensorflow::GetFilteredRegisteredKernels(is_cpu_kernel);
  std::set<std::string> result;

  for (int i = 0; i < kernel_list.kernel_size(); ++i) {
    tensorflow::KernelDef kernel_def = kernel_list.kernel(i);
    result.insert(kernel_def.op());
  }
  return result;
}

// Test if every flex op has their kernel included in the flex delegate library.
// This test must be run on both Linux and Android.
TEST(DISABLED_AllowlistedFlexOpsTest, EveryOpHasKernel) {
  const std::set<std::string>& allowlist = GetFlexAllowlist();
  std::set<std::string> all_kernels = GetAllCpuKernels();

  for (const std::string& op_name : allowlist) {
    EXPECT_EQ(all_kernels.count(op_name), 1)
        << op_name << " op is added to flex allowlist "
        << "but its kernel is not found.";
  }
}

TEST(TfTextUtilsTest, TestFlexOpAllowed) {
  // Expect false since ConstrainedSequence kernel is not registered.
  EXPECT_FALSE(IsAllowedTFTextOpForFlex("ConstrainedSequence"));
}

TEST(TfTextUtilsTest, TestFlexOpNotAllowed) {
  EXPECT_FALSE(IsAllowedTFTextOpForFlex("ngrams"));
}

}  // namespace flex
}  // namespace tflite
