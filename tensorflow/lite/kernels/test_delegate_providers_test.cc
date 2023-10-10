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
#include "tensorflow/lite/kernels/test_delegate_providers.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace {
TEST(KernelTestDelegateProvidersTest, DelegateProvidersParams) {
  KernelTestDelegateProviders providers;
  const auto& params = providers.ConstParams();
  EXPECT_TRUE(params.HasParam("use_xnnpack"));
  EXPECT_TRUE(params.HasParam("use_nnapi"));

  int argc = 3;
  const char* argv[] = {"program_name", "--use_nnapi=true",
                        "--other_undefined_flag=1"};
  EXPECT_TRUE(providers.InitFromCmdlineArgs(&argc, argv));
  EXPECT_TRUE(params.Get<bool>("use_nnapi"));
  EXPECT_EQ(2, argc);
  EXPECT_EQ("--other_undefined_flag=1", argv[1]);
}

TEST(KernelTestDelegateProvidersTest, CreateTfLiteDelegates) {
#if !defined(__Fuchsia__) && !defined(__s390x__) && \
    !defined(TFLITE_WITHOUT_XNNPACK)
  KernelTestDelegateProviders providers;
  providers.MutableParams()->Set<bool>("use_xnnpack", true);
  EXPECT_GE(providers.CreateAllDelegates().size(), 1);

  tools::ToolParams local_params;
  local_params.Merge(providers.ConstParams());
  local_params.Set<bool>("use_xnnpack", false);
  EXPECT_TRUE(providers.CreateAllDelegates(local_params).empty());
#endif
}
}  // namespace
}  // namespace tflite
