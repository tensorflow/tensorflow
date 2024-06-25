/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/tool_params.h"

namespace tflite {
namespace tools {
namespace {

TEST(XNNPackDelegateProviderTest, Test) {
  const std::string kFakeCacheParam =
      testing::TempDir() + "/XNNPackDelegateProviderTest.xnnpack_cache";

  const auto& providers = GetRegisteredDelegateProviders();
  ASSERT_EQ(providers.size(), 1);
  ToolParams params;

  const auto& xnnpack_provider = providers[0];
  ASSERT_NE(xnnpack_provider, nullptr);

  params.Merge(xnnpack_provider->DefaultParams());
  params.AddParam("num_threads", ToolParam::Create<int32_t>(-1));

  EXPECT_TRUE(params.HasParam("use_xnnpack"));
  EXPECT_FALSE(params.HasValueSet<bool>("use_xnnpack"));
  ASSERT_NE(params.GetParam("use_xnnpack"), nullptr);

  EXPECT_TRUE(params.HasParam("xnnpack_force_fp16"));
  EXPECT_FALSE(params.HasValueSet<bool>("xnnpack_force_fp16"));
  ASSERT_NE(params.GetParam("xnnpack_force_fp16"), nullptr);

  EXPECT_TRUE(params.HasParam("xnnpack_experimental_weight_cache_file_path"));
  EXPECT_FALSE(params.HasValueSet<std::string>(
      "xnnpack_experimental_weight_cache_file_path"));
  ASSERT_NE(params.GetParam("xnnpack_experimental_weight_cache_file_path"),
            nullptr);

  params.Set<bool>("use_xnnpack", true, /*position=*/0);

  {
    TfLiteDelegatePtr delegate = xnnpack_provider->CreateTfLiteDelegate(params);
    const TfLiteXNNPackDelegateOptions* options =
        TfLiteXNNPackDelegateGetOptions(delegate.get());
    ASSERT_NE(options, nullptr);
    EXPECT_EQ(options->experimental_weight_cache_file_path, nullptr);
  }

  params.Set<bool>("xnnpack_force_fp16", true, /*position=*/1);
  params.Set<std::string>("xnnpack_experimental_weight_cache_file_path",
                          kFakeCacheParam, /*position=*/2);
  {
    TfLiteDelegatePtr delegate = xnnpack_provider->CreateTfLiteDelegate(params);
    const TfLiteXNNPackDelegateOptions* options =
        TfLiteXNNPackDelegateGetOptions(delegate.get());
    ASSERT_NE(options, nullptr);
    EXPECT_THAT(options->experimental_weight_cache_file_path,
                testing::StrEq(kFakeCacheParam));
    EXPECT_TRUE(options->flags & TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16);
  }
}

}  // namespace
}  // namespace tools
}  // namespace tflite
