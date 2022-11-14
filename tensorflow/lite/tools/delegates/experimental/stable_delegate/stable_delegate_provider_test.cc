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
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/tool_params.h"

namespace tflite {
namespace tools {
namespace {

TEST(StableAbiDelegateProviderTest, CreateDelegate) {
  ToolParams params;
  ProvidedDelegateList providers(&params);
  providers.AddAllDelegateParams();
  params.Set<std::string>(
      "stable_delegate_path",
      "third_party/tensorflow/lite/delegates/utils/experimental/"
      "sample_stable_delegate/libtensorflowlite_sample_stable_delegate.so",
      /*position=*/1);

  auto delegates = providers.CreateAllRankedDelegates();

  // Only the stable ABI delegate is registered.
  EXPECT_EQ(1, delegates.size());
  EXPECT_EQ("STABLE_DELEGATE", delegates.front().provider->GetName());
  EXPECT_NE(nullptr, delegates.front().delegate.get());
  EXPECT_EQ(1, delegates.front().rank);
}

TEST(StableAbiDelegateProviderTest, CreateDelegateFailedWithInvalidLibPath) {
  ToolParams params;
  ProvidedDelegateList providers(&params);
  providers.AddAllDelegateParams();
  params.Set<std::string>("stable_delegate_path", "invalid.so",
                          /*position=*/1);

  auto delegates = providers.CreateAllRankedDelegates();

  EXPECT_EQ(0, delegates.size());
}

TEST(StableAbiDelegateProviderTest, CreateDelegateFailedWithInvalidSymbolName) {
  ToolParams params;
  ProvidedDelegateList providers(&params);
  providers.AddAllDelegateParams();
  params.Set<std::string>(
      "stable_delegate_path",
      "third_party/tensorflow/lite/delegates/utils/experimental/"
      "sample_stable_delegate/libtensorflowlite_sample_stable_delegate.so",
      /*position=*/1);
  std::vector<std::string> invalid_symbol_names = {"", "invalid_symbol"};
  for (const auto& symbol_name : invalid_symbol_names) {
    params.Set<std::string>("stable_delegate_plugin_symbol", symbol_name);

    auto delegates = providers.CreateAllRankedDelegates();

    EXPECT_EQ(0, delegates.size());
  }
}

}  // namespace
}  // namespace tools
}  // namespace tflite
