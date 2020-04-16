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
#include "tensorflow/lite/tools/evaluation/evaluation_delegate_provider.h"

#include <gtest/gtest.h>
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"

namespace tflite {
namespace evaluation {
namespace {
TEST(EvaluationDelegateProviderTest, ParseStringToDelegateType) {
  EXPECT_EQ(TfliteInferenceParams::NNAPI, ParseStringToDelegateType("nnapi"));
  EXPECT_EQ(TfliteInferenceParams::GPU, ParseStringToDelegateType("gpu"));
  EXPECT_EQ(TfliteInferenceParams::HEXAGON,
            ParseStringToDelegateType("hexagon"));
  EXPECT_EQ(TfliteInferenceParams::XNNPACK,
            ParseStringToDelegateType("xnnpack"));

  EXPECT_EQ(TfliteInferenceParams::NONE, ParseStringToDelegateType("Gpu"));
  EXPECT_EQ(TfliteInferenceParams::NONE, ParseStringToDelegateType("Testing"));
}

TEST(EvaluationDelegateProviderTest, CreateTfLiteDelegate) {
  TfliteInferenceParams params;
  params.set_delegate(TfliteInferenceParams::NONE);
  // A NONE delegate type will return a nullptr TfLite delegate ptr.
  EXPECT_TRUE(!CreateTfLiteDelegate(params));
}

TEST(EvaluationDelegateProviderTest, DelegateProvidersParams) {
  DelegateProviders providers;
  const auto& params = providers.GetAllParams();
  EXPECT_TRUE(params.HasParam("use_nnapi"));
  EXPECT_TRUE(params.HasParam("use_gpu"));

  int argc = 3;
  const char* argv[] = {"program_name", "--use_gpu=true",
                        "--other_undefined_flag=1"};
  EXPECT_TRUE(providers.InitFromCmdlineArgs(&argc, argv));
  EXPECT_TRUE(params.Get<bool>("use_gpu"));
  EXPECT_EQ(2, argc);
  EXPECT_EQ("--other_undefined_flag=1", argv[1]);
}

}  // namespace
}  // namespace evaluation
}  // namespace tflite
