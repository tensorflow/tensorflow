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
#include "tensorflow/lite/tools/optimize/reduced_precision_support.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/tools/optimize/test_util.h"

namespace tflite {
namespace optimize {
namespace utils {
namespace {

class ReducedPrecisionSupportTest : public testing::Test {
 protected:
  tflite::TestErrorReporter error_reporter_;
};

TEST_F(ReducedPrecisionSupportTest, BitwiseOps) {
  ReducedPrecisionSupport mask0 = ReducedPrecisionSupport::None;
  ReducedPrecisionSupport mask1 = ReducedPrecisionSupport::Float16Inference;
  ReducedPrecisionSupport bf16 = ReducedPrecisionSupport::Bfloat16Inference;
  ReducedPrecisionSupport fp16 = ReducedPrecisionSupport::Float16Inference;
  EXPECT_EQ(mask0, mask0 & mask1);
  EXPECT_EQ(mask1, mask0 | mask1);
  mask0 |= fp16;
  EXPECT_EQ(true, SupportsFP16Inference(mask0));
  mask0 |= bf16;
  EXPECT_EQ(true, SupportsBfloat16Inference(mask0));
  ReducedPrecisionSupport mask2 = ReducedPrecisionSupport::Float16Accumulation;
  mask2 &= fp16;
  EXPECT_EQ(mask2, ReducedPrecisionSupport::None);
}

TEST_F(ReducedPrecisionSupportTest, SupportTests) {
  ReducedPrecisionSupport bf16 = ReducedPrecisionSupport::Bfloat16Inference;
  ReducedPrecisionSupport fp16 = ReducedPrecisionSupport::Float16Inference;
  ReducedPrecisionSupport mask = bf16 | fp16;
  EXPECT_EQ(true, SupportsFP16Inference(mask));
  EXPECT_EQ(true, SupportsBfloat16Inference(mask));
  EXPECT_EQ(false, SupportsFP16Accumulation(mask));
  EXPECT_EQ(false, SupportsFP32Accumulation(mask));
  EXPECT_EQ(true, SupportsReducedPrecisionInference(mask));
  EXPECT_EQ(true, SupportsReducedPrecisionInference(mask));
  EXPECT_EQ(false, SupportsEitherFP16OrFP32Accumulation(mask));
  mask = mask | ReducedPrecisionSupport::Float16Accumulation;
  EXPECT_EQ(true, SupportsFP16Accumulation(mask));
  EXPECT_EQ(false, SupportsFP32Accumulation(mask));
  EXPECT_EQ(true, SupportsEitherFP16OrFP32Accumulation(mask));
}

TEST_F(ReducedPrecisionSupportTest, MetadataStrings) {
  ReducedPrecisionSupport bf16 = ReducedPrecisionSupport::Bfloat16Inference;
  ReducedPrecisionSupport fp16 = ReducedPrecisionSupport::Float16Inference;
  ReducedPrecisionSupport accfp32 =
      ReducedPrecisionSupport::Float32Accumulation;
  ReducedPrecisionSupport accfp16 =
      ReducedPrecisionSupport::Float16Accumulation;
  ReducedPrecisionSupport maskA = bf16 | fp16 | accfp32;
  std::pair<std::string, std::string> ans =
      MetadataForReducedPrecisionSupport(maskA);
  EXPECT_EQ("fp16bf16accfp32", ans.second);
  EXPECT_EQ("reduced_precision_support", ans.first);
  ReducedPrecisionSupport maskB = fp16 | accfp16;
  EXPECT_EQ("fp16accfp16", MetadataForReducedPrecisionSupport(maskB).second);
}

TEST_F(ReducedPrecisionSupportTest, ReadStringsIntoMasks) {
  ReducedPrecisionSupport fp16 = ReducedPrecisionSupport::Float16Inference;
  ReducedPrecisionSupport accfp16 =
      ReducedPrecisionSupport::Float16Accumulation;
  ReducedPrecisionSupport maskfp16 = fp16;
  ReducedPrecisionSupport maskfp16accfp16 = fp16 | accfp16;
  ReducedPrecisionSupport mask = ReducedPrecisionSupport::None;
  size_t idx = 0;
  std::string metadata = "fp16accfp16";
  EXPECT_EQ(true, ReadInferenceType(metadata, &idx, &mask));
  EXPECT_EQ(maskfp16, mask);
  EXPECT_EQ(idx, 4);
  idx = 7;
  EXPECT_EQ(true, ReadAccumulationType(metadata, &idx, &mask));
  EXPECT_EQ(maskfp16accfp16, mask);
  EXPECT_EQ(idx, 11);
}

TEST_F(ReducedPrecisionSupportTest, SetMasks) {
  ReducedPrecisionSupport fp16 = ReducedPrecisionSupport::Float16Inference;
  ReducedPrecisionSupport bf16 = ReducedPrecisionSupport::Bfloat16Inference;
  ReducedPrecisionSupport accfp16 =
      ReducedPrecisionSupport::Float16Accumulation;
  ReducedPrecisionSupport accfp32 =
      ReducedPrecisionSupport::Float32Accumulation;
  ReducedPrecisionSupport mask = ReducedPrecisionSupport::None;
  EXPECT_EQ(true, SetMaskFromReducedPrecisionMetadata("bf16accfp32", &mask));
  EXPECT_EQ(mask, bf16 | accfp32);
  mask = ReducedPrecisionSupport::None;
  EXPECT_EQ(true, SetMaskFromReducedPrecisionMetadata("fp16accfp16", &mask));
  EXPECT_EQ(mask, fp16 | accfp16);
  mask = ReducedPrecisionSupport::None;
  EXPECT_EQ(true,
            SetMaskFromReducedPrecisionMetadata("fp16bf16accfp32", &mask));
  EXPECT_EQ(mask, fp16 | bf16 | accfp32);
  mask = ReducedPrecisionSupport::None;
  EXPECT_EQ(false, SetMaskFromReducedPrecisionMetadata("accfp32", &mask));
  EXPECT_EQ(mask, ReducedPrecisionSupport::None);
  EXPECT_EQ(false, SetMaskFromReducedPrecisionMetadata("qwerwer", &mask));
  EXPECT_EQ(mask, ReducedPrecisionSupport::None);
  EXPECT_EQ(false,
            SetMaskFromReducedPrecisionMetadata("fp16accfp32fp16", &mask));
  EXPECT_EQ(mask, ReducedPrecisionSupport::None);
  EXPECT_EQ(false, SetMaskFromReducedPrecisionMetadata("fp16accbf16", &mask));
  EXPECT_EQ(mask, ReducedPrecisionSupport::None);
}

}  // namespace
}  // namespace utils
}  // namespace optimize
}  // namespace tflite

int main(int argc, char** argv) {
  ::tensorflow::port::InitMain(argv[0], &argc, &argv);
  return RUN_ALL_TESTS();
}
