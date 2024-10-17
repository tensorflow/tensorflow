// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/lrt/core/experimental/litert_model_serialize.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/core/flatbuffer_utils.h"
#include "tensorflow/lite/experimental/lrt/core/graph_tools.h"
#include "tensorflow/lite/experimental/lrt/core/lite_rt_model_init.h"
#include "tensorflow/lite/experimental/lrt/core/model.h"
#include "tensorflow/lite/experimental/lrt/test/common.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

using ::graph_tools::GetMetadata;
using ::litert::internal::FbBufToStr;
using ::litert::internal::FbCharT;
using ::litert::internal::VerifyFlatbuffer;
using ::lrt::testing::LoadTestFileModel;
using ::testing::HasSubstr;

static constexpr absl::string_view kSocModel = "TestSocModel";
static constexpr absl::string_view kSocMan = "TestSocMan";

// Gets a test model with a single custom op with empty attributes.
UniqueLrtModel GetTestModel() {
  static constexpr absl::string_view kTestModel = "one_mul.tflite";
  return LoadTestFileModel(kTestModel);
}

UniqueLrtModel RoundTrip(UniqueLrtModel model) {
  uint8_t* buf;
  size_t size;
  size_t offset;
  if (SerializeModel(model.release(), &buf, &size, &offset) != kLrtStatusOk) {
    delete[] buf;
    return {};
  }

  auto out_buf = buf + offset;
  const size_t out_size = size - offset;
  if (!VerifyFlatbuffer(out_buf, out_size)) {
    delete[] buf;
    return {};
  }

  LrtModel new_model;
  if (LoadModel(out_buf, out_size, &new_model) != kLrtStatusOk) {
    delete[] buf;
    return {};
  }
  delete[] buf;

  return UniqueLrtModel(new_model);
}

bool HasCustomCode(const LrtModelT& model,
                   const absl::string_view custom_code) {
  const auto& fb = model.flatbuffer_model;
  for (auto& c : fb->operator_codes) {
    if (c->custom_code == custom_code &&
        c->builtin_code == tflite::BuiltinOperator_CUSTOM) {
      return true;
    }
  }
  return false;
}

TEST(TestByteCodePacking, MetadataStrategy) {
  static constexpr absl::string_view kByteCode = "some_byte_code";

  auto model = GetTestModel();
  ASSERT_STATUS_OK(LiteRtModelAddByteCodeMetadata(
      model.get(), kSocMan.data(), kSocModel.data(), kByteCode.data(),
      kByteCode.size()));

  model = RoundTrip(std::move(model));
  ASSERT_NE(model, nullptr);

  EXPECT_TRUE(HasCustomCode(*model, kLiteRtDispatchOpCustomCode));

  ASSERT_RESULT_OK_ASSIGN(auto build_tag,
                          GetMetadata(model.get(), kLiteRtBuildTagKey));
  EXPECT_EQ(FbBufToStr(build_tag),
            "soc_man:TestSocMan,soc_model:TestSocModel,serialization_strategy:"
            "METADATA");

  ASSERT_RESULT_OK_ASSIGN(auto byte_code,
                          GetMetadata(model.get(), kLiteRtMetadataByteCodeKey));
  EXPECT_EQ(FbBufToStr(byte_code), kByteCode);
}

TEST(TestByteCodePacking, AppendStrategy) {
  auto model = GetTestModel();
  ASSERT_STATUS_OK(LiteRtModelPrepareForByteCodeAppend(
      model.get(), kSocMan.data(), kSocModel.data()));

  model = RoundTrip(std::move(model));
  ASSERT_NE(model, nullptr);

  EXPECT_TRUE(HasCustomCode(*model, kLiteRtDispatchOpCustomCode));

  ASSERT_RESULT_OK_ASSIGN(auto build_tag,
                          GetMetadata(model.get(), kLiteRtBuildTagKey));
  EXPECT_EQ(FbBufToStr(build_tag),
            "soc_man:TestSocMan,soc_model:TestSocModel,serialization_strategy:"
            "APPEND");

  ASSERT_RESULT_OK_ASSIGN(auto byte_code_placeholder,
                          GetMetadata(model.get(), kLiteRtMetadataByteCodeKey));
  EXPECT_THAT(FbBufToStr(byte_code_placeholder),
              HasSubstr(kLiteRtAppendedByteCodePlaceholder));
}

}  // namespace
