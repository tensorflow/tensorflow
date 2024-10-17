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

#include "tensorflow/lite/experimental/lrt/core/litert_model_serialize.h"

#include <cstddef>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/lrt/c/litert_model.h"
#include "tensorflow/lite/experimental/lrt/c/litert_support.h"
#include "tensorflow/lite/experimental/lrt/cc/litert_support.h"
#include "tensorflow/lite/experimental/lrt/core/graph_tools.h"
#include "tensorflow/lite/experimental/lrt/core/litert_model_init.h"
#include "tensorflow/lite/experimental/lrt/core/model.h"
#include "tensorflow/lite/experimental/lrt/core/util/buffer_ref.h"
#include "tensorflow/lite/experimental/lrt/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/experimental/lrt/test/common.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

using ::graph_tools::GetMetadata;
using ::litert::internal::FinishByteCodeAppend;
using ::litert::internal::ParseByteCodeOffsetFromMetadata;
using ::litert::internal::VerifyFlatbuffer;
using ::litert::testing::LoadTestFileModel;
using ::testing::EndsWith;
using ::testing::HasSubstr;
using ::testing::StartsWith;

static constexpr absl::string_view kSocModel = "TestSocModel";
static constexpr absl::string_view kSocMan = "TestSocMan";

// Gets a test model with a single custom op with empty attributes.
UniqueLiteRtModel GetTestModel() {
  static constexpr absl::string_view kTestModel = "one_mul.tflite";
  return LoadTestFileModel(kTestModel);
}

UniqueLiteRtModel RoundTrip(UniqueLiteRtModel model) {
  LITERT_ASSIGN_OR_RETURN_VAL(auto serialized, SerializeModel(std::move(model)),
                              {});
  LITERT_ENSURE(VerifyFlatbuffer(serialized.Span()), {},
                "Failed to verify flatbuffer");

  LiteRtModel new_model;
  LITERT_RETURN_VAL_IF_NOT_OK(
      LoadModel(serialized.Data(), serialized.Size(), &new_model), {});

  return UniqueLiteRtModel(new_model);
}

bool HasCustomCode(const LiteRtModelT& model,
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
  EXPECT_EQ(build_tag.StrView(),
            "soc_man:TestSocMan,soc_model:TestSocModel,serialization_strategy:"
            "METADATA");

  ASSERT_RESULT_OK_ASSIGN(auto byte_code,
                          GetMetadata(model.get(), kLiteRtMetadataByteCodeKey));
  EXPECT_EQ(byte_code.StrView(), kByteCode);
}

TEST(TestByteCodePacking, AppendStrategyPrepare) {
  auto model = GetTestModel();
  ASSERT_STATUS_OK(LiteRtModelPrepareForByteCodeAppend(
      model.get(), kSocMan.data(), kSocModel.data()));

  model = RoundTrip(std::move(model));
  ASSERT_NE(model, nullptr);

  EXPECT_TRUE(HasCustomCode(*model, kLiteRtDispatchOpCustomCode));

  ASSERT_RESULT_OK_ASSIGN(auto build_tag,
                          GetMetadata(model.get(), kLiteRtBuildTagKey));
  EXPECT_EQ(build_tag.StrView(),
            "soc_man:TestSocMan,soc_model:TestSocModel,serialization_strategy:"
            "APPEND");

  ASSERT_RESULT_OK_ASSIGN(auto byte_code_placeholder,
                          GetMetadata(model.get(), kLiteRtMetadataByteCodeKey));
  EXPECT_THAT(byte_code_placeholder.StrView(),
              HasSubstr(kLiteRtAppendedByteCodePlaceholder));
}

TEST(TestByteCodePacking, AppendStrategyFinish) {
  auto model = GetTestModel();
  ASSERT_STATUS_OK(LiteRtModelPrepareForByteCodeAppend(
      model.get(), kSocMan.data(), kSocModel.data()));

  ASSERT_RESULT_OK_MOVE(auto serialized, SerializeModel(std::move(model)));
  const size_t fixed_model_size = serialized.Size();

  ASSERT_STATUS_OK(FinishByteCodeAppend(serialized, 512));
  ASSERT_EQ(serialized.Size(), fixed_model_size);

  ASSERT_RESULT_OK_MOVE(auto new_model, LoadModel(serialized));
  ASSERT_RESULT_OK_ASSIGN(
      auto new_metadata,
      GetMetadata(new_model.get(), kLiteRtMetadataByteCodeKey));
  EXPECT_THAT(new_metadata.StrView(), StartsWith("<npu_byte_code>[*"));
  EXPECT_THAT(new_metadata.StrView(), EndsWith("**512]"));

  ASSERT_RESULT_OK_ASSIGN(auto offset_size,
                          ParseByteCodeOffsetFromMetadata(new_metadata));
  auto [offset, size] = offset_size;

  EXPECT_EQ(offset, fixed_model_size);
  EXPECT_EQ(size, 512);
}

}  // namespace
