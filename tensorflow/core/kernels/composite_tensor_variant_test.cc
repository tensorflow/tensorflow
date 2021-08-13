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

#include "tensorflow/core/kernels/composite_tensor_variant.h"

#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/composite_tensor_variant.pb.h"

namespace tensorflow {
namespace {

// TypeSpecProto for a 2D Ragged Tensor.
constexpr const char* k2DRaggedTensorSpec = R"(
type_spec_proto: {
  type_spec_class: RAGGED_TENSOR_SPEC
  type_state: {
    tuple_value: {
      values: [
        {tensor_shape_value:
          {dim: [{size: -1}, {size: -1}]}},      # shape
        {tensor_dtype_value: DT_INT32},          # dtype
        {int64_value: 1},                        # ragged_rank
        {tensor_dtype_value: DT_INT64}           # row_splits_dtype
      ]
    }
  }
})";

// Returns an CompositeTensorVariant encoding for a 2D ragged tensor with
// the specified values and row_splits.
CompositeTensorVariant Make2DRaggedTensor(const std::vector<int32>& values,
                                          const std::vector<int64_t>& splits) {
  CompositeTensorVariantMetadata metadata;
  EXPECT_TRUE(
      protobuf::TextFormat::ParseFromString(k2DRaggedTensorSpec, &metadata));
  std::vector<Tensor> components;
  components.push_back(test::AsTensor<int32>(values));
  components.push_back(test::AsTensor<int64_t>(splits));
  CompositeTensorVariant v(metadata, absl::MakeSpan(components));
  return v;
}

TEST(CompositeTensorVariantTest, EncodeAndDecodeRagged) {
  CompositeTensorVariant v = Make2DRaggedTensor(
      /* values = */ {5, 5, 3, 4, 1, 8},
      /* splits = */ {0, 2, 3, 6});
  Tensor t(DT_VARIANT, {});

  t.flat<Variant>()(0) = v;  // Encode to variant.
  auto* decoded = t.flat<Variant>()(0).get<CompositeTensorVariant>();

  EXPECT_EQ(v.metadata().SerializeAsString(),
            decoded->metadata().SerializeAsString());
  EXPECT_EQ(v.flat_components().size(), 2);
  test::ExpectTensorEqual<int32>(v.flat_components()[0],
                                 decoded->flat_components()[0]);
  test::ExpectTensorEqual<int64_t>(v.flat_components()[1],
                                   decoded->flat_components()[1]);
}

TEST(CompositeTensorVariantTest, DebugStringForDefaultConstructed) {
  CompositeTensorVariant v;
  EXPECT_EQ(v.DebugString(),
            "<CompositeTensorVariant type=UNKNOWN, components=[]>");
}

TEST(CompositeTensorVariantTest, DebugStringForRagged) {
  CompositeTensorVariant v = Make2DRaggedTensor(
      /* values = */ {5, 5, 3, 4, 1},
      /* splits = */ {0, 2, 3, 5});
  EXPECT_EQ(v.DebugString(),
            "<CompositeTensorVariant type=RAGGED_TENSOR_SPEC, "
            "components=[Tensor<type: int32 shape: [5] values: 5 5 3...>, "
            "Tensor<type: int64 shape: [4] values: 0 2 3...>]>");
}

TEST(CompositeTensorVariantTest, TypeName) {
  CompositeTensorVariant v;
  EXPECT_EQ(v.TypeName(), "CompositeTensorVariant");
}

}  // namespace
}  // namespace tensorflow
