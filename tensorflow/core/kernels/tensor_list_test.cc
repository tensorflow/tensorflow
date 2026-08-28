/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/tensor_list.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

std::string TensorListMetadata(const std::vector<uint64_t>& invalid_indices,
                               uint64_t element_dtype,
                               uint64_t max_num_elements,
                               const std::string& shape_proto) {
  std::string metadata;
  core::PutVarint64(&metadata, invalid_indices.size());
  for (uint64_t invalid_index : invalid_indices) {
    core::PutVarint64(&metadata, invalid_index);
  }
  core::PutVarint64(&metadata, element_dtype);
  core::PutVarint64(&metadata, max_num_elements);
  metadata.append(shape_proto);
  return metadata;
}

Tensor ScalarInt32Tensor(int32_t value) {
  Tensor t(DT_INT32, TensorShape({}));
  t.scalar<int32_t>()() = value;
  return t;
}

TEST(TensorListTest, DecodeIsTransactionalOnFailure) {
  VariantTensorData data;
  data.set_metadata(TensorListMetadata(
      /*invalid_indices=*/{}, static_cast<uint64_t>(DT_FLOAT),
      std::numeric_limits<uint64_t>::max(), "not a shape proto"));
  *data.add_tensors() = ScalarInt32Tensor(123);

  TensorList list;
  list.element_dtype = DT_INT32;
  list.max_num_elements = 7;
  list.element_shape = PartialTensorShape({2});
  list.tensors().push_back(ScalarInt32Tensor(42));

  EXPECT_FALSE(list.Decode(data));
  EXPECT_EQ(list.element_dtype, DT_INT32);
  EXPECT_EQ(list.max_num_elements, 7);
  EXPECT_EQ(list.element_shape.dim_size(0), 2);
  ASSERT_EQ(list.tensors().size(), 1);
  EXPECT_EQ(list.tensors()[0].dtype(), DT_INT32);
  EXPECT_EQ(list.tensors()[0].scalar<int32_t>()(), 42);
}

TEST(TensorListTest, DecodeRejectsMalformedShapeMetadata) {
  TensorShapeProto shape;
  shape.add_dim()->set_size(1);
  VariantTensorData data;

  data.set_metadata(TensorListMetadata(
      /*invalid_indices=*/{}, static_cast<uint64_t>(DT_FLOAT),
      std::numeric_limits<uint64_t>::max(), shape.SerializeAsString()));
  EXPECT_TRUE(TensorList().Decode(data));

  data.set_metadata(TensorListMetadata(
      /*invalid_indices=*/{}, static_cast<uint64_t>(DT_FLOAT),
      std::numeric_limits<uint64_t>::max(), "not a shape proto"));
  EXPECT_FALSE(TensorList().Decode(data));
}

TEST(TensorListTest, DecodeRejectsDuplicateInvalidIndices) {
  TensorShapeProto shape;
  VariantTensorData data;
  data.set_metadata(TensorListMetadata(
      /*invalid_indices=*/{1, 1}, static_cast<uint64_t>(DT_INT32),
      std::numeric_limits<uint64_t>::max(), shape.SerializeAsString()));
  *data.add_tensors() = ScalarInt32Tensor(1);
  EXPECT_FALSE(TensorList().Decode(data));
}

TEST(TensorListTest, DecodeRejectsDescendingInvalidIndices) {
  TensorShapeProto shape;
  VariantTensorData data;
  data.set_metadata(TensorListMetadata(
      /*invalid_indices=*/{2, 1}, static_cast<uint64_t>(DT_INT32),
      std::numeric_limits<uint64_t>::max(), shape.SerializeAsString()));
  *data.add_tensors() = ScalarInt32Tensor(1);
  EXPECT_FALSE(TensorList().Decode(data));
}

TEST(TensorListTest, DecodePreservesSentinelMinusOne) {
  TensorShapeProto shape;
  VariantTensorData data;
  data.set_metadata(TensorListMetadata(
      /*invalid_indices=*/{}, static_cast<uint64_t>(DT_INT32),
      std::numeric_limits<uint64_t>::max(), shape.SerializeAsString()));

  TensorList decoded;
  EXPECT_TRUE(decoded.Decode(data));
  EXPECT_EQ(decoded.max_num_elements, -1);
}

TEST(TensorListTest, DecodeValidRoundTripCompatibility) {
  TensorList list;
  list.element_dtype = DT_INT32;
  list.max_num_elements = -1;
  list.element_shape = PartialTensorShape({});
  list.tensors().push_back(Tensor(DT_INVALID));
  list.tensors().push_back(ScalarInt32Tensor(7));
  list.tensors().push_back(Tensor(DT_INVALID));
  list.tensors().push_back(ScalarInt32Tensor(9));

  VariantTensorData data;
  list.Encode(&data);

  TensorList decoded;
  EXPECT_TRUE(decoded.Decode(data));
  EXPECT_EQ(decoded.element_dtype, DT_INT32);
  EXPECT_EQ(decoded.max_num_elements, -1);
  EXPECT_TRUE(decoded.element_shape.IsIdenticalTo(list.element_shape));
  ASSERT_EQ(decoded.tensors().size(), 4);
  EXPECT_EQ(decoded.tensors()[0].dtype(), DT_INVALID);
  EXPECT_EQ(decoded.tensors()[1].dtype(), DT_INT32);
  EXPECT_EQ(decoded.tensors()[1].scalar<int32_t>()(), 7);
  EXPECT_EQ(decoded.tensors()[2].dtype(), DT_INVALID);
  EXPECT_EQ(decoded.tensors()[3].dtype(), DT_INT32);
  EXPECT_EQ(decoded.tensors()[3].scalar<int32_t>()(), 9);
}

TEST(TensorListTest, DecodeRejectsOutOfRangeMaxNumElements) {
  TensorShapeProto shape;
  VariantTensorData data;
  data.set_metadata(TensorListMetadata(
      /*invalid_indices=*/{}, static_cast<uint64_t>(DT_INT32),
      static_cast<uint64_t>(std::numeric_limits<int>::max()) + 1,
      shape.SerializeAsString()));
  EXPECT_FALSE(TensorList().Decode(data));
}

}  // namespace
}  // namespace tensorflow
