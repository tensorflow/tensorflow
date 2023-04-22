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

#include <vector>

#include "tensorflow/c/experimental/saved_model/core/saved_model_utils.h"
#include "tensorflow/c/experimental/saved_model/core/tf_concrete_function_test_protos.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace tensorflow {
namespace {

// Validates names, shapes, and dtypes of two tensorspecprotos are equivalent.
bool TensorSpecsAreEqual(const TensorSpecProto& spec,
                         const std::string& expected_name,
                         const PartialTensorShape& expected_shape,
                         DataType expected_dtype) {
  return spec.name() == expected_name &&
         PartialTensorShape(spec.shape()).IsIdenticalTo(expected_shape) &&
         spec.dtype() == expected_dtype;
}

// This tests the common case for a tf.function w/o inputs. This ends up
// being serialized as a tuple of an empty tuple + empty dictionary
// (corresponding to the args, kwargs) of the function.
TEST(SignatureFlatteningTest, ZeroArgInputSignature) {
  std::vector<const TensorSpecProto*> flattened;
  StructuredValue value = testing::ZeroArgInputSignature();
  TF_EXPECT_OK(internal::FlattenSignature(value, &flattened));
  EXPECT_EQ(flattened.size(), 0);
}

// This tests the common case for a tf.function w/o outputs. This ends up
// being serialized as a "NoneValue".
TEST(SignatureFlatteningTest, ZeroRetOutputSignature) {
  std::vector<const TensorSpecProto*> flattened;
  StructuredValue value = testing::ZeroReturnOutputSignature();
  TF_EXPECT_OK(internal::FlattenSignature(value, &flattened));
  EXPECT_EQ(flattened.size(), 0);
}

TEST(SignatureFlatteningTest, SingleArgInputSignature) {
  std::vector<const TensorSpecProto*> flattened;
  StructuredValue value = testing::SingleArgInputSignature();
  TF_EXPECT_OK(internal::FlattenSignature(value, &flattened));
  EXPECT_EQ(flattened.size(), 1);
  EXPECT_TRUE(TensorSpecsAreEqual(*flattened[0],
                                  /* expected_name = */ "x",
                                  /* expected_shape = */ {1, 10},
                                  /* expected_dtype = */ DT_FLOAT))
      << "Expected " << flattened[0]->DebugString();
}

TEST(SignatureFlatteningTest, SingleReturnOutputSignature) {
  std::vector<const TensorSpecProto*> flattened;
  StructuredValue value = testing::SingleReturnOutputSignature();
  TF_EXPECT_OK(internal::FlattenSignature(value, &flattened));
  EXPECT_EQ(flattened.size(), 1);
  EXPECT_TRUE(TensorSpecsAreEqual(*flattened[0],
                                  /* expected_name = */ "",
                                  /* expected_shape = */ {1},
                                  /* expected_dtype = */ DT_FLOAT))
      << "Expected " << flattened[0]->DebugString();
}

TEST(SignatureFlatteningTest, ThreeArgInputSignature) {
  std::vector<const TensorSpecProto*> flattened;
  StructuredValue value = testing::ThreeArgInputSignature();
  TF_EXPECT_OK(internal::FlattenSignature(value, &flattened));
  EXPECT_EQ(flattened.size(), 3);
  EXPECT_TRUE(TensorSpecsAreEqual(*flattened[0],
                                  /* expected_name = */ "x",
                                  /* expected_shape = */ {1},
                                  /* expected_dtype = */ DT_FLOAT))
      << "Expected " << flattened[0]->DebugString();

  EXPECT_TRUE(TensorSpecsAreEqual(*flattened[1],
                                  /* expected_name = */ "y",
                                  /* expected_shape = */ {1},
                                  /* expected_dtype = */ DT_FLOAT))
      << "Expected " << flattened[1]->DebugString();

  EXPECT_TRUE(TensorSpecsAreEqual(*flattened[2],
                                  /* expected_name = */ "z",
                                  /* expected_shape = */ {1},
                                  /* expected_dtype = */ DT_FLOAT))
      << "Expected " << flattened[2]->DebugString();
}

// This test has an exotic outputsignature of tuple of a
// dictionary<string,tensor>, tensor
TEST(SignatureFlatteningTest, ThreeReturnOutputSignature) {
  std::vector<const TensorSpecProto*> flattened;
  StructuredValue value = testing::ThreeReturnOutputSignature();
  TF_EXPECT_OK(internal::FlattenSignature(value, &flattened));
  EXPECT_EQ(flattened.size(), 3);
  EXPECT_TRUE(TensorSpecsAreEqual(*flattened[0],
                                  /* expected_name = */ "0/a",
                                  /* expected_shape = */ {1},
                                  /* expected_dtype = */ DT_FLOAT))
      << "Expected " << flattened[0]->DebugString();

  EXPECT_TRUE(TensorSpecsAreEqual(*flattened[1],
                                  /* expected_name = */ "0/b",
                                  /* expected_shape = */ {1},
                                  /* expected_dtype = */ DT_FLOAT))
      << "Expected " << flattened[1]->DebugString();

  EXPECT_TRUE(TensorSpecsAreEqual(*flattened[2],
                                  /* expected_name = */ "1",
                                  /* expected_shape = */ {1},
                                  /* expected_dtype = */ DT_FLOAT))
      << "Expected " << flattened[2]->DebugString();
}

}  // namespace
}  // namespace tensorflow
