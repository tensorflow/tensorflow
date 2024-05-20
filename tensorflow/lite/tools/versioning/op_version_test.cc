/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/versioning/op_version.h"

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/schema/mutable/schema_generated.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/versioning/op_signature.h"

namespace tflite {
namespace {

// Creates vector of OpSignatureTensorSpec with the given TfLiteType vector.
std::vector<OpSignatureTensorSpec> CreateOpSignatureTensorSpecs(
    const std::vector<TfLiteType>& types) {
  std::vector<OpSignatureTensorSpec> tensor_specs;
  for (auto type : types) {
    OpSignatureTensorSpec tensor_spec = {};
    tensor_spec.type = type;
    tensor_specs.push_back(tensor_spec);
  }
  return tensor_specs;
}

// Creates vector of OpSignatureTensorSpec with the given TfLiteType vector,
// each with rank 'rank'
std::vector<OpSignatureTensorSpec> CreateOpSignatureTensorSpecs(
    const std::vector<TfLiteType>& types, int rank) {
  std::vector<OpSignatureTensorSpec> tensor_specs;
  for (auto type : types) {
    OpSignatureTensorSpec tensor_spec = {};
    tensor_spec.type = type;
    for (int i = 0; i < rank; i++) {
      tensor_spec.dims.push_back(4);
    }
    tensor_specs.push_back(tensor_spec);
  }
  return tensor_specs;
}

// Creates vector of OpSignatureTensorSpec of single tensor spec of TfLiteType.
std::vector<OpSignatureTensorSpec> CreateOpSignatureTensorSpecs(
    const TfLiteType type) {
  std::vector<OpSignatureTensorSpec> tensor_specs;
  OpSignatureTensorSpec tensor_spec = {};
  tensor_spec.type = type;
  tensor_specs.push_back(tensor_spec);
  return tensor_specs;
}

// Creates vector of OpSignatureTensorSpec of single tensor spec of TfLiteType
// with shapes.
std::vector<OpSignatureTensorSpec> CreateOpSignatureTensorSpecs(
    const TfLiteType type, const int dim) {
  std::vector<OpSignatureTensorSpec> tensor_specs;
  OpSignatureTensorSpec tensor_spec = {};
  tensor_spec.type = type;
  for (int i = 0; i < dim; i++) {
    tensor_spec.dims.push_back(4);
  }
  tensor_specs.push_back(tensor_spec);
  return tensor_specs;
}

// Creates vector of OpSignatureTensorSpec of two tensor specs of TfLiteType
// with shapes.
std::vector<OpSignatureTensorSpec> CreateOpSignatureTensorSpecs(
    const TfLiteType type, const int dim1, const int dim2) {
  std::vector<OpSignatureTensorSpec> tensor_specs;
  OpSignatureTensorSpec tensor_spec1 = {};
  tensor_spec1.type = type;
  for (int i = 0; i < dim1; i++) {
    tensor_spec1.dims.push_back(4);
  }
  tensor_specs.push_back(tensor_spec1);

  OpSignatureTensorSpec tensor_spec2 = {};
  tensor_spec2.type = type;
  for (int i = 0; i < dim2; i++) {
    tensor_spec2.dims.push_back(4);
  }
  tensor_specs.push_back(tensor_spec2);
  return tensor_specs;
}

}  // namespace

TEST(OpVersionTest, VersioningSpareToDense) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_SPARSE_TO_DENSE,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt8, kTfLiteInt8, kTfLiteInt8}),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_SPARSE_TO_DENSE,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteUInt8, kTfLiteUInt8, kTfLiteUInt8}),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_SPARSE_TO_DENSE,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt64, kTfLiteInt64, kTfLiteInt64}),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_SPARSE_TO_DENSE,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt32, kTfLiteInt32, kTfLiteInt32}),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

// Test version for a simple Op with 2 versions and the input type controls the
// version.
void SimpleVersioningTest(BuiltinOperator op) {
  OpSignature fake_op_sig = {
      .op = op,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = op,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

// Similar to SimpleVersioningTest function, but
// op has 3 versions and the input type includes kTfLiteInt16.
void SimpleVersioningTestExtended(BuiltinOperator op) {
  OpSignature fake_op_sig = {
      .op = op,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  SimpleVersioningTest(op);
}

// Test version for a simple Op with 2 versions and the output type controls the
void SimpleOutputVersioningTest(BuiltinOperator op) {
  OpSignature fake_op_sig = {
      .op = op,
      .inputs = std::vector<OpSignatureTensorSpec>{},
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = op,
      .inputs = std::vector<OpSignatureTensorSpec>{},
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningEqualTest) {
  SimpleVersioningTest(BuiltinOperator_EQUAL);
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_EQUAL,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteString),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
}

TEST(OpVersionTest, VersioningNotEqualTest) {
  SimpleVersioningTest(BuiltinOperator_NOT_EQUAL);
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_NOT_EQUAL,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteString),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
}

TEST(OpVersionTest, VersioningLessTest) {
  SimpleVersioningTest(BuiltinOperator_LESS);
}

TEST(OpVersionTest, VersioningLessEqualTest) {
  SimpleVersioningTest(BuiltinOperator_LESS_EQUAL);
}

TEST(OpVersionTest, VersioningGreaterTest) {
  SimpleVersioningTest(BuiltinOperator_GREATER);
}

TEST(OpVersionTest, VersioningGreaterEqualTest) {
  SimpleVersioningTest(BuiltinOperator_GREATER_EQUAL);
}

TEST(OpVersionTest, VersioningSpaceToBatchNDTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_SPACE_TO_BATCH_ND,
  };
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16, 3);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8, 3);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8, 3);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningLogSoftmaxTest) {
  SimpleVersioningTest(BuiltinOperator_LOG_SOFTMAX);
}

TEST(OpVersionTest, VersioningPackTest) {
  OpSignature fake_op_sig = {};
  fake_op_sig.op = BuiltinOperator_PACK;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {};
  fake_op_sig.op = BuiltinOperator_PACK;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16);
  fake_op_sig.outputs = CreateOpSignatureTensorSpecs(kTfLiteInt16);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {};
  fake_op_sig.op = BuiltinOperator_PACK;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt32);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);

  fake_op_sig = {};
  fake_op_sig.op = BuiltinOperator_PACK;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt32);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningUnpackTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_UNPACK,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_UNPACK,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_UNPACK,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt32),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningRangeTest) {
  OpSignature fake_op_sig = {};
  fake_op_sig.op = BuiltinOperator_RANGE;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt64);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt32);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningReluTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_RELU,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_RELU,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_RELU,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_RELU,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt32),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningBatchToSpaceNDTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_BATCH_TO_SPACE_ND,
  };
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16, 3);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8, 3);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8, 3);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningTanhTest) {
  SimpleVersioningTest(BuiltinOperator_TANH);
}

TEST(OpVersionTest, VersioningStridedSliceTest) {
  TfLiteStridedSliceParams strided_slice_params = {};
  OpSignature fake_op_sig = {};
  fake_op_sig.op = BuiltinOperator_STRIDED_SLICE;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8);
  fake_op_sig.outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8);
  fake_op_sig.builtin_data = reinterpret_cast<void*>(&strided_slice_params);
  strided_slice_params.ellipsis_mask = 0;
  strided_slice_params.new_axis_mask = 2;
  fake_op_sig.ext_options.strided_slice.num_dims = 5;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 6);

  strided_slice_params.new_axis_mask = 0;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);

  fake_op_sig.ext_options.strided_slice.num_dims = 4;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt32);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 7);

  strided_slice_params.offset = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 8);
}

TEST(OpVersionTest, VersioningSpaceToDepthTest) {
  SimpleVersioningTest(BuiltinOperator_SPACE_TO_DEPTH);
}

TEST(OpVersionTest, VersioningSliceTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_SLICE,
  };
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16, 5);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 5);

  fake_op_sig = {
      .op = BuiltinOperator_SLICE,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
  };
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);

  fake_op_sig = {
      .op = BuiltinOperator_SLICE,
  };
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteString, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_SLICE,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
  };
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_SLICE,
  };
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig = {};
  fake_op_sig.op = BuiltinOperator_SLICE;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt32, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 6);
}

TEST(OpVersionTest, VersioningLogisticTest) {
  SimpleVersioningTest(BuiltinOperator_SPACE_TO_DEPTH);
}

TEST(OpVersionTest, VersioningL2NormTest) {
  SimpleOutputVersioningTest(BuiltinOperator_L2_NORMALIZATION);
}

TEST(OpVersionTest, VersioningMaxTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_MAXIMUM,
  };

  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8, 4, 5);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8, 5, 5);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8, 4, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_MAXIMUM,
  };
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8, 4, 5);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8, 4, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningMinTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_MINIMUM,
  };

  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8, 4, 5);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8, 5, 5);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8, 4, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_MINIMUM,
  };
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8, 4, 5);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8, 4, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningMeanTest) {
  SimpleVersioningTestExtended(BuiltinOperator_MEAN);
}

TEST(OpVersionTest, VersioningSumTest) {
  SimpleVersioningTest(BuiltinOperator_SUM);
}

TEST(OpVersionTest, VersioningReduceMinTest) {
  SimpleVersioningTestExtended(BuiltinOperator_REDUCE_MIN);
}

TEST(OpVersionTest, VersioningReduceMaxTest) {
  SimpleVersioningTestExtended(BuiltinOperator_REDUCE_MAX);
}

TEST(OpVersionTest, VersioningMirrorPadTest) {
  SimpleVersioningTestExtended(BuiltinOperator_MIRROR_PAD);
}

TEST(OpVersionTest, VersioningReduceProdTest) {
  OpSignature fake_op_sig;
  fake_op_sig.op = BuiltinOperator_REDUCE_PROD;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningAddTest) {
  TfLiteAddParams add_params = {};
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_ADD,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
      .builtin_data = reinterpret_cast<void*>(&add_params)};
  add_params.pot_scale_int16 = false;
  fake_op_sig.ext_options.add.input_quantized = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig.ext_options.add.input_quantized = false;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 5);

  SimpleVersioningTest(BuiltinOperator_ADD);
}

TEST(OpVersionTest, VersioningSubTest) {
  TfLiteSubParams sub_params = {};
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_SUB,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
      .builtin_data = reinterpret_cast<void*>(&sub_params)};
  sub_params.pot_scale_int16 = false;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 5);

  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt64);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);

  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8, 4, 5);
  fake_op_sig.outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  SimpleVersioningTest(BuiltinOperator_SUB);
}

TEST(OpVersionTest, VersioningMUL7TestInt16) {
  OpSignature fake_op_sig;
  fake_op_sig.op = BuiltinOperator_MUL;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16);
  fake_op_sig.ext_options.mul.input_quantized = false;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 7);
}

TEST(OpVersionTest, VersioningMUL7TestUInt32) {
  OpSignature fake_op_sig;
  fake_op_sig.op = BuiltinOperator_MUL;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt32);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 7);
}

TEST(OpVersionTest, VersioningMUL6Test) {
  OpSignature fake_op_sig;
  fake_op_sig.op = BuiltinOperator_MUL;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteComplex64);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 6);
}

TEST(OpVersionTest, VersioningMUL5Test) {
  OpSignature fake_op_sig;
  fake_op_sig.op = BuiltinOperator_MUL;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt64);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 5);
}

TEST(OpVersionTest, VersioningSub4Test) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_SUB,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt64),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);
}

void SimpleMulVersioningTest(TfLiteType data_type, float multiplier,
                             int version) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_MUL,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{data_type, data_type}),
      .outputs = CreateOpSignatureTensorSpecs(data_type),
  };
  fake_op_sig.ext_options.mul = {1.0f, 1.0f, 1.0f / multiplier};
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), version);
}

TEST(OpVersionTest, VersioningMulTest) {
  SimpleMulVersioningTest(kTfLiteUInt8, 0.5f, 1);
  SimpleMulVersioningTest(kTfLiteInt8, 0.5f, 2);
  SimpleMulVersioningTest(kTfLiteInt8, 2.0f, 3);
}

TEST(OpVersionTest, VersioningPadTest) {
  SimpleVersioningTest(BuiltinOperator_PAD);
}

TEST(OpVersionTest, VersioningPadV2Test) {
  SimpleVersioningTest(BuiltinOperator_PADV2);
}

TEST(OpVersionTest, VersioningConcatenationTest) {
  OpSignature fake_op_sig = {};
  fake_op_sig.op = BuiltinOperator_CONCATENATION;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt32);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);
}

TEST(OpVersionTest, VersioningSelectTest) {
  OpSignature fake_op_sig = {};
  fake_op_sig.op = BuiltinOperator_SELECT;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(
      std::vector<TfLiteType>{kTfLiteUInt32, kTfLiteUInt32, kTfLiteUInt32}, 5);
  fake_op_sig.outputs = CreateOpSignatureTensorSpecs(kTfLiteUInt32);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);

  fake_op_sig = {};
  fake_op_sig.op = BuiltinOperator_SELECT;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(
      std::vector<TfLiteType>{kTfLiteUInt8, kTfLiteUInt8, kTfLiteUInt8}, 5);
  fake_op_sig.outputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {};
  fake_op_sig.op = BuiltinOperator_SELECT;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(
      std::vector<TfLiteType>{kTfLiteInt8, kTfLiteInt8, kTfLiteInt8}, 4);
  fake_op_sig.outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {};
  fake_op_sig.op = BuiltinOperator_SELECT;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(
      std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteFloat32, kTfLiteFloat32},
      4);
  fake_op_sig.outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningSelectV2Test) {
  OpSignature fake_op_sig = {};
  fake_op_sig.op = BuiltinOperator_SELECT_V2;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(
      std::vector<TfLiteType>{kTfLiteUInt32, kTfLiteUInt32, kTfLiteUInt32}, 5);
  fake_op_sig.outputs = CreateOpSignatureTensorSpecs(kTfLiteUInt32);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
  fake_op_sig = {};
  fake_op_sig.op = BuiltinOperator_SELECT_V2;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(
      std::vector<TfLiteType>{kTfLiteInt32, kTfLiteInt32, kTfLiteInt32}, 5);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningRelu6Test) {
  SimpleVersioningTestExtended(BuiltinOperator_RELU6);
}

TEST(OpVersionTest, VersioningFullyConnectedTest) {
  TfLiteFullyConnectedParams fully_connected_params = {};
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_FULLY_CONNECTED,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteUInt8, kTfLiteUInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8),
      .builtin_data = reinterpret_cast<void*>(&fully_connected_params),
  };
  fully_connected_params.weights_format =
      kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 6);

  fake_op_sig = {
      .op = BuiltinOperator_FULLY_CONNECTED,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt8, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
      .builtin_data = reinterpret_cast<void*>(&fully_connected_params),
  };
  fully_connected_params.weights_format =
      kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 6);

  fake_op_sig = {
      .op = BuiltinOperator_FULLY_CONNECTED,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt8, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
      .builtin_data = reinterpret_cast<void*>(&fully_connected_params),
  };
  fully_connected_params.weights_format =
      kTfLiteFullyConnectedWeightsFormatDefault;
  fake_op_sig.ext_options.fully_connected.sparse_weight = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 8);

  fake_op_sig = {
      .op = BuiltinOperator_FULLY_CONNECTED,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteInt8, kTfLiteFloat32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .builtin_data = reinterpret_cast<void*>(&fully_connected_params),
  };
  fully_connected_params.asymmetric_quantize_inputs = false;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fully_connected_params.asymmetric_quantize_inputs = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 9);

  fake_op_sig = {
      .op = BuiltinOperator_FULLY_CONNECTED,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt16, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
      .builtin_data = reinterpret_cast<void*>(&fully_connected_params),
  };
  fully_connected_params.quantized_bias_type = kTfLiteInt32;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 11);

  fake_op_sig = {
      .op = BuiltinOperator_FULLY_CONNECTED,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .builtin_data = reinterpret_cast<void*>(&fully_connected_params),
  };
  fake_op_sig.ext_options.fully_connected.is_per_channel_quantized = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 12);
}

TEST(OpVersionTest, VersioningDequantizeTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_DEQUANTIZE,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_DEQUANTIZE,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteFloat16),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_DEQUANTIZE,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig.ext_options.dequantize.is_per_channel_quantized = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 5);

  fake_op_sig = {
      .op = BuiltinOperator_DEQUANTIZE,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningQuantizeTest) {
  OpSignature fake_op_sig;
  fake_op_sig.op = BuiltinOperator_QUANTIZE;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32);
  fake_op_sig.ext_options.quantize.is_per_channel_quantized = false;

  fake_op_sig.outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig.outputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig.outputs = CreateOpSignatureTensorSpecs(kTfLiteInt16);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig.ext_options.quantize.is_per_channel_quantized = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
}

TEST(OpVersionTest, VersioningConv2DTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_CONV_2D,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteUInt8, kTfLiteUInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig = {
      .op = BuiltinOperator_CONV_2D,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt8, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_CONV_2D,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_CONV_2D,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
  };
  fake_op_sig.ext_options.conv_2d.is_per_channel_quantized = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 5);

  fake_op_sig.op = BuiltinOperator_CONV_2D;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(
      std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteInt8});
  fake_op_sig.outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32);
  fake_op_sig.ext_options.conv_2d.is_grouped_convolution = true;

  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 6);

  TfLiteConvParams conv_params = {};
  fake_op_sig = {
      .op = BuiltinOperator_CONV_2D,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt16, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
      .builtin_data = reinterpret_cast<void*>(&conv_params),
  };
  conv_params.quantized_bias_type = kTfLiteInt32;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 8);
}

TEST(OpVersionTest, VersioningFloorDivOperatorTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_FLOOR_DIV,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt32),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig = {
      .op = BuiltinOperator_FLOOR_DIV,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_FLOOR_DIV,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
}

TEST(OpVersionTest, VersioningFloorModOperatorTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_FLOOR_MOD,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt32),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig = {
      .op = BuiltinOperator_FLOOR_MOD,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
}

TEST(OpVersionTest, VersioningTransposeConvOperatorTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE_CONV,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteUInt8}),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE_CONV,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt32, kTfLiteInt8, kTfLiteInt8}),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE_CONV,
      .inputs = CreateOpSignatureTensorSpecs(std::vector<TfLiteType>{
          kTfLiteInt32, kTfLiteInt8, kTfLiteInt8, kTfLiteInt32}),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  const auto none_type = kTfLiteNoType;
  fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE_CONV,
      .inputs = CreateOpSignatureTensorSpecs(std::vector<TfLiteType>{
          kTfLiteInt32, kTfLiteInt8, kTfLiteInt8, none_type}),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  TfLiteTransposeConvParams transpose_conv_params = {};
  transpose_conv_params.activation = kTfLiteActRelu;
  fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE_CONV,
      .inputs = CreateOpSignatureTensorSpecs(std::vector<TfLiteType>{
          kTfLiteInt32, kTfLiteInt8, kTfLiteInt8, none_type}),
      .builtin_data = reinterpret_cast<void*>(&transpose_conv_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);

  transpose_conv_params = {};
  fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE_CONV,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt16, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
      .builtin_data = reinterpret_cast<void*>(&transpose_conv_params),
  };
  transpose_conv_params.quantized_bias_type = kTfLiteInt32;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 5);
}

TEST(OpVersionTest, VersioningSVDFOperatorTest) {
  TfLiteSVDFParams svdf_params = {};
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_SVDF,
      .inputs = CreateOpSignatureTensorSpecs(std::vector<TfLiteType>{
          kTfLiteFloat32, kTfLiteFloat32, kTfLiteFloat32, kTfLiteFloat32,
          kTfLiteFloat32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .builtin_data = reinterpret_cast<void*>(&svdf_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig = {
      .op = BuiltinOperator_SVDF,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteInt8, kTfLiteFloat32,
                                  kTfLiteFloat32, kTfLiteFloat32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .builtin_data = reinterpret_cast<void*>(&svdf_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
  svdf_params.asymmetric_quantize_inputs = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);

  svdf_params = {};
  fake_op_sig = {
      .op = BuiltinOperator_SVDF,
      .inputs = CreateOpSignatureTensorSpecs(std::vector<TfLiteType>{
          kTfLiteInt8, kTfLiteInt8, kTfLiteInt32, kTfLiteInt32, kTfLiteInt16}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
      .builtin_data = reinterpret_cast<void*>(&svdf_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
}

TEST(OpVersionTest, VersioningDepthwiseConv2DTest) {
  TfLiteDepthwiseConvParams depthwise_conv_params = {};
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_DEPTHWISE_CONV_2D,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .builtin_data = reinterpret_cast<void*>(&depthwise_conv_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);
  fake_op_sig.ext_options.depthwise_conv_2d.is_per_channel_quantized = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 6);

  depthwise_conv_params = {};
  fake_op_sig = {
      .op = BuiltinOperator_DEPTHWISE_CONV_2D,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt8, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
      .builtin_data = reinterpret_cast<void*>(&depthwise_conv_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_DEPTHWISE_CONV_2D,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteFloat32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .builtin_data = reinterpret_cast<void*>(&depthwise_conv_params),
  };
  depthwise_conv_params.dilation_width_factor = 2;
  depthwise_conv_params.dilation_height_factor = 2;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_DEPTHWISE_CONV_2D,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteFloat32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .builtin_data = reinterpret_cast<void*>(&depthwise_conv_params),
  };
  depthwise_conv_params.dilation_width_factor = 1;
  depthwise_conv_params.dilation_height_factor = 1;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}
TEST(OpVersionTest, VersioningTileOperatorTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_TILE,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt32),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig = {
      .op = BuiltinOperator_TILE,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteString),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
}
TEST(OpVersionTest, VersioningTransposeTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 5);

  fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE,
  };
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteBool, 5);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteBool, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}
TEST(OpVersionTest, VersioningGatherNdOperatorTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_GATHER_ND,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt32, kTfLiteInt32}),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig = {
      .op = BuiltinOperator_GATHER_ND,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteString, kTfLiteInt32}),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_GATHER_ND,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt16, kTfLiteInt32}),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_GATHER_ND,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt32, kTfLiteInt16}),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);

  fake_op_sig = {
      .op = BuiltinOperator_GATHER_ND,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteBool, kTfLiteInt16}),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 5);
}
TEST(OpVersionTest, VersioningDivTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_DIV,
  };
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8, 5, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8, 5, 5);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8, 4, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}
TEST(OpVersionTEst, VersioningFillTest) {
  OpSignature fake_op_sig = {BuiltinOperator_FILL};
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(
      std::vector<TfLiteType>{kTfLiteInt32, kTfLiteFloat16});
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(
      std::vector<TfLiteType>{kTfLiteInt64, kTfLiteFloat16});
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(
      std::vector<TfLiteType>{kTfLiteInt32, kTfLiteInt8});
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(
      std::vector<TfLiteType>{kTfLiteInt64, kTfLiteInt16});
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(
      std::vector<TfLiteType>{kTfLiteInt32, kTfLiteBool});
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(
      std::vector<TfLiteType>{kTfLiteInt32, kTfLiteString});
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(
      std::vector<TfLiteType>{kTfLiteInt32, kTfLiteInt32});
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}
TEST(OpVersionTest, VersioningResizeBilinearTest) {
  // Default.
  TfLiteResizeBilinearParams resize_bilinear_params = {};
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_RESIZE_BILINEAR,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteInt32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .builtin_data = reinterpret_cast<void*>(&resize_bilinear_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  // align_corners=true is still version 1.
  resize_bilinear_params.align_corners = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  // half_pixel_centers=true must be version 3.
  resize_bilinear_params.align_corners = false;
  resize_bilinear_params.half_pixel_centers = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  // int8 input is version 2.
  resize_bilinear_params = {};
  fake_op_sig = {
      .op = BuiltinOperator_RESIZE_BILINEAR,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt8, kTfLiteInt32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
      .builtin_data = reinterpret_cast<void*>(&resize_bilinear_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  resize_bilinear_params.half_pixel_centers = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  // int16 input is version 4.
  resize_bilinear_params = {};
  fake_op_sig = {
      .op = BuiltinOperator_RESIZE_BILINEAR,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt16, kTfLiteInt32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
      .builtin_data = reinterpret_cast<void*>(&resize_bilinear_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);
}
TEST(OpVersionTest, VersioningResizeNearestNeighborTest) {
  // Default.
  TfLiteResizeNearestNeighborParams resize_nearest_neighbor_params = {};
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteInt32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .builtin_data = reinterpret_cast<void*>(&resize_nearest_neighbor_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  // align_corners=true is version 3.
  resize_nearest_neighbor_params.align_corners = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  // half_pixel_centers=true must be version 3.
  resize_nearest_neighbor_params.align_corners = false;
  resize_nearest_neighbor_params.half_pixel_centers = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  // int8 input is version 2.
  resize_nearest_neighbor_params = {};
  fake_op_sig = {
      .op = BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt8, kTfLiteInt32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
      .builtin_data = reinterpret_cast<void*>(&resize_nearest_neighbor_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  resize_nearest_neighbor_params.align_corners = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  // int16 input is version 4.
  resize_nearest_neighbor_params = {};
  fake_op_sig = {
      .op = BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt16, kTfLiteInt32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
      .builtin_data = reinterpret_cast<void*>(&resize_nearest_neighbor_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);
}
TEST(OpVersionTest, VersioningAbsTest) {
  // Default.
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_ABS,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  // int8 input is version 2.
  fake_op_sig = {
      .op = BuiltinOperator_ABS,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  // int16 quantized input is version 3.
  fake_op_sig = {
      .op = BuiltinOperator_ABS,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
  };
  fake_op_sig.ext_options.abs.input_quantized = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  // int16 non-quantized input is version 4.
  fake_op_sig = {
      .op = BuiltinOperator_ABS,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);
  fake_op_sig = {};
  fake_op_sig.op = BuiltinOperator_ABS;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt32);
  fake_op_sig.outputs = CreateOpSignatureTensorSpecs(kTfLiteInt32);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 5);
}
TEST(OpVersionTest, VersioningSignTest) {
  // Default.
  OpSignature fake_op_sig;
  fake_op_sig.op = BuiltinOperator_SIGN;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32);
  fake_op_sig.outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  // int32 input is version 2.
  fake_op_sig = {};
  fake_op_sig.op = BuiltinOperator_SIGN;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt32);
  fake_op_sig.outputs = CreateOpSignatureTensorSpecs(kTfLiteInt32);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
}
TEST(OpVersionTest, VersioningBatchMatMulTest) {
  // Default.
  TfLiteBatchMatMulParams batch_mat_mul_params = {};
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_BATCH_MATMUL,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteFloat32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .builtin_data = reinterpret_cast<void*>(&batch_mat_mul_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  // int8 input is version 2.
  batch_mat_mul_params = {};
  fake_op_sig = {
      .op = BuiltinOperator_BATCH_MATMUL,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt8, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
      .builtin_data = reinterpret_cast<void*>(&batch_mat_mul_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  // int16 input is version 3.
  fake_op_sig = {
      .op = BuiltinOperator_BATCH_MATMUL,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt16, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
      .builtin_data = reinterpret_cast<void*>(&batch_mat_mul_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  // Symmetric hybrid quantized input is version 1.
  fake_op_sig = {
      .op = BuiltinOperator_BATCH_MATMUL,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .builtin_data = reinterpret_cast<void*>(&batch_mat_mul_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  // Asymmetric hybrid quantized input is version 4.
  fake_op_sig = {
      .op = BuiltinOperator_BATCH_MATMUL,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .builtin_data = reinterpret_cast<void*>(&batch_mat_mul_params),
  };
  batch_mat_mul_params.asymmetric_quantize_inputs = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);
}
TEST(OpVersionTest, VersioningSquaredDifferenceTest) {
  // Default.
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_SQUARED_DIFFERENCE,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteFloat32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  // int8 input is version 2.
  fake_op_sig = {
      .op = BuiltinOperator_SQUARED_DIFFERENCE,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt8, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
}
TEST(OpVersionTest, VersioningRsqrtTest) {
  OpSignature fake_op_sig = {};
  fake_op_sig.op = BuiltinOperator_RSQRT;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
}
TEST(OpVersionTest, VersioningBroadcastToTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_BROADCAST_TO,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  // Quantized broadcast_to op is version 3.
  fake_op_sig = {
      .op = BuiltinOperator_BROADCAST_TO,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_BROADCAST_TO,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
}

TEST(OpVersionTest, VersioningGeluTest) {
  OpSignature fake_op_sig;
  fake_op_sig.op = BuiltinOperator_GELU;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig.op = BuiltinOperator_GELU;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig.op = BuiltinOperator_GELU;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
}

TEST(OpVersionTest, VersioningUnidirectionalLstmTest) {
  TfLiteUnidirectionalSequenceLSTMParams params = {};
  OpSignature fake_op_sig = {};
  fake_op_sig.op = BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(
      std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteFloat32, kTfLiteFloat32});
  fake_op_sig.outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32);
  fake_op_sig.builtin_data = reinterpret_cast<void*>(&params);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(
      std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteFloat32, kTfLiteInt8});
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  params.asymmetric_quantize_inputs = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  params.diagonal_recurrent_tensors = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);
}

TEST(OpVersionTest, VersioningExpTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_EXP,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
  fake_op_sig = {
      .op = BuiltinOperator_EXP,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
  fake_op_sig = {
      .op = BuiltinOperator_EXP,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
}
TEST(OpVersionTest, VersioningLogTest) {
  OpSignature fake_op_sig = {};
  fake_op_sig.op = BuiltinOperator_LOG;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
}
}  // namespace tflite
