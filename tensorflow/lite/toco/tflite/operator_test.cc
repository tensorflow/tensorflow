/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/toco/tflite/operator.h"

#include "flatbuffers/flexbuffers.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"

namespace toco {

namespace tflite {
namespace {

class OperatorTest : public ::testing::Test {
 protected:
  // Return the operator for the given name and type.
  const BaseOperator& GetOperator(const string& name, OperatorType type) {
    using OpsByName = std::map<string, std::unique_ptr<BaseOperator>>;
    using OpsByType = std::map<OperatorType, std::unique_ptr<BaseOperator>>;

    static auto* by_name = new OpsByName(BuildOperatorByNameMap());
    static auto* by_type = new OpsByType(BuildOperatorByTypeMap());

    // Make sure the two maps were consitently built.
    CHECK(by_name->count(name)) << "No operator for '" << name << "'.";
    BaseOperator* op1 = by_name->at(name).get();
    CHECK(op1->type() == type) << "while verifying '" << name << "'.";

    CHECK(by_type->count(type))
        << "No operator for '" << OperatorTypeName(type) << "'.";
    BaseOperator* op2 = by_type->at(type).get();
    CHECK(op2->name() == name)
        << "while verifying '" << OperatorTypeName(type) << "'.";

    return *op1;
  }

  // Use the given BaseOperator to serialize the tf.mini operator into a set of
  // TF Lite options. Proceed to deserialize the options back into a new
  // tf.mini operator, which is then returned. If `options` is given, it will
  // be populated with the serialized options.
  template <typename T>
  std::unique_ptr<T> SerializeAndDeserialize(const BaseOperator& op,
                                             const T& toco_op,
                                             Options* options = nullptr) {
    flatbuffers::FlatBufferBuilder builder;
    Options input_options = op.Serialize(toco_op, &builder);

    if (options) {
      *options = input_options;
    }

    builder.Finish(CreateOperator(builder, 0, 0, 0, input_options.type,
                                  input_options.builtin, input_options.custom,
                                  ::tflite::CustomOptionsFormat_FLEXBUFFERS));
    auto* output_options =
        flatbuffers::GetRoot<::tflite::Operator>(builder.GetBufferPointer());
    auto new_toco_op = op.Deserialize(output_options->builtin_options(),
                                      output_options->custom_options());

    CHECK(new_toco_op->type == toco_op.type)
        << "The type of the serialized and deserialized"
        << HelpfulOperatorTypeName(*new_toco_op)
        << " does not match the type of the original "
        << HelpfulOperatorTypeName(toco_op);

    return std::unique_ptr<T>(dynamic_cast<T*>(new_toco_op.release()));
  }

  // Verify serialization and deserialization of simple operators (those
  // that don't have any configuration parameters).
  template <typename T>
  void CheckSimpleOperator(const string& name, OperatorType type) {
    Options options;
    auto output_toco_op =
        SerializeAndDeserialize(GetOperator(name, type), T(), &options);

    ASSERT_EQ(0, options.builtin.o);
    ASSERT_EQ(0, options.custom.o);
    ASSERT_EQ(::tflite::BuiltinOptions_NONE, options.type);

    ASSERT_NE(nullptr, output_toco_op.get());
  }

  template <typename T>
  void CheckReducerOperator(const string& name, OperatorType type) {
    T op;

    op.keep_dims = false;

    auto output_toco_op = SerializeAndDeserialize(GetOperator(name, type), op);
    EXPECT_EQ(op.keep_dims, output_toco_op->keep_dims);
  }
};

TEST_F(OperatorTest, SimpleOperators) {
  CheckSimpleOperator<DequantizeOperator>("DEQUANTIZE",
                                          OperatorType::kDequantize);
  CheckSimpleOperator<FloorOperator>("FLOOR", OperatorType::kFloor);
  CheckSimpleOperator<ReluOperator>("RELU", OperatorType::kRelu);
  CheckSimpleOperator<Relu1Operator>("RELU_N1_TO_1", OperatorType::kRelu1);
  CheckSimpleOperator<Relu6Operator>("RELU6", OperatorType::kRelu6);
  CheckSimpleOperator<LogisticOperator>("LOGISTIC", OperatorType::kLogistic);
  CheckSimpleOperator<TanhOperator>("TANH", OperatorType::kTanh);
  CheckSimpleOperator<ExpOperator>("EXP", OperatorType::kExp);
  CheckSimpleOperator<LogSoftmaxOperator>("LOG_SOFTMAX",
                                          OperatorType::kLogSoftmax);
  CheckSimpleOperator<TensorFlowMaximumOperator>(
      "MAXIMUM", OperatorType::kMaximum);  //  Element-wise Maximum
  CheckSimpleOperator<TensorFlowMinimumOperator>(
      "MINIMUM", OperatorType::kMinimum);  //  Element-wise Minimum
  CheckSimpleOperator<TensorFlowLessOperator>("LESS", OperatorType::kLess);
  CheckSimpleOperator<NegOperator>("NEG", OperatorType::kNeg);
  CheckSimpleOperator<SelectOperator>("SELECT", OperatorType::kSelect);
  CheckSimpleOperator<SliceOperator>("SLICE", OperatorType::kSlice);
  CheckSimpleOperator<SinOperator>("SIN", OperatorType::kSin);
  CheckSimpleOperator<TensorFlowEqualOperator>("EQUAL", OperatorType::kEqual);
  CheckSimpleOperator<TensorFlowNotEqualOperator>("NOT_EQUAL",
                                                  OperatorType::kNotEqual);
  CheckSimpleOperator<LogOperator>("LOG", OperatorType::kLog);
  CheckSimpleOperator<TensorFlowSqrtOperator>("SQRT", OperatorType::kSqrt);
  CheckSimpleOperator<TensorFlowRsqrtOperator>("RSQRT", OperatorType::kRsqrt);
  CheckSimpleOperator<PowOperator>("POW", OperatorType::kPow);
  CheckSimpleOperator<LogicalOrOperator>("LOGICAL_OR",
                                         OperatorType::kLogicalOr);
  CheckSimpleOperator<LogicalAndOperator>("LOGICAL_AND",
                                          OperatorType::kLogicalAnd);
  CheckSimpleOperator<LogicalNotOperator>("LOGICAL_NOT",
                                          OperatorType::kLogicalNot);
  CheckSimpleOperator<FloorDivOperator>("FLOOR_DIV", OperatorType::kFloorDiv);
  CheckSimpleOperator<TensorFlowSquareOperator>("SQUARE",
                                                OperatorType::kSquare);
  CheckSimpleOperator<TensorFlowZerosLikeOperator>("ZEROS_LIKE",
                                                   OperatorType::kZerosLike);
  CheckSimpleOperator<FloorModOperator>("FLOOR_MOD", OperatorType::kFloorMod);
  CheckSimpleOperator<RangeOperator>("RANGE", OperatorType::kRange);
}

TEST_F(OperatorTest, BuiltinAdd) {
  AddOperator op;
  op.fused_activation_function = FusedActivationFunctionType::kRelu6;
  auto output_toco_op =
      SerializeAndDeserialize(GetOperator("ADD", OperatorType::kAdd), op);
  EXPECT_EQ(op.fused_activation_function,
            output_toco_op->fused_activation_function);
}

TEST_F(OperatorTest, BuiltinReducerOps) {
  CheckReducerOperator<MeanOperator>("MEAN", OperatorType::kMean);
  CheckReducerOperator<TensorFlowSumOperator>("SUM", OperatorType::kSum);
  CheckReducerOperator<TensorFlowProdOperator>("REDUCE_PROD",
                                               OperatorType::kReduceProd);
  CheckReducerOperator<TensorFlowMaxOperator>("REDUCE_MAX",
                                              OperatorType::kReduceMax);
  CheckReducerOperator<TensorFlowMinOperator>("REDUCE_MIN",
                                              OperatorType::kReduceMin);
  CheckReducerOperator<TensorFlowAnyOperator>("REDUCE_ANY", OperatorType::kAny);
}

TEST_F(OperatorTest, BuiltinCast) {
  CastOperator op;
  op.src_data_type = ArrayDataType::kFloat;
  op.dst_data_type = ArrayDataType::kUint8;
  auto output_toco_op =
      SerializeAndDeserialize(GetOperator("CAST", OperatorType::kCast), op);
  EXPECT_EQ(op.src_data_type, output_toco_op->src_data_type);
  EXPECT_EQ(op.dst_data_type, output_toco_op->dst_data_type);
}

TEST_F(OperatorTest, CustomConcatenation) {
  ConcatenationOperator op;
  op.axis = 123;
  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("CONCATENATION", OperatorType::kConcatenation), op);
  EXPECT_EQ(op.axis, output_toco_op->axis);
}

TEST_F(OperatorTest, CustomDepthToSpace) {
  DepthToSpaceOperator op;
  op.block_size = 123;
  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("DEPTH_TO_SPACE", OperatorType::kDepthToSpace), op);
  EXPECT_EQ(op.block_size, output_toco_op->block_size);
}

TEST_F(OperatorTest, CustomFakeQuant) {
  FakeQuantOperator op;
  auto* minmax = new MinMax;
  minmax->min = -10;
  minmax->max = 200;
  op.minmax.reset(minmax);
  op.num_bits = 16;
  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("FAKE_QUANT", OperatorType::kFakeQuant), op);
  EXPECT_EQ(op.minmax->min, output_toco_op->minmax->min);
  EXPECT_EQ(op.minmax->max, output_toco_op->minmax->max);
  EXPECT_EQ(op.num_bits, output_toco_op->num_bits);
}

TEST_F(OperatorTest, CustomFullyConnected) {
  FullyConnectedOperator op;
  op.fused_activation_function = FusedActivationFunctionType::kRelu6;
  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("FULLY_CONNECTED", OperatorType::kFullyConnected), op);
  EXPECT_EQ(op.fused_activation_function,
            output_toco_op->fused_activation_function);
}

TEST_F(OperatorTest, BuiltinGather) {
  GatherOperator op;
  auto output_toco_op =
      SerializeAndDeserialize(GetOperator("GATHER", OperatorType::kGather), op);
  ASSERT_NE(nullptr, output_toco_op.get());
}

TEST_F(OperatorTest, BuiltinL2Pool) {
  L2PoolOperator op;
  op.stride_width = 123;
  op.stride_height = 124;
  op.padding.type = PaddingType::kValid;
  op.kwidth = 480;
  op.kheight = 1080;
  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("L2_POOL_2D", OperatorType::kL2Pool), op);
  EXPECT_EQ(op.stride_width, output_toco_op->stride_width);
  EXPECT_EQ(op.stride_height, output_toco_op->stride_height);
  EXPECT_EQ(op.padding.type, output_toco_op->padding.type);
  EXPECT_EQ(op.kwidth, output_toco_op->kwidth);
  EXPECT_EQ(op.kheight, output_toco_op->kheight);
}

TEST_F(OperatorTest, BuiltinLocalResponseNormalization) {
  LocalResponseNormalizationOperator op;
  op.range = 123;
  op.bias = 1.23;
  op.alpha = 12.3;
  op.beta = .123;
  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("LOCAL_RESPONSE_NORMALIZATION",
                  OperatorType::kLocalResponseNormalization),
      op);
  EXPECT_EQ(op.range, output_toco_op->range);
  EXPECT_EQ(op.bias, output_toco_op->bias);
  EXPECT_EQ(op.alpha, output_toco_op->alpha);
  EXPECT_EQ(op.beta, output_toco_op->beta);
}

TEST_F(OperatorTest, BuiltinMaxPool) {
  MaxPoolOperator op;
  op.stride_width = 123;
  op.stride_height = 124;
  op.padding.type = PaddingType::kValid;
  op.kwidth = 480;
  op.kheight = 1080;
  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("MAX_POOL_2D", OperatorType::kMaxPool), op);
  EXPECT_EQ(op.stride_width, output_toco_op->stride_width);
  EXPECT_EQ(op.stride_height, output_toco_op->stride_height);
  EXPECT_EQ(op.padding.type, output_toco_op->padding.type);
  EXPECT_EQ(op.kwidth, output_toco_op->kwidth);
  EXPECT_EQ(op.kheight, output_toco_op->kheight);
}

TEST_F(OperatorTest, BuiltinReshape) {
  TensorFlowReshapeOperator op;
  op.shape = {1, 2, 4, 5, 8};
  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("RESHAPE", OperatorType::kReshape), op);
  EXPECT_EQ(op.shape, output_toco_op->shape);
}

TEST_F(OperatorTest, CustomSoftmax) {
  SoftmaxOperator op;
  op.beta = 123.1;
  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("SOFTMAX", OperatorType::kSoftmax), op);
  EXPECT_EQ(op.beta, output_toco_op->beta);
}

TEST_F(OperatorTest, BuiltinSpaceToDepth) {
  SpaceToDepthOperator op;
  op.block_size = 123;
  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("SPACE_TO_DEPTH", OperatorType::kSpaceToDepth), op);
  EXPECT_EQ(op.block_size, output_toco_op->block_size);
}

TEST_F(OperatorTest, CustomSplit) {
  TensorFlowSplitOperator op;
  op.num_split = 123;
  auto output_toco_op =
      SerializeAndDeserialize(GetOperator("SPLIT", OperatorType::kSplit), op);
  EXPECT_EQ(op.num_split, output_toco_op->num_split);
}

TEST_F(OperatorTest, BuiltinAveragePool) {
  AveragePoolOperator op;
  op.fused_activation_function = FusedActivationFunctionType::kRelu6;
  op.stride_width = 123;
  op.stride_height = 124;
  op.padding.type = PaddingType::kValid;
  op.kwidth = 480;
  op.kheight = 1080;
  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("AVERAGE_POOL_2D", OperatorType::kAveragePool), op);
  EXPECT_EQ(op.fused_activation_function,
            output_toco_op->fused_activation_function);
  EXPECT_EQ(op.stride_width, output_toco_op->stride_width);
  EXPECT_EQ(op.stride_height, output_toco_op->stride_height);
  EXPECT_EQ(op.padding.type, output_toco_op->padding.type);
  EXPECT_EQ(op.kwidth, output_toco_op->kwidth);
  EXPECT_EQ(op.kheight, output_toco_op->kheight);
}

TEST_F(OperatorTest, BuiltinConvolution) {
  ConvOperator op;
  op.stride_width = 123;
  op.stride_height = 124;
  op.padding.type = PaddingType::kValid;
  op.fused_activation_function = FusedActivationFunctionType::kRelu6;
  auto output_toco_op =
      SerializeAndDeserialize(GetOperator("CONV_2D", OperatorType::kConv), op);
  EXPECT_EQ(op.stride_width, output_toco_op->stride_width);
  EXPECT_EQ(op.stride_height, output_toco_op->stride_height);
  EXPECT_EQ(op.padding.type, output_toco_op->padding.type);
  EXPECT_EQ(op.fused_activation_function,
            output_toco_op->fused_activation_function);
}

TEST_F(OperatorTest, BuiltinDepthwiseConvolution) {
  DepthwiseConvOperator op;
  op.stride_width = 123;
  op.stride_height = 124;
  op.padding.type = PaddingType::kValid;
  op.depth_multiplier = 6;
  op.fused_activation_function = FusedActivationFunctionType::kRelu6;
  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("DEPTHWISE_CONV_2D", OperatorType::kDepthwiseConv), op);
  EXPECT_EQ(op.stride_width, output_toco_op->stride_width);
  EXPECT_EQ(op.stride_height, output_toco_op->stride_height);
  EXPECT_EQ(op.padding.type, output_toco_op->padding.type);
  EXPECT_EQ(op.depth_multiplier, output_toco_op->depth_multiplier);
  EXPECT_EQ(op.fused_activation_function,
            output_toco_op->fused_activation_function);
}

TEST_F(OperatorTest, BuiltinL2Norm) {
  L2NormalizationOperator op;
  op.fused_activation_function = FusedActivationFunctionType::kRelu6;
  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("L2_NORMALIZATION", OperatorType::kL2Normalization), op);
  EXPECT_EQ(op.fused_activation_function,
            output_toco_op->fused_activation_function);
}

TEST_F(OperatorTest, BuiltinMul) {
  MulOperator op;
  op.fused_activation_function = FusedActivationFunctionType::kRelu6;
  auto output_toco_op =
      SerializeAndDeserialize(GetOperator("MUL", OperatorType::kMul), op);
  EXPECT_EQ(op.fused_activation_function,
            output_toco_op->fused_activation_function);
}

TEST_F(OperatorTest, ResizeBilinear) {
  ResizeBilinearOperator op;
  op.align_corners = true;
  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("RESIZE_BILINEAR", OperatorType::kResizeBilinear), op);
  EXPECT_EQ(op.align_corners, output_toco_op->align_corners);
}

TEST_F(OperatorTest, ResizeNearestNeighbor) {
  ResizeNearestNeighborOperator op;
  op.align_corners = true;
  auto output_toco_op =
      SerializeAndDeserialize(GetOperator("RESIZE_NEAREST_NEIGHBOR",
                                          OperatorType::kResizeNearestNeighbor),
                              op);
  EXPECT_EQ(op.align_corners, output_toco_op->align_corners);
}

TEST_F(OperatorTest, Svdf) {
  SvdfOperator op;
  op.fused_activation_function = FusedActivationFunctionType::kRelu;
  op.rank = 1;
  auto output_toco_op =
      SerializeAndDeserialize(GetOperator("SVDF", OperatorType::kSvdf), op);
  EXPECT_EQ(op.fused_activation_function,
            output_toco_op->fused_activation_function);
  EXPECT_EQ(op.rank, output_toco_op->rank);
}

TEST_F(OperatorTest, Squeeze) {
  SqueezeOperator op;
  op.squeeze_dims = {-2, -3, 4, 1, 4};

  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("SQUEEZE", OperatorType::kSqueeze), op);
  EXPECT_EQ(op.squeeze_dims, output_toco_op->squeeze_dims);
}

TEST_F(OperatorTest, StridedSlice) {
  StridedSliceOperator op;

  op.begin_mask = 1;
  op.end_mask = 2;
  op.ellipsis_mask = 1;
  op.new_axis_mask = 1;
  op.shrink_axis_mask = 2;

  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("STRIDED_SLICE", OperatorType::kStridedSlice), op);
  EXPECT_EQ(op.start_indices, output_toco_op->start_indices);
  EXPECT_EQ(op.stop_indices, output_toco_op->stop_indices);
  EXPECT_EQ(op.strides, output_toco_op->strides);
  EXPECT_EQ(op.begin_mask, output_toco_op->begin_mask);
  EXPECT_EQ(op.end_mask, output_toco_op->end_mask);
  EXPECT_EQ(op.end_mask, output_toco_op->end_mask);
  EXPECT_EQ(op.ellipsis_mask, output_toco_op->ellipsis_mask);
  EXPECT_EQ(op.new_axis_mask, output_toco_op->new_axis_mask);
  EXPECT_EQ(op.shrink_axis_mask, output_toco_op->shrink_axis_mask);
}

TEST_F(OperatorTest, BuiltinTopKV2) {
  TopKV2Operator op;
  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("TOPK_V2", OperatorType::kTopK_V2), op);
  ASSERT_NE(nullptr, output_toco_op.get());
}

TEST_F(OperatorTest, BuiltinArgMax) {
  ArgMaxOperator op;
  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("ARG_MAX", OperatorType::kArgMax), op);
  EXPECT_EQ(op.output_data_type, output_toco_op->output_data_type);
}

TEST_F(OperatorTest, BuiltinArgMin) {
  ArgMinOperator op;
  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("ARG_MIN", OperatorType::kArgMin), op);
  EXPECT_EQ(op.output_data_type, output_toco_op->output_data_type);
}

TEST_F(OperatorTest, BuiltinTransposeConv) {
  TransposeConvOperator op;
  op.stride_width = 123;
  op.stride_height = 124;
  op.padding.type = PaddingType::kValid;
  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("TRANSPOSE_CONV", OperatorType::kTransposeConv), op);
  EXPECT_EQ(op.stride_width, output_toco_op->stride_width);
  EXPECT_EQ(op.stride_height, output_toco_op->stride_height);
  EXPECT_EQ(op.padding.type, output_toco_op->padding.type);
}

TEST_F(OperatorTest, BuiltinShape) {
  TensorFlowShapeOperator op;
  op.output_data_type = ArrayDataType::kInt64;
  auto output_toco_op =
      SerializeAndDeserialize(GetOperator("SHAPE", OperatorType::kShape), op);
  EXPECT_EQ(op.output_data_type, output_toco_op->output_data_type);
}

TEST_F(OperatorTest, BuiltinSparseToDense) {
  SparseToDenseOperator op;
  op.validate_indices = false;
  std::unique_ptr<toco::SparseToDenseOperator> output_toco_op =
      SerializeAndDeserialize(
          GetOperator("SPARSE_TO_DENSE", OperatorType::kSparseToDense), op);
  EXPECT_EQ(op.validate_indices, output_toco_op->validate_indices);
}

TEST_F(OperatorTest, BuiltinPack) {
  PackOperator op;
  op.values_count = 3;
  op.axis = 1;
  std::unique_ptr<toco::PackOperator> output_toco_op =
      SerializeAndDeserialize(GetOperator("PACK", OperatorType::kPack), op);
  EXPECT_EQ(op.values_count, output_toco_op->values_count);
  EXPECT_EQ(op.axis, output_toco_op->axis);
}

TEST_F(OperatorTest, BuiltinOneHot) {
  OneHotOperator op;
  op.axis = 2;
  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("ONE_HOT", OperatorType::kOneHot), op);
  EXPECT_EQ(op.axis, output_toco_op->axis);
}

TEST_F(OperatorTest, BuiltinUnpack) {
  UnpackOperator op;
  op.num = 5;
  op.axis = 2;
  auto output_toco_op =
      SerializeAndDeserialize(GetOperator("UNPACK", OperatorType::kUnpack), op);
  EXPECT_EQ(op.num, output_toco_op->num);
  EXPECT_EQ(op.axis, output_toco_op->axis);
}

TEST_F(OperatorTest, BuiltinLeakyRelu) {
  LeakyReluOperator op;
  op.alpha = 3;
  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("LEAKY_RELU", OperatorType::kLeakyRelu), op);
  EXPECT_EQ(op.alpha, output_toco_op->alpha);
}

TEST_F(OperatorTest, BuiltinSquaredDifference) {
  SquaredDifferenceOperator op;
  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("SQUARED_DIFFERENCE", OperatorType::kSquaredDifference), op);
  ASSERT_NE(nullptr, output_toco_op.get());
}

TEST_F(OperatorTest, CustomCTCBeamSearchDecoder) {
  CTCBeamSearchDecoderOperator op;
  op.beam_width = 3;
  op.top_paths = 2;
  op.merge_repeated = false;
  std::unique_ptr<toco::CTCBeamSearchDecoderOperator> output_toco_op =
      SerializeAndDeserialize(GetOperator("CTC_BEAM_SEARCH_DECODER",
                                          OperatorType::kCTCBeamSearchDecoder),
                              op);
  EXPECT_EQ(op.beam_width, output_toco_op->beam_width);
  EXPECT_EQ(op.top_paths, output_toco_op->top_paths);
  EXPECT_EQ(op.merge_repeated, output_toco_op->merge_repeated);
}

TEST_F(OperatorTest, TensorFlowUnsupported) {
  TensorFlowUnsupportedOperator op;
  op.tensorflow_op = "MyCustomUnsupportedOp";

  ::tensorflow::NodeDef node_def;
  auto attr = node_def.mutable_attr();
  (*attr)["float_attr"].set_f(2.0);
  (*attr)["str_attr"].set_s("Hello World");
  (*attr)["int_attr"].set_i(17);
  (*attr)["bool_attr"].set_b(true);
  {
    auto* list = (*attr)["list_int_attr"].mutable_list();
    list->add_i(1);
    list->add_i(20);
    list->add_i(1LL << 40);
    list->add_i(-(1LL << 40));
  }
  node_def.SerializeToString(&op.tensorflow_node_def);

  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("TENSORFLOW_UNSUPPORTED", OperatorType::kUnsupported), op);

  ::tensorflow::NodeDef output_node_def;
  output_node_def.ParseFromString(output_toco_op->tensorflow_node_def);
  const auto& output_attr = output_node_def.attr();
  EXPECT_EQ(2.0, output_attr.at("float_attr").f());
  EXPECT_EQ("Hello World", output_attr.at("str_attr").s());
  EXPECT_EQ(17, output_attr.at("int_attr").i());
  EXPECT_EQ(true, output_attr.at("bool_attr").b());

  {
    const auto& list = output_attr.at("list_int_attr").list();
    ASSERT_EQ(4, list.i_size());
    EXPECT_EQ(1, list.i(0));
    EXPECT_EQ(20, list.i(1));
    EXPECT_EQ(1LL << 40, list.i(2));
    EXPECT_EQ(-(1LL << 40), list.i(3));
  }
}

TEST_F(OperatorTest, TensorFlowUnsupportedWithoutAttr) {
  TensorFlowUnsupportedOperator op;
  op.tensorflow_op = "MyCustomUnsupportedOp";
  auto output_toco_op = SerializeAndDeserialize(
      GetOperator("TENSORFLOW_UNSUPPORTED", OperatorType::kUnsupported), op);

  ::tensorflow::NodeDef output_node_def;
  output_node_def.ParseFromString(output_toco_op->tensorflow_node_def);
  EXPECT_TRUE(output_node_def.attr().empty());
}

TEST_F(OperatorTest, TestShouldExportAsFlexOp) {
  EXPECT_FALSE(ShouldExportAsFlexOp(false, "Conv2D"));
  EXPECT_TRUE(ShouldExportAsFlexOp(true, "Conv2D"));
  EXPECT_TRUE(ShouldExportAsFlexOp(true, "EluGrad"));
  EXPECT_FALSE(ShouldExportAsFlexOp(true, "MyAwesomeCustomOp"));
  // While the RFFT op is available on desktop, it is not in the kernel
  // set available on mobile and should be excluded.
  EXPECT_FALSE(ShouldExportAsFlexOp(true, "RFFT"));
}

}  // namespace
}  // namespace tflite

}  // namespace toco
