/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/core/api/flatbuffer_conversions.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/compiler/mlir/lite/core/c/builtin_op_data.h"
#include "tensorflow/compiler/mlir/lite/core/c/tflite_types.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"

using testing::AllOf;
using testing::Each;
using testing::ElementsAre;
using testing::Eq;
using testing::HasSubstr;
using testing::StrEq;
using tflite::BuiltinOptions;
using tflite::BuiltinOptions2;
using tflite::BuiltinOptions_SqueezeOptions;
using tflite::CustomOptionsFormat_FLEXBUFFERS;

namespace tflite_file {
namespace flatbuffer_conversions {
using tflite::ActivationFunctionType_RELU;
using tflite::BuiltinOperator_CONV_2D;
using tflite::BuiltinOperator_CUSTOM;
using tflite::BuiltinOperator_FULLY_CONNECTED;
using tflite::BuiltinOperator_RESHAPE;
using tflite::BuiltinOperator_SQUEEZE;
using tflite::BuiltinOperator_STABLEHLO_PAD;
using tflite::BuiltinOperator_STABLEHLO_REDUCE_WINDOW;
using tflite::BuiltinOptions2_StablehloPadOptions;
using tflite::BuiltinOptions2_StablehloReduceWindowOptions;
using tflite::BuiltinOptions_Conv2DOptions;
using tflite::BuiltinOptions_FullyConnectedOptions;
using tflite::BuiltinOptions_NONE;
using tflite::BuiltinOptions_ReshapeOptions;
using tflite::CreateReshapeOptions;
using tflite::CreateSqueezeOptions;
using tflite::CreateStablehloPadOptions;
using tflite::CreateStablehloReduceWindowOptions;
using tflite::FullyConnectedOptionsWeightsFormat;
using tflite::Padding_SAME;
using tflite::TensorType_BFLOAT16;
using tflite::TensorType_FLOAT16;
using tflite::TensorType_FLOAT32;
using tflite::TensorType_INT4;

namespace {

using std::string;

// Used to determine how the op data parsing function creates its working space.
class MockDataAllocator : public BuiltinDataAllocator {
 public:
  MockDataAllocator() : is_allocated_(false) {}
  void* Allocate(size_t size, size_t alignment_hint) override {
    EXPECT_FALSE(is_allocated_);
    const int max_size = kBufferSize;
    EXPECT_LE(size, max_size);
    is_allocated_ = true;
    return buffer_;
  }
  void Deallocate(void* data) override { is_allocated_ = false; }

 private:
  static constexpr int kBufferSize = 1024;
  char buffer_[kBufferSize];
  bool is_allocated_;
};

}  // namespace

class FlatbufferConversionsTest : public ::testing::Test {
 public:
  const Operator* BuildTestOperator(BuiltinOptions op_type,
                                    flatbuffers::Offset<void> options) {
    flatbuffers::Offset<Operator> offset =
        CreateOperatorDirect(builder_, 0, nullptr, nullptr, op_type, options,
                             nullptr, CustomOptionsFormat_FLEXBUFFERS, nullptr);
    builder_.Finish(offset);
    void* pointer = builder_.GetBufferPointer();
    return flatbuffers::GetRoot<Operator>(pointer);
  }

  const Operator* BuildTestOperator(BuiltinOptions2 op_type,
                                    flatbuffers::Offset<void> options) {
    flatbuffers::Offset<Operator> offset = CreateOperatorDirect(
        builder_, /*opcode_index=*/0, /*inputs=*/nullptr, /*outputs=*/nullptr,
        /*builtin_options_type=*/tflite::BuiltinOptions_NONE,
        /*builtin_options=*/0, /*custom_options=*/nullptr,
        /*custom_options_format=*/tflite::CustomOptionsFormat_FLEXBUFFERS,
        /*mutating_variable_inputs=*/nullptr, /*intermediates=*/nullptr,
        /*large_custom_options_offset=*/0, /*large_custom_options_size=*/0,
        /*builtin_options_2_type=*/op_type,
        /*builtin_options_2=*/options);
    builder_.Finish(offset);
    void* pointer = builder_.GetBufferPointer();
    return flatbuffers::GetRoot<Operator>(pointer);
  }

 protected:
  MockDataAllocator mock_allocator_;
  flatbuffers::FlatBufferBuilder builder_;
};

TEST_F(FlatbufferConversionsTest, ParseSqueezeAll) {
  const Operator* op = BuildTestOperator(
      BuiltinOptions_SqueezeOptions, CreateSqueezeOptions(builder_).Union());
  void* output_data = nullptr;
  EXPECT_TRUE(
      ParseOpData(op, BuiltinOperator_SQUEEZE, &mock_allocator_, &output_data)
          .ok());
}

TEST_F(FlatbufferConversionsTest, ParseDynamicReshape) {
  const Operator* op = BuildTestOperator(
      BuiltinOptions_ReshapeOptions, CreateReshapeOptions(builder_).Union());
  void* output_data = nullptr;
  EXPECT_TRUE(
      ParseOpData(op, BuiltinOperator_RESHAPE, &mock_allocator_, &output_data)
          .ok());
}

TEST_F(FlatbufferConversionsTest, TestParseOpDataConv) {
  const Operator* conv_op =
      BuildTestOperator(BuiltinOptions_Conv2DOptions,
                        CreateConv2DOptions(builder_, Padding_SAME, 1, 2,
                                            ActivationFunctionType_RELU, 3, 4)
                            .Union());
  void* output_data = nullptr;
  EXPECT_TRUE(ParseOpData(conv_op, BuiltinOperator_CONV_2D, &mock_allocator_,
                          &output_data)
                  .ok());
  EXPECT_NE(nullptr, output_data);
  TfLiteConvParams* params = reinterpret_cast<TfLiteConvParams*>(output_data);
  EXPECT_EQ(kTfLitePaddingSame, params->padding);
  EXPECT_EQ(1, params->stride_width);
  EXPECT_EQ(2, params->stride_height);
  EXPECT_EQ(kTfLiteActRelu, params->activation);
  EXPECT_EQ(3, params->dilation_width_factor);
  EXPECT_EQ(4, params->dilation_height_factor);
}

TEST_F(FlatbufferConversionsTest, ParseBadFullyConnected) {
  const Operator* conv_op = BuildTestOperator(
      BuiltinOptions_FullyConnectedOptions,
      CreateFullyConnectedOptions(
          builder_, ActivationFunctionType_RELU,
          static_cast<FullyConnectedOptionsWeightsFormat>(-1), true)
          .Union());
  void* output_data = nullptr;
  EXPECT_FALSE(ParseOpData(conv_op, BuiltinOperator_FULLY_CONNECTED,
                           &mock_allocator_, &output_data)
                   .ok());
}

TEST_F(FlatbufferConversionsTest, TestParseOpDataCustom) {
  const Operator* custom_op =
      BuildTestOperator(BuiltinOptions_NONE, flatbuffers::Offset<void>());
  void* output_data = nullptr;
  EXPECT_TRUE(ParseOpData(custom_op, BuiltinOperator_CUSTOM, &mock_allocator_,
                          &output_data)
                  .ok());
  EXPECT_EQ(nullptr, output_data);
}

TEST_F(FlatbufferConversionsTest, TestConvertTensorType) {
  TfLiteType type;
  EXPECT_TRUE(ConvertTensorType(TensorType_FLOAT32, &type).ok());
  EXPECT_EQ(kTfLiteFloat32, type);
}

TEST_F(FlatbufferConversionsTest, TestConvertTensorTypeFloat16) {
  TfLiteType type;
  EXPECT_TRUE(ConvertTensorType(TensorType_FLOAT16, &type).ok());
  EXPECT_EQ(kTfLiteFloat16, type);
}

TEST_F(FlatbufferConversionsTest, TestConvertTensorTypeBFloat16) {
  TfLiteType type;
  EXPECT_TRUE(ConvertTensorType(TensorType_BFLOAT16, &type).ok());
  EXPECT_EQ(kTfLiteBFloat16, type);
}

TEST_F(FlatbufferConversionsTest, TestConvertTensorTypeInt4) {
  TfLiteType type;
  EXPECT_TRUE(ConvertTensorType(TensorType_INT4, &type).ok());
  EXPECT_EQ(kTfLiteInt4, type);
}

class StablehloReduceWindowFlatbufferConversionsTest
    : public FlatbufferConversionsTest {
 public:
  static constexpr int kMaxDims =
      TFLITE_STABLEHLO_REDUCE_WINDOW_PARAMS_MAX_DIMENSION_COUNT;
  static constexpr int64_t kValidValue = 5;

  auto ValidAttr() {
    return builder_.CreateVector(std::vector<int64_t>(kMaxDims, kValidValue));
  }

  auto InvalidAttr() {
    return builder_.CreateVector(
        std::vector<int64_t>(kMaxDims + 1, kValidValue));
  }

  auto ValidPaddingAttr() {
    return builder_.CreateVector(
        std::vector<int64_t>(2 * kMaxDims, kValidValue));
  }

  auto InvalidPaddingAttr() {
    return builder_.CreateVector(
        std::vector<int64_t>(2 * kMaxDims + 1, kValidValue));
  }

  auto EmptyAttr() { return builder_.CreateVector<int64_t>({}); }
};

TEST_F(StablehloReduceWindowFlatbufferConversionsTest, Succeeds) {
  const Operator* stablehlo_reduce_window_op = BuildTestOperator(
      BuiltinOptions2_StablehloReduceWindowOptions,
      CreateStablehloReduceWindowOptions(
          builder_,
          /*window_dimensions=*/builder_.CreateVector<int64_t>({1, 2}),
          /*window_strides=*/builder_.CreateVector<int64_t>({3, 4}),
          /*base_dilations=*/builder_.CreateVector<int64_t>({5, 6}),
          /*window_dilations=*/builder_.CreateVector<int64_t>({7, 8}),
          /*padding=*/builder_.CreateVector<int64_t>({9, 10, 11, 12}),
          /*body_subgraph_index=*/13)
          .Union());
  TfLiteStablehloReduceWindowParams* output_data = nullptr;
  EXPECT_TRUE(ParseOpData(stablehlo_reduce_window_op,
                          BuiltinOperator_STABLEHLO_REDUCE_WINDOW,
                          &mock_allocator_, (void**)&output_data)
                  .ok());

  EXPECT_THAT(std::make_tuple(output_data->window_dimensions, 2),
              ElementsAre(1, 2));
  EXPECT_THAT(std::make_tuple(output_data->window_strides, 2),
              ElementsAre(3, 4));
  EXPECT_THAT(std::make_tuple(output_data->base_dilations, 2),
              ElementsAre(5, 6));
  EXPECT_THAT(std::make_tuple(output_data->window_dilations, 2),
              ElementsAre(7, 8));
  EXPECT_THAT(std::make_tuple(output_data->padding, 4),
              ElementsAre(9, 10, 11, 12));
  EXPECT_THAT(output_data->body_subgraph_index, Eq(13));
}

TEST_F(StablehloReduceWindowFlatbufferConversionsTest,
       FailsWithNoWindowDimensions) {
  TfLiteStablehloReduceWindowParams* output_data = nullptr;
  auto status = ParseOpData(
      BuildTestOperator(
          BuiltinOptions2_StablehloReduceWindowOptions,
          CreateStablehloReduceWindowOptions(builder_,
                                             /*window_dimensions=*/0,
                                             /*window_strides=*/ValidAttr(),
                                             /*base_dilations=*/ValidAttr(),
                                             /*window_dilations=*/ValidAttr(),
                                             /*padding=*/ValidPaddingAttr(),
                                             /*body_subgraph_index=*/13)
              .Union()),
      BuiltinOperator_STABLEHLO_REDUCE_WINDOW, &mock_allocator_,
      (void**)&output_data);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("'window_dimensions' attribute is not optional for "
                        "'stablehlo.reduce_window' and cannot be empty."));
}

TEST_F(StablehloReduceWindowFlatbufferConversionsTest,
       SucceedsWithNoWindowStrides) {
  TfLiteStablehloReduceWindowParams* output_data = nullptr;
  auto status = ParseOpData(
      BuildTestOperator(
          BuiltinOptions2_StablehloReduceWindowOptions,
          CreateStablehloReduceWindowOptions(builder_,
                                             /*window_dimensions=*/ValidAttr(),
                                             /*window_strides=*/0,
                                             /*base_dilations=*/ValidAttr(),
                                             /*window_dilations=*/ValidAttr(),
                                             /*padding=*/ValidPaddingAttr(),
                                             /*body_subgraph_index=*/13)
              .Union()),
      BuiltinOperator_STABLEHLO_REDUCE_WINDOW, &mock_allocator_,
      (void**)&output_data);
  EXPECT_TRUE(status.ok());
  EXPECT_THAT(status.message(), StrEq(""));
  EXPECT_THAT(std::make_tuple(output_data->window_dimensions, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->window_strides, kMaxDims), Each(1));
  EXPECT_THAT(std::make_tuple(output_data->base_dilations, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->window_dilations, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->padding, 2 * kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(output_data->body_subgraph_index, Eq(13));
}

TEST_F(StablehloReduceWindowFlatbufferConversionsTest,
       SucceedsWithNoBaseDilations) {
  TfLiteStablehloReduceWindowParams* output_data = nullptr;
  auto status = ParseOpData(
      BuildTestOperator(
          BuiltinOptions2_StablehloReduceWindowOptions,
          CreateStablehloReduceWindowOptions(builder_,
                                             /*window_dimensions=*/ValidAttr(),
                                             /*window_strides=*/ValidAttr(),
                                             /*base_dilations=*/0,
                                             /*window_dilations=*/ValidAttr(),
                                             /*padding=*/ValidPaddingAttr(),
                                             /*body_subgraph_index=*/13)
              .Union()),
      BuiltinOperator_STABLEHLO_REDUCE_WINDOW, &mock_allocator_,
      (void**)&output_data);
  EXPECT_TRUE(status.ok());
  EXPECT_THAT(status.message(), StrEq(""));
  EXPECT_THAT(std::make_tuple(output_data->window_dimensions, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->window_strides, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->base_dilations, kMaxDims), Each(1));
  EXPECT_THAT(std::make_tuple(output_data->window_dilations, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->padding, 2 * kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(output_data->body_subgraph_index, Eq(13));
}

TEST_F(StablehloReduceWindowFlatbufferConversionsTest,
       SucceedsWithNoWindowDilations) {
  TfLiteStablehloReduceWindowParams* output_data = nullptr;
  auto status = ParseOpData(
      BuildTestOperator(
          BuiltinOptions2_StablehloReduceWindowOptions,
          CreateStablehloReduceWindowOptions(builder_,
                                             /*window_dimensions=*/ValidAttr(),
                                             /*window_strides=*/ValidAttr(),
                                             /*base_dilations=*/ValidAttr(),
                                             /*window_dilations=*/0,
                                             /*padding=*/ValidPaddingAttr(),
                                             /*body_subgraph_index=*/13)
              .Union()),
      BuiltinOperator_STABLEHLO_REDUCE_WINDOW, &mock_allocator_,
      (void**)&output_data);
  EXPECT_TRUE(status.ok());
  EXPECT_THAT(status.message(), StrEq(""));
  EXPECT_THAT(std::make_tuple(output_data->window_dimensions, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->window_strides, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->base_dilations, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->window_dilations, kMaxDims),
              Each(1));
  EXPECT_THAT(std::make_tuple(output_data->padding, 2 * kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(output_data->body_subgraph_index, Eq(13));
}

TEST_F(StablehloReduceWindowFlatbufferConversionsTest, SucceedsWithNoPadding) {
  TfLiteStablehloReduceWindowParams* output_data = nullptr;
  auto status = ParseOpData(
      BuildTestOperator(
          BuiltinOptions2_StablehloReduceWindowOptions,
          CreateStablehloReduceWindowOptions(builder_,
                                             /*window_dimensions=*/ValidAttr(),
                                             /*window_strides=*/ValidAttr(),
                                             /*base_dilations=*/ValidAttr(),
                                             /*window_dilations=*/ValidAttr(),
                                             /*padding=*/0,
                                             /*body_subgraph_index=*/13)
              .Union()),
      BuiltinOperator_STABLEHLO_REDUCE_WINDOW, &mock_allocator_,
      (void**)&output_data);
  EXPECT_TRUE(status.ok());
  EXPECT_THAT(status.message(), StrEq(""));
  EXPECT_THAT(std::make_tuple(output_data->window_dimensions, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->window_strides, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->base_dilations, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->window_dilations, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->padding, 2 * kMaxDims), Each(0));
  EXPECT_THAT(output_data->body_subgraph_index, Eq(13));
}

TEST_F(StablehloReduceWindowFlatbufferConversionsTest,
       FailsWithEmptyWindowDimensions) {
  TfLiteStablehloReduceWindowParams* output_data = nullptr;
  auto status = ParseOpData(
      BuildTestOperator(
          BuiltinOptions2_StablehloReduceWindowOptions,
          CreateStablehloReduceWindowOptions(builder_,
                                             /*window_dimensions=*/EmptyAttr(),
                                             /*window_strides=*/ValidAttr(),
                                             /*base_dilations=*/ValidAttr(),
                                             /*window_dilations=*/ValidAttr(),
                                             /*padding=*/ValidPaddingAttr(),
                                             /*body_subgraph_index=*/13)
              .Union()),
      BuiltinOperator_STABLEHLO_REDUCE_WINDOW, &mock_allocator_,
      (void**)&output_data);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("'window_dimensions' attribute is not optional for "
                        "'stablehlo.reduce_window' and cannot be empty."));
}

TEST_F(StablehloReduceWindowFlatbufferConversionsTest,
       SucceedsWithEmptyWindowStrides) {
  TfLiteStablehloReduceWindowParams* output_data = nullptr;
  auto status = ParseOpData(
      BuildTestOperator(
          BuiltinOptions2_StablehloReduceWindowOptions,
          CreateStablehloReduceWindowOptions(builder_,
                                             /*window_dimensions=*/ValidAttr(),
                                             /*window_strides=*/EmptyAttr(),
                                             /*base_dilations=*/ValidAttr(),
                                             /*window_dilations=*/ValidAttr(),
                                             /*padding=*/ValidPaddingAttr(),
                                             /*body_subgraph_index=*/13)
              .Union()),
      BuiltinOperator_STABLEHLO_REDUCE_WINDOW, &mock_allocator_,
      (void**)&output_data);
  EXPECT_TRUE(status.ok());
  EXPECT_THAT(status.message(), StrEq(""));
  EXPECT_THAT(std::make_tuple(output_data->window_dimensions, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->window_strides, kMaxDims), Each(1));
  EXPECT_THAT(std::make_tuple(output_data->base_dilations, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->window_dilations, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->padding, 2 * kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(output_data->body_subgraph_index, Eq(13));
}

TEST_F(StablehloReduceWindowFlatbufferConversionsTest,
       SucceedsWithEmptyBaseDilations) {
  TfLiteStablehloReduceWindowParams* output_data = nullptr;
  auto status = ParseOpData(
      BuildTestOperator(
          BuiltinOptions2_StablehloReduceWindowOptions,
          CreateStablehloReduceWindowOptions(builder_,
                                             /*window_dimensions=*/ValidAttr(),
                                             /*window_strides=*/ValidAttr(),
                                             /*base_dilations=*/EmptyAttr(),
                                             /*window_dilations=*/ValidAttr(),
                                             /*padding=*/ValidPaddingAttr(),
                                             /*body_subgraph_index=*/13)
              .Union()),
      BuiltinOperator_STABLEHLO_REDUCE_WINDOW, &mock_allocator_,
      (void**)&output_data);
  EXPECT_TRUE(status.ok());
  EXPECT_THAT(status.message(), StrEq(""));
  EXPECT_THAT(std::make_tuple(output_data->window_dimensions, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->window_strides, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->base_dilations, kMaxDims), Each(1));
  EXPECT_THAT(std::make_tuple(output_data->window_dilations, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->padding, 2 * kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(output_data->body_subgraph_index, Eq(13));
}

TEST_F(StablehloReduceWindowFlatbufferConversionsTest,
       SucceedsWithEmptyWindowDilations) {
  TfLiteStablehloReduceWindowParams* output_data = nullptr;
  auto status = ParseOpData(
      BuildTestOperator(
          BuiltinOptions2_StablehloReduceWindowOptions,
          CreateStablehloReduceWindowOptions(builder_,
                                             /*window_dimensions=*/ValidAttr(),
                                             /*window_strides=*/ValidAttr(),
                                             /*base_dilations=*/ValidAttr(),
                                             /*window_dilations=*/EmptyAttr(),
                                             /*padding=*/ValidPaddingAttr(),
                                             /*body_subgraph_index=*/13)
              .Union()),
      BuiltinOperator_STABLEHLO_REDUCE_WINDOW, &mock_allocator_,
      (void**)&output_data);
  EXPECT_TRUE(status.ok());
  EXPECT_THAT(status.message(), StrEq(""));
  EXPECT_THAT(std::make_tuple(output_data->window_dimensions, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->window_strides, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->base_dilations, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->window_dilations, kMaxDims),
              Each(1));
  EXPECT_THAT(std::make_tuple(output_data->padding, 2 * kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(output_data->body_subgraph_index, Eq(13));
}

TEST_F(StablehloReduceWindowFlatbufferConversionsTest,
       SucceedsWithEmptyPadding) {
  TfLiteStablehloReduceWindowParams* output_data = nullptr;
  auto status = ParseOpData(
      BuildTestOperator(
          BuiltinOptions2_StablehloReduceWindowOptions,
          CreateStablehloReduceWindowOptions(builder_,
                                             /*window_dimensions=*/ValidAttr(),
                                             /*window_strides=*/ValidAttr(),
                                             /*base_dilations=*/ValidAttr(),
                                             /*window_dilations=*/ValidAttr(),
                                             /*padding=*/EmptyAttr(),
                                             /*body_subgraph_index=*/13)
              .Union()),
      BuiltinOperator_STABLEHLO_REDUCE_WINDOW, &mock_allocator_,
      (void**)&output_data);
  EXPECT_TRUE(status.ok());
  EXPECT_THAT(status.message(), StrEq(""));
  EXPECT_THAT(std::make_tuple(output_data->window_dimensions, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->window_strides, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->base_dilations, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->window_dilations, kMaxDims),
              Each(kValidValue));
  EXPECT_THAT(std::make_tuple(output_data->padding, 2 * kMaxDims), Each(0));
  EXPECT_THAT(output_data->body_subgraph_index, Eq(13));
}

TEST_F(StablehloReduceWindowFlatbufferConversionsTest,
       SucceedsWithParamsAtMaxDims) {
  TfLiteStablehloReduceWindowParams* output_data = nullptr;
  auto status = ParseOpData(
      BuildTestOperator(
          BuiltinOptions2_StablehloReduceWindowOptions,
          CreateStablehloReduceWindowOptions(builder_,
                                             /*window_dimensions=*/ValidAttr(),
                                             /*window_strides=*/ValidAttr(),
                                             /*base_dilations=*/ValidAttr(),
                                             /*window_dilations=*/ValidAttr(),
                                             /*padding=*/ValidPaddingAttr(),
                                             /*body_subgraph_index=*/13)
              .Union()),
      BuiltinOperator_STABLEHLO_REDUCE_WINDOW, &mock_allocator_,
      (void**)&output_data);
  EXPECT_TRUE(status.ok());
  EXPECT_THAT(status.message(), StrEq(""));
}

TEST_F(StablehloReduceWindowFlatbufferConversionsTest,
       FailsWhenWindowDimensionsHasMoreThanMaxDims) {
  TfLiteStablehloReduceWindowParams* output_data = nullptr;
  auto status = ParseOpData(
      BuildTestOperator(BuiltinOptions2_StablehloReduceWindowOptions,
                        CreateStablehloReduceWindowOptions(
                            builder_,
                            /*window_dimensions=*/InvalidAttr(),
                            /*window_strides=*/ValidAttr(),
                            /*base_dilations=*/ValidAttr(),
                            /*window_dilations=*/ValidAttr(),
                            /*padding=*/ValidPaddingAttr(),
                            /*body_subgraph_index=*/13)
                            .Union()),
      BuiltinOperator_STABLEHLO_REDUCE_WINDOW, &mock_allocator_,
      (void**)&output_data);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              AllOf(HasSubstr("Found too many dimensions in the input array of "
                              "operation 'stablehlo.reduce_window'."),
                    HasSubstr("Check the 'window_dimensions' attribute.")));
}

TEST_F(StablehloReduceWindowFlatbufferConversionsTest,
       FailsWhenWindowStridesHasWrongDimCount) {
  TfLiteStablehloReduceWindowParams* output_data = nullptr;
  auto status = ParseOpData(
      BuildTestOperator(
          BuiltinOptions2_StablehloReduceWindowOptions,
          CreateStablehloReduceWindowOptions(builder_,
                                             /*window_dimensions=*/ValidAttr(),
                                             /*window_strides=*/InvalidAttr(),
                                             /*base_dilations=*/ValidAttr(),
                                             /*window_dilations=*/ValidAttr(),
                                             /*padding=*/ValidPaddingAttr(),
                                             /*body_subgraph_index=*/13)
              .Union()),
      BuiltinOperator_STABLEHLO_REDUCE_WINDOW, &mock_allocator_,
      (void**)&output_data);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr("'window_strides' attribute of 'stablehlo.reduce_window' does "
                "not have the expected size"));
}

TEST_F(StablehloReduceWindowFlatbufferConversionsTest,
       FailsWhenBaseDilationsHasWrongDimCount) {
  TfLiteStablehloReduceWindowParams* output_data = nullptr;
  auto status = ParseOpData(
      BuildTestOperator(
          BuiltinOptions2_StablehloReduceWindowOptions,
          CreateStablehloReduceWindowOptions(builder_,
                                             /*window_dimensions=*/ValidAttr(),
                                             /*window_strides=*/ValidAttr(),
                                             /*base_dilations=*/InvalidAttr(),
                                             /*window_dilations=*/ValidAttr(),
                                             /*padding=*/ValidPaddingAttr(),
                                             /*body_subgraph_index=*/13)
              .Union()),
      BuiltinOperator_STABLEHLO_REDUCE_WINDOW, &mock_allocator_,
      (void**)&output_data);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr("'base_dilations' attribute of 'stablehlo.reduce_window' does "
                "not have the expected size"));
}

TEST_F(StablehloReduceWindowFlatbufferConversionsTest,
       FailsWhenWindowDilationsHasWrongDimCount) {
  TfLiteStablehloReduceWindowParams* output_data = nullptr;
  auto status = ParseOpData(
      BuildTestOperator(
          BuiltinOptions2_StablehloReduceWindowOptions,
          CreateStablehloReduceWindowOptions(builder_,
                                             /*window_dimensions=*/ValidAttr(),
                                             /*window_strides=*/ValidAttr(),
                                             /*base_dilations=*/ValidAttr(),
                                             /*window_dilations=*/InvalidAttr(),
                                             /*padding=*/ValidPaddingAttr(),
                                             /*body_subgraph_index=*/13)
              .Union()),
      BuiltinOperator_STABLEHLO_REDUCE_WINDOW, &mock_allocator_,
      (void**)&output_data);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr(
          "'window_dilations' attribute of 'stablehlo.reduce_window' does "
          "not have the expected size"));
}

TEST_F(StablehloReduceWindowFlatbufferConversionsTest,
       FailsWhenPaddingHasWrongDimCount) {
  TfLiteStablehloReduceWindowParams* output_data = nullptr;
  auto status = ParseOpData(
      BuildTestOperator(
          BuiltinOptions2_StablehloReduceWindowOptions,
          CreateStablehloReduceWindowOptions(builder_,
                                             /*window_dimensions=*/ValidAttr(),
                                             /*window_strides=*/ValidAttr(),
                                             /*base_dilations=*/ValidAttr(),
                                             /*window_dilations=*/ValidAttr(),
                                             /*padding=*/InvalidPaddingAttr(),
                                             /*body_subgraph_index=*/13)
              .Union()),
      BuiltinOperator_STABLEHLO_REDUCE_WINDOW, &mock_allocator_,
      (void**)&output_data);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("'padding' attribute of 'stablehlo.reduce_window' does "
                        "not have the expected size"));
}

TEST_F(StablehloReduceWindowFlatbufferConversionsTest, FailsWithWrongOptions) {
  const Operator* stablehlo_reduce_window_op =
      BuildTestOperator(BuiltinOptions2_StablehloReduceWindowOptions, 0);
  TfLiteStablehloReduceWindowParams* output_data = nullptr;
  auto status = ParseOpData(stablehlo_reduce_window_op,
                            BuiltinOperator_STABLEHLO_REDUCE_WINDOW,
                            &mock_allocator_, (void**)&output_data);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr(
          "Could not get 'stablehlo.reduce_window' operation parameters."));
}

TEST_F(StablehloReduceWindowFlatbufferConversionsTest, DeathTests) {
  const Operator* stablehlo_reduce_window_op = BuildTestOperator(
      BuiltinOptions2_StablehloReduceWindowOptions,
      CreateStablehloReduceWindowOptions(
          builder_, /*window_dimensions=*/ValidAttr(),
          /*window_strides=*/ValidAttr(),
          /*base_dilations=*/ValidAttr(),
          /*window_dilations=*/ValidAttr(),
          /*padding=*/ValidPaddingAttr(), /*body_subgraph_index=*/13)
          .Union());
  TfLiteStablehloReduceWindowParams* output_data = nullptr;
#ifdef NDEBUG
  GTEST_SKIP();
#endif
  EXPECT_DEATH(ParseOpData(nullptr, BuiltinOperator_STABLEHLO_REDUCE_WINDOW,
                           &mock_allocator_, (void**)&output_data)
                   .IgnoreError(),
               "");
  EXPECT_DEATH(ParseOpData(stablehlo_reduce_window_op,
                           BuiltinOperator_STABLEHLO_REDUCE_WINDOW, nullptr,
                           (void**)&output_data)
                   .IgnoreError(),
               "");
  EXPECT_DEATH(ParseOpData(stablehlo_reduce_window_op,
                           BuiltinOperator_STABLEHLO_REDUCE_WINDOW,
                           &mock_allocator_, nullptr)
                   .IgnoreError(),
               "");
}

class StablehloPadFlatbufferConversionsTest : public FlatbufferConversionsTest {
 public:
  static constexpr int kMaxDims =
      TFLITE_STABLEHLO_PAD_PARAMS_MAX_DIMENSION_COUNT;
  static constexpr int64_t kValidValue = 5;
};

TEST_F(StablehloPadFlatbufferConversionsTest, Succeeds) {
  const Operator* stablehlo_pad_op = BuildTestOperator(
      BuiltinOptions2_StablehloPadOptions,
      CreateStablehloPadOptions(
          builder_,
          /*edge_padding_low=*/builder_.CreateVector<int64_t>({1, 0, -1}),
          /*edge_padding_high=*/builder_.CreateVector<int64_t>({2, 0, -2}),
          /*interior_padding=*/builder_.CreateVector<int64_t>({3, 0, 3}))
          .Union());
  TfLiteStablehloPadParams* output_data = nullptr;
  EXPECT_TRUE(ParseOpData(stablehlo_pad_op, BuiltinOperator_STABLEHLO_PAD,
                          &mock_allocator_, (void**)&output_data)
                  .ok());
  EXPECT_THAT(std::make_tuple(output_data->edge_padding_low, 3),
              ElementsAre(1, 0, -1));
  EXPECT_THAT(std::make_tuple(output_data->edge_padding_high, 3),
              ElementsAre(2, 0, -2));
  EXPECT_THAT(std::make_tuple(output_data->interior_padding, 3),
              ElementsAre(3, 0, 3));
}

TEST_F(StablehloPadFlatbufferConversionsTest, FailsWithMissingLowPadding) {
  const Operator* stablehlo_pad_op = BuildTestOperator(
      BuiltinOptions2_StablehloPadOptions,
      CreateStablehloPadOptions(
          builder_,
          /*edge_padding_low=*/0,
          /*edge_padding_high=*/builder_.CreateVector<int64_t>({2, 0, -2}),
          /*interior_padding=*/builder_.CreateVector<int64_t>({3, 0, 3}))
          .Union());
  TfLiteStablehloPadParams* output_data = nullptr;
  auto status = ParseOpData(stablehlo_pad_op, BuiltinOperator_STABLEHLO_PAD,
                            &mock_allocator_, (void**)&output_data);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      AllOf(
          HasSubstr("Input array not provided for operation 'stablehlo.pad'."),
          HasSubstr("Check the 'edge_padding_low' attribute.")));
}

TEST_F(StablehloPadFlatbufferConversionsTest, FailsWithMissingHighPadding) {
  const Operator* stablehlo_pad_op = BuildTestOperator(
      BuiltinOptions2_StablehloPadOptions,
      CreateStablehloPadOptions(
          builder_,
          /*edge_padding_low=*/builder_.CreateVector<int64_t>({1, 0, -1}),
          /*edge_padding_high=*/0,
          /*interior_padding=*/builder_.CreateVector<int64_t>({3, 0, 3}))
          .Union());
  TfLiteStablehloPadParams* output_data = nullptr;
  auto status = ParseOpData(stablehlo_pad_op, BuiltinOperator_STABLEHLO_PAD,
                            &mock_allocator_, (void**)&output_data);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      AllOf(
          HasSubstr("Input array not provided for operation 'stablehlo.pad'."),
          HasSubstr("Check the 'edge_padding_high' attribute.")));
}

TEST_F(StablehloPadFlatbufferConversionsTest, FailsWithMissingInteriorPadding) {
  const Operator* stablehlo_pad_op = BuildTestOperator(
      BuiltinOptions2_StablehloPadOptions,
      CreateStablehloPadOptions(
          builder_,
          /*edge_padding_low=*/builder_.CreateVector<int64_t>({1, 0, -1}),
          /*edge_padding_high=*/builder_.CreateVector<int64_t>({2, 0, -2}),
          /*interior_padding=*/0)
          .Union());
  TfLiteStablehloPadParams* output_data = nullptr;
  auto status = ParseOpData(stablehlo_pad_op, BuiltinOperator_STABLEHLO_PAD,
                            &mock_allocator_, (void**)&output_data);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      AllOf(
          HasSubstr("Input array not provided for operation 'stablehlo.pad'."),
          HasSubstr("Check the 'interior_padding' attribute.")));
}

TEST_F(StablehloPadFlatbufferConversionsTest, FailsInconsistentSizes) {
  const Operator* stablehlo_pad_op = BuildTestOperator(
      BuiltinOptions2_StablehloPadOptions,
      CreateStablehloPadOptions(
          builder_,
          /*edge_padding_low=*/builder_.CreateVector<int64_t>({1, 0, -1}),
          /*edge_padding_high=*/builder_.CreateVector<int64_t>({2, 0, -2}),
          /*interior_padding=*/builder_.CreateVector<int64_t>({3, 0, -3, 5}))
          .Union());
  TfLiteStablehloPadParams* output_data = nullptr;
  auto status = ParseOpData(stablehlo_pad_op, BuiltinOperator_STABLEHLO_PAD,
                            &mock_allocator_, (void**)&output_data);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("'stablehlo.pad' operation parameter array sizes are "
                        "not consistent."));
}

TEST_F(StablehloPadFlatbufferConversionsTest, FailsWithWrongOptions) {
  const Operator* stablehlo_pad_op = BuildTestOperator(BuiltinOptions_NONE, 0);
  TfLiteStablehloPadParams* output_data = nullptr;
  auto status = ParseOpData(stablehlo_pad_op, BuiltinOperator_STABLEHLO_PAD,
                            &mock_allocator_, (void**)&output_data);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Could not get 'stablehlo.pad' operation parameters."));
}

TEST_F(StablehloPadFlatbufferConversionsTest, DeathTests) {
  const Operator* stablehlo_pad_op = BuildTestOperator(BuiltinOptions_NONE, 0);
  TfLiteStablehloPadParams* output_data = nullptr;
#ifdef NDEBUG
  GTEST_SKIP();
#endif
  EXPECT_DEATH(ParseOpData(nullptr, BuiltinOperator_STABLEHLO_PAD,
                           &mock_allocator_, (void**)&output_data)
                   .IgnoreError(),
               "");
  EXPECT_DEATH(ParseOpData(stablehlo_pad_op, BuiltinOperator_STABLEHLO_PAD,
                           nullptr, (void**)&output_data)
                   .IgnoreError(),
               "");
  EXPECT_DEATH(ParseOpData(stablehlo_pad_op, BuiltinOperator_STABLEHLO_PAD,
                           &mock_allocator_, nullptr)
                   .IgnoreError(),
               "");
}

}  // namespace flatbuffer_conversions
}  // namespace tflite_file
