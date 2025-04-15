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

#include <cstdint>
#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class StablehloGatherOpModel : public SingleOpModel {
 public:
  StablehloGatherOpModel(const TensorData& input, const TensorData& indices,
                         const TfLiteStablehloGatherParams& params) {
    input_ = AddInput(input);
    indices_ = AddInput(indices);
    output_ = AddOutput(TensorData(input.type, {2, 3, 2, 2}));
    SetBuiltinOp(
        BuiltinOperator_STABLEHLO_GATHER,
        BuiltinOptions2_StablehloGatherOptions,
        CreateStablehloGatherOptions(
            builder_,
            builder_.CreateVector(
                std::vector(params.offset_dims,
                            params.offset_dims + params.num_offset_dims)),
            builder_.CreateVector(std::vector(
                params.collapsed_slice_dims,
                params.collapsed_slice_dims + params.num_collapsed_slice_dims)),
            builder_.CreateVector(std::vector(
                params.start_index_map,
                params.start_index_map + params.num_start_index_map)),
            params.index_vector_dim,
            builder_.CreateVector(
                std::vector(params.slice_sizes,
                            params.slice_sizes + params.num_slice_sizes)),
            params.indices_are_sorted)
            .Union());
    BuildInterpreter({GetShape(input_), GetShape(indices_)});
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor<T>(input_, data);
  }

  template <typename T>
  void SetIndices(std::initializer_list<T> data) {
    PopulateTensor<T>(indices_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 protected:
  int input_;
  int indices_;
  int output_;
};

TEST(StablehloScatterOpTest, GathersSlices) {
  TfLiteStablehloGatherParams params = {
      {2, 3},     // offset_dims
      2,          // num_offset_dims;
      {0},        // collapsed_slice_dims
      1,          // num_collapsed_slice_dims;
      {1, 0},     // start_index_map
      2,          // num_start_index_map;
      2,          // index_vector_dim;
      {1, 2, 2},  // slice_sizes
      3,          // num_slice_sizes;
      false       // indices_are_sorted;
  };
  StablehloGatherOpModel model({TensorType_FLOAT32, {3, 4, 2}},
                               {TensorType_INT64, {2, 3, 2}}, params);

  model.SetInput<float>({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                         13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  model.SetIndices<int64_t>({0, 0, 1, 0, 2, 1, 0, 1, 1, 1, 0, 2});

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<float> expected_values = {1,  2,  3,  4,  3,  4,  5,  6,
                                        13, 14, 15, 16, 9,  10, 11, 12,
                                        11, 12, 13, 14, 17, 18, 19, 20};
  EXPECT_THAT(model.GetOutput<float>(), ElementsAreArray(expected_values));
}

TEST(StablehloScatterOpTest, ClipsStartingIndices) {
  TfLiteStablehloGatherParams params = {
      {2, 3},     // offset_dims
      2,          // num_offset_dims;
      {0},        // collapsed_slice_dims
      1,          // num_collapsed_slice_dims;
      {1, 0},     // start_index_map
      2,          // num_start_index_map;
      2,          // index_vector_dim;
      {1, 2, 2},  // slice_sizes
      3,          // num_slice_sizes;
      false       // indices_are_sorted;
  };
  StablehloGatherOpModel model({TensorType_FLOAT32, {3, 4, 2}},
                               {TensorType_INT64, {2, 3, 2}}, params);

  model.SetInput<float>({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                         13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  model.SetIndices<int64_t>({0, 0, 1, 0, 2, 1, 0, 1, 1, 1, 0, 9});

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<float> expected_values = {1,  2,  3,  4,  3,  4,  5,  6,
                                        13, 14, 15, 16, 9,  10, 11, 12,
                                        11, 12, 13, 14, 17, 18, 19, 20};
  EXPECT_THAT(model.GetOutput<float>(), ElementsAreArray(expected_values));
}

TEST(StablehloScatterOpTest, WorksWithDynamicShapes) {
  TfLiteStablehloGatherParams params = {
      {2, 3},     // offset_dims
      2,          // num_offset_dims;
      {0},        // collapsed_slice_dims
      1,          // num_collapsed_slice_dims;
      {1, 0},     // start_index_map
      2,          // num_start_index_map;
      2,          // index_vector_dim;
      {1, 2, 2},  // slice_sizes
      3,          // num_slice_sizes;
      false       // indices_are_sorted;
  };

  TensorData indices_tensor = {TensorType_INT64,
                               /*shape*/ {2, 3, 2},
                               /*min*/ 0.0f,
                               /*max*/ 0.0f,
                               /*scale*/ 0.0f,
                               /*zero_point*/ 0,
                               /*per_channel_quantization*/ false,
                               /*per_channel_quantization_scales*/ {},
                               /*per_channel_quantization_offsets*/ {},
                               /*channel_index*/ 0,
                               /*traversal_order*/ {},
                               /*format*/ {},
                               /*block_size*/ {},
                               /*block_map*/ {},
                               /*shape_signature*/ {{-1, -1, 2}}};

  // shape_signature when creating the model has -1 for unknown dimension sizes.
  // After building the interpreter, `model.BuildInterpreter` resizes the
  // tensors with the actual shape.
  StablehloGatherOpModel model({TensorType_FLOAT32, {3, 4, 2}}, indices_tensor,
                               params);

  model.SetInput<float>({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                         13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  model.SetIndices<int64_t>({0, 0, 1, 0, 2, 1, 0, 1, 1, 1, 0, 9});

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<float> expected_values = {1,  2,  3,  4,  3,  4,  5,  6,
                                        13, 14, 15, 16, 9,  10, 11, 12,
                                        11, 12, 13, 14, 17, 18, 19, 20};
  EXPECT_THAT(model.GetOutput<float>(), ElementsAreArray(expected_values));
}

}  // namespace
}  // namespace tflite
