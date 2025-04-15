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
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

enum class StablehloScatterOpType { kAdd, kMul, kMax, kMin, kUpdate };

class StablehloScatterOpModel : public SingleOpModel {
 public:
  StablehloScatterOpModel(const TensorData& input, const TensorData& indices,
                          const TensorData& updates,
                          const TfLiteStablehloScatterParams& params,
                          StablehloScatterOpType op_type) {
    input_ = AddInput(input);
    indices_ = AddInput(indices);
    updates_ = AddInput(updates);
    output_ = AddOutput(input.type);
    SetBuiltinOp(
        BuiltinOperator_STABLEHLO_SCATTER,
        BuiltinOptions2_StablehloScatterOptions,
        CreateStablehloScatterOptions(
            builder_, params.indices_are_sorted,
            builder_.CreateVector(std::vector(
                params.update_window_dims,
                params.update_window_dims + params.num_update_window_dims)),
            builder_.CreateVector(std::vector(
                params.inserted_window_dims,
                params.inserted_window_dims + params.num_inserted_window_dims)),
            builder_.CreateVector(
                std::vector(params.scatter_dims_to_operand_dims,
                            params.scatter_dims_to_operand_dims +
                                params.num_scatter_dims_to_operand_dims)),
            params.index_vector_dim, params.unique_indices, 1)
            .Union());
    BuildInterpreter({GetShape(input_), GetShape(indices_), GetShape(updates_)},
                     /*num_threads=*/-1, /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/false, /*allocate_and_delegate=*/false,
                     /*use_simple_allocator=*/false);

    int* dummy = nullptr;
    AddSubgraphs(1, dummy);
    if (op_type == StablehloScatterOpType::kAdd) {
      subgraph_builder_.BuildStablehloAddSubgraph(interpreter_->subgraph(1));
    } else if (op_type == StablehloScatterOpType::kMul) {
      subgraph_builder_.BuildStablehloMulSubgraph(interpreter_->subgraph(1));
    } else if (op_type == StablehloScatterOpType::kMax) {
      subgraph_builder_.BuildStablehloMaximumSubgraph(
          interpreter_->subgraph(1));
    } else if (op_type == StablehloScatterOpType::kMin) {
      subgraph_builder_.BuildStablehloMinimumSubgraph(
          interpreter_->subgraph(1));
    } else if (op_type == StablehloScatterOpType::kUpdate) {
      subgraph_builder_.BuildOutputIsSecondInputSubgraph(
          interpreter_->subgraph(1));
    }

    // This calls Prepare on ops, so it's important to call this *after*
    // all the subgraphs are added.
    AllocateAndDelegate(true);
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
  void SetUpdates(std::initializer_list<T> data) {
    PopulateTensor<T>(updates_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 protected:
  Subgraph* subgraph_;
  int input_;
  int indices_;
  int updates_;
  int output_;

  subgraph_test_util::SubgraphBuilder subgraph_builder_;
};

TEST(StablehloScatterOpTest, PerformsAddition) {
  StablehloScatterOpType op_type = StablehloScatterOpType::kAdd;

  TfLiteStablehloScatterParams params = {
      false,   // indices_are_sorted
      {2, 3},  // std::vector<update_window_dims>
      2,       // num_update_window_dims
      {0},     // std::vector<inserted_window_dims>
      1,       // num_inserted_window_dims
      {1, 0},  // std::vector<scatter_dims_to_operand_dims>
      2,       // num_scatter_dims_to_operand_dims
      2,       // index_vector_dim
      false,   // unique_indices
      1        // update_computation_subgraph_index
  };
  StablehloScatterOpModel model(
      {TensorType_FLOAT32, {3, 4, 2}}, {TensorType_INT64, {2, 3, 2}},
      {TensorType_FLOAT32, {2, 3, 2, 2}}, params, op_type);
  model.SetInput<float>({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                         13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  model.SetIndices<int64_t>({0, 2, 1, 0, 2, 1, 0, 1, 1, 0, 0, 9});
  model.SetUpdates<float>(
      {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<float> expected_values = {1,  2,  7,  8,  9,  10, 7,  8,
                                        11, 12, 13, 14, 15, 16, 17, 18,
                                        19, 20, 21, 22, 21, 22, 23, 24};
  EXPECT_THAT(model.GetOutput<float>(), ElementsAreArray(expected_values));
}

TEST(StablehloScatterOpTest, PerformsMultiplication) {
  StablehloScatterOpType op_type = StablehloScatterOpType::kMul;

  TfLiteStablehloScatterParams params = {
      false,   // indices_are_sorted
      {2, 3},  // std::vector<update_window_dims>
      2,       // num_update_window_dims
      {0},     // std::vector<inserted_window_dims>
      1,       // num_inserted_window_dims
      {1, 0},  // std::vector<scatter_dims_to_operand_dims>
      2,       // num_scatter_dims_to_operand_dims
      2,       // index_vector_dim
      false,   // unique_indices
      1        // update_computation_subgraph_index
  };
  StablehloScatterOpModel model(
      {TensorType_FLOAT32, {3, 4, 2}}, {TensorType_INT64, {2, 3, 2}},
      {TensorType_FLOAT32, {2, 3, 2, 2}}, params, op_type);
  model.SetInput<float>({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                         13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  model.SetIndices<int64_t>({0, 2, 1, 0, 2, 1, 0, 1, 1, 0, 0, 9});
  model.SetUpdates<float>(
      {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<float> expected_values = {1,  2,  12, 16, 20, 24, 7,  8,
                                        18, 20, 22, 24, 26, 28, 30, 32,
                                        34, 36, 38, 40, 21, 22, 23, 24};
  EXPECT_THAT(model.GetOutput<float>(), ElementsAreArray(expected_values));
}

TEST(StablehloScatterOpTest, PerformsMaximum) {
  StablehloScatterOpType op_type = StablehloScatterOpType::kMax;

  TfLiteStablehloScatterParams params = {
      false,   // indices_are_sorted
      {2, 3},  // std::vector<update_window_dims>
      2,       // num_update_window_dims
      {0},     // std::vector<inserted_window_dims>
      1,       // num_inserted_window_dims
      {1, 0},  // std::vector<scatter_dims_to_operand_dims>
      2,       // num_scatter_dims_to_operand_dims
      2,       // index_vector_dim
      false,   // unique_indices
      1        // update_computation_subgraph_index
  };
  StablehloScatterOpModel model(
      {TensorType_FLOAT32, {3, 4, 2}}, {TensorType_INT64, {2, 3, 2}},
      {TensorType_FLOAT32, {2, 3, 2, 2}}, params, op_type);
  model.SetInput<float>({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                         13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  model.SetIndices<int64_t>({0, 2, 1, 0, 2, 1, 0, 1, 1, 0, 0, 9});
  model.SetUpdates<float>(
      {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<float> expected_values = {1,  2,  3,  4,  5,  6,  7,  8,
                                        9,  10, 11, 12, 13, 14, 15, 16,
                                        17, 18, 19, 20, 21, 22, 23, 24};
  EXPECT_THAT(model.GetOutput<float>(), ElementsAreArray(expected_values));
}

TEST(StablehloScatterOpTest, PerformsMinimum) {
  StablehloScatterOpType op_type = StablehloScatterOpType::kMin;

  TfLiteStablehloScatterParams params = {
      false,   // indices_are_sorted
      {2, 3},  // std::vector<update_window_dims>
      2,       // num_update_window_dims
      {0},     // std::vector<inserted_window_dims>
      1,       // num_inserted_window_dims
      {1, 0},  // std::vector<scatter_dims_to_operand_dims>
      2,       // num_scatter_dims_to_operand_dims
      2,       // index_vector_dim
      false,   // unique_indices
      1        // update_computation_subgraph_index
  };
  StablehloScatterOpModel model(
      {TensorType_FLOAT32, {3, 4, 2}}, {TensorType_INT64, {2, 3, 2}},
      {TensorType_FLOAT32, {2, 3, 2, 2}}, params, op_type);
  model.SetInput<float>({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                         13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  model.SetIndices<int64_t>({0, 2, 1, 0, 2, 1, 0, 1, 1, 0, 0, 9});
  model.SetUpdates<float>(
      {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<float> expected_values = {1, 2, 2, 2, 2, 2, 7, 8, 2,  2,  2,  2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 21, 22, 23, 24};
  EXPECT_THAT(model.GetOutput<float>(), ElementsAreArray(expected_values));
}

TEST(StablehloScatterOpTest, PerformsUpdate) {
  StablehloScatterOpType op_type = StablehloScatterOpType::kUpdate;

  TfLiteStablehloScatterParams params = {
      false,   // indices_are_sorted
      {2, 3},  // std::vector<update_window_dims>
      2,       // num_update_window_dims
      {0},     // std::vector<inserted_window_dims>
      1,       // num_inserted_window_dims
      {1, 0},  // std::vector<scatter_dims_to_operand_dims>
      2,       // num_scatter_dims_to_operand_dims
      2,       // index_vector_dim
      false,   // unique_indices
      1        // update_computation_subgraph_index
  };
  StablehloScatterOpModel model(
      {TensorType_FLOAT32, {3, 4, 2}}, {TensorType_INT64, {2, 3, 2}},
      {TensorType_FLOAT32, {2, 3, 2, 2}}, params, op_type);
  model.SetInput<float>({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                         13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  model.SetIndices<int64_t>({0, 2, 1, 0, 2, 1, 0, 1, 1, 0, 0, 9});
  model.SetUpdates<float>(
      {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<float> expected_values = {1, 2, 2, 2, 2, 2, 7, 8, 2,  2,  2,  2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 21, 22, 23, 24};
  EXPECT_THAT(model.GetOutput<float>(), ElementsAreArray(expected_values));
}

}  // namespace
}  // namespace tflite
