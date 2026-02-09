// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

#include <initializer_list>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace ops {
namespace custom {
namespace text {
TfLiteRegistration* Register_RAGGED_TENSOR_TO_TENSOR();
}  // namespace text
}  // namespace custom
}  // namespace ops

namespace {

class RaggedTensorToTensorOpModel : public SingleOpModel {
 public:
  RaggedTensorToTensorOpModel(int output_shape_dims,
                              std::initializer_list<int> values_shape,
                              std::initializer_list<std::initializer_list<int>>
                                  partition_tensors_shapes,
                              std::vector<std::string> partition_types,
                              TensorType value_type = TensorType_FLOAT32,
                              TensorType index_type = TensorType_INT32,
                              bool allocate_and_delegate = true) {
    // A structure to collect shapes for the input.
    std::vector<std::vector<int>> shapes;
    input_shape_ = AddInput(index_type);
    shapes.push_back({output_shape_dims});
    input_values_ = AddInput(value_type);
    shapes.emplace_back(values_shape);
    input_default_values_ = AddInput(value_type);
    shapes.push_back({1});
    for (const auto& p : partition_tensors_shapes) {
      partition_tensors_.push_back(AddInput(TensorType_INT32));
      shapes.emplace_back(p);
    }
    output_ = AddOutput(value_type);

    flexbuffers::Builder fbb;
    size_t start = fbb.StartMap();
    {
      size_t start = fbb.StartVector("row_partition_types");
      for (const auto& s : partition_types) {
        fbb.String(s);
      }
      fbb.EndVector(start, /*typed=*/true, /*fixed=*/false);
    }
    fbb.Int("num_row_partition_tensors", partition_types.size());
    fbb.EndMap(start);
    fbb.Finish();
    SetCustomOp("RaggedTensorToTensor", fbb.GetBuffer(),
                ops::custom::text::Register_RAGGED_TENSOR_TO_TENSOR);
    BuildInterpreter(shapes, /*num_threads=*/-1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/true,
                     /*allocate_and_delegate=*/allocate_and_delegate,
                     /*use_simple_allocator=*/false);
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  std::vector<float> GetOutputFloat() { return ExtractVector<float>(output_); }
  std::vector<int32_t> GetOutputInt() {
    return ExtractVector<int32_t>(output_);
  }

  void InvokeFloat(const std::vector<int>& shape,
                   const std::vector<float>& values, float default_value,
                   const std::vector<std::vector<int>>& partition_values) {
    PopulateTensor(input_shape_, shape);
    PopulateTensor(input_values_, values);
    PopulateTensor(input_default_values_, {default_value});
    for (int i = 0; i < partition_values.size(); ++i) {
      PopulateTensor(partition_tensors_[i], partition_values[i]);
    }
    SingleOpModel::Invoke();
  }
  void InvokeInt(const std::vector<int>& shape,
                 const std::vector<int32_t>& values, int32_t default_value,
                 const std::vector<std::vector<int>>& partition_values) {
    PopulateTensor(input_shape_, shape);
    PopulateTensor(input_values_, values);
    PopulateTensor(input_default_values_, {default_value});
    for (int i = 0; i < partition_values.size(); ++i) {
      PopulateTensor(partition_tensors_[i], partition_values[i]);
    }
    SingleOpModel::Invoke();
  }
  TfLiteStatus TryAllocateTensors() { return interpreter_->AllocateTensors(); }

 private:
  int input_shape_;
  int input_values_;
  int input_default_values_;
  std::vector<int> partition_tensors_;
  int output_;
};

TEST(RaggedTensorToTensorTest, RaggedTensorToTensor) {
  // indices = [2, 1, 0, 3]
  // params = [[.1, .2, .3], [], [.4, .5, .6, .7], [.8, .9]]
  // params.shape = [4, None]
  RaggedTensorToTensorOpModel model(
      2,           // output_shape_dims
      {9},         // values_shape
      {{1}, {9}},  // partition_tensors_shapes
      std::vector<std::string>({"FIRST_DIM_SIZE", "VALUE_ROWIDS"}));
  model.InvokeFloat({4, 4},                                // shape
                    {.1, .2, .3, .4, .5, .6, .7, .8, .9},  // values
                    1.5,                                   // default_value
                    std::vector<std::vector<int>>(
                        {std::vector<int>({4}),
                         std::vector<int>({0, 0, 0, 2, 2, 2, 2, 3, 3})}));
  EXPECT_THAT(model.GetOutputShape(), testing::ElementsAreArray({4, 4}));
  EXPECT_THAT(model.GetOutputFloat(),
              testing::ElementsAreArray({.1, .2, .3, 1.5, 1.5, 1.5, 1.5, 1.5,
                                         .4, .5, .6, .7, .8, .9, 1.5, 1.5}));
}

TEST(RaggedTensorToTensorTest, RaggedTensorToTensorRowSplits) {
  // indices = [2, 1, 0, 3]
  // params = [[.1, .2, .3], [], [.4, .5, .6, .7], [.8, .9]]
  RaggedTensorToTensorOpModel model(2,      // output_shape_dims
                                    {9},    // values_shape
                                    {{5}},  // partition_tensors_shapes
                                    std::vector<std::string>({"ROW_SPLITS"}));
  model.InvokeFloat(
      {4, 4},                                // shape
      {.1, .2, .3, .4, .5, .6, .7, .8, .9},  // values
      1.5,                                   // default_value
      std::vector<std::vector<int>>({std::vector<int>({0, 3, 3, 7, 9})}));
  EXPECT_THAT(model.GetOutputShape(), testing::ElementsAreArray({4, 4}));
  EXPECT_THAT(model.GetOutputFloat(),
              testing::ElementsAreArray({.1, .2, .3, 1.5, 1.5, 1.5, 1.5, 1.5,
                                         .4, .5, .6, .7, .8, .9, 1.5, 1.5}));
}

TEST(RaggedTensorToTensorTest, RaggedTensorToTensor_3DParams) {
  // params = [
  //           [[]],
  //           [[.1, .2], [.3]],
  //           [],
  //           [[.4, .5], [.6, .7, .8]],
  //           [[.9]]
  //          ]
  RaggedTensorToTensorOpModel model(
      3,                // output_shape_dims
      {9},              // values_shape
      {{1}, {6}, {9}},  // partition_tensors_shapes
      std::vector<std::string>(
          {"FIRST_DIM_SIZE", "VALUE_ROWIDS", "VALUE_ROWIDS"}));
  model.InvokeFloat(
      {5, 2, 3},                             // shape
      {.1, .2, .3, .4, .5, .6, .7, .8, .9},  // values
      1.5,                                   // default_value
      std::vector<std::vector<int>>(
          {std::vector<int>({5}), std::vector<int>({0, 1, 1, 3, 3, 4}),
           std::vector<int>({1, 1, 2, 3, 3, 4, 4, 4, 5})}));

  EXPECT_THAT(model.GetOutputShape(), testing::ElementsAreArray({5, 2, 3}));
  EXPECT_THAT(model.GetOutputFloat(),
              testing::ElementsAreArray({1.5, 1.5, 1.5, 1.5, 1.5, 1.5, .1,  .2,
                                         1.5, .3,  1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
                                         1.5, 1.5, .4,  .5,  1.5, .6,  .7,  .8,
                                         .9,  1.5, 1.5, 1.5, 1.5, 1.5}));
}

TEST(RaggedTensorToTensorOpTest, RaggedTensorToTensor_3DParamsRowSplits) {
  // params = [
  //           [[]],
  //           [[.1, .2], [.3]],
  //           [],
  //           [[.4, .5], [.6, .7, .8]],
  //           [[.9]]
  //          ]
  RaggedTensorToTensorOpModel model(
      3,           // output_shape_dims
      {9},         // values_shape
      {{6}, {7}},  // partition_tensors_shapes
      std::vector<std::string>({"ROW_SPLITS", "ROW_SPLITS"}));
  model.InvokeFloat(
      {5, 2, 3},                             // shape
      {.1, .2, .3, .4, .5, .6, .7, .8, .9},  // values
      1.5,                                   // default_value
      std::vector<std::vector<int>>({std::vector<int>({0, 1, 3, 3, 5, 6}),
                                     std::vector<int>({0, 0, 2, 3, 5, 8, 9})}));
  EXPECT_THAT(model.GetOutputShape(), testing::ElementsAreArray({5, 2, 3}));
  EXPECT_THAT(model.GetOutputFloat(),
              testing::ElementsAreArray({1.5, 1.5, 1.5, 1.5, 1.5, 1.5, .1,  .2,
                                         1.5, .3,  1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
                                         1.5, 1.5, .4,  .5,  1.5, .6,  .7,  .8,
                                         .9,  1.5, 1.5, 1.5, 1.5, 1.5}));
}

TEST(RaggedTensorToTensorTest, RaggedTensorToTensor_3DParamsRowSplits2) {
  // params = [
  //           [[0, 1, 2], []],
  //           [],
  //           [[3]]
  //          ]

  RaggedTensorToTensorOpModel model(
      3,           // output_shape_dims
      {4},         // values_shape
      {{4}, {4}},  // partition_tensors_shapes
      std::vector<std::string>({"ROW_SPLITS", "ROW_SPLITS"}), TensorType_INT32);
  model.InvokeInt(
      {3, 2, 3},     // shape
      {0, 1, 2, 3},  // values
      5,             // default_value
      std::vector<std::vector<int>>(
          {std::vector<int>({0, 2, 2, 3}), std::vector<int>({0, 3, 3, 4})}));

  EXPECT_THAT(model.GetOutputShape(), testing::ElementsAreArray({3, 2, 3}));

  EXPECT_THAT(model.GetOutputInt(),
              testing::ElementsAreArray(
                  {0, 1, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 5, 5, 5, 5, 5}));
}

TEST(RaggedTensorToTensorTest, RaggedTensorToTensorContractExpanded) {
  // params = [[.1, .2, .3], [], [.4, .5, .6, .7], [.8, .9]]
  RaggedTensorToTensorOpModel model(
      2,           // output_shape_dims
      {9},         // values_shape
      {{1}, {9}},  // partition_tensors_shapes
      std::vector<std::string>({"FIRST_DIM_SIZE", "VALUE_ROWIDS"}));
  model.InvokeFloat({3, 5},                                // shape
                    {.1, .2, .3, .4, .5, .6, .7, .8, .9},  // values
                    1.5,                                   // default_value
                    std::vector<std::vector<int>>(
                        {std::vector<int>({4}),
                         std::vector<int>({0, 0, 0, 2, 2, 2, 2, 3, 3})}));
  EXPECT_THAT(model.GetOutputShape(), testing::ElementsAreArray({3, 5}));

  EXPECT_THAT(model.GetOutputFloat(),
              testing::ElementsAreArray({.1, .2, .3, 1.5, 1.5,     //
                                         1.5, 1.5, 1.5, 1.5, 1.5,  //
                                         .4, .5, .6, .7, 1.5}));
}

// Adds a dense dimension.
TEST(RaggedTensorToTensorTest, RaggedTensorToTensorContractExpandedDense) {
  // params = [[.1, .2, .3], [], [.4, .5, .6, .7], [.8, .9]]
  RaggedTensorToTensorOpModel model(
      3,           // output_shape_dims
      {9, 2},      // values_shape
      {{1}, {9}},  // partition_tensors_shapes
      std::vector<std::string>({"FIRST_DIM_SIZE", "VALUE_ROWIDS"}));

  model.InvokeFloat({3, 5, 2},  // shape
                    {.1, 1.1, .2, 1.2, .3, 1.3, .4, 1.4, .5, 1.5, .6, 1.6, .7,
                     1.7, .8, 1.8, .9, 1.9},  // values
                    1.5,                      // default_value
                    std::vector<std::vector<int>>(
                        {std::vector<int>({4}),
                         std::vector<int>({0, 0, 0, 2, 2, 2, 2, 3, 3})}));

  EXPECT_THAT(model.GetOutputShape(), testing::ElementsAreArray({3, 5, 2}));
  EXPECT_THAT(model.GetOutputFloat(),
              testing::ElementsAreArray(
                  {.1,  1.1, .2,  1.2, .3,  1.3, 1.5, 1.5, 1.5, 1.5,  //
                   1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,  //
                   .4,  1.4, .5,  1.5, .6,  1.6, .7,  1.7, 1.5, 1.5}));
}

TEST(RaggedTensorToTensorTest, StringType) {
  RaggedTensorToTensorOpModel model(
      2,           // output_shape_dims
      {9},         // values_shape
      {{1}, {9}},  // partition_tensors_shapes
      std::vector<std::string>({"FIRST_DIM_SIZE", "VALUE_ROWIDS"}),
      TensorType_STRING, TensorType_INT32, /*allocate_and_delegate=*/false);
  EXPECT_EQ(model.TryAllocateTensors(), kTfLiteError);
}

}  // namespace
}  // namespace tflite
