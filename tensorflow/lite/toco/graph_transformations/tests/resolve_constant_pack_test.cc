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
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"

namespace toco {

namespace {
std::vector<testing::Matcher<float>> ArrayFloatNear(
    const std::vector<float>& values, float max_abs_error = 1e-5) {
  std::vector<testing::Matcher<float>> matchers;
  matchers.reserve(values.size());
  for (const float& v : values) {
    matchers.emplace_back(testing::FloatNear(v, max_abs_error));
  }
  return matchers;
}
}  // namespace

class ResolveConstantPackTest : public ::testing::Test {
 protected:
  ResolveConstantPackTest() = default;

  void PrepareModel(Model* model, int axis, int output_dim0 = 2,
                    int input_buf_size = 2) {
    const std::string output_name("pack_op_output");
    model->flags.add_output_arrays(output_name);
    std::vector<std::string> pack_input_names = {"array0", "array1"};

    const int kBufSize = 2;
    static const float in_buf[2][2] = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    int cnt = 0;
    for (const std::string& pack_input_name : pack_input_names) {
      Array& in_array = model->GetOrCreateArray(pack_input_name);
      in_array.data_type = ArrayDataType::kFloat;

      Shape* in_array_shape = in_array.mutable_shape();
      std::vector<int>* in_array_shape_dim = in_array_shape->mutable_dims();
      in_array_shape_dim->push_back(kBufSize);

      auto& in_array_buffer =
          in_array.GetMutableBuffer<toco::ArrayDataType::kFloat>();
      in_array_buffer.data.resize(input_buf_size);
      float* buf_ptr = in_array_buffer.data.data();
      std::copy(in_buf[cnt], in_buf[cnt] + std::min(input_buf_size, kBufSize),
                buf_ptr);
      cnt++;
    }

    auto pack_op = std::make_unique<PackOperator>();
    pack_op->axis = axis;
    pack_op->inputs = pack_input_names;
    pack_op->outputs = {output_name};

    Array& out_array = model->GetOrCreateArray(pack_op->outputs[0]);
    out_array.data_type = ArrayDataType::kFloat;
    Shape* out_array_shape = out_array.mutable_shape();
    std::vector<int>* out_array_shape_dim = out_array_shape->mutable_dims();
    out_array_shape_dim->push_back(output_dim0);
    out_array_shape_dim->push_back(kBufSize);

    model->operators.push_back(std::move(pack_op));
  }
};

TEST_F(ResolveConstantPackTest, PackAtAxis0) {
  Model model;
  PrepareModel(&model, /*axis=*/0);

  GraphTransformationsSet graph_transformation_set;
  graph_transformation_set.Add(new toco::ResolveConstantPack);
  EXPECT_THAT(model.GetArrayMap().size(), 3);
  bool modified;
  ASSERT_TRUE((*graph_transformation_set.begin())
                  ->Run(&model, /*op_index=*/0, &modified)
                  .ok());
  EXPECT_THAT(model.GetArrayMap().size(), 1);

  const auto& packed_array = model.GetArray(model.flags.output_arrays(0));
  EXPECT_THAT(packed_array.GetBuffer<toco::ArrayDataType::kFloat>().data,
              ElementsAreArray(ArrayFloatNear({1.0f, 2.0f, 3.0f, 4.0f})));
}

TEST_F(ResolveConstantPackTest, PackAtNegativeAxis) {
  Model model;
  // For a 1D input, output is 2D (rank 2). axis=-2 resolves to 0.
  PrepareModel(&model, /*axis=*/-2);

  GraphTransformationsSet graph_transformation_set;
  graph_transformation_set.Add(new toco::ResolveConstantPack);
  EXPECT_THAT(model.GetArrayMap().size(), 3);
  bool modified;
  ASSERT_TRUE((*graph_transformation_set.begin())
                  ->Run(&model, /*op_index=*/0, &modified)
                  .ok());
  EXPECT_THAT(model.GetArrayMap().size(), 1);

  const auto& packed_array = model.GetArray(model.flags.output_arrays(0));
  EXPECT_THAT(packed_array.GetBuffer<toco::ArrayDataType::kFloat>().data,
              ElementsAreArray(ArrayFloatNear({1.0f, 2.0f, 3.0f, 4.0f})));
}

TEST_F(ResolveConstantPackTest, DeathOnInputBufferTooSmall) {
  Model model;
  // Set input_buf_size=1, while shape expects 2.
  PrepareModel(&model, /*axis=*/0, /*output_dim0=*/2, /*input_buf_size=*/1);

  GraphTransformationsSet graph_transformation_set;
  graph_transformation_set.Add(new toco::ResolveConstantPack);
  bool modified;
  EXPECT_DEATH((*graph_transformation_set.begin())
                   ->Run(&model, /*op_index=*/0, &modified)
                   .IgnoreError(),
               "input_size");
}

TEST_F(ResolveConstantPackTest, DeathOnOutputBufferOverflow) {
  Model model;
  // Set output_dim0=1 (so output buffer size is 2), while inputs expect 4
  // elements.
  PrepareModel(&model, /*axis=*/0, /*output_dim0=*/1, /*input_buf_size=*/2);

  GraphTransformationsSet graph_transformation_set;
  graph_transformation_set.Add(new toco::ResolveConstantPack);
  bool modified;
  EXPECT_DEATH((*graph_transformation_set.begin())
                   ->Run(&model, /*op_index=*/0, &modified)
                   .IgnoreError(),
               "output_data\\.size");
}

}  // namespace toco
