/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

// The following tests make sure the quantize operation for unpack has the
// correct quantization parameters
namespace {

class UnpackQuantizeTest : public ::testing::Test {
 protected:
  UnpackQuantizeTest() {}

  // Prepare a hypothetical TOCO model with one unpack operator in it
  // together with 2 arrays as its outputs.

  // Since we are testing quantization in action, we are going to have all
  // inputs as kFloat. Outputs are also kFloat. This will force the
  // transformation operation to
  // 1. calculate min and max of the input.
  // 2. insert dequantization nodes after quantized outputs of Unpack operation.
  void PrepareModel(Model* model, int axis) {
    std::vector<string> unpack_output_names = {"unpack_out0", "unpack_out1"};
    model->flags.add_output_arrays(unpack_output_names[0]);
    model->flags.add_output_arrays(unpack_output_names[1]);
    const string unpack_input_name("unpack_op_input");

    const int kDim = 2;
    const int kElementPerDim = 2;
    const int kBufSize = 4;
    static float in_buf[kBufSize] = {0.0, 1.0, 2.0, 3.0};

    // Input arrays is going to be in kFloat since in this case quantization
    // transformation will be forced to calculate min and max of the input.
    Array& in_array = model->GetOrCreateArray(unpack_input_name);
    in_array.data_type = ArrayDataType::kFloat;

    // Initialize shape for the input array.
    Shape* in_array_shape = in_array.mutable_shape();
    std::vector<int>* in_array_shape_dim = in_array_shape->mutable_dims();
    for (int i = 0; i < kDim; i++) {
      in_array_shape_dim->push_back(kElementPerDim);
    }
    auto& in_array_buffer =
        in_array.GetMutableBuffer<toco::ArrayDataType::kFloat>();
    in_array_buffer.data.resize(kBufSize);
    auto* buf_ptr = in_array_buffer.data.data();
    std::copy(in_buf, in_buf + kBufSize, buf_ptr);

    auto* unpack_op = new UnpackOperator;
    unpack_op->axis = axis;
    unpack_op->inputs = {unpack_input_name};
    unpack_op->outputs = unpack_output_names;

    // Configuring the necessary outputs. The outputs also happen to be in
    // kFloat. This is because during quantization transformation data types for
    // these arrays are going to be forced to be kUint8.
    for (const string& unpack_output_name : unpack_output_names) {
      Array& out_array = model->GetOrCreateArray(unpack_output_name);
      out_array.GetOrCreateMinMax();
      out_array.data_type = ArrayDataType::kFloat;
      out_array.GetMutableBuffer<ArrayDataType::kFloat>().data.resize(
          kElementPerDim);

      Shape* out_array_shape = out_array.mutable_shape();
      std::vector<int>* out_array_shape_dim = out_array_shape->mutable_dims();
      out_array_shape_dim->resize(kDim - 1);
      for (int i = 0; i < kDim - 1; i++) {
        (*out_array_shape_dim)[i] = kElementPerDim;
      }
    }

    model->operators.push_back(std::unique_ptr<Operator>(unpack_op));
  }
};
}  // namespace

TEST_F(UnpackQuantizeTest, CheckUnpackPreservesQuantizationParameters) {
  using testing::ElementsAre;
  using testing::ElementsAreArray;
  Model model;
  const int axis = 0;
  PrepareModel(&model, axis);

  GraphTransformationsSet graph_transformation_set;
  graph_transformation_set.Add(new toco::Quantize);
  bool modified;
  ASSERT_TRUE((*graph_transformation_set.begin())
                  ->Run(&model, /*op_index=*/0, &modified)
                  .ok());

  const string output_name = model.flags.output_arrays(0);

  // Quantization transformation inserts NODE_NAME_DEQUANTIZE operations,
  // effectively making them the new outputs of the array. Old outputs of the
  // array are being fed into dequantization nodes. Furthermore, dequantize
  // nodes are being set as model outputs in model flags.  Therefore, we get the
  // following configuration OriginalInput->Unpack->OriginalOutputQuantized->
  // ->Dequantize. In fact we are interested in quantization parameters of
  // OriginalOutputQuantized array, hence using the original string constants
  // from the test fixture preparation code.
  const auto& unpack_input_array = model.GetArray("unpack_op_input");
  const auto& unpack_array0 = model.GetArray("unpack_out0");
  const auto& unpack_array1 = model.GetArray("unpack_out1");
  // Checking quantization params match, minmax match for array1
  EXPECT_THAT(unpack_input_array.quantization_params->zero_point,
              unpack_array0.quantization_params->zero_point);
  EXPECT_THAT(unpack_input_array.quantization_params->scale,
              unpack_array0.quantization_params->scale);
  EXPECT_THAT(unpack_input_array.minmax->min, unpack_array0.minmax->min);
  EXPECT_THAT(unpack_input_array.minmax->max, unpack_array0.minmax->max);

  // ...and for array2
  EXPECT_THAT(unpack_input_array.quantization_params->zero_point,
              unpack_array1.quantization_params->zero_point);
  EXPECT_THAT(unpack_input_array.quantization_params->scale,
              unpack_array1.quantization_params->scale);
  EXPECT_THAT(unpack_input_array.minmax->min, unpack_array1.minmax->min);
  EXPECT_THAT(unpack_input_array.minmax->max, unpack_array1.minmax->max);
}
}  // namespace toco
