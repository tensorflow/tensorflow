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

#include "tensorflow/lite/delegates/gpu/common/testing/feature_parity/generators/add.h"

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/testing/feature_parity/utils.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace {

class AddModelBuilder {
 public:
  std::vector<uint8_t> Build() {
    flatbuffers::FlatBufferBuilder builder;
    flatbuffers::Offset<OperatorCode> operator_code =
        CreateOperatorCode(builder, BuiltinOperator_ADD, 0);

    flatbuffers::Offset<AddOptions> add_options =
        CreateAddOptions(builder, ActivationFunctionType_NONE);

    flatbuffers::Offset<Buffer> buffers[1] = {
        CreateBuffer(builder, builder.CreateVector({})),
    };
    std::vector<flatbuffers::Offset<Tensor>> tensors;
    for (int8_t i = 0; i < input_shapes_.size(); i++) {
      tensors.push_back(CreateTensor(
          builder, builder.CreateVector<int32_t>(input_shapes_[i].data(), 4),
          TensorType_FLOAT32,
          /*buffer=*/0, builder.CreateString(std::to_string(i))));
    }
    tensors.push_back(CreateTensor(
        builder, builder.CreateVector<int32_t>(output_shape_.data(), 4),
        TensorType_FLOAT32,
        /*buffer=*/0,
        builder.CreateString(std::to_string(input_shapes_.size()))));

    const int32_t op_inputs[2] = {0, 1};
    const int32_t op_outputs[1] = {2};

    flatbuffers::Offset<Operator> op =
        CreateOperator(builder, /*opcode_index=*/0,
                       builder.CreateVector<int32_t>(op_inputs, 2),
                       builder.CreateVector<int32_t>(op_outputs, 1),
                       BuiltinOptions_AddOptions, add_options.Union());

    int32_t subgraph_inputs[2] = {0, 1};
    int32_t subgraph_outputs[1] = {2};
    flatbuffers::Offset<SubGraph> subgraph =
        CreateSubGraph(builder, builder.CreateVector(&tensors[0], 3),
                       builder.CreateVector<int32_t>(subgraph_inputs, 2),
                       builder.CreateVector<int32_t>(subgraph_outputs, 1),
                       builder.CreateVector(&op, 1));

    flatbuffers::Offset<flatbuffers::String> description =
        builder.CreateString("Add model");
    flatbuffers::Offset<Model> model_buffer = CreateModel(
        builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&operator_code, 1),
        builder.CreateVector(&subgraph, 1), description,
        builder.CreateVector(buffers, 1));

    builder.Finish(model_buffer);
    return std::vector<uint8_t>(builder.GetBufferPointer(),
                                builder.GetBufferPointer() + builder.GetSize());
  }

  void SetInputShape(uint32_t input, std::vector<int32_t>&& shape) {
    if (input_shapes_.size() <= input) {
      input_shapes_.resize(input + 1);
    }

    input_shapes_[input] = std::move(shape);
  }

  void SetOutputShape(std::vector<int32_t>&& shape) {
    output_shape_ = std::move(shape);
  }

 private:
  std::vector<std::vector<int32_t>> input_shapes_;
  std::vector<int32_t> output_shape_;
};
}  // namespace

TestParams Add2SameShapeTensors() {
  AddModelBuilder builder;
  builder.SetInputShape(0, {1, 2, 2, 2});
  builder.SetInputShape(1, {1, 2, 2, 2});
  builder.SetOutputShape({1, 2, 2, 2});
  return {"Add2SameShapeTensors", builder.Build()};
}

TestParams AddBroadcast() {
  AddModelBuilder builder;
  builder.SetInputShape(0, {1, 2, 2, 2});
  builder.SetInputShape(1, {1, 1, 1, 2});
  builder.SetOutputShape({1, 2, 2, 2});
  return {"AddBroadcast", builder.Build()};
}

}  // namespace tflite
