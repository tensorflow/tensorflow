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
#include "tensorflow/contrib/lite/toco/tflite/import.h"

#include "flatbuffers/flexbuffers.h"
#include "tensorflow/contrib/lite/schema/schema_generated.h"
#include "tensorflow/contrib/lite/toco/tflite/operator.h"
#include "tensorflow/contrib/lite/toco/tflite/types.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"

namespace toco {

namespace tflite {

namespace details {
void LoadTensorsTable(const ::tflite::Model& input_model,
                      TensorsTable* tensors_table) {
  // TODO(aselle): add support to toco for multiple subgraphs.
  auto tensors = (*input_model.subgraphs())[0]->tensors();
  if (!tensors) return;
  for (const auto* tensor : *tensors) {
    tensors_table->push_back(tensor->name()->c_str());
  }
}

void LoadOperatorsTable(const ::tflite::Model& input_model,
                        OperatorsTable* operators_table) {
  auto opcodes = input_model.operator_codes();
  if (!opcodes) return;
  for (const auto* opcode : *opcodes) {
    if (opcode->builtin_code() != ::tflite::BuiltinOperator_CUSTOM) {
      operators_table->push_back(
          EnumNameBuiltinOperator(opcode->builtin_code()));
    } else {
      operators_table->push_back(opcode->custom_code()->c_str());
    }
  }
}
}  // namespace details

void ImportTensors(const ::tflite::Model& input_model, Model* model) {
  auto tensors = (*input_model.subgraphs())[0]->tensors();
  auto* buffers = input_model.buffers();
  // auto tensors = input_model.tensors();
  if (!tensors) return;
  for (const auto* input_tensor : *tensors) {
    Array& array = model->GetOrCreateArray(input_tensor->name()->c_str());
    array.data_type = DataType::Deserialize(input_tensor->type());
    int buffer_index = input_tensor->buffer();
    auto* buffer = buffers->Get(buffer_index);
    DataBuffer::Deserialize(*input_tensor, *buffer, &array);

    auto shape = input_tensor->shape();
    if (shape) {
      // If the shape is 0-dimensional, make sure to record it as such,
      // as oppose to leaving the array without a shape.
      array.mutable_shape()->mutable_dims()->clear();
      for (int i = 0; i < shape->Length(); ++i) {
        auto d = shape->Get(i);
        array.mutable_shape()->mutable_dims()->push_back(d);
      }
    }

    auto quantization = input_tensor->quantization();
    if (quantization) {
      // Note that tf.mini only supports a single quantization parameters for
      // the whole array.
      if (quantization->min() && quantization->max()) {
        CHECK_EQ(1, quantization->min()->Length());
        CHECK_EQ(1, quantization->max()->Length());
        MinMax& minmax = array.GetOrCreateMinMax();
        minmax.min = quantization->min()->Get(0);
        minmax.max = quantization->max()->Get(0);
      }
      if (quantization->scale() && quantization->zero_point()) {
        CHECK_EQ(1, quantization->scale()->Length());
        CHECK_EQ(1, quantization->zero_point()->Length());
        QuantizationParams& q = array.GetOrCreateQuantizationParams();
        q.scale = quantization->scale()->Get(0);
        q.zero_point = quantization->zero_point()->Get(0);
      }
    }
  }
}

void ImportOperators(
    const ::tflite::Model& input_model,
    const std::map<string, std::unique_ptr<BaseOperator>>& ops_by_name,
    const details::TensorsTable& tensors_table,
    const details::OperatorsTable& operators_table, Model* model) {
  // TODO(aselle): add support for multiple subgraphs.
  auto ops = (*input_model.subgraphs())[0]->operators();

  if (!ops) return;
  for (const auto* input_op : *ops) {
    int index = input_op->opcode_index();
    if (index < 0 || index > operators_table.size()) {
      LOG(FATAL) << "Index " << index << " must be between zero and "
                 << operators_table.size();
    }
    string opname = operators_table.at(index);
    if (ops_by_name.count(opname) == 0) {
      LOG(FATAL) << "Op '" << opname << "' not supported";
    }

    auto new_op = ops_by_name.at(opname)->Deserialize(
        input_op->builtin_options(), input_op->custom_options());
    model->operators.emplace_back(new_op.release());
    auto* op = model->operators.back().get();

    auto inputs = input_op->inputs();
    for (int i = 0; i < inputs->Length(); i++) {
      auto input_index = inputs->Get(i);
      // input_index == -1 indicates optional tensor.
      if (input_index != -1) {
        const string& input_name = tensors_table.at(input_index);
        op->inputs.push_back(input_name);
      } else {
        const string& tensor_name =
            toco::AvailableArrayName(*model, "OptionalTensor");
        model->CreateOptionalArray(tensor_name);
        op->inputs.push_back(tensor_name);
      }
    }
    auto outputs = input_op->outputs();
    for (int i = 0; i < outputs->Length(); i++) {
      auto output_index = outputs->Get(i);
      const string& output_name = tensors_table.at(output_index);
      op->outputs.push_back(output_name);
    }
  }
}

void ImportIOTensors(const ::tflite::Model& input_model,
                     const details::TensorsTable& tensors_table, Model* model) {
  auto inputs = (*input_model.subgraphs())[0]->inputs();
  if (inputs) {
    for (int input : *inputs) {
      const string& input_name = tensors_table.at(input);
      model->flags.add_input_arrays()->set_name(input_name);
    }
  }

  auto outputs = (*input_model.subgraphs())[0]->outputs();
  if (outputs) {
    for (int output : *outputs) {
      const string& output_name = tensors_table.at(output);
      model->flags.add_output_arrays(output_name);
    }
  }
}

std::unique_ptr<Model> Import(const ModelFlags& model_flags,
                              const string& input_file_contents) {
  const ::tflite::Model* input_model =
      ::tflite::GetModel(input_file_contents.data());

  // Full list of all known operators.
  const auto ops_by_name = BuildOperatorByNameMap();

  if (input_model->subgraphs()->size() != 1) {
    LOG(FATAL) << "# of subgraphs in tflite should be exactly 1 for now.";
  }
  std::unique_ptr<Model> model;
  model.reset(new Model);

  details::TensorsTable tensors_table;
  details::LoadTensorsTable(*input_model, &tensors_table);

  details::OperatorsTable operators_table;
  details::LoadOperatorsTable(*input_model, &operators_table);

  ImportTensors(*input_model, model.get());
  ImportOperators(*input_model, ops_by_name, tensors_table, operators_table,
                  model.get());
  ImportIOTensors(*input_model, tensors_table, model.get());

  return model;
}

}  // namespace tflite

}  // namespace toco
