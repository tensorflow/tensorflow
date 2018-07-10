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
#include "tensorflow/contrib/lite/toco/tflite/export.h"

#include "flatbuffers/flexbuffers.h"
#include "absl/strings/str_join.h"
#include "tensorflow/contrib/lite/schema/schema_generated.h"
#include "tensorflow/contrib/lite/toco/tflite/operator.h"
#include "tensorflow/contrib/lite/toco/tflite/types.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/contrib/lite/version.h"

namespace toco {

namespace tflite {

using flatbuffers::FlatBufferBuilder;
using flatbuffers::Offset;
using flatbuffers::Vector;
using ::tflite::Buffer;
using ::tflite::BuiltinOperator;
using ::tflite::BuiltinOperator_CUSTOM;
using ::tflite::BuiltinOperator_MAX;
using ::tflite::BuiltinOperator_MIN;
using ::tflite::CreateBuffer;
using ::tflite::CreateModel;
using ::tflite::CreateOperator;
using ::tflite::CreateTensor;
using ::tflite::Operator;
using ::tflite::OperatorCode;
using ::tflite::SubGraph;
using ::tflite::Tensor;

namespace {

details::OperatorKey GetOperatorKey(
    const ::toco::Operator& op,
    const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type) {
  string custom_code;
  if (op.type == OperatorType::kUnsupported) {
    const TensorFlowUnsupportedOperator& unsupported_op =
        static_cast<const TensorFlowUnsupportedOperator&>(op);
    custom_code = unsupported_op.tensorflow_op;
  }
  int version = 1;
  if (ops_by_type.count(op.type) != 0) {
    version = ops_by_type.at(op.type)->GetVersion(op);
  }
  return details::OperatorKey(op.type, custom_code, version);
}

}  // Anonymous namespace.

namespace details {

void LoadTensorsMap(const Model& model, TensorsMap* tensors_map) {
  // First find a list of unique array names.
  std::set<string> names;
  for (const auto& array_pair : model.GetArrayMap()) {
    names.insert(array_pair.first);
  }

  // Now assign indices to them and fill in the map.
  int index = 0;
  for (const auto& name : names) {
    (*tensors_map)[name] = index;
    ++index;
  }
}

void LoadOperatorsMap(
    const Model& model, OperatorsMap* operators_map,
    const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type) {
  // First find a list of unique operator types.
  std::set<OperatorKey> keys;
  for (const auto& op : model.operators) {
    keys.insert(GetOperatorKey(*op, ops_by_type));
  }
  // Now assign indices to them and fill in the map.
  int index = 0;
  for (const auto& key : keys) {
    (*operators_map)[key] = index;
    ++index;
  }
}
}  // namespace details

Offset<Vector<Offset<Tensor>>> ExportTensors(
    const Model& model, const details::TensorsMap& tensors_map,
    FlatBufferBuilder* builder, std::vector<const Array*>* buffers_to_write,
    const std::set<int32_t>& variable_tensor_indices) {
  // In the end we will need to produce a vector sorted by the indices of the
  // tensors in the tensors_map.
  std::map<int, Offset<Tensor>> ordered_tensors;

  for (const auto& array_pair : model.GetArrayMap()) {
    const string& tensor_name = array_pair.first;
    const toco::Array& array = *array_pair.second;

    int buffer_index = buffers_to_write->size();
    auto type = DataType::Serialize(array.data_type);
    buffers_to_write->push_back(&array);

    std::vector<int> shape;
    if (array.has_shape()) {
      for (int d : array.shape().dims()) {
        shape.push_back(d);
      }
    }

    Offset<Vector<float>> min;
    Offset<Vector<float>> max;
    Offset<Vector<float>> scale;
    Offset<Vector<int64_t>> zero_point;
    if (array.minmax) {
      min = builder->CreateVector(
          std::vector<float>{static_cast<float>(array.minmax->min)});
      max = builder->CreateVector(
          std::vector<float>{static_cast<float>(array.minmax->max)});
    }
    if (array.quantization_params) {
      scale = builder->CreateVector(std::vector<float>{
          static_cast<float>(array.quantization_params->scale)});
      zero_point = builder->CreateVector(
          std::vector<int64_t>{array.quantization_params->zero_point});
    }
    auto q_param = ::tflite::CreateQuantizationParameters(*builder, min, max,
                                                          scale, zero_point);

    int index = tensors_map.at(tensor_name);
    bool is_variable =
        variable_tensor_indices.find(index) != variable_tensor_indices.end();
    ordered_tensors[index] =
        CreateTensor(*builder, builder->CreateVector(shape), type, buffer_index,
                     builder->CreateString(tensor_name), q_param, is_variable);
  }

  std::vector<Offset<Tensor>> tensor_vector;
  tensor_vector.reserve(ordered_tensors.size());
  for (const auto& tensor : ordered_tensors) {
    tensor_vector.push_back(tensor.second);
  }

  return builder->CreateVector(tensor_vector);
}

Offset<Vector<int32_t>> ExportInputTensors(
    const Model& model, const details::TensorsMap& tensors_map,
    FlatBufferBuilder* builder) {
  std::vector<int32_t> inputs;
  for (const auto& input : model.flags.input_arrays()) {
    inputs.push_back(tensors_map.at(input.name()));
  }
  return builder->CreateVector<int32_t>(inputs);
}

Offset<Vector<int32_t>> ExportOutputTensors(
    const Model& model, const details::TensorsMap& tensors_map,
    FlatBufferBuilder* builder) {
  std::vector<int32_t> outputs;
  for (const string& output : model.flags.output_arrays()) {
    outputs.push_back(tensors_map.at(output));
  }
  return builder->CreateVector<int32_t>(outputs);
}

Offset<Vector<Offset<OperatorCode>>> ExportOperatorCodes(
    const Model& model,
    const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type,
    const details::OperatorsMap& operators_map, FlatBufferBuilder* builder,
    std::set<string>* error_summary) {
  // Map from operator name to TF Lite enum value, for all builtins.
  std::map<string, BuiltinOperator> builtin_ops;
  for (int i = BuiltinOperator_MIN; i <= BuiltinOperator_MAX; ++i) {
    BuiltinOperator op = static_cast<BuiltinOperator>(i);
    string name = EnumNameBuiltinOperator(op);
    if (op != BuiltinOperator_CUSTOM && !name.empty()) {
      builtin_ops[name] = op;
    }
  }

  // We will need to produce a vector of codes in the same order as they
  // appear in the operators_map.
  std::map<int, Offset<OperatorCode>> ordered_opcodes;

  for (const auto& op : model.operators) {
    const details::OperatorKey operator_key = GetOperatorKey(*op, ops_by_type);
    int op_index = operators_map.at(operator_key);
    int op_version = operator_key.version;

    string name = HelpfulOperatorTypeName(*op);
    bool is_builtin = false;
    if (ops_by_type.count(op->type) != 0) {
      name = ops_by_type.at(op->type)->name();
      is_builtin = (builtin_ops.count(name) > 0);
    }

    if (is_builtin) {
      ordered_opcodes[op_index] =
          CreateOperatorCode(*builder, builtin_ops[name], 0, op_version);
    } else {
      // This could be a kUnsupported, in which case we should be
      // able to retrieve the original Tensorflow name from the OperatorKey, or
      // this could be a proper TOCO operator that is completely unknown to TF
      // Lite.
      if (!operator_key.custom_code.empty()) {
        name = operator_key.custom_code;
      }
      // Either way, this is an operator that is not supported by TF Lite,
      // so we output it as a custom op and add it to the error summary.
      if (error_summary) {
        error_summary->insert(name);
      }
      ordered_opcodes[op_index] =
          CreateOperatorCode(*builder, BuiltinOperator_CUSTOM,
                             builder->CreateString(name), op_version);
    }
  }

  std::vector<Offset<OperatorCode>> opcode_vector;
  opcode_vector.reserve(ordered_opcodes.size());
  for (const auto& opcode : ordered_opcodes) {
    opcode_vector.push_back(opcode.second);
  }

  return builder->CreateVector(opcode_vector);
}

Offset<Vector<Offset<Operator>>> ExportOperators(
    const Model& model,
    const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type,
    const details::OperatorsMap& operators_map,
    const details::TensorsMap& tensors_map, FlatBufferBuilder* builder,
    std::set<int32_t>* variable_tensor_indices) {
  variable_tensor_indices->clear();

  // The operators are in execution order, so we just follow tf.mini order.
  std::vector<Offset<Operator>> op_vector;
  for (const auto& op : model.operators) {
    std::vector<int32_t> inputs;
    for (const string& input : op->inputs) {
      // -1 is the ID for optional tensor in TFLite output
      int id = model.IsOptionalArray(input) ? -1 : tensors_map.at(input);
      inputs.push_back(id);
    }
    std::vector<int32_t> outputs;
    for (const string& output : op->outputs) {
      outputs.push_back(tensors_map.at(output));
    }

    int op_index = operators_map.at(GetOperatorKey(*op, ops_by_type));

    auto tflite_op_it = ops_by_type.find(op->type);
    BaseOperator* tflite_op = tflite_op_it == ops_by_type.end()
                                  ? nullptr
                                  : tflite_op_it->second.get();

    // This is a custom op unless we can find it in ops_by_type, and even then
    // it could be a custom op (such as kUnsupported).
    auto options = Options::Custom(0);

    std::vector<bool> mutating_input_variables;
    if (tflite_op) {
      options = tflite_op->Serialize(*op, builder);
      mutating_input_variables = tflite_op->GetMutatingInputVariables(*op);

      if (!mutating_input_variables.empty()) {
        for (int i = 0; i < op->inputs.size(); ++i) {
          if (!mutating_input_variables[i]) {
            continue;
          }
          int32_t variable_tensor_index = tensors_map.at(op->inputs[i]);
          variable_tensor_indices->insert(variable_tensor_index);
        }
      }
    }
    // The only supported CustomOptionFormat is FLEXBUFFERS now.
    op_vector.push_back(CreateOperator(
        *builder, op_index, builder->CreateVector(inputs),
        builder->CreateVector(outputs), options.type, options.builtin,
        options.custom, ::tflite::CustomOptionsFormat_FLEXBUFFERS,
        builder->CreateVector(mutating_input_variables)));
  }

  return builder->CreateVector(op_vector);
}

Offset<Vector<Offset<Buffer>>> ExportBuffers(
    const Model& model, const std::vector<const Array*>& buffers_to_write,
    FlatBufferBuilder* builder) {
  std::vector<Offset<Buffer>> buffer_vector;
  size_t index = 0;
  for (const Array* array_ptr : buffers_to_write) {
    const Array& array = *array_ptr;
    Offset<Vector<uint8_t>> data_buffer = DataBuffer::Serialize(array, builder);
    buffer_vector.push_back(CreateBuffer(*builder, data_buffer));
    index++;
  }
  return builder->CreateVector(buffer_vector);
}

void Export(const Model& model, bool allow_custom_ops,
            string* output_file_contents) {
  const auto ops_by_type = BuildOperatorByTypeMap();
  Export(model, allow_custom_ops, output_file_contents, ops_by_type);
}

void Export(
    const Model& model, bool allow_custom_ops, string* output_file_contents,
    const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type) {
  flatbuffers::FlatBufferBuilder builder(/*initial_size=*/10240);

  details::TensorsMap tensors_map;
  details::LoadTensorsMap(model, &tensors_map);

  details::OperatorsMap operators_map;
  details::LoadOperatorsMap(model, &operators_map, ops_by_type);

  std::vector<const Array*> buffers_to_write;
  Array empty_array;
  buffers_to_write.push_back(&empty_array);

  std::set<string> error_summary;
  auto op_codes = ExportOperatorCodes(model, ops_by_type, operators_map,
                                      &builder, &error_summary);

  const string fake_quant_operation_name = "FAKE_QUANT";

  if (error_summary.count(fake_quant_operation_name) != 0) {
    LOG(ERROR)
        << fake_quant_operation_name
        << " operation was not converted. If running quantized make sure you "
           "are passing --inference_type=QUANTIZED_UINT8 and values for "
           "--std_values and --mean_values.";
    // Remove the fake quant operation from the errors, since it shouldn't
    // be provided a custom implementation.
    error_summary.erase(fake_quant_operation_name);
  }
  if (!allow_custom_ops && !error_summary.empty()) {
    // Remove ExpandDims and ReorderAxes from unimplemented list unless they
    // compose the list. Both ops are removed during graph transformations.
    // However, if an op is unimplemented earlier in the model, the graph
    // transformation is unable to run because the output shape is not defined.
    // This causes unnecessary confusion during model conversion time.
    std::set<string> error_summary_final;
    for (const auto& op_type : error_summary) {
      if (op_type != "ReorderAxes" && op_type != "ExpandDims") {
        error_summary_final.insert(op_type);
      }
    }
    if (error_summary_final.empty()) {
      error_summary_final = error_summary;
    }

    LOG(QFATAL)
        << "Some of the operators in the model are not supported by "
           "the standard TensorFlow Lite runtime. If you have a custom "
           "implementation for them you can disable this error with "
           "--allow_custom_ops, or by setting allow_custom_ops=True "
           "when calling tf.contrib.lite.toco_convert(). Here is a list "
           "of operators for which  you will need custom implementations: "
        << absl::StrJoin(error_summary_final, ", ") << ".";
  }

  std::set<int32_t> variable_tensor_indices;
  auto ops = ExportOperators(model, ops_by_type, operators_map, tensors_map,
                             &builder, &variable_tensor_indices);

  auto tensors = ExportTensors(model, tensors_map, &builder, &buffers_to_write,
                               variable_tensor_indices);
  auto inputs = ExportInputTensors(model, tensors_map, &builder);
  auto outputs = ExportOutputTensors(model, tensors_map, &builder);

  // TODO(aselle): add support to toco for multiple subgraphs.
  auto subgraph = CreateSubGraph(builder, tensors, inputs, outputs, ops,
                                 /* name */ 0);
  std::vector<flatbuffers::Offset<SubGraph>> subgraphs = {subgraph};

  auto buffers = ExportBuffers(model, buffers_to_write, &builder);
  auto description = builder.CreateString("TOCO Converted.");
  auto new_model_location =
      CreateModel(builder, TFLITE_SCHEMA_VERSION, op_codes,
                  builder.CreateVector(subgraphs), description, buffers);
  ::tflite::FinishModelBuffer(builder, new_model_location);
  const uint8_t* buffer = builder.GetBufferPointer();
  int size = builder.GetSize();
  *output_file_contents = string(reinterpret_cast<const char*>(buffer), size);
}

}  // namespace tflite

}  // namespace toco
