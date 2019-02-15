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
#include "tensorflow/lite/experimental/writer/writer_lib.h"
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/experimental/writer/enum_mapping.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/reflection/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
template <class T>
using Offset = flatbuffers::Offset<T>;
template <class T>
using Vector = flatbuffers::Vector<T>;
using FlatBufferBuilder = flatbuffers::FlatBufferBuilder;

std::pair<BuiltinOptions, Offset<void>> CreateBuiltinUnion(
    FlatBufferBuilder* fbb, enum BuiltinOperator op, void* builtin_op_data) {
  switch (op) {
#include "tensorflow/lite/experimental/writer/option_writer_generated.h"
  }
  return std::make_pair(BuiltinOptions_NONE, Offset<void>());
}

template <class T_OUTPUT, class T_INPUT>
Offset<Vector<T_OUTPUT>> InterpreterWriter::ExportVector(FlatBufferBuilder* fbb,
                                                         const T_INPUT& v) {
  std::vector<T_OUTPUT> inputs(v.begin(), v.end());
  return fbb->template CreateVector<T_OUTPUT>(inputs);
}

Offset<Vector<Offset<Operator>>> InterpreterWriter::ExportOperators(
    FlatBufferBuilder* fbb) {
  std::vector<Offset<Operator>> operators;

  std::vector<int> operator_to_opcode;
  // TODO(aselle): Augment this once we put execution plan in schema.
  operator_to_opcode.resize(interpreter_->nodes_size(), -1);
  for (int op_index : interpreter_->execution_plan()) {
    const auto* node_and_registration =
        interpreter_->node_and_registration(op_index);
    const TfLiteRegistration* registration = &node_and_registration->second;
    if (!registration->custom_name) {
      operator_to_opcode[op_index] =
          GetOpCodeForBuiltin(registration->builtin_code);
    } else {
      operator_to_opcode[op_index] =
          GetOpCodeForCustom(registration->custom_name);
    }
  }
  // second pass serialize operators
  for (int op_index : interpreter_->execution_plan()) {
    const auto* node_and_registration =
        interpreter_->node_and_registration(op_index);
    const TfLiteNode& node = node_and_registration->first;
    const TfLiteRegistration& registration = node_and_registration->second;
    Offset<void> builtin_options;
    BuiltinOptions builtin_options_type = BuiltinOptions_NONE;
    // Custom data
    // TODO(aselle): Custom options format is not known by default. Just assume
    // for now.
    auto custom_options_format = CustomOptionsFormat_FLEXBUFFERS;
    Offset<Vector<uint8_t>> custom_options = 0;

    if (!registration.custom_name) {
      // builtin
      auto builtin_options_and_type = CreateBuiltinUnion(
          fbb, static_cast<enum BuiltinOperator>(registration.builtin_code),
          node.builtin_data);
      builtin_options = builtin_options_and_type.second;
      builtin_options_type = builtin_options_and_type.first;
    } else {
      auto custom_writer = custom_op_to_writer_.find(registration.custom_name);
      if (custom_writer != custom_op_to_writer_.end() &&
          custom_writer->second) {
        // delegate to custom writer if it exists
        custom_writer->second(fbb, interpreter_, op_index, &custom_options,
                              &custom_options_format);
      } else {
        // use the custom data as fact
        custom_options = fbb->CreateVector(
            reinterpret_cast<const uint8_t*>(node.custom_initial_data),
            node.custom_initial_data_size);
      }
    }

    int opcode_index = operator_to_opcode[op_index];
    std::vector<int> written_inputs =
        RemapTensorIndicesToWritten(TfLiteIntArrayView(node.inputs));
    std::vector<int> written_outputs =
        RemapTensorIndicesToWritten(TfLiteIntArrayView(node.outputs));
    auto inputs = ExportVector<int32_t>(fbb, written_inputs);
    auto outputs = ExportVector<int32_t>(fbb, written_outputs);
    operators.push_back(CreateOperator(*fbb, opcode_index, inputs, outputs,
                                       builtin_options_type, builtin_options,
                                       custom_options, custom_options_format));
  }

  return fbb->template CreateVector<Offset<Operator>>(operators);
}

Offset<Vector<Offset<Tensor>>> InterpreterWriter::ExportTensors(
    FlatBufferBuilder* fbb) {
  // Initialized to -1.
  // A value of -1 means this tensor will not be exported.
  tensor_to_written_tensor_.resize(interpreter_->tensors_size(), -1);

  std::vector<Offset<Tensor>> tensors;

  // Make a map from tensor index to whether the tensor is a temporary.
  std::vector<bool> tensor_is_temporary(interpreter_->tensors_size(), false);
  for (int op_index = 0; op_index < interpreter_->nodes_size(); ++op_index) {
    const auto* node_and_registration =
        interpreter_->node_and_registration(op_index);
    for (auto tensor_index :
         TfLiteIntArrayView(node_and_registration->first.temporaries))
      tensor_is_temporary[tensor_index] = true;
  }

  // Now we need to remap all used tensor indices
  int curr_output_index = 0;
  for (int tensor_index = 0; tensor_index < interpreter_->tensors_size();
       tensor_index++) {
    // Temporary tensors and unused tensors will not be written.
    if (!tensor_is_temporary[tensor_index] &&
        unused_tensors_.find(tensor_index) == unused_tensors_.end()) {
      tensor_to_written_tensor_[tensor_index] = curr_output_index++;
    }
  }

  for (int tensor_index = 0; tensor_index < interpreter_->tensors_size();
       ++tensor_index) {
    // Tensor not exported.
    if (tensor_to_written_tensor_[tensor_index] == -1) continue;

    if (TfLiteTensor* tensor = interpreter_->tensor(tensor_index)) {
      // We only need to convert non temporaries
      if (tensor->allocation_type != kTfLiteArenaRw &&
          tensor->allocation_type != kTfLiteMmapRo &&
          tensor->allocation_type != kTfLiteArenaRwPersistent)
        continue;
      // Allocate a buffer index
      int buffer_index = 0;  // This is null
      if (tensor->allocation_type == kTfLiteMmapRo) {
        buffer_index = buffers_.size();
        buffers_.push_back(std::make_pair(
            reinterpret_cast<const uint8_t*>(tensor->data.raw), tensor->bytes));
      }
      // Primitive type.
      TensorType type = TfLiteTypeToSchemaType(tensor->type);
      // Handle quantization
      const Offset<Vector<float>> null_array;
      Offset<Vector<float>> scale_array;
      Offset<Vector<int64_t>> zero_point_array;
      if (tensor->params.scale != 0.f) {
        // We have quantization, make a single arugment array (multi channel
        // quant needs updating here).
        scale_array = fbb->CreateVector<float>({tensor->params.scale});
        zero_point_array =
            fbb->CreateVector<int64_t>({tensor->params.zero_point});
      }
      Offset<QuantizationParameters> quantization_params =
          CreateQuantizationParameters(*fbb, null_array, null_array,
                                       scale_array, zero_point_array);
      // Shape
      TfLiteIntArrayView shape_view(tensor->dims);
      std::vector<int> shape =
          std::vector<int>(shape_view.begin(), shape_view.end());

      tensors.push_back(CreateTensor(*fbb, ExportVector<int32_t>(fbb, shape),
                                     type, buffer_index,
                                     fbb->CreateString(tensor->name),
                                     quantization_params, tensor->is_variable));
    }
  }
  return fbb->template CreateVector<Offset<Tensor>>(tensors);
}

Offset<Vector<Offset<Buffer>>> InterpreterWriter::ExportBuffers(
    FlatBufferBuilder* fbb) {
  std::vector<Offset<Buffer>> buffer_vector;
  for (auto buffer : buffers_) {
    auto data_offset = fbb->CreateVector(buffer.first, buffer.second);
    buffer_vector.push_back(CreateBuffer(*fbb, data_offset));
  }
  return fbb->template CreateVector<Offset<Buffer>>(buffer_vector);
}

Offset<Vector<Offset<OperatorCode>>> InterpreterWriter::CreateOpCodeTable(
    FlatBufferBuilder* fbb) {
  std::vector<Offset<OperatorCode>> codes;
  for (auto it : opcodes_) {
    const char* custom_name = it.custom.empty() ? nullptr : it.custom.c_str();
    codes.push_back(CreateOperatorCodeDirect(
        *fbb, static_cast<BuiltinOperator>(it.builtin), custom_name));
  }
  return fbb->template CreateVector<Offset<OperatorCode>>(codes);
}

template <class T>
std::vector<int> InterpreterWriter::RemapTensorIndicesToWritten(
    const T& input) {
  std::vector<int> output;
  output.reserve(input.size());
  for (int x : input) {
    // Special value representing an optional tensor which is not present.
    if (x == -1) {
      output.push_back(x);
      continue;
    }
    if (tensor_to_written_tensor_[x] != -1) {
      output.push_back(tensor_to_written_tensor_[x]);
    }
  }
  return output;
}

TfLiteStatus InterpreterWriter::GetBuffer(std::unique_ptr<uint8_t[]>* out,
                                          size_t* size) {
  if (!out || !size) return kTfLiteError;
  FlatBufferBuilder builder(/*initial_size=*/10240);

  std::vector<Offset<SubGraph>> subgraphs_as_vector;
  {  // subgraph specific stuff
    auto tensors = ExportTensors(&builder);
    std::vector<int> written_inputs =
        RemapTensorIndicesToWritten(interpreter_->inputs());
    std::vector<int> written_outputs =
        RemapTensorIndicesToWritten(interpreter_->outputs());
    auto inputs = ExportVector<int32_t>(&builder, written_inputs);
    auto outputs = ExportVector<int32_t>(&builder, written_outputs);

    auto ops = ExportOperators(&builder);
    subgraphs_as_vector.push_back(
        CreateSubGraph(builder, tensors, inputs, outputs, ops, /* name */ 0));
  }
  Offset<Vector<Offset<Buffer>>> buffers = ExportBuffers(&builder);

  auto description = builder.CreateString("Exported from Interpreter.");

  auto op_codes = CreateOpCodeTable(&builder);
  auto model = CreateModel(builder, TFLITE_SCHEMA_VERSION, op_codes,
                           builder.CreateVector(subgraphs_as_vector),
                           description, buffers);
  ::tflite::FinishModelBuffer(builder, model);
  const uint8_t* buffer = builder.GetBufferPointer();
  *size = builder.GetSize();
  (*out).reset(new uint8_t[*size]);
  memcpy(out->get(), buffer, *size);
  return kTfLiteOk;
}

TfLiteStatus InterpreterWriter::Write(const std::string& filename) {
  std::unique_ptr<uint8_t[]> buffer;
  size_t size;
  TF_LITE_ENSURE_STATUS(GetBuffer(&buffer, &size));

  FILE* fp = fopen(filename.c_str(), "wb");
  if (!fp) return kTfLiteError;

  if (fwrite(buffer.get(), 1, size, fp) != size) return kTfLiteError;
  if (fclose(fp)) return kTfLiteError;

  return kTfLiteOk;
}

TfLiteStatus InterpreterWriter::RegisterCustomWriter(
    const std::string& custom_name, CustomWriter custom_writer) {
  if (custom_op_to_writer_.find(custom_name) != custom_op_to_writer_.end()) {
    return kTfLiteError;
  }
  custom_op_to_writer_.insert(std::make_pair(custom_name, custom_writer));
  return kTfLiteOk;
}

}  // namespace tflite
