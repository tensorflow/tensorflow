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
#include "tensorflow/lite/tools/serialization/writer_lib.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/schema/reflection/schema_generated.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/tools/serialization/enum_mapping.h"
#include "tensorflow/lite/tools/versioning/op_version.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace {

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<OperatorCode>>>
CreateOpCodeTableImpl(flatbuffers::FlatBufferBuilder* fbb,
                      std::vector<OpCode>* opcodes) {
  std::vector<flatbuffers::Offset<OperatorCode>> codes;
  for (const auto& it : *opcodes) {
    const char* custom_name = it.custom.empty() ? nullptr : it.custom.c_str();
    // Use version 0 for builtin op. This is a way to serialize version field to
    // flatbuffer (since 0 is non default) and it will be corrected later.
    int32_t op_version = it.builtin != tflite::BuiltinOperator_CUSTOM ? 0 : 1;
    codes.push_back(
        CreateOperatorCodeDirect(*fbb, static_cast<BuiltinOperator>(it.builtin),
                                 custom_name, op_version));
  }
  return fbb->template CreateVector<flatbuffers::Offset<OperatorCode>>(codes);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Buffer>>>
ExportBuffersImpl(flatbuffers::FlatBufferBuilder* fbb,
                  std::vector<std::pair<const uint8_t*, size_t>>* buffers) {
  std::vector<flatbuffers::Offset<Buffer>> buffer_vector;
  for (auto buffer : *buffers) {
    auto data_offset = fbb->CreateVector(buffer.first, buffer.second);
    buffer_vector.push_back(CreateBuffer(*fbb, data_offset));
  }
  return fbb->template CreateVector<flatbuffers::Offset<Buffer>>(buffer_vector);
}

TfLiteStatus WriteImpl(const std::string& filename, void* data, size_t size) {
  FILE* fp = fopen(filename.c_str(), "wb");
  if (!fp) return kTfLiteError;

  const int result_size = fwrite(data, 1, size, fp);
  fclose(fp);
  if (result_size != size) return kTfLiteError;

  return kTfLiteOk;
}

std::pair<BuiltinOptions, flatbuffers::Offset<void>> CreateBuiltinUnion(
    flatbuffers::FlatBufferBuilder* fbb, enum BuiltinOperator op,
    void* builtin_op_data, const TfLiteNode& node) {
  switch (op) {
#include "tensorflow/lite/tools/serialization/option_writer_generated.h"
  }
  return std::make_pair(BuiltinOptions_NONE, flatbuffers::Offset<void>());
}

}  // namespace

template <class T_OUTPUT, class T_INPUT>
flatbuffers::Offset<flatbuffers::Vector<T_OUTPUT>> SubgraphWriter::ExportVector(
    flatbuffers::FlatBufferBuilder* fbb, const T_INPUT& v) {
  std::vector<T_OUTPUT> inputs(v.begin(), v.end());
  return fbb->template CreateVector<T_OUTPUT>(inputs);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Operator>>>
SubgraphWriter::ExportOperators(flatbuffers::FlatBufferBuilder* fbb) {
  std::vector<flatbuffers::Offset<Operator>> operators;

  std::vector<int> operator_to_opcode;
  // TODO(aselle): Augment this once we put execution plan in schema.
  operator_to_opcode.resize(subgraph_->nodes_size(), -1);
  for (int op_index : execution_plan_) {
    const auto* node_and_registration =
        subgraph_->node_and_registration(op_index);
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
  for (int op_index : execution_plan_) {
    const auto* node_and_registration =
        subgraph_->node_and_registration(op_index);
    const TfLiteNode& node = node_and_registration->first;
    const TfLiteRegistration& registration = node_and_registration->second;
    flatbuffers::Offset<void> builtin_options;
    BuiltinOptions builtin_options_type = BuiltinOptions_NONE;
    // Custom data
    // TODO(aselle): Custom options format is not known by default. Just assume
    // for now.
    auto custom_options_format = CustomOptionsFormat_FLEXBUFFERS;
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> custom_options = 0;

    if (!registration.custom_name) {
      // builtin
      auto builtin_options_and_type = CreateBuiltinUnion(
          fbb, static_cast<enum BuiltinOperator>(registration.builtin_code),
          node.builtin_data, node);
      builtin_options = builtin_options_and_type.second;
      builtin_options_type = builtin_options_and_type.first;
    } else {
      auto custom_writer = custom_op_to_writer_.find(registration.custom_name);
      if (custom_writer != custom_op_to_writer_.end() &&
          custom_writer->second) {
        // delegate to custom writer if it exists
        custom_writer->second(fbb, subgraph_, op_index, &custom_options,
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

  return fbb->template CreateVector<flatbuffers::Offset<Operator>>(operators);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Tensor>>>
SubgraphWriter::ExportTensors(flatbuffers::FlatBufferBuilder* fbb) {
  // Initialized to -1.
  // A value of -1 means this tensor will not be exported.
  tensor_to_written_tensor_.resize(subgraph_->tensors_size(), -1);

  std::vector<flatbuffers::Offset<Tensor>> tensors;

  // Make a map from tensor index to whether the tensor is a temporary.
  std::vector<bool> tensor_is_temporary(subgraph_->tensors_size(), false);
  for (int op_index = 0; op_index < subgraph_->nodes_size(); ++op_index) {
    const auto* node_and_registration =
        subgraph_->node_and_registration(op_index);
    for (auto tensor_index :
         TfLiteIntArrayView(node_and_registration->first.temporaries))
      tensor_is_temporary[tensor_index] = true;
  }

  // Now we need to remap all used tensor indices
  int curr_output_index = 0;
  for (int tensor_index = 0; tensor_index < subgraph_->tensors_size();
       tensor_index++) {
    // Temporary tensors and unused tensors will not be written.
    if (!tensor_is_temporary[tensor_index] &&
        unused_tensors_.find(tensor_index) == unused_tensors_.end()) {
      tensor_to_written_tensor_[tensor_index] = curr_output_index++;
    }
  }

  for (int tensor_index = 0; tensor_index < subgraph_->tensors_size();
       ++tensor_index) {
    // Tensor not exported.
    if (tensor_to_written_tensor_[tensor_index] == -1) continue;

    if (TfLiteTensor* tensor = subgraph_->tensor(tensor_index)) {
      // Allocate a buffer index
      int buffer_index = 0;  // This is null
      if (tensor->allocation_type == kTfLiteMmapRo) {
        buffer_index = buffers_->size();
        buffers_->push_back(std::make_pair(
            reinterpret_cast<const uint8_t*>(tensor->data.raw), tensor->bytes));
      }
      // Primitive type.
      TensorType type = TfLiteTypeToSchemaType(tensor->type);
      // Handle quantization
      flatbuffers::Offset<QuantizationParameters> quantization_params;

      const flatbuffers::Offset<flatbuffers::Vector<float>> null_array;
      flatbuffers::Offset<flatbuffers::Vector<float>> scale_array;
      flatbuffers::Offset<flatbuffers::Vector<int64_t>> zero_point_array;

      if (tensor->quantization.type == kTfLiteAffineQuantization) {
        if (tensor->params.scale != 0.f) {
          // Quantization with a single argument array.
          scale_array = fbb->CreateVector<float>({tensor->params.scale});
          zero_point_array =
              fbb->CreateVector<int64_t>({tensor->params.zero_point});
          quantization_params = CreateQuantizationParameters(
              *fbb, null_array, null_array, scale_array, zero_point_array);
        } else {  // Multi channel quantization.
          const TfLiteAffineQuantization* params =
              reinterpret_cast<TfLiteAffineQuantization*>(
                  tensor->quantization.params);
          const size_t num_scales = params->scale->size;

          std::vector<float> scale_vector(params->scale->data,
                                          params->scale->data + num_scales);
          std::vector<int64_t> zero_point_vector(
              params->zero_point->data, params->zero_point->data + num_scales);
          scale_array = fbb->CreateVector<float>(scale_vector);
          zero_point_array = fbb->CreateVector<int64_t>(zero_point_vector);
          quantization_params = CreateQuantizationParameters(
              *fbb, null_array, null_array, scale_array, zero_point_array,
              QuantizationDetails_NONE, 0, params->quantized_dimension);
        }
      }

      // Shape
      // Some tensors added during op init are not registered formally as
      // node temporaries. Some didn't get memory allocated for them, and we
      // should avoid serializing those tensors.
      if (tensor->dims) {
        TfLiteIntArrayView shape_view(tensor->dims);
        std::vector<int> shape =
            std::vector<int>(shape_view.begin(), shape_view.end());

        Offset<flatbuffers::String> tensor_name_offset = 0;
        if (tensor->name != nullptr) {
          tensor_name_offset = fbb->CreateString(tensor->name);
        }

        tensors.push_back(CreateTensor(
            *fbb, ExportVector<int32_t>(fbb, shape), type, buffer_index,
            tensor_name_offset, quantization_params, tensor->is_variable));
      }
    }
  }
  return fbb->template CreateVector<flatbuffers::Offset<Tensor>>(tensors);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Buffer>>>
SubgraphWriter::ExportBuffers(flatbuffers::FlatBufferBuilder* fbb) {
  return ExportBuffersImpl(fbb, buffers_);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<OperatorCode>>>
SubgraphWriter::CreateOpCodeTable(flatbuffers::FlatBufferBuilder* fbb) {
  return CreateOpCodeTableImpl(fbb, opcodes_);
}

template <class T>
std::vector<int> SubgraphWriter::RemapTensorIndicesToWritten(const T& input) {
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

TfLiteStatus SubgraphWriter::GetBuffer(std::unique_ptr<uint8_t[]>* out,
                                       size_t* size) {
  if (!out || !size) return kTfLiteError;
  flatbuffers::FlatBufferBuilder builder(/*initial_size=*/10240);
  std::vector<flatbuffers::Offset<SubGraph>> subgraphs_as_vector;
  subgraphs_as_vector.push_back(
      PopulateAndGetOffset(&builder, subgraph_->GetName()));

  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Buffer>>>
      buffers = ExportBuffers(&builder);

  auto description = builder.CreateString("Exported from Subgraph.");

  auto op_codes = CreateOpCodeTable(&builder);
  auto model = CreateModel(builder, TFLITE_SCHEMA_VERSION, op_codes,
                           builder.CreateVector(subgraphs_as_vector),
                           description, buffers);
  ::tflite::FinishModelBuffer(builder, model);
  ::tflite::UpdateOpVersion(builder.GetBufferPointer());
  const uint8_t* buffer = builder.GetBufferPointer();
  *size = builder.GetSize();
  (*out).reset(new uint8_t[*size]);
  memcpy(out->get(), buffer, *size);
  return kTfLiteOk;
}

flatbuffers::Offset<SubGraph> SubgraphWriter::PopulateAndGetOffset(
    flatbuffers::FlatBufferBuilder* builder, const std::string& subgraph_name) {
  auto tensors = ExportTensors(builder);
  std::vector<int> written_inputs = RemapTensorIndicesToWritten(inputs_);
  std::vector<int> written_outputs = RemapTensorIndicesToWritten(outputs_);
  auto inputs = ExportVector<int32_t>(builder, written_inputs);
  auto outputs = ExportVector<int32_t>(builder, written_outputs);

  auto ops = ExportOperators(builder);
  auto name = builder->CreateString(subgraph_name);
  return CreateSubGraph(*builder, tensors, inputs, outputs, ops, name);
}

TfLiteStatus SubgraphWriter::Write(const std::string& filename) {
  std::unique_ptr<uint8_t[]> buffer;
  size_t size;
  TF_LITE_ENSURE_STATUS(GetBuffer(&buffer, &size));
  return WriteImpl(filename, buffer.get(), size);
}

TfLiteStatus SubgraphWriter::RegisterCustomWriter(
    const std::string& custom_name, CustomWriter custom_writer) {
  if (custom_op_to_writer_.find(custom_name) != custom_op_to_writer_.end()) {
    return kTfLiteError;
  }
  custom_op_to_writer_.insert(std::make_pair(custom_name, custom_writer));
  return kTfLiteOk;
}

TfLiteStatus SubgraphWriter::CheckInputOutput(
    const std::vector<int>& inputs, const std::vector<int>& outputs,
    const std::vector<int>& execution_plan) {
  std::unordered_set<int> known_tensors(inputs.begin(), inputs.end());
  known_tensors.insert(subgraph_->variables().begin(),
                       subgraph_->variables().end());
  // Scan execution plan and confirm input tensors are known before each node
  // executes. Then append output tensors to known tensors.
  for (int op_index : execution_plan) {
    const auto* node_and_registration =
        subgraph_->node_and_registration(op_index);
    const TfLiteNode& node = node_and_registration->first;
    for (int tensor_index : TfLiteIntArrayView(node.inputs)) {
      if (tensor_index < 0) {
        // Skip if optional input not present.
        if (tensor_index == kTfLiteOptionalTensor) {
          continue;
        } else {
          return kTfLiteError;
        }
      }
      if (TfLiteTensor* tensor = subgraph_->tensor(tensor_index)) {
        // Skip constant tensors.
        if (tensor->allocation_type == kTfLiteMmapRo) {
          continue;
        }
      }

      if (known_tensors.find(tensor_index) == known_tensors.end()) {
        subgraph_->context()->ReportError(
            subgraph_->context(),
            "Node (%d) uses an input (%d) that is not provided.", op_index,
            tensor_index);
        return kTfLiteError;
      }
    }
    TfLiteIntArrayView outputs(node.outputs);
    known_tensors.insert(outputs.begin(), outputs.end());
  }

  // Check if outputs are known tensors or constants.
  for (int tensor_index : outputs) {
    if (TfLiteTensor* tensor = subgraph_->tensor(tensor_index)) {
      // Skip constant tensors.
      if (tensor->allocation_type == kTfLiteMmapRo) {
        continue;
      }
    }

    if (known_tensors.find(tensor_index) == known_tensors.end()) {
      subgraph_->context()->ReportError(
          subgraph_->context(),
          "Output (%d) is not produced by the execution plan.", tensor_index);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus SubgraphWriter::SetCustomInputOutput(
    const std::vector<int>& inputs, const std::vector<int>& outputs,
    const std::vector<int>& execution_plan) {
  TF_LITE_ENSURE_STATUS(CheckInputOutput(inputs, outputs, execution_plan));
  inputs_ = inputs;
  outputs_ = outputs;
  execution_plan_ = execution_plan;
  return kTfLiteOk;
}

ModelWriter::ModelWriter(Interpreter* interpreter) {
  std::vector<Subgraph*> subgraphs;

  // Retrieves the list of the subgraphs from the interpreter for constructing
  // a list of SubgraphWriters.
  subgraphs.reserve(interpreter->subgraphs_size());
  for (int i = 0; i < interpreter->subgraphs_size(); ++i) {
    subgraphs.push_back(interpreter->subgraph(i));
  }

  Init(subgraphs);
}

ModelWriter::ModelWriter(const std::vector<Subgraph*>& subgraphs) {
  Init(subgraphs);
}

void ModelWriter::Init(const std::vector<Subgraph*>& subgraphs) {
  buffers_.push_back(std::make_pair(nullptr, 0));
  subgraph_writers_.reserve(subgraphs.size());
  for (auto* subgraph : subgraphs) {
    SubgraphWriter writer(subgraph, &buffers_, &opcodes_,
                          &builtin_op_to_opcode_);
    subgraph_writers_.push_back(writer);
  }
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Buffer>>>
ModelWriter::ExportBuffers(flatbuffers::FlatBufferBuilder* fbb) {
  return ExportBuffersImpl(fbb, &buffers_);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<OperatorCode>>>
ModelWriter::CreateOpCodeTable(flatbuffers::FlatBufferBuilder* fbb) {
  return CreateOpCodeTableImpl(fbb, &opcodes_);
}

TfLiteStatus ModelWriter::GetBuffer(std::unique_ptr<uint8_t[]>* out,
                                    size_t* size) {
  if (!out || !size) return kTfLiteError;
  flatbuffers::FlatBufferBuilder builder(/*initial_size=*/10240);

  std::vector<flatbuffers::Offset<SubGraph>> subgraphs_as_vector;
  subgraphs_as_vector.reserve(subgraph_writers_.size());
  for (auto& subgraph_writer : subgraph_writers_) {
    subgraphs_as_vector.push_back(subgraph_writer.PopulateAndGetOffset(
        &builder, subgraph_writer.subgraph_->GetName()));
  }

  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Buffer>>>
      buffers = ExportBuffers(&builder);

  auto description = builder.CreateString("Exported from Subgraph.");

  auto op_codes = CreateOpCodeTable(&builder);
  auto model = CreateModel(builder, TFLITE_SCHEMA_VERSION, op_codes,
                           builder.CreateVector(subgraphs_as_vector),
                           description, buffers);
  ::tflite::FinishModelBuffer(builder, model);
  ::tflite::UpdateOpVersion(builder.GetBufferPointer());
  const uint8_t* buffer = builder.GetBufferPointer();
  *size = builder.GetSize();
  (*out).reset(new uint8_t[*size]);
  memcpy(out->get(), buffer, *size);
  return kTfLiteOk;
}

TfLiteStatus ModelWriter::Write(const std::string& filename) {
  std::unique_ptr<uint8_t[]> buffer;
  size_t size;
  TF_LITE_ENSURE_STATUS(GetBuffer(&buffer, &size));
  return WriteImpl(filename, buffer.get(), size);
}

void ModelWriter::SetUnusedTensors(int subgraph_index,
                                   const std::set<int>& unused_tensors) {
  subgraph_writers_[subgraph_index].SetUnusedTensors(unused_tensors);
}

TfLiteStatus ModelWriter::SetCustomInputOutput(
    int subgraph_index, const std::vector<int>& inputs,
    const std::vector<int>& outputs, const std::vector<int>& execution_plan) {
  return subgraph_writers_[subgraph_index].SetCustomInputOutput(inputs, outputs,
                                                                execution_plan);
}

TfLiteStatus ModelWriter::RegisterCustomWriter(const std::string& custom_name,
                                               CustomWriter custom_writer) {
  for (auto& subgraph_writer : subgraph_writers_) {
    subgraph_writer.RegisterCustomWriter(custom_name, custom_writer);
  }
  return kTfLiteOk;
}

}  // namespace tflite
