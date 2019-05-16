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

#include "tensorflow/lite/tools/verifier.h"
#include <climits>
#include "absl/container/flat_hash_set.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/version.h"

namespace tflite {

namespace {

// Reports error message when the reporter is set.
void ReportError(ErrorReporter* error_reporter, const char* format, ...) {
  if (error_reporter) {
    va_list args;
    va_start(args, format);
    error_reporter->Report(format, args);
    va_end(args);
  }
}

// Returns the int32_t value pointed by ptr.
const uint32_t* GetIntPtr(const char* ptr) {
  return reinterpret_cast<const uint32_t*>(ptr);
}

// Verifies flatbuffer format of the model contents and returns the in-memory
// model.
const Model* VerifyFlatbufferAndGetModel(const void* buf, size_t len) {
  ::flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buf), len);
  if (VerifyModelBuffer(verifier)) {
    return ::tflite::GetModel(buf);
  } else {
    return nullptr;
  }
}

const uint32_t kMaxNumString = UINT_MAX / sizeof(int32_t) - 2;

// Verifies string tensor has legit buffer contents that follow the schema
// defined in lite/string_util.h
bool VerifyStringTensorBuffer(const Tensor& tensor, const Buffer& buffer,
                              ErrorReporter* error_reporter) {
  uint32_t buffer_size = buffer.data()->size();
  if (buffer_size < sizeof(uint32_t)) {
    ReportError(error_reporter, "String tensor %s is invalid (empty)",
                tensor.name()->c_str());
    return false;
  }
  const char* buffer_ptr = reinterpret_cast<const char*>(buffer.data()->data());

  uint32_t num_strings = *GetIntPtr(buffer_ptr);
  if (num_strings > kMaxNumString) {
    ReportError(error_reporter,
                "String tensor %s has invalid num of string set: %d",
                tensor.name()->c_str(), num_strings);
    return false;
  }
  uint32_t header_offsets =
      static_cast<uint32_t>(num_strings + 2) * sizeof(int32_t);

  if (buffer_size < header_offsets) {
    ReportError(error_reporter,
                "String tensor %s buffer requires at least %d bytes, but is "
                "allocated with %d bytes",
                tensor.name()->c_str(), header_offsets, buffer_size);
    return false;
  }

  uint32_t prev_ptr = header_offsets;
  uint32_t offset = sizeof(int32_t);

  if (*GetIntPtr(buffer_ptr + offset) != header_offsets) {
    ReportError(error_reporter,
                "String tensor %s buffer initial offset must be: %d",
                tensor.name()->c_str(), header_offsets);
    return false;
  }
  offset += sizeof(int32_t);
  for (int i = 1; i <= num_strings; i++, offset += sizeof(int32_t)) {
    int string_offset = *GetIntPtr(buffer_ptr + offset);
    if (string_offset < prev_ptr || string_offset > buffer_size) {
      ReportError(error_reporter,
                  "String tensor %s buffer is invalid: index %d",
                  tensor.name()->c_str(), i);
      return false;
    }
  }
  if (*GetIntPtr(buffer_ptr + offset - sizeof(int32_t)) != buffer_size) {
    ReportError(error_reporter,
                "String tensor %s buffer last offset must be %d",
                tensor.name()->c_str(), buffer_size);
    return false;
  }
  return true;
}

// Verifies numeric tensor has legit buffer.
bool VerifyNumericTensorBuffer(const Tensor& tensor, const Buffer& buffer,
                               ErrorReporter* error_reporter) {
  uint64_t bytes_required = 1;
  if (!tensor.shape()) {
    // Empty tensor. Avoid further checks.
    return true;
  }
  for (int dim : *tensor.shape()) {
    bytes_required *= dim;
    if (bytes_required > UINT_MAX) {
      ReportError(error_reporter, "Tensor %s dimension overflow",
                  tensor.name()->c_str());
      return false;
    }
  }
  switch (tensor.type()) {
    case TensorType_FLOAT32:
      bytes_required *= sizeof(float);
      break;
    case TensorType_INT8:
      bytes_required *= sizeof(int8_t);
      break;
    case TensorType_UINT8:
      bytes_required *= sizeof(uint8_t);
      break;
    case TensorType_INT32:
      bytes_required *= sizeof(int32_t);
      break;
    case TensorType_INT64:
      bytes_required *= sizeof(int64_t);
      break;
    case TensorType_FLOAT16:
      // FALLTHROUGH_INTENDED;
    default:
      ReportError(error_reporter, "Tensor %s invalid type: %d",
                  tensor.name()->c_str(), tensor.type());
      return false;
  }
  if (bytes_required > UINT_MAX) {
    ReportError(error_reporter, "Tensor %s dimension overflow",
                tensor.name()->c_str());
    return false;
  }

  if (bytes_required != buffer.data()->size()) {
    ReportError(
        error_reporter,
        "Tensor %s requires %d bytes, but is allocated with %d bytes buffer",
        tensor.name()->c_str(), bytes_required, buffer.data()->size());
    return false;
  }
  return true;

  // TODO(yichengfan): verify quantized tensors.
}

using flatbuffers::Offset;
using flatbuffers::Vector;

bool VerifyOperators(const Vector<Offset<Operator>>& operators,
                     ErrorReporter* error_reporter) {
  for (const auto& op : operators) {
    if (!op->inputs()) {
      ReportError(error_reporter, "Missing 'inputs' for operator.");
      return false;
    }
    if (!op->outputs()) {
      ReportError(error_reporter, "Missing 'outputs' for operator.");
      return false;
    }
  }
  return true;
}

bool IsConstantTensor(const Tensor& tensor, const Model& model) {
  if (!tensor.buffer() || !model.buffers()) return false;
  if (tensor.buffer() > 0 && tensor.buffer() < model.buffers()->size()) {
    auto* buffer = model.buffers()->Get(tensor.buffer());
    if (buffer && buffer->data()) {
      return true;
    }
  }
  return false;
}

// Performs basic consistency checks on a sub-graph.
bool VerifySubGraphConsistency(const Model& model, const SubGraph& subgraph,
                               ErrorReporter* error_reporter) {
  absl::flat_hash_set<int> subgraph_input_tensors, constant_tensors,
      variable_tensors, output_tensors;
  if (subgraph.tensors()) {
    for (int i = 0; i < subgraph.tensors()->Length(); ++i) {
      const auto* tensor = subgraph.tensors()->Get(i);
      if (IsConstantTensor(*tensor, model)) {
        constant_tensors.insert(i);
      } else if (tensor->is_variable()) {
        variable_tensors.insert(i);
      }
    }
  }
  if (subgraph.inputs()) {
    for (const int tensor_idx : *subgraph.inputs()) {
      subgraph_input_tensors.insert(tensor_idx);
    }
  }

  if (subgraph.operators()) {
    for (int op_idx = 0; op_idx < subgraph.operators()->Length(); ++op_idx) {
      const auto* op = subgraph.operators()->Get(op_idx);
      if (!model.operator_codes() ||
          (op->opcode_index() >= model.operator_codes()->size())) {
        ReportError(error_reporter,
                    "Operator %d does not exist in model op codes",
                    op->opcode_index());
        return false;
      }
      const auto& opcode = model.operator_codes()->Get(op->opcode_index());
      // Check for invalid inputs by ensuring all exist in produced_tensors.
      for (const int input_idx : *op->inputs()) {
        if (input_idx == kOptionalTensor) continue;
        if (constant_tensors.find(input_idx) == constant_tensors.end() &&
            variable_tensors.find(input_idx) == variable_tensors.end() &&
            subgraph_input_tensors.find(input_idx) ==
                subgraph_input_tensors.end() &&
            output_tensors.find(input_idx) == output_tensors.end()) {
          ReportError(error_reporter,
                      "Input tensor %d to op %d (%s) is not produced",
                      input_idx, op_idx,
                      EnumNameBuiltinOperator(opcode->builtin_code()));
          return false;
        }
      }
      // Check for cycles/invalid outputs by ensuring that none exist in
      // produced_tensors.
      for (const int output_idx : *op->outputs()) {
        if (constant_tensors.find(output_idx) != constant_tensors.end()) {
          ReportError(error_reporter,
                      "Output tensor %d to op %d (%s) is a constant",
                      output_idx, op_idx,
                      EnumNameBuiltinOperator(opcode->builtin_code()));
          return false;
        } else if (variable_tensors.find(output_idx) !=
                   variable_tensors.end()) {
          ReportError(error_reporter,
                      "Output tensor %d to op %d (%s) is a variable",
                      output_idx, op_idx,
                      EnumNameBuiltinOperator(opcode->builtin_code()));
          return false;
        } else if (subgraph_input_tensors.find(output_idx) !=
                   subgraph_input_tensors.end()) {
          ReportError(error_reporter,
                      "Output tensor %d to op %d (%s) is a subgraph input",
                      output_idx, op_idx,
                      EnumNameBuiltinOperator(opcode->builtin_code()));
          return false;
        } else if (output_tensors.find(output_idx) != output_tensors.end()) {
          ReportError(error_reporter,
                      "Output tensor %d to op %d (%s) is an output from "
                      "another op. There is a cycle in the graph",
                      output_idx, op_idx,
                      EnumNameBuiltinOperator(opcode->builtin_code()));
          return false;
        }
        // This can be an input to a subsequent op.
        output_tensors.insert(output_idx);
      }
    }
  }
  return true;
}

bool VerifySubGraphs(const Model& model, ErrorReporter* error_reporter) {
  if (!model.subgraphs()) {
    ReportError(error_reporter, "Missing 'subgraphs' section.");
    return false;
  }
  for (const auto& subgraph : *model.subgraphs()) {
    if (!subgraph->operators()) {
      ReportError(error_reporter, "Missing 'operators' section in subgraph.");
      return false;
    }

    if (!VerifyOperators(*subgraph->operators(), error_reporter)) {
      return false;
    }

    if (!VerifySubGraphConsistency(model, *subgraph, error_reporter)) {
      return false;
    }
  }
  return true;
}

// Verifies tensors have valid properties and legit buffer if set.
bool VerifyTensors(const Model& model, ErrorReporter* error_reporter) {
  if (!model.subgraphs()) {
    return true;
  }
  if (!model.buffers()) {
    ReportError(error_reporter, "Missing 'buffers' section.");
    return false;
  }

  for (const auto& subgraph : *model.subgraphs()) {
    if (!subgraph->tensors()) {
      continue;
    }
    for (const auto& tensor : *subgraph->tensors()) {
      if (!tensor->buffer()) {
        continue;
      }
      if (tensor->buffer() >= model.buffers()->size()) {
        ReportError(error_reporter, "Tensor %s invalid buffer index: %d",
                    tensor->name(), tensor->buffer());
        return false;
      }
      auto* buffer = model.buffers()->Get(tensor->buffer());
      if (!buffer) {
        ReportError(error_reporter, "Tensor %s buffer %d not set",
                    tensor->name(), tensor->buffer());
        return false;
      }

      // Many transient tensors don't have data in the flatbuffer. Their
      // buffers will be allocated by the interpreter at run-time.
      if (buffer->data()) {
        if (tensor->type() == TensorType_STRING) {
          if (!VerifyStringTensorBuffer(*tensor, *buffer, error_reporter)) {
            return false;
          }
        } else {
          if (!VerifyNumericTensorBuffer(*tensor, *buffer, error_reporter)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

bool VerifyOps(const Model& model, const OpResolver& resolver,
               ErrorReporter* error_reporter) {
  if (!model.operator_codes()) {
    return true;
  }
  for (const auto& opcode : *model.operator_codes()) {
    if (opcode->builtin_code() < BuiltinOperator_MIN ||
        opcode->builtin_code() > BuiltinOperator_MAX) {
      ReportError(error_reporter, "Operator id '%d' is out of range.",
                  opcode->builtin_code());
      return false;
    }

    if (opcode->builtin_code() == BuiltinOperator_CUSTOM) {
      if (!resolver.FindOp(opcode->custom_code()->c_str(), opcode->version())) {
        ReportError(error_reporter, "Unsupported custom op: %s, version: %d",
                    opcode->custom_code()->c_str(), opcode->version());
        return false;
      }
    } else {
      if (!resolver.FindOp(opcode->builtin_code(), opcode->version())) {
        ReportError(error_reporter, "Unsupported builtin op: %s, version: %d",
                    EnumNameBuiltinOperator(opcode->builtin_code()),
                    opcode->version());
        return false;
      }
    }
  }
  return true;
}

}  // namespace

bool Verify(const void* buf, size_t len, const OpResolver& resolver,
            ErrorReporter* error_reporter) {
  const Model* model = VerifyFlatbufferAndGetModel(buf, len);
  if (model == nullptr) {
    ReportError(error_reporter, "Invalid flatbuffer format");
    return false;
  }
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ReportError(error_reporter, "Invalid model version %d", model->version());
    return false;
  }
  if (!VerifySubGraphs(*model, error_reporter)) {
    return false;
  }
  if (!VerifyTensors(*model, error_reporter)) {
    return false;
  }
  if (!VerifyOps(*model, resolver, error_reporter)) {
    return false;
  }
  return true;
}
}  // namespace tflite
