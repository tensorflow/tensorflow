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
bool VerifyStringTensorBuffer(const Buffer& buffer,
                              ErrorReporter* error_reporter) {
  uint32_t buffer_size = buffer.data()->size();
  const char* buffer_ptr = reinterpret_cast<const char*>(buffer.data()->data());

  uint32_t num_strings = *GetIntPtr(buffer_ptr);
  if (num_strings > kMaxNumString) {
    ReportError(error_reporter,
                "String tensor has invalid num of string set: %d", num_strings);
    return false;
  }
  uint32_t header_offsets =
      static_cast<uint32_t>(num_strings + 2) * sizeof(int32_t);

  if (buffer_size < header_offsets) {
    ReportError(error_reporter,
                "String tensor buffer requires at least %d bytes, but is "
                "allocated with %d bytes",
                header_offsets, buffer_size);
    return false;
  }

  uint32_t prev_ptr = header_offsets;
  uint32_t offset = sizeof(int32_t);

  if (*GetIntPtr(buffer_ptr + offset) != header_offsets) {
    ReportError(error_reporter,
                "String tensor buffer initial offset must be: %d",
                header_offsets);
    return false;
  }
  offset += sizeof(int32_t);
  for (int i = 1; i <= num_strings; i++, offset += sizeof(int32_t)) {
    int string_offset = *GetIntPtr(buffer_ptr + offset);
    if (string_offset < prev_ptr || string_offset > buffer_size) {
      ReportError(error_reporter, "String tensor buffer is invalid: index %d",
                  i);
      return false;
    }
  }
  if (*GetIntPtr(buffer_ptr + offset - sizeof(int32_t)) != buffer_size) {
    ReportError(error_reporter, "String tensor buffer last offset must be %d",
                buffer_size);
    return false;
  }
  return true;
}

// Verifies numeric tensor has legit buffer.
bool VerifyNumericTensorBuffer(const Tensor& tensor, const Buffer& buffer,
                               ErrorReporter* error_reporter) {
  uint64_t bytes_required = 1;
  for (int dim : *tensor.shape()) {
    bytes_required *= dim;
    if (bytes_required > UINT_MAX) {
      ReportError(error_reporter, "Tensor dimension overflow");
      return false;
    }
  }
  switch (tensor.type()) {
    case TensorType_FLOAT32:
      bytes_required *= sizeof(float);
      break;
    case TensorType_INT32:
      bytes_required *= sizeof(int32_t);
      break;
    case TensorType_UINT8:
      bytes_required *= sizeof(uint8_t);
      break;
    case TensorType_INT64:
      bytes_required *= sizeof(int64_t);
      break;
    case TensorType_FLOAT16:
      // FALLTHROUGH_INTENDED;
    default:
      ReportError(error_reporter, "Invalid tensor type: %d", tensor.type());
      return false;
  }
  if (bytes_required > UINT_MAX) {
    ReportError(error_reporter, "Tensor dimension overflow");
    return false;
  }

  if (bytes_required != buffer.data()->size()) {
    ReportError(
        error_reporter,
        "Tensor requires %d bytes, but is allocated with %d bytes buffer",
        bytes_required, buffer.data()->size());
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
        ReportError(error_reporter, "Invalid tensor buffer index: %d",
                    tensor->buffer());
        return false;
      }
      auto* buffer = model.buffers()->Get(tensor->buffer());
      if (!buffer) {
        ReportError(error_reporter, "Tensor buffer %d not set",
                    tensor->buffer());
        return false;
      }

      // Many transient tensors don't have data in the flatbuffer. Their
      // buffers will be allocated by the interpreter at run-time.
      if (buffer->data()) {
        if (tensor->type() == TensorType_STRING) {
          if (!VerifyStringTensorBuffer(*buffer, error_reporter)) {
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
