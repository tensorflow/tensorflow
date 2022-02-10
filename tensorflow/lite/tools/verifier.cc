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

#include <algorithm>
#include <climits>
#include <complex>
#include <cstdint>
#include <cstring>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/verifier_internal.h"
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace {

const char* NameOrEmptyString(const flatbuffers::String* str) {
  if (str == nullptr || str->c_str() == nullptr) {
    return "";
  }
  return str->c_str();
}

bool IsNullOrEmptyString(const flatbuffers::String* str) {
  return strcmp(NameOrEmptyString(str), "") == 0;
}

void ReportError(ErrorReporter* error_reporter, const char* format, ...) {
  if (error_reporter) {
    va_list args;
    va_start(args, format);
    TF_LITE_REPORT_ERROR(error_reporter, format, args);
    va_end(args);
  }
}

// Returns the int32_t value pointed by ptr.
const uint32_t GetIntPtr(const char* ptr) {
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  return flatbuffers::EndianScalar(*reinterpret_cast<const uint32_t*>(ptr));
#else
  return *reinterpret_cast<const uint32_t*>(ptr);
#endif
}

const uint32_t kMaxNumString = UINT_MAX / sizeof(int32_t) - 2;

// Verifies string tensor has legit buffer contents that follow the schema
// defined in lite/string_util.h
bool VerifyStringTensorBuffer(const Tensor& tensor, const Buffer& buffer,
                              ErrorReporter* error_reporter) {
  uint32_t buffer_size = buffer.data()->size();
  if (buffer_size < sizeof(uint32_t)) {
    ReportError(error_reporter, "String tensor %s is invalid (empty)",
                NameOrEmptyString(tensor.name()));
    return false;
  }
  const char* buffer_ptr = reinterpret_cast<const char*>(buffer.data()->data());

  uint32_t num_strings = GetIntPtr(buffer_ptr);
  if (num_strings > kMaxNumString) {
    ReportError(error_reporter,
                "String tensor %s has invalid num of string set: %d",
                NameOrEmptyString(tensor.name()), num_strings);
    return false;
  }
  uint32_t header_offsets =
      static_cast<uint32_t>(num_strings + 2) * sizeof(int32_t);

  if (buffer_size < header_offsets) {
    ReportError(error_reporter,
                "String tensor %s buffer requires at least %d bytes, but is "
                "allocated with %d bytes",
                NameOrEmptyString(tensor.name()), header_offsets, buffer_size);
    return false;
  }

  uint32_t prev_ptr = header_offsets;
  uint32_t offset = sizeof(int32_t);

  if (GetIntPtr(buffer_ptr + offset) != header_offsets) {
    ReportError(error_reporter,
                "String tensor %s buffer initial offset must be: %d",
                NameOrEmptyString(tensor.name()), header_offsets);
    return false;
  }
  offset += sizeof(int32_t);
  for (int i = 1, end = num_strings; i <= end; i++, offset += sizeof(int32_t)) {
    int string_offset = GetIntPtr(buffer_ptr + offset);
    if (string_offset < static_cast<int>(prev_ptr) ||
        string_offset > static_cast<int>(buffer_size)) {
      ReportError(error_reporter,
                  "String tensor %s buffer is invalid: index %d",
                  NameOrEmptyString(tensor.name()), i);
      return false;
    }
  }
  if (GetIntPtr(buffer_ptr + offset - sizeof(int32_t)) != buffer_size) {
    ReportError(error_reporter,
                "String tensor %s buffer last offset must be %d",
                NameOrEmptyString(tensor.name()), buffer_size);
    return false;
  }
  return true;
}

bool CheckArraySegments(const DimensionMetadata* dim_metadata) {
  if (dim_metadata->array_segments() == nullptr) {
    return false;
  }
  switch (dim_metadata->array_segments_type()) {
    case SparseIndexVector_Int32Vector:
      return (dim_metadata->array_segments_as_Int32Vector()->values() !=
              nullptr);
    case SparseIndexVector_Uint16Vector:
      return (dim_metadata->array_segments_as_Uint16Vector()->values() !=
              nullptr);
    case SparseIndexVector_Uint8Vector:
      return (dim_metadata->array_segments_as_Uint8Vector()->values() !=
              nullptr);
    default:
      return false;
  }
}

int GetSizeOfSegments(const DimensionMetadata* dim_metadata) {
  switch (dim_metadata->array_segments_type()) {
    case SparseIndexVector_Int32Vector:
      return dim_metadata->array_segments_as_Int32Vector()->values()->size();
    case SparseIndexVector_Uint16Vector:
      return dim_metadata->array_segments_as_Uint16Vector()->values()->size();
    case SparseIndexVector_Uint8Vector:
      return dim_metadata->array_segments_as_Uint8Vector()->values()->size();
    default:
      return -1;
  }
}

int GetValueOfSegmentsAt(const DimensionMetadata* dim_metadata, const int i) {
  switch (dim_metadata->array_segments_type()) {
    case SparseIndexVector_Int32Vector:
      return static_cast<int>(
          dim_metadata->array_segments_as_Int32Vector()->values()->Get(i));
    case SparseIndexVector_Uint16Vector:
      return static_cast<int>(
          dim_metadata->array_segments_as_Uint16Vector()->values()->Get(i));
    case SparseIndexVector_Uint8Vector:
      return static_cast<int>(
          dim_metadata->array_segments_as_Uint8Vector()->values()->Get(i));
    default:
      return -1;
  }
}

bool CheckArrayIndices(const DimensionMetadata* dim_metadata) {
  if (dim_metadata->array_indices() == nullptr) {
    return false;
  }
  switch (dim_metadata->array_indices_type()) {
    case SparseIndexVector_Int32Vector:
      return (dim_metadata->array_indices_as_Int32Vector()->values() !=
              nullptr);
    case SparseIndexVector_Uint16Vector:
      return (dim_metadata->array_indices_as_Uint16Vector()->values() !=
              nullptr);
    case SparseIndexVector_Uint8Vector:
      return (dim_metadata->array_indices_as_Uint8Vector()->values() !=
              nullptr);
    default:
      return false;
  }
}

int GetSizeOfIndices(const DimensionMetadata* dim_metadata) {
  switch (dim_metadata->array_indices_type()) {
    case SparseIndexVector_Int32Vector:
      return dim_metadata->array_indices_as_Int32Vector()->values()->size();
    case SparseIndexVector_Uint16Vector:
      return dim_metadata->array_indices_as_Uint16Vector()->values()->size();
    case SparseIndexVector_Uint8Vector:
      return dim_metadata->array_indices_as_Uint8Vector()->values()->size();
    default:
      return -1;
  }
}

int GetValueOfIndicesAt(const DimensionMetadata* dim_metadata, const int i) {
  switch (dim_metadata->array_indices_type()) {
    case SparseIndexVector_Int32Vector:
      return static_cast<int>(
          dim_metadata->array_indices_as_Int32Vector()->values()->Get(i));
    case SparseIndexVector_Uint16Vector:
      return static_cast<int>(
          dim_metadata->array_indices_as_Uint16Vector()->values()->Get(i));
    case SparseIndexVector_Uint8Vector:
      return static_cast<int>(
          dim_metadata->array_indices_as_Uint8Vector()->values()->Get(i));
    default:
      return -1;
  }
  return -1;
}

// The sparsity parameter defines a tree structure to map each non-zero element
// stored in the flattened buffer back to its index in the conceptual dense
// tensor.
// Traverse the tree level by level, count total number of elements, and
// validate the sparsity parameters along the way.
absl::optional<uint64_t> VerifyAndCountElements(
    const SparsityParameters& sparsity, const std::vector<int>& dim_sizes) {
  const int total_level = sparsity.traversal_order()->size();
  uint64_t num_elements = 1;
  for (int i = 0; i < total_level; i++) {
    const int original_dim = sparsity.traversal_order()->Get(i);
    const auto* dim_metadata = sparsity.dim_metadata()->Get(i);
    if (dim_metadata->format() == DimensionType_DENSE) {
      if (dim_metadata->dense_size() != dim_sizes[original_dim]) {
        return absl::nullopt;
      }

      // Each index in a dense dimension is stored implicitly.
      num_elements *= dim_metadata->dense_size();
    } else {
      if (!CheckArraySegments(dim_metadata) ||
          !CheckArrayIndices(dim_metadata)) {
        return absl::nullopt;
      }

      int array_segments_size = GetSizeOfSegments(dim_metadata);
      int array_indices_size = GetSizeOfIndices(dim_metadata);

      for (int j = 0; j < array_segments_size - 1; j++) {
        if (GetValueOfSegmentsAt(dim_metadata, j) < 0 ||
            GetValueOfSegmentsAt(dim_metadata, j + 1) < 0 ||
            GetValueOfSegmentsAt(dim_metadata, j) >
                GetValueOfSegmentsAt(dim_metadata, j + 1)) {
          return absl::nullopt;
        }
      }

      if (static_cast<int>(num_elements) != array_segments_size - 1) {
        return absl::nullopt;
      }

      if (array_indices_size !=
          GetValueOfSegmentsAt(dim_metadata, array_segments_size - 1)) {
        return absl::nullopt;
      }

      for (int j = 0; j < array_indices_size; j++) {
        if (GetValueOfIndicesAt(dim_metadata, j) < 0 ||
            GetValueOfIndicesAt(dim_metadata, j) >= dim_sizes[original_dim]) {
          return absl::nullopt;
        }
      }

      // Need to reset num_elements when seeing a sparse dimension.
      num_elements = array_indices_size;
    }
  }

  return num_elements;
}

absl::optional<uint64_t> VerifyAndCountSparseElements(const Tensor& tensor) {
  const auto* sparsity = tensor.sparsity();
  if (sparsity->traversal_order() == nullptr ||
      sparsity->dim_metadata() == nullptr) {
    return absl::nullopt;
  }

  const int total_dims = sparsity->traversal_order()->size();
  const int original_rank = tensor.shape()->size();
  const int sparsity_dim_metadata_size = sparsity->dim_metadata()->size();
  if (total_dims < original_rank || sparsity_dim_metadata_size != total_dims) {
    return absl::nullopt;
  }

  const int block_rank = total_dims - original_rank;
  if (block_rank > 0) {
    if (sparsity->block_map() == nullptr) {
      return absl::nullopt;
    }
    const int sparse_rank = sparsity->block_map()->size();
    if (sparse_rank != block_rank) {
      return absl::nullopt;
    }
  }

  // For a n-dimensional tensor (d0, ..., dn-1) with k-dimensional block (dn,
  // ..., dn+k-1), the first n elements in the traversal order should be a
  // permutation of (d0, ..., dn-1), and the last k elements should be a
  // permutation of (dn, ..., dn+k-1).
  std::vector<int> traversal_order(total_dims);
  for (int i = 0; i < total_dims; i++) {
    traversal_order[i] = sparsity->traversal_order()->Get(i);
  }

  std::sort(traversal_order.begin(), traversal_order.begin() + original_rank);
  for (int i = 0; i < original_rank; i++) {
    if (traversal_order[i] != i) {
      return absl::nullopt;
    }
  }

  std::sort(traversal_order.begin() + original_rank, traversal_order.end());
  for (int i = original_rank; i < total_dims; i++) {
    if (traversal_order[i] != i) {
      return absl::nullopt;
    }
  }

  // For a n-dimensional tensor (d0, ..., dn-1) with k-dimensional block (dn,
  // ..., dn+k-1), the expanded_dim_sizes holds the size of each dimension in
  // the order of (d0, ..., dn-1, dn, ..., dn+k-1), not the traversal order.
  // For example, a 4x4 tensor with 2x2 block has expanded_dim_sizes = {2, 2, 2,
  // 2}.
  std::vector<int> expanded_dim_sizes;
  expanded_dim_sizes.resize(total_dims);
  // First go through the original tensor dimensions, populate their sizes.
  for (int i = 0; i < original_rank; i++) {
    expanded_dim_sizes[i] = tensor.shape()->Get(i);
  }
  // Then go through the block dimensions, and
  //   1. populate block dimension size.
  //   2. block_map[i] has the original dimension that block dimension i maps
  //   to. Divide the size of the original dimension by the size of the ith
  //   block dimension.
  for (int i = 0; i < block_rank; i++) {
    int original_block_dim =
        sparsity->traversal_order()->Get(i + original_rank);
    if (original_block_dim < 0 || original_block_dim >= total_dims) {
      return absl::nullopt;
    }
    int block_dim_size =
        sparsity->dim_metadata()->Get(i + original_rank)->dense_size();
    // If size is <= 0 we just return as it is invalid.
    if (block_dim_size <= 0) {
      return absl::nullopt;
    }

    expanded_dim_sizes[original_block_dim] = block_dim_size;

    int mapped_block_dim = sparsity->block_map()->Get(i);
    if (mapped_block_dim < 0 || mapped_block_dim >= total_dims) {
      return absl::nullopt;
    }
    expanded_dim_sizes[mapped_block_dim] /= block_dim_size;
  }

  return VerifyAndCountElements(*sparsity, expanded_dim_sizes);
}

// Verifies numeric tensor has legit buffer.
bool VerifyNumericTensorBuffer(const Tensor& tensor, const Buffer& buffer,
                               ErrorReporter* error_reporter) {
  uint64_t bytes_required = 1;
  if (!tensor.shape()) {
    // Empty tensor. Avoid further checks.
    return true;
  }
  if (tensor.sparsity() != nullptr) {
    const auto num_elements = VerifyAndCountSparseElements(tensor);
    if (!num_elements.has_value()) {
      ReportError(error_reporter, "Tensor %s has invalid sparsity parameters",
                  NameOrEmptyString(tensor.name()));
      return false;
    }
    bytes_required = num_elements.value();
    if (bytes_required > UINT_MAX) {
      ReportError(error_reporter, "Tensor %s dimension overflow",
                  NameOrEmptyString(tensor.name()));
      return false;
    }
  } else {
    for (int dim : *tensor.shape()) {
      bytes_required *= dim;
      if (bytes_required > UINT_MAX) {
        ReportError(error_reporter, "Tensor %s dimension overflow",
                    NameOrEmptyString(tensor.name()));
        return false;
      }
    }
  }

  switch (tensor.type()) {
    case TensorType_FLOAT32:
      bytes_required *= sizeof(float);
      break;
    case TensorType_FLOAT16:
      bytes_required *= sizeof(uint16_t);
      break;
    case TensorType_FLOAT64:
      bytes_required *= sizeof(double);
      break;
    case TensorType_INT32:
      bytes_required *= sizeof(int32_t);
      break;
    case TensorType_UINT32:
      bytes_required *= sizeof(uint32_t);
      break;
    case TensorType_UINT8:
      bytes_required *= sizeof(uint8_t);
      break;
    case TensorType_INT8:
      bytes_required *= sizeof(int8_t);
      break;
    case TensorType_INT64:
      bytes_required *= sizeof(int64_t);
      break;
    case TensorType_UINT64:
      bytes_required *= sizeof(uint64_t);
      break;
    case TensorType_BOOL:
      bytes_required *= sizeof(bool);
      break;
    case TensorType_INT16:
      bytes_required *= sizeof(uint16_t);
      break;
    case TensorType_UINT16:
      bytes_required *= sizeof(uint16_t);
      break;
    case TensorType_COMPLEX64:
      bytes_required *= sizeof(std::complex<float>);
      break;
    case TensorType_COMPLEX128:
      bytes_required *= sizeof(std::complex<double>);
      break;
    default:
      ReportError(error_reporter, "Tensor %s invalid type: %d",
                  NameOrEmptyString(tensor.name()), tensor.type());
      return false;
  }
  if (bytes_required > UINT_MAX) {
    ReportError(error_reporter, "Tensor %s dimension overflow",
                NameOrEmptyString(tensor.name()));
    return false;
  }

  if (bytes_required != buffer.data()->size()) {
    ReportError(
        error_reporter,
        "Tensor %s requires %d bytes, but is allocated with %d bytes buffer",
        NameOrEmptyString(tensor.name()), bytes_required,
        buffer.data()->size());
    return false;
  }
  return true;

  // TODO(yichengfan): verify quantized tensors.
}

using flatbuffers::Offset;
using flatbuffers::Vector;

bool VerifyOperators(const Vector<Offset<Operator>>& operators,
                     ErrorReporter* error_reporter) {
  for (const auto* op : operators) {
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
    for (int i = 0, end = subgraph.tensors()->size(); i < end; ++i) {
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
    for (int op_idx = 0, end = subgraph.operators()->size(); op_idx < end;
         ++op_idx) {
      const auto* op = subgraph.operators()->Get(op_idx);
      if (!model.operator_codes() ||
          (op->opcode_index() >= model.operator_codes()->size())) {
        ReportError(error_reporter,
                    "Operator %d does not exist in model op codes",
                    op->opcode_index());
        return false;
      }
      const auto& opcode = model.operator_codes()->Get(op->opcode_index());
      auto builtin_code = GetBuiltinCode(opcode);
      // Check for invalid inputs by ensuring all exist in produced_tensors.
      for (const int input_idx : *op->inputs()) {
        if (input_idx == kTfLiteOptionalTensor) continue;
        if (constant_tensors.find(input_idx) == constant_tensors.end() &&
            variable_tensors.find(input_idx) == variable_tensors.end() &&
            subgraph_input_tensors.find(input_idx) ==
                subgraph_input_tensors.end() &&
            output_tensors.find(input_idx) == output_tensors.end()) {
          ReportError(error_reporter,
                      "Input tensor %d to op %d (%s) is not produced",
                      input_idx, op_idx, EnumNameBuiltinOperator(builtin_code));
          return false;
        }
      }
      // Check for cycles/invalid outputs by ensuring that none exist in
      // produced_tensors.
      for (const int output_idx : *op->outputs()) {
        if (constant_tensors.find(output_idx) != constant_tensors.end()) {
          ReportError(
              error_reporter, "Output tensor %d to op %d (%s) is a constant",
              output_idx, op_idx, EnumNameBuiltinOperator(builtin_code));
          return false;
        } else if (variable_tensors.find(output_idx) !=
                   variable_tensors.end()) {
          ReportError(
              error_reporter, "Output tensor %d to op %d (%s) is a variable",
              output_idx, op_idx, EnumNameBuiltinOperator(builtin_code));
          return false;
        } else if (subgraph_input_tensors.find(output_idx) !=
                   subgraph_input_tensors.end()) {
          ReportError(error_reporter,
                      "Output tensor %d to op %d (%s) is a subgraph input",
                      output_idx, op_idx,
                      EnumNameBuiltinOperator(builtin_code));
          return false;
        } else if (output_tensors.find(output_idx) != output_tensors.end()) {
          ReportError(error_reporter,
                      "Output tensor %d to op %d (%s) is an output from "
                      "another op. There is a cycle in the graph",
                      output_idx, op_idx,
                      EnumNameBuiltinOperator(builtin_code));
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
  for (const auto* subgraph : *model.subgraphs()) {
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

  for (const auto* subgraph : *model.subgraphs()) {
    if (!subgraph->tensors()) {
      continue;
    }
    for (const auto* tensor : *subgraph->tensors()) {
      if (!tensor->buffer()) {
        continue;
      }
      if (tensor->buffer() >= model.buffers()->size()) {
        ReportError(error_reporter, "Tensor %s invalid buffer index: %d",
                    NameOrEmptyString(tensor->name()), tensor->buffer());
        return false;
      }
      auto* buffer = model.buffers()->Get(tensor->buffer());
      if (!buffer) {
        ReportError(error_reporter, "Tensor %s buffer %d not set",
                    NameOrEmptyString(tensor->name()), tensor->buffer());
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

  // Track whichs ops are used in only the validation subgraphs. Validation
  // subgraphs are allowed to contain custom ops that are not in the resolver,
  // as they will be run with a custom resolver.
  absl::flat_hash_set<int> regular_code_indices;
  absl::flat_hash_set<int> validation_code_indices;
  for (const auto* subgraph : *model.subgraphs()) {
    if (!subgraph->operators()) {
      continue;
    }
    if (subgraph->name() && IsValidationSubgraph(subgraph->name()->c_str())) {
      for (const auto& op : *(subgraph->operators())) {
        validation_code_indices.insert(op->opcode_index());
      }
    } else {
      for (const auto* op : *(subgraph->operators())) {
        regular_code_indices.insert(op->opcode_index());
      }
    }
  }
  for (int i = 0; i < model.operator_codes()->size(); i++) {
    const auto* opcode = model.operator_codes()->Get(i);
    auto builtin_code = GetBuiltinCode(opcode);
    if (builtin_code < BuiltinOperator_MIN ||
        builtin_code > BuiltinOperator_MAX) {
      ReportError(error_reporter, "Operator id '%d' is out of range.",
                  builtin_code);
      return false;
    }

    if (builtin_code == BuiltinOperator_CUSTOM) {
      if (IsNullOrEmptyString(opcode->custom_code())) {
        ReportError(error_reporter,
                    "Invalid custom op name, cannot be null/empty.");
        return false;
      } else if (!resolver.FindOp(opcode->custom_code()->c_str(),
                                  opcode->version())) {
        if (regular_code_indices.contains(i) ||
            !validation_code_indices.contains(i)) {
          ReportError(error_reporter, "Unsupported custom op: %s, version: %d",
                      opcode->custom_code()->c_str(), opcode->version());
          return false;
        }
      }
    } else {
      if (!resolver.FindOp(builtin_code, opcode->version())) {
        ReportError(error_reporter, "Unsupported builtin op: %s, version: %d",
                    EnumNameBuiltinOperator(builtin_code), opcode->version());
        return false;
      }
    }
  }
  return true;
}

bool VerifyModel(const Model* model, ErrorReporter* error_reporter) {
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
  return true;
}

}  // namespace

bool Verify(const void* buf, size_t len, ErrorReporter* error_reporter) {
  const Model* model = internal::VerifyFlatBufferAndGetModel(buf, len);
  return VerifyModel(model, error_reporter);
}

// Deprecated: see comments in header.
bool Verify(const void* buf, size_t len, const OpResolver& resolver,
            ErrorReporter* error_reporter) {
  const Model* model = internal::VerifyFlatBufferAndGetModel(buf, len);
  if (!VerifyModel(model, error_reporter)) {
    return false;
  }
  if (!VerifyOps(*model, resolver, error_reporter)) {
    return false;
  }
  return true;
}

}  // namespace tflite
