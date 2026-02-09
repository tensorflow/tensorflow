// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

#include <cstdint>
#include <memory>

#include "flatbuffers/flexbuffers.h"
#include "tensorflow/core/util/ragged_to_dense_util_common.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace text {
namespace ragged_tensor_to_tensor {
namespace {

constexpr int kShapeInput = 0;
constexpr int kValuesInput = 1;
constexpr int kDefaultValueInput = 2;
constexpr int kFirstPartitionInputIndex = 3;

constexpr int kOutputTensor = 0;

constexpr char kRowPartitionTypesAttr[] = "row_partition_types";

// The following three functions are copied from
// .../tensorflow/lite/kernels/internal/tensor_ctypes.h
// This header is not available in tensorflow package when building.
template <typename T>
inline T* GetTensorData(TfLiteTensor* tensor) {
  return tensor != nullptr ? reinterpret_cast<T*>(tensor->data.raw) : nullptr;
}

template <typename T>
inline const T* GetTensorData(const TfLiteTensor* tensor) {
  return tensor != nullptr ? reinterpret_cast<const T*>(tensor->data.raw)
                           : nullptr;
}

inline RuntimeShape GetTensorShape(const TfLiteTensor* tensor) {
  if (tensor == nullptr) {
    return RuntimeShape();
  }

  TfLiteIntArray* dims = tensor->dims;
  const int dims_size = dims->size;
  const int32_t* dims_data = reinterpret_cast<const int32_t*>(dims->data);
  return RuntimeShape(dims_size, dims_data);
}

struct ConversionAttributes {
  std::vector<tensorflow::RowPartitionType> partition_types;
  int ragged_rank = 0;

  tensorflow::RowPartitionType GetRowPartitionTypeByDimension(
      int dimension) const {
    if (partition_types.front() ==
        tensorflow::RowPartitionType::FIRST_DIM_SIZE) {
      return partition_types[dimension + 1];
    } else {
      return partition_types[dimension];
    }
  }
};
template <typename INDEX_TYPE>
int GetFirstDimensionSizeT(TfLiteContext* context,
                           const TfLiteTensor& first_partition_input,
                           const ConversionAttributes* attributes) {
  const tensorflow::RowPartitionType first_partition_type =
      attributes->partition_types.front();
  switch (first_partition_type) {
    case tensorflow::RowPartitionType::FIRST_DIM_SIZE:
      return *GetTensorData<INDEX_TYPE>(&first_partition_input);
    case tensorflow::RowPartitionType::VALUE_ROWIDS:
      context->ReportError(context,
                           "Cannot handle VALUE_ROWIDS in first dimension.");
      return -1;
    case tensorflow::RowPartitionType::ROW_SPLITS: {
      const auto shape = GetTensorShape(&first_partition_input);
      return shape.Dims(0) - 1;
    }

    default:
      context->ReportError(
          context, "Cannot handle type ",
          RowPartitionTypeToString(first_partition_type).c_str());
      return -1;
  }
}

int GetFirstDimensionSize(TfLiteContext* context,
                          const TfLiteTensor& first_partition_input,
                          const ConversionAttributes* attributes) {
  switch (first_partition_input.type) {
    case kTfLiteInt32:
      return GetFirstDimensionSizeT<int32_t>(context, first_partition_input,
                                             attributes);
    case kTfLiteInt64:
      return GetFirstDimensionSizeT<int64_t>(context, first_partition_input,
                                             attributes);
    default:
      context->ReportError(context,
                           "Not supported row partitioning tensor type");
      return -1;
  }
}

bool ValidateDefaultValueShape(TfLiteContext* context,
                               const RuntimeShape& default_value_shape,
                               const RuntimeShape& /*value_shape*/) {
  // TF implementation also checks that shapes are not defined, not needed in
  // TFLite.
  // TODO(mgubin): Only scalar default value sizes are supported.
  if (default_value_shape.FlatSize() != 1) {
    context->ReportError(context, "Only scalar default value is supported");
    return false;
  }
  return true;
}

RuntimeShape TensorShapeFromTensor(const TfLiteTensor& tensor) {
  // TODO(mgubin): No checks, see
  // third_party/tensorflow/core/kernels/list_kernels.cc
  const RuntimeShape tensor_shape(tensor.dims->size, tensor.dims->data);
  if (0 == tensor.dims->size) {
    // If the input tensor is scalar then the shape is empty (also scalar).
    return RuntimeShape{};
  }
  RuntimeShape result(tensor_shape.FlatSize());
  switch (tensor.type) {
    case kTfLiteInt32: {
      for (int i = 0; i < tensor_shape.FlatSize(); ++i) {
        result.SetDim(i, GetTensorData<int32_t>(&tensor)[i]);
      }
    } break;
    case kTfLiteInt64: {
      for (int i = 0; i < tensor_shape.FlatSize(); ++i) {
        result.SetDim(i, GetTensorData<int64_t>(&tensor)[i]);
      }
    } break;
    default: {
      // Checked in Prepare.
    }
  }
  return result;
}

const TfLiteTensor* GetRowPartitionTensor(
    const ConversionAttributes& conversion_attributes, TfLiteContext* context,
    TfLiteNode* node, int dimension) {
  if (conversion_attributes.partition_types.front() ==
      tensorflow::RowPartitionType::FIRST_DIM_SIZE) {
    return &context->tensors[node->inputs->data[kFirstPartitionInputIndex + 1 +
                                                dimension]];
  } else {
    return &context->tensors[node->inputs
                                 ->data[kFirstPartitionInputIndex + dimension]];
  }
}

int GetMaxWidthValueRowID(const TfLiteTensor* tensor) {
  const RuntimeShape tensor_shape(tensor->dims->size, tensor->dims->data);
  const int index_length = tensor_shape.FlatSize();
  if (index_length == 0) {
    return 0;
  }
  auto value_rowids = [tensor](int index) {
    switch (tensor->type) {
      case kTfLiteInt32:
        return static_cast<int>(tensor->data.i32[index]);
      case kTfLiteInt64:
        return static_cast<int>(tensor->data.i64[index]);
      default:
        // TODO(mgubin): Add error checks.
        return 0;
    }
  };
  int first_equal_index = 0;
  int first_equal_index_value = value_rowids(0);
  int max_width = 0;
  for (int i = 0; i < index_length; ++i) {
    const int value = value_rowids(i);
    if (value != first_equal_index_value) {
      first_equal_index_value = value;
      max_width = std::max(i - first_equal_index, max_width);
      first_equal_index = i;
    }
  }
  return std::max(index_length - first_equal_index, max_width);
}

int GetMaxWidthRowSplit(const TfLiteTensor* tensor) {
  const RuntimeShape tensor_shape(tensor->dims->size, tensor->dims->data);
  const int tensor_length = tensor_shape.FlatSize();
  if (tensor_length == 0 || tensor_length == 1) {
    return 0;
  }
  auto value_rowsplit = [tensor](int index) {
    switch (tensor->type) {
      case kTfLiteInt32:
        return static_cast<int>(tensor->data.i32[index]);
      case kTfLiteInt64:
        return static_cast<int>(tensor->data.i64[index]);
      default:
        // TODO(mgubin): Add error checks.
        return 0;
    }
  };
  int max_width = 1;
  int prev_split = value_rowsplit(0);
  for (int i = 1; i < tensor_length; ++i) {
    const int split = value_rowsplit(i);
    max_width = std::max(max_width, split - prev_split);
    prev_split = split;
  }
  return max_width;
}

int GetMaxWidth(const ConversionAttributes& conversion_attributes,
                TfLiteContext* context, TfLiteNode* node, int dimension) {
  const TfLiteTensor* tensor = GetRowPartitionTensor(
      conversion_attributes, context, node, dimension - 1);
  switch (conversion_attributes.GetRowPartitionTypeByDimension(dimension - 1)) {
    case tensorflow::RowPartitionType::VALUE_ROWIDS:
      return GetMaxWidthValueRowID(tensor);
    case tensorflow::RowPartitionType::ROW_SPLITS:
      return GetMaxWidthRowSplit(tensor);
    default:
      context->ReportError(context, "Cannot handle partition type");
      return -1;
  }
}

RuntimeShape CombineRaggedTensorToTensorShapes(
    int ragged_rank, const RuntimeShape& output_shape,
    const RuntimeShape& value_shape) {
  // TODO(mgubin): No checks, see
  // third_party/tensorflow/core/ops/ragged_to_dense_util.cc
  RuntimeShape result(output_shape);
  if (output_shape.DimensionsCount() == 0) {
    const int output_shape_rank = ragged_rank + value_shape.DimensionsCount();
    result.Resize(output_shape_rank);
    for (int i = 0; i < output_shape_rank; ++i) {
      result.SetDim(i, -1);
    }
  }
  const int need_to_set =
      output_shape.DimensionsCount() - value_shape.DimensionsCount();
  for (int i = 1; i < value_shape.DimensionsCount(); ++i) {
    result.SetDim(need_to_set + i, value_shape.Dims(i));
  }
  return result;
}

RuntimeShape CalculateOutputSize(
    const ConversionAttributes& conversion_attributes, TfLiteContext* context,
    TfLiteNode* node, int first_dimension, int ragged_rank,
    const TfLiteTensor& values, const TfLiteTensor& default_value,
    const TfLiteTensor& output_shape) {
  RuntimeShape values_shape(values.dims->size, values.dims->data);
  RuntimeShape default_value_shape(default_value.dims->size,
                                   default_value.dims->data);

  if (!ValidateDefaultValueShape(context, default_value_shape, values_shape)) {
    return {};
  }
  RuntimeShape output_shape_shape = TensorShapeFromTensor(output_shape);

  RuntimeShape result_shape = CombineRaggedTensorToTensorShapes(
      ragged_rank, output_shape_shape, values_shape);
  if (result_shape.Dims(0) < 0) {
    result_shape.SetDim(0, first_dimension);
  }
  for (int i = 1; i <= ragged_rank; ++i) {
    if (result_shape.Dims(i) < 0) {
      result_shape.SetDim(i,
                          GetMaxWidth(conversion_attributes, context, node, i));
    }
  }
  return result_shape;
}

TfLiteIntArray* IntArrayFromShape(const RuntimeShape& shape) {
  TfLiteIntArray* result = TfLiteIntArrayCreate(shape.DimensionsCount());
  for (int i = 0; i < shape.DimensionsCount(); ++i) {
    result->data[i] = shape.Dims(i);
  }
  return result;
}

/**
 * The output_index represents the index in the output tensor
 * where the first element of a particular dimension would be written.
 * If it is -1, it indicates that the index is out of scope.
 * Example, given first_dimension = 10, first_dimension_output = 6,
 * and output_index_multiplier = 100:
 * result = [0 100 200 300 400 500 -1 -1 -1 -1]
 * If first_dimension_output = 11 instead, then:
 * result = [0 100 200 300 400 500 600 700 800 900]
 */
void CalculateFirstParentOutputIndex(int first_dimension,
                                     int output_index_multiplier,
                                     int first_dimension_output,
                                     std::vector<int>* result) {
  const int min_dimension = std::min(first_dimension, first_dimension_output);
  result->reserve(first_dimension);
  int current_output_index = 0;
  for (int i = 0; i < min_dimension;
       ++i, current_output_index += output_index_multiplier) {
    result->push_back(current_output_index);
  }
  for (int i = min_dimension; i < first_dimension; ++i) {
    result->push_back(-1);
  }
}
// Calculate the output index of the first element of a list.
// The parent_output_index is the same computation for the previous list.
// -1 indicates an element or list that is out of range.
// The output_index_multiplier is the number of output indices one moves
// forward for each column.
// E.g., given:
// value_rowids:[0 1 2 2 2 3 5 5 6]
// parent_output_index:[1000 1100 2000 2100 -1 3000 4000]
// output_index_multiplier: 10
// output_size: 2
// You get:
// result = [1000 1100 2000 2010 -1 2100 -1 -1 3000]
// result[0] = parent_output_index[value_rowids[0]]
// result[1] = parent_output_index[value_rowids[1]]
// result[2] = parent_output_index[value_rowids[2]]
// result[3] = parent_output_index[value_rowids[2] + 10]
// result[4] = -1 because it is the third element the size is 2.
// result[5] = parent_output_index[value_rowids[3]]
// result[6] = -1 because parent_output_index[value_rowids[6]] == -1
// result[7] = -1 because parent_output_index[value_rowids[6]] == -1
// result[8] = parent_output_index[value_rowids[7]]
void CalculateOutputIndexValueRowID(const TfLiteTensor& value_rowids,
                                    const std::vector<int>& parent_output_index,
                                    int output_index_multiplier,
                                    int output_size, std::vector<int>* result) {
  const RuntimeShape tensor_shape(value_rowids.dims->size,
                                  value_rowids.dims->data);
  const int index_size = tensor_shape.FlatSize();
  result->reserve(index_size);
  if (index_size == 0) {
    return;
  }

  auto value_rowids_val = [value_rowids](int index) {
    switch (value_rowids.type) {
      case kTfLiteInt32:
        return static_cast<int>(value_rowids.data.i32[index]);
      case kTfLiteInt64:
        return static_cast<int>(value_rowids.data.i64[index]);
      default:
        // TODO(mgubin): Add error checks.
        return 0;
    }
  };
  int current_output_column = 0;
  int current_value_rowid = value_rowids_val(0);
  // DCHECK_LT(current_value_rowid, parent_output_index.size());
  int current_output_index = parent_output_index[current_value_rowid];
  result->push_back(current_output_index);
  for (int i = 1; i < index_size; ++i) {
    int next_value_rowid = value_rowids_val(i);
    if (next_value_rowid == current_value_rowid) {
      if (current_output_index >= 0) {
        ++current_output_column;
        if (current_output_column < output_size) {
          current_output_index += output_index_multiplier;
        } else {
          current_output_index = -1;
        }
      }
    } else {
      current_output_column = 0;
      current_value_rowid = next_value_rowid;
      // DCHECK_LT(next_value_rowid, parent_output_index.size());
      current_output_index = parent_output_index[next_value_rowid];
    }
    result->push_back(current_output_index);
  }
  // DCHECK_EQ(result->size(), value_rowids.size());
}

void CalculateOutputIndexRowSplit(const TfLiteTensor& row_split,
                                  const std::vector<int>& parent_output_index,
                                  int output_index_multiplier, int output_size,
                                  std::vector<int>* result) {
  const RuntimeShape row_split_shape(row_split.dims->size,
                                     row_split.dims->data);
  const int row_split_size = row_split_shape.FlatSize();
  auto row_split_val = [row_split](int index) {
    switch (row_split.type) {
      case kTfLiteInt32:
        return static_cast<int>(row_split.data.i32[index]);
      case kTfLiteInt64:
        return static_cast<int>(row_split.data.i64[index]);
      default:
        // TODO(mgubin): Add error checks.
        return 0;
    }
  };
  if (row_split_size > 0) {
    result->reserve(row_split_val(row_split_size - 1));
  }
  for (int i = 0; i < row_split_size - 1; ++i) {
    const int row_length = row_split_val(i + 1) - row_split_val(i);
    int real_length = std::min(output_size, row_length);
    int parent_output_index_current = parent_output_index[i];

    if (parent_output_index_current == -1) {
      real_length = 0;
    }
    for (int j = 0; j < real_length; ++j) {
      result->push_back(parent_output_index_current);
      parent_output_index_current += output_index_multiplier;
    }
    for (int j = 0; j < row_length - real_length; ++j) {
      result->push_back(-1);
    }
  }
  // if (row_split_size > 0) {
  //  DCHECK_EQ(result->size(), row_split(row_split_size - 1));
  //}
}

TfLiteStatus CalculateOutputIndex(
    const ConversionAttributes& conversion_attributes, TfLiteContext* context,
    TfLiteNode* node, int dimension,
    const std::vector<int>& parent_output_index, int output_index_multiplier,
    int output_size, std::vector<int>* result) {
  const TfLiteTensor* row_partition_tensor =
      GetRowPartitionTensor(conversion_attributes, context, node, dimension);
  auto partition_type =
      conversion_attributes.GetRowPartitionTypeByDimension(dimension);
  switch (partition_type) {
    case tensorflow::RowPartitionType::VALUE_ROWIDS:
      CalculateOutputIndexValueRowID(*row_partition_tensor, parent_output_index,
                                     output_index_multiplier, output_size,
                                     result);
      return kTfLiteOk;
    case tensorflow::RowPartitionType::ROW_SPLITS:
      CalculateOutputIndexRowSplit(*row_partition_tensor, parent_output_index,
                                   output_index_multiplier, output_size,
                                   result);
      return kTfLiteOk;
    default:
      context->ReportError(context, "Unsupported partition type");
      return kTfLiteError;
  }
}

template <typename VALUE_TYPE>
void SetOutputT(TfLiteContext* context, int ragged_rank,
                const std::vector<int>& output_index,
                const TfLiteTensor& values_tensor,
                const TfLiteTensor& default_value_tensor,
                TfLiteTensor* output_tensor) {
  const VALUE_TYPE* values_base = GetTensorData<VALUE_TYPE>(&values_tensor);
  VALUE_TYPE* output_base = GetTensorData<VALUE_TYPE>(output_tensor);
  const VALUE_TYPE* default_value =
      GetTensorData<VALUE_TYPE>(&default_value_tensor);

  RuntimeShape output_shape = GetTensorShape(output_tensor);
  RuntimeShape element_shape =
      RuntimeShape(output_shape.DimensionsCount() - ragged_rank - 1,
                   output_shape.DimsData() + ragged_rank + 1);

  // element_shape.RemoveDimRange(0, ragged_rank + 1);
  const int value_element_size = element_shape.FlatSize();
  size_t output_index_size = output_index.size();

  // Loop through the output_index vector, finding contiguous regions that
  // should be copied.  Once we find the end of a contiguous region, copy it
  // and add any necessary padding (with default_value).
  int src_start = 0;  // Start of contiguous region (in values)
  int dst_start = 0;  // Destination for contiguous region (in output)
  int dst_end = 0;    // Destination for contiguous region (in output)
  for (int src_i = 0; src_i <= output_index_size; ++src_i) {
    // dst_i is the destination where the value at src_i should be copied.
    int dst_i = src_i < output_index_size ? output_index[src_i] : -1;

    // If we're still in a contiguous region, then update dst_end go to the
    // next src_i.
    if (dst_i == dst_end) {
      ++dst_end;
      continue;
    }

    // We found the end of contiguous region.  This can be because we found
    // a gap (dst_i > dst_end), or a source value that shouldn't be copied
    // because it's out-of-bounds (dst_i == -1), or the end of the tensor
    // (dst_i = -1).
    if (dst_start < dst_end) {
      // Copy the contiguous region.
      const VALUE_TYPE* src = values_base + src_start * value_element_size;
      VALUE_TYPE* dst = output_base + dst_start * value_element_size;
      int nvals = (dst_end - dst_start) * value_element_size;
      std::copy(src, src + nvals, dst);
      // copy_array<VALUE_TYPE, int>(dst, src, nvals);
    }

    // Add any necessary padding (w/ default_value).
    if (src_i >= output_index_size) {
      // We reached the end of values: pad to the end of output.
      const int output_size = output_shape.FlatSize();
      dst_i = output_size / value_element_size;
    }
    if (dst_i > dst_end) {
      std::fill(output_base + dst_end * value_element_size,
                output_base + dst_i * value_element_size, *default_value);
      dst_end = dst_i;
    }

    // Update indices.
    if (dst_i < 0) {
      // src_i should be skipped -- leave it out of the contiguous region.
      src_start = src_i + 1;
      dst_start = dst_end;
    } else {
      // src_i should be copied -- include it in the contiguous region.
      src_start = src_i;
      dst_start = dst_end;
      dst_end = dst_start + 1;
    }
  }
}

bool IsSupportedTensorType(TfLiteType type) {
  // Should reflect SetOutput capabilities.
  return type == kTfLiteInt32 || type == kTfLiteInt64 || type == kTfLiteFloat32;
}

TfLiteStatus SetOutput(TfLiteContext* context, int ragged_rank,
                       const std::vector<int>& output_index,
                       const TfLiteTensor& values_tensor,
                       const TfLiteTensor& default_value_tensor,
                       TfLiteTensor* output_tensor) {
  switch (output_tensor->type) {
    case kTfLiteInt32:
      SetOutputT<int32_t>(context, ragged_rank, output_index, values_tensor,
                          default_value_tensor, output_tensor);
      return kTfLiteOk;
    case kTfLiteInt64:
      SetOutputT<int64_t>(context, ragged_rank, output_index, values_tensor,
                          default_value_tensor, output_tensor);
      return kTfLiteOk;
    case kTfLiteFloat32:
      SetOutputT<float>(context, ragged_rank, output_index, values_tensor,
                        default_value_tensor, output_tensor);
      return kTfLiteOk;
    default:
      // Should not happen, checked in Prepare.
      // Left as a defensive programming artifact for future updates.
      context->ReportError(context, "Not supported values type");
      return kTfLiteError;
  }
}

}  // namespace

void* Initialize(TfLiteContext* context, const char* buffer, size_t length) {
  auto attributes = std::make_unique<ConversionAttributes>();

  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);

  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  // TODO (mgubin): Converting flat buffer to a vector of strings looks not very
  // effective but simple. A cleaner way is needed.
  const flexbuffers::TypedVector row_partition_types_attr =
      m[kRowPartitionTypesAttr].AsTypedVector();
  std::vector<std::string> row_partition_types_attr_strings;
  row_partition_types_attr_strings.reserve(row_partition_types_attr.size());
  for (int i = 0; i < row_partition_types_attr.size(); ++i) {
    row_partition_types_attr_strings.emplace_back(
        row_partition_types_attr[i].AsString().str());
  }
  attributes->partition_types =
      tensorflow::GetRowPartitionTypesHelper(row_partition_types_attr_strings);
  if (attributes->partition_types.size() !=
      row_partition_types_attr_strings.size()) {
    context->ReportError(context, "Can't parse partition type attribute");
    return nullptr;
  }
  attributes->ragged_rank =
      tensorflow::GetRaggedRank(attributes->partition_types);
  return attributes.release();
}
void Free(TfLiteContext* /*context*/, void* buffer) {
  ConversionAttributes* attributes =
      reinterpret_cast<ConversionAttributes*>(buffer);
  delete attributes;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const ConversionAttributes* attributes =
      reinterpret_cast<ConversionAttributes*>(node->user_data);
  if (attributes == nullptr) {
    // Parsing attributes failed, can't prepare.
    context->ReportError(context, "Attributes are not initialized");
    return kTfLiteError;
  }
  TfLiteTensor& output_tensor =
      context->tensors[node->outputs->data[kOutputTensor]];
  if (!IsSupportedTensorType(output_tensor.type)) {
    context->ReportError(context, "Unsupported ragged tensor type");
    return kTfLiteError;
  }
  // The output tensor needs to be set to dynamic because it can have different
  // size.
  SetTensorToDynamic(&output_tensor);

  // Check that input shape tensor is int32 or int64
  TfLiteTensor& input_shape = context->tensors[node->inputs->data[kShapeInput]];
  if (input_shape.type != kTfLiteInt32 && input_shape.type != kTfLiteInt64) {
    context->ReportError(context,
                         "Input shape tensor could be only int32 or int64");
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const ConversionAttributes* attributes =
      reinterpret_cast<ConversionAttributes*>(node->user_data);
  TfLiteTensor& input_shape = context->tensors[node->inputs->data[kShapeInput]];
  TfLiteTensor& input_values =
      context->tensors[node->inputs->data[kValuesInput]];
  TfLiteTensor& default_value =
      context->tensors[node->inputs->data[kDefaultValueInput]];
  // TODO (mgubin): Only scallar default value is supported.
  if (RuntimeShape(default_value.dims->size, default_value.dims->data)
          .FlatSize() != 1) {
    context->ReportError(context, "Only scallar default value is supported");
    return kTfLiteError;
  }
  TfLiteTensor& first_partition_input =
      context->tensors[node->inputs->data[kFirstPartitionInputIndex]];

  // Calculate dimensions.
  const int first_dimension =
      GetFirstDimensionSize(context, first_partition_input, attributes);
  if (first_dimension < 0) {
    return kTfLiteError;
  }
  RuntimeShape output_shape = CalculateOutputSize(
      *attributes, context, node, first_dimension, attributes->ragged_rank,
      input_values, default_value, input_shape);
  if (output_shape.DimensionsCount() == 0) {
    return kTfLiteError;
  }

  std::vector<int> multiplier;
  multiplier.resize(attributes->ragged_rank + 1);
  multiplier.back() = 1;
  for (int i = multiplier.size() - 2; i >= 0; --i) {
    multiplier[i] = multiplier[i + 1] * output_shape.Dims(i + 1);
  }

  // Allocate output tensor.
  TfLiteTensor& output_tensor =
      context->tensors[node->outputs->data[kOutputTensor]];

  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, &output_tensor,
                                          IntArrayFromShape(output_shape)));

  // Copy data.
  const int full_size = multiplier.front() * output_shape.Dims(0);
  if (full_size > 0) {
    std::vector<int> output_index, new_output_index;
    int nvals = input_values.dims->data[0];
    output_index.reserve(nvals);
    new_output_index.reserve(nvals);

    CalculateFirstParentOutputIndex(first_dimension, multiplier[0],
                                    output_shape.Dims(0), &output_index);
    for (int i = 1; i <= attributes->ragged_rank; ++i) {
      TF_LITE_ENSURE_OK(
          context, CalculateOutputIndex(
                       *attributes, context, node, i - 1, output_index,
                       multiplier[i], output_shape.Dims(i), &new_output_index));
      output_index.swap(new_output_index);
      new_output_index.clear();
    }

    TF_LITE_ENSURE_OK(context,
                      SetOutput(context, attributes->ragged_rank, output_index,
                                input_values, default_value, &output_tensor));
  }
  return kTfLiteOk;
}

static TfLiteRegistration* GetTfLiteRegistration() {
  static TfLiteRegistration r = {Initialize, Free, Prepare, Eval};
  return &r;
}

}  // namespace ragged_tensor_to_tensor

extern "C" void AddRaggedTensorToTensor(tflite::MutableOpResolver* resolver) {
  resolver->AddCustom("RaggedTensorToTensor",
                      ragged_tensor_to_tensor::GetTfLiteRegistration());
}

TfLiteRegistration* Register_RAGGED_TENSOR_TO_TENSOR() {
  return ragged_tensor_to_tensor::GetTfLiteRegistration();
}

}  // namespace text
}  // namespace custom
}  // namespace ops
}  // namespace tflite
