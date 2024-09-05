/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/tensor_slice_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_gather {
namespace {

constexpr int kOperandTensor = 0;
constexpr int kStartIndicesTensor = 1;
constexpr int kOutputTensor = 0;

using TfLiteIntArrayUniquePtr =
    std::unique_ptr<TfLiteIntArray, decltype(&TfLiteIntArrayFree)>;

// Clips the starting indices given the operand_shape and slice_sizes. This
// means the starting index in a dimension will be shifted back if necessary so
// that the whole slice can fit in the operand.
// Example:
// starting_index = [i, j], operand_shape = [oi, oj], slice_sizes = [si, sj]
// starting_index will be transformed to [min(i, oi - si), min(j, oj - sj)]
template <typename IndexType>
TfLiteStatus ClipStartingIndex(const RuntimeShape& operand_shape,
                               const int64_t* slice_sizes, int num_slice_sizes,
                               Index<IndexType>& starting_index) {
  if (operand_shape.DimensionsCount() != starting_index.size() ||
      operand_shape.DimensionsCount() != num_slice_sizes) {
    return kTfLiteError;
  }
  for (int dim = 0; dim < starting_index.size(); ++dim) {
    starting_index[dim] = std::min((int64_t)starting_index[dim],
                                   operand_shape.Dims(dim) - slice_sizes[dim]);
  }
  return kTfLiteOk;
}

// Returns a vector containing slice_sizes with all the entries with indices
// that are present in collapsed_slice_dims removed.
// Example: slice_sizes = {3, 5, 2, 7}, collapsed_slice_dims = {1, 3}
// Result: {3, 2}
static std::vector<int64_t> GetCollapsedSliceShape(
    const int64_t* slice_sizes, int num_slice_sizes,
    const int64_t* collapsed_slice_dims, int num_collapsed_slice_dims) {
  std::vector<int64_t> result(num_slice_sizes - num_collapsed_slice_dims);
  int result_ctr = 0;
  for (int dim = 0; dim < num_slice_sizes; dim++) {
    if (!ArrayContains(collapsed_slice_dims, num_collapsed_slice_dims, dim)) {
      result[result_ctr] = slice_sizes[dim];
      result_ctr++;
    }
  }
  return result;
}

// Creates the result shape based on the rank of the result, options and
// shape of the result_indices operand.
// Refer to the spec for a full explanation:
// https://github.com/openxla/stablehlo/blob/main/docs/spec.md#gather
static TfLiteIntArrayUniquePtr GetResultShape(
    int64_t result_rank, const TfLiteStablehloGatherParams* data,
    const RuntimeShape& start_indices_shape) {
  TfLiteIntArrayUniquePtr result = TfLiteIntArrayUniquePtr(
      TfLiteIntArrayCreate(result_rank), &TfLiteIntArrayFree);
  int result_ctr = 0;

  std::vector<int64_t> collapsed_slice_shape = GetCollapsedSliceShape(
      data->slice_sizes, data->num_slice_sizes, data->collapsed_slice_dims,
      data->num_collapsed_slice_dims);
  int64_t slice_shape_ctr = 0;
  int64_t start_indices_shape_ctr = 0;

  for (int64_t dim = 0; dim < result_rank; dim++) {
    if (ArrayContains(data->offset_dims, data->num_offset_dims, dim)) {
      result->data[result_ctr] = collapsed_slice_shape[slice_shape_ctr];
      slice_shape_ctr++;
    } else {
      if (start_indices_shape_ctr == data->index_vector_dim) {
        start_indices_shape_ctr++;
      }
      result->data[result_ctr] =
          start_indices_shape.Dims(start_indices_shape_ctr);
      start_indices_shape_ctr++;
    }
    result_ctr++;
  }
  return result;
}

// Extracts the batch and offset indices out of a given result index.
// Result index is the index of an element in the output(result) tensor.
// The location of the offset dims is given in the offset_dims argument and
// the rest are batch dimensions.
template <typename IndexType>
TfLiteStatus SetBatchAndOffsetIndices(const Index<IndexType>& result_index,
                                      const int64_t* offset_dims,
                                      int num_offset_dims,
                                      Index<IndexType>& batch_index,
                                      Index<IndexType>& offset_index) {
  int offset_index_ctr = 0;
  int batch_index_ctr = 0;
  for (int result_dim = 0; result_dim < result_index.size(); ++result_dim) {
    if (ArrayContains(offset_dims, num_offset_dims, result_dim)) {
      if (offset_index_ctr >= num_offset_dims) {
        return kTfLiteError;
      }
      offset_index[offset_index_ctr] = result_index[result_dim];
      offset_index_ctr++;
    } else {
      if (batch_index_ctr >= result_index.size() - num_offset_dims) {
        return kTfLiteError;
      }
      batch_index[batch_index_ctr] = result_index[result_dim];
      batch_index_ctr++;
    }
  }
  return kTfLiteOk;
}

// Evaluates this node given the type of the elements in the start_indices
// and the type of the elements in the operand tensor.
template <typename IndexType, typename DataType>
TfLiteStatus EvalWithTypes(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* operand;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kOperandTensor, &operand));
  int operand_rank = operand->dims->size;
  RuntimeShape operand_shape = GetTensorShape(operand);

  const TfLiteTensor* start_indices;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kStartIndicesTensor,
                                          &start_indices));

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  const TfLiteStablehloGatherParams* data =
      reinterpret_cast<TfLiteStablehloGatherParams*>(node->builtin_data);

  RuntimeShape start_indices_shape = GetTensorShape(start_indices);
  int result_rank = output->dims->size;
  RuntimeShape result_runtime_shape(result_rank, output->dims->data);
  Index<IndexType> result_index = Index<IndexType>(result_rank, 0);

  int64_t num_batch_dims = result_rank - data->num_offset_dims;

  Index<IndexType> batch_index(num_batch_dims);
  Index<IndexType> offset_index(data->num_offset_dims);
  do {
    TF_LITE_ENSURE_OK(
        context, SetBatchAndOffsetIndices(result_index, data->offset_dims,
                                          data->num_offset_dims, batch_index,
                                          offset_index));

    Index<IndexType> starting_index_vector =
        ReadIndexVector(start_indices, start_indices_shape, batch_index,
                        data->index_vector_dim);

    Index<IndexType> final_starting_index;
    ScatterIndex(starting_index_vector, data->start_index_map,
                 data->num_start_index_map, operand_rank,
                 &final_starting_index);

    TF_LITE_ENSURE_OK(
        context,
        ClipStartingIndex(operand_shape, data->slice_sizes,
                          data->num_slice_sizes, final_starting_index));

    Index<IndexType> full_offset_index;
    ExpandDims(offset_index, data->collapsed_slice_dims,
               data->num_collapsed_slice_dims, &full_offset_index);

    Index<IndexType> operand_lookup_index =
        AddIndices(final_starting_index, full_offset_index);

    const DataType* operand_data = GetTensorData<DataType>(operand);
    IndexType flat_operand_index =
        TensorIndexToFlat(operand_lookup_index.data(),
                          operand_lookup_index.size(), GetTensorShape(operand));
    DataType looked_up_value = operand_data[flat_operand_index];

    DataType* result_data = GetTensorData<DataType>(output);
    IndexType flat_result_index = TensorIndexToFlat(
        result_index.data(), result_index.size(), GetTensorShape(output));
    result_data[flat_result_index] = looked_up_value;
  } while (NextIndex(result_rank, result_runtime_shape.DimsData(),
                     result_index.data()));

  return TfLiteStatus::kTfLiteOk;
}

// Evaluates this node given the type of the elements in the scatter_indices
// tensor.
template <typename IndexType>
TfLiteStatus EvalWithIndexType(TfLiteContext* context, TfLiteNode* node,
                               TfLiteType index_type, TfLiteType data_type) {
  switch (data_type) {
    case kTfLiteFloat16:
      return EvalWithTypes<IndexType, Eigen::half>(context, node);
    case kTfLiteFloat32:
      return EvalWithTypes<IndexType, float>(context, node);
    case kTfLiteFloat64:
      return EvalWithTypes<IndexType, double>(context, node);
    case kTfLiteInt8:
      return EvalWithTypes<IndexType, int8_t>(context, node);
    case kTfLiteInt16:
      return EvalWithTypes<IndexType, int16_t>(context, node);
    case kTfLiteInt32:
      return EvalWithTypes<IndexType, int32_t>(context, node);
    case kTfLiteInt64:
      return EvalWithTypes<IndexType, int64_t>(context, node);
    case kTfLiteUInt8:
      return EvalWithTypes<IndexType, uint8_t>(context, node);
    case kTfLiteUInt16:
      return EvalWithTypes<IndexType, uint16_t>(context, node);
    case kTfLiteUInt32:
      return EvalWithTypes<IndexType, uint32_t>(context, node);
    case kTfLiteUInt64:
      return EvalWithTypes<IndexType, uint64_t>(context, node);
    default:
      TF_LITE_KERNEL_LOG(
          context, "(Index Type: %s, Data Type: %s) currently not supported.\n",
          TfLiteTypeGetName(index_type), TfLiteTypeGetName(data_type));
      return TfLiteStatus::kTfLiteError;
  }
}

}  // namespace

// This is the kernel for stablehlo.gather which receives `slice_sizes` as a
// static attribute.
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* operand;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kOperandTensor, &operand));
  const TfLiteTensor* start_indices;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kStartIndicesTensor,
                                          &start_indices));

  TfLiteType index_type = start_indices->type;
  TfLiteType data_type = operand->type;

  if (index_type == kTfLiteInt32) {
    return EvalWithIndexType<int32_t>(context, node, index_type, data_type);
  } else if (index_type == kTfLiteInt64) {
    return EvalWithIndexType<int64_t>(context, node, index_type, data_type);
  } else {
    TF_LITE_KERNEL_LOG(context, "(Index Type: %s) currently not supported.\n",
                       TfLiteTypeGetName(index_type));
    return TfLiteStatus::kTfLiteError;
  }
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* operand;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kOperandTensor, &operand));
  const TfLiteTensor* start_indices;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kStartIndicesTensor,
                                          &start_indices));

  TfLiteType index_type = start_indices->type;
  if (index_type != kTfLiteInt32 && index_type != kTfLiteInt64) {
    TF_LITE_KERNEL_LOG(context, "(Index Type: %s) currently not supported.\n",
                       TfLiteTypeGetName(index_type));
    return TfLiteStatus::kTfLiteError;
  }

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  const TfLiteStablehloGatherParams* data =
      reinterpret_cast<TfLiteStablehloGatherParams*>(node->builtin_data);

  RuntimeShape start_indices_shape = GetTensorShape(start_indices);

  TfLiteIntArrayUniquePtr result_shape =
      GetResultShape(output->dims->size, data, start_indices_shape);

  // ResizeTensor takes ownership of result_shape
  TF_LITE_ENSURE_STATUS(
      context->ResizeTensor(context, output, result_shape.release()));

  return TfLiteStatus::kTfLiteOk;
}

}  // namespace stablehlo_gather

TfLiteRegistration* Register_STABLEHLO_GATHER() {
  static TfLiteRegistration r = {nullptr, nullptr, stablehlo_gather::Prepare,
                                 stablehlo_gather::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
