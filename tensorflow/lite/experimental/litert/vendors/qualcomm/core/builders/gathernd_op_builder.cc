// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/gathernd_op_builder.h"

#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/utils/log.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

namespace {

constexpr size_t kInputIndex = 0;
constexpr size_t kIndicesIndex = 1;
constexpr size_t kOutputIndex = 0;
constexpr size_t kRangeNumElements = 3;  // [begin, end, stride]

bool TransformToStrideSliceOp(TensorPool& tensor_pool,
                              std::vector<OpWrapper>& res,
                              const TensorWrapper& input,
                              const TensorWrapper& indices,
                              const TensorWrapper& output) {
  if (indices.GetRank() != 2 || indices.GetDim(1) != 1) {
    QNN_LOG_WARNING(
        "Failed to transform GatherNd into StrideSlice because the shape of "
        "the indices tensor is not appropriate.");
    return false;
  }

  const auto indices_data = indices.GetStaticTensorData<std::int32_t>();
  if (!indices_data.has_value()) {
    QNN_LOG_WARNING(
        "Failed to get the static data when transforming GatherNd into "
        "StrideSlice.");
    return false;
  }

  // Compute and check the range data for the first dimension.
  const std::int32_t begin = (*indices_data).front();
  const std::int32_t end = (*indices_data).back();
  const std::int32_t stride =
      (begin == end) ? 1 : (end - begin) / ((*indices_data).size() - 1);
  for (size_t i = 0; i < (*indices_data).size(); ++i) {
    if (begin + i * stride != (*indices_data)[i]) {
      QNN_LOG_WARNING(
          "Failed to transform GatherNd into StrideSlice because the indices "
          "are not appropriate.");
      return false;
    }
  }

  // Fill the range data in the format [begin, end, stride, ...].
  std::vector<std::int32_t> range_data(input.GetRank() * kRangeNumElements);
  range_data[0] = begin;
  range_data[1] = end + 1;
  range_data[2] = stride;
  for (size_t i = 1; i < input.GetRank(); ++i) {
    range_data[i * kRangeNumElements] = 0;
    range_data[i * kRangeNumElements + 1] = input.GetDim(i);
    range_data[i * kRangeNumElements + 2] = 1;
  }
  TensorWrapper& range_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {input.GetRank(), 3},
      sizeof(std::int32_t) * range_data.size(), range_data.data());

  auto& slice_op = CreateOpWrapper(res, QNN_OP_STRIDED_SLICE);
  slice_op.AddInputTensor(input);
  slice_op.AddOutputTensor(output);
  slice_op.AddTensorParam(QNN_OP_STRIDED_SLICE_PARAM_RANGES, range_tensor);
  return true;
}

}  // namespace

std::vector<OpWrapper> BuildGatherNdOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs,
    const std::uint32_t batch_dims) {
  std::vector<OpWrapper> res;

  auto& input_tensor = inputs[kInputIndex].get();
  auto& indices_tensor = inputs[kIndicesIndex].get();

  if (indices_tensor.IsTensorStatic()) {
    if (TransformToStrideSliceOp(tensor_pool, res, input_tensor, indices_tensor,
                                 outputs[kOutputIndex])) {
      return res;
    } else {
      QNN_LOG_ERROR("Static indices is not supported for GatherNd op.");
      return res;
    }
  }

  OpWrapper& gathernd_op = CreateOpWrapper(res, QNN_OP_GATHER_ND);
  gathernd_op.AddInputTensor(input_tensor);
  gathernd_op.AddInputTensor(indices_tensor);
  gathernd_op.AddOutputTensor(outputs[kOutputIndex]);
  gathernd_op.AddScalarParam<std::uint32_t>(QNN_OP_GATHER_ND_PARAM_BATCH_DIMS,
                                            batch_dims);

  return res;
}

}  // namespace qnn
