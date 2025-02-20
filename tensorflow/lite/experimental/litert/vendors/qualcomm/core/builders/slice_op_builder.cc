//  Copyright (c) Qualcomm Innovation Center, Inc.
//  All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/slice_op_builder.h"

namespace qnn {

namespace {
constexpr int kDefaultStrideValue = 1;
constexpr int kSizeNegative = -1;
constexpr int kRangeNumElements = 3;
}  // namespace

std::vector<OpWrapper> BuildSliceOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  TensorWrapper& input_tensor = inputs[0];
  TensorWrapper& begin_tensor = inputs[1];
  TensorWrapper& size_tensor = inputs[2];
  if (!begin_tensor.IsTensorStatic() || !size_tensor.IsTensorStatic()) {
    // TODO: error log
    return res;
  }

  const auto input_rank = input_tensor.GetRank();
  auto begin_data =
      reinterpret_cast<const std::int32_t*>(begin_tensor.GetStaticTensorData());
  auto size_data =
      reinterpret_cast<const std::int32_t*>(size_tensor.GetStaticTensorData());
  std::vector<std::int32_t> range_data;
  range_data.reserve(input_rank * kRangeNumElements);
  for (size_t i = 0; i < input_rank; ++i) {
    range_data.emplace_back(begin_data[i]);
    if (size_data[i] == kSizeNegative) {
      range_data.emplace_back(input_tensor.GetDim(i));
    } else {
      range_data.emplace_back(begin_data[i] + size_data[i]);
    }
    range_data.emplace_back(kDefaultStrideValue);
  }
  TensorWrapper& range_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, begin_tensor.GetQuantParams(),
      {input_rank, kRangeNumElements}, sizeof(std::int32_t) * range_data.size(),
      range_data.data());

  auto& slice_op = CreateOpWrapper(res, QNN_OP_STRIDED_SLICE);
  slice_op.AddTensorParam(QNN_OP_STRIDED_SLICE_PARAM_RANGES, range_tensor);
  slice_op.AddInputTensor(input_tensor);
  slice_op.AddOutputTensor(outputs[0]);

  return res;
}

}  // namespace qnn
