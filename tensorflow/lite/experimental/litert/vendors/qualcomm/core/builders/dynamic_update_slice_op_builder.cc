//  Copyright (c) Qualcomm Innovation Center, Inc.
//  All Rights Reserved.
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/dynamic_update_slice_op_builder.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {
namespace {
constexpr int kInputIdx = 0;
constexpr int kUpdateIdx = 1;
constexpr int kIndicesIdx = 2;
constexpr int kOutputIdx = 0;
}  // namespace

std::vector<OpWrapper> BuildDynamicUpdateSliceOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;
  // Dynamic Update Slice:
  //  in[0] operand: [1, 64, 4, 64]
  //  in[1] updates: [1, 1, 4, 64]
  //  in[2] start_indices: [4] -> data: [0, x, 0, 0]

  // Transpose in[0] and in[1]
  // Slice and reshape in[2]

  // QNN ScatterNd:
  //  in[0] input: [64, 1, 4, 64]
  //  in[1] indices: [1, 1] -> data: [[x]]
  //  in[2] updates: [1, 1, 4, 64]

  // TODO: check support, only support gemma2 case now

  auto& input_tensor = inputs[kInputIdx].get();
  auto& update_tensor = inputs[kUpdateIdx].get();
  auto& indices_tensor = inputs[kIndicesIdx].get();
  auto& output_tensor = outputs[kOutputIdx].get();

  if (input_tensor.GetRank() != update_tensor.GetRank()) {
    // TODO: log
    std::cout << "LiteRT QNN Delegate only supports Dynamic Update Slice when "
                 "operand "
                 "and updates have the same rank."
              << std::endl;
    return {};
  }

  if (input_tensor.GetRank() < 2) {
    // TODO: log
    std::cout << "LiteRT QNN Delegate does not support Dynamic Update Slice "
                 "operand rank < 2."
              << std::endl;
    return {};
  }
  std::vector<std::uint32_t> perm_dims = {input_tensor.GetRank()};
  std::vector<std::uint32_t> perm_data = {1, 0};
  for (size_t i = 2; i < perm_dims[0]; i++) {
    perm_data.emplace_back(i);
  }

  // transpose input
  auto& input_transpose = CreateOpWrapper(res, QNN_OP_TRANSPOSE);
  input_transpose.AddInputTensor(input_tensor);
  TensorWrapper& transpose_param_0 = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, perm_dims,
      sizeof(std::uint32_t) * perm_dims[0], perm_data.data());
  input_transpose.AddTensorParam(QNN_OP_TRANSPOSE_PARAM_PERM,
                                 transpose_param_0);

  auto& input_dims = input_tensor.GetDims();
  // check dims large enough
  std::vector<std::uint32_t> transposed_in_dims = {input_dims[1],
                                                   input_dims[0]};

  for (size_t i = 2; i < input_dims.size(); i++) {
    transposed_in_dims.emplace_back(input_dims[i]);
  }
  // create intermediate tensor
  TensorWrapper& transposed_input =
      tensor_pool.CloneNativeTensorFrom(input_tensor, transposed_in_dims);
  input_transpose.AddOutputTensor(transposed_input);

  // transpose update
  OpWrapper& update_transpose = CreateOpWrapper(res, QNN_OP_TRANSPOSE);
  update_transpose.AddInputTensor(update_tensor);
  update_transpose.AddTensorParam(QNN_OP_TRANSPOSE_PARAM_PERM,
                                  transpose_param_0);

  auto& update_dims = update_tensor.GetDims();
  std::vector<std::uint32_t> transposed_update_dims = {update_dims[1],
                                                       update_dims[0]};

  for (size_t i = 2; i < update_dims.size(); i++) {
    transposed_update_dims.emplace_back(update_dims[i]);
  }
  // create intermediate tensor
  TensorWrapper& transposed_update =
      tensor_pool.CloneNativeTensorFrom(update_tensor, transposed_update_dims);
  update_transpose.AddOutputTensor(transposed_update);

  // slice indices
  OpWrapper& strided_slice_op = CreateOpWrapper(res, QNN_OP_STRIDED_SLICE);

  strided_slice_op.AddInputTensor(indices_tensor);

  std::vector<std::int32_t> ranges = {1, 2, 1};
  TensorWrapper& range_tensor_param = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, QuantizeParamsWrapperVariant{}, {1, 3},
      sizeof(std::uint32_t) * 3, ranges.data());

  strided_slice_op.AddTensorParam(QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                  range_tensor_param);

  TensorWrapper& sliced_index =
      tensor_pool.CloneNativeTensorFrom(indices_tensor, {1});
  strided_slice_op.AddOutputTensor(sliced_index);

  // reshape
  OpWrapper& reshape_op = CreateOpWrapper(res, QNN_OP_RESHAPE);

  reshape_op.AddInputTensor(sliced_index);
  TensorWrapper& reshaped_sliced_index =
      tensor_pool.CloneNativeTensorFrom(sliced_index, {1, 1});
  reshape_op.AddOutputTensor(reshaped_sliced_index);

  // scatterNd
  OpWrapper& scatter_nd_op = CreateOpWrapper(res, QNN_OP_SCATTER_ND);

  scatter_nd_op.AddInputTensor(transposed_input);
  scatter_nd_op.AddInputTensor(reshaped_sliced_index);
  scatter_nd_op.AddInputTensor(transposed_update);

  // check dims large enough
  std::vector<std::uint32_t> scatter_nd_out_dims = transposed_in_dims;

  TensorWrapper& scatter_nd_out =
      tensor_pool.CloneNativeTensorFrom(output_tensor, scatter_nd_out_dims);
  scatter_nd_op.AddOutputTensor(scatter_nd_out);

  // transpose output
  OpWrapper& output_transpose = CreateOpWrapper(res, QNN_OP_TRANSPOSE);
  output_transpose.AddInputTensor(scatter_nd_out);
  TensorWrapper& transpose_param_2 = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, perm_dims,
      sizeof(std::uint32_t) * perm_dims[0], perm_data.data());
  output_transpose.AddTensorParam(QNN_OP_TRANSPOSE_PARAM_PERM,
                                  transpose_param_2);

  output_transpose.AddOutputTensor(output_tensor);
  return res;
}

}  // namespace qnn
