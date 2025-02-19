//  Copyright (c) Qualcomm Innovation Center, Inc.
//  All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/dynamic_update_slice_op_builder.h"

#include <cstdint>
#include <numeric>
#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
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

  // reduceSum and reshape in[2] -> index tensor

  // Create static tensor table
  //  shape: [64]
  //  data: [0,...,63]

  // QNN ElementWiseNotEqual:
  //  in[0]: table
  //  in[1]: index tensor
  //  out[0]: condition tensor

  // reshape condition tensor due to QNN broadcast rules
  //  in[0]: [64]
  //  out[0]: [64, 1, 1]

  // QNN ElementWiseSelect:
  //  in[0] condition: [64, 1, 1]
  //  in[1] input: [1, 64, 4, 64]
  //  in[2] updates: [1, 1, 4, 64]

  // CAUTION!!! only support Gemma2 use case now

  auto& input_tensor = inputs[kInputIdx].get();
  auto& update_tensor = inputs[kUpdateIdx].get();
  auto& indices_tensor = inputs[kIndicesIdx].get();
  auto& output_tensor = outputs[kOutputIdx].get();

  if (input_tensor.GetRank() != update_tensor.GetRank()) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "QNN LiteRT Delegate only supports Dynamic Update Slice when "
               "operand and updates have the same rank.");
    return {};
  }

  if (indices_tensor.GetDataType() != QNN_DATATYPE_INT_32) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "Dynamic Update Slice only supports QNN_DATATYPE_INT_32 "
               "start_indices.");
    return {};
  }

  // reduce sum
  auto& reduce_sum_op = CreateOpWrapper(res, QNN_OP_REDUCE_SUM);
  reduce_sum_op.AddInputTensor(indices_tensor);

  std::vector<uint32_t> axis_data = {0};
  TensorWrapper& axis_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, {1},
      sizeof(std::uint32_t), axis_data.data());
  reduce_sum_op.AddTensorParam(QNN_OP_REDUCE_SUM_PARAM_AXES, axis_tensor);

  // create intermediate tensor
  TensorWrapper& one_dim_index =
      tensor_pool.CloneNativeTensorFrom(indices_tensor, {1});
  reduce_sum_op.AddOutputTensor(one_dim_index);

  // ElementwiseNotEqual
  // get table dims from in[0]->Dims[1]
  if (input_tensor.GetRank() < 2) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "Dynamic Update Slice only supports operand tensor rank >= 2");
    return {};
  }
  uint32_t table_size = input_tensor.GetDim(1);
  std::vector<uint32_t> static_table_dims = {table_size};
  std::vector<int32_t> table_data(table_size);
  std::iota(table_data.begin(), table_data.end(), 0);

  // create static table tensor
  TensorWrapper& static_table = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, QuantizeParamsWrapperVariant{}, static_table_dims,
      table_size * sizeof(std::int32_t), table_data.data());

  OpWrapper& not_equal_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_NOT_EQUAL);
  not_equal_op.AddInputTensor(static_table);
  not_equal_op.AddInputTensor(one_dim_index);

  TensorWrapper& not_equal_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, QuantizeParamsWrapperVariant{}, static_table_dims);
  not_equal_op.AddOutputTensor(not_equal_out);

  // reshape not equal output to [N, 1, 1]
  OpWrapper& reshape_op = CreateOpWrapper(res, QNN_OP_RESHAPE);

  reshape_op.AddInputTensor(not_equal_out);
  TensorWrapper& reshape_out =
      tensor_pool.CloneNativeTensorFrom(not_equal_out, {table_size, 1, 1});
  reshape_op.AddOutputTensor(reshape_out);

  // Select
  OpWrapper& select_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_SELECT);

  select_op.AddInputTensor(reshape_out);
  select_op.AddInputTensor(input_tensor);
  select_op.AddInputTensor(update_tensor);
  select_op.AddOutputTensor(output_tensor);
  return res;
}

}  // namespace qnn
