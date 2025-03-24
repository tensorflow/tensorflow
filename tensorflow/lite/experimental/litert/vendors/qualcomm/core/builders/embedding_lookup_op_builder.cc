// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/embedding_lookup_op_builder.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/utils/log.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {
namespace {
constexpr int kTableIdx = 1;
constexpr int kIndicesIdx = 0;
constexpr int kOutputIdx = 0;
}  // namespace

std::vector<OpWrapper> BuildEmbeddingLookupOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  TensorWrapper& table_tensor = inputs[kTableIdx];
  TensorWrapper& indices_tensor = inputs[kIndicesIdx];
  TensorWrapper& output_tensor = outputs[kOutputIdx];

  auto& gather_op = CreateOpWrapper(res, QNN_OP_GATHER);
  // Case: QInt8 table with QInt16 output
  if (table_tensor.IsQuant8() && output_tensor.IsQuant16()) {
    QNN_LOG_WARNING(
        "The data type of embedding lookup table is int8, but output data type "
        "is int16. Int8 table will be cast to int16.");
    std::vector<std::int16_t> int16_data;
    size_t data_len = table_tensor.GetTensorNumElements();
    auto int8_data = table_tensor.GetStaticTensorData<std::int8_t>();
    if (!int8_data.has_value()) {
      QNN_LOG_ERROR("Embedding lookup get int8 table failed.");
      return res;
    }
    for (int i = 0; i < data_len; ++i) {
      int16_data.emplace_back(static_cast<std::int16_t>((*int8_data)[i]));
    }

    TensorWrapper& int16_table_tensor = tensor_pool.CreateStaticTensor(
        output_tensor.GetDataType(), table_tensor.GetQuantParams(),
        table_tensor.GetDims(),
        sizeof(decltype(int16_data)::value_type) * int16_data.size(),
        reinterpret_cast<void*>(int16_data.data()));

    gather_op.AddInputTensor(int16_table_tensor);
  } else {
    gather_op.AddInputTensor(table_tensor);
  }

  gather_op.AddInputTensor(indices_tensor);
  gather_op.AddOutputTensor(output_tensor);
  gather_op.AddScalarParam<std::int32_t>(QNN_OP_GATHER_PARAM_AXIS, 0);
  return res;
}

}  // namespace qnn
