// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/gathernd_op_builder.h"

#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

namespace {

constexpr size_t kInputIndex = 0;
constexpr size_t kIndicesIndex = 1;
constexpr size_t kOutputIndex = 0;

}  // namespace

std::vector<OpWrapper> BuildGatherNdOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs,
    const std::uint32_t batch_dims) {
  std::vector<OpWrapper> res;

  OpWrapper& gathernd_op = CreateOpWrapper(res, QNN_OP_GATHER_ND);
  gathernd_op.AddInputTensor(inputs[kInputIndex]);
  gathernd_op.AddInputTensor(inputs[kIndicesIndex]);
  gathernd_op.AddOutputTensor(outputs[kOutputIndex]);
  gathernd_op.AddScalarParam<std::uint32_t>(QNN_OP_GATHER_ND_PARAM_BATCH_DIMS,
                                            batch_dims);

  return res;
}

}  // namespace qnn
