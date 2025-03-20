// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_OP_BUILDER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_OP_BUILDER_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

enum class PaddingType {
  Unkown = 0,
  Same,
  Valid,
};

std::pair<std::uint32_t, std::uint32_t> ComputePaddingBeforeAfter(
    const std::uint32_t input_size, const std::uint32_t filter_size,
    const std::uint32_t stride, const std::uint32_t dilation_rate,
    const PaddingType padding_type);

OpWrapper& CreateOpWrapper(std::vector<OpWrapper>& ops, const char* op_type);

OpWrapper& CreateSimpleActivationOp(std::vector<OpWrapper>& ops,
                                    const char* op_type,
                                    const TensorWrapper& input_tensor,
                                    const TensorWrapper& output_tensor);

void ConvertFp32ActivationToFp16IfWeightOnlyQuantized(
    std::vector<OpWrapper>& res, TensorWrapper& fp32_input_activation,
    TensorWrapper& fp32_output_activation, TensorWrapper*& input_activation,
    TensorWrapper*& output_activation, bool is_int8_weight_only_quantized,
    TensorPool& tensor_pool);

void ConvertFp32ActivationToFp16(std::vector<OpWrapper>& res,
                                 TensorWrapper& fp32_activation,
                                 TensorWrapper*& activation,
                                 bool is_int8_weight_only_quantized,
                                 TensorPool& tensor_pool);

void ConvertFp16ActivationToFp32IfWeightOnlyQuantized(
    std::vector<OpWrapper>& res, TensorWrapper* fp16_output_activation,
    TensorWrapper& output_activation, bool is_int8_weight_only_quantized);

void AddFusedActivationNode(std::vector<OpWrapper>& res,
                            const uint32_t fused_activation_function,
                            const TensorWrapper& input_tensor,
                            const TensorWrapper& output_tensor);

}  // namespace qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_OP_BUILDER_H_
