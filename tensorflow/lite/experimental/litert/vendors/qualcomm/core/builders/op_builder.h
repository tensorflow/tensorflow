//  Copyright (c) Qualcomm Innovation Center, Inc.
//  All Rights Reserved.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_OP_BUILDER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_OP_BUILDER_H_

#include <vector>

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

OpWrapper& CreateOpWrapper(std::vector<OpWrapper>& ops, const char* op_type);

OpWrapper& CreateSimpleActivationOp(std::vector<OpWrapper>& ops,
                                    const char* op_type,
                                    const TensorWrapper& input_tensor,
                                    const TensorWrapper& output_tensor);

}  // namespace qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_OP_BUILDER_H_
