//  Copyright (c) Qualcomm Innovation Center, Inc.
//  All Rights Reserved.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_PACK_OP_BUILDER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_PACK_OP_BUILDER_H_

#include <cstdint>
#include <vector>

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

std::vector<OpWrapper> BuildPackOp(TensorPool& tensor_pool,
                                   const std::vector<TensorWrapperRef>& inputs,
                                   const std::vector<TensorWrapperRef>& outputs,
                                   const int32_t axis);

}  // namespace qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_PACK_OP_BUILDER_H_
