/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/shim/test_op/tmpl_tf_op.h"

#include <cstdint>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/lite/kernels/shim/tf_op_shim.h"

namespace tflite {
namespace shim {

using TmplOpKernelInstance = TmplOpKernel<float, int32_t>;

REGISTER_TF_OP_SHIM(TmplOpKernelInstance);

REGISTER_KERNEL_BUILDER(Name(TmplOpKernelInstance::OpName())
                            .Device(::tensorflow::DEVICE_CPU)
                            .TypeConstraint<float>("AType")
                            .TypeConstraint<int32_t>("BType"),
                        TmplOpKernel<float, int32_t>);

REGISTER_KERNEL_BUILDER(Name(TmplOpKernelInstance::OpName())
                            .Device(::tensorflow::DEVICE_CPU)
                            .TypeConstraint<int32_t>("AType")
                            .TypeConstraint<int64_t>("BType"),
                        TmplOpKernel<int32_t, int64_t>);

}  // namespace shim
}  // namespace tflite
