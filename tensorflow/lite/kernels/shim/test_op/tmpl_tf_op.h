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
#ifndef TENSORFLOW_LITE_KERNELS_SHIM_TEST_OP_TMPL_TF_OP_H_
#define TENSORFLOW_LITE_KERNELS_SHIM_TEST_OP_TMPL_TF_OP_H_

#include "tensorflow/lite/kernels/shim/test_op/tmpl_op.h"
#include "tensorflow/lite/kernels/shim/tf_op_shim.h"

namespace tflite {
namespace shim {

template <typename AType, typename BType>
class TmplOpKernel : public TfOpKernel<TmplOp, AType, BType> {
 public:
  using TfOpKernel<TmplOp, AType, BType>::TfOpKernel;
};

}  // namespace shim
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SHIM_TEST_OP_TMPL_TF_OP_H_
