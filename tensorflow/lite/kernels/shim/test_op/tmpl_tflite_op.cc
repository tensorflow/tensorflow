/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/shim/test_op/tmpl_tflite_op.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/shim/test_op/tmpl_op.h"
#include "tensorflow/lite/kernels/shim/tflite_op_shim.h"
#include "tensorflow/lite/kernels/shim/tflite_op_wrapper.h"

namespace tflite {
namespace ops {
namespace custom {
namespace {
const char a_type[]("AType"), b_type[]("BType");
}  // namespace

using ::tflite::shim::op_wrapper::Attr;
using ::tflite::shim::op_wrapper::AttrName;
using ::tflite::shim::op_wrapper::OpWrapper;

template <shim::Runtime Rt>
using Op = OpWrapper<Rt, shim::TmplOp, Attr<AttrName<a_type>, int32_t, float>,
                     Attr<AttrName<b_type>, int32_t, int64_t, bool>>;

using OpKernel = ::tflite::shim::TfLiteOpKernel<Op>;

void AddTmplOp(MutableOpResolver* resolver) { OpKernel::Add(resolver); }

TfLiteRegistration* Register_TMPL_OP() {
  return OpKernel::GetTfLiteRegistration();
}

const char* OpName_TMPL_OP() { return OpKernel::OpName(); }

}  // namespace custom
}  // namespace ops
}  // namespace tflite
