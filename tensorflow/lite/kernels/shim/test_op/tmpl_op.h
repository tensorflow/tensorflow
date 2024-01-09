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
#ifndef TENSORFLOW_LITE_KERNELS_SHIM_TEST_OP_TMPL_OP_H_
#define TENSORFLOW_LITE_KERNELS_SHIM_TEST_OP_TMPL_OP_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"

namespace tflite {
namespace shim {

// A simple operation for demonstration and testing purposes.
// See the kDoc member for documentation.

template <Runtime Rt, typename AType, typename BType>
class TmplOp : public OpKernelShim<TmplOp, Rt, AType, BType> {
 protected:
  enum Inputs { kInput0 = 0, kInput1 };
  enum Outputs { kOutput0 = 0 };

 public:
  using typename OpKernelShim<TmplOp, Rt, AType, BType>::InitContext;
  using typename OpKernelShim<TmplOp, Rt, AType, BType>::InvokeContext;
  using typename OpKernelShim<TmplOp, Rt, AType, BType>::ShapeInferenceContext;

  TmplOp() = default;
  static constexpr char kOpName[] = "TemplatizedOperation";
  static constexpr char kDoc[] = R"doc(
Description:
  Templatized op for testing and demonstration purposes.

Attrs
  AType: The type for input0
  BType: The type for input1
Inputs
  in0: AType, shape=[] - A scalar input
  in1: BType, shape=[] - A scalar input
Outputs
  out0: int, shape=[] - first output
)doc";

  static const char* OpName() { return kOpName; }
  static const char* Doc() { return kDoc; }

  // Attributes declaration (syntax: https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Attrs() {
    return {"AType: {int32, float} = DT_INT32", "BType: type"};
  }
  // Input tensors declaration (syntax:
  // https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Inputs() {
    return {"in0: AType", "in1: BType"};
  }
  // Output tensors declaration (syntax:
  // https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Outputs() { return {"out0: float"}; }

  // Initializes the op
  absl::Status Init(InitContext* ctx) { return absl::OkStatus(); }

  // Shape inference
  static absl::Status ShapeInference(ShapeInferenceContext* ctx) {
    // outpu0
    SH_RETURN_IF_ERROR(ctx->SetOutputShape(kOutput0, Shape({})));
    return absl::OkStatus();
  }

  // Runs the operation
  absl::Status Invoke(InvokeContext* ctx) {
    using std::int32_t;
    // input 0
    SH_ASSIGN_OR_RETURN(const auto input0_t, ctx->GetInput(kInput0));
    const auto in0 = input0_t->template AsScalar<AType>();
    // input 1
    SH_ASSIGN_OR_RETURN(const auto input1_t, ctx->GetInput(kInput1));
    const auto in1 = input1_t->template AsScalar<BType>();
    // output 0
    SH_ASSIGN_OR_RETURN(auto output0_t, ctx->GetOutput(kOutput0, Shape({})));
    auto& out0 = output0_t->template AsScalar<float>();
    out0 = in0 + in1;
    return absl::OkStatus();
  }
};

}  // namespace shim
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SHIM_TEST_OP_TMPL_OP_H_
