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
#ifndef TENSORFLOW_LITE_KERNELS_SHIM_TEST_OP_SIMPLE_OP_H_
#define TENSORFLOW_LITE_KERNELS_SHIM_TEST_OP_SIMPLE_OP_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow/lite/kernels/shim/tensor_view.h"

namespace tflite {
namespace shim {

// A simple operation for demonstration and testing purposes.
// See the kDoc member for documentation.

template <Runtime Rt>
class SimpleOp : public OpKernelShim<SimpleOp, Rt> {
 protected:
  enum Inputs { kInput1 = 0 };
  enum Outputs { kOutput1 = 0, kOutput2, kOutput3 };
  static constexpr int kOutput1Size = 5;
  int64_t output2_size_;
  static const char kOutput2SizeAttr[];

 public:
  using typename OpKernelShim<SimpleOp, Rt>::InitContext;
  using typename OpKernelShim<SimpleOp, Rt>::InvokeContext;
  using typename OpKernelShim<SimpleOp, Rt>::ShapeInferenceContext;

  SimpleOp() = default;
  static const char kOpName[];
  static const char kDoc[];

  // Input tensors declaration (name, type, shape)
  static std::vector<TensorDeclaration> Inputs() {
    return {{"in1: string", {}}};
  }
  // Output tensors declaration (name, type, shape)
  static std::vector<TensorDeclaration> Outputs() {
    return {{"out1: int32", Shape({kOutput1Size})},
            {"out2: float", Shape({Shape::kUnknownDim})},
            {"out3: int32", Shape({Shape::kUnknownDim})}};
  }
  // Attributes declaration (name, type)
  static std::vector<std::string> Attrs() {
    return {absl::StrCat(kOutput2SizeAttr, ": int")};
  }

  // Parses a serialized config being passed in
  absl::Status Init(InitContext* ctx) {
    SH_RETURN_IF_ERROR(ctx->GetAttr(kOutput2SizeAttr, &output2_size_));
    if (output2_size_ < 1) {
      return absl::InternalError("output_size should be >= 1");
    }
    return absl::OkStatus();
  }

  // Runs the operation
  absl::Status Invoke(InvokeContext* ctx) {
    using std::int32_t;
    // read input
    SH_ASSIGN_OR_RETURN(const auto input_t, ctx->GetInput(kInput1));
    const auto input_str = input_t->template AsScalar<::tensorflow::tstring>();
    // output 1 whose size is static
    SH_ASSIGN_OR_RETURN(auto output1_t,
                        ctx->GetOutput(kOutput1, Shape({kOutput1Size})));
    auto output1 = output1_t->template As<int32_t, 1>();
    for (int i = 0; i < output1.Dim(0); ++i) output1(i) = i;
    // output 2 whose size is based on the attr
    SH_ASSIGN_OR_RETURN(
        auto output2_t,
        ctx->GetOutput(kOutput2, Shape({static_cast<int>(output2_size_)})));
    auto output2 = output2_t->template As<float, 1>();
    for (int i = 0; i < output2.Dim(0); ++i) output2(i) = 0.5 * i;
    // output 3 whose size is based on input
    const int output3_size = input_str.length();
    SH_ASSIGN_OR_RETURN(auto output3_t,
                        ctx->GetOutput(kOutput3, Shape({output3_size})));
    auto output3 = output3_t->template As<int32_t, 1>();
    for (int i = 0; i < output3.Dim(0); ++i) output3(i) = i;
    return absl::OkStatus();
  }

  // Shape inference
  static absl::Status ShapeInference(ShapeInferenceContext* ctx) {
    SH_RETURN_IF_ERROR(ctx->SetOutputShape(kOutput1, Shape({kOutput1Size})));
    SH_RETURN_IF_ERROR(
        ctx->SetOutputShape(kOutput2, Shape({Shape::kUnknownDim})));
    SH_RETURN_IF_ERROR(
        ctx->SetOutputShape(kOutput3, Shape({Shape::kUnknownDim})));
    return absl::OkStatus();
  }
};

// Static member definitions.
// These can be inlined once the toolchain is bumped up to C++17

template <Runtime Rt>
const char SimpleOp<Rt>::kOutput2SizeAttr[] = "output2_size";

template <Runtime Rt>
const char SimpleOp<Rt>::kOpName[] = "SimpleOperation";

template <Runtime Rt>
const char SimpleOp<Rt>::kDoc[] = R"doc(
Description:
  Simple example op for testing and demonstration purposes.

Inputs
  in1: str, shape=[] - input as string
Outputs
  out1: int, shape=[5] - first output
  out2: float, shape=[?] - second output
  out3: int, shape[?] - third output
Attrs
  output2_size: int - the size of the second output
)doc";

}  // namespace shim
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SHIM_TEST_OP_SIMPLE_OP_H_
