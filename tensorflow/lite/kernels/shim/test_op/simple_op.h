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

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"

namespace tflite {
namespace shim {

// A simple operation for demonstration and testing purposes.
// See the kDoc member for documentation.

template <Runtime Rt>
class SimpleOp : public OpKernelShim<SimpleOp, Rt> {
 protected:
  enum Inputs { kInput0 = 0, kInput1 };
  enum Outputs { kOutput0 = 0, kOutput1, kOutput2, kOutput3 };
  int64_t output1_size_;
  std::string output2_suffix_;
  int64_t n_;
  static constexpr int kOutput0Size = 5;
  static constexpr char kOutput1SizeAttr[] = "output1_size";

 public:
  using typename OpKernelShim<SimpleOp, Rt>::InitContext;
  using typename OpKernelShim<SimpleOp, Rt>::InvokeContext;
  using typename OpKernelShim<SimpleOp, Rt>::ShapeInferenceContext;

  SimpleOp() = default;
  static constexpr char kOpName[] = "SimpleOperation";
  static constexpr char kDoc[] = R"doc(
Description:
  Simple example op for testing and demonstration purposes.

Attrs
  output1_size: int - the size of the second output
  output2_suffix: string - the string value to be appended to the end of out2
  N: int - the number of tensors for the second input and last output
Inputs
  in0: str, shape=[] - A scalar input
  in1: int64, list<shape=?> - A list of tensors as input
Outputs
  out0: int, shape=[5] - first output
  out1: float, shape=[?] - second output
  out2: string, shape=[?] - third output
  out3: int64, list<shape=?> - fourth output that is in1 but incremented.
)doc";

  static const char* OpName() { return kOpName; }
  static const char* Doc() { return kDoc; }

  // Attributes declaration (syntax: https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Attrs() {
    return {absl::StrCat(kOutput1SizeAttr, ": int"), "output2_suffix: string",
            "N: int >= 0"};
  }
  // Input tensors declaration (syntax:
  // https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Inputs() {
    return {"in0: string", "in1: N*int64"};
  }
  // Output tensors declaration (syntax:
  // https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Outputs() {
    return {"out0: int32", "out1: float", "out2: string", "out3: N*int64"};
  }

  // Initializes the op
  absl::Status Init(InitContext* ctx) {
    SH_RETURN_IF_ERROR(ctx->GetAttr(kOutput1SizeAttr, &output1_size_));
    if (output1_size_ < 1) {
      return absl::InternalError(
          absl::StrCat(kOutput1SizeAttr, " should be >= 1"));
    }
    SH_RETURN_IF_ERROR(ctx->GetAttr("N", &n_));
    absl::string_view output2_suffix;
    SH_RETURN_IF_ERROR(ctx->GetAttr("output2_suffix", &output2_suffix));
    output2_suffix_ = std::string(output2_suffix);
    return absl::OkStatus();
  }

  // Runs the operation
  absl::Status Invoke(InvokeContext* ctx) {
    using std::int32_t;
    // read input
    SH_ASSIGN_OR_RETURN(const auto input_t, ctx->GetInput(kInput0));
    const auto input_str = input_t->template AsScalar<::tensorflow::tstring>();
    // output0 whose size is static
    SH_ASSIGN_OR_RETURN(auto output0_t,
                        ctx->GetOutput(kOutput0, Shape({kOutput0Size})));
    auto output0 = output0_t->template As<int32_t, 1>();
    for (int i = 0; i < output0.Dim(0); ++i) output0(i) = i;
    // output1 whose size is based on the attr
    SH_ASSIGN_OR_RETURN(
        auto output1_t,
        ctx->GetOutput(kOutput1, Shape({static_cast<int>(output1_size_)})));
    auto output1 = output1_t->template As<float, 1>();
    for (int i = 0; i < output1.Dim(0); ++i) output1(i) = 0.5 * i;
    // output2 whose size is based on input
    const int output2_size = input_str.length() + 1;
    SH_ASSIGN_OR_RETURN(auto output2_t,
                        ctx->GetOutput(kOutput2, Shape({output2_size})));
    auto output2 = output2_t->template As<tensorflow::tstring, 1>();
    for (int i = 0; i < output2.Dim(0) - 1; ++i) output2(i) = std::to_string(i);
    output2(output2.Dim(0) - 1) = output2_suffix_;
    // output3 which is a list of length N
    // The values in output3 are element wise equal to input2 + 1.
    if (ctx->NumInputs() < kInput1 + n_) {
      return absl::InternalError(absl::StrCat(
          "out of bounds: num_inputs=", ctx->NumInputs(), " N=", n_));
    }
    if (ctx->NumOutputs() < kOutput3 + n_) {
      return absl::InternalError(absl::StrCat(
          "out of bounds: num_outputs=", ctx->NumOutputs(), " N=", n_));
    }
    for (int i = 0; i < n_; ++i) {
      SH_ASSIGN_OR_RETURN(const auto input_t, ctx->GetInput(kInput1 + i));
      Shape output_shape(input_t->Shape());
      SH_ASSIGN_OR_RETURN(auto output_t,
                          ctx->GetOutput(kOutput3 + i, output_shape));
      const auto input_data = input_t->template Data<int64_t>();
      auto output_buffer = output_t->template Data<int64_t>().data();
      std::copy(input_data.begin(), input_data.end(), output_buffer);
      // Increment the values of the output
      for (auto& v : output_t->template Data<int64_t>()) ++v;
    }
    return absl::OkStatus();
  }

  // Shape inference
  static absl::Status ShapeInference(ShapeInferenceContext* ctx) {
    // outpu0
    SH_RETURN_IF_ERROR(ctx->SetOutputShape(kOutput0, Shape({kOutput0Size})));
    // output1
    SH_RETURN_IF_ERROR(
        ctx->SetOutputShape(kOutput1, Shape({Shape::kUnknownDim})));
    // output2
    const auto input_t_or = ctx->GetInputTensor(kInput0);
    Shape output2_shape;
    if (input_t_or.ok()) {
      const auto& input_t = input_t_or.value();
      const auto input_str =
          input_t->template AsScalar<::tensorflow::tstring>();
      output2_shape = Shape({static_cast<int>(input_str.length() + 1)});
    } else {
      output2_shape = Shape({Shape::kUnknownDim});
    }
    SH_RETURN_IF_ERROR(ctx->SetOutputShape(kOutput2, output2_shape));
    // output3
    for (int i = kOutput3; i < ctx->NumOutputs(); ++i) {
      SH_RETURN_IF_ERROR(ctx->SetOutputShape(kOutput3, Shape()));
    }
    int64_t n;
    SH_RETURN_IF_ERROR(ctx->GetAttr("N", &n));
    if (n + 1 != ctx->NumInputs()) {
      return absl::InternalError(absl::StrCat("n + 1 != num_inputs: ", n + 1,
                                              " != ", ctx->NumInputs()));
    }
    if (n + 3 != ctx->NumOutputs()) {
      return absl::InternalError(absl::StrCat("n + 1 != num_inputs: ", n + 1,
                                              " != ", ctx->NumOutputs()));
    }
    return absl::OkStatus();
  }
};


}  // namespace shim
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SHIM_TEST_OP_SIMPLE_OP_H_
