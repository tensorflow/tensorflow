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

#include <cstdint>

#include "absl/log/log.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

// Please use the appropriate namespace for your project
namespace tensorflow {
namespace custom_op_examples {

using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Tensor;
using ::tensorflow::errors::Internal;
using ::tensorflow::errors::InvalidArgument;

// Multiple types for the values inside two of the input tensors
// (e.g. int32, float) are supported by using a template where the type is T.
template <typename T>
class MultiplexDenseOp : public OpKernel {
 public:
  explicit MultiplexDenseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &num_cond_a_));
  }

  MultiplexDenseOp(const MultiplexDenseOp& other) = delete;
  MultiplexDenseOp& operator=(const MultiplexDenseOp& other) = delete;
  ~MultiplexDenseOp() override = default;

  void Compute(OpKernelContext* ctx) override {
    // Optional error checking: cond and a_values are lists of N, so there are
    // a total of 2N+1 inputs. Check that the  number of inputs and the
    // `N` Attr is consistent.
    const int64_t expected_inputs = 2 * num_cond_a_ + 1;
    OP_REQUIRES(ctx, expected_inputs == ctx->num_inputs(),
                Internal("expected_inputs != num_inputs(): ", expected_inputs,
                         " != ", ctx->num_inputs()));
    VLOG(1) << "N " << num_cond_a_;

    const auto& first_cond_tensor = ctx->input(0);
    const auto& first_a_values_tensor = ctx->input(num_cond_a_);
    const auto& b_values_tensor = ctx->input(2 * num_cond_a_);

    // Allow any shape, but require that a_values, b_values, and cond all
    // have the same shape.
    // Note that ::tensorflow::TensorShapeUtils has some useful functions
    // for checking shapes.
    for (int64_t i = 0; i < num_cond_a_; i++) {
      const auto& cond_tensor_i = ctx->input(i);
      const auto& a_values_tensor_i = ctx->input(num_cond_a_ + i);
      OP_REQUIRES(
          ctx, a_values_tensor_i.shape() == b_values_tensor.shape(),
          InvalidArgument(
              "a_values[", i,
              "] and b_values must have the same shape. "
              "a_values[",
              i, "] shape: ", a_values_tensor_i.DebugString(),
              " b_values shape: ", b_values_tensor.shape().DebugString()));
      OP_REQUIRES(
          ctx, cond_tensor_i.shape() == b_values_tensor.shape(),
          InvalidArgument(
              "cond_values[", i,
              "] and b_valuesmust have the same shape. "
              "cond_values[",
              i, "] shape: ", first_a_values_tensor.shape().DebugString(),
              " b_values shape: ", first_cond_tensor.shape().DebugString()));
    }

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, b_values_tensor.shape(), &output_tensor));
    auto output = output_tensor->template flat<T>();

    const auto b_values = b_values_tensor.template flat<T>();
    // np.select style behavior, `cond` and `a_values` are lists of tensors.
    // Also works for the np.where style case where there is only one `cond`
    // and one `a_values` tensor.
    const int64_t N = first_a_values_tensor.NumElements();
    for (int64_t i = 0; i < N; i++) {
      bool flag = false;
      for (int64_t list_index = 0; list_index < num_cond_a_; list_index++) {
        const auto& cond_tensor = ctx->input(list_index);
        const auto& a_values_tensor = ctx->input(num_cond_a_ + list_index);
        const auto cond = cond_tensor.template flat<bool>();
        const auto a_values = a_values_tensor.template flat<T>();
        if (cond(i)) {
          output(i) = a_values(i);
          flag = true;
          VLOG(1) << "A " << list_index << " for " << i;
          break;
        }
      }
      if (!flag) {
        output(i) = b_values(i);
        VLOG(1) << "B for " << i;
      }
    }
  }

 private:
  int64_t num_cond_a_;  // the number of `cond` and `a` input tensors
};

// The "Name" used by REGISTER_KERNEL_BUILDER is defined by REGISTER_OP,
// see multiplex_4_op.cc.
// To support tensors containing different types (e.g. int32, float), one
// kernel per type is registered and is templatized by the "T" attr value.
// The TF_CALL_ALL_TYPES macro registers the op for all types appropriate for
// the target platform. See go/tf-custom-ops-guide
#define REGISTER_KERNELS(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("Examples>MultiplexDense")       \
                              .Device(::tensorflow::DEVICE_CPU) \
                              .TypeConstraint<type>("T"),       \
                          MultiplexDenseOp<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

}  // namespace custom_op_examples
}  // namespace tensorflow
