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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

// Please use the appropriate namespace for your project
namespace tensorflow {
namespace custom_op_examples {

using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Tensor;
using ::tensorflow::errors::InvalidArgument;

// Multiple types for the values inside two of the input tensors
// (e.g. int32, float) are supported by using a template where the type is T.
template <typename T>
class MultiplexDenseOp : public OpKernel {
 public:
  explicit MultiplexDenseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  MultiplexDenseOp(const MultiplexDenseOp& other) = delete;
  MultiplexDenseOp& operator=(const MultiplexDenseOp& other) = delete;
  ~MultiplexDenseOp() override = default;

  void Compute(OpKernelContext* ctx) override {
    const auto& cond_tensor = ctx->input(0);
    const auto& a_values_tensor = ctx->input(1);
    const auto& b_values_tensor = ctx->input(2);

    // Allow any shape, but require that a_values, b_values, and cond all
    // have the same shape.
    // Note that ::tensorflow::TensorShapeUtils has some useful functions
    // for checking shapes.
    OP_REQUIRES(ctx, a_values_tensor.shape() == b_values_tensor.shape(),
                InvalidArgument(
                    "a_values and b_values must have the same shape. "
                    "a_values shape: ",
                    a_values_tensor.shape().DebugString(), " b_values shape: ",
                    b_values_tensor.shape().DebugString()));
    OP_REQUIRES(
        ctx, a_values_tensor.shape() == cond_tensor.shape(),
        InvalidArgument("a_values and cond must have the same shape. "
                        "a_values shape: ",
                        a_values_tensor.shape().DebugString(),
                        " cond shape: ", cond_tensor.shape().DebugString()));

    const auto a_values = a_values_tensor.flat<T>();
    const auto b_values = b_values_tensor.flat<T>();
    const auto cond = cond_tensor.flat<bool>();

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, a_values_tensor.shape(), &output_tensor));
    auto output = output_tensor->template flat<T>();
    const int64_t N = a_values_tensor.NumElements();

    // Here is an example of processing tensors in a simple loop directly
    // without relying on any libraries. For intensive math operations, it is
    // a good practice to use libraries such as Eigen that support
    // tensors when possible, e.g. "output = cond.select(a_values, b_values);"
    // Eigen supports chunking into blocks and multi-threading.
    // See
    // https://eigen.tuxfamily.org/dox/unsupported/eigen_tensors.html#title55
    for (int64_t i = 0; i < N; i++) {
      if (cond(i)) {
        output(i) = a_values(i);
      } else {
        output(i) = b_values(i);
      }
    }
  }
};

// The "Name" used by REGISTER_KERNEL_BUILDER is defined by REGISTER_OP,
// see multiplex_1_op.cc.
// To support tensors containing different types (e.g. int32, float), one
// kernel per type is registered and is templatized by the "T" attr value.
// The TF_CALL_ALL_TYPES macro registers the op for all types appropriate for
// the target platform. See go/tf-custom-ops-guide
#define REGISTER_KERNELS(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("Examples1>MultiplexDense")      \
                              .Device(::tensorflow::DEVICE_CPU) \
                              .TypeConstraint<type>("T"),       \
                          MultiplexDenseOp<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

}  // namespace custom_op_examples
}  // namespace tensorflow
