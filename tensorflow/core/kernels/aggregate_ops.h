/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_AGGREGATE_OPS_H_
#define TENSORFLOW_CORE_KERNELS_AGGREGATE_OPS_H_

#include <numeric>

#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace tensorflow {
namespace functor {

// Functor definitions for Aggregate ops, must be compilable by nvcc.
template <typename Device, typename T>
struct Add2Functor {
  void operator()(const Device& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2);
};

template <typename Device, typename T>
struct Add2EigenImpl {
  static void Compute(const Device& d, typename TTypes<T>::Flat out,
                      typename TTypes<T>::ConstFlat in1,
                      typename TTypes<T>::ConstFlat in2) {
    out.device(d) = in1 + in2;
  }
};

template <typename Device, typename T>
struct Add3Functor {
  void operator()(const Device& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3);
};

template <typename Device, typename T>
struct Add3EigenImpl {
  static void Compute(const Device& d, typename TTypes<T>::Flat out,
                      typename TTypes<T>::ConstFlat in1,
                      typename TTypes<T>::ConstFlat in2,
                      typename TTypes<T>::ConstFlat in3) {
    out.device(d) = in1 + in2 + in3;
  }
};

template <typename Device, typename T>
struct Add4Functor {
  void operator()(const Device& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3,
                  typename TTypes<T>::ConstFlat in4);
};

template <typename Device, typename T>
struct Add4EigenImpl {
  static void Compute(const Device& d, typename TTypes<T>::Flat out,
                      typename TTypes<T>::ConstFlat in1,
                      typename TTypes<T>::ConstFlat in2,
                      typename TTypes<T>::ConstFlat in3,
                      typename TTypes<T>::ConstFlat in4) {
    out.device(d) = in1 + in2 + in3 + in4;
  }
};

template <typename Device, typename T>
struct Add5Functor {
  void operator()(const Device& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3,
                  typename TTypes<T>::ConstFlat in4,
                  typename TTypes<T>::ConstFlat in5);
};

template <typename Device, typename T>
struct Add5EigenImpl {
  static void Compute(const Device& d, typename TTypes<T>::Flat out,
                      typename TTypes<T>::ConstFlat in1,
                      typename TTypes<T>::ConstFlat in2,
                      typename TTypes<T>::ConstFlat in3,
                      typename TTypes<T>::ConstFlat in4,
                      typename TTypes<T>::ConstFlat in5) {
    out.device(d) = in1 + in2 + in3 + in4 + in5;
  }
};

template <typename Device, typename T>
struct Add6Functor {
  void operator()(const Device& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3,
                  typename TTypes<T>::ConstFlat in4,
                  typename TTypes<T>::ConstFlat in5,
                  typename TTypes<T>::ConstFlat in6);
};

template <typename Device, typename T>
struct Add6EigenImpl {
  static void Compute(const Device& d, typename TTypes<T>::Flat out,
                      typename TTypes<T>::ConstFlat in1,
                      typename TTypes<T>::ConstFlat in2,
                      typename TTypes<T>::ConstFlat in3,
                      typename TTypes<T>::ConstFlat in4,
                      typename TTypes<T>::ConstFlat in5,
                      typename TTypes<T>::ConstFlat in6) {
    out.device(d) = in1 + in2 + in3 + in4 + in5 + in6;
  }
};

template <typename Device, typename T>
struct Add7Functor {
  void operator()(const Device& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3,
                  typename TTypes<T>::ConstFlat in4,
                  typename TTypes<T>::ConstFlat in5,
                  typename TTypes<T>::ConstFlat in6,
                  typename TTypes<T>::ConstFlat in7);
};

template <typename Device, typename T>
struct Add7EigenImpl {
  static void Compute(const Device& d, typename TTypes<T>::Flat out,
                      typename TTypes<T>::ConstFlat in1,
                      typename TTypes<T>::ConstFlat in2,
                      typename TTypes<T>::ConstFlat in3,
                      typename TTypes<T>::ConstFlat in4,
                      typename TTypes<T>::ConstFlat in5,
                      typename TTypes<T>::ConstFlat in6,
                      typename TTypes<T>::ConstFlat in7) {
    out.device(d) = in1 + in2 + in3 + in4 + in5 + in6 + in7;
  }
};

template <typename Device, typename T>
struct Add8Functor {
  void operator()(
      const Device& d, typename TTypes<T>::Flat out,
      typename TTypes<T>::ConstFlat in1, typename TTypes<T>::ConstFlat in2,
      typename TTypes<T>::ConstFlat in3, typename TTypes<T>::ConstFlat in4,
      typename TTypes<T>::ConstFlat in5, typename TTypes<T>::ConstFlat in6,
      typename TTypes<T>::ConstFlat in7, typename TTypes<T>::ConstFlat in8);
};

template <typename Device, typename T>
struct Add8EigenImpl {
  static void Compute(
      const Device& d, typename TTypes<T>::Flat out,
      typename TTypes<T>::ConstFlat in1, typename TTypes<T>::ConstFlat in2,
      typename TTypes<T>::ConstFlat in3, typename TTypes<T>::ConstFlat in4,
      typename TTypes<T>::ConstFlat in5, typename TTypes<T>::ConstFlat in6,
      typename TTypes<T>::ConstFlat in7, typename TTypes<T>::ConstFlat in8) {
    out.device(d) = in1 + in2 + in3 + in4 + in5 + in6 + in7 + in8;
  }
};

// Add8p is like Add8 except the underlying implementation should +=
// rather than assign to the output.
template <typename Device, typename T>
struct Add8pFunctor {
  void operator()(
      const Device& d, typename TTypes<T>::Flat out,
      typename TTypes<T>::ConstFlat in1, typename TTypes<T>::ConstFlat in2,
      typename TTypes<T>::ConstFlat in3, typename TTypes<T>::ConstFlat in4,
      typename TTypes<T>::ConstFlat in5, typename TTypes<T>::ConstFlat in6,
      typename TTypes<T>::ConstFlat in7, typename TTypes<T>::ConstFlat in8);
};

template <typename Device, typename T>
struct Add8pEigenImpl {
  static void Compute(
      const Device& d, typename TTypes<T>::Flat out,
      typename TTypes<T>::ConstFlat in1, typename TTypes<T>::ConstFlat in2,
      typename TTypes<T>::ConstFlat in3, typename TTypes<T>::ConstFlat in4,
      typename TTypes<T>::ConstFlat in5, typename TTypes<T>::ConstFlat in6,
      typename TTypes<T>::ConstFlat in7, typename TTypes<T>::ConstFlat in8) {
    out.device(d) += in1 + in2 + in3 + in4 + in5 + in6 + in7 + in8;
  }
};

template <typename Device, typename T>
struct Add9Functor {
  void operator()(
      const Device& d, typename TTypes<T>::Flat out,
      typename TTypes<T>::ConstFlat in1, typename TTypes<T>::ConstFlat in2,
      typename TTypes<T>::ConstFlat in3, typename TTypes<T>::ConstFlat in4,
      typename TTypes<T>::ConstFlat in5, typename TTypes<T>::ConstFlat in6,
      typename TTypes<T>::ConstFlat in7, typename TTypes<T>::ConstFlat in8,
      typename TTypes<T>::ConstFlat in9);
};

template <typename Device, typename T>
struct Add9EigenImpl {
  static void Compute(
      const Device& d, typename TTypes<T>::Flat out,
      typename TTypes<T>::ConstFlat in1, typename TTypes<T>::ConstFlat in2,
      typename TTypes<T>::ConstFlat in3, typename TTypes<T>::ConstFlat in4,
      typename TTypes<T>::ConstFlat in5, typename TTypes<T>::ConstFlat in6,
      typename TTypes<T>::ConstFlat in7, typename TTypes<T>::ConstFlat in8,
      typename TTypes<T>::ConstFlat in9) {
    out.device(d) = in1 + in2 + in3 + in4 + in5 + in6 + in7 + in8 + in9;
  }
};

}  // namespace functor

template <typename Device, typename T, class OpKernelT,
          class OpKernelConstructionT, class OpKernelContextT>
class AddNOp : public OpKernelT {
 public:
  explicit AddNOp(OpKernelConstructionT* context) : OpKernelT(context) {}

  void Compute(OpKernelContextT* ctx) override {
    if (!ctx->ValidateInputsAreSameShape(this)) return;

    const Tensor& input0 = ctx->input(0);
    const int num = ctx->num_inputs();

    if (num == 1) {
      ctx->set_output(0, input0);
      return;
    }

    // Try to forward and accumulate the result in one of the input buffers.
    int reused_input = -1;
    gtl::InlinedVector<int, 8> input_indices(num);
    std::iota(input_indices.begin(), input_indices.end(), 0);
    Tensor* output = nullptr;
    for (int input_idx = 0; input_idx < num; ++input_idx) {
      if (ctx->forward_input_to_output_with_shape(input_idx, 0, input0.shape(),
                                                  &output)) {
        reused_input = input_idx;
        break;
      }
    }
    if (reused_input == -1) {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input0.shape(), &output));
    } else if (reused_input > 0) {
      // Move the forwarded buffer to the front so we don't double count
      // anything if there are more than 8 inputs.
      input_indices[0] = reused_input;
      input_indices[reused_input] = 0;
    }
    auto To = output->flat<T>();

#define I(IDX) ctx->input(input_indices[IDX]).template flat<T>()

#if defined(__ANDROID_TYPES_SLIM__)
    // On Android by default,we only support additions of two arguments, so we
    // can reduce the number of template instantiations.
    OP_REQUIRES(ctx, num == 2,
                errors::InvalidArgument("Only additions of two arguments "
                                        "supported. Num inputs: ",
                                        num));
    functor::Add2Functor<Device, T> functor2;
    functor2(ctx->template eigen_device<Device>(), To, I(0), I(1));
#else
    static const int kWidth = 8;
    int r = num % kWidth;

    switch (r) {
      case 2: {
        functor::Add2Functor<Device, T> functor2;
        functor2(ctx->template eigen_device<Device>(), To, I(0), I(1));
        break;
      }
      case 3: {
        functor::Add3Functor<Device, T> functor3;
        functor3(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2));
        break;
      }
      case 4: {
        functor::Add4Functor<Device, T> functor4;
        functor4(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2),
                 I(3));
        break;
      }
      case 5: {
        functor::Add5Functor<Device, T> functor5;
        functor5(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2),
                 I(3), I(4));
        break;
      }
      case 6: {
        functor::Add6Functor<Device, T> functor6;
        functor6(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2),
                 I(3), I(4), I(5));
        break;
      }
      case 7: {
        functor::Add7Functor<Device, T> functor7;
        functor7(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2),
                 I(3), I(4), I(5), I(6));
        break;
      }
      case 0: {
        functor::Add8Functor<Device, T> functor8;
        functor8(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2),
                 I(3), I(4), I(5), I(6), I(7));
        r = 8;
        break;
      }
      case 1: {
        functor::Add9Functor<Device, T> functor9;
        functor9(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2),
                 I(3), I(4), I(5), I(6), I(7), I(8));
        r = 9;
        break;
      }
    }

    for (; r < num; r += kWidth) {
      functor::Add8pFunctor<Device, T> functor8p;
      functor8p(ctx->template eigen_device<Device>(), To, I(r), I(r + 1),
                I(r + 2), I(r + 3), I(r + 4), I(r + 5), I(r + 6), I(r + 7));
    }
#endif  // defined(__ANDROID_TYPES_SLIM__)

#undef I
  }
};

template <typename Device, class OpKernelT, class OpKernelConstructionT,
          class OpKernelContextT>
class AddNOp<Device, Variant, OpKernelT, OpKernelConstructionT,
             OpKernelContextT> : public OpKernelT {
 public:
  explicit AddNOp(OpKernelConstructionT* context) : OpKernelT(context) {}

  void Compute(OpKernelContextT* ctx) override {
    if (!ctx->ValidateInputsAreSameShape(this)) return;

    const Tensor& input0 = ctx->input(0);
    const int num = ctx->num_inputs();

    if (num == 1) {
      ctx->set_output(0, input0);
      return;
    }

    for (int i = 0; i < num; ++i) {
      // Step 1: ensure unary variants.
      OP_REQUIRES(
          ctx, ctx->input(i).dims() == 0,
          errors::InvalidArgument(
              "AddN of non-scalar Tensor with dtype=DT_VARIANT is not "
              "supported; inputs[",
              i, " has shape: ", ctx->input(i).shape().DebugString(), "."));
    }

    // Step 2: attempt to add using
    //   BinaryOpVariants(ADD_VARIANT_BINARY_OP, ...)
    //   For the output create a default-constructed variant object.
    // TODO(ebrevdo): Perform summation in a tree-structure.
    Tensor out(cpu_allocator(), DT_VARIANT, TensorShape({}));
    Variant* v_out = &(out.scalar<Variant>()());
    OP_REQUIRES_OK(ctx, BinaryOpVariants<Device>(
                            ctx, ADD_VARIANT_BINARY_OP,
                            ctx->input(0).template scalar<Variant>()(),
                            ctx->input(1).template scalar<Variant>()(), v_out));
    for (int i = 2; i < num; ++i) {
      const Variant tmp = std::move(*v_out);
      const Variant& inp = ctx->input(i).template scalar<Variant>()();
      OP_REQUIRES_OK(ctx, BinaryOpVariants<Device>(ctx, ADD_VARIANT_BINARY_OP,
                                                   inp, tmp, v_out));
    }
    ctx->set_output(0, out);
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_AGGREGATE_OPS_H_
