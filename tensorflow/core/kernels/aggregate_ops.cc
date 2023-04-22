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

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/aggregate_ops.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/aggregate_ops_cpu.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class AddNOp : public OpKernel {
 public:
  explicit AddNOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
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

template <typename Device>
class AddNOp<Device, Variant> : public OpKernel {
 public:
  explicit AddNOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
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

    // Step 2: Sum input variants in a tree-like structure using
    //   BinaryOpVariants(ADD_VARIANT_BINARY_OP, ...)
    //   For the output create a default-constructed variant object.
    //
    // Pairwise summation provides better numerical precision by
    // reducing round-off error:
    //
    //   https://en.wikipedia.org/wiki/Pairwise_summation
    //
    // These two vectors are used to store and mark intermediate sums.
    gtl::InlinedVector<bool, 4> temp_filled(num, false);
    gtl::InlinedVector<Variant, 4> temp(num);

    // Tree-based summation.
    int skip = 1;
    int n = num;
    while (skip < n) {
      int i = skip;
      while (i < n) {
        // TODO(ebrevdo, rmlarsen): Parallelize the pairwise summations in the
        // inner loop if the variants are "large".

        // x[i - skip] += x[i]
        OP_REQUIRES_OK(ctx,
                       AddVariantTo(ctx, i - skip, i, &temp, &temp_filled));
        // We won't use this index again, recover its memory.
        temp[i].clear();
        i += 2 * skip;
      }
      if (i == n) {
        // x[0] += x[i - skip]
        OP_REQUIRES_OK(ctx,
                       AddVariantTo(ctx, 0, i - skip, &temp, &temp_filled));
        // We won't use this index again, recover its memory.
        temp[i - skip].clear();
        n -= skip;
      }
      skip *= 2;
    }

    Tensor out(cpu_allocator(), DT_VARIANT, TensorShape({}));
    out.scalar<Variant>()() = std::move(temp[0]);
    ctx->set_output(0, out);
  }

 private:
  // AddVariantTo efficiently performs:
  //    temp[lhs_ix] <- array(lhs_ix) + array(rhs_ix)
  // where array(ix) := (temp_filled[ix]
  //                     ? temp[ix]
  //                     : ctx->input(ix).scalar<Variant>()())
  // This reduces (possibly expensive) copying of Variants from
  // the inputs into temp at the lowest levels of the summation tree.
  static inline Status AddVariantTo(OpKernelContext* ctx, const int lhs_ix,
                                    const int rhs_ix,
                                    gtl::InlinedVector<Variant, 4>* temp,
                                    gtl::InlinedVector<bool, 4>* temp_filled) {
    Variant tmp;
    if (temp_filled->at(lhs_ix)) tmp = std::move(temp->at(lhs_ix));
    const Variant& a = temp_filled->at(lhs_ix)
                           ? tmp
                           : ctx->input(lhs_ix).template scalar<Variant>()();
    const Variant& b = temp_filled->at(rhs_ix)
                           ? temp->at(rhs_ix)
                           : ctx->input(rhs_ix).template scalar<Variant>()();
    Variant* c = &temp->at(lhs_ix);
    TF_RETURN_IF_ERROR(
        BinaryOpVariants<Device>(ctx, ADD_VARIANT_BINARY_OP, a, b, c));
    temp_filled->at(lhs_ix) = true;
    return Status::OK();
  }
};

#define REGISTER_ADDN(type, dev)                                   \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("AddN").Device(DEVICE_##dev).TypeConstraint<type>("T"), \
      AddNOp<dev##Device, type>)

#define REGISTER_ADDN_CPU(type) REGISTER_ADDN(type, CPU)

TF_CALL_NUMBER_TYPES(REGISTER_ADDN_CPU);
REGISTER_ADDN_CPU(Variant);

#undef REGISTER_ADDN_CPU

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#define REGISTER_ADDN_GPU(type) REGISTER_ADDN(type, GPU)
TF_CALL_int64(REGISTER_ADDN_GPU);
TF_CALL_uint32(REGISTER_ADDN_GPU);
TF_CALL_variant(REGISTER_ADDN_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_ADDN_GPU);
TF_CALL_COMPLEX_TYPES(REGISTER_ADDN_GPU);
#undef REGISTER_ADDN_GPU

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("AddN")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("inputs")
                            .HostMemory("sum"),
                        AddNOp<CPUDevice, int32>);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef REGISTER_ADDN

}  // namespace tensorflow
