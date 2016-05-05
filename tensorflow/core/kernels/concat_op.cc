/* Copyright 2015 Google Inc. All Rights Reserved.

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

// See docs in ../ops/array_ops.cc.

#include <limits>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// --------------------------------------------------------------------------
template <typename Device, typename T>
class ConcatOp : public OpKernel {
 public:
  typedef std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>
      ConstMatrixVector;

  explicit ConcatOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    const Tensor* concat_dim_tensor;
    OP_REQUIRES_OK(c, c->input("concat_dim", &concat_dim_tensor));
    OP_REQUIRES(
        c, IsLegacyScalar(concat_dim_tensor->shape()),
        errors::InvalidArgument(
            "Concat dim tensor should be a scalar integer, but got shape ",
            concat_dim_tensor->shape().DebugString()));
    const int32 concat_dim =
        internal::SubtleMustCopy(concat_dim_tensor->scalar<int32>()());
    OpInputList values;
    OP_REQUIRES_OK(c, c->input_list("values", &values));
    const int N = values.size();
    const int input_dims = values[0].dims();
    const TensorShape& input_shape = values[0].shape();
    OP_REQUIRES(
        c, FastBoundsCheck(concat_dim, input_dims) ||
               (allow_legacy_scalars() && concat_dim == 0),
        errors::InvalidArgument(
            "ConcatOp : Expected concatenating dimensions in the range [", 0,
            ", ", input_dims, "), but got ", concat_dim));

    // Note that we reduce the concat of n-dimensional tensors into a two
    // dimensional concat. Assuming the dimensions of any input/output
    // tensor are {x0, x1,...,xn-1, y0, y1,...,ym-1}, where the concat is along
    // the dimension indicated with size y0, we flatten it to {x, y}, where y =
    // Prod_i(yi) and x = ((n > 0) ? Prod_i(xi) : 1).
    ConstMatrixVector inputs_flat;
    inputs_flat.reserve(N);
    int64 inputs_flat_dim0 = 1;
    for (int d = 0; d < concat_dim; ++d) {
      inputs_flat_dim0 *= input_shape.dim_size(d);
    }
    int output_concat_dim = 0;
    const bool input_is_scalar = IsLegacyScalar(input_shape);
    for (int i = 0; i < N; ++i) {
      const auto in = values[i];
      const bool in_is_scalar = IsLegacyScalar(in.shape());
      OP_REQUIRES(
          c, in.dims() == input_dims || (input_is_scalar && in_is_scalar),
          errors::InvalidArgument(
              "ConcatOp : Ranks of all input tensors should match: shape[0] = ",
              input_shape.DebugString(), " vs. shape[", i, "] = ",
              in.shape().DebugString()));
      for (int j = 0; j < input_dims; ++j) {
        if (j == concat_dim) {
          continue;
        }
        OP_REQUIRES(
            c, in.dim_size(j) == input_shape.dim_size(j),
            errors::InvalidArgument(
                "ConcatOp : Dimensions of inputs should match: shape[0] = ",
                input_shape.DebugString(), " vs. shape[", i, "] = ",
                in.shape().DebugString()));
      }
      if (in.NumElements() > 0) {
        int64 inputs_flat_dim1 = in.NumElements() / inputs_flat_dim0;
        inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
            in.shaped<T, 2>({inputs_flat_dim0, inputs_flat_dim1})));
      }
      // TODO(irving): Remove check once !allow_legacy_scalars().
      output_concat_dim += in.dims() > 0 ? in.dim_size(concat_dim) : 1;
    }

    TensorShape output_shape(input_shape);
    // TODO(irving): Remove rank 0 case once !allow_legacy_scalars().
    if (output_shape.dims() == 0) {
      output_shape.AddDim(output_concat_dim);
    } else {
      output_shape.set_dim(concat_dim, output_concat_dim);
    }
    Tensor* output = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output));
    if (output->NumElements() > 0) {
      int64 output_dim1 = output->NumElements() / inputs_flat_dim0;
      auto output_flat = output->shaped<T, 2>({inputs_flat_dim0, output_dim1});
      if (std::is_same<Device, GPUDevice>::value) {
        // Switching indexing to int64 might cause performance issues.
        // Hence, we keep int32 indexing in the GPU kernel unless we need to
        // switch to int64.
        if (output->NumElements() < std::numeric_limits<int32>::max()) {
          ConcatGPU32<T>(c->eigen_gpu_device(), inputs_flat, &output_flat);
        } else {
          ConcatGPU64<T>(c->eigen_gpu_device(), inputs_flat, &output_flat);
        }
      } else {
        ConcatCPU<T>(c->device(), inputs_flat, &output_flat);
      }
    }
  }
};

#define REGISTER_CONCAT(type)                            \
  REGISTER_KERNEL_BUILDER(Name("Concat")                 \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("concat_dim"), \
                          ConcatOp<CPUDevice, type>)

TF_CALL_ALL_TYPES(REGISTER_CONCAT);
REGISTER_CONCAT(quint8);
REGISTER_CONCAT(qint8);
REGISTER_CONCAT(quint16);
REGISTER_CONCAT(qint16);
REGISTER_CONCAT(qint32);
REGISTER_CONCAT(bfloat16);

#undef REGISTER_CONCAT

#if GOOGLE_CUDA

#define REGISTER_GPU(type)                               \
  REGISTER_KERNEL_BUILDER(Name("Concat")                 \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("concat_dim"), \
                          ConcatOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
REGISTER_GPU(bfloat16);
#undef REGISTER_GPU

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Concat")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("concat_dim")
                            .HostMemory("values")
                            .HostMemory("output"),
                        ConcatOp<CPUDevice, int32>);

#endif  // GOOGLE_CUDA

class ConcatOffsetOp : public OpKernel {
 public:
  explicit ConcatOffsetOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& concat_dim = ctx->input(0);
    OP_REQUIRES(
        ctx, IsLegacyScalar(concat_dim.shape()),
        errors::InvalidArgument(
            "Concat dim tensor should be a scalar integer, but got shape ",
            concat_dim.shape().DebugString()));
    for (int i = 1; i < ctx->num_inputs(); ++i) {
      const Tensor& inp = ctx->input(i);
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(inp.shape()),
                  errors::InvalidArgument("input ", i,
                                          " should be a vector, but got shape ",
                                          inp.shape().DebugString()));
    }
    // Suppose a Concat() op needs to Concatenate N tensors, each of
    // which has the same number of dimensions.  Their shapes match
    // except the concat dimension.
    //
    // E.g., say, we want to concatenate 3 tensors in the 2nd
    // dimension, and their shapes are:
    //
    //  [2, 2, 5, 7]
    //  [2, 3, 5, 7]
    //  [2, 4, 5, 7]
    //
    // Here, N=3, cdim=1, dims=4. The concatenated tensor has shape
    // [2,9,5,7]. We will compute the cumulative sum along the 2nd
    // dimension to figure out each input's offset in the concatenated
    // output:
    //  [0, 0, 0, 0]
    //  [0, 2, 0, 0]
    //  [0, 5, 0, 0]
    const int32 N = ctx->num_inputs() - 1;
    const Tensor& inp0 = ctx->input(1);
    auto inp0_vec = inp0.vec<int32>();
    const int64 cdim = internal::SubtleMustCopy(concat_dim.scalar<int32>()());
    const int64 dims = inp0.NumElements();
    OP_REQUIRES(ctx, FastBoundsCheck(cdim, dims),
                errors::InvalidArgument("Concat dim is out of range: ", cdim,
                                        " vs. ", dims));
    int32 offset = 0;
    for (int i = 0; i < N; ++i) {
      const Tensor& inp = ctx->input(1 + i);
      OP_REQUIRES(
          ctx, dims == inp.NumElements(),
          errors::InvalidArgument("input ", i, " should contain ", dims,
                                  " elements, but got", inp.NumElements()));
      auto inp_vec = inp.vec<int32>();
      Tensor* out = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, {dims}, &out));
      auto out_vec = out->vec<int32>();
      for (int64 j = 0; j < dims; ++j) {
        if (j == cdim) {
          out_vec(j) = offset;
          offset += inp_vec(j);
        } else {
          OP_REQUIRES(
              ctx, (inp0_vec(j) == inp_vec(j)),
              errors::InvalidArgument("input[", i, ",", j, "] mismatch: ",
                                      inp0_vec(j), " vs. ", inp_vec(j)));
          out_vec(j) = 0;
        }
      }
    }
  }

  bool IsExpensive() override { return false; }
};

REGISTER_KERNEL_BUILDER(Name("ConcatOffset").Device(DEVICE_CPU),
                        ConcatOffsetOp);

REGISTER_KERNEL_BUILDER(Name("ConcatOffset")
                            .Device(DEVICE_GPU)
                            .HostMemory("concat_dim")
                            .HostMemory("shape")
                            .HostMemory("offset"),
                        ConcatOffsetOp);

}  // namespace tensorflow
