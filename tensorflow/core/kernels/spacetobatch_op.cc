/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include <memory>
#include <string>
#include <utility>

#include "tensorflow/core/kernels/spacetobatch_functor.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace {

template <typename Device, typename T>
Status SpaceToBatchOpCompute(OpKernelContext* context,
                             const Tensor& orig_input_tensor,
                             const Tensor& orig_block_shape,
                             const Tensor& orig_paddings) {
  const int input_dims = orig_input_tensor.dims();
  if (!TensorShapeUtils::IsVector(orig_block_shape.shape())) {
    return errors::InvalidArgument("block_shape rank should be 1 instead of ",
                                   orig_block_shape.dims());
  }

  const int block_dims = orig_block_shape.dim_size(0);
  if (orig_input_tensor.dims() < 1 + block_dims) {
    return errors::InvalidArgument("input rank should be >= ", 1 + block_dims,
                                   " instead of ", orig_input_tensor.dims());
  }

  if (!(TensorShapeUtils::IsMatrix(orig_paddings.shape()) &&
        block_dims == orig_paddings.dim_size(0) &&
        2 == orig_paddings.dim_size(1))) {
    return errors::InvalidArgument("paddings should have shape [", block_dims,
                                   ", 2] instead of ",
                                   orig_paddings.shape().DebugString());
  }

  // To avoid out-of-bounds access in the case that the block_shape and/or
  // paddings tensors are concurrently modified, we must copy the values.
  gtl::InlinedVector<int64_t, 4> block_shape;
  gtl::InlinedVector<int64_t, 8> paddings;
  internal::spacetobatch::SubtleMustCopyFlat(orig_block_shape, &block_shape);
  internal::spacetobatch::SubtleMustCopyFlat(orig_paddings, &paddings);

  // Determine the length of the prefix of block dims that can be combined
  // into the batch dimension due to having no padding and block_shape=1.
  int removed_prefix_block_dims = 0;
  for (; removed_prefix_block_dims < block_dims; ++removed_prefix_block_dims) {
    const int dim = removed_prefix_block_dims;
    if (paddings[2 * dim] != 0 || paddings[2 * dim + 1] != 0 ||
        block_shape[dim] != 1) {
      break;
    }
  }

  // Determine the length of the suffix of block dims that can be combined
  // into the depth dimension due to having no padding and block_shape=1.
  int removed_suffix_block_dims = 0;
  for (; removed_suffix_block_dims < block_dims - removed_prefix_block_dims;
       ++removed_suffix_block_dims) {
    const int dim = block_dims - 1 - removed_suffix_block_dims;
    if (paddings[dim * 2] != 0 || paddings[dim * 2 + 1] != 0 ||
        block_shape[dim] != 1) {
      break;
    }
  }

  // Compute the product of the block_shape values.
  int64_t block_shape_product = 1;
  for (int block_dim = 0; block_dim < block_dims; ++block_dim) {
    block_shape_product *= block_shape[block_dim];
  }
  if (block_shape_product <= 0) {
    return errors::InvalidArgument(
        "Product of block sizes must be positive, got ", block_shape_product);
  }

  const int internal_block_dims =
      block_dims - removed_prefix_block_dims - removed_suffix_block_dims;
  if (internal_block_dims > kMaxSpaceToBatchBlockDims) {
    return errors::InvalidArgument(
        "Maximum number of non-combined block dimensions is ",
        internal_block_dims, " but must not exceed ",
        kMaxSpaceToBatchBlockDims);
  }

  if (internal_block_dims == 0) {
    context->set_output(0, orig_input_tensor);
    return Status::OK();
  }

  // For the purpose of computing the result, the input will be treated as
  // having this shape, of rank 2 + internal_block_dims.
  TensorShape internal_input_shape;

  // For the purpose of computing the result, the output will be treated as
  // having this shape, of rank 2 + internal_block_dims.
  TensorShape internal_output_shape;

  // The actual output shape exposed to callers.
  TensorShape external_output_shape;

  external_output_shape.AddDim(orig_input_tensor.dim_size(0) *
                               block_shape_product);

  int64_t input_batch_size = orig_input_tensor.dim_size(0);
  for (int block_dim = 0; block_dim < removed_prefix_block_dims; ++block_dim) {
    const int64_t size = orig_input_tensor.dim_size(block_dim + 1);
    input_batch_size *= size;
    external_output_shape.AddDim(size);
  }
  internal_input_shape.AddDim(input_batch_size);
  internal_output_shape.AddDim(input_batch_size * block_shape_product);

  for (int block_dim = removed_prefix_block_dims;
       block_dim < block_dims - removed_suffix_block_dims; ++block_dim) {
    const int64_t pad_start = paddings[2 * block_dim],
                  pad_end = paddings[2 * block_dim + 1];
    if (pad_start < 0 || pad_end < 0) {
      return errors::InvalidArgument("Paddings must be non-negative");
    }
    const int64_t input_size = orig_input_tensor.dim_size(block_dim + 1);
    const int64_t block_shape_value = block_shape[block_dim];
    const int64_t padded_size = input_size + pad_start + pad_end;
    if (padded_size % block_shape_value != 0) {
      return errors::InvalidArgument("padded_shape[", block_dim,
                                     "]=", padded_size,
                                     " is not divisible by block_shape[",
                                     block_dim, "]=", block_shape_value);
    }
    internal_input_shape.AddDim(input_size);
    const int64_t output_size = padded_size / block_shape_value;
    internal_output_shape.AddDim(output_size);
    external_output_shape.AddDim(output_size);
  }

  int64_t depth = 1;
  for (int dim = block_dims - removed_suffix_block_dims + 1; dim < input_dims;
       ++dim) {
    const int64_t size = orig_input_tensor.dim_size(dim);
    external_output_shape.AddDim(size);
    depth *= size;
  }
  internal_input_shape.AddDim(depth);
  internal_output_shape.AddDim(depth);

  // Allocate output tensor.
  Tensor* output_tensor = nullptr;
  TF_RETURN_IF_ERROR(
      context->allocate_output(0, external_output_shape, &output_tensor));

  const int64_t* internal_paddings = &paddings[2 * removed_prefix_block_dims];
  const int64_t* internal_block_shape = &block_shape[removed_prefix_block_dims];

  switch (internal_block_dims) {
#define TF_SPACETOBATCH_BLOCK_DIMS_CASE(NUM_BLOCK_DIMS)                   \
  case NUM_BLOCK_DIMS: {                                                  \
    TF_RETURN_IF_ERROR(                                                   \
        functor::SpaceToBatchFunctor<Device, T, NUM_BLOCK_DIMS, false>()( \
            context->eigen_device<Device>(),                              \
            orig_input_tensor.shaped<T, NUM_BLOCK_DIMS + 2>(              \
                internal_input_shape.dim_sizes()),                        \
            internal_block_shape, internal_paddings,                      \
            output_tensor->shaped<T, NUM_BLOCK_DIMS + 2>(                 \
                internal_output_shape.dim_sizes())));                     \
  } break;                                                                \
    /**/
    TF_SPACETOBATCH_FOR_EACH_NUM_BLOCK_DIMS(TF_SPACETOBATCH_BLOCK_DIMS_CASE)
#undef TF_SPACETOBATCH_BLOCK_DIMS_CASE
  }
  return Status::OK();
}

}  // namespace

template <typename Device, typename T>
class SpaceToBatchNDOp : public OpKernel {
 public:
  explicit SpaceToBatchNDOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& orig_input_tensor = context->input(0);
    const Tensor& orig_block_shape = context->input(1);
    const Tensor& orig_paddings = context->input(2);
    OP_REQUIRES_OK(context, SpaceToBatchOpCompute<Device, T>(
                                context, orig_input_tensor, orig_block_shape,
                                orig_paddings));
  }
};

template <typename Device, typename T>
class SpaceToBatchOp : public OpKernel {
 public:
  explicit SpaceToBatchOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("block_size", &block_size_));
    OP_REQUIRES(
        context, block_size_ > 1,
        errors::InvalidArgument("Block size should be > 1: ", block_size_));
    block_shape_ = Tensor(tensorflow::DT_INT64, TensorShape({2}));
    auto block_shape_vec = block_shape_.vec<int64_t>();
    block_shape_vec(0) = block_size_;
    block_shape_vec(1) = block_size_;
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& in0 = context->input(0);
    const Tensor& in1 = context->input(1);
    const int dims = in0.dims();

    static const int kRequiredDims = 4;
    OP_REQUIRES(context, kRequiredDims == dims,
                errors::InvalidArgument("Input rank should be: ", kRequiredDims,
                                        "instead of: ", dims));
    OP_REQUIRES_OK(context, SpaceToBatchOpCompute<Device, T>(
                                context, in0, block_shape_, in1));
  }

 private:
  int block_size_;
  Tensor block_shape_;
};

#define REGISTER(T)                                        \
  REGISTER_KERNEL_BUILDER(Name("SpaceToBatchND")           \
                              .Device(DEVICE_CPU)          \
                              .TypeConstraint<T>("T")      \
                              .HostMemory("block_shape")   \
                              .HostMemory("paddings"),     \
                          SpaceToBatchNDOp<CPUDevice, T>); \
  REGISTER_KERNEL_BUILDER(Name("SpaceToBatch")             \
                              .Device(DEVICE_CPU)          \
                              .TypeConstraint<T>("T")      \
                              .HostMemory("paddings"),     \
                          SpaceToBatchOp<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER);
#undef REGISTER

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER(T)                                        \
  REGISTER_KERNEL_BUILDER(Name("SpaceToBatchND")           \
                              .Device(DEVICE_GPU)          \
                              .TypeConstraint<T>("T")      \
                              .HostMemory("block_shape")   \
                              .HostMemory("paddings"),     \
                          SpaceToBatchNDOp<GPUDevice, T>); \
  REGISTER_KERNEL_BUILDER(Name("SpaceToBatch")             \
                              .Device(DEVICE_GPU)          \
                              .TypeConstraint<T>("T")      \
                              .HostMemory("paddings"),     \
                          SpaceToBatchOp<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER);
#undef REGISTER
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // end namespace tensorflow
