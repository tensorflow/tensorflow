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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/sparse_reorder_op.h"

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <utility>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/sparse_utils.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
using GPUDevice = Eigen::GpuDevice;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace functor {

template <typename T>
struct SparseReorderFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& input_ind,
                  const Tensor& input_val, const Tensor& input_shape_in) {
    absl::Span<const int64_t> input_shape(input_shape_in.vec<int64_t>().data(),
                                          input_shape_in.NumElements());

    absl::InlinedVector<int64_t, 8UL> std_order(input_shape.size());
    std::iota(std_order.begin(), std_order.end(), 0);

    // Check if the sparse tensor is already ordered correctly
    sparse::SparseTensor input_sp;
    OP_REQUIRES_OK(
        context, sparse::SparseTensor::Create(input_ind, input_val, input_shape,
                                              std_order, &input_sp));

    if (input_sp.IndicesValid().ok()) {
      context->set_output(0, input_sp.indices());
      context->set_output(1, input_sp.values());
    } else {
      // Deep-copy the input Tensors, then reorder in-place
      sparse::SparseTensor reordered_sp;
      OP_REQUIRES_OK(context,
                     sparse::SparseTensor::Create(tensor::DeepCopy(input_ind),
                                                  tensor::DeepCopy(input_val),
                                                  input_shape, &reordered_sp));
      reordered_sp.Reorder<T>(std_order);
      context->set_output(0, reordered_sp.indices());
      context->set_output(1, reordered_sp.values());
    }
  }
};

}  // namespace functor

template <typename Device, typename T>
class SparseReorderOp : public OpKernel {
 public:
  explicit SparseReorderOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_ind = context->input(0);
    const Tensor& input_val = context->input(1);
    const Tensor& input_shape_in = context->input(2);
    // Indices aren't used, and some ops use -1 as a placeholder for missing
    // values.
    OP_REQUIRES_OK(context, (sparse_utils::ValidateSparseTensor<int64_t>(
                                input_ind, input_val, input_shape_in,
                                sparse_utils::IndexValidation::kNone)));
    functor::SparseReorderFunctor<Device, T>()(context, input_ind, input_val,
                                               input_shape_in);
  }
};

#define REGISTER_KERNELS(type)                                            \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("SparseReorder").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseReorderOp<CPUDevice, type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU_KERNELS(type)                                        \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("SparseReorder").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SparseReorderOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
TF_CALL_INTEGRAL_TYPES(REGISTER_GPU_KERNELS);
REGISTER_GPU_KERNELS(bool);
#undef REGISTER_GPU_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
