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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow/core/util/cuda_kernel_helper.h"

#include "tensorflow/core/kernels/reduction_gpu_kernels.cu.h"
#include "tensorflow/core/kernels/reduction_ops_common.h"

namespace tensorflow {

namespace {

template <typename T>
__global__ void GenerateNormalizedProb(const T* logits, const T* sum_probs,
                                       const T* max_logits, T* output,
                                       const int num_rows, const int num_cols,
                                       const bool in_log_space) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  const int row = tid / num_cols;
  const int col = tid % num_cols;

  if (row < num_rows && col < num_cols) {
    if (in_log_space)
      output[tid] =
          logits[tid] - ldg(max_logits + row) - log(ldg(sum_probs + row));
    else
      output[tid] =
          exp(logits[tid] - ldg(max_logits + row)) / ldg(sum_probs + row);
  }
}

template <typename T>
struct SubtractAndExpFunctor {
  __host__ __device__ SubtractAndExpFunctor(const T* logits,
                                            const T* max_logits,
                                            const int num_cols)
      : logits_(logits), max_logits_(max_logits), num_cols_(num_cols) {}

  __host__ __device__ T operator()(const int gid) const {
    return exp(logits_[gid] - ldg(max_logits_ + gid / num_cols_));
  }

  const T* logits_;
  const T* max_logits_;
  const int num_cols_;
};

template <typename T, typename Op, typename InputIter>
void DoRowReduction(OpKernelContext* context, T* output, InputIter input,
                    int rows, int cols) {
  typedef const Eigen::array<TTypes<float>::Tensor::Index, 1>& ReductionAxes;
  Constants<GPUDevice> constants;

  Op op;

  functor::ReduceImpl<T, Op, T*, InputIter, ReductionAxes>(
      context, output, input, 2, rows, cols, 1, 1, constants.kOne, op);
}

}  // namespace

template <typename T>
class SoftmaxOpGPU : public OpKernel {
 public:
  explicit SoftmaxOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    log_ = StringPiece(type_string()).starts_with("Log");
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& logits_in_ = context->input(0);
    auto logits_in = logits_in_.matrix<T>();
    const int rows = logits_in.dimension(0);
    const int cols = logits_in.dimension(1);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(logits_in_.shape()),
                errors::InvalidArgument("logits must be 2-dimensional"));
    Tensor* softmax_out = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, logits_in_.shape(), &softmax_out));

    const cudaStream_t& cu_stream = GetCudaStream(context);
    if (logits_in_.NumElements() > 0) {
      Tensor max_logits;
      Tensor sum_probs;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::value,
                                            softmax_out->shape(), &max_logits));
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::value,
                                            softmax_out->shape(), &sum_probs));

      DoRowReduction<T, cub::Max, const T*>(
          context, const_cast<T*>(max_logits.flat<T>().data()),
          reinterpret_cast<const T*>(logits_in_.flat<T>().data()), rows, cols);

      const int numThreads = 128;
      const int numBlocks = Eigen::divup(rows * cols, numThreads);

      cub::CountingInputIterator<int> counting_iterator(0);
      typedef cub::TransformInputIterator<T, SubtractAndExpFunctor<T>,
                                          cub::CountingInputIterator<int>>
          InputIterType;

      InputIterType input_itr(
          counting_iterator,
          SubtractAndExpFunctor<T>(
              reinterpret_cast<const T*>(logits_in_.flat<T>().data()),
              reinterpret_cast<const T*>(max_logits.flat<T>().data()), cols));

      DoRowReduction<T, cub::Sum, InputIterType>(
          context, const_cast<T*>(sum_probs.flat<T>().data()), input_itr, rows,
          cols);

      GenerateNormalizedProb<<<numBlocks, numThreads, 0, cu_stream>>>(
          reinterpret_cast<const T*>(logits_in_.flat<T>().data()),
          reinterpret_cast<const T*>(sum_probs.flat<T>().data()),
          reinterpret_cast<const T*>(max_logits.flat<T>().data()),
          const_cast<T*>(softmax_out->flat<T>().data()), rows, cols, log_);
    }
  }

 private:
  bool log_;
};

REGISTER_KERNEL_BUILDER(
    Name("Softmax").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
    SoftmaxOpGPU<Eigen::half>);
REGISTER_KERNEL_BUILDER(
    Name("Softmax").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    SoftmaxOpGPU<float>);
REGISTER_KERNEL_BUILDER(
    Name("Softmax").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    SoftmaxOpGPU<double>);
REGISTER_KERNEL_BUILDER(
    Name("LogSoftmax").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
    SoftmaxOpGPU<Eigen::half>);
REGISTER_KERNEL_BUILDER(
    Name("LogSoftmax").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    SoftmaxOpGPU<float>);

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
