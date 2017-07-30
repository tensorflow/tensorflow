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

// See docs in ../ops/linalg_ops.cc.
#include <cmath>

#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

#include "cuda/include/thrust/device_ptr.h"
#include "cuda/include/thrust/fill.h"
#include "cuda/include/thrust/functional.h"
#include "cuda/include/thrust/iterator/counting_iterator.h"
#include "cuda/include/thrust/iterator/zip_iterator.h"
#include "cuda/include/thrust/reduce.h"
#include "cuda/include/thrust/tuple.h"
#include "tensorflow/core/kernels/cuda_solvers.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

typedef Eigen::GpuDevice GPUDevice;

namespace tensorflow {

namespace {
// https://stackoverflow.com/a/29485908
template <typename Iterator>
class strided_range {
 public:
  typedef typename thrust::iterator_difference<Iterator>::type difference_type;

  struct stride_functor
      : public thrust::unary_function<difference_type, difference_type> {
    difference_type stride;

    stride_functor(difference_type stride) : stride(stride) {}

    __host__ __device__ difference_type
    operator()(const difference_type& i) const {
      return stride * i;
    }
  };

  typedef typename thrust::counting_iterator<difference_type> CountingIterator;
  typedef typename thrust::transform_iterator<stride_functor, CountingIterator>
      TransformIterator;
  typedef typename thrust::permutation_iterator<Iterator, TransformIterator>
      PermutationIterator;

  // type of the strided_range iterator
  typedef PermutationIterator iterator;

  // construct strided_range for the range [first,last)
  strided_range(Iterator first, Iterator last, difference_type stride)
      : first(first), last(last), stride(stride) {}

  iterator begin(void) const {
    return PermutationIterator(
        first, TransformIterator(CountingIterator(0), stride_functor(stride)));
  }

  iterator end(void) const {
    return begin() + ((last - first) + (stride - 1)) / stride;
  }

 protected:
  Iterator first;
  Iterator last;
  difference_type stride;
};

// This took some experiments, cuSolver stores the permutation in a very strange
// way.
struct compute_parity : public thrust::unary_function<int, int> {
  int* perm;
  int n;
  compute_parity(int* perm, int n) : perm(perm), n(n) {}
  __device__ int32 operator()(int i) {
    int pi = perm[i];
    return (i + 1 != pi) ? -1 : 1;
  }
};
}

template <class Scalar>
class DeterminantOpGpu : public OpKernel {
 public:
  explicit DeterminantOpGpu(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) final {
    const Tensor& input = context->input(0);
    const int ndims = input.dims();
    const int64 n = input.dim_size(ndims - 1);
    // Validate inputs.
    OP_REQUIRES(
        context, ndims >= 2,
        errors::InvalidArgument("Input must have rank >= 2, got ", ndims));
    OP_REQUIRES(context, input.dim_size(ndims - 2) == n,
                errors::InvalidArgument("Input matrices must be squares, got",
                                        input.dim_size(ndims - 2), " != ", n));

    // Allocate output.
    Tensor* output;
    TensorShape outputShape = input.shape();
    outputShape.RemoveDim(outputShape.dims() - 1);  // remove matrix dimension,
    outputShape.RemoveDim(outputShape.dims() - 1);  // leave only batch shape
    OP_REQUIRES_OK(context, context->allocate_output(0, outputShape, &output));
    Scalar* outputData = output->flat<Scalar>().data();

    if (n == 0) {
      // If X is an empty matrix (0 rows, 0 col),
      // the determinant is one by definition
      for (int64 i = 0; i < outputShape.num_elements(); ++i) {
        outputData[i] = 1;
      }
      return;
    }

    // This tensor will hold the intermediate LU factorization
    Tensor lu;
    OP_REQUIRES_OK(context, context->allocate_temp(output->dtype(),
                                                   TensorShape({n, n}), &lu));
    Scalar* lu_ptr = lu.flat<Scalar>().data();
    Tensor pivot;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_INT32, TensorShape({n}), &pivot));
    int* pivot_ptr = pivot.flat<int>().data();

    auto input_reshaped = input.template flat_inner_dims<Scalar, 3>();
    auto output_reshaped = output->template flat_inner_dims<Scalar, 1>();

    // Launch a Cholesky kernel for each matrix in the batch.
    const int64 batch_size = input_reshaped.dimension(0);
    int* dev_info_ptr;
    cudaMalloc(&dev_info_ptr, sizeof(int));
    // TODO(rmlarsen): Parallelize over batches if it turns out to be
    // an important use case.
    Scalar* output_ptr = output_reshaped.data();
    CudaSolver solver(context);
    for (int64 i = 0; i < batch_size; ++i) {
      const Scalar* input_ptr = input_reshaped.data() + i * n * n;

      // 1. copy sub-matrix to temp memory
      cudaMemcpy(lu_ptr, input_ptr, sizeof(Scalar) * n * n,
                 cudaMemcpyDeviceToDevice);

      // 2. perform LU factorization
      OP_REQUIRES_OK(context,
                     solver.Getrf(n, n, lu_ptr, n, pivot_ptr, dev_info_ptr));
      int host_info;
      cudaMemcpy(&host_info, dev_info_ptr, sizeof(int), cudaMemcpyDeviceToHost);

      OP_REQUIRES(context, host_info >= 0,
                  errors::InvalidArgument("LU factorization failed for batch ",
                                          i, ", error code: ", host_info));

      if (host_info > 0) {
        // singular matrix -> determinant is zero
        output_ptr[i] = 0;
        continue;
      }

      // 3. perform a strided reduction to multiply the diagonal entries
      // together
      //   https://stackoverflow.com/a/29485908
      int stride = n + 1;
      thrust::device_ptr<Scalar> dev_ptr(lu_ptr);
      strided_range<thrust::device_ptr<Scalar>> pos(dev_ptr, dev_ptr + n * n,
                                                    stride);
      Scalar det = thrust::reduce(pos.begin(), pos.end(), (Scalar)1,
                                  thrust::multiplies<Scalar>());

      // 4. compute the sign of the permutation
      thrust::counting_iterator<int> counting_it(0);
      auto transform_it = thrust::make_transform_iterator(
          counting_it, compute_parity(pivot_ptr, n));
      int sign = thrust::reduce(transform_it, transform_it + n, 1,
                                thrust::multiplies<int>());

      det *= sign;

      // 5. Set output
      output_ptr[i] = det;
    }
  }
};

#define REGISTER_DETERMINANT_OP(OpName, Scalar)            \
  REGISTER_KERNEL_BUILDER(Name(OpName)                     \
                              .Device(DEVICE_GPU)          \
                              .TypeConstraint<Scalar>("T") \
                              .HostMemory("output"),       \
                          DeterminantOpGpu<Scalar>)

REGISTER_DETERMINANT_OP("MatrixDeterminant", float);
REGISTER_DETERMINANT_OP("MatrixDeterminant", double);
REGISTER_DETERMINANT_OP("BatchMatrixDeterminant", float);
REGISTER_DETERMINANT_OP("BatchMatrixDeterminant", double);

#undef REGISTER_DETERMINANT_OP
}

#endif  // GOOGLE_CUDA