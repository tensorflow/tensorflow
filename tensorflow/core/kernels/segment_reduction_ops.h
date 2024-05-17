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

#ifndef TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_OPS_H_
#define TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_OPS_H_

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

class OpKernelContext;

bool UseDeterministicSegmentReductions();
bool DisableSegmentReductionOpDeterminismExceptions();

// Type of SparseSegmentReduction operation to perform gradient of.
enum class SparseSegmentReductionOperation { kSum, kMean, kSqrtN };

namespace functor {

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Note that we define this ourselves to avoid a dependency on gpuprim.
struct Sum {
  template <typename T>
  __host__ __device__ T operator()(const T& a, const T& b) const {
    return a + b;
  }
};

struct Prod {
  template <typename T>
  __host__ __device__ T operator()(const T& a, const T& b) const {
    return a * b;
  }
};

// Note that we don't use gpuprim::Min/Max because they use operator<, which is
// not implemented for AlignedVector types.
struct Min {
  template <typename T>
  __host__ __device__ T operator()(const T& a, const T& b) const {
    return min(a, b);
  }
};

struct Max {
  template <typename T>
  __host__ __device__ T operator()(const T& a, const T& b) const {
    return max(a, b);
  }
};

template <typename ReduceOp, typename T>
struct ReduceOpIsAssociative {};
template <typename T>
struct ReduceOpIsAssociative<functor::Sum, T> : std::is_integral<T> {};
template <typename T>
struct ReduceOpIsAssociative<functor::Prod, T> : std::is_integral<T> {};
template <typename T>
struct ReduceOpIsAssociative<functor::Max, T> : std::true_type {};
template <typename T>
struct ReduceOpIsAssociative<functor::Min, T> : std::true_type {};

typedef Eigen::GpuDevice GPUDevice;
// Functor for SegmentReductionGPUOp.
// output_rows: the number of output segments (unique segment ids in
//                'segment_ids').
// segment_ids_shape: shape of 'segment_ids' tensor.
// segment_ids: unsorted map from input to output segment ids at which to
//                perform segment sum operation.
// data_size: size of input data tensor.
// data: input data tensor.
// output: output reshaped to {output_rows, output.size/output_rows}
template <typename T, typename Index, typename InitialValueF,
          typename EmptySegmentValueF, typename ReductionF>
struct SegmentReductionFunctor {
  void operator()(OpKernelContext* ctx, const GPUDevice& d,
                  const Index output_rows, const TensorShape& segment_ids_shape,
                  bool is_mean, typename TTypes<Index>::ConstFlat segment_ids,
                  const Index data_size, const T* data,
                  typename TTypes<T, 2>::Tensor output);
  static constexpr bool atomic_reduction_is_associative =
      ReduceOpIsAssociative<ReductionF, T>::value;
};

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename Device, typename T, typename Index, typename InitialValueF,
          typename ReductionF>
struct UnsortedSegmentFunctor {
  void operator()(OpKernelContext* ctx, const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  typename TTypes<T, 2>::ConstTensor data,
                  typename TTypes<T, 2>::Tensor output);
};

// Initial value functors.
template <typename T>
struct Zero {
  EIGEN_STRONG_INLINE T operator()() const { return T(0); }
};

template <typename T>
struct One {
  EIGEN_STRONG_INLINE T operator()() const { return T(1); }
};

template <typename T>
struct Lowest {
  EIGEN_STRONG_INLINE T operator()() const {
    return Eigen::NumTraits<T>::lowest();
  }
};

template <typename T>
struct Highest {
  EIGEN_STRONG_INLINE T operator()() const {
    return Eigen::NumTraits<T>::highest();
  }
};

template <typename T, typename Index, typename SegmentId>
struct SparseSegmentReductionFunctor {
  Status operator()(OpKernelContext* context, bool is_mean, bool is_sqrtn,
                    T default_value, typename TTypes<T, 2>::ConstTensor input,
                    typename TTypes<Index>::ConstVec indices,
                    typename TTypes<SegmentId>::ConstVec segment_ids,
                    typename TTypes<T, 2>::Tensor output);
};

template <class Device, typename T, typename Index, typename SegmentId>
struct SparseSegmentGradFunctor {
  void operator()(OpKernelContext* context,
                  SparseSegmentReductionOperation operation,
                  typename TTypes<T>::ConstMatrix input_flat,
                  typename TTypes<Index>::ConstVec indices_vec,
                  typename TTypes<SegmentId>::ConstVec segment_vec,
                  typename TTypes<T>::Matrix output_flat);
};

template <class Device, typename T, typename Index, typename SegmentId>
struct SparseSegmentGradV2Functor {
  void operator()(OpKernelContext* context,
                  SparseSegmentReductionOperation operation,
                  typename TTypes<T>::ConstMatrix input_flat,
                  typename TTypes<Index>::ConstVec indices_vec,
                  typename TTypes<SegmentId>::ConstVec segment_vec,
                  const TensorShape& dense_output_shape,
                  typename AsyncOpKernel::DoneCallback done);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_OPS_H_
