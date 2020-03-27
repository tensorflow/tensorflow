/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_OPS_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_OPS_IMPL_H_

#define EIGEN_USE_THREADS
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <vector>

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/segment_reduction_ops.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/util.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/cuda_solvers.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"

using stream_executor::cuda::ScopedActivateExecutorContext;
#elif TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/cuda_solvers.h"
#include "tensorflow/core/platform/rocm.h"
using stream_executor::rocm::ScopedActivateExecutorContext;
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace internal {
extern void SegmentReductionValidationHelper(OpKernelContext* context,
                                             const Tensor& input,
                                             const Tensor& segment_ids);
extern bool SegmentReductionDoValidation(OpKernelContext* c,
                                         const Tensor& input,
                                         const Tensor& segment_ids);
extern void UnsortedSegmentReductionValidation(OpKernel* op_kernel,
                                               OpKernelContext* context,
                                               const Tensor& data,
                                               const Tensor& segment_ids,
                                               const Tensor& num_segments);
extern bool UnsortedSegmentReductionDoValidation(OpKernel* op_kernel,
                                                 OpKernelContext* context,
                                                 const Tensor& data,
                                                 const Tensor& segment_ids,
                                                 const Tensor& num_segments);
}  // namespace internal

// This operator handles reducing segments along the first dimension.
// See core/ops/math_ops.cc for more details.
template <typename Device, class T, class Index, typename Reducer,
          int default_value>
class SegmentReductionOp : public OpKernel {
 public:
  explicit SegmentReductionOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& segment_ids = context->input(1);

    if (!internal::SegmentReductionDoValidation(context, input, segment_ids)) {
      return;
    }

    const int64 num_indices = segment_ids.NumElements();
    auto input_flat = input.flat_outer_dims<T>();
    const int64 num_col = input_flat.dimension(1);

    const auto segment_vec = segment_ids.vec<Index>();
    // Note that the current implementation assumes that segment_vec values are
    // sorted.
    const Index output_rows =
        num_indices > 0
            ? internal::SubtleMustCopy(segment_vec(num_indices - 1)) + 1
            : 0;
    OP_REQUIRES(context, output_rows >= 0,
                errors::InvalidArgument("segment ids must be >= 0"));

    TensorShape output_shape = input.shape();
    output_shape.set_dim(0, output_rows);

    // Note that we do not initialize the output buffer with a default value, so
    // we need to explicitly set missing indices to the default value.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (num_indices == 0) return;
    OP_REQUIRES(context, output_rows > 0,
                errors::InvalidArgument("segment ids must be >= 0"));
    auto output_flat = output->flat_outer_dims<T>();

#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::DSizes<Eigen::DenseIndex, 1> dims_to_reduce;
    dims_to_reduce[0] = 0;
#else
    Eigen::IndexList<Eigen::type2index<0> > dims_to_reduce;
#endif
    Index start = 0, end = 1;

    Index uninitialized_index = 0;  // Index from which the output is not set.
    Index out_index = internal::SubtleMustCopy(segment_vec(start));

    // TODO(agarwal): if this loop becomes a bottleneck, consider sharding it
    // across threads.
    Eigen::DSizes<Eigen::DenseIndex, 1> out_slice_shape(num_col);
    while (end <= num_indices) {
      // We initialize next_index to 0 to avoid "warning: 'next_index' may be
      // used uninitialized in this function" in the Mac build (since the
      // compiler isn't smart enough to realize the code is safe).
      Index next_index = 0;
      if (end < num_indices) {
        next_index = internal::SubtleMustCopy(segment_vec(end));
        if (out_index == next_index) {
          ++end;
          continue;
        }
        // We have a new segment here.  Verify that the segment ids are growing.
        OP_REQUIRES(context, out_index < next_index,
                    errors::InvalidArgument("segment ids are not increasing"));
      }

      // Process segment [start, end)
      const T* in_slice_ptr = &input_flat(start, 0);
      typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>,
                               Eigen::Unaligned>
          OutT;

      OP_REQUIRES(
          context, FastBoundsCheck(out_index, output_rows),
          errors::InvalidArgument(
              "Segment id ", out_index, " out of range [0, ", output_rows,
              "), possibly because 'segment_ids' input is not sorted."));

      // If there is a gap between two indices, we need to set that gap to the
      // default value.
      if (out_index > uninitialized_index) {
        Eigen::DSizes<Eigen::DenseIndex, 2> gap_slice_shape(
            out_index - uninitialized_index, num_col);
        Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Unaligned>
            gap_slice(&output_flat(uninitialized_index, 0), gap_slice_shape);
        gap_slice.setConstant(T(default_value));
      }

      T* out_slice_ptr = &output_flat(out_index, 0);
      OutT out_slice(out_slice_ptr, out_slice_shape);
      // We don't use out_slice.device(context->eigen_device<Device>)
      // because these pieces of work are likely to be very small and
      // the context switching overhead dwarfs any benefit we get from
      // using another thread to do this work.
      if (start == end - 1) {
        typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>,
                                 Eigen::Unaligned>
            InT;
        InT in_slice(in_slice_ptr, out_slice_shape);
        out_slice = in_slice;
      } else {
        Eigen::DSizes<Eigen::DenseIndex, 2> in_slice_shape(end - start,
                                                           num_col);
        typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>,
                                 Eigen::Unaligned>
            InT;
        InT in_slice(in_slice_ptr, in_slice_shape);

        out_slice = in_slice.reduce(dims_to_reduce, Reducer());
      }
      if (end >= num_indices) break;
      start = end;
      ++end;
      uninitialized_index = out_index + 1;
      out_index = next_index;
    }
  }
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
//  SegmentSumGPUOp is a segment sum operator implemented for GPU only.
//  TODO: This implementation of SegmentSumGPUOp is sometimes slower than
//  its unsorted counterpart (mostly when problem size is small).
//  This is due to the following two main reasons and a cost-effective way
//  to resolve these problems is desirable.
//  1. Sorted segment sum requires a memory transfer from device to host in
//     order to know the size of the output dimension whereas unsorted segment
//     sum receives the size of the output dimension as an input parameter.
//  2. Sorted segment sum is essentially a tiled version of unsorted segment
//     sum and therefore such optimization comes at an inherent cost. However
//     such cost may not be justified when the problem size is small. When to
//     use the tiled version or the untiled version depends on many factors
//     including data alignments, ratio of calculation to memory traffic and
//     obviously, the problem sizes.
template <class T, class Index>
class SegmentSumGPUOp : public AsyncOpKernel {
 public:
  explicit SegmentSumGPUOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    const Tensor& input = context->input(0);
    const Tensor& segment_ids = context->input(1);

    OP_REQUIRES_ASYNC(
        context, TensorShapeUtils::IsVector(segment_ids.shape()),
        errors::InvalidArgument("segment_ids should be a vector."), done);

    const int64 num_indices = segment_ids.NumElements();
    OP_REQUIRES_ASYNC(
        context, num_indices == input.dim_size(0),
        errors::InvalidArgument(
            "segment_ids should be the same size as dimension 0 of"
            " input."),
        done);

    if (num_indices == 0) {
      TensorShape output_shape = input.shape();
      output_shape.set_dim(0, 0);

      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_output(0, output_shape, &output), done);
      done();
      return;
    }

    se::DeviceMemoryBase output_rows_device(
        const_cast<Tensor&>(segment_ids).template flat<Index>().data() +
        (num_indices - 1));
    ScratchSpace<Index> output_rows_host(context, 1, /* on_host */ true);

    auto stream = context->op_device_context()->stream();
    OP_REQUIRES_ASYNC(
        context,
        stream
            ->ThenMemcpy(output_rows_host.mutable_data(), output_rows_device,
                         sizeof(Index))
            .ok(),
        errors::Internal(
            "SegmentSumGPUOp: failed to copy output_rows from device"),
        done);

    functor::SegmentSumFunctor<T, Index> functor_;
    auto create_and_check_output = [context, output_rows_host, &input,
                                    &segment_ids, &functor_, done]() {
      // Ensure that within the callback, the proper GPU settings are
      // configured.
      auto stream = context->op_device_context()->stream();
      ScopedActivateExecutorContext scoped_activation{stream->parent()};

      Index output_rows = *output_rows_host.data();
      output_rows++;
      OP_REQUIRES_ASYNC(context, output_rows > 0,
                        errors::InvalidArgument("segment ids must be >= 0"),
                        done);

      TensorShape output_shape = input.shape();
      output_shape.set_dim(0, output_rows);

      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_output(0, output_shape, &output), done);

      auto output_flat = output->flat_outer_dims<T>();
      auto data_ptr = input.template flat<T>().data();
      auto segment_flat = segment_ids.flat<Index>();
      functor_(context, context->eigen_device<GPUDevice>(), output_rows,
               segment_ids.shape(), segment_flat, input.NumElements(), data_ptr,
               output_flat);

      done();
    };

    context->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
        stream, create_and_check_output);
  }
};
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// ____________________________________________________________________________
// Unsorted segment reduction ops.

namespace functor {

// The ReductionFunctor implementation for CPU.
template <typename T, typename Index, typename InitialValueF,
          typename ReductionF>
struct UnsortedSegmentFunctor<CPUDevice, T, Index, InitialValueF, ReductionF> {
  void operator()(OpKernelContext* ctx, const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  typename TTypes<T, 2>::ConstTensor data,
                  typename TTypes<T, 2>::Tensor output) {
    output.setConstant(InitialValueF()());
    if (data.size() == 0) {
      return;
    }
    const int64 N = segment_ids.dimension(0);
    const int64 num_segments = output.dimension(0);
    ReductionF reduction;
    for (int64 i = 0; i < N; ++i) {
      Index j = internal::SubtleMustCopy(segment_ids(i));
      if (j < 0) {
        continue;
      }
      OP_REQUIRES(ctx, FastBoundsCheck(j, num_segments),
                  errors::InvalidArgument(
                      "segment_ids", SliceDebugString(segment_ids_shape, i),
                      " = ", j, " is out of range [0, ", num_segments, ")"));
      reduction(data.template chip<0>(i), output.template chip<0>(j));
    }
  }
};

template <typename T>
using MatrixChip = Eigen::TensorChippingOp<0l, typename TTypes<T, 2>::Matrix>;

template <typename T>
using constMatrixChip =
    Eigen::TensorChippingOp<0l, const typename TTypes<T, 2>::ConstMatrix>;

// reduction functors
template <typename T>
struct SumOp {
  void operator()(const constMatrixChip<T> data, MatrixChip<T> output) {
    output += data;
  }
};

template <typename T>
struct MaxOp {
  void operator()(const constMatrixChip<T> data, MatrixChip<T> output) {
    output = data.cwiseMax(output);
  }
};

template <typename T>
struct MinOp {
  void operator()(const constMatrixChip<T> data, MatrixChip<T> output) {
    output = data.cwiseMin(output);
  }
};

template <typename T>
struct ProdOp {
  void operator()(const constMatrixChip<T> data, MatrixChip<T> output) {
    output *= data;
  }
};
}  // namespace functor

// The UnsortedSegmentReduction OpKernel. The DeviceReductionFunctor
// is the device specific implementation of the reduction. These device
// specific implementations are templated themselves with the corresponding
// initial value functors and reduction functors.
template <typename T, typename Index, typename DeviceReductionFunctor>
class UnsortedSegmentReductionOp : public OpKernel {
 public:
  explicit UnsortedSegmentReductionOp(OpKernelConstruction* context)
      : OpKernel(context), reduction_functor_(DeviceReductionFunctor()) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& data = context->input(0);
    const Tensor& segment_ids = context->input(1);
    const Tensor& num_segments = context->input(2);
    if (!internal::UnsortedSegmentReductionDoValidation(
            this, context, data, segment_ids, num_segments)) {
      return;
    }
    const auto segment_flat = segment_ids.flat<Index>();
    const int64 output_rows = internal::SubtleMustCopy(static_cast<int64>(
        num_segments.dtype() == DT_INT32 ? num_segments.scalar<int32>()()
                                         : num_segments.scalar<int64>()()));
    OP_REQUIRES(context, output_rows >= 0,
                errors::InvalidArgument("Input num_segments == ", output_rows,
                                        " must not be negative."));
    TensorShape output_shape;
    output_shape.AddDim(output_rows);
    for (int i = segment_ids.dims(); i < data.dims(); i++) {
      output_shape.AddDim(data.dim_size(i));
    }
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto output_flat = output->flat_outer_dims<T>();
    auto data_flat = data.flat_inner_outer_dims<T, 2>(segment_ids.dims() - 1);
    reduction_functor_(context, segment_ids.shape(), segment_flat, data_flat,
                       output_flat);
  }

 protected:
  DeviceReductionFunctor reduction_functor_;
};

// ____________________________________________________________________________
// Sparse segment reduction ops.

// Same as SegmentReductionOp but takes as input a "sparse" tensor, represented
// by two dense tensors, one containing the data, and the other containing
// indices into the data.
template <typename Device, class T>
class SparseSegmentReductionOpBase : public OpKernel {
 public:
  explicit SparseSegmentReductionOpBase(OpKernelConstruction* context,
                                        bool is_mean, bool is_sqrtn,
                                        bool has_num_segments, T default_value)
      : OpKernel(context),
        is_mean_(is_mean),
        is_sqrtn_(is_sqrtn),
        has_num_segments_(has_num_segments),
        default_value_(default_value) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& segment_ids = context->input(2);

    Index output_rows = -1;
    if (has_num_segments_) {
      const Tensor& num_segments = context->input(3);

      OP_REQUIRES(
          context, num_segments.shape().dims() == 0,
          errors::InvalidArgument("num_segments should be a scalar, not shape ",
                                  num_segments.shape().DebugString()));
      output_rows = internal::SubtleMustCopy(num_segments.scalar<int32>()());
      OP_REQUIRES(context, output_rows >= 0,
                  errors::InvalidArgument("segment ids must be >= 0"));
    }

    OP_REQUIRES(context, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices should be a vector."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(segment_ids.shape()),
                errors::InvalidArgument("segment_ids should be a vector."));

    const int64 num_indices = indices.NumElements();
    OP_REQUIRES(context, num_indices == segment_ids.NumElements(),
                errors::InvalidArgument(
                    "segment_ids and indices should have same size."));

    auto input_flat = input.flat_outer_dims<T>();
    const int64 num_col = input_flat.dimension(1);
    const auto indices_vec = indices.vec<Index>();
    typedef int32 OutputRow;
    const auto segment_vec = segment_ids.vec<OutputRow>();
    // Note that the current implementation assumes that segment_vec values are
    // sorted.
    const OutputRow last_segment_id_plus_one =
        num_indices > 0
            ? internal::SubtleMustCopy(segment_vec(num_indices - 1)) + 1
            : 0;
    if (has_num_segments_) {
      OP_REQUIRES(
          context, output_rows >= last_segment_id_plus_one,
          errors::InvalidArgument("segment ids must be < num_segments"));
    } else {
      output_rows = last_segment_id_plus_one;
    }
    OP_REQUIRES(context, output_rows >= 0,
                errors::InvalidArgument("segment ids must be >= 0"));

    TensorShape output_shape = input.shape();
    output_shape.set_dim(0, output_rows);

    // Note that we do not initialize the output buffer with a default value, so
    // we need to explicitly set missing indices to the default value.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (num_indices == 0) {
      if (output_rows > 0) {
        output->flat_outer_dims<T>().setConstant(default_value_);
      }
      return;
    }
    OP_REQUIRES(context, output_rows > 0,
                errors::InvalidArgument("segment ids must be >= 0"));
    auto output_flat = output->flat_outer_dims<T>();

    int64 start = 0, end = 1;
    // Index from which the output is not initialized.
    OutputRow uninitialized_index = 0;
    OutputRow out_index = internal::SubtleMustCopy(segment_vec(start));

    while (true) {
      // We initialize next_index to 0 to avoid "warning: 'next_index' may be
      // used uninitialized in this function" in the Mac build (since the
      // compiler isn't smart enough to realize the code is safe).
      OutputRow next_index = 0;
      if (end < num_indices) {
        next_index = internal::SubtleMustCopy(segment_vec(end));
        if (out_index == next_index) {
          ++end;
          continue;
        }
        // We have a new segment here.  Verify that the segment ids are growing.
        OP_REQUIRES(context, out_index < next_index,
                    errors::InvalidArgument("segment ids are not increasing"));
      }

      OP_REQUIRES(
          context, FastBoundsCheck(out_index, output_rows),
          errors::InvalidArgument(
              "Segment id ", out_index, " out of range [0, ", output_rows,
              "), possibly because 'segment_ids' input is not sorted."));

      // If there is a gap between two indices, we need to set that gap to the
      // default value.
      if (out_index > uninitialized_index) {
        Eigen::DSizes<Eigen::DenseIndex, 2> gap_slice_shape(
            out_index - uninitialized_index, num_col);
        Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Unaligned>
            gap_slice(&output_flat(uninitialized_index, 0), gap_slice_shape);
        gap_slice.setConstant(default_value_);
      }

      auto out = output_flat.template chip<0>(out_index);
      const int bad_offset =
          Reduce(input_flat, indices_vec, start, end - start, out);
      OP_REQUIRES(context, bad_offset < 0,
                  errors::InvalidArgument(
                      "Bad: indices[", start + bad_offset,
                      "] == ", indices_vec(start + bad_offset),
                      " out of range [0, ", input_flat.dimension(0), ")"));

      start = end;
      ++end;
      uninitialized_index = out_index + 1;
      out_index = next_index;
      if (end > num_indices) break;
    }

    // Fill the gap at the end with the default value.
    if (uninitialized_index < output_rows) {
      Eigen::DSizes<Eigen::DenseIndex, 2> gap_slice_shape(
          output_rows - uninitialized_index, num_col);
      Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Unaligned>
          gap_slice(&output_flat(uninitialized_index, 0), gap_slice_shape);
      gap_slice.setConstant(default_value_);
    }
  }

 private:
  typedef int32 Index;

  int64 Reduce(const typename TTypes<T>::ConstMatrix& input_flat,
               const typename TTypes<Index>::ConstVec& indices_vec, int64 start,
               int64 num,
               Eigen::TensorChippingOp<0, typename TTypes<T>::Matrix> out) {
#define INDEX(n, i)                               \
  const auto index##n = indices_vec(start + (i)); \
  if (!FastBoundsCheck(index##n, input_flat.dimension(0))) return (i);

#define L(n) input_flat.template chip<0>(index##n)

    if (num == 1) {
      INDEX(0, 0);
      out = L(0);
    } else {
      int64 r = num % 8;
      T m(1);
      if (is_mean_ && (num < 10)) {
        m = T(num);
      }
      if (is_sqrtn_ && (num < 10)) {
        m = T(sqrt(num));
      }
      switch (r) {
        case 2: {
          INDEX(0, 0);
          INDEX(1, 1);
          out = (L(0) + L(1)) / m;
          break;
        }
        case 3: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          out = (L(0) + L(1) + L(2)) / m;
          break;
        }
        case 4: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          out = (L(0) + L(1) + L(2) + L(3)) / m;
          break;
        }
        case 5: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          INDEX(4, 4);
          out = (L(0) + L(1) + L(2) + L(3) + L(4)) / m;
          break;
        }
        case 6: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          INDEX(4, 4);
          INDEX(5, 5);
          out = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5)) / m;
          break;
        }
        case 7: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          INDEX(4, 4);
          INDEX(5, 5);
          INDEX(6, 6);
          out = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6)) / m;
          break;
        }
        case 0: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          INDEX(4, 4);
          INDEX(5, 5);
          INDEX(6, 6);
          INDEX(7, 7);
          out = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6) + L(7)) / m;
          r = 8;
          break;
        }
        case 1: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          INDEX(4, 4);
          INDEX(5, 5);
          INDEX(6, 6);
          INDEX(7, 7);
          INDEX(8, 8);
          out = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6) + L(7) + L(8)) /
                m;
          r = 9;
          break;
        }
      }
      for (; r < num; r += 8) {
        INDEX(0, r);
        INDEX(1, r + 1);
        INDEX(2, r + 2);
        INDEX(3, r + 3);
        INDEX(4, r + 4);
        INDEX(5, r + 5);
        INDEX(6, r + 6);
        INDEX(7, r + 7);
        out += L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6) + L(7);
      }
      if (is_mean_ && num >= 10) {
        out = out / static_cast<T>(num);
      }
      if (is_sqrtn_ && num >= 10) {
        out = out / static_cast<T>(sqrt(num));
      }
    }

    return -1;
#undef L
#undef INDEX
  }

  const bool is_mean_;
  const bool is_sqrtn_;
  const bool has_num_segments_;
  const T default_value_;
};

template <typename Device, class T>
class SparseSegmentReductionMeanOp
    : public SparseSegmentReductionOpBase<Device, T> {
 public:
  explicit SparseSegmentReductionMeanOp(OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T>(
            context, true /*is_mean*/, false /*is_sqrtn*/,
            false /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T>
class SparseSegmentReductionMeanWithNumSegmentsOp
    : public SparseSegmentReductionOpBase<Device, T> {
 public:
  explicit SparseSegmentReductionMeanWithNumSegmentsOp(
      OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T>(
            context, true /*is_mean*/, false /*is_sqrtn*/,
            true /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T>
class SparseSegmentReductionSqrtNOp
    : public SparseSegmentReductionOpBase<Device, T> {
 public:
  explicit SparseSegmentReductionSqrtNOp(OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T>(
            context, false /*is_mean*/, true /*is_sqrtn*/,
            false /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T>
class SparseSegmentReductionSqrtNWithNumSegmentsOp
    : public SparseSegmentReductionOpBase<Device, T> {
 public:
  explicit SparseSegmentReductionSqrtNWithNumSegmentsOp(
      OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T>(
            context, false /*is_mean*/, true /*is_sqrtn*/,
            true /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T>
class SparseSegmentReductionSumOp
    : public SparseSegmentReductionOpBase<Device, T> {
 public:
  explicit SparseSegmentReductionSumOp(OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T>(
            context, false /*is_mean*/, false /*is_sqrtn*/,
            false /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T>
class SparseSegmentReductionSumWithNumSegmentsOp
    : public SparseSegmentReductionOpBase<Device, T> {
 public:
  explicit SparseSegmentReductionSumWithNumSegmentsOp(
      OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T>(
            context, false /*is_mean*/, false /*is_sqrtn*/,
            true /* has_num_segments */, T(0) /* default_value */) {}
};

template <class T>
class SparseSegmentGradOpBase : public OpKernel {
 public:
  explicit SparseSegmentGradOpBase(OpKernelConstruction* context, bool is_sqrtn)
      : OpKernel(context), is_sqrtn_(is_sqrtn) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& segment_ids = context->input(2);
    const Tensor& output_dim0 = context->input(3);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices should be a vector."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(segment_ids.shape()),
                errors::InvalidArgument("segment_ids should be a vector."));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(output_dim0.shape()),
                errors::InvalidArgument("output_dim0 should be a scalar."));

    const int64 N = indices.NumElements();
    OP_REQUIRES(context, N == segment_ids.NumElements(),
                errors::InvalidArgument(
                    "segment_ids and indices should have same size."));
    typedef int32 SegmentId;
    const SegmentId M =
        internal::SubtleMustCopy(output_dim0.scalar<SegmentId>()());

    auto input_flat = input.flat_outer_dims<T>();
    typedef int32 Index;
    const auto indices_vec = indices.vec<Index>();
    const auto segment_vec = segment_ids.vec<SegmentId>();

    TensorShape output_shape = input.shape();
    output_shape.set_dim(0, M);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (M == 0 || N == 0) return;

    // Note that similar to SparseSegmentMean, we assume that segment_vec is
    // already sorted and has non-negative values.
    const SegmentId num_segments = input.dim_size(0);
    const SegmentId last_segment_id_plus_one =
        internal::SubtleMustCopy(segment_vec(N - 1)) + 1;
    OP_REQUIRES(context, last_segment_id_plus_one <= num_segments,
                errors::InvalidArgument("Invalid number of segments"));

    // Compute scaling factors for input.
    std::vector<double> scaling(num_segments, 0.0);
    for (int64 i = 0; i < N; ++i) {
      const SegmentId idx = internal::SubtleMustCopy(segment_vec(i));
      OP_REQUIRES(
          context, FastBoundsCheck(idx, num_segments),
          errors::InvalidArgument("Segment id ", idx, " out of range [0, ",
                                  num_segments, ")."));
      scaling[idx] += 1;
    }
    for (size_t i = 0; i < scaling.size(); ++i) {
      if (is_sqrtn_) {
        scaling[i] = 1.0 / sqrt(std::max(scaling[i], 1.0));
      } else {
        scaling[i] = 1.0 / std::max(scaling[i], 1.0);
      }
    }

    auto output_flat = output->flat_outer_dims<T>();
    output_flat.setZero();
    std::vector<bool> is_modified(M, false);

    for (int64 i = 0; i < N; ++i) {
      const Index output_idx = internal::SubtleMustCopy(indices_vec(i));
      OP_REQUIRES(context, FastBoundsCheck(output_idx, M),
                  errors::InvalidArgument("Index ", output_idx,
                                          " out of range [0, ", M, ")."));

      const SegmentId idx = internal::SubtleMustCopy(segment_vec(i));
      OP_REQUIRES(
          context, FastBoundsCheck(idx, num_segments),
          errors::InvalidArgument("Segment id ", idx, " out of range [0, ",
                                  num_segments, ")."));

      const T scale = static_cast<T>(scaling[idx]);
      if (is_modified[output_idx]) {
        if (scale == 1.0) {
          output_flat.template chip<0>(output_idx) +=
              input_flat.template chip<0>(idx);
        } else {
          output_flat.template chip<0>(output_idx) +=
              input_flat.template chip<0>(idx) * scale;
        }
      } else {
        if (scale == 1.0) {
          output_flat.template chip<0>(output_idx) =
              input_flat.template chip<0>(idx);
        } else {
          output_flat.template chip<0>(output_idx) =
              input_flat.template chip<0>(idx) * scale;
        }
      }
      is_modified[output_idx] = true;
    }
  }

 private:
  const bool is_sqrtn_;
};

template <class T>
class SparseSegmentMeanGradOp : public SparseSegmentGradOpBase<T> {
 public:
  explicit SparseSegmentMeanGradOp(OpKernelConstruction* context)
      : SparseSegmentGradOpBase<T>(context, false /*is_sqrtn*/) {}
};

template <class T>
class SparseSegmentSqrtNGradOp : public SparseSegmentGradOpBase<T> {
 public:
  explicit SparseSegmentSqrtNGradOp(OpKernelConstruction* context)
      : SparseSegmentGradOpBase<T>(context, true /*is_sqrtn*/) {}
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_OPS_IMPL_H_
