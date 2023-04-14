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

#include <cstdint>

#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/platform/types.h"
#define EIGEN_USE_THREADS
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

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
#include "tensorflow/core/util/determinism.h"
#include "tensorflow/core/util/util.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/util/gpu_solvers.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if GOOGLE_CUDA
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/core/util/gpu_solvers.h"

using stream_executor::cuda::ScopedActivateExecutorContext;
#elif TENSORFLOW_USE_ROCM
#include "tensorflow/core/platform/rocm.h"
#include "tensorflow/core/util/gpu_solvers.h"
using stream_executor::rocm::ScopedActivateExecutorContext;
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace internal {
Status ValidateSegmentReduction(OpKernelContext* c, const Tensor& input,
                                const Tensor& segment_ids);
Status ValidateUnsortedSegmentReduction(OpKernel* op_kernel,
                                        OpKernelContext* context,
                                        const Tensor& data,
                                        const Tensor& segment_ids,
                                        const Tensor& num_segments);
Status ValidateSparseSegmentReduction(OpKernelContext* context,
                                      const Tensor& input,
                                      const Tensor& indices,
                                      const Tensor& segment_ids,
                                      bool has_num_segments);
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

    OP_REQUIRES_OK(context, internal::ValidateSegmentReduction(context, input,
                                                               segment_ids));

    const int64_t num_indices = segment_ids.NumElements();
    auto input_flat = input.flat_outer_dims<T>();
    const int64_t num_col = input_flat.dimension(1);

    const auto segment_vec = segment_ids.vec<Index>();
    // Note that the current implementation assumes that segment_vec values are
    // sorted.
    const Index output_rows =
        num_indices > 0
            ? internal::SubtleMustCopy(segment_vec(num_indices - 1)) + 1
            : 0;
    OP_REQUIRES(context, output_rows >= 0,
                errors::InvalidArgument("segment ids must be >= 0"));

    OP_REQUIRES(context, input.dims() >= 1,
                errors::InvalidArgument("Shape must be at least rank 1"));

    TensorShape output_shape = input.shape();
    // Since we're changing the first dimension of the shape, we need to make
    // sure the new shape won't overflow.
    OP_REQUIRES_OK(context, output_shape.SetDimWithStatus(0, output_rows));

    // Note that we do not initialize the output buffer with a default value, so
    // we need to explicitly set missing indices to the default value.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (num_indices == 0) return;
    OP_REQUIRES(context, output_rows > 0,
                errors::InvalidArgument("segment ids must be >= 0"));
    auto output_flat = output->flat_outer_dims<T>();

    Eigen::IndexList<Eigen::type2index<0> > dims_to_reduce;
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

//  SegmentReductionGPUOp is a segment reduction operator implemented for GPU
//  only.
//  TODO: This implementation of SegmentReductionGPUOp is sometimes slower than
//  its unsorted counterpart (mostly when problem size is small).
//  This is due to the following two main reasons and a cost-effective way
//  to resolve these problems is desirable.
//  1. Sorted segment reduction requires a memory transfer from device to host
//     in order to know the size of the output dimension whereas unsorted
//     segment reduction receives the size of the output dimension as an input
//     parameter.
//  2. Sorted segment reduction is essentially a tiled version of unsorted
//     segment reduction and therefore such optimization comes at an inherent
//     cost. However such cost may not be justified when the problem size is
//     small. When to use the tiled version or the untiled version depends on
//     many factors including data alignments, ratio of calculation to memory
//     traffic and obviously, the problem sizes.
template <class T, class Index, class SegmentReductionFunctor, bool IsMean>
class SegmentReductionGPUOp : public AsyncOpKernel {
 public:
  explicit SegmentReductionGPUOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    const Tensor& input = context->input(0);
    const Tensor& segment_ids = context->input(1);

    OP_REQUIRES_ASYNC(
        context, TensorShapeUtils::IsVector(segment_ids.shape()),
        errors::InvalidArgument("segment_ids should be a vector."), done);

    OP_REQUIRES_ASYNC(context, input.dims() >= 1,
                      errors::InvalidArgument("Shape must be at least rank 1"),
                      done);

    const int64_t num_indices = segment_ids.NumElements();
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
        errors::Internal(type_string() +
                         ": failed to copy output_rows from device"),
        done);

    SegmentReductionFunctor functor_;
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
      // Since we're changing the first dimension of the shape, we need to make
      // sure the new shape won't overflow.
      OP_REQUIRES_OK_ASYNC(context,
                           output_shape.SetDimWithStatus(0, output_rows), done);

      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_output(0, output_shape, &output), done);

      bool use_deterministic_kernels =
          UseDeterministicSegmentReductions() ||
          (!SegmentReductionFunctor::atomic_reduction_is_associative &&
           OpDeterminismRequired());

      // The determinism check is here, rather than inside the functor (as it is
      // for the unsorted segment reduction ops) because the done callback
      // (required for OP_REQUIRES_ASYNC) is not available inside the functor.
      bool determinism_requirement_met =
          use_deterministic_kernels ||
          SegmentReductionFunctor::atomic_reduction_is_associative ||
          !OpDeterminismRequired() ||
          DisableSegmentReductionOpDeterminismExceptions();
      OP_REQUIRES_ASYNC(
          context, determinism_requirement_met,
          errors::Unimplemented(
              "Deterministic GPU implementation of sorted segment reduction op"
              " not available."),
          done);

      auto output_flat = output->flat_outer_dims<T>();
      auto data_ptr = input.template flat<T>().data();
      auto segment_flat = segment_ids.flat<Index>();
      functor_(context, context->eigen_device<GPUDevice>(), output_rows,
               segment_ids.shape(), IsMean, segment_flat, input.NumElements(),
               data_ptr, output_flat);

      done();
    };

    context->device()
        ->tensorflow_accelerator_device_info()
        ->event_mgr->ThenExecute(stream, create_and_check_output);
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
    auto cpu_device = ctx->eigen_cpu_device();
    output.device(cpu_device) = output.constant(InitialValueF()());
    if (data.size() == 0) {
      return;
    }

    // This functor will reduce `N` rows input to `num_segments` rows output.
    const int64_t N = segment_ids.dimension(0);
    const int64_t num_segments = output.dimension(0);
    const int64_t inner_dim = data.dimension(1);
    ReductionF reduction;

    // `num_real_segment` counts the rows actually reduced from input,
    // the rows with negative segment index will be excluded.
    // It will be used for cost model.
    int64_t num_real_segment = N;
    // `num_reductions` counts the rows actually reduced in output,
    // the rows only filled with InitialValueF() will be excluded.
    int64_t num_reductions = 0;
    // `row_counter` records how many input rows will be reduced in each
    // output row, the row only fills with InitialValueF() will keep 0.
    // Length of non-zero elements is `num_reductions`.
    std::vector<Index> row_counter(num_segments, 0);

    for (int64_t i = 0; i < N; ++i) {
      Index j = internal::SubtleMustCopy(segment_ids(i));
      if (j < 0) {
        --num_real_segment;
        continue;
      }
      OP_REQUIRES(ctx, FastBoundsCheck(j, num_segments),
                  errors::InvalidArgument(
                      "segment_ids", SliceDebugString(segment_ids_shape, i),
                      " = ", j, " is out of range [0, ", num_segments, ")"));
      if (row_counter[j] == 0) num_reductions++;
      row_counter[j]++;
    }

    // Nothing to reduce. All output values equal to `InitialValueF()`.
    if (num_reductions == 0) return;

    // Parallelize by `num_segments`. It's simple, efficient and safe
    // (no data dependency):
    //
    //   input   segment_ids                 num_segments  operation
    //   | a0 |  | 0 |            worker 1:  |0|           f(a0, a1)
    //   | b0 |  | 1 |            worker 2:  |1|           f(b0, b1)
    // N | c0 |  | 2 |       -->  worker 3:  |2|           f(c0)
    //   | b1 |  | 1 |
    //   | a1 |  | 0 |
    //
    // TODO(intel-tf): Balance workload in `row_counter` to make parallelism
    //                 more efficient.
    auto reductionWorker = [&](int64_t begin, int64_t end) -> void {
      for (int64_t i = 0; i < N; i++) {
        Index j = internal::SubtleMustCopy(segment_ids(i));
        // If `j` is in work scope of this worker, do the reduction.
        if (j >= begin && j < end) {
          reduction(data.template chip<0>(i), output.template chip<0>(j));
        }
      }
    };

    // Reduction functors includes Sum, Max, Min, etc. Simply consider it
    // will cost 5 cycles per operation.
    const int64_t kAverTaskSize = num_real_segment / num_segments;
    const int64_t compute_cycles = 5 * inner_dim * kAverTaskSize;
    const int64_t input_bytes = sizeof(T) * inner_dim * kAverTaskSize;
    const int64_t output_bytes = sizeof(T) * inner_dim * kAverTaskSize;
    const Eigen::TensorOpCost cost(input_bytes, output_bytes, compute_cycles);
    cpu_device.parallelFor(num_segments, cost, reductionWorker);
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
    OP_REQUIRES_OK(context,
                   internal::ValidateUnsortedSegmentReduction(
                       this, context, data, segment_ids, num_segments));
    const auto segment_flat = segment_ids.flat<Index>();
    const int64_t output_rows = internal::SubtleMustCopy(static_cast<int64_t>(
        num_segments.dtype() == DT_INT32 ? num_segments.scalar<int32>()()
                                         : num_segments.scalar<int64_t>()()));
    OP_REQUIRES(context, output_rows >= 0,
                errors::InvalidArgument("Input num_segments == ", output_rows,
                                        " must not be negative."));
    TensorShape output_shape;
    OP_REQUIRES_OK(context, output_shape.AddDimWithStatus(output_rows));
    for (int i = segment_ids.dims(); i < data.dims(); i++) {
      OP_REQUIRES_OK(context, output_shape.AddDimWithStatus(data.dim_size(i)));
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
//
// The template parameters are:
// * Device: An Eigen device object, on which the kernel will execute.
// * T: The value type.
// * Index: The element type of the indices tensor (int32 or int64).
// * SegmentId: The element type of the segment_ids tensor (int32 or int64).
template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentReductionOpBase : public OpKernel {
 public:
  explicit SparseSegmentReductionOpBase(OpKernelConstruction* context,
                                        bool is_mean, bool is_sqrtn,
                                        bool has_num_segments, T default_value)
      : OpKernel(context),
        dtidx_(DataTypeToEnum<Index>::v()),
        is_mean_(is_mean),
        is_sqrtn_(is_sqrtn),
        has_num_segments_(has_num_segments),
        default_value_(default_value) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& segment_ids = context->input(2);

    OP_REQUIRES_OK(
        context, internal::ValidateSparseSegmentReduction(
                     context, input, indices, segment_ids, has_num_segments_));

    Index output_rows = -1;
    if (has_num_segments_) {
      const Tensor& num_segments = context->input(3);
      // Note that there is a Tnumsegments parameter on the op, but it is not
      // plumbed through to here and so always takes its default value of int32.
      output_rows = internal::SubtleMustCopy(num_segments.scalar<int32>()());
    }
    const int64_t num_indices = indices.NumElements();

    auto input_flat = input.flat_outer_dims<T>();
    const int64_t num_col = input_flat.dimension(1);
    const auto indices_vec = indices.vec<Index>();
    const auto segment_vec = segment_ids.vec<SegmentId>();
    // Note that the current implementation assumes that segment_vec values are
    // sorted.
    const SegmentId last_segment_id =
        num_indices > 0 ? segment_vec(num_indices - 1) : 0;
    int64_t limit = dtidx_ == DataType::DT_INT32 ? kint32max : kint64max;

    OP_REQUIRES(
        context, last_segment_id < limit,
        errors::InvalidArgument("Last segment id must be < kintmax, got ",
                                last_segment_id, " limit ", limit));

    const SegmentId last_segment_id_plus_one =
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
    OP_REQUIRES_OK(context, output_shape.SetDimWithStatus(0, output_rows));

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

    Tensor temp;
    if (input.dtype() == DT_BFLOAT16 || input.dtype() == DT_HALF) {
      temp = tensorflow::Tensor(DT_FLOAT, output_shape);
    }
    auto temp_flat = temp.flat_outer_dims<float>();

    int64_t start = 0, end = 1;
    // Index from which the output is not initialized.
    SegmentId uninitialized_index = 0;
    SegmentId out_index = internal::SubtleMustCopy(segment_vec(start));

    while (true) {
      // We initialize next_index to 0 to avoid "warning: 'next_index' may be
      // used uninitialized in this function" in the Mac build (since the
      // compiler isn't smart enough to realize the code is safe).
      SegmentId next_index = 0;
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
      auto temp = temp_flat.template chip<0>(out_index);
      const int bad_offset = Reduce<T, Index>(input_flat, indices_vec, start,
                                              end - start, out, temp);
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
  const DataType dtidx_;
  template <typename Tin>
  using EnableIfBfloat16OrHalf =
      typename std::enable_if<std::is_same<Tin, bfloat16>::value ||
                                  std::is_same<Tin, Eigen::half>::value,
                              int>::type;
  template <typename Tin>
  using EnableIfNotBfloat16OrHalf =
      typename std::enable_if<!std::is_same<Tin, bfloat16>::value &&
                                  !std::is_same<Tin, Eigen::half>::value,
                              int>::type;

  template <typename Tin, typename Tindex, EnableIfNotBfloat16OrHalf<Tin> = 0>
  EIGEN_ALWAYS_INLINE auto fetch_val(
      const typename TTypes<Tin>::ConstMatrix& input_flat, Tindex index) {
    return input_flat.template chip<0>(index);
  }

  template <typename Tin, typename Tindex, EnableIfBfloat16OrHalf<Tin> = 0>
  EIGEN_ALWAYS_INLINE auto fetch_val(
      const typename TTypes<Tin>::ConstMatrix& input_flat, Tindex index) {
    return input_flat.template chip<0>(index).template cast<float>();
  }

  template <typename Tout>
  EIGEN_ALWAYS_INLINE Tout get_scaling_factor(int64_t num) {
    Tout m(1);
    if (is_mean_ && (num < 10)) {
      m = Tout(num);
    }
    if (is_sqrtn_ && (num < 10)) {
      m = Tout(sqrt(num));
    }
    return Tout(1) / m;
  }

  template <typename Tin, typename Tindex, EnableIfNotBfloat16OrHalf<Tin> = 0>
  int64_t Reduce(
      const typename TTypes<Tin>::ConstMatrix& input_flat,
      const typename TTypes<Tindex>::ConstVec& indices_vec, int64_t start,
      int64_t num, Eigen::TensorChippingOp<0, typename TTypes<Tin>::Matrix> out,
      Eigen::TensorChippingOp<0, typename TTypes<float>::Matrix> temp) {
    return ReduceImpl<Tin, Tindex, Tin>(input_flat, indices_vec, start, num,
                                        out, get_scaling_factor<Tin>(num));
  }

  template <typename Tin, typename Tindex, EnableIfBfloat16OrHalf<Tin> = 0>
  int64_t Reduce(
      const typename TTypes<Tin>::ConstMatrix& input_flat,
      const typename TTypes<Tindex>::ConstVec& indices_vec, int64_t start,
      int64_t num, Eigen::TensorChippingOp<0, typename TTypes<Tin>::Matrix> out,
      Eigen::TensorChippingOp<0, typename TTypes<float>::Matrix> temp) {
    int64_t res =
        ReduceImpl<Tin, Tindex, float>(input_flat, indices_vec, start, num,
                                       temp, get_scaling_factor<float>(num));
    out = temp.template cast<Tin>();
    return res;
  }

  template <typename Tin, typename Tindex, typename Tout>
  int64_t ReduceImpl(
      const typename TTypes<Tin>::ConstMatrix& input_flat,
      const typename TTypes<Tindex>::ConstVec& indices_vec, int64_t start,
      int64_t num,
      Eigen::TensorChippingOp<0, typename TTypes<Tout>::Matrix> out,
      const Tout scaling_factor) {
#define INDEX(n, i)                               \
  const auto index##n = indices_vec(start + (i)); \
  if (!FastBoundsCheck(index##n, input_flat.dimension(0))) return (i);

#define L(n) fetch_val<Tin, Tindex>(input_flat, index##n)

    if (num == 1) {
      INDEX(0, 0);
      out = L(0);
    } else {
      int64_t r = num & 7;
      switch (r) {
        case 2: {
          INDEX(0, 0);
          INDEX(1, 1);
          out = (L(0) + L(1)) * scaling_factor;
          break;
        }
        case 3: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          out = (L(0) + L(1) + L(2)) * scaling_factor;
          break;
        }
        case 4: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          out = (L(0) + L(1) + L(2) + L(3)) * scaling_factor;
          break;
        }
        case 5: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          INDEX(4, 4);
          out = (L(0) + L(1) + L(2) + L(3) + L(4)) * scaling_factor;
          break;
        }
        case 6: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          INDEX(4, 4);
          INDEX(5, 5);
          out = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5)) * scaling_factor;
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
          out =
              (L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6)) * scaling_factor;
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
          out = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6) + L(7)) *
                scaling_factor;
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
          out = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6) + L(7) + L(8)) *
                scaling_factor;
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
        out = out / static_cast<Tout>(num);
      }
      if (is_sqrtn_ && num >= 10) {
        out = out / static_cast<Tout>(sqrt(num));
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Specialization for GPU. Must be Async because may need to wait for a host to
// device memcpy before allocating output.
template <class T, typename Index, typename SegmentId>
class SparseSegmentReductionOpBase<GPUDevice, T, Index, SegmentId>
    : public AsyncOpKernel {
 public:
  explicit SparseSegmentReductionOpBase(OpKernelConstruction* context,
                                        bool is_mean, bool is_sqrtn,
                                        bool has_num_segments, T default_value)
      : AsyncOpKernel(context),
        is_mean_(is_mean),
        is_sqrtn_(is_sqrtn),
        has_num_segments_(has_num_segments),
        default_value_(default_value) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    const Tensor& input = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& segment_ids = context->input(2);

    OP_REQUIRES_OK_ASYNC(
        context,
        internal::ValidateSparseSegmentReduction(
            context, input, indices, segment_ids, has_num_segments_),
        done);

    ScratchSpace<SegmentId> last_segment_id_host(context, 1, /*on_host=*/true);

    auto create_and_check_output = [this, context, input, indices, segment_ids,
                                    last_segment_id_host, done]() {
      // Ensure that within the callback, the proper GPU settings are
      // configured.
      auto stream = context->op_device_context()->stream();
      ScopedActivateExecutorContext scoped_activation{stream->parent()};

      SegmentId last_segment_id = *last_segment_id_host.data();
      SegmentId output_rows = last_segment_id + 1;
      OP_REQUIRES_ASYNC(context, output_rows > 0,
                        errors::InvalidArgument("segment ids must be >= 0"),
                        done);

      TensorShape output_shape = input.shape();
      output_shape.set_dim(0, output_rows);

      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_output(0, output_shape, &output), done);

      auto input_flat = input.flat_outer_dims<T>();
      const auto indices_vec = indices.vec<Index>();
      const auto segment_ids_vec = segment_ids.vec<SegmentId>();
      auto output_flat = output->flat_outer_dims<T>();

      functor::SparseSegmentReductionFunctor<T, Index, SegmentId> functor;
      OP_REQUIRES_OK_ASYNC(
          context,
          functor(context, is_mean_, is_sqrtn_, default_value_, input_flat,
                  indices_vec, segment_ids_vec, output_flat),
          done);
      done();
    };

    if (has_num_segments_) {
      // No need to do any device to host memcpy, just compute synchronously.
      const Tensor& num_segments_t = context->input(3);
      SegmentId num_segments =
          internal::SubtleMustCopy(num_segments_t.dtype() == DT_INT32
                                       ? num_segments_t.scalar<int32>()()
                                       : num_segments_t.scalar<int64_t>()());
      *last_segment_id_host.mutable_data() = num_segments - 1;
      create_and_check_output();
    } else {
      const int64_t num_indices = indices.NumElements();
      if (num_indices == 0) {
        TensorShape output_shape = input.shape();
        output_shape.set_dim(0, 0);

        Tensor* output = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, output_shape, &output), done);
        done();
        return;
      }

      // Need to copy last element of segment_ids from device to host, and then
      // asynchronously allocate the output and finish the computation.
      se::DeviceMemoryBase last_segment_id_device(
          const_cast<Tensor&>(segment_ids).template flat<SegmentId>().data() +
          (num_indices - 1));
      auto stream = context->op_device_context()->stream();
      OP_REQUIRES_ASYNC(
          context,
          stream
              ->ThenMemcpy(last_segment_id_host.mutable_data(),
                           last_segment_id_device, sizeof(SegmentId))
              .ok(),
          errors::Internal(type_string() +
                           ": failed to copy last_segment_id from device"),
          done);
      context->device()
          ->tensorflow_accelerator_device_info()
          ->event_mgr->ThenExecute(stream, create_and_check_output);
    }
  }

 private:
  const bool is_mean_;
  const bool is_sqrtn_;
  const bool has_num_segments_;
  const T default_value_;
};

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentReductionMeanOp
    : public SparseSegmentReductionOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentReductionMeanOp(OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T, Index, SegmentId>(
            context, true /*is_mean*/, false /*is_sqrtn*/,
            false /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentReductionMeanWithNumSegmentsOp
    : public SparseSegmentReductionOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentReductionMeanWithNumSegmentsOp(
      OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T, Index, SegmentId>(
            context, true /*is_mean*/, false /*is_sqrtn*/,
            true /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentReductionSqrtNOp
    : public SparseSegmentReductionOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentReductionSqrtNOp(OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T, Index, SegmentId>(
            context, false /*is_mean*/, true /*is_sqrtn*/,
            false /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentReductionSqrtNWithNumSegmentsOp
    : public SparseSegmentReductionOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentReductionSqrtNWithNumSegmentsOp(
      OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T, Index, SegmentId>(
            context, false /*is_mean*/, true /*is_sqrtn*/,
            true /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentReductionSumOp
    : public SparseSegmentReductionOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentReductionSumOp(OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T, Index, SegmentId>(
            context, false /*is_mean*/, false /*is_sqrtn*/,
            false /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentReductionSumWithNumSegmentsOp
    : public SparseSegmentReductionOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentReductionSumWithNumSegmentsOp(
      OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T, Index, SegmentId>(
            context, false /*is_mean*/, false /*is_sqrtn*/,
            true /* has_num_segments */, T(0) /* default_value */) {}
};

namespace functor {

template <typename T, typename Index, typename SegmentId>
struct SparseSegmentGradFunctor<CPUDevice, T, Index, SegmentId> {
  void operator()(OpKernelContext* context,
                  SparseSegmentReductionOperation operation,
                  typename TTypes<T>::ConstMatrix input_flat,
                  typename TTypes<Index>::ConstVec indices_vec,
                  typename TTypes<SegmentId>::ConstVec segment_vec,
                  typename TTypes<T>::Matrix output_flat) {
    const int64_t N = indices_vec.size();
    const SegmentId M = output_flat.dimension(0);

    // Note that similar to SparseSegmentMean, we assume that segment_vec is
    // already sorted and has non-negative values.
    const SegmentId num_segments = input_flat.dimension(0);
    const SegmentId last_segment_id_plus_one =
        internal::SubtleMustCopy(segment_vec(N - 1)) + 1;
    OP_REQUIRES(context, last_segment_id_plus_one <= num_segments,
                errors::InvalidArgument("Invalid number of segments"));

    // Compute scaling factors for input.
    std::vector<double> scaling(
        (operation == SparseSegmentReductionOperation::kSum ? 0 : num_segments),
        0.0);
    if (operation != SparseSegmentReductionOperation::kSum) {
      for (int64_t i = 0; i < N; ++i) {
        const SegmentId idx = internal::SubtleMustCopy(segment_vec(i));
        OP_REQUIRES(
            context, FastBoundsCheck(idx, num_segments),
            errors::InvalidArgument("Segment id ", idx, " out of range [0, ",
                                    num_segments, ")."));
        scaling[idx] += 1;
      }
      for (size_t i = 0; i < scaling.size(); ++i) {
        switch (operation) {
          case SparseSegmentReductionOperation::kSum: {
            OP_REQUIRES(
                context, false,
                errors::Internal(
                    "Should not happen: sum inside SparseSegmentReductionOp "
                    "scaling generation."));
          }
          case SparseSegmentReductionOperation::kMean: {
            scaling[i] = 1.0 / std::max(scaling[i], 1.0);
            break;
          }
          case SparseSegmentReductionOperation::kSqrtN: {
            scaling[i] = 1.0 / sqrt(std::max(scaling[i], 1.0));
            break;
          }
            // No default to get compiler warnings for missing cases.
        }
      }
    }

    output_flat.setZero();
    std::vector<bool> is_modified(M, false);

    for (int64_t i = 0; i < N; ++i) {
      const Index output_idx = internal::SubtleMustCopy(indices_vec(i));
      OP_REQUIRES(context, FastBoundsCheck(output_idx, M),
                  errors::InvalidArgument("Index ", output_idx,
                                          " out of range [0, ", M, ")."));

      const SegmentId idx = internal::SubtleMustCopy(segment_vec(i));
      OP_REQUIRES(
          context, FastBoundsCheck(idx, num_segments),
          errors::InvalidArgument("Segment id ", idx, " out of range [0, ",
                                  num_segments, ")."));

      const T scale = (operation == SparseSegmentReductionOperation::kSum
                           ? static_cast<T>(1)
                           : static_cast<T>(scaling[idx]));
      if (is_modified[output_idx]) {
        if (scale == T{1.0}) {
          output_flat.template chip<0>(output_idx) +=
              input_flat.template chip<0>(idx);
        } else {
          output_flat.template chip<0>(output_idx) +=
              input_flat.template chip<0>(idx) * scale;
        }
      } else {
        if (scale == T{1.0}) {
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
};

}  // namespace functor

// Implements the common logic for the gradients of SparseSegmentReduction
// kernels.
//
// The template parameters are:
// * Device: An Eigen device object, on which the kernel will execute.
// * T: The value type.
// * Index: The element type of the indices tensor (int32 or int64).
// * SegmentId: The element type of the segment_ids tensor (int32 or int64).
template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentGradOpBase : public OpKernel {
 public:
  explicit SparseSegmentGradOpBase(OpKernelConstruction* context,
                                   SparseSegmentReductionOperation operation)
      : OpKernel(context), operation_(operation) {}

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

    const int64_t N = indices.NumElements();
    OP_REQUIRES(context, N == segment_ids.NumElements(),
                errors::InvalidArgument(
                    "segment_ids and indices should have same size."));
    const SegmentId M = internal::SubtleMustCopy(output_dim0.scalar<int32>()());

    auto input_flat = input.flat_outer_dims<T>();
    const auto indices_vec = indices.vec<Index>();
    const auto segment_vec = segment_ids.vec<SegmentId>();

    TensorShape output_shape = input.shape();
    OP_REQUIRES_OK(context, output_shape.SetDimWithStatus(0, M));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (M == 0 || N == 0) return;

    auto output_flat = output->flat_outer_dims<T>();
    functor::SparseSegmentGradFunctor<Device, T, Index, SegmentId>()(
        context, operation_, input_flat, indices_vec, segment_vec, output_flat);
  }

 private:
  const SparseSegmentReductionOperation operation_;
};

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentSumGradOp
    : public SparseSegmentGradOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentSumGradOp(OpKernelConstruction* context)
      : SparseSegmentGradOpBase<Device, T, Index, SegmentId>(
            context, SparseSegmentReductionOperation::kSum) {}
};

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentMeanGradOp
    : public SparseSegmentGradOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentMeanGradOp(OpKernelConstruction* context)
      : SparseSegmentGradOpBase<Device, T, Index, SegmentId>(
            context, SparseSegmentReductionOperation::kMean) {}
};

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentSqrtNGradOp
    : public SparseSegmentGradOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentSqrtNGradOp(OpKernelConstruction* context)
      : SparseSegmentGradOpBase<Device, T, Index, SegmentId>(
            context, SparseSegmentReductionOperation::kSqrtN) {}
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_OPS_IMPL_H_
