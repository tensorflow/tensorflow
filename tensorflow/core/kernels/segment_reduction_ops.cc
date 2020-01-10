// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/tensor.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/public/status.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

// This operator handles reducing segments along the first dimension.
// See core/ops/math_ops.cc for more details.
template <typename Device, class T, class Index, typename Reducer>
class SegmentReductionOp : public OpKernel {
 public:
  explicit SegmentReductionOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& segment_ids = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(segment_ids.shape()),
                errors::InvalidArgument("segment_ids should be a vector."));
    const int64 num_indices = segment_ids.NumElements();
    OP_REQUIRES(context, num_indices == input.dim_size(0),
                errors::InvalidArgument(
                    "segment_ids should be the same size as dimension 0 of"
                    " input."));

    auto input_flat = input.flat_outer_dims<T>();
    const int64 num_col = input_flat.dimension(1);

    const auto segment_vec = segment_ids.vec<Index>();
    // Note that the current implementation assumes that segment_vec values are
    // sorted.
    const Index output_rows =
        num_indices > 0 ? segment_vec(num_indices - 1) + 1 : 0;

    TensorShape output_shape = input.shape();
    output_shape.set_dim(0, output_rows);

    // Note that we do not initialize the output buffer with a default value.
    // We require that segment ids be sorted and cover all values (otherwise we
    // return an error).
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto output_flat = output->flat_outer_dims<T>();

#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::DSizes<Eigen::DenseIndex, 1> dims_to_reduce;
    dims_to_reduce[0] = 0;
#else
    Eigen::IndexList<Eigen::type2index<0>> dims_to_reduce;
#endif
    Index start = 0, end = 1;
    // TODO(agarwal): if this loop becomes a bottleneck, consider sharding it
    // across threads.
    Eigen::DSizes<Eigen::DenseIndex, 1> out_slice_shape(num_col);
    while (end <= num_indices) {
      if (end < num_indices) {
        if (segment_vec(start) == segment_vec(end)) {
          ++end;
          continue;
        }
        // We have a new segment here.  Verify that the segment ids grow by one
        // each time, so that we cover every possible output value.
        OP_REQUIRES(
            context, segment_vec(start) + 1 == segment_vec(end),
            errors::InvalidArgument("segment ids are not increasing by 1"));
      }

      // Process segment [start, end)
      const T* in_slice_ptr = &input_flat(start, 0);
      typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>,
                               Eigen::Unaligned> OutT;
      T* out_slice_ptr = &output_flat(segment_vec(start), 0);
      OutT out_slice(out_slice_ptr, out_slice_shape);
      // We don't use out_slice.device(context->egien_device<Device>)
      // because these pieces of work are likely to be very small and
      // the context switching overhead dwarfs any benefit we get from
      // using another thread to do this work.
      if (start == end - 1) {
        typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>,
                                 Eigen::Unaligned> InT;
        InT in_slice(in_slice_ptr, out_slice_shape);
        out_slice = in_slice;
      } else {
        Eigen::DSizes<Eigen::DenseIndex, 2> in_slice_shape(end - start,
                                                           num_col);
        typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>,
                                 Eigen::Unaligned> InT;
        InT in_slice(in_slice_ptr, in_slice_shape);

        out_slice = in_slice.reduce(dims_to_reduce, Reducer());
      }
      start = end;
      ++end;
    }
  }
};

#define REGISTER_CPU_KERNELS(type, index_type)                 \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("SegmentSum")                                       \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .TypeConstraint<index_type>("Tindices"),             \
      SegmentReductionOp<CPUDevice, type, index_type,          \
                         Eigen::internal::SumReducer<type>>);  \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("SegmentMean")                                      \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .TypeConstraint<index_type>("Tindices"),             \
      SegmentReductionOp<CPUDevice, type, index_type,          \
                         Eigen::internal::MeanReducer<type>>); \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("SegmentProd")                                      \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .TypeConstraint<index_type>("Tindices"),             \
      SegmentReductionOp<CPUDevice, type, index_type,          \
                         Eigen::internal::ProdReducer<type>>); \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("SegmentMin")                                       \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .TypeConstraint<index_type>("Tindices"),             \
      SegmentReductionOp<CPUDevice, type, index_type,          \
                         Eigen::internal::MinReducer<type>>);  \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("SegmentMax")                                       \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .TypeConstraint<index_type>("Tindices"),             \
      SegmentReductionOp<CPUDevice, type, index_type,          \
                         Eigen::internal::MaxReducer<type>>);

#define REGISTER_CPU_KERNELS_ALL(type) \
  REGISTER_CPU_KERNELS(type, int32);   \
  REGISTER_CPU_KERNELS(type, int64);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_KERNELS_ALL);
#undef REGISTER_CPU_KERNELS
#undef REGISTER_CPU_KERNELS_ALL

// Similar to SegmentReductionOp but can handle unsorted segment definitions and
// specifying size of output.
template <typename Device, class T, class Index>
class UnsortedSegmentSumOp : public OpKernel {
 public:
  explicit UnsortedSegmentSumOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& data = context->input(0);
    const Tensor& segment_ids = context->input(1);
    const Tensor& num_segments = context->input(2);

    OP_REQUIRES(
        context, TensorShapeUtils::IsLegacyScalar(num_segments.shape()),
        errors::InvalidArgument("num_segments should be a scalar, not shape ",
                                num_segments.shape().ShortDebugString()));

    OP_REQUIRES(context,
                TensorShapeUtils::StartsWith(data.shape(), segment_ids.shape()),
                errors::InvalidArgument(
                    "data.shape = ", data.shape().ShortDebugString(),
                    " does not start with segment_ids.shape = ",
                    segment_ids.shape().ShortDebugString()));

    const auto segment_flat = segment_ids.flat<Index>();
    const int32 N = segment_flat.dimension(0);
    const int32 output_rows = num_segments.scalar<int32>()();

    if (N > 0) {
      Eigen::Tensor<Index, 0, Eigen::RowMajor> m = segment_flat.maximum();
      OP_REQUIRES(
          context, m() < output_rows,
          errors::InvalidArgument("More segments found than output size"));
    }

    TensorShape output_shape;
    output_shape.AddDim(output_rows);
    for (int i = segment_ids.dims(); i < data.dims(); i++) {
      output_shape.AddDim(data.dim_size(i));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto output_flat = output->flat_outer_dims<T>();
    output_flat.setZero();

    if (data.NumElements() > 0) {
      auto data_flat = data.shaped<T, 2>({N, data.NumElements() / N});
      for (int i = 0; i < N; ++i) {
        output_flat.template chip<0>(segment_flat(i)) +=
            data_flat.template chip<0>(i);
      }
    }
  }
};

#define REGISTER_CPU_UNSORTED_KERNELS(type, index_type)                \
  REGISTER_KERNEL_BUILDER(Name("UnsortedSegmentSum")                   \
                              .Device(DEVICE_CPU)                      \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices"), \
                          UnsortedSegmentSumOp<CPUDevice, type, index_type>);

#define REGISTER_CPU_UNSORTED_KERNELS_ALL(type) \
  REGISTER_CPU_UNSORTED_KERNELS(type, int32);   \
  REGISTER_CPU_UNSORTED_KERNELS(type, int64);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_UNSORTED_KERNELS_ALL);
#undef REGISTER_CPU_UNSORTED_KERNELS
#undef REGISTER_CPU_UNSORTED_KERNELS_ALL

// Same as SegmentReductionOp but takes as input a "sparse" tensor, represented
// by two dense tensors, one containing the data, and the other containing
// indices into the data.
template <typename Device, class T>
class SparseSegmentReductionOpBase : public OpKernel {
 public:
  explicit SparseSegmentReductionOpBase(OpKernelConstruction* context,
                                        bool is_mean)
      : OpKernel(context), is_mean_(is_mean) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& segment_ids = context->input(2);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices should be a vector."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(segment_ids.shape()),
                errors::InvalidArgument("segment_ids should be a vector."));

    const int32 num_indices = indices.NumElements();
    OP_REQUIRES(context, num_indices == segment_ids.NumElements(),
                errors::InvalidArgument(
                    "segment_ids and indices should have same size."));

    auto input_flat = input.flat_outer_dims<T>();

    const auto indices_vec = indices.vec<int32>();
    const auto segment_vec = segment_ids.vec<int32>();
    // Note that the current implementation assumes that segment_vec values are
    // sorted.
    const int32 output_rows =
        num_indices > 0 ? segment_vec(num_indices - 1) + 1 : 0;

    TensorShape output_shape = input.shape();
    output_shape.set_dim(0, output_rows);

    // Note that we do not initialize the output buffer with a default value.
    // We require that segment ids be sorted and cover all values (otherwise we
    // return an error).
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (num_indices == 0) return;
    auto output_flat = output->flat_outer_dims<T>();

    int32 start = 0, end = 1;
    while (end <= num_indices) {
      if (end < num_indices) {
        if (segment_vec(start) == segment_vec(end)) {
          ++end;
          continue;
        }
        // We have a new segment here.  Verify that the segment ids grow by one
        // each time, so that we cover every possible output value.
        OP_REQUIRES(
            context, segment_vec(start) + 1 == segment_vec(end),
            errors::InvalidArgument("segment ids are not increasing by 1"));
      }

      auto out = output_flat.template chip<0>(segment_vec(start));
#define I(i) input_flat.template chip<0>(indices_vec(start + i))
      int num = end - start;
      if (num == 1) {
        out = I(0);
      } else {
        int r = num % 8;
        T m = (is_mean_ && (num < 10)) ? num : 1;
        switch (r) {
          case 2:
            out = (I(0) + I(1)) / m;
            break;
          case 3:
            out = (I(0) + I(1) + I(2)) / m;
            break;
          case 4:
            out = (I(0) + I(1) + I(2) + I(3)) / m;
            break;
          case 5:
            out = (I(0) + I(1) + I(2) + I(3) + I(4)) / m;
            break;
          case 6:
            out = (I(0) + I(1) + I(2) + I(3) + I(4) + I(5)) / m;
            break;
          case 7:
            out = (I(0) + I(1) + I(2) + I(3) + I(4) + I(5) + I(6)) / m;
            break;
          case 0:
            out = (I(0) + I(1) + I(2) + I(3) + I(4) + I(5) + I(6) + I(7)) / m;
            r = 8;
            break;
          case 1:
            out =
                (I(0) + I(1) + I(2) + I(3) + I(4) + I(5) + I(6) + I(7) + I(8)) /
                m;
            r = 9;
            break;
        }
        for (; r < num; r += 8) {
          out += I(r) + I(r + 1) + I(r + 2) + I(r + 3) + I(r + 4) + I(r + 5) +
                 I(r + 6) + I(r + 7);
        }
#undef I
        if (is_mean_ && num >= 10) {
          out = out / static_cast<T>(num);
        }
      }
      start = end;
      ++end;
    }
  }

 private:
  bool is_mean_;
};

template <typename Device, class T>
class SparseSegmentReductionMeanOp
    : public SparseSegmentReductionOpBase<Device, T> {
 public:
  explicit SparseSegmentReductionMeanOp(OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T>(context, true /*is_mean*/) {}
};

template <typename Device, class T>
class SparseSegmentReductionSumOp
    : public SparseSegmentReductionOpBase<Device, T> {
 public:
  explicit SparseSegmentReductionSumOp(OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T>(context, false /*is_mean*/) {}
};

#define REGISTER_CPU_SPARSE_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("SparseSegmentSum").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseSegmentReductionSumOp<CPUDevice, type>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_SPARSE_KERNELS);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type)                                     \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("SparseSegmentMean").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseSegmentReductionMeanOp<CPUDevice, type>);
REGISTER_CPU_SPARSE_KERNELS(float);
REGISTER_CPU_SPARSE_KERNELS(double);
#undef REGISTER_CPU_SPARSE_KERNELS

template <class T>
class SparseSegmentMeanGradOp : public OpKernel {
 public:
  explicit SparseSegmentMeanGradOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& segment_ids = context->input(2);
    const Tensor& output_dim0 = context->input(3);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices should be a vector."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(segment_ids.shape()),
                errors::InvalidArgument("segment_ids should be a vector."));
    OP_REQUIRES(context, TensorShapeUtils::IsLegacyScalar(output_dim0.shape()),
                errors::InvalidArgument("output_dim0 should be a scalar."));

    const int64 N = indices.NumElements();
    OP_REQUIRES(context, N == segment_ids.NumElements(),
                errors::InvalidArgument(
                    "segment_ids and indices should have same size."));
    const int32 M = output_dim0.scalar<int32>()();

    auto input_flat = input.flat_outer_dims<T>();
    const auto indices_vec = indices.vec<int32>();
    const auto segment_vec = segment_ids.vec<int32>();

    TensorShape output_shape = input.shape();
    output_shape.set_dim(0, M);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (M == 0 || N == 0) return;

    // Note that similar to SparseSegmentMean, we assume that segment_vec is
    // already sorted and has non-negative values.
    int num_segments = segment_vec(N - 1) + 1;
    OP_REQUIRES(context, input.dim_size(0) == num_segments,
                errors::InvalidArgument("Invalid number of segments"));

    // Compute scaling factors for input.
    std::vector<double> scaling(num_segments, 0.0);
    for (int64 i = 0; i < N; ++i) {
      scaling[segment_vec(i)] += 1;
    }
    for (int i = 0; i < scaling.size(); ++i) {
      scaling[i] = 1.0 / std::max(scaling[i], 1.0);
    }

    auto output_flat = output->flat_outer_dims<T>();
    output_flat.setZero();
    std::vector<bool> is_modified(M, false);

    for (int64 i = 0; i < N; ++i) {
      int output_idx = indices_vec(i);
      int idx = segment_vec(i);
      T scale = static_cast<T>(scaling[idx]);
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
};

#define REGISTER_CPU_SPARSE_KERNELS(type)                 \
  REGISTER_KERNEL_BUILDER(Name("SparseSegmentMeanGrad")   \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<type>("T"), \
                          SparseSegmentMeanGradOp<type>);

REGISTER_CPU_SPARSE_KERNELS(float);
REGISTER_CPU_SPARSE_KERNELS(double);

#undef REGISTER_CPU_SPARSE_KERNELS
}  // namespace tensorflow
