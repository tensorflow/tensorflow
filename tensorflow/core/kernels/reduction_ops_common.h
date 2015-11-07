// This is an internal header file intended to only be included as the
// front-matter in the implementation files of various reduction ops.  It
// is a header file because we split the various reduction ops into their
// own compilation units to get more parallelism in compilation.

#ifndef TENSORFLOW_KERNELS_REDUCTION_OPS_COMMON_H_
#define TENSORFLOW_KERNELS_REDUCTION_OPS_COMMON_H_

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/reduction_ops.h"

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/tensor.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/public/status.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device>
struct Constants {
  // Derive Index type. int (32-bit) or long (64-bit) depending on the
  // compile-time configuration. "float" here is not relevant.
  // TODO(zhifengc): Moves the definition to TTypes.
  typedef TTypes<float>::Tensor::Index Index;
  Eigen::array<Index, 1> kZero;
  Eigen::array<Index, 1> kOne;
  Eigen::array<Index, 2> kZeroTwo;

  Constants() {
    kZero[0] = 0;
    kOne[0] = 1;
    kZeroTwo[0] = 0;
    kZeroTwo[1] = 2;
  }
};

#if defined(EIGEN_HAS_INDEX_LIST)
template <>
struct Constants<CPUDevice> {
  const Eigen::IndexList<Eigen::type2index<0>> kZero;
  const Eigen::IndexList<Eigen::type2index<1>> kOne;
  const Eigen::IndexList<Eigen::type2index<0>, Eigen::type2index<2>> kZeroTwo;
};
#endif

namespace {

class ReductionHelper {
 public:
  ReductionHelper() : reduce_first_axis_(false) {}

  Status Simplify(const Tensor& data, const Tensor& axis,
                       const bool keep_dims) {
    // bitmap[i] indicates whether to reduce data along i-th axis.
    std::vector<bool> bitmap(data.dims(), false);
    auto axis_vec = axis.flat<int32>();
    for (int64 i = 0; i < axis.NumElements(); ++i) {
      const int32 index = axis_vec(i);
      if (index < 0 || index >= data.dims()) {
        return errors::OutOfRange("Invalid reduction dimension (", index,
                                  " for input with ", data.dims(),
                                  " dimension(s)");
      }
      bitmap[index] = true;
    }

    // Output tensor's dim sizes.
    out_shape_.clear();
    for (int i = 0; i < data.dims(); ++i) {
      if (!bitmap[i]) {
        // If we are not reducing along dimension i.
        out_shape_.push_back(data.dim_size(i));
      } else if (keep_dims) {
        // We are reducing along dimension i, but we want to keep the
        // same number of dimensions, so we set the dimension of i to
        // '1'.
        out_shape_.push_back(1);
      }
    }

    // Depending on bitmap[i] and bitmap[i-1], we can collapse axis of
    // the input data before doing the reduction on the resulting
    // tensor.  The shape of the reduction is a reshape of the final
    // output.

    // We'll skip the leading 1s.
    int dim_index = 0;
    for (; dim_index < data.dims(); ++dim_index) {
      if (data.dim_size(dim_index) != 1) break;
    }
    if (dim_index >= data.dims()) {
      // Special case. The input is essentially a scalar.
      reduce_first_axis_ = true;
    } else {
      // Starting from the (dim_index)-th dimension, dimensions
      // alternates between runs that need to be reduced and runs that
      // don't.
      //
      // NOTE: If a dimension has size 1, we group it as the current
      // run so that we can minimize the number of runs.
      //
      // E.g., when we want to reduce a tensor of shape [2, 1, 3, 1,
      // 5] by axes = [1, 4], we should treat the tensor as a [6, 5]
      // and reduce by axes = [1] (i.e., the output is shape [6]).
      reduce_first_axis_ = bitmap[dim_index];
      data_reshape_.push_back(data.dim_size(dim_index));
      ++dim_index;
      for (; dim_index < data.dims(); ++dim_index) {
        const auto size = data.dim_size(dim_index);
        if (size == 1) {
          bitmap[dim_index] = bitmap[dim_index - 1];
        }
        if (bitmap[dim_index - 1] != bitmap[dim_index]) {
          // Starts a new run of reduce or !reduce.
          data_reshape_.push_back(size);
        } else {
          // Continue a run of reduce or !reduce.
          data_reshape_.back() *= size;
        }
      }
      // If reduce_first_axis_ is true (input's dimension 0, 2, 4, etc
      // are reduced), data_reshape_[1, 3, 5, ...]  is out_reshape_,
      // otherwise, data_reshape_[0, 2, 4, ...] is.
      for (size_t i = reduce_first_axis_ ? 1 : 0; i < data_reshape_.size();
           i += 2) {
        out_reshape_.push_back(data_reshape_[i]);
      }
    }

    VLOG(1) << "data reshape: " << str_util::Join(data_reshape_, ",");
    VLOG(1) << "out  reshape: " << str_util::Join(out_reshape_, ",");
    VLOG(1) << "out    shape: " << str_util::Join(out_shape_, ",");
    return Status::OK();
  }

  // We need to do roughly:
  //   tmp_out = allocate(out_reshape())
  //   tmp_out.reshape(out_reshape) = data.reshape(data_reshape).reduce(axes)
  //   out = tmp_out.reshape(out_shape)

  // The reduction result must be allocated with this shape.
  TensorShape out_reshape() const {
    TensorShape shape;
    for (auto size : out_reshape_) shape.AddDim(size);
    return shape;
  }

  // The final output shape must be allocated with this shape.
  TensorShape out_shape() const {
    TensorShape shape;
    for (auto size : out_shape_) shape.AddDim(size);
    return shape;
  }

  // The reduction is on a reshaped tensor of this rank.
  int ndims() const { return data_reshape_.size(); }

  // True if need to reduce the 0-th dimension.
  bool reduce_first_axis() const { return reduce_first_axis_; }

  // The output is reshaped.
  template <typename T, int N>
  typename TTypes<T, N>::Tensor out(Tensor* out) {
    return out->shaped<T, N>(out_reshape_);
  }

  // The input is reshaped.
  template <typename T, int N>
  typename TTypes<T, N>::ConstTensor in(const Tensor& data) {
    return data.shaped<T, N>(data_reshape_);
  }

 private:
  bool reduce_first_axis_;      // True if need to reduce the 0-th dimension.
  std::vector<int64> data_reshape_;  // Reshape the data before reduction.
  std::vector<int64> out_shape_;     // The final output shape.
  std::vector<int64> out_reshape_;   // Reshape the output for reduction.
};

}  // end namespace

// For operations where the output is a reduction function along some
// dimensions of the input.
template <typename Device, class T, typename Reducer>
class ReductionOp : public OpKernel {
 public:
  explicit ReductionOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(ctx, ctx->MatchSignature({dt, DT_INT32}, {dt}));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("keep_dims", &keep_dims_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& data = ctx->input(0);
    const Tensor& axes = ctx->input(1);
    VLOG(1) << "data shape: " << data.shape().ShortDebugString();
    VLOG(1) << "axes      : " << axes.SummarizeValue(10);

    ReductionHelper helper;
    OP_REQUIRES_OK(ctx, helper.Simplify(data, axes, keep_dims_));
    CHECK_GE(helper.ndims(), 0);

    // The real output shape will be assigned below.
    TensorShape empty_shape;
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, empty_shape, &out));

    if (helper.ndims() == 0 ||
        (helper.ndims() == 1 && !helper.reduce_first_axis())) {
      // Special case. Reduces nothing.  It is unclear why this is
      // necessary, but tests fail without it.  Look into why this
      // case occurs.
      if (!out->CopyFrom(data, helper.out_shape())) {
        ctx->SetStatus(errors::Internal("Error during reduction copy."));
      }
      return;
    }

    // A temporary tensor whose size matches the size of the reduced
    // output.
    Tensor tmp_out;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(out->dtype(), helper.out_reshape(), &tmp_out));

    typedef functor::ReduceFunctor<Device> Functor;
    Constants<Device> constants;
    const Device& d = ctx->eigen_device<Device>();
    Reducer reducer;

    if ((helper.ndims() == 1) && helper.reduce_first_axis()) {
      // Reduce to a scalar.
      Functor::Reduce(d, helper.out<T, 0>(&tmp_out), helper.in<T, 1>(data),
                      constants.kZero, reducer);
    } else if ((helper.ndims() == 2) && helper.reduce_first_axis()) {
      // Can be viewed as a reduction of a matrix along 1st dimension.
      Functor::Reduce(d, helper.out<T, 1>(&tmp_out), helper.in<T, 2>(data),
                      constants.kZero, reducer);
    } else if ((helper.ndims() == 2) && !helper.reduce_first_axis()) {
      // Can be viewed as a reduction of a matrix along 2nd dimension.
      Functor::Reduce(d, helper.out<T, 1>(&tmp_out), helper.in<T, 2>(data),
                      constants.kOne, reducer);
    } else if ((helper.ndims() == 3) && helper.reduce_first_axis()) {
      // Can be viewed as a reduction of a 3D tensor along 1st and 3rd
      // dimensions.
      Functor::Reduce(d, helper.out<T, 1>(&tmp_out), helper.in<T, 3>(data),
                      constants.kZeroTwo, reducer);
    } else if ((helper.ndims() == 3) && !helper.reduce_first_axis()) {
      // Can be viewed as a reduction of a 3D tensor along 2nd dimension.
      Functor::Reduce(d, helper.out<T, 2>(&tmp_out), helper.in<T, 3>(data),
                      constants.kOne, reducer);
    } else {
      // TODO(zhifengc): We can implement reduction for arbitrary rank
      // tensor and arbitrary reduction axes by iterating the reduction
      // multiple times. This may also be accomplished in the graph
      // construction.
      ctx->SetStatus(
          errors::Unimplemented("Reducing ", data.shape().ShortDebugString(),
                                " axes [", axes.SummarizeValue(10), "] to ",
                                tmp_out.shape().ShortDebugString()));
      return;
    }

    // Set the real output using the contents of the reduction but the
    // real expected output shape.  The number of elements should
    // match between the two shapes.
    if (!out->CopyFrom(tmp_out, helper.out_shape())) {
      ctx->SetStatus(errors::Internal("Error during reduction copy."));
    }
  }

 private:
  // True if the number of dimensions should be maintained.
  bool keep_dims_;
};

namespace functor {

template <>
struct ReduceFunctor<CPUDevice> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes,
            typename Reducer>
  static void Reduce(const CPUDevice& d, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Reducer& reducer) {
    ReduceEigenImpl(d, out, in, reduction_axes, reducer);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_REDUCTION_OPS_COMMON_H_
