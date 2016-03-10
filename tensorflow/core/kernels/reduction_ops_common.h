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

// This is an internal header file intended to only be included as the
// front-matter in the implementation files of various reduction ops.  It
// is a header file because we split the various reduction ops into their
// own compilation units to get more parallelism in compilation.

#ifndef TENSORFLOW_KERNELS_REDUCTION_OPS_COMMON_H_
#define TENSORFLOW_KERNELS_REDUCTION_OPS_COMMON_H_

#define EIGEN_USE_THREADS

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/reduction_ops.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"

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

class ReductionHelper {
 public:
  ReductionHelper() : reduce_first_axis_(false) {}

  Status Simplify(const Tensor& data, const Tensor& axis, const bool keep_dims);

  // We need to do roughly:
  //   tmp_out = allocate(out_reshape())
  //   tmp_out.reshape(out_reshape) = data.reshape(data_reshape).reduce(axes)
  //   out = tmp_out.reshape(out_shape)

  // The reduction result must be allocated with this shape.
  TensorShape out_reshape() const;

  // The final output shape must be allocated with this shape.
  TensorShape out_shape() const;

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

  // Shape of shuffled input
  TensorShape data_reshape() const {
    TensorShape shape;
    for (auto s : data_reshape_) shape.AddDim(s);
    return shape;
  }

  // Shape with all reduction dimensions at the end
  TensorShape shuffled_shape();

  // Permutation of reduced dims needed to put reduction dimensions at the end
  gtl::InlinedVector<int32, 8> permutation();

 private:
  bool reduce_first_axis_;  // True if need to reduce the 0-th dimension.
  gtl::InlinedVector<int64, 4> data_reshape_;  // Reshape data before reduction.
  gtl::InlinedVector<int64, 4> out_shape_;     // The final output shape.
  gtl::InlinedVector<int64, 4> out_reshape_;   // Reshape output for reduction.
};

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
    VLOG(1) << "data shape: " << data.shape().DebugString();
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

    // We must allocate temp tensors using the same alloc attr as
    // output(0) because it is returned as output(0) in the end.
    const AllocatorAttributes alloc_attr = ctx->output_alloc_attr(0);

    // A temporary tensor whose size matches the size of the reduced
    // output.
    Tensor tmp_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(out->dtype(), helper.out_reshape(),
                                           &tmp_out, alloc_attr));

    typedef functor::ReduceFunctor<Device, Reducer> Functor;
    Constants<Device> constants;
    const Device& d = ctx->eigen_device<Device>();
    Reducer reducer;

    if (tmp_out.NumElements() == 0) {
      // Nothing to do, fall through to final reshaping.
    }
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
      // If we don't hit one of the cases above, transpose the data so that
      // all reduced dimensions are last and reuse the 2-D -> 1-D case.
      Tensor data_reshaped;
      CHECK(data_reshaped.CopyFrom(data, helper.data_reshape()));
      Tensor shuffled;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                             helper.shuffled_shape(), &shuffled,
                                             alloc_attr));
      OP_REQUIRES_OK(
          ctx, DoTranspose(d, data_reshaped, helper.permutation(), &shuffled));
      const int64 unreduced = tmp_out.NumElements();
      const int64 reduced = shuffled.NumElements() / unreduced;
      const Tensor& const_shuffled = shuffled;
      Functor::Reduce(d, tmp_out.flat<T>(),
                      const_shuffled.shaped<T, 2>({unreduced, reduced}),
                      constants.kOne, reducer);
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

template <typename Reducer>
struct ReduceFunctor<CPUDevice, Reducer> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(const CPUDevice& d, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Reducer& reducer) {
    ReduceEigenImpl(d, out, in, reduction_axes, reducer);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_REDUCTION_OPS_COMMON_H_
