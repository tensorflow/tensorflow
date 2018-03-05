/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <array>
#include <limits>
#include <type_traits>
#include <vector>

#include "tensorflow/contrib/coder/kernels/range_coder.h"
#include "tensorflow/contrib/coder/kernels/range_coder_ops_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {
// A helper class to iterate over data and cdf simultaneously, while cdf is
// broadcasted to data.
// NOTE: Moving this class out of anonymous namespace impacts compiler
// optimization and affects performance. When moving this code around (e.g.,
// into a library header), be sure to check the benchmark tests.
template <typename T, typename U, int N>
class BroadcastRange {
 public:
  BroadcastRange(T* data_pointer, gtl::ArraySlice<int64> data_shape,
                 const U* cdf_pointer, gtl::ArraySlice<int64> cdf_shape)
      : data_pointer_(data_pointer), cdf_pointer_(cdf_pointer) {
    CHECK(!data_shape.empty());
    CHECK_EQ(data_shape.size(), N);
    CHECK_EQ(cdf_shape.size(), N + 1);

    std::copy(data_shape.begin(), data_shape.end(), &data_shape_[0]);
    data_index_.fill(0);

    const int64 innermost_stride = cdf_shape[N];
    cdf_displace_.fill(innermost_stride);

    // Pre-compute the pointer displacement for cdf.
    int64 stride = innermost_stride;
    for (int i = N - 1; i >= 0; --i) {
      const bool broadcasting = (cdf_shape[i] <= 1);

      // When the data linear index advances by one, the cdf linear index
      // advances by `innermost_stride`.
      //
      // Suppose that the i-th axis coordinate of data increased by one, and
      // that i-th axis is broadcasting. The cdf linear index should be wound
      // back by i-th axis stride, so that i-th axis coordinate of cdf is
      // effectively kept at 0.
      if (broadcasting) {
        cdf_displace_[i] -= stride;
      }
      stride *= cdf_shape[i];
    }
  }

  // Returns the pointers to the current iterating locations to data and cdf
  // tensors.
  //
  // Note that this function does not track whether data pointer is running past
  // the end of data buffer. The caller has to make sure Next() is called no
  // more than that.
  std::pair<T*, const U*> Next() {
    std::pair<T*, const U*> return_value = {data_pointer_, cdf_pointer_};

    int i = N - 1;
    for (; i > 0; --i) {
      ++data_index_[i];
      if (data_index_[i] < data_shape_[i]) {
        break;
      }
      data_index_[i] = 0;
    }

    // Advance data pointer by one.
    data_pointer_ += 1;

    // For cdf pointer, it's more complicated because of broadcasting. When i-th
    // coordinate increase by one, and if i-th axis is broadcasting, then we
    // need to rewind back the pointer so that the effective i-th axis
    // coordinate for cdf is always 0. This value is precomputed as
    // cdf_displace_.
    cdf_pointer_ += cdf_displace_[i];
    return return_value;
  }

 private:
  std::array<int64, N> data_shape_;
  std::array<int64, N> cdf_displace_;
  std::array<int64, N> data_index_;

  T* data_pointer_;
  const U* cdf_pointer_;
};

Status CheckCdfShape(const TensorShape& data_shape,
                     const TensorShape& cdf_shape) {
  if (TF_PREDICT_FALSE(cdf_shape.dims() != data_shape.dims() + 1)) {
    return errors::InvalidArgument(
        "`cdf` should have one more axis than `data`: data shape=",
        data_shape.DebugString(), ", cdf shape=", cdf_shape.DebugString());
  }

  if (TF_PREDICT_FALSE(cdf_shape.dim_size(cdf_shape.dims() - 1) <= 1)) {
    return errors::InvalidArgument(
        "The last dimension of `cdf` should be > 1: ", cdf_shape.DebugString());
  }

  return Status::OK();
}

// Non-incremental encoder op -------------------------------------------------
class RangeEncodeOp : public OpKernel {
 public:
  explicit RangeEncodeOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("precision", &precision_));
    OP_REQUIRES(context, 0 < precision_ && precision_ <= 16,
                errors::InvalidArgument("`precision` must be in [1, 16]: ",
                                        precision_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& data = context->input(0);
    const Tensor& cdf = context->input(1);

    OP_REQUIRES_OK(context, CheckCdfShape(data.shape(), cdf.shape()));

    std::vector<int64> data_shape, cdf_shape;
    OP_REQUIRES_OK(
        context, MergeAxes(data.shape(), cdf.shape(), &data_shape, &cdf_shape));

    Tensor* output_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape{}, &output_tensor));
    string* output = &output_tensor->scalar<string>()();

    switch (data_shape.size()) {
#define RANGE_ENCODE_CASE(dims)                                                \
  case dims: {                                                                 \
    RangeEncodeImpl<dims>(data.flat<int16>(), data_shape,                      \
                          cdf.flat_inner_dims<int32, 2>(), cdf_shape, output); \
  } break
      RANGE_ENCODE_CASE(1);
      RANGE_ENCODE_CASE(2);
      RANGE_ENCODE_CASE(3);
      RANGE_ENCODE_CASE(4);
      RANGE_ENCODE_CASE(5);
      RANGE_ENCODE_CASE(6);
#undef RANGE_ENCODE_CASE
      default:
        context->CtxFailure(errors::InvalidArgument(
            "Irregular broadcast pattern: ", data.shape().DebugString(), ", ",
            cdf.shape().DebugString()));
        return;
    }
  }

 private:
  template <int N>
  void RangeEncodeImpl(TTypes<int16>::ConstFlat data,
                       gtl::ArraySlice<int64> data_shape,
                       TTypes<int32>::ConstMatrix cdf,
                       gtl::ArraySlice<int64> cdf_shape, string* output) const {
    const int64 data_size = data.size();
    const int64 cdf_size = cdf.size();
    const int64 chip_size = cdf.dimension(1);

    BroadcastRange<const int16, int32, N> view{data.data(), data_shape,
                                               cdf.data(), cdf_shape};
    RangeEncoder encoder{precision_};
    for (int64 linear = 0; linear < data_size; ++linear) {
      const auto pair = view.Next();

      const int64 index = *pair.first;
      DCHECK_GE(index, 0);
      DCHECK_LT(index + 1, chip_size);

      const int32* cdf_slice = pair.second;
      DCHECK_LE(cdf_slice + chip_size, cdf.data() + cdf_size);

      const int32 lower = cdf_slice[index];
      const int32 upper = cdf_slice[index + 1];
      encoder.Encode(lower, upper, output);
    }

    encoder.Finalize(output);
  }

  int precision_;
};

REGISTER_KERNEL_BUILDER(Name("RangeEncode").Device(DEVICE_CPU), RangeEncodeOp);

// Non-incremental decoder op -------------------------------------------------
class RangeDecodeOp : public OpKernel {
 public:
  explicit RangeDecodeOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("precision", &precision_));
    OP_REQUIRES(context, 0 < precision_ && precision_ <= 16,
                errors::InvalidArgument("`precision` must be in [1, 16]: ",
                                        precision_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& encoded_tensor = context->input(0);
    const Tensor& shape = context->input(1);
    const Tensor& cdf = context->input(2);

    OP_REQUIRES(context, TensorShapeUtils::IsScalar(encoded_tensor.shape()),
                errors::InvalidArgument("Invalid `encoded` shape: ",
                                        encoded_tensor.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(shape.shape()),
                errors::InvalidArgument("Invalid `shape` shape: ",
                                        shape.shape().DebugString()));
    TensorShape output_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(shape.vec<int32>(),
                                                        &output_shape));
    OP_REQUIRES_OK(context, CheckCdfShape(output_shape, cdf.shape()));

    std::vector<int64> data_shape, cdf_shape;
    OP_REQUIRES_OK(
        context, MergeAxes(output_shape, cdf.shape(), &data_shape, &cdf_shape));

    const string& encoded = encoded_tensor.scalar<string>()();

    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    switch (data_shape.size()) {
#define RANGE_DECODE_CASE(dim)                                              \
  case dim: {                                                               \
    RangeDecodeImpl<dim>(output->flat<int16>(), data_shape,                 \
                         cdf.flat_inner_dims<int32>(), cdf_shape, encoded); \
  } break
      RANGE_DECODE_CASE(1);
      RANGE_DECODE_CASE(2);
      RANGE_DECODE_CASE(3);
      RANGE_DECODE_CASE(4);
      RANGE_DECODE_CASE(5);
      RANGE_DECODE_CASE(6);
#undef RANGE_DECODE_CASE
      default:
        context->CtxFailure(errors::InvalidArgument(
            "Irregular broadcast pattern: ", output_shape.DebugString(), ", ",
            cdf.shape().DebugString()));
        return;
    }
  }

 private:
  template <int N>
  void RangeDecodeImpl(TTypes<int16>::Flat output,
                       gtl::ArraySlice<int64> output_shape,
                       TTypes<int32>::ConstMatrix cdf,
                       gtl::ArraySlice<int64> cdf_shape,
                       const string& encoded) const {
    BroadcastRange<int16, int32, N> view{output.data(), output_shape,
                                         cdf.data(), cdf_shape};

    RangeDecoder decoder{encoded, precision_};

    const int64 output_size = output.size();
    const int64 cdf_size = cdf.size();
    const auto chip_size =
        static_cast<gtl::ArraySlice<int32>::size_type>(cdf.dimension(1));

    for (int64 i = 0; i < output_size; ++i) {
      const auto pair = view.Next();

      int16* data = pair.first;
      DCHECK_LT(data, output.data() + output_size);

      const int32* cdf_slice = pair.second;
      DCHECK_LE(cdf_slice + chip_size, cdf.data() + cdf_size);

      *data = decoder.Decode(gtl::ArraySlice<int32>{cdf_slice, chip_size});
    }
  }

  int precision_;
};

REGISTER_KERNEL_BUILDER(Name("RangeDecode").Device(DEVICE_CPU), RangeDecodeOp);
}  // namespace
}  // namespace tensorflow
