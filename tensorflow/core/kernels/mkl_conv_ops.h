/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_MKL_CONV_OPS_H_
#define TENSORFLOW_CORE_KERNELS_MKL_CONV_OPS_H_

#include <limits>
#include <vector>
#include <memory>

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/conv_grad_ops.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

#include "tensorflow/core/util/mkl_util.h"

#ifndef INTEL_MKL_ML
#include "mkldnn.hpp"

using mkldnn::prop_kind;
using mkldnn::stream;

using mkldnn::convolution_direct;
using mkldnn::convolution_forward;
#endif

namespace tensorflow {

#ifndef INTEL_MKL_ML

class MklDnnConvUtil {
 protected:
  OpKernelContext* context_;  // We don't own this.
  std::vector<int32> strides_;
  std::vector<int32> dilations_;
  Padding padding_;
  TensorFormat data_format_;

 public:
  MklDnnConvUtil(OpKernelContext* context, const std::vector<int32>& strides,
                 Padding pad, TensorFormat fm,
                 const std::vector<int32>& dilations)
      : context_(context),
        strides_(strides),
        dilations_(dilations),
        padding_(pad),
        data_format_(fm) {}

  virtual ~MklDnnConvUtil() { context_ = nullptr; }

  // Calculate Convolution strides
  virtual inline void GetStridesInMklOrder(memory::dims* strides) {
    // For now we take the stride from the second and third dimensions only
    // (we do not support striding on the batch or depth dimension).
    CHECK_NOTNULL(strides);
    int stride_rows = GetTensorDim(strides_, data_format_, 'H');
    int stride_cols = GetTensorDim(strides_, data_format_, 'W');
    *strides = {stride_rows, stride_cols};
  }

  // Calculate Convolution dilations
  virtual inline void GetDilationsInMklOrder(memory::dims *dilations) {
    // For now we take the dilation from the second and third dimensions only
    // (we do not support dilation on the batch or depth dimension).
    CHECK_NOTNULL(dilations);
    int dilations_rows = GetTensorDim(dilations_, data_format_, 'H');
    int dilations_cols = GetTensorDim(dilations_, data_format_, 'W');
    *dilations = {dilations_rows, dilations_cols};
  }

  // Calculate Convolution input size in MKL-DNN order. MKL-DNN
  // requires input in NCHW format. Function does not return anything.
  // But errors arising from sanity checks are returned in context's
  // status.
  virtual inline void GetInputSizeInMklOrder(const TensorShape& input_shape,
                                             memory::dims* input_dims) {
#define CHECK_BOUNDS(val, err_msg)                                     \
  do {                                                                 \
    OP_REQUIRES(context_,                                              \
                FastBoundsCheck(val, std::numeric_limits<int>::max()), \
                errors::InvalidArgument(err_msg));                     \
  } while (0)

    CHECK_NOTNULL(input_dims);

    // Input channel
    int64 input_depth_raw = GetTensorDim(input_shape, data_format_, 'C');
    int input_depth = static_cast<int>(input_depth_raw);

    // Input rows/height
    int64 input_rows_raw = GetTensorDim(input_shape, data_format_, 'H');
    CHECK_BOUNDS(input_rows_raw, "Input rows too large");
    int input_rows = static_cast<int>(input_rows_raw);

    // Input columns/width
    int64 input_cols_raw = GetTensorDim(input_shape, data_format_, 'W');
    CHECK_BOUNDS(input_cols_raw, "Input cols too large");
    int input_cols = static_cast<int>(input_cols_raw);

    // Input batch
    int64 input_batch_raw = GetTensorDim(input_shape, data_format_, 'N');
    CHECK_BOUNDS(input_batch_raw, "Input batch too large");
    int input_batch = static_cast<int>(input_batch_raw);

#undef CHECK_BOUNDS

    // MKL-DNN always requires input in NCHW format.
    std::vector<int> mkldnn_sizes(4, -1);
    mkldnn_sizes[MklDnnDims::Dim_N] = input_batch;
    mkldnn_sizes[MklDnnDims::Dim_C] = input_depth;
    mkldnn_sizes[MklDnnDims::Dim_H] = input_rows;
    mkldnn_sizes[MklDnnDims::Dim_W] = input_cols;

    *input_dims = mkldnn_sizes;
  }

  // Calculate Convolution filter size in MKL-DNN order. MKL-DNN
  // requires filter in OIHW format. Function does not return anything.
  // But errors arising from sanity checks are returned in context's
  // status.
  //
  // Calculate Convolution filter size in MKL-DNN order. MKL-DNN
  // requires filter in OIHW format. Function does not return anything.
  // But errors arising from sanity checks are returned in context's
  // status. This function differs from GetConvFilterSizeInMklOrder in
  // parameter for input - it accepts src_shape since Convolution Backward
  // Input gets shape of input tensor rather than actual tensor (Convolution
  // forward gets actual tensor as input).
  //
  // TODO(nhasabni): Add similar function for input and filter in MklShape.
  virtual inline void GetFilterSizeInMklOrder(const TensorShape& input_shape,
                                              const TensorShape& filter_shape,
                                              memory::dims* filter_dims) {
    CHECK_NOTNULL(filter_dims);

    OP_REQUIRES(context_, filter_shape.dims() == 4,
                errors::InvalidArgument("filter must be 4-dimensional: ",
                                        filter_shape.DebugString()));

    for (int i = 0; i < 3; i++) {
      OP_REQUIRES(context_,
                  FastBoundsCheck(filter_shape.dim_size(i),
                                  std::numeric_limits<int>::max()),
                  errors::InvalidArgument("filter too large"));
    }

    int input_depth = GetTensorDim(input_shape, data_format_, 'C');

    OP_REQUIRES(context_, input_depth == filter_shape.dim_size(2),
                errors::InvalidArgument(
                    "input and filter must have the same depth: ", input_depth,
                    " vs ", filter_shape.dim_size(2)));

    // TF filter is always in (rows, cols, in_depth, out_depth) order.
    int filter_rows = static_cast<int>(filter_shape.dim_size(0));
    int filter_cols = static_cast<int>(filter_shape.dim_size(1));
    int in_depth = static_cast<int>(filter_shape.dim_size(2));
    int out_depth = static_cast<int>(filter_shape.dim_size(3));

    // MKL-DNN always needs filter in OIHW format.
    // OIHW = (out_depth, in_depth, rows, cols)
    std::vector<int> mkldnn_sizes(4, -1);
    mkldnn_sizes[MklDnnDims::Dim_O] = out_depth;
    mkldnn_sizes[MklDnnDims::Dim_I] = in_depth;
    mkldnn_sizes[MklDnnDims::Dim_H] = filter_rows;
    mkldnn_sizes[MklDnnDims::Dim_W] = filter_cols;

    *filter_dims = mkldnn_sizes;
  }

  // Calculate Convolution filter size in MKL-DNN order. MKL-DNN
  // requires filter in OIHW format. Function does not return anything.
  // But errors arising from sanity checks are returned in context's
  // status.
  virtual inline void GetFilterSizeInMklOrder(size_t src_index,
                                              size_t filter_index,
                                              memory::dims* filter_dims) {
    CHECK_NOTNULL(filter_dims);
    GetFilterSizeInMklOrder(GetTfShape(context_, src_index),
                            GetTfShape(context_, filter_index), filter_dims);
  }

  // Calculate Bias size for 2D Convolution. Function does not return
  // anything, but sets error in context status.
  virtual inline void GetBiasSizeInMklOrder(size_t bias_index,
                                            memory::dims* bias_dims) {
    const Tensor& bias = MklGetInput(context_, bias_index);
    OP_REQUIRES(context_, bias.dims() == 1,
                errors::InvalidArgument("bias must be 1-dimensional: ",
                                        bias.shape().DebugString()));

    *bias_dims = {static_cast<int>(bias.dim_size(0))};
  }

  // Function to calculate output and padding size for 2D convolution.
  //
  // Calculate output shape of Convolution in MKL-DNN and TensorFlow order.
  // MKL-DNN uses NCHW for output order. But TensorFlow output will be in
  // NHWC or NCHW format depending on data format. Function also calculates
  // left, right, top and bottom pads. Function does not return any status -
  // status is returned via context status.
  //
  // TODO(nhasabni): Add similar function for input and filter in MklShape.
  virtual inline void GetOutputAndPadSizeInMklOrder(
      const TensorShape& input_shape, const TensorShape& filter_shape,
      const memory::dims& strides, const memory::dims& dilations,
      memory::dims* output_dims_tf_order,
      memory::dims* output_dims_mkl_order, memory::dims* pad_l,
      memory::dims* pad_r) {
    CHECK_NOTNULL(output_dims_tf_order);
    CHECK_NOTNULL(output_dims_mkl_order);
    CHECK_NOTNULL(pad_l);
    CHECK_NOTNULL(pad_r);

    int input_rows = GetTensorDim(input_shape, data_format_, 'H');
    int input_cols = GetTensorDim(input_shape, data_format_, 'W');

    // The first dimension for filter is rows/height.
    int filter_rows = filter_shape.dim_size(0);
    // The second dimension for filter is cols/width.
    int filter_cols = filter_shape.dim_size(1);

    // Stride is vector of 2 elements: {s_r, s_c}
    int stride_rows = strides[0];
    int stride_cols = strides[1];
    int dilation_rows = dilations[0];
    int dilation_cols = dilations[1];

    // Output batch is same as input batch.
    int out_batch = GetTensorDim(input_shape, data_format_, 'N');
    // Output depth is same as last dimension for filter.
    int out_depth = filter_shape.dim_size(3);

    int64 out_rows = 0, out_cols = 0;
    int64 pad_top = 0, pad_bottom = 0, pad_left, pad_right;

    OP_REQUIRES_OK(context_,
            GetWindowedOutputSizeVerboseV2(input_rows, filter_rows,
                                 dilation_rows, stride_rows, padding_,
                                 &out_rows, &pad_top, &pad_bottom));
    OP_REQUIRES_OK(context_,
            GetWindowedOutputSizeVerboseV2(input_cols, filter_cols,
                                 dilation_cols, stride_cols, padding_,
                                 &out_cols, &pad_left, &pad_right));

    // Tensorflow output is in data_format order. (NHWC or NCHW)
    TensorShape out_shape =
        ShapeFromFormat(data_format_, out_batch, out_rows, out_cols, out_depth);
    *output_dims_tf_order = TFShapeToMklDnnDims(out_shape);

    // MKL-DNN always needs output in NCHW format.
    std::vector<int> mkldnn_sizes(4, -1);
    mkldnn_sizes[MklDnnDims::Dim_N] = out_batch;
    mkldnn_sizes[MklDnnDims::Dim_C] = out_depth;
    mkldnn_sizes[MklDnnDims::Dim_H] = static_cast<int>(out_rows);
    mkldnn_sizes[MklDnnDims::Dim_W] = static_cast<int>(out_cols);
    *output_dims_mkl_order = mkldnn_sizes;

    // Now handle padding. MKL-DNN uses asymetric padding.
    *pad_l = {static_cast<int>(pad_top), static_cast<int>(pad_left)};
    *pad_r = {static_cast<int>(pad_bottom), static_cast<int>(pad_right)};
  }

  // Calculate output and pad size of forward Convolution operator.
  // See comment on GetConvOutputAndPadSizeInMklOrder for parameters.
  //
  // Function does not return anything, but sets error in context status.
  inline void GetOutputAndPadSizeInMklOrder(
      size_t src_index, size_t filter_index,
      const memory::dims& strides, const memory::dims& dilations,
      memory::dims* output_dims_tf_order, memory::dims* output_dims_mkl_order,
      memory::dims* pad_l, memory::dims* pad_r) {
    CHECK_NOTNULL(output_dims_tf_order);
    CHECK_NOTNULL(output_dims_mkl_order);
    CHECK_NOTNULL(pad_l);
    CHECK_NOTNULL(pad_r);

    auto input_tf_shape = GetTfShape(context_, src_index);
    auto filter_tf_shape = GetTfShape(context_, filter_index);

    OP_REQUIRES(context_, input_tf_shape.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input_tf_shape.DebugString()));

    GetOutputAndPadSizeInMklOrder(input_tf_shape, filter_tf_shape,
                                  strides, dilations, output_dims_tf_order,
                                  output_dims_mkl_order, pad_l, pad_r);
  }

  // Wrapper function to calculate input, filter, and output sizes of
  // 2D Convolution in MKL order (NCHW for input and output; OIHW for filter.)
  // Function also calculates output shape in Tensorflow order. Additionally, it
  // also calculates strides and paddings for 2D Convolution.
  //
  // Function does not return anything, but sets error in context status.
  inline void GetConvFwdSizesInMklOrder(
      const TensorShape& input_shape, const TensorShape& filter_shape,
      memory::dims* input_dims, memory::dims* filter_dims,
      memory::dims* strides, memory::dims *dilations,
      memory::dims* output_dims_tf_order,
      memory::dims* output_dims_mkl_order, memory::dims* pad_l,
      memory::dims* pad_r) {
    CHECK_NOTNULL(input_dims);
    CHECK_NOTNULL(filter_dims);
    CHECK_NOTNULL(strides);
    CHECK_NOTNULL(dilations);
    CHECK_NOTNULL(output_dims_tf_order);
    CHECK_NOTNULL(output_dims_mkl_order);
    CHECK_NOTNULL(pad_l);
    CHECK_NOTNULL(pad_r);

    GetInputSizeInMklOrder(input_shape, input_dims);
    if (!context_->status().ok()) return;
    GetFilterSizeInMklOrder(input_shape, filter_shape, filter_dims);
    if (!context_->status().ok()) return;
    GetStridesInMklOrder(strides);
    GetDilationsInMklOrder(dilations);
    GetOutputAndPadSizeInMklOrder(input_shape, filter_shape,
                                  *strides, *dilations,
                                  output_dims_tf_order, output_dims_mkl_order,
                                  pad_l, pad_r);
    if (!context_->status().ok()) return;
  }
};


/////////////////////////////////////////////////////////////////////
///  Common class that implements Conv2DBackpropFilter and Input
/////////////////////////////////////////////////////////////////////

template <typename Device, class T>
class MklConv2DBackpropCommonOp : public OpKernel {
 public:
  ~MklConv2DBackpropCommonOp() {}
  explicit MklConv2DBackpropCommonOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format_str;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(context, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    int stride_n = GetTensorDim(strides_, data_format_, 'N');
    int stride_c = GetTensorDim(strides_, data_format_, 'C');
    OP_REQUIRES(
        context, (stride_n == 1 && stride_c == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));
    OP_REQUIRES(context, dilations_.size() == 4,
                errors::InvalidArgument("Sliding window dilations field must "
                                        "specify 4 dimensions"));
    int dilation_n = GetTensorDim(dilations_, data_format_, 'N');
    int dilation_c = GetTensorDim(dilations_, data_format_, 'C');
    int dilation_h = GetTensorDim(dilations_, data_format_, 'H');
    int dilation_w = GetTensorDim(dilations_, data_format_, 'W');
    OP_REQUIRES(context, (dilation_n == 1 && dilation_c == 1),
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilations in the batch and depth dimensions."));
    OP_REQUIRES(
        context, dilation_h > 0 && dilation_w > 0,
        errors::InvalidArgument("Dilated rates should be larger than 0."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

 protected:
  // data members accessible to derived classes.
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;  // NCHW or NHWC
};

#endif  // INTEL_MKL_ML


/////////////////////////////////////////////////////////////////////
///  Dummy Mkl op that is just used for operators that are intermediate
///  output of node fusion in the graph
/////////////////////////////////////////////////////////////////////

template <typename Device, typename T>
class MklDummyOp : public OpKernel {
 public:
  ~MklDummyOp() {}

  explicit MklDummyOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TF_CHECK_OK(
        errors::Unimplemented("This is a dummy op."
                              "It should not have been invoked."));
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MKL_CONV_OPS_H_
