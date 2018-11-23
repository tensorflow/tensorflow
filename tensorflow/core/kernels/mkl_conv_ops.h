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
#include <memory>
#include <vector>

#include "mkldnn.hpp"
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
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

using mkldnn::convolution_direct;
using mkldnn::convolution_forward;
using mkldnn::prop_kind;
using mkldnn::stream;

namespace tensorflow {

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
    if (strides_.size() == 4) {
      int stride_rows = GetTensorDim(strides_, data_format_, 'H');
      int stride_cols = GetTensorDim(strides_, data_format_, 'W');
      *strides = {stride_rows, stride_cols};
    } else if (strides_.size() == 5) {
      int stride_planes = GetTensorDim(strides_, data_format_, '0');
      int stride_rows = GetTensorDim(strides_, data_format_, '1');
      int stride_cols = GetTensorDim(strides_, data_format_, '2');
      *strides = {stride_planes, stride_rows, stride_cols};
    }
  }

  // Calculate Convolution dilations
  virtual inline void GetDilationsInMklOrder(memory::dims* dilations) {
    // For now we take the dilation from the second and third dimensions only
    // (we do not support dilation on the batch or depth dimension).
    CHECK_NOTNULL(dilations);
    if (dilations_.size() == 4) {
      int dilations_rows = GetTensorDim(dilations_, data_format_, 'H');
      int dilations_cols = GetTensorDim(dilations_, data_format_, 'W');
      *dilations = {dilations_rows, dilations_cols};
    } else if (dilations_.size() == 5) {
      int dilations_planes = GetTensorDim(dilations_, data_format_, '0');
      int dilations_rows = GetTensorDim(dilations_, data_format_, '1');
      int dilations_cols = GetTensorDim(dilations_, data_format_, '2');
      *dilations = {dilations_planes, dilations_rows, dilations_cols};
    }
  }

  // Calculate Convolution input size in MKL-DNN order. MKL-DNN
  // requires input in NCHW/NCDHW format. Function does not return anything.
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

    // Input batch
    int64 input_batch_raw = GetTensorDim(input_shape, data_format_, 'N');
    CHECK_BOUNDS(input_batch_raw, "Input batch too large");
    int input_batch = static_cast<int>(input_batch_raw);

    if (strides_.size() == 4) {  // NCHW format for Conv2D
      // Input rows/height
      int64 input_rows_raw = GetTensorDim(input_shape, data_format_, 'H');
      CHECK_BOUNDS(input_rows_raw, "Input rows too large");
      int input_rows = static_cast<int>(input_rows_raw);

      // Input columns/width
      int64 input_cols_raw = GetTensorDim(input_shape, data_format_, 'W');
      CHECK_BOUNDS(input_cols_raw, "Input cols too large");
      int input_cols = static_cast<int>(input_cols_raw);

      // MKL-DNN always requires input in NCHW format Conv2D.
      std::vector<int> mkldnn_sizes(4, -1);
      mkldnn_sizes[MklDnnDims::Dim_N] = input_batch;
      mkldnn_sizes[MklDnnDims::Dim_C] = input_depth;
      mkldnn_sizes[MklDnnDims::Dim_H] = input_rows;
      mkldnn_sizes[MklDnnDims::Dim_W] = input_cols;

      *input_dims = mkldnn_sizes;
    } else if (strides_.size() == 5) {  // NCDHW format for Conv3D
      // Input planes/third-dimension
      int64 input_planes_raw = GetTensorDim(input_shape, data_format_, '0');
      CHECK_BOUNDS(input_planes_raw, "Input depth too large");
      int input_planes = static_cast<int>(input_planes_raw);

      // Input rows/height
      int64 input_rows_raw = GetTensorDim(input_shape, data_format_, '1');
      CHECK_BOUNDS(input_rows_raw, "Input rows too large");
      int input_rows = static_cast<int>(input_rows_raw);

      // Input columns/width
      int64 input_cols_raw = GetTensorDim(input_shape, data_format_, '2');
      CHECK_BOUNDS(input_cols_raw, "Input cols too large");
      int input_cols = static_cast<int>(input_cols_raw);

      // MKL-DNN always requires input in NCDHW format for Conv3D.
      std::vector<int> mkldnn_sizes(5, -1);
      mkldnn_sizes[MklDnnDims3D::Dim3d_N] = input_batch;
      mkldnn_sizes[MklDnnDims3D::Dim3d_C] = input_depth;
      mkldnn_sizes[MklDnnDims3D::Dim3d_D] = input_planes;
      mkldnn_sizes[MklDnnDims3D::Dim3d_H] = input_rows;
      mkldnn_sizes[MklDnnDims3D::Dim3d_W] = input_cols;

      *input_dims = mkldnn_sizes;
    }
#undef CHECK_BOUNDS
  }

  // Calculate Convolution filter size in MKL-DNN order.
  // MKL-DNN requires filter in OIHW (Conv2D) or OIDHW (Conv3D) format.
  // Function does not return anything.
  // But errors arising from sanity checks are returned in context's
  // status. This function differs from GetConvFilterSizeInMklOrder in
  // parameter for input - it accepts src_shape since Convolution Backward
  // Input gets shape of input tensor rather than actual tensor (Convolution
  // forward gets actual tensor as input).
  //
  // TODO(nhasabni): Add similar function for input and filter in MklShape.
  virtual inline void GetFilterSizeInMklOrder(const TensorShape& input_shape,
                                              const TensorShape& filter_shape,
                                              memory::dims* filter_dims,
                                              bool is_Depthwise) {
    CHECK_NOTNULL(filter_dims);

    OP_REQUIRES(context_, filter_shape.dims() == strides_.size(),
                errors::InvalidArgument((strides_.size() == 4)
                                            ? "filter must be 4-dimensional: "
                                            : "filter must be 5-dimensional: ",
                                        filter_shape.DebugString()));

    for (int i = 0; i < ((strides_.size() == 4) ? 3 : 5); i++) {
      OP_REQUIRES(context_, FastBoundsCheck(filter_shape.dim_size(i),
                                            std::numeric_limits<int>::max()),
                  errors::InvalidArgument("filter too large"));
    }

    int input_depth = GetTensorDim(input_shape, data_format_, 'C');

    if (strides_.size() == 4) {  // Conv2D
      OP_REQUIRES(context_, input_depth == filter_shape.dim_size(2),
                  errors::InvalidArgument(
                      "input and filter must have the same depth: ",
                      input_depth, " vs ", filter_shape.dim_size(2)));

      // TF filter is always in (rows, cols, in_depth, out_depth) order.
      int filter_rows =
          static_cast<int>(filter_shape.dim_size(TF_2DFILTER_DIM_H));
      int filter_cols =
          static_cast<int>(filter_shape.dim_size(TF_2DFILTER_DIM_W));
      int filter_in_depth =
          static_cast<int>(filter_shape.dim_size(TF_2DFILTER_DIM_I));
      int filter_out_depth =
          static_cast<int>(filter_shape.dim_size(TF_2DFILTER_DIM_O));
      // MKL-DNN always needs filter in OIHW format for regular convolutions
      // and GOIHW for grouped/depthwise convolutions,
      // OIHW = (out_depth, in_depth, rows, cols)
      // GOIHW = (group, out_depth, in_depth, rows, cols)
      // Specifically for depthwise G=filer_indepth, O=filter_outdepth, I=1
      if (is_Depthwise) {
        std::vector<int> mkldnn_sizes(5, -1);
        mkldnn_sizes[MKL_GROUP_FILTER_DIM_G] = filter_in_depth;
        mkldnn_sizes[MKL_GROUP_FILTER_DIM_O] = filter_out_depth;
        mkldnn_sizes[MKL_GROUP_FILTER_DIM_I] = 1;
        mkldnn_sizes[MKL_GROUP_FILTER_DIM_H] = filter_rows;
        mkldnn_sizes[MKL_GROUP_FILTER_DIM_W] = filter_cols;

        *filter_dims = mkldnn_sizes;
      } else {
        std::vector<int> mkldnn_sizes(4, -1);
        mkldnn_sizes[MklDnnDims::Dim_O] = filter_out_depth;
        mkldnn_sizes[MklDnnDims::Dim_I] = filter_in_depth;
        mkldnn_sizes[MklDnnDims::Dim_H] = filter_rows;
        mkldnn_sizes[MklDnnDims::Dim_W] = filter_cols;

        *filter_dims = mkldnn_sizes;
      }
    } else {  // Conv3D
      OP_REQUIRES(context_, input_depth == filter_shape.dim_size(3),
                  errors::InvalidArgument(
                      "input and filter must have the same depth: ",
                      input_depth, " vs ", filter_shape.dim_size(3)));

      // TF filter is always in (planes, rows, cols, in_depth, out_depth) order.
      int filter_planes =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_P));
      int filter_rows =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_H));
      int filter_cols =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_W));
      int filter_in_depth =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_I));
      int filter_out_depth =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_O));

      // MKL-DNN always needs filter in OIDHW format.
      // OIDHW = (out_depth, in_depth, planes, rows, cols)
      std::vector<int> mkldnn_sizes(5, -1);
      mkldnn_sizes[MklDnnDims3D::Dim3d_O] = filter_out_depth;
      mkldnn_sizes[MklDnnDims3D::Dim3d_I] = filter_in_depth;
      mkldnn_sizes[MklDnnDims3D::Dim3d_D] = filter_planes;
      mkldnn_sizes[MklDnnDims3D::Dim3d_H] = filter_rows;
      mkldnn_sizes[MklDnnDims3D::Dim3d_W] = filter_cols;

      *filter_dims = mkldnn_sizes;
    }
  }

  // Calculate Convolution filter size in MKL-DNN order.
  // MKL-DNN requires filter in OIHW (Conv2D) or OIDHW(Conv3D format.
  // Function does not return anything. But errors arising from sanity
  // checks are returned in context's status.
  virtual inline void GetFilterSizeInMklOrder(size_t src_index,
                                              size_t filter_index,
                                              memory::dims* filter_dims,
                                              bool is_Depthwise) {
    CHECK_NOTNULL(filter_dims);
    GetFilterSizeInMklOrder(GetTfShape(context_, src_index),
                            GetTfShape(context_, filter_index), filter_dims,
                            is_Depthwise);
  }

  // Calculate Bias size for 2D or 3D Convolution. Function does not
  // return anything, but may set an error in context status.
  virtual inline void GetBiasSizeInMklOrder(size_t bias_index,
                                            memory::dims* bias_dims) {
    const Tensor& bias = MklGetInput(context_, bias_index);
    OP_REQUIRES(context_, bias.dims() == 1,
                errors::InvalidArgument("bias must be 1-dimensional: ",
                                        bias.shape().DebugString()));

    *bias_dims = {static_cast<int>(bias.dim_size(0))};
  }

  // Function to calculate output and padding size for 2D/3D convolution.
  //
  // Calculate output shape of Convolution in MKL-DNN and TensorFlow order.
  // MKL-DNN uses NCHW(Conv2D) or NCDHW(Conv3D) for output order.
  // But TensorFlow output will be in NHWC||NCHW(Conv2D) or
  // NDHWC||NCDHW(Conv3D) format depending on data format.
  // Function also calculates left, right, top and bottom pads.
  // Function does not return any status which is set with context status.
  //
  // TODO(nhasabni): Add similar function for input and filter in MklShape.
  virtual inline void GetOutputAndPadSizeInMklOrder(
      const TensorShape& input_shape, const TensorShape& filter_shape,
      const memory::dims& strides, const memory::dims& dilations,
      memory::dims* output_dims_tf_order, memory::dims* output_dims_mkl_order,
      memory::dims* pad_l, memory::dims* pad_r, bool is_Depthwise) {
    CHECK_NOTNULL(output_dims_tf_order);
    CHECK_NOTNULL(output_dims_mkl_order);
    CHECK_NOTNULL(pad_l);
    CHECK_NOTNULL(pad_r);

    bool is_Conv2D = (strides_.size() == 4);
    int input_planes, input_rows, input_cols;
    if (is_Conv2D) {
      input_rows = GetTensorDim(input_shape, data_format_, 'H');
      input_cols = GetTensorDim(input_shape, data_format_, 'W');
    } else {
      input_planes = GetTensorDim(input_shape, data_format_, '0');
      input_rows = GetTensorDim(input_shape, data_format_, '1');
      input_cols = GetTensorDim(input_shape, data_format_, '2');
    }

    // Filter dimension
    // Conv2D:
    //    First dimension: rows/height.
    //    Second dimension: cols/width.
    // Conv3D:
    //    First dimension: planes/depth.
    //    Second dimension: rows/height.
    //    Third dimension: cols/width.

    int filter_planes, filter_rows, filter_cols;
    if (is_Conv2D) {
      filter_rows = filter_shape.dim_size(TF_2DFILTER_DIM_H);
      filter_cols = filter_shape.dim_size(TF_2DFILTER_DIM_W);
    } else {
      filter_planes = filter_shape.dim_size(TF_3DFILTER_DIM_P);
      filter_rows = filter_shape.dim_size(TF_3DFILTER_DIM_H);
      filter_cols = filter_shape.dim_size(TF_3DFILTER_DIM_W);
    }

    int stride_planes, stride_rows, stride_cols;
    int dilation_planes, dilation_rows, dilation_cols;
    if (is_Conv2D) {
      // Conv2D stride is a vector of 2 elements: {s_r, s_c}
      stride_rows = strides[0];
      stride_cols = strides[1];
      dilation_rows = dilations[0];
      dilation_cols = dilations[1];
    } else {
      // Conv3D stride is a vector of 3 elements: {s_d, s_r, s_c}
      stride_planes = strides[0];
      stride_rows = strides[1];
      stride_cols = strides[2];
      dilation_planes = dilations[0];
      dilation_rows = dilations[1];
      dilation_cols = dilations[2];
    }

    // Output batch is same as input batch.
    int out_batch = GetTensorDim(input_shape, data_format_, 'N');
    int out_depth;

    // TODO add support for 3-D Depthwise

    // Output depth is same as last dimension for filters for regular
    // convolutions.
    // For depthwise it is in_depth * channel_multiplier. The channel_miltipler
    // is the last dimension of TF filter for depthwise convolutions.
    if (is_Depthwise) {
      out_depth = (filter_shape.dim_size(TF_2DFILTER_DIM_I) *
                   filter_shape.dim_size(TF_2DFILTER_DIM_O));
    } else {
      out_depth = filter_shape.dim_size(
          is_Conv2D ? static_cast<int>(TF_2DFILTER_DIM_O)
                    : static_cast<int>(TF_3DFILTER_DIM_O));
    }

    int64 out_rows = 0, out_cols = 0, out_planes = 0;
    int64 pad_top = 0, pad_bottom = 0, pad_left, pad_right;
    int64 pad_D1, pad_D2;

    if (is_Conv2D) {
      OP_REQUIRES_OK(context_,
                     GetWindowedOutputSizeVerboseV2(
                         input_rows, filter_rows, dilation_rows, stride_rows,
                         padding_, &out_rows, &pad_top, &pad_bottom));
      OP_REQUIRES_OK(context_,
                     GetWindowedOutputSizeVerboseV2(
                         input_cols, filter_cols, dilation_cols, stride_cols,
                         padding_, &out_cols, &pad_left, &pad_right));
    } else {
      OP_REQUIRES_OK(context_, GetWindowedOutputSizeVerbose(
                                   input_planes, filter_planes, stride_planes,
                                   padding_, &out_planes, &pad_D1, &pad_D2));
      OP_REQUIRES_OK(context_, GetWindowedOutputSizeVerbose(
                                   input_rows, filter_rows, stride_rows,
                                   padding_, &out_rows, &pad_top, &pad_bottom));
      OP_REQUIRES_OK(context_, GetWindowedOutputSizeVerbose(
                                   input_cols, filter_cols, stride_cols,
                                   padding_, &out_cols, &pad_left, &pad_right));
    }

    // Tensorflow output is in data_format order.
    //     Conv2D: NHWC or NCHW
    //     Conv3D: NDHWC or NCDHW
    // MKL-DNN uses asymetric padding.
    TensorShape out_shape =
        is_Conv2D
            ? ShapeFromFormat(data_format_, out_batch, out_rows, out_cols,
                              out_depth)
            : ShapeFromFormat(data_format_, out_batch,
                              {{out_planes, out_rows, out_cols}}, out_depth);
    *output_dims_tf_order = TFShapeToMklDnnDims(out_shape);

    if (is_Conv2D) {
      // For Conv2D, MKL-DNN always needs output in NCHW format.
      std::vector<int> mkldnn_sizes(4, -1);
      mkldnn_sizes[MklDnnDims::Dim_N] = out_batch;
      mkldnn_sizes[MklDnnDims::Dim_C] = out_depth;
      mkldnn_sizes[MklDnnDims::Dim_H] = static_cast<int>(out_rows);
      mkldnn_sizes[MklDnnDims::Dim_W] = static_cast<int>(out_cols);
      *output_dims_mkl_order = mkldnn_sizes;

      *pad_l = {static_cast<int>(pad_top), static_cast<int>(pad_left)};
      *pad_r = {static_cast<int>(pad_bottom), static_cast<int>(pad_right)};
    } else {
      std::vector<int> mkldnn_sizes(5, -1);
      mkldnn_sizes[MklDnnDims3D::Dim3d_N] = out_batch;
      mkldnn_sizes[MklDnnDims3D::Dim3d_C] = out_depth;
      mkldnn_sizes[MklDnnDims3D::Dim3d_D] = static_cast<int>(out_planes);
      mkldnn_sizes[MklDnnDims3D::Dim3d_H] = static_cast<int>(out_rows);
      mkldnn_sizes[MklDnnDims3D::Dim3d_W] = static_cast<int>(out_cols);
      *output_dims_mkl_order = mkldnn_sizes;

      *pad_l = {static_cast<int>(pad_D1), static_cast<int>(pad_top),
                static_cast<int>(pad_left)};
      *pad_r = {static_cast<int>(pad_D2), static_cast<int>(pad_bottom),
                static_cast<int>(pad_right)};
    }
  }

  // Calculate output and pad size of forward Convolution operator.
  // See comment on GetConvOutputAndPadSizeInMklOrder for parameters.
  //
  // Function does not return anything, but sets error in context status.
  inline void GetOutputAndPadSizeInMklOrder(
      size_t src_index, size_t filter_index, const memory::dims& strides,
      const memory::dims& dilations, memory::dims* output_dims_tf_order,
      memory::dims* output_dims_mkl_order, memory::dims* pad_l,
      memory::dims* pad_r, bool is_Depthwise) {
    CHECK_NOTNULL(output_dims_tf_order);
    CHECK_NOTNULL(output_dims_mkl_order);
    CHECK_NOTNULL(pad_l);
    CHECK_NOTNULL(pad_r);

    auto input_tf_shape = GetTfShape(context_, src_index);
    auto filter_tf_shape = GetTfShape(context_, filter_index);

    if (strides_.size() == 4) {
      // Conv2D
      OP_REQUIRES(context_, input_tf_shape.dims() == 4,
                  errors::InvalidArgument("input must be 4-dimensional",
                                          input_tf_shape.DebugString()));
    } else {
      // Conv3D
      OP_REQUIRES(context_, input_tf_shape.dims() == 5,
                  errors::InvalidArgument("input must be 5-dimensional",
                                          input_tf_shape.DebugString()));
    }

    GetOutputAndPadSizeInMklOrder(input_tf_shape, filter_tf_shape, strides,
                                  dilations, output_dims_tf_order,
                                  output_dims_mkl_order, pad_l, pad_r,
                                  is_Depthwise);
  }

  // Wrapper function to calculate input, filter, and output sizes of
  // Conv2D/Conv3D in MKL order:
  //     Conv2D: NCHW for input and output; OIHW for filter.
  //     Conv3D: NCDHW for input and output; OIDHW for filter.
  // Function also calculates output shape in Tensorflow order.
  // Additionally, it also calculates strides and paddings.
  //
  // Function does not return anything, but sets error in context status.
  inline void GetConvFwdSizesInMklOrder(
      const TensorShape& input_shape, const TensorShape& filter_shape,
      memory::dims* input_dims, memory::dims* filter_dims,
      memory::dims* strides, memory::dims* dilations,
      memory::dims* output_dims_tf_order, memory::dims* output_dims_mkl_order,
      memory::dims* pad_l, memory::dims* pad_r, bool is_Depthwise) {
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
    GetFilterSizeInMklOrder(input_shape, filter_shape, filter_dims,
                            is_Depthwise);
    if (!context_->status().ok()) return;
    GetStridesInMklOrder(strides);
    GetDilationsInMklOrder(dilations);
    GetOutputAndPadSizeInMklOrder(
        input_shape, filter_shape, *strides, *dilations, output_dims_tf_order,
        output_dims_mkl_order, pad_l, pad_r, is_Depthwise);
    if (!context_->status().ok()) return;
  }
};

/////////////////////////////////////////////////////////////////////
///  Common class that implements ConvBackpropFilter and Input
/////////////////////////////////////////////////////////////////////

template <typename Device, class T>
class MklConvBackpropCommonOp : public OpKernel {
 public:
  ~MklConvBackpropCommonOp() {}
  explicit MklConvBackpropCommonOp(OpKernelConstruction* context)
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

    if (strides_.size() == 4) {
      // Check Conv2D dilations
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
    }

    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

 protected:
  // data members accessible to derived classes.
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;  // NCHW or NHWC
};

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
