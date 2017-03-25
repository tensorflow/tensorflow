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

// See docs in ../ops/nn_ops.cc.

#ifdef INTEL_MKL

#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/conv_grad_ops.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "tensorflow/core/util/work_sharder.h"

#include "tensorflow/core/util/mkl_util.h"
#include "third_party/mkl/include/mkl_dnn.h"
#include "third_party/mkl/include/mkl_dnn_types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, class T>
class MklConv2DCustomBackpropFilterOp : public OpKernel {
 public:
  explicit MklConv2DCustomBackpropFilterOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));

    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    int stride_n = GetTensorDim(strides_, data_format_, 'N');
    int stride_c = GetTensorDim(strides_, data_format_, 'C');
    OP_REQUIRES(
        context, (stride_n == 1 && stride_c == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = MklGetInput(context, 0);
    GetMklShape(context, 0, &(mkl_shapes_.input_shape));
    bool input_in_mkl_format = mkl_shapes_.input_shape.IsMklTensor();

    const Tensor& filter_sizes = MklGetInput(context, 1);

    const Tensor& out_backprop = MklGetInput(context, 2);
    GetMklShape(context, 2, &(mkl_shapes_.out_backprop_shape));
    bool out_backprop_in_mkl_format =
        mkl_shapes_.out_backprop_shape.IsMklTensor();

    TensorShape input_shape, filter_shape, out_backprop_shape;

    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(filter_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DCustomBackpropFilter: filter_sizes input must be 1-dim, "
            "not ",
            filter_sizes.dims()));
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                filter_sizes.vec<int32>(), &filter_shape));

    Conv2DBackpropDimensions backprop_dims;

    // Generate shape for input if input is in MKL format.
    if (input_in_mkl_format) {
      OP_REQUIRES(context, mkl_shapes_.input_shape.GetDimension() == 4,
                  errors::InvalidArgument(
                      "Conv2DCustomBackpropFilter: input size must be 4-dim"));

      MklSizesToTFSizes(context, data_format_, mkl_shapes_.input_shape,
                        &input_shape);
    } else {
      input_shape = input.shape();
    }

    // Generate shape for outback prop if input is in MKL format.
    if (out_backprop_in_mkl_format) {
      OP_REQUIRES(
          context, mkl_shapes_.out_backprop_shape.GetDimension() == 4,
          errors::InvalidArgument(
              "Conv2DCustomBackpropFilter: outbackprop size must be 4-dim"));

      MklSizesToTFSizes(context, data_format_, mkl_shapes_.out_backprop_shape,
                        &out_backprop_shape);
    } else {
      out_backprop_shape = out_backprop.shape();
    }

    OP_REQUIRES_OK(context, Conv2DBackpropComputeDimensions(
                                "Conv2DCustomBackpropFilter", input_shape,
                                filter_shape, out_backprop_shape, strides_,
                                padding_, data_format_, &backprop_dims));

    int64 pad_top, pad_bottom;
    int64 pad_left, pad_right;
    OP_REQUIRES_OK(
        context,
        GetWindowedOutputSizeVerbose(
            backprop_dims.rows.input_size, backprop_dims.rows.filter_size,
            backprop_dims.rows.stride, padding_,
            &backprop_dims.rows.output_size, &pad_top, &pad_bottom));
    OP_REQUIRES_OK(
        context,
        GetWindowedOutputSizeVerbose(
            backprop_dims.cols.input_size, backprop_dims.cols.filter_size,
            backprop_dims.cols.stride, padding_,
            &backprop_dims.cols.output_size, &pad_left, &pad_right));

    // Create MKL primitives for convolution filter grad
    mkl_params_.in_dims = input_in_mkl_format
                              ? mkl_shapes_.input_shape.GetDimension()
                              : input.dims();
    mkl_params_.out_dims = out_backprop_in_mkl_format
                               ? mkl_shapes_.out_backprop_shape.GetDimension()
                               : out_backprop.dims();
    mkl_params_.in_sizes[0] =
        static_cast<size_t>(backprop_dims.cols.input_size);
    mkl_params_.in_sizes[1] =
        static_cast<size_t>(backprop_dims.rows.input_size);
    mkl_params_.in_sizes[2] = static_cast<size_t>(backprop_dims.in_depth);
    mkl_params_.in_sizes[3] = static_cast<size_t>(backprop_dims.batch_size);
    mkl_params_.out_sizes[0] =
        static_cast<size_t>(backprop_dims.cols.output_size);
    mkl_params_.out_sizes[1] =
        static_cast<size_t>(backprop_dims.rows.output_size);
    mkl_params_.out_sizes[2] = static_cast<size_t>(backprop_dims.out_depth);
    mkl_params_.out_sizes[3] = static_cast<size_t>(backprop_dims.batch_size);
    mkl_params_.input_offsets[0] = static_cast<int>(-pad_left);
    mkl_params_.input_offsets[1] = static_cast<int>(-pad_top);
    mkl_params_.conv_strides[0] =
        static_cast<size_t>(backprop_dims.cols.stride);
    mkl_params_.conv_strides[1] =
        static_cast<size_t>(backprop_dims.rows.stride);

    GetStridesFromSizes(data_format_, mkl_params_.in_strides,
                        mkl_params_.in_sizes);
    GetStridesFromSizes(data_format_, mkl_params_.out_strides,
                        mkl_params_.out_sizes);

    // MKL understands dimensions in 0, 1, 2, and 3 indices denotes
    // filter cols, rows, input channels, and output depth/channels.
    mkl_params_.filter_dims = 4;
    mkl_params_.filter_sizes[0] = backprop_dims.cols.filter_size;
    mkl_params_.filter_sizes[1] = backprop_dims.rows.filter_size;
    mkl_params_.filter_sizes[2] = backprop_dims.in_depth;
    mkl_params_.filter_sizes[3] = backprop_dims.out_depth;

    // We want filter grad to be in TF format, so
    // make the strides accordingly to reflect this fact.
    // Note TF filter layout : (rows, cols, in_depth, out_depth),
    // while row is the innermost dimension.
    mkl_params_.filter_strides[0] =
        backprop_dims.out_depth * backprop_dims.in_depth;
    mkl_params_.filter_strides[1] = backprop_dims.out_depth *
                                    backprop_dims.in_depth *
                                    backprop_dims.cols.filter_size;
    mkl_params_.filter_strides[2] = backprop_dims.out_depth;
    mkl_params_.filter_strides[3] = 1;

    mkl_params_.conv_strides[0] = backprop_dims.cols.stride;
    mkl_params_.conv_strides[1] = backprop_dims.rows.stride;

    // Create convolution-grad-filter primitive
    CHECK_EQ(
        dnnConvolutionCreateBackwardFilter_F32(
            &mkl_prim_conv_grad_filter_, nullptr, dnnAlgorithmConvolutionDirect,
            mkl_params_.in_dims, mkl_params_.in_sizes, mkl_params_.out_sizes,
            mkl_params_.filter_sizes, mkl_params_.conv_strides,
            mkl_params_.input_offsets, dnnBorderZeros),
        E_SUCCESS);

    // Create the layouts for entities in received context.
    MklCreateInputLayouts(context);

    // Mkl needs the entities in its native format.
    // So create temporary tensors along with buffers to
    // convert the received entities.
    Tensor mkl_tmp_input_buf_tensor, mkl_tmp_out_backprop_buf_tensor;
    // This preparation sets (1) dnnResourceSrc (2) dnnResourceDiffDst
    MklPrepareInputs(context, &mkl_tmp_input_buf_tensor,
                     &mkl_tmp_out_backprop_buf_tensor);

    // Final conv-grad-filter should be in TF layout.
    Tensor* grad_filter;
    mkl_shapes_.grad_filter_shape.SetMklTensor(false);
    mkl_shapes_.grad_filter_shape.SetTfLayout(mkl_params_.filter_dims,
                                              mkl_params_.filter_sizes,
                                              mkl_params_.filter_strides);
    AllocateOutputSetMklshape(context, 0, &grad_filter, filter_shape,
                              mkl_shapes_.grad_filter_shape);

    // Need to set member variable for TF layout
    mkl_lt_grad_filter_ = mkl_shapes_.grad_filter_shape.GetTfLayout();

    // MKL conv-grad-filter might produce grad in its internal layout
    Tensor mkl_tmp_grad_filter_buf_tensor;
    // This preparation sets conversion primitive if required
    // and allocates temporary tensor and its buffer without doing conversions.
    // Also sets (3) dnnResourceDiffFilter accordingly
    MklPrepareGradFilter(context, grad_filter, &mkl_tmp_grad_filter_buf_tensor);

    // After setting all the required dnnResources, ready for execution!
    CHECK_EQ(
        dnnExecute_F32(mkl_prim_conv_grad_filter_, mkl_conv_grad_filter_res_),
        E_SUCCESS);

    // Convert grad-filter to TF layout
    if (mkl_prim_convert_grad_filter_ != nullptr) {
      void* mkl_buf_convert_grad_filter =
          const_cast<void*>(static_cast<const void*>(
              mkl_tmp_grad_filter_buf_tensor.flat<T>().data()));
      void* mkl_buf_grad_filter = const_cast<void*>(
          static_cast<const void*>(grad_filter->flat<T>().data()));
      CHECK_EQ(dnnConversionExecute_F32(mkl_prim_convert_grad_filter_,
                                        mkl_buf_convert_grad_filter,
                                        mkl_buf_grad_filter),
               E_SUCCESS);
    }

    MklCleanup();
  }

 private:
  typedef struct {
    int in_dims;
    size_t in_sizes[4];
    size_t in_strides[4];
    int out_dims;
    size_t out_sizes[4];
    size_t out_strides[4];
    int filter_dims;
    size_t filter_sizes[4];
    size_t filter_strides[4];
    int input_offsets[2];
    size_t conv_strides[2];
  } MklConv2DGradParams;

  typedef struct {
    MklShape input_shape;
    MklShape grad_filter_shape;
    MklShape out_backprop_shape;
  } MklConv2DGradShapes;

  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;

  MklConv2DGradParams mkl_params_;
  MklConv2DGradShapes mkl_shapes_;
  dnnPrimitive_t mkl_prim_conv_grad_filter_ = nullptr;
  dnnPrimitive_t mkl_prim_convert_grad_filter_ = nullptr;
  dnnLayout_t mkl_lt_input_ = nullptr, mkl_lt_grad_filter_ = nullptr,
              mkl_lt_out_backprop_ = nullptr;
  void* mkl_conv_grad_filter_res_[dnnResourceNumber];

  void MklCleanup() {
    // Cleanup member layouts and primitives except "mkl_lt_grad_filter_"
    // which points to MklShape's TFLayout
    bool input_in_mkl_format = mkl_shapes_.input_shape.IsMklTensor();
    bool out_backprop_in_mkl_format =
        mkl_shapes_.out_backprop_shape.IsMklTensor();
    if (!input_in_mkl_format) dnnLayoutDelete_F32(mkl_lt_input_);
    if (!out_backprop_in_mkl_format) dnnLayoutDelete_F32(mkl_lt_out_backprop_);
    if (mkl_prim_convert_grad_filter_ != nullptr)
      dnnDelete_F32(mkl_prim_convert_grad_filter_);
    dnnDelete_F32(mkl_prim_conv_grad_filter_);
  }

  // Create MKL dnnLayout_t objects for tensors coming into the layer
  void MklCreateInputLayouts(OpKernelContext* context) {
    bool input_in_mkl_format = mkl_shapes_.input_shape.IsMklTensor();
    if (input_in_mkl_format) {
      mkl_lt_input_ =
          static_cast<dnnLayout_t>(mkl_shapes_.input_shape.GetCurLayout());
    } else {
      CHECK_EQ(
          dnnLayoutCreate_F32(&mkl_lt_input_, mkl_params_.in_dims,
                              mkl_params_.in_sizes, mkl_params_.in_strides),
          E_SUCCESS);
    }

    bool out_backprop_in_mkl_format =
        mkl_shapes_.out_backprop_shape.IsMklTensor();
    if (out_backprop_in_mkl_format) {
      mkl_lt_out_backprop_ = static_cast<dnnLayout_t>(
          mkl_shapes_.out_backprop_shape.GetCurLayout());
    } else {
      CHECK_EQ(
          dnnLayoutCreate_F32(&mkl_lt_out_backprop_, mkl_params_.out_dims,
                              mkl_params_.out_sizes, mkl_params_.out_strides),
          E_SUCCESS);
    }
  }

  // Compare incoming tensor layouts with MKL preferred layouts and convert
  // data to the preferred layout if necessary
  void MklPrepareInputs(OpKernelContext* context,
                        Tensor* mkl_tmp_input_buf_tensor,
                        Tensor* mkl_tmp_out_backprop_buf_tensor) {
    bool mkl_convert_input, mkl_convert_out_backprop;
    dnnPrimitive_t mkl_prim_convert_input, mkl_prim_convert_out_backprop;
    dnnLayout_t mkl_lt_internal_input, mkl_lt_internal_out_backprop;
    void *mkl_buf_convert_input, *mkl_buf_convert_out_backprop;

    mkl_prim_convert_input = nullptr;
    mkl_prim_convert_out_backprop = nullptr;
    mkl_lt_internal_input = nullptr;
    mkl_lt_internal_out_backprop = nullptr;
    mkl_buf_convert_input = nullptr;
    mkl_buf_convert_out_backprop = nullptr;

    // Compare with internal layouts and convert if needed
    const Tensor& input = MklGetInput(context, 0);
    void* mkl_buf_input =
        const_cast<void*>(static_cast<const void*>(input.flat<T>().data()));
    CHECK_EQ(
        dnnLayoutCreateFromPrimitive_F32(
            &mkl_lt_internal_input, mkl_prim_conv_grad_filter_, dnnResourceSrc),
        E_SUCCESS);
    mkl_convert_input =
        !dnnLayoutCompare_F32(mkl_lt_internal_input, mkl_lt_input_);
    if (mkl_convert_input) {
      CHECK_EQ(dnnConversionCreate_F32(&mkl_prim_convert_input, mkl_lt_input_,
                                       mkl_lt_internal_input),
               E_SUCCESS);
      AllocTmpBuffer(context, mkl_tmp_input_buf_tensor, mkl_lt_internal_input,
                     &mkl_buf_convert_input);
      CHECK_EQ(dnnConversionExecute_F32(mkl_prim_convert_input, mkl_buf_input,
                                        mkl_buf_convert_input),
               E_SUCCESS);
      dnnDelete_F32(mkl_prim_convert_input);
    }
    dnnLayoutDelete_F32(mkl_lt_internal_input);

    mkl_conv_grad_filter_res_[dnnResourceSrc] =
        (mkl_convert_input) ? mkl_buf_convert_input : mkl_buf_input;

    const Tensor& out_backprop = MklGetInput(context, 2);
    void* mkl_buf_out_backprop = const_cast<void*>(
        static_cast<const void*>(out_backprop.flat<T>().data()));
    CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&mkl_lt_internal_out_backprop,
                                              mkl_prim_conv_grad_filter_,
                                              dnnResourceDiffDst),
             E_SUCCESS);
    mkl_convert_out_backprop = !dnnLayoutCompare_F32(
        mkl_lt_internal_out_backprop, mkl_lt_out_backprop_);
    if (mkl_convert_out_backprop) {
      CHECK_EQ(dnnConversionCreate_F32(&mkl_prim_convert_out_backprop,
                                       mkl_lt_out_backprop_,
                                       mkl_lt_internal_out_backprop),
               E_SUCCESS);
      AllocTmpBuffer(context, mkl_tmp_out_backprop_buf_tensor,
                     mkl_lt_out_backprop_, &mkl_buf_convert_out_backprop);
      CHECK_EQ(dnnConversionExecute_F32(mkl_prim_convert_out_backprop,
                                        mkl_buf_out_backprop,
                                        mkl_buf_convert_out_backprop),
               E_SUCCESS);
      dnnDelete_F32(mkl_prim_convert_out_backprop);
    }
    dnnLayoutDelete_F32(mkl_lt_internal_out_backprop);

    mkl_conv_grad_filter_res_[dnnResourceDiffDst] =
        (mkl_convert_out_backprop) ? mkl_buf_convert_out_backprop
                                   : mkl_buf_out_backprop;
  }

  void MklPrepareGradFilter(OpKernelContext* context, Tensor* grad_filter,
                            Tensor* mkl_tmp_grad_filter_buf_tensor) {
    bool mkl_convert_grad_filter;
    dnnLayout_t mkl_lt_internal_grad_filter = nullptr;
    void* mkl_buf_convert_grad_filter = nullptr;
    void* mkl_buf_grad_filter = const_cast<void*>(
        static_cast<const void*>(grad_filter->flat<T>().data()));
    CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&mkl_lt_internal_grad_filter,
                                              mkl_prim_conv_grad_filter_,
                                              dnnResourceDiffFilter),
             E_SUCCESS);
    mkl_convert_grad_filter =
        !dnnLayoutCompare_F32(mkl_lt_internal_grad_filter, mkl_lt_grad_filter_);
    if (mkl_convert_grad_filter) {
      CHECK_EQ(dnnConversionCreate_F32(&mkl_prim_convert_grad_filter_,
                                       mkl_lt_internal_grad_filter,
                                       mkl_lt_grad_filter_),
               E_SUCCESS);
      AllocTmpBuffer(context, mkl_tmp_grad_filter_buf_tensor,
                     mkl_lt_internal_grad_filter, &mkl_buf_convert_grad_filter);
    }
    dnnLayoutDelete_F32(mkl_lt_internal_grad_filter);

    mkl_conv_grad_filter_res_[dnnResourceDiffFilter] =
        (mkl_convert_grad_filter) ? mkl_buf_convert_grad_filter
                                  : mkl_buf_grad_filter;
  }
};

#define REGISTER_MKL_FILTER_KERNELS(T)                    \
  REGISTER_KERNEL_BUILDER(Name("MklConv2DBackpropFilter") \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<T>("T")     \
                              .Label(mkl_layer_registry::kMklLayerLabel), \
                          MklConv2DCustomBackpropFilterOp<CPUDevice, T>);

TF_CALL_float(REGISTER_MKL_FILTER_KERNELS);
#undef REGISTER_MKL_FILTER_KERNELS
}  // namespace tensorflow

#endif  // INTEL_MKL
