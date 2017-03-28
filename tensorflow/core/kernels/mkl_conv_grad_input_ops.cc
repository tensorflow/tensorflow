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
// See docs in ../ops/nn_ops.cc. This opkernel uses MKL library, create MKL
// layout and primitives, use MKL dnn primitives to compute convolution backward
// input

#ifdef INTEL_MKL

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS
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
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "tensorflow/core/util/work_sharder.h"
#include "third_party/mkl/include/mkl_dnn.h"
#include "third_party/mkl/include/mkl_dnn_types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, class T>
class MklConv2DCustomBackpropInputOp : public OpKernel {
 public:
  ~MklConv2DCustomBackpropInputOp() {}
  explicit MklConv2DCustomBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string dataformat;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &dataformat));
    OP_REQUIRES(context, FormatFromString(dataformat, &data_format),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides));
    int stride_n = GetTensorDim(strides, data_format, 'N');
    int stride_c = GetTensorDim(strides, data_format, 'C');
    OP_REQUIRES(
        context, (stride_n == 1 && stride_c == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));

    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
  }

  typedef struct {
    int in_dims;
    size_t in_sizes[4];
    size_t in_strides[4];
    size_t out_sizes[4];
    size_t out_strides[4];
    int input_offset[2];
    size_t filter_size[4];
    size_t filter_stride[4];
    size_t conv_strides[2];
    MklShape mkl_filter_shape, mkl_outback_shape;
  } MklConvBackInputOpParams;

  // Create MKL dnnLayout_t objects for tensors coming into the layer
  void MklCreateInputLayouts(OpKernelContext* context) {
    bool filter_in_mkl_format = mkl_params.mkl_filter_shape.IsMklTensor();
    bool outback_in_mkl_format = mkl_params.mkl_outback_shape.IsMklTensor();
    if (filter_in_mkl_format) {
      mkl_lt_filter = (dnnLayout_t)mkl_params.mkl_filter_shape.GetCurLayout();
    } else {
      CHECK_EQ(
          dnnLayoutCreate_F32(&mkl_lt_filter, mkl_params.in_dims,
                              mkl_params.filter_size, mkl_params.filter_stride),
          E_SUCCESS);
    }

    if (outback_in_mkl_format) {
      mkl_lt_outbackprop =
          (dnnLayout_t)mkl_params.mkl_outback_shape.GetCurLayout();
    } else {
      CHECK_EQ(
          dnnLayoutCreate_F32(&mkl_lt_outbackprop, mkl_params.in_dims,
                              mkl_params.out_sizes, mkl_params.out_strides),
          E_SUCCESS);
    }
  }

  // Compare incoming input tensor layouts with MKL preferred layouts and
  // convert data to the preferred layout if necessary
  void MklPrepareConvolutionInputs(OpKernelContext* context,
                                   Tensor* mkl_tmp_outbackprop_buf_tensor,
                                   Tensor* mkl_tmp_filter_buf_tensor) {
    dnnPrimitive_t mkl_convert_filter = nullptr,
                   mkl_convert_outbackprop = nullptr;
    void *mkl_filter_buf = nullptr, *mkl_outbackprop_buf = nullptr;
    dnnLayout_t mkl_lt_filter_internal = nullptr,
                mkl_lt_outbackprop_internal = nullptr;
    CHECK_EQ(
        dnnLayoutCreateFromPrimitive_F32(
            &mkl_lt_filter_internal, mkl_convolutionbwdata, dnnResourceFilter),
        E_SUCCESS);

    const Tensor& filter = MklGetInput(context, 1);

    CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&mkl_lt_outbackprop_internal,
                                              mkl_convolutionbwdata,
                                              dnnResourceDiffDst),
             E_SUCCESS);
    if (!dnnLayoutCompare_F32(mkl_lt_filter_internal, mkl_lt_filter)) {
      // Create conversion primitive
      CHECK_EQ(dnnConversionCreate_F32(&mkl_convert_filter, mkl_lt_filter,
                                       mkl_lt_filter_internal),
               E_SUCCESS);

      AllocTmpBuffer(context, mkl_tmp_filter_buf_tensor, mkl_lt_filter_internal,
                     &mkl_filter_buf);
      CHECK_EQ(dnnConversionExecute_F32(
                   mkl_convert_filter,
                   static_cast<void*>(const_cast<T*>(filter.flat<T>().data())),
                   mkl_filter_buf),
               E_SUCCESS);

      // Assign filter buf to resources[] for convolution.
      conv_res[dnnResourceFilter] = mkl_filter_buf;
      dnnDelete_F32(mkl_convert_filter);
    } else {
      // If we do not need any layout conversion for filter, then
      // we direclty assign input filter to resources[].
      conv_res[dnnResourceFilter] =
          static_cast<void*>(const_cast<T*>(filter.flat<T>().data()));
    }
    dnnLayoutDelete_F32(mkl_lt_filter_internal);
    const Tensor& out_backprop = MklGetInput(context, 2);
    // --
    // We do similar steps as above for outputbackprop.
    if (!dnnLayoutCompare_F32(mkl_lt_outbackprop_internal,
                              mkl_lt_outbackprop)) {
      CHECK_EQ(
          dnnConversionCreate_F32(&mkl_convert_outbackprop, mkl_lt_outbackprop,
                                  mkl_lt_outbackprop_internal),
          E_SUCCESS);
      AllocTmpBuffer(context, mkl_tmp_outbackprop_buf_tensor,
                     mkl_lt_outbackprop_internal, &mkl_outbackprop_buf);

      CHECK_EQ(
          dnnConversionExecute_F32(
              mkl_convert_outbackprop,
              static_cast<void*>(const_cast<T*>(out_backprop.flat<T>().data())),
              mkl_outbackprop_buf),
          E_SUCCESS);

      conv_res[dnnResourceDiffDst] = mkl_outbackprop_buf;
      dnnDelete_F32(mkl_convert_outbackprop);
    } else {
      conv_res[dnnResourceDiffDst] =
          static_cast<void*>(const_cast<T*>(out_backprop.flat<T>().data()));
    }
    dnnLayoutDelete_F32(mkl_lt_outbackprop_internal);
  }

  // Cleanup member layouts and primitives
  void MklCleanup() {
    bool filter_in_mkl_format = mkl_params.mkl_filter_shape.IsMklTensor();
    bool outback_in_mkl_format = mkl_params.mkl_outback_shape.IsMklTensor();
    if (!filter_in_mkl_format) dnnLayoutDelete_F32(mkl_lt_filter);
    if (!outback_in_mkl_format) dnnLayoutDelete_F32(mkl_lt_outbackprop);
    dnnDelete_F32(mkl_convolutionbwdata);
  }
  void Compute(OpKernelContext* context) override {
    const Tensor& input = MklGetInput(context, 0);
    const Tensor& filter = MklGetInput(context, 1);

    GetMklShape(context, 1, &(mkl_params.mkl_filter_shape));
    bool filter_in_mkl_format = mkl_params.mkl_filter_shape.IsMklTensor();

    const Tensor& out_backprop = MklGetInput(context, 2);
    GetMklShape(context, 2, &(mkl_params.mkl_outback_shape));
    bool outback_in_mkl_format = mkl_params.mkl_outback_shape.IsMklTensor();

    TensorShape input_shape, filter_shape, outback_shape;

    // Generate input shape.
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropInput: input_sizes input must be 1-dim, not ",
            input.dims()));
    OP_REQUIRES_OK(
        context, TensorShapeUtils::MakeShape(input.vec<int32>(), &input_shape));

    // Generate shape for filter prop if input is in MKL format.
    if (filter_in_mkl_format) {
      OP_REQUIRES(context, mkl_params.mkl_filter_shape.GetDimension() == 4,
                  errors::InvalidArgument(
                      "Conv2DCustomBackpropInput: size must be 4-dim"));

      MklSizesToTFSizes(context, data_format, mkl_params.mkl_filter_shape,
                        &filter_shape);
    } else {
      filter_shape = filter.shape();
    }

    // Generate shape for outback prop if input is in MKL format.
    if (outback_in_mkl_format) {
      OP_REQUIRES(context, mkl_params.mkl_outback_shape.GetDimension() == 4,
                  errors::InvalidArgument(
                      "Conv2DCustomBackpropInput: size must be 4-dim"));

      MklSizesToTFSizes(context, data_format, mkl_params.mkl_outback_shape,
                        &outback_shape);
    } else {
      outback_shape = out_backprop.shape();
    }

    Conv2DBackpropDimensions dims;
    OP_REQUIRES_OK(context,
                   Conv2DBackpropComputeDimensions(
                       "Conv2DCustomBackpropInput", input_shape, filter_shape,
                       outback_shape, strides, padding, data_format, &dims));

    int64 pad_top, pad_bottom;
    int64 pad_left, pad_right;
    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                dims.rows.input_size, dims.rows.filter_size,
                                dims.rows.stride, padding,
                                &dims.rows.output_size, &pad_top, &pad_bottom));
    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                dims.cols.input_size, dims.cols.filter_size,
                                dims.cols.stride, padding,
                                &dims.cols.output_size, &pad_left, &pad_right));

    mkl_params.in_dims = 4;

    mkl_params.in_sizes[0] = static_cast<size_t>(dims.cols.input_size);
    mkl_params.in_sizes[1] = static_cast<size_t>(dims.rows.input_size);
    mkl_params.in_sizes[2] = static_cast<size_t>(dims.in_depth);
    mkl_params.in_sizes[3] = static_cast<size_t>(dims.batch_size);

    mkl_params.out_sizes[0] = static_cast<size_t>(dims.cols.output_size);
    mkl_params.out_sizes[1] = static_cast<size_t>(dims.rows.output_size);
    mkl_params.out_sizes[2] = static_cast<size_t>(dims.out_depth);
    mkl_params.out_sizes[3] = static_cast<size_t>(dims.batch_size);

    mkl_params.input_offset[0] = static_cast<int>(-pad_left);
    mkl_params.input_offset[1] = static_cast<int>(-pad_top);

    mkl_params.conv_strides[0] = static_cast<size_t>(dims.cols.stride);
    mkl_params.conv_strides[1] = static_cast<size_t>(dims.rows.stride);

    GetStridesFromSizes(data_format, mkl_params.out_strides,
                        mkl_params.out_sizes);
    GetStridesFromSizes(data_format, mkl_params.in_strides,
                        mkl_params.in_sizes);

    mkl_params.filter_size[0] = dims.cols.filter_size;
    mkl_params.filter_size[1] = dims.rows.filter_size;
    mkl_params.filter_size[2] = dims.in_depth;
    mkl_params.filter_size[3] = dims.out_depth;

    mkl_params.filter_stride[0] =
        mkl_params.filter_size[2] * mkl_params.filter_size[3];
    mkl_params.filter_stride[1] = mkl_params.filter_size[2] *
                                  mkl_params.filter_size[0] *
                                  mkl_params.filter_size[3];
    mkl_params.filter_stride[2] = mkl_params.filter_size[3];
    mkl_params.filter_stride[3] = 1;

    CHECK_EQ(dnnConvolutionCreateBackwardData_F32(
                 &mkl_convolutionbwdata, NULL, dnnAlgorithmConvolutionDirect,
                 mkl_params.in_dims, mkl_params.in_sizes, mkl_params.out_sizes,
                 mkl_params.filter_size, mkl_params.conv_strides,
                 mkl_params.input_offset, dnnBorderZeros),
             E_SUCCESS);

    // Allocate output tensor and shape
    TensorShape mkl_out_shape;
    MklShape mklOutputShape;
    mklOutputShape.SetMklTensor(true);
    mklOutputShape.SetMklLayout(mkl_convolutionbwdata, dnnResourceDiffSrc);
    mklOutputShape.SetTfLayout(mkl_params.in_dims, mkl_params.in_sizes,
                               mkl_params.in_strides);

    Tensor* in_backprop = nullptr;
    mkl_out_shape.AddDim(dnnLayoutGetMemorySize_F32(static_cast<dnnLayout_t>(
                             mklOutputShape.GetMklLayout())) /
                         sizeof(T));
    AllocateOutputSetMklshape(context, 0, &in_backprop, mkl_out_shape,
                              mklOutputShape);

    conv_res[dnnResourceDiffSrc] =
        static_cast<void*>(const_cast<T*>(in_backprop->flat<T>().data()));

    MklCreateInputLayouts(context);
    Tensor mkl_tmp_outbackprop_buf_tensor, mkl_tmp_filter_buf_tensor;
    MklPrepareConvolutionInputs(context, &mkl_tmp_outbackprop_buf_tensor,
                                &mkl_tmp_filter_buf_tensor);

    CHECK_EQ(dnnExecute_F32(mkl_convolutionbwdata, conv_res), E_SUCCESS);
    MklCleanup();
  }

 private:
  std::vector<int32> strides;
  Padding padding;
  TensorFormat data_format;
  MklConvBackInputOpParams mkl_params;
  dnnPrimitive_t mkl_convolutionbwdata = nullptr;
  void* conv_res[dnnResourceNumber];
  dnnLayout_t mkl_lt_filter = nullptr, mkl_lt_outbackprop = nullptr;
};

#define REGISTER_MKL_CPU_KERNELS(T)                                       \
  REGISTER_KERNEL_BUILDER(Name("MklConv2DBackpropInput")                  \
                              .Device(DEVICE_CPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .Label(mkl_layer_registry::kMklLayerLabel), \
                          MklConv2DCustomBackpropInputOp<CPUDevice, T>);

TF_CALL_float(REGISTER_MKL_CPU_KERNELS);
#undef REGISTER_MKL_CPU_KERNELS

}  // namespace tensorflow
#endif  // INTEL_MKL
