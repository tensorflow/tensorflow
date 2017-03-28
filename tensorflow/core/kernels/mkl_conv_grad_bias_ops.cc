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

// See docs in ../ops/nn_ops.cc.This opkernel uses MKL library, create MKL
// layout and primitives, use MKL dnn primitives to compute convolution backward
// bias.

#ifdef INTEL_MKL

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

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
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, class T>
class MklConv2DCustomBackpropBiasOp : public OpKernel {
 public:
  explicit MklConv2DCustomBackpropBiasOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
  }
  ~MklConv2DCustomBackpropBiasOp() {}

  typedef struct {
    int in_dims;
    int c_size;
    size_t in_sizes[4];
    size_t in_strides[4];
    size_t out_sizes[4];
    size_t out_strides[4];
    size_t filter_sizes[4];
    size_t filter_strides[4];
    int input_offset[2];
    size_t conv_stride[2];
    MklShape mkl_input_shape;
  } MklConvBackBiasOpParams;

  // Create MKL dnnLayout_t objects for tensors coming into the layer
  void MklCreateInputLayouts(OpKernelContext* context) {
    bool input_is_mkl = mkl_params_.mkl_input_shape.IsMklTensor();
    CHECK_EQ(dnnLayoutCreate_F32(&mkl_lt_outbackprop_, 1, mkl_params_.out_sizes,
                                 mkl_params_.out_strides),
             E_SUCCESS);
    if (input_is_mkl) {
      mkl_lt_input_ =
          static_cast<dnnLayout_t>(mkl_params_.mkl_input_shape.GetCurLayout());
    } else {
      CHECK_EQ(
          dnnLayoutCreate_F32(&mkl_lt_input_, mkl_params_.in_dims,
                              mkl_params_.in_sizes, mkl_params_.in_strides),
          E_SUCCESS);
    }
  }

  // Compare incoming output tensor layouts with MKL preferred layouts and
  // convert data to the preferred layout if necessary
  void MklPrepareConvolutionOutputs(OpKernelContext* context,
                                    Tensor* mkl_tmp_outbackprop_buf_,
                                    Tensor* bias_backprop) {
    dnnLayout_t mkl_prim_internal_outbackprop = nullptr;
    CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&mkl_prim_internal_outbackprop,
                                              mkl_convolution_bwd_bias_,
                                              dnnResourceDiffBias),
             E_SUCCESS);

    if (!dnnLayoutCompare_F32(mkl_lt_outbackprop_,
                              mkl_prim_internal_outbackprop)) {
      CHECK_EQ(dnnConversionCreate_F32(&mkl_convert_outbackprop_,
                                       mkl_prim_internal_outbackprop,
                                       mkl_lt_outbackprop_),
               E_SUCCESS);
      AllocTmpBuffer(context, mkl_tmp_outbackprop_buf_,
                     mkl_prim_internal_outbackprop, &outbackprop_buf_);
    }

    if (mkl_convert_outbackprop_ == nullptr) {
      conv_res_[dnnResourceDiffBias] =
          static_cast<void*>(const_cast<T*>(bias_backprop->flat<T>().data()));
    } else {
      conv_res_[dnnResourceDiffBias] = outbackprop_buf_;
    }

    dnnLayoutDelete_F32(mkl_prim_internal_outbackprop);
  }

  // Compare incoming input tensor layouts with MKL preferred layouts and
  // convert data to the preferred layout if necessary
  void MklPrepareConvolutionInputs(OpKernelContext* context,
                                   Tensor* mkl_tmp_input_buf) {
    dnnLayout_t mkl_prim_internal_input = nullptr;
    dnnPrimitive_t mkl_convert_input = nullptr;
    void* input_buf = nullptr;

    CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&mkl_prim_internal_input,
                                              mkl_convolution_bwd_bias_,
                                              dnnResourceDiffDst),
             E_SUCCESS);

    if (!dnnLayoutCompare_F32(mkl_lt_input_, mkl_prim_internal_input)) {
      CHECK_EQ(dnnConversionCreate_F32(&mkl_convert_input, mkl_lt_input_,
                                       mkl_prim_internal_input),
               E_SUCCESS);
      AllocTmpBuffer(context, mkl_tmp_input_buf, mkl_prim_internal_input,
                     &input_buf);
    }

    const Tensor& input = MklGetInput(context, 0);
    // performs conversions
    if (mkl_convert_input == nullptr) {
      conv_res_[dnnResourceDiffDst] =
          static_cast<void*>(const_cast<T*>(input.flat<T>().data()));
    } else {
      CHECK_EQ(dnnConversionExecute_F32(
                   mkl_convert_input,
                   static_cast<void*>(const_cast<T*>(input.flat<T>().data())),
                   input_buf),
               E_SUCCESS);

      conv_res_[dnnResourceDiffDst] = input_buf;
      dnnDelete_F32(mkl_convert_input);
    }
    dnnLayoutDelete_F32(mkl_prim_internal_input);
  }

  // Cleanup member layouts and primitives
  void MklCleanup() {
    bool input_is_mkl = mkl_params_.mkl_input_shape.IsMklTensor();
    if (!input_is_mkl) dnnLayoutDelete_F32(mkl_lt_input_);
    dnnLayoutDelete_F32(mkl_lt_outbackprop_);

    if (mkl_convert_outbackprop_) dnnDelete_F32(mkl_convert_outbackprop_);
    dnnDelete_F32(mkl_convolution_bwd_bias_);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = MklGetInput(context, 0);
    GetMklShape(context, 0, &mkl_params_.mkl_input_shape);
    bool input_is_mkl = mkl_params_.mkl_input_shape.IsMklTensor();

    if (input_is_mkl) {
      OP_REQUIRES(
          context, mkl_params_.mkl_input_shape.GetDimension() == 4,
          errors::InvalidArgument("Input tensor must be 4-dimensional"));
    } else {
      OP_REQUIRES(context, input.dims() == 4,
                  errors::InvalidArgument("input must be 4-dimensional",
                                          input.shape().DebugString()));
    }

    if (input_is_mkl) {
      mkl_params_.c_size = mkl_params_.mkl_input_shape.GetSizes()[2];
    } else if (data_format_ == FORMAT_NHWC || data_format_ == FORMAT_NCHW) {
      mkl_params_.c_size = GetTensorDim(input, data_format_, 'C');
    } else {
      errors::InvalidArgument("Unknown format ",
                              " Format must be either NCHW or NHWC. ");
    }
    TensorShape output_shape{mkl_params_.c_size};

    Tensor* bias_backprop = nullptr;
    MklShape output_mkl_shape;
    output_mkl_shape.SetMklTensor(false);
    AllocateOutputSetMklshape(context, 0, &bias_backprop, output_shape,
                              output_mkl_shape);

    mkl_params_.in_dims = 4;

    if (input_is_mkl) {  // get the shape from the mkl shape
      mkl_params_.c_size = mkl_params_.mkl_input_shape.GetSizes()[2];
      mkl_params_.in_sizes[0] = mkl_params_.mkl_input_shape.GetSizes()[0];
      mkl_params_.in_sizes[1] = mkl_params_.mkl_input_shape.GetSizes()[1];
      mkl_params_.in_sizes[2] = mkl_params_.mkl_input_shape.GetSizes()[2];
      mkl_params_.in_sizes[3] = mkl_params_.mkl_input_shape.GetSizes()[3];

      mkl_params_.in_strides[0] = mkl_params_.mkl_input_shape.GetStrides()[0];
      mkl_params_.in_strides[1] = mkl_params_.mkl_input_shape.GetStrides()[1];
      mkl_params_.in_strides[2] = mkl_params_.mkl_input_shape.GetStrides()[2];
      mkl_params_.in_strides[3] = mkl_params_.mkl_input_shape.GetStrides()[3];
    } else {
      mkl_params_.in_sizes[0] = GetTensorDim(input, data_format_, 'W');
      mkl_params_.in_sizes[1] = GetTensorDim(input, data_format_, 'H');
      mkl_params_.in_sizes[2] = GetTensorDim(input, data_format_, 'C');
      mkl_params_.in_sizes[3] = GetTensorDim(input, data_format_, 'N');
      GetStridesFromSizes(data_format_, mkl_params_.in_strides,
                          mkl_params_.in_sizes);
    }
    mkl_params_.out_sizes[0] = mkl_params_.c_size;
    mkl_params_.out_strides[0] = 1;

    CHECK_EQ(
        dnnConvolutionCreateBackwardBias_F32(
            &mkl_convolution_bwd_bias_, NULL, dnnAlgorithmConvolutionDirect,
            mkl_params_.in_dims, mkl_params_.in_sizes),
        E_SUCCESS);

    MklCreateInputLayouts(context);

    Tensor mkl_tmp_input_buf, mkl_tmp_outbackprop_buf_;
    MklPrepareConvolutionInputs(context, &mkl_tmp_input_buf);
    MklPrepareConvolutionOutputs(context, &mkl_tmp_outbackprop_buf_,
                                 bias_backprop);
    CHECK_EQ(dnnExecute_F32(mkl_convolution_bwd_bias_, conv_res_), E_SUCCESS);
    if (mkl_convert_outbackprop_ != nullptr) {
      CHECK_EQ(dnnConversionExecute_F32(
                   mkl_convert_outbackprop_, outbackprop_buf_,
                   static_cast<void*>(bias_backprop->flat<T>().data())),
               E_SUCCESS);
    }
    // deletes layouts
    MklCleanup();
  }

 private:
  TensorFormat data_format_;
  MklConvBackBiasOpParams mkl_params_;
  dnnPrimitive_t mkl_convolution_bwd_bias_ = nullptr;
  void* conv_res_[dnnResourceNumber];
  dnnLayout_t mkl_lt_input_ = nullptr, mkl_lt_outbackprop_ = nullptr;
  dnnPrimitive_t mkl_convert_outbackprop_ = nullptr;
  void* outbackprop_buf_ = nullptr;
  TF_DISALLOW_COPY_AND_ASSIGN(MklConv2DCustomBackpropBiasOp);
};

#define REGISTER_CPU_KERNELS(T)                                           \
  REGISTER_KERNEL_BUILDER(Name("MklConv2DWithBiasBackpropBias")           \
                              .Device(DEVICE_CPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .Label(mkl_layer_registry::kMklLayerLabel), \
                          MklConv2DCustomBackpropBiasOp<CPUDevice, T>);

TF_CALL_float(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS
} /* namespace tensorflow */
#endif /* INTEL_MKL */
