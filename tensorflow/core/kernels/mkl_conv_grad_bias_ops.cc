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
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "tensorflow/core/util/work_sharder.h"

#include "mkl_dnn.h"
#include "mkl_dnn_types.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

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

  void Compute(OpKernelContext* context) override {
    MklConvBackBiasOpContext mkl_context;
    const Tensor& input = MklGetInput(context, 0);
    GetMklShape(context, 0, &mkl_context.input_shape);
    bool input_is_mkl = mkl_context.input_shape.IsMklTensor();

    if (input_is_mkl) {
      OP_REQUIRES(
          context, mkl_context.input_shape.GetDimension() == 4,
          errors::InvalidArgument("Input tensor must be 4-dimensional"));
    } else {
      OP_REQUIRES(context, input.dims() == 4,
                  errors::InvalidArgument("input must be 4-dimensional",
                                          input.shape().DebugString()));
    }

    if (input_is_mkl) {
      mkl_context.c_size = mkl_context.input_shape.GetSizes()[MklDims::C];
    } else if (data_format_ == FORMAT_NHWC || data_format_ == FORMAT_NCHW) {
      mkl_context.c_size = GetTensorDim(input, data_format_, 'C');
    } else {
      errors::InvalidArgument("Unknown format ",
                              " Format must be either NCHW or NHWC. ");
    }
    TensorShape output_shape{mkl_context.c_size};

    Tensor* bias_backprop = nullptr;
    MklShape output_mkl_shape;
    output_mkl_shape.SetMklTensor(false);
    AllocateOutputSetMklShape(context, 0, &bias_backprop, output_shape,
                              output_mkl_shape);

    mkl_context.in_dims = 4;

    if (input_is_mkl) {  // get the shape from the mkl shape
      mkl_context.in_sizes[MklDims::W] =
          mkl_context.input_shape.GetSizes()[MklDims::W];
      mkl_context.in_sizes[MklDims::H] =
          mkl_context.input_shape.GetSizes()[MklDims::H];
      mkl_context.in_sizes[MklDims::C] =
          mkl_context.input_shape.GetSizes()[MklDims::C];
      mkl_context.in_sizes[MklDims::N] =
          mkl_context.input_shape.GetSizes()[MklDims::N];
    } else {
      mkl_context.in_sizes[MklDims::W] = GetTensorDim(input, data_format_, 'W');
      mkl_context.in_sizes[MklDims::H] = GetTensorDim(input, data_format_, 'H');
      mkl_context.in_sizes[MklDims::C] = GetTensorDim(input, data_format_, 'C');
      mkl_context.in_sizes[MklDims::N] = GetTensorDim(input, data_format_, 'N');
      GetStridesFromSizes(data_format_, mkl_context.in_strides,
                          mkl_context.in_sizes);
    }

    mkl_context.out_sizes[0] = mkl_context.c_size;
    mkl_context.out_strides[0] = 1;

    CHECK_EQ(
        dnnConvolutionCreateBackwardBias_F32(
            &mkl_context.prim_conv_bwdbias, NULL, dnnAlgorithmConvolutionDirect,
            mkl_context.in_dims, mkl_context.in_sizes),
        E_SUCCESS);

    mkl_context.MklCreateInputLayouts(context);

    Tensor mkl_tmp_input_buf, mkl_tmp_outbackprop_buf;
    mkl_context.MklPrepareConvolutionInputs(context, &mkl_tmp_input_buf);
    mkl_context.MklPrepareConvolutionOutputs(context, &mkl_tmp_outbackprop_buf,
                                             bias_backprop);

    CHECK_EQ(
        dnnExecute_F32(mkl_context.prim_conv_bwdbias, mkl_context.conv_res),
        E_SUCCESS);
    if (mkl_context.should_convert_output) {
      CHECK_EQ(dnnConversionExecute_F32(
                   mkl_context.convert_outbackprop, mkl_context.outbackprop_buf,
                   static_cast<void*>(bias_backprop->flat<T>().data())),
               E_SUCCESS);
    }
    // deletes layouts
    mkl_context.MklCleanup();
  }

 private:
  typedef struct {
    int in_dims;
    int c_size;
    size_t in_sizes[4];
    size_t in_strides[4];
    size_t out_sizes[1];
    size_t out_strides[1];
    size_t filter_sizes[4];
    size_t filter_strides[4];
    int input_offset[2];
    size_t conv_stride[2];
    MklShape input_shape;
    dnnPrimitive_t prim_conv_bwdbias;
    void* conv_res[dnnResourceNumber];
    dnnLayout_t lt_input, lt_outbackprop;
    bool should_convert_output;
    dnnPrimitive_t convert_outbackprop;
    void* outbackprop_buf;

    // Create MKL dnnLayout_t objects for tensors coming into the layer
    void MklCreateInputLayouts(OpKernelContext* context) {
      bool input_is_mkl = input_shape.IsMklTensor();

      CHECK_EQ(dnnLayoutCreate_F32(&lt_outbackprop, 1, out_sizes, out_strides),
               E_SUCCESS);
      if (input_is_mkl) {
        lt_input = static_cast<dnnLayout_t>(input_shape.GetCurLayout());
      } else {
        CHECK_EQ(dnnLayoutCreate_F32(&lt_input, in_dims, in_sizes, in_strides),
                 E_SUCCESS);
      }
    }

    // Compare incoming output tensor layouts with MKL preferred layouts and
    // convert data to the preferred layout if necessary
    void MklPrepareConvolutionOutputs(OpKernelContext* context,
                                      Tensor* mkl_tmp_outbackprop_buf,
                                      Tensor* bias_backprop) {
      dnnLayout_t mkl_prim_internal_outbackprop = nullptr;
      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&mkl_prim_internal_outbackprop,
                                                prim_conv_bwdbias,
                                                dnnResourceDiffBias),
               E_SUCCESS);
      should_convert_output =
          !dnnLayoutCompare_F32(lt_outbackprop, mkl_prim_internal_outbackprop);
      if (should_convert_output) {
        CHECK_EQ(dnnConversionCreate_F32(&convert_outbackprop,
                                         mkl_prim_internal_outbackprop,
                                         lt_outbackprop),
                 E_SUCCESS);
        AllocTmpBuffer(context, mkl_tmp_outbackprop_buf,
                       mkl_prim_internal_outbackprop, &outbackprop_buf);
        conv_res[dnnResourceDiffBias] = outbackprop_buf;
      } else {
        conv_res[dnnResourceDiffBias] =
            static_cast<void*>(const_cast<T*>(bias_backprop->flat<T>().data()));
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
      const Tensor& input = MklGetInput(context, 0);

      CHECK_EQ(
          dnnLayoutCreateFromPrimitive_F32(
              &mkl_prim_internal_input, prim_conv_bwdbias, dnnResourceDiffDst),
          E_SUCCESS);

      if (!dnnLayoutCompare_F32(lt_input, mkl_prim_internal_input)) {
        CHECK_EQ(dnnConversionCreate_F32(&mkl_convert_input, lt_input,
                                         mkl_prim_internal_input),
                 E_SUCCESS);
        AllocTmpBuffer(context, mkl_tmp_input_buf, mkl_prim_internal_input,
                       &input_buf);
        CHECK_EQ(dnnConversionExecute_F32(
                     mkl_convert_input,
                     static_cast<void*>(const_cast<T*>(input.flat<T>().data())),
                     input_buf),
                 E_SUCCESS);
        conv_res[dnnResourceDiffDst] = input_buf;
        dnnDelete_F32(mkl_convert_input);
      } else {
        conv_res[dnnResourceDiffDst] =
            static_cast<void*>(const_cast<T*>(input.flat<T>().data()));
      }

      dnnLayoutDelete_F32(mkl_prim_internal_input);
    }

    // Cleanup member layouts and primitives
    void MklCleanup() {
      bool input_is_mkl = input_shape.IsMklTensor();
      if (!input_is_mkl) dnnLayoutDelete_F32(lt_input);
      dnnLayoutDelete_F32(lt_outbackprop);

      if (should_convert_output) dnnDelete_F32(convert_outbackprop);
      dnnDelete_F32(prim_conv_bwdbias);
    }
  } MklConvBackBiasOpContext;

  TensorFormat data_format_;
  TF_DISALLOW_COPY_AND_ASSIGN(MklConv2DCustomBackpropBiasOp);
};

#define REGISTER_CPU_KERNELS(T)                                     \
  REGISTER_KERNEL_BUILDER(Name("_MklConv2DWithBiasBackpropBias")    \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T")               \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklConv2DCustomBackpropBiasOp<CPUDevice, T>);

TF_CALL_float(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS
} /* namespace tensorflow */
#endif /* INTEL_MKL */
