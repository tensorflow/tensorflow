/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#ifdef INTEL_MKL

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/tensor_format.h"


#ifndef INTEL_MKL_ML
#include "mkldnn.hpp"
using mkldnn::batch_normalization_backward;
using mkldnn::batch_normalization_forward;
using mkldnn::prop_kind;
using mkldnn::stream;
using mkldnn::use_global_stats;
using mkldnn::use_scale_shift;
#else
#include "mkl_dnn.h"
#include "mkl_dnn_types.h"
#endif

#include "tensorflow/core/util/mkl_util.h"
// TODO(inteltf) Address comments from PR 8968.

namespace tensorflow {
using CPUDevice = Eigen::ThreadPoolDevice;

#ifdef INTEL_MKL_ML

template <typename Device, typename T>
class MklFusedBatchNormOp : public OpKernel {
 public:
  explicit MklFusedBatchNormOp(OpKernelConstruction* context)
      : OpKernel(context) {
    float epsilon;
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
    epsilon_ = T(epsilon);
    string tensor_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &tensor_format));
    OP_REQUIRES(context, FormatFromString(tensor_format, &tensor_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));
  }

  void Compute(OpKernelContext* context) override {
    MklFusedBatchNormOpContext mkl_context;
    const Tensor& input = MklGetInput(context, 0);
    const Tensor& scale = MklGetInput(context, 1);
    const Tensor& shift = MklGetInput(context, 2);
    const Tensor& est_mean = MklGetInput(context, 3);
    const Tensor& est_variance = MklGetInput(context, 4);

    GetMklShape(context, 0, &(mkl_context.mkl_shape_input_shape));
    bool input_in_mkl_format = mkl_context.mkl_shape_input_shape.IsMklTensor();

    if (!input_in_mkl_format) {
      OP_REQUIRES(context, input.dims() == 4,
                  errors::InvalidArgument("input must be 4-dimensional",
                                          input.shape().DebugString()));
    }
    OP_REQUIRES(context, scale.dims() == 1,
                errors::InvalidArgument("scale must be 1-dimensional",
                                        scale.shape().DebugString()));
    OP_REQUIRES(context, shift.dims() == 1,
                errors::InvalidArgument("offset must be 1-dimensional",
                                        shift.shape().DebugString()));
    OP_REQUIRES(context, est_mean.dims() == 1,
                errors::InvalidArgument("estimated_mean must be 1-dimensional",
                                        est_mean.shape().DebugString()));

    OP_REQUIRES(
        context, est_variance.dims() == 1,
        errors::InvalidArgument("estimated_variance must be 1-dimensional",
                                est_variance.shape().DebugString()));

    if (is_training_) {
      OP_REQUIRES(context, est_mean.dim_size(0) == 0,
                  errors::InvalidArgument("estimated_mean empty for training",
                                          est_mean.shape().DebugString()));
      OP_REQUIRES(context, est_variance.dim_size(0) == 0,
                  errors::InvalidArgument(
                      "estimated_variance must be empty for training",
                      est_variance.shape().DebugString()));
    }

    unsigned int flag_batch_norm =
        is_training_ ? dnnUseScaleShift
                     : (dnnUseInputMeanVariance | dnnUseScaleShift);

    mkl_context.MklExtractParams(context, tensor_format_);

    // Create layout only for input data as it is used in Op primitive.
    mkl_context.MklCreateInputLayout(context);

    // Create Op primitive.
    CHECK_EQ(dnnBatchNormalizationCreateForward_v2_F32(
                 &(mkl_context.mkl_prim_batchnorm), nullptr,
                 mkl_context.mkl_lt_input, static_cast<float>(epsilon_),
                 flag_batch_norm),
             E_SUCCESS);

    // Temporary tensors with buffers for the context inputs, if
    // conversion to MKL-Op specific layouts are required. It is assumed here
    // that TF's 1D tensors (scale, shift, est_mean, and est_variance) won't
    // require any conversion.
    // Since scale-shift is combined in MKL, a buffer is required.
    Tensor mkl_tmp_input_buf_tensor, mkl_tmp_scale_shift_buf_tensor;
    mkl_context.MklPrepareContextInputs(context, &mkl_tmp_input_buf_tensor,
                                        &mkl_tmp_scale_shift_buf_tensor);

    // Output data in MKL layout
    Tensor* output = nullptr;
    TensorShape tf_shape_output;
    MklShape mkl_shape_output;
    mkl_shape_output.SetMklTensor(true);
    mkl_shape_output.SetMklLayout(mkl_context.mkl_prim_batchnorm,
                                  dnnResourceDst);
    mkl_shape_output.SetTfLayout(mkl_context.mkl_params.in_dim,
                                 mkl_context.mkl_params.in_sizes,
                                 mkl_context.mkl_params.in_strides);
    mkl_shape_output.SetTfDimOrder(mkl_context.mkl_params.in_dim,
                                   tensor_format_);
    tf_shape_output.AddDim(dnnLayoutGetMemorySize_F32(static_cast<dnnLayout_t>(
                               mkl_shape_output.GetMklLayout())) /
                           sizeof(T));
    AllocateOutputSetMklShape(context, 0, &output, tf_shape_output,
                              mkl_shape_output);
    mkl_context.mkl_res_batchnorm[dnnResourceDst] =
        static_cast<void*>(output->flat<T>().data());

    // Batch mean in TF layout
    Tensor* batch_mean = nullptr;
    MklShape mkl_shape_batch_mean;
    mkl_shape_batch_mean.SetMklTensor(false);
    AllocateOutputSetMklShape(context, 1, &batch_mean, scale.shape(),
                              mkl_shape_batch_mean);
    // Batch variance in TF layout
    Tensor* batch_variance = nullptr;
    MklShape mkl_shape_batch_variance;
    mkl_shape_batch_variance.SetMklTensor(false);
    AllocateOutputSetMklShape(context, 2, &batch_variance, scale.shape(),
                              mkl_shape_batch_variance);
    // If training mode, set dnnResourceMean and dnnResourceVariance to
    // output tensors for batch mean and variance.
    // Otherwise, set dnnResourceMean and dnnResourceVariance to
    // estimated mean and variance.
    if (is_training_)
      mkl_context.MklSetMeanVariance(*batch_mean, *batch_variance);
    else
      mkl_context.MklSetMeanVariance(est_mean, est_variance);

    // Now that all resources are set, it is ready for dnnExecute
    CHECK_EQ(dnnExecute_F32(mkl_context.mkl_prim_batchnorm,
                            mkl_context.mkl_res_batchnorm),
             E_SUCCESS);

    // Mean and variance (without Bessel's correction) saved for backward
    // computation to serve as pre-computed mean and variance.
    Tensor* saved_mean = nullptr;
    MklShape mkl_shape_saved_mean;
    mkl_shape_saved_mean.SetMklTensor(false);
    AllocateOutputSetMklShape(context, 3, &saved_mean, scale.shape(),
                              mkl_shape_saved_mean);
    std::memcpy(
        reinterpret_cast<char*>(saved_mean->flat<float>().data()),
        reinterpret_cast<char*>(mkl_context.mkl_res_batchnorm[dnnResourceMean]),
        scale.NumElements() * sizeof(float));
    Tensor* saved_variance = nullptr;
    MklShape mkl_shape_saved_variance;
    mkl_shape_saved_variance.SetMklTensor(false);
    AllocateOutputSetMklShape(context, 4, &saved_variance, scale.shape(),
                              mkl_shape_saved_variance);
    std::memcpy(reinterpret_cast<char*>(saved_variance->flat<float>().data()),
                reinterpret_cast<char*>(
                    mkl_context.mkl_res_batchnorm[dnnResourceVariance]),
                scale.NumElements() * sizeof(float));

    // Bessel's correction on variance, if training mode is on
    if (is_training_) {
      float* p_var = static_cast<float*>(batch_variance->flat<T>().data());
      auto depth = mkl_context.mkl_params.depth;
      size_t orig_size = mkl_context.mkl_params.in_sizes[0] *
                         mkl_context.mkl_params.in_sizes[1] *
                         mkl_context.mkl_params.in_sizes[3];
      size_t adjust_size = orig_size - 1;
      float adjust_factor = (static_cast<float>(orig_size)) / adjust_size;
      for (int i = 0; i < depth; i++) p_var[i] = adjust_factor * p_var[i];
    }

    mkl_context.MklCleanup();
  }

 private:
  T epsilon_;
  TensorFormat tensor_format_;
  bool is_training_;

  // Structure containing all info for MklOp
  typedef struct {
    // Parameters used for input and output layouts
    struct MklBatchNormParams {
      // BatchNormOp src and
      size_t in_dim;
      size_t in_sizes[4];
      size_t in_strides[4];
      size_t depth;  // Batch normalization is done for per channel.
    } mkl_params;

    MklShape mkl_shape_input_shape;

    // MKL primitive and resources for BatchNormOp
    dnnPrimitive_t mkl_prim_batchnorm = nullptr;
    void* mkl_res_batchnorm[dnnResourceNumber];

    // MKL layouts for inputs in the context
    dnnLayout_t mkl_lt_input = nullptr;

    void MklCleanup() {
      bool input_in_mkl_format = mkl_shape_input_shape.IsMklTensor();
      if (!input_in_mkl_format) dnnLayoutDelete_F32(mkl_lt_input);
      if (mkl_prim_batchnorm != nullptr) dnnDelete_F32(mkl_prim_batchnorm);
    }

    void MklExtractParams(OpKernelContext* context,
                          const TensorFormat& tensor_format) {
      const Tensor& input = MklGetInput(context, 0);
      bool input_in_mkl_format = mkl_shape_input_shape.IsMklTensor();
      mkl_params.in_dim = input_in_mkl_format
                              ? mkl_shape_input_shape.GetDimension()
                              : input.dims();
      mkl_params.in_sizes[0] = static_cast<size_t>(
          input_in_mkl_format ? mkl_shape_input_shape.GetSizes()[0]
                              : GetTensorDim(input, tensor_format, 'W'));
      mkl_params.in_sizes[1] = static_cast<size_t>(
          input_in_mkl_format ? mkl_shape_input_shape.GetSizes()[1]
                              : GetTensorDim(input, tensor_format, 'H'));
      mkl_params.in_sizes[2] = static_cast<size_t>(
          input_in_mkl_format ? mkl_shape_input_shape.GetSizes()[2]
                              : GetTensorDim(input, tensor_format, 'C'));
      mkl_params.in_sizes[3] = static_cast<size_t>(
          input_in_mkl_format ? mkl_shape_input_shape.GetSizes()[3]
                              : GetTensorDim(input, tensor_format, 'N'));
      mkl_params.depth = mkl_params.in_sizes[2];
      GetStridesFromSizes(tensor_format, mkl_params.in_strides,
                          mkl_params.in_sizes);
    }

    void MklCreateInputLayout(OpKernelContext* context) {
      const Tensor& input = MklGetInput(context, 0);
      bool input_in_mkl_format = mkl_shape_input_shape.IsMklTensor();
      if (input_in_mkl_format) {
        mkl_lt_input =
            static_cast<dnnLayout_t>(mkl_shape_input_shape.GetCurLayout());
      } else {
        CHECK_EQ(
            dnnLayoutCreate_F32(&mkl_lt_input, mkl_params.in_dim,
                                mkl_params.in_sizes, mkl_params.in_strides),
            E_SUCCESS);
      }
    }
    void MklPrepareContextInputs(OpKernelContext* context,
                                 Tensor* mkl_tmp_input_buf_tensor,
                                 Tensor* mkl_tmp_scale_shift_buf_tensor) {
      bool mkl_convert_input;
      dnnPrimitive_t mkl_prim_convert_input = nullptr;
      dnnLayout_t mkl_lt_internal_input = nullptr;
      void* mkl_buf_converted_input = nullptr;
      // Compare with internal layouts and convert if needed
      const Tensor& input = MklGetInput(context, 0);
      void* mkl_buf_input =
          const_cast<void*>(static_cast<const void*>(input.flat<T>().data()));
      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(
                   &mkl_lt_internal_input, mkl_prim_batchnorm, dnnResourceSrc),
               E_SUCCESS);
      mkl_convert_input =
          !dnnLayoutCompare_F32(mkl_lt_internal_input, mkl_lt_input);
      if (mkl_convert_input) {
        CHECK_EQ(dnnConversionCreate_F32(&mkl_prim_convert_input, mkl_lt_input,
                                         mkl_lt_internal_input),
                 E_SUCCESS);
        AllocTmpBuffer(context, mkl_tmp_input_buf_tensor, mkl_lt_internal_input,
                       &mkl_buf_converted_input);
        CHECK_EQ(dnnConversionExecute_F32(mkl_prim_convert_input, mkl_buf_input,
                                          mkl_buf_converted_input),
                 E_SUCCESS);
        dnnDelete_F32(mkl_prim_convert_input);
      }
      dnnLayoutDelete_F32(mkl_lt_internal_input);
      mkl_res_batchnorm[dnnResourceSrc] =
          (mkl_convert_input) ? mkl_buf_converted_input : mkl_buf_input;

      // scale-shift layout is created from primitive. So no conversion
      // is needed, however, a buffer has to be allocated.
      dnnLayout_t mkl_lt_scale_shift = nullptr;
      void* mkl_buf_scale_shift = nullptr;
      CHECK_EQ(
          dnnLayoutCreateFromPrimitive_F32(
              &mkl_lt_scale_shift, mkl_prim_batchnorm, dnnResourceScaleShift),
          E_SUCCESS);
      AllocTmpBuffer(context, mkl_tmp_scale_shift_buf_tensor,
                     mkl_lt_scale_shift, &mkl_buf_scale_shift);
      // Fill the scale-shift buffer with data, presumably buffer is 2D array
      const Tensor& scale = MklGetInput(context, 1);
      const Tensor& shift = MklGetInput(context, 2);
      float* buf_scale_shift = static_cast<float*>(mkl_buf_scale_shift);
      float* buf_scale = const_cast<float*>(
          static_cast<const float*>(scale.flat<float>().data()));
      float* buf_shift = const_cast<float*>(
          static_cast<const float*>(shift.flat<float>().data()));
      auto depth = mkl_params.depth;
      for (int i = 0; i < depth; i++) {
        buf_scale_shift[i] = buf_scale[i];
        buf_scale_shift[i + depth] = buf_shift[i];
      }
      mkl_res_batchnorm[dnnResourceScaleShift] = mkl_buf_scale_shift;
    }

    inline void MklSetMeanVariance(const Tensor& mean, const Tensor& variance) {
      mkl_res_batchnorm[dnnResourceMean] = const_cast<void*>(
          static_cast<const void*>(mean.flat<float>().data()));
      mkl_res_batchnorm[dnnResourceVariance] = const_cast<void*>(
          static_cast<const void*>(variance.flat<float>().data()));
    }
  } MklFusedBatchNormOpContext;
};

template <typename Device, typename T>
class MklFusedBatchNormGradOp : public OpKernel {
 public:
  explicit MklFusedBatchNormGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    float epsilon;
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
    epsilon_ = T(epsilon);
    string tensor_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &tensor_format));
    OP_REQUIRES(context, FormatFromString(tensor_format, &tensor_format_),
                errors::InvalidArgument("Invalid data format"));
  }

  void Compute(OpKernelContext* context) override {
    MklFusedBatchNormGradOpContext mkl_context;

    const Tensor& out_backprop = MklGetInput(context, 0);
    const Tensor& input = MklGetInput(context, 1);
    const Tensor& scale = MklGetInput(context, 2);
    const Tensor& saved_mean = MklGetInput(context, 3);
    const Tensor& saved_var = MklGetInput(context, 4);

    // Here scale, mean, and variance are 1D and considered
    // those having same layout in MKL and TF
    GetMklShape(context, 0, &(mkl_context.mkl_shape_out_backprop));
    GetMklShape(context, 1, &(mkl_context.mkl_shape_input_shape));

    bool input_in_mkl_format = mkl_context.mkl_shape_input_shape.IsMklTensor();
    bool out_backprop_in_mkl_format =
        mkl_context.mkl_shape_out_backprop.IsMklTensor();
    if (!out_backprop_in_mkl_format) {
      OP_REQUIRES(context, out_backprop.dims() == 4,
                  errors::InvalidArgument("input must be 4-dimensional",
                                          out_backprop.shape().DebugString()));
    }
    if (!input_in_mkl_format) {
      OP_REQUIRES(context, input.dims() == 4,
                  errors::InvalidArgument("input must be 4-dimensional",
                                          input.shape().DebugString()));
    }
    OP_REQUIRES(context, scale.dims() == 1,
                errors::InvalidArgument("scale must be 1-dimensional",
                                        scale.shape().DebugString()));
    OP_REQUIRES(context, saved_mean.dims() == 1,
                errors::InvalidArgument("saved mean must be 1-dimensional",
                                        saved_mean.shape().DebugString()));
    OP_REQUIRES(context, saved_var.dims() == 1,
                errors::InvalidArgument("saved variance must be 1-dimensional",
                                        saved_var.shape().DebugString()));

    mkl_context.MklExtractParams(context, tensor_format_);

    mkl_context.MklCreateInputLayout(context);

    unsigned int flag_batch_norm_grad = dnnUseScaleShift;

    // Create Backward Op primitive.
    CHECK_EQ(dnnBatchNormalizationCreateBackward_v2_F32(
                 &(mkl_context.mkl_prim_batchnorm_bwd), nullptr,
                 mkl_context.mkl_lt_input, static_cast<float>(epsilon_),
                 flag_batch_norm_grad),
             E_SUCCESS);

    // Temporary tensors and their buffers if conversion is required
    Tensor mkl_tmp_input_buf_tensor, mkl_tmp_outbackprop_buf_tensor,
        mkl_tmp_scaleshift_buf_tensor;
    mkl_context.MklPrepareContextInputs(context, &mkl_tmp_input_buf_tensor,
                                        &mkl_tmp_outbackprop_buf_tensor,
                                        &mkl_tmp_scaleshift_buf_tensor);

    // Allocate tensor for grad w.r.t. input(x)
    Tensor* in_backprop = nullptr;
    TensorShape tf_shape_in_backprop;
    MklShape mkl_shape_in_backprop;
    mkl_shape_in_backprop.SetMklTensor(true);
    mkl_shape_in_backprop.SetMklLayout(mkl_context.mkl_prim_batchnorm_bwd,
                                       dnnResourceDiffSrc);
    mkl_shape_in_backprop.SetTfLayout(mkl_context.mkl_params.in_dims,
                                      mkl_context.mkl_params.in_sizes,
                                      mkl_context.mkl_params.in_strides);
    mkl_shape_in_backprop.SetTfDimOrder(mkl_context.mkl_params.in_dims,
                                        tensor_format_);
    tf_shape_in_backprop.AddDim(
        dnnLayoutGetMemorySize_F32(
            static_cast<dnnLayout_t>(mkl_shape_in_backprop.GetMklLayout())) /
        sizeof(T));
    AllocateOutputSetMklShape(context, 0, &in_backprop, tf_shape_in_backprop,
                              mkl_shape_in_backprop);
    mkl_context.mkl_res_batchnorm_bwd[dnnResourceDiffSrc] =
        static_cast<void*>(in_backprop->flat<T>().data());

    // grad_scale and grad_shift are combined together in MKL
    // So create a single temporary buffer for those.
    // Also set dnnResourceDiffScaleShift to the temporary buffer
    Tensor mkl_tmp_grad_scale_shift_buf_tensor;
    mkl_context.MklPrepareGradScaleShift(context,
                                         &mkl_tmp_grad_scale_shift_buf_tensor);

    // All dnn resources are set now, ready to execute
    CHECK_EQ(dnnExecute_F32(mkl_context.mkl_prim_batchnorm_bwd,
                            mkl_context.mkl_res_batchnorm_bwd),
             E_SUCCESS);

    // Now separate out scale and shift grad and copy to individual tensors
    const TensorShape& tf_shape_scale_shift = scale.shape();
    // Allocate tensor for grad w.r.t. scale (beta)
    Tensor* scale_backprop = nullptr;
    MklShape mkl_shape_scale_backprop;
    AllocateOutputSetMklShape(context, 1, &scale_backprop, tf_shape_scale_shift,
                              mkl_shape_scale_backprop);

    // Allocate tensor for grad w.r.t. shift(gamma)
    Tensor* shift_backprop = nullptr;
    MklShape mkl_shape_shift_backprop;
    AllocateOutputSetMklShape(context, 2, &shift_backprop, tf_shape_scale_shift,
                              mkl_shape_shift_backprop);

    // copy scale and shift grads to tensors
    float* mkl_buf_scale_shift = const_cast<float*>(static_cast<const float*>(
        mkl_tmp_grad_scale_shift_buf_tensor.flat<T>().data()));
    float* tf_buf_scale = const_cast<float*>(
        static_cast<const float*>(scale_backprop->flat<T>().data()));
    float* tf_buf_shift = const_cast<float*>(
        static_cast<const float*>(shift_backprop->flat<T>().data()));
    auto depth = mkl_context.mkl_params.depth;
    for (int i = 0; i < depth; i++) {
      tf_buf_scale[i] = mkl_buf_scale_shift[i];
      tf_buf_shift[i] = mkl_buf_scale_shift[i + depth];
    }

    // Two placeholders for estimated_mean and estimated_variance, which are
    // used for inference and thus not needed here for gradient computation.
    Tensor* placeholder_1 = nullptr;
    MklShape mkl_shape_placeholder_1;
    AllocateOutputSetMklShape(context, 3, &placeholder_1, TensorShape({}),
                              mkl_shape_placeholder_1);
    Tensor* placeholder_2 = nullptr;
    MklShape mkl_shape_placeholder_2;
    AllocateOutputSetMklShape(context, 4, &placeholder_2, TensorShape({}),
                              mkl_shape_placeholder_2);

    mkl_context.MklCleanup();
  }

 private:
  T epsilon_;
  TensorFormat tensor_format_;

  // Structure containing all info for MklOp
  typedef struct {
    // Parameters used for input and output layouts
    struct MklBatchNormParams {
      // BatchNormOp src and
      size_t in_dims;
      size_t in_sizes[4];
      size_t in_strides[4];
      size_t depth;  // Batch normalization is done for per channel.
    } mkl_params;

    MklShape mkl_shape_out_backprop;
    MklShape mkl_shape_input_shape;

    // MKL primitive and resources for BatchNormOp
    dnnPrimitive_t mkl_prim_batchnorm_bwd = nullptr;
    void* mkl_res_batchnorm_bwd[dnnResourceNumber];

    // MKL layouts for inputs in the context
    dnnLayout_t mkl_lt_out_backprop = nullptr;
    dnnLayout_t mkl_lt_input = nullptr;

    void MklCleanup() {
      bool input_in_mkl_format = mkl_shape_input_shape.IsMklTensor();
      bool out_backprop_in_mkl_format = mkl_shape_out_backprop.IsMklTensor();
      if (!input_in_mkl_format) dnnLayoutDelete_F32(mkl_lt_input);
      if (!out_backprop_in_mkl_format) dnnLayoutDelete_F32(mkl_lt_out_backprop);

      dnnDelete_F32(mkl_prim_batchnorm_bwd);
    }

    void MklExtractParams(OpKernelContext* context,
                          const TensorFormat& tensor_format) {
      const Tensor& input = MklGetInput(context, 1);
      bool input_in_mkl_format = mkl_shape_input_shape.IsMklTensor();
      mkl_params.in_dims = input_in_mkl_format
                               ? mkl_shape_input_shape.GetDimension()
                               : input.dims();
      mkl_params.in_sizes[0] = static_cast<size_t>(
          input_in_mkl_format ? mkl_shape_input_shape.GetSizes()[0]
                              : GetTensorDim(input, tensor_format, 'W'));
      mkl_params.in_sizes[1] = static_cast<size_t>(
          input_in_mkl_format ? mkl_shape_input_shape.GetSizes()[1]
                              : GetTensorDim(input, tensor_format, 'H'));
      mkl_params.in_sizes[2] = static_cast<size_t>(
          input_in_mkl_format ? mkl_shape_input_shape.GetSizes()[2]
                              : GetTensorDim(input, tensor_format, 'C'));
      mkl_params.in_sizes[3] = static_cast<size_t>(
          input_in_mkl_format ? mkl_shape_input_shape.GetSizes()[3]
                              : GetTensorDim(input, tensor_format, 'N'));
      mkl_params.depth = mkl_params.in_sizes[2];
      GetStridesFromSizes(tensor_format, mkl_params.in_strides,
                          mkl_params.in_sizes);
    }

    void MklCreateInputLayout(OpKernelContext* context) {
      const Tensor& input = MklGetInput(context, 0);
      bool input_in_mkl_format = mkl_shape_input_shape.IsMklTensor();
      if (input_in_mkl_format) {
        mkl_lt_input =
            static_cast<dnnLayout_t>(mkl_shape_input_shape.GetCurLayout());
      } else {
        CHECK_EQ(
            dnnLayoutCreate_F32(&mkl_lt_input, mkl_params.in_dims,
                                mkl_params.in_sizes, mkl_params.in_strides),
            E_SUCCESS);
      }

      bool out_backprop_in_mkl_format = mkl_shape_out_backprop.IsMklTensor();
      if (out_backprop_in_mkl_format) {
        mkl_lt_out_backprop =
            static_cast<dnnLayout_t>(mkl_shape_out_backprop.GetCurLayout());
      } else {
        CHECK_EQ(
            dnnLayoutCreate_F32(&mkl_lt_out_backprop, mkl_params.in_dims,
                                mkl_params.in_sizes, mkl_params.in_strides),
            E_SUCCESS);
      }
    }

    void MklPrepareContextInputs(OpKernelContext* context,
                                 Tensor* mkl_tmp_input_buf_tensor,
                                 Tensor* mkl_tmp_outbackprop_buf_tensor,
                                 Tensor* mkl_tmp_scaleshift_buf_tensor) {
      bool mkl_convert_input;
      dnnPrimitive_t mkl_prim_convert_input = nullptr;
      dnnLayout_t mkl_lt_internal_input = nullptr;
      void* mkl_buf_converted_input = nullptr;
      // Compare with internal layouts and convert if needed
      const Tensor& input = MklGetInput(context, 1);
      void* mkl_buf_input =
          const_cast<void*>(static_cast<const void*>(input.flat<T>().data()));
      CHECK_EQ(
          dnnLayoutCreateFromPrimitive_F32(
              &mkl_lt_internal_input, mkl_prim_batchnorm_bwd, dnnResourceSrc),
          E_SUCCESS);
      mkl_convert_input =
          !dnnLayoutCompare_F32(mkl_lt_internal_input, mkl_lt_input);
      if (mkl_convert_input) {
        CHECK_EQ(dnnConversionCreate_F32(&mkl_prim_convert_input, mkl_lt_input,
                                         mkl_lt_internal_input),
                 E_SUCCESS);
        AllocTmpBuffer(context, mkl_tmp_input_buf_tensor, mkl_lt_internal_input,
                       &mkl_buf_converted_input);
        CHECK_EQ(dnnConversionExecute_F32(mkl_prim_convert_input, mkl_buf_input,
                                          mkl_buf_converted_input),
                 E_SUCCESS);
        dnnDelete_F32(mkl_prim_convert_input);
      }
      dnnLayoutDelete_F32(mkl_lt_internal_input);
      mkl_res_batchnorm_bwd[dnnResourceSrc] =
          (mkl_convert_input) ? mkl_buf_converted_input : mkl_buf_input;

      bool mkl_convert_out_backprop;
      dnnPrimitive_t mkl_prim_convert_out_backprop = nullptr;
      dnnLayout_t mkl_lt_internal_out_backprop = nullptr;
      void* mkl_buf_converted_out_backprop = nullptr;
      // Compare with internal layouts and convert if needed
      const Tensor& out_backprop = MklGetInput(context, 0);
      void* mkl_buf_out_backprop = const_cast<void*>(
          static_cast<const void*>(out_backprop.flat<T>().data()));
      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&mkl_lt_internal_out_backprop,
                                                mkl_prim_batchnorm_bwd,
                                                dnnResourceDiffDst),
               E_SUCCESS);
      mkl_convert_out_backprop = !dnnLayoutCompare_F32(
          mkl_lt_internal_out_backprop, mkl_lt_out_backprop);
      if (mkl_convert_out_backprop) {
        CHECK_EQ(dnnConversionCreate_F32(&mkl_prim_convert_out_backprop,
                                         mkl_lt_out_backprop,
                                         mkl_lt_internal_out_backprop),
                 E_SUCCESS);
        AllocTmpBuffer(context, mkl_tmp_outbackprop_buf_tensor,
                       mkl_lt_internal_out_backprop,
                       &mkl_buf_converted_out_backprop);
        CHECK_EQ(dnnConversionExecute_F32(mkl_prim_convert_out_backprop,
                                          mkl_buf_out_backprop,
                                          mkl_buf_converted_out_backprop),
                 E_SUCCESS);
        dnnDelete_F32(mkl_prim_convert_out_backprop);
      }
      dnnLayoutDelete_F32(mkl_lt_internal_out_backprop);
      mkl_res_batchnorm_bwd[dnnResourceDiffDst] =
          (mkl_convert_out_backprop) ? mkl_buf_converted_out_backprop
                                     : mkl_buf_out_backprop;

      // Set dnnResourceMean and dnnResourceVariance
      const Tensor& saved_mean = MklGetInput(context, 3);
      const Tensor& saved_var = MklGetInput(context, 4);
      void* mkl_buf_saved_mean = const_cast<void*>(
          static_cast<const void*>(saved_mean.flat<T>().data()));
      void* mkl_buf_saved_var = const_cast<void*>(
          static_cast<const void*>(saved_var.flat<T>().data()));
      mkl_res_batchnorm_bwd[dnnResourceMean] = mkl_buf_saved_mean;
      mkl_res_batchnorm_bwd[dnnResourceVariance] = mkl_buf_saved_var;

      // Set dnnResourceScaleShift
      // Note backward Op needs only current values of scale parameters,
      // shift parameters could be garbage and won't be used
      const Tensor& scale = MklGetInput(context, 2);
      dnnLayout_t mkl_lt_scale_shift = nullptr;
      void* mkl_buf_scale_shift = nullptr;
      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&mkl_lt_scale_shift,
                                                mkl_prim_batchnorm_bwd,
                                                dnnResourceScaleShift),
               E_SUCCESS);
      AllocTmpBuffer(context, mkl_tmp_scaleshift_buf_tensor, mkl_lt_scale_shift,
                     &mkl_buf_scale_shift);
      float* pscale =
          const_cast<float*>(static_cast<const float*>(scale.flat<T>().data()));
      float* pscale_shift = static_cast<float*>(mkl_buf_scale_shift);
      auto depth = mkl_params.depth;
      for (int i = 0; i < depth; i++) pscale_shift[i] = pscale[i];
      mkl_res_batchnorm_bwd[dnnResourceScaleShift] = mkl_buf_scale_shift;
      dnnLayoutDelete_F32(mkl_lt_scale_shift);
    }

    void MklPrepareGradScaleShift(OpKernelContext* context,
                                  Tensor* mkl_tmp_grad_scale_shift_buf_tensor) {
      dnnLayout_t mkl_lt_grad_scaleshift = nullptr;
      void* mkl_buf_grad_scaleshift = nullptr;
      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&mkl_lt_grad_scaleshift,
                                                mkl_prim_batchnorm_bwd,
                                                dnnResourceDiffScaleShift),
               E_SUCCESS);
      AllocTmpBuffer(context, mkl_tmp_grad_scale_shift_buf_tensor,
                     mkl_lt_grad_scaleshift, &mkl_buf_grad_scaleshift);
      mkl_res_batchnorm_bwd[dnnResourceDiffScaleShift] =
          mkl_buf_grad_scaleshift;
      dnnLayoutDelete_F32(mkl_lt_grad_scaleshift);
    }
  } MklFusedBatchNormGradOpContext;
};
#endif

#ifndef INTEL_MKL_ML

struct MklBatchNormFwdParams {
  memory::dims src_dims;
  int depth;
  float eps;
  bool training;

  MklBatchNormFwdParams(const memory::dims &src_dims,
      int depth, float eps, bool training) : src_dims(src_dims),
      depth(depth), eps(eps), training(training) {
  }
};

template <typename T>
class MklFusedBatchNormFwdPrimitive : public MklPrimitive {
 public:
  explicit MklFusedBatchNormFwdPrimitive(
      const MklBatchNormFwdParams& fwdParams) :
           cpu_engine_(engine::cpu, 0) {
    context_.fwd_stream.reset(
        new mkldnn::stream(mkldnn::stream::kind::eager));
    if (context_.bn_fwd == nullptr)
        Setup(fwdParams);
  }

  ~MklFusedBatchNormFwdPrimitive() {}

  // BatchNormalization forward execute
  //   src_data:     input data buffer of src
  //   weights_data: input data buffer of weights
  //   dst_data:     output data buffer of dst
  //   mean_data:     output data buffer of means
  //   variance_data: output data buffer of variances
  void Execute(const T* src_data, const T* weights_data, T* dst_data,
      T* mean_data, T* variance_data) {
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_data)));
    context_.dst_mem->set_data_handle(
        static_cast<void*>(dst_data));

    if (context_.flags & use_scale_shift)
      context_.weights_mem->set_data_handle(
          static_cast<void*>(const_cast<T*>(weights_data)));

    if ((context_.pkind == prop_kind::forward_training) ||
      (context_.flags & use_global_stats)) {
      context_.mean_mem->set_data_handle(
          static_cast<void*>(mean_data));
      context_.variance_mem->set_data_handle(
          static_cast<void*>(variance_data));
    }

    // execution
    context_.fwd_stream->submit(context_.fwd_primitives);

    context_.src_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);

    if (context_.flags & use_scale_shift)
      context_.weights_mem->set_data_handle(DummyData);

    if ((context_.pkind == prop_kind::forward_training) ||
      (context_.flags & use_global_stats)) {
      context_.mean_mem->set_data_handle(DummyData);
      context_.variance_mem->set_data_handle(DummyData);
    }
  }

  memory::primitive_desc GetDstPd() const {
    return (*context_.dst_mem).get_primitive_desc();
  }

  mkldnn_memory_format_t GetSrcFmt() const {
    return (*context_.src_mem).get_primitive_desc().desc().data.format;
  }

  mkldnn_memory_format_t GetDstFmt() const {
    return (*context_.dst_mem).get_primitive_desc().desc().data.format;
  }

 private:
  // Primitive reuse context for BatchNorm fwd op
  struct BatchNormFwdContext {
    // flags indict if it is training or inference mode
    int64 flags;

    // algorithm
    mkldnn::prop_kind pkind;

    // Mkldnn Memory
    std::shared_ptr<mkldnn::memory> src_mem;
    std::shared_ptr<mkldnn::memory> weights_mem;
    std::shared_ptr<mkldnn::memory> dst_mem;
    std::shared_ptr<mkldnn::memory> mean_mem;
    std::shared_ptr<mkldnn::memory> variance_mem;

    // BatchNorm forward primitive
    std::shared_ptr<mkldnn::primitive> bn_fwd;
    std::shared_ptr<mkldnn::stream> fwd_stream;
    std::vector<mkldnn::primitive> fwd_primitives;

    BatchNormFwdContext() :
      flags(0), pkind(mkldnn::forward_training), src_mem(nullptr),
      weights_mem(nullptr), dst_mem(nullptr),  mean_mem(nullptr),
      variance_mem(nullptr), bn_fwd(nullptr), fwd_stream(nullptr) {
    }
  };

  void Setup(const MklBatchNormFwdParams& fwdParams) {
    context_.flags = fwdParams.training ? use_scale_shift
                 : (use_scale_shift | use_global_stats);
    context_.pkind = fwdParams.training ? prop_kind::forward_training
                           : prop_kind::forward_scoring;

    // memory desc
    auto src_md = memory::desc({fwdParams.src_dims},
        MklDnnType<T>(), get_desired_format(fwdParams.src_dims[1]));

    // fwd desc & primitive desc
    auto fwd_desc = batch_normalization_forward::desc(
        context_.pkind, src_md, fwdParams.eps, context_.flags);
    auto fwd_pd = batch_normalization_forward::primitive_desc(
        fwd_desc, cpu_engine_);

    // memory primitive
    context_.src_mem.reset(new memory({src_md, cpu_engine_}, DummyData));
    context_.dst_mem.reset(new memory(fwd_pd.dst_primitive_desc(), DummyData));

    if (context_.flags & use_scale_shift) {
      auto weights_desc = memory::desc({2, fwdParams.depth},
          MklDnnType<T>(), memory::format::nc);
      context_.weights_mem.reset(new memory({weights_desc, cpu_engine_},
          DummyData));
    }

    if (fwdParams.training || (context_.flags & use_global_stats)) {
      auto mean_desc = memory::desc({1, fwdParams.depth},
          MklDnnType<T>(), memory::format::nc);
      context_.mean_mem.reset(new memory({mean_desc, cpu_engine_}, DummyData));

      auto variance_desc = memory::desc({1, fwdParams.depth},
          MklDnnType<T>(), memory::nc);
      context_.variance_mem.reset(new memory({variance_desc, cpu_engine_},
          DummyData));
    }

    // BatchNorm forward primitive
    if (!fwdParams.training && !(context_.flags & use_global_stats)) {
      if ((context_.flags & use_scale_shift) && mkldnn_use_scaleshift) {
        context_.bn_fwd.reset(new batch_normalization_forward(fwd_pd,
            *context_.src_mem, *context_.weights_mem, *context_.dst_mem));
      } else {
        context_.bn_fwd.reset(new batch_normalization_forward(
            fwd_pd, *context_.src_mem, *context_.dst_mem));
      }
    } else if (context_.flags & use_global_stats) {
      if ((context_.flags & use_scale_shift) && mkldnn_use_scaleshift) {
        context_.bn_fwd.reset(new batch_normalization_forward(
            fwd_pd, *context_.src_mem, (const primitive::at)*context_.mean_mem,
            (const primitive::at)*context_.variance_mem, *context_.weights_mem,
            *context_.dst_mem));
      } else {
        context_.bn_fwd.reset(new batch_normalization_forward(
            fwd_pd, *context_.src_mem, (const primitive::at)*context_.mean_mem,
            (const primitive::at)*context_.variance_mem, *context_.dst_mem));
      }
    } else {
      if ((context_.flags & use_scale_shift) && mkldnn_use_scaleshift) {
        context_.bn_fwd.reset(new batch_normalization_forward(
            fwd_pd, *context_.src_mem, *context_.weights_mem, *context_.dst_mem,
            *context_.mean_mem, *context_.variance_mem));
      } else {
        context_.bn_fwd.reset(new batch_normalization_forward(
            fwd_pd, *context_.src_mem, *context_.dst_mem,
            *context_.mean_mem, *context_.variance_mem));
      }
    }

    context_.fwd_primitives.push_back(*context_.bn_fwd);
  }

  mkldnn::memory::desc get_desc_data(const mkldnn::memory &m) const {
    return m.get_primitive_desc().desc().data;
  }

  struct BatchNormFwdContext context_;
  engine cpu_engine_;
};

template <typename T>
class MklFusedBatchNormFwdPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklFusedBatchNormFwdPrimitive<T>* Get(
            const MklBatchNormFwdParams& fwdParams) {
    auto bn_fwd = static_cast<MklFusedBatchNormFwdPrimitive<T>*>(
        MklFusedBatchNormFwdPrimitiveFactory<T>
          ::GetInstance().GetBatchNormFwd(fwdParams));

    if (bn_fwd == nullptr) {
      bn_fwd = new MklFusedBatchNormFwdPrimitive<T>(fwdParams);
      MklFusedBatchNormFwdPrimitiveFactory<T>::GetInstance().SetBatchNormFwd(
          fwdParams, bn_fwd);
    }
    return bn_fwd;
  }

  static MklFusedBatchNormFwdPrimitiveFactory & GetInstance() {
    static MklFusedBatchNormFwdPrimitiveFactory instance_;
    return instance_;
  }

 private:
  MklFusedBatchNormFwdPrimitiveFactory() {}
  ~MklFusedBatchNormFwdPrimitiveFactory() {}

  static std::string CreateKey(const MklBatchNormFwdParams& fwdParams) {
    std::string prefix = "bn_fwd";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(fwdParams.src_dims);
    key_creator.AddAsKey<int>(fwdParams.depth);
    key_creator.AddAsKey<float>(fwdParams.eps);
    key_creator.AddAsKey<bool>(fwdParams.training);
    return key_creator.GetKey();
  }

  MklPrimitive* GetBatchNormFwd(const MklBatchNormFwdParams& fwdParams) {
    std::string key = CreateKey(fwdParams);
    return this->GetOp(key);
  }

  void SetBatchNormFwd(const MklBatchNormFwdParams& fwdParams,
                       MklPrimitive *op) {
    std::string key = CreateKey(fwdParams);
    this->SetOp(key, op);
  }
};

struct MklBatchNormBwdParams {
  memory::dims src_dims;
  memory::dims diff_dst_dims;
  int depth;
  float eps;
  bool training;

  MklBatchNormBwdParams(memory::dims src_dims, memory::dims diff_dst_dims,
      int depth, float eps, bool training) : src_dims(src_dims),
      diff_dst_dims(diff_dst_dims), depth(depth), eps(eps),
      training(training) {
  }
};


template <typename T>
class MklFusedBatchNormBwdPrimitive : public MklPrimitive {
 public:
  explicit MklFusedBatchNormBwdPrimitive(
      const MklBatchNormBwdParams& bwdParams) :
           cpu_engine_(engine::cpu, 0) {
    context_.bwd_stream.reset(
          new mkldnn::stream(mkldnn::stream::kind::eager));
    if (context_.bn_bwd == nullptr)
      Setup(bwdParams);
  }

  ~MklFusedBatchNormBwdPrimitive() {}

  // BatchNormalization backward execute
  //   src_data:       input data buffer of src
  //   mean_data:      input data buffer of mean
  //   variance_data:  input data buffer of variance
  //   diff_dst_data:  input data buffer of diff_dst
  //   weights_data:   input data buffer of weights
  //   diff_src_data:      output data buffer of diff_src
  //   diff_weights_data:  output data buffer of diff_weights
  void Execute(const T* src_data, const T* mean_data, const T* variance_data,
      const T* diff_dst_data, const T* weights_data,
      T* diff_src_data, T* diff_weights_data) {
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_data)));
    context_.mean_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(mean_data)));
    context_.variance_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(variance_data)));
    context_.diff_dst_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(diff_dst_data)));

    if (context_.flags & use_scale_shift) {
      context_.weights_mem->set_data_handle(
          static_cast<void*>(const_cast<T*>(weights_data)));
      context_.diff_weights_mem->set_data_handle(
          static_cast<void*>(diff_weights_data));
    }

    context_.diff_src_mem->set_data_handle(
        static_cast<void*>(diff_src_data));

    // execution
    context_.bwd_stream->submit(context_.bwd_primitives);

    context_.src_mem->set_data_handle(DummyData);
    context_.mean_mem->set_data_handle(DummyData);
    context_.variance_mem->set_data_handle(DummyData);
    context_.diff_dst_mem->set_data_handle(DummyData);
    if (context_.flags & use_scale_shift) {
      context_.weights_mem->set_data_handle(DummyData);
      context_.diff_weights_mem->set_data_handle(DummyData);
    }
    context_.diff_src_mem->set_data_handle(DummyData);
  }

  mkldnn_memory_format_t GetSrcFmt() {
    return(*context_.src_mem).get_primitive_desc().desc().data.format;
  }

  mkldnn_memory_format_t GetDiffDstFmt() {
    return(*context_.diff_dst_mem).get_primitive_desc().desc().data.format;
  }

  memory::primitive_desc GetDiffSrcPd() {
    return(*context_.diff_src_mem).get_primitive_desc();
  }

 private:
  struct BatchNormBwdContext {
    // Flags to indicate whether it is training or inference
    int64 flags;

    // MKLDNN memory
    std::shared_ptr<mkldnn::memory> src_mem;
    std::shared_ptr<mkldnn::memory> mean_mem;
    std::shared_ptr<mkldnn::memory> variance_mem;
    std::shared_ptr<mkldnn::memory> diff_dst_mem;
    std::shared_ptr<mkldnn::memory> weights_mem;
    std::shared_ptr<mkldnn::memory> diff_weights_mem;
    std::shared_ptr<mkldnn::memory> diff_src_mem;

    // Batch Norm primitive
    std::shared_ptr<mkldnn::primitive> bn_bwd;
    std::vector<mkldnn::primitive> bwd_primitives;
    std::shared_ptr<mkldnn::stream> bwd_stream;

    BatchNormBwdContext() :
      src_mem(nullptr), mean_mem(nullptr), variance_mem(nullptr),
      diff_dst_mem(nullptr), weights_mem(nullptr), diff_weights_mem(nullptr),
      diff_src_mem(nullptr), bwd_stream(nullptr) {
    }
  };

  void Setup(const MklBatchNormBwdParams& bwdParams) {
    context_.flags = bwdParams.training ? use_scale_shift
           : (use_scale_shift | use_global_stats);

    // memory desc
    auto src_md = memory::desc({bwdParams.src_dims},
        MklDnnType<T>(), get_desired_format(bwdParams.src_dims[1]));
    auto diff_dst_md = memory::desc({bwdParams.diff_dst_dims},
        MklDnnType<T>(), get_desired_format(bwdParams.diff_dst_dims[1]));
    auto variance_desc = memory::desc({1, bwdParams.depth}, MklDnnType<T>(),
         memory::nc);
    auto mean_desc = memory::desc({1, bwdParams.depth},
         MklDnnType<T>(), memory::format::nc);
    auto weights_desc = memory::desc({2, bwdParams.depth},
         MklDnnType<T>(), memory::format::nc);
    auto diff_weights_desc = weights_desc;

    // fwd desc & primitive desc
    auto fwd_desc = batch_normalization_forward::desc(
        prop_kind::forward_training, src_md, bwdParams.eps,
        bwdParams.training
        ? use_scale_shift
        : (use_scale_shift | use_global_stats));
    auto fwd_pd = batch_normalization_forward::primitive_desc(
        fwd_desc, cpu_engine_);

    // BatchNorm backward primtive
    //
    // For inference, specify use_global_stats
    //   1. on fwd propagation, use mean and variance provided as inputs.
    //   2. on bwd propagation, mean and variance are considered as constants.
    //      Thus, reduce the amount of MKL computation.
    auto bwd_desc = batch_normalization_backward::desc(
        prop_kind::backward, diff_dst_md, src_md, bwdParams.eps,
        bwdParams.training ? use_scale_shift
                 : (use_scale_shift | use_global_stats));
    auto bn_bwd_pd = batch_normalization_backward::primitive_desc(
        bwd_desc, cpu_engine_, fwd_pd);

    // memory primitive
    context_.src_mem.reset(new memory({src_md, cpu_engine_}, DummyData));
    context_.diff_dst_mem.reset(new memory({diff_dst_md, cpu_engine_},
        DummyData));
    context_.variance_mem.reset(new memory({variance_desc, cpu_engine_},
        DummyData));
    context_.mean_mem.reset(new memory({mean_desc, cpu_engine_}, DummyData));
    context_.weights_mem.reset(new memory({weights_desc, cpu_engine_},
        DummyData));
    context_.diff_weights_mem.reset(new memory({diff_weights_desc, cpu_engine_},
          DummyData));
    context_.diff_src_mem.reset(new memory({src_md, cpu_engine_}, DummyData));

    context_.bn_bwd.reset(new batch_normalization_backward(
          bn_bwd_pd, *context_.src_mem, *context_.mean_mem,
          *context_.variance_mem, *context_.diff_dst_mem, *context_.weights_mem,
          *context_.diff_src_mem, *context_.diff_weights_mem));
    context_.bwd_primitives.push_back(*context_.bn_bwd);
  }

  struct BatchNormBwdContext context_;
  engine cpu_engine_;
};

template <typename T>
class MklFusedBatchNormBwdPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklFusedBatchNormBwdPrimitive<T>* Get(
            const MklBatchNormBwdParams& bwdParams) {
    auto bn_bwd = static_cast<MklFusedBatchNormBwdPrimitive<T>*>(
        MklFusedBatchNormBwdPrimitiveFactory<T>
          ::GetInstance().GetBatchNormBwd(bwdParams));
    if (bn_bwd == nullptr) {
      bn_bwd = new MklFusedBatchNormBwdPrimitive<T>(bwdParams);
      MklFusedBatchNormBwdPrimitiveFactory<T>::GetInstance().SetBatchNormBwd(
           bwdParams, bn_bwd);
    }
    return bn_bwd;
  }

  static MklFusedBatchNormBwdPrimitiveFactory& GetInstance() {
    static MklFusedBatchNormBwdPrimitiveFactory instance_;
    return instance_;
  }

 private:
  MklFusedBatchNormBwdPrimitiveFactory() {}
  ~MklFusedBatchNormBwdPrimitiveFactory() {}

  static std::string CreateKey(const MklBatchNormBwdParams& bwdParams) {
    std::string prefix = "bn_bwd";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(bwdParams.src_dims);
    key_creator.AddAsKey(bwdParams.diff_dst_dims);
    key_creator.AddAsKey<int>(bwdParams.depth);
    key_creator.AddAsKey<float>(bwdParams.eps);
    key_creator.AddAsKey<bool>(bwdParams.training);
    return key_creator.GetKey();
  }

  MklPrimitive* GetBatchNormBwd(const MklBatchNormBwdParams& bwdParams) {
    std::string key = CreateKey(bwdParams);
    return this->GetOp(key);
  }

  void SetBatchNormBwd(const MklBatchNormBwdParams& bwdParams,
                       MklPrimitive* op) {
    std::string key = CreateKey(bwdParams);
    this->SetOp(key, op);
  }
};

template <typename Device, typename T>
class MklFusedBatchNormOp : public OpKernel {
 public:
  explicit MklFusedBatchNormOp(OpKernelConstruction* context)
      : OpKernel(context) {
    float epsilon;
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
    epsilon_ = T(epsilon);
    string tensor_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &tensor_format));
    OP_REQUIRES(context, FormatFromString(tensor_format, &tensor_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));
  }

  void Compute(OpKernelContext* context) override {
    try {
      const size_t kSrcIndex = 0;       // index of src input tensor
      const size_t kScaleIndex = 1;     // index of scale tensor
      const size_t kShiftIndex = 2;     // index of shift tensor
      const size_t kMeanIndex = 3;      // index of est_mean tensor
      const size_t kVarianceIndex = 4;  // index of est_variance tensor

      const Tensor& src_tensor = MklGetInput(context, kSrcIndex);
      const Tensor& scale_tensor = MklGetInput(context, kScaleIndex);
      const Tensor& shift_tensor = MklGetInput(context, kShiftIndex);
      const Tensor& est_mean_tensor = MklGetInput(context, kMeanIndex);
      const Tensor& est_variance_tensor = MklGetInput(context, kVarianceIndex);

      TensorShape tf_shape_src;
      MklDnnShape dnn_shape_src;
      GetMklShape(context, kSrcIndex, &dnn_shape_src);

      if (dnn_shape_src.IsMklTensor()) {
        tf_shape_src = dnn_shape_src.GetTfShape();
        OP_REQUIRES(context, dnn_shape_src.GetDimension() == 4,
                    errors::InvalidArgument("input must be 4-dimensional",
                                            src_tensor.shape().DebugString()));
      } else {
        tf_shape_src = src_tensor.shape();
        OP_REQUIRES(context, src_tensor.dims() == 4,
                    errors::InvalidArgument("input must be 4-dimensional",
                                            src_tensor.shape().DebugString()));
      }
      OP_REQUIRES(context, scale_tensor.dims() == 1,
                  errors::InvalidArgument("scale must be 1-dimensional",
                                          scale_tensor.shape().DebugString()));
      OP_REQUIRES(context, shift_tensor.dims() == 1,
                  errors::InvalidArgument("offset must be 1-dimensional",
                                          shift_tensor.shape().DebugString()));
      OP_REQUIRES(
          context, est_mean_tensor.dims() == 1,
          errors::InvalidArgument("estimated_mean must be 1-dimensional",
                                  est_mean_tensor.shape().DebugString()));
      OP_REQUIRES(
          context, est_variance_tensor.dims() == 1,
          errors::InvalidArgument("estimated_variance must be 1-dimensional",
                                  est_variance_tensor.shape().DebugString()));

      if (is_training_) {
        OP_REQUIRES(
            context, est_mean_tensor.dim_size(0) == 0,
            errors::InvalidArgument("estimated_mean must be empty for training",
                                    est_mean_tensor.shape().DebugString()));
        OP_REQUIRES(context, est_variance_tensor.dim_size(0) == 0,
                    errors::InvalidArgument(
                        "estimated_variance must be empty for training",
                        est_variance_tensor.shape().DebugString()));
      }

      // special case: input with 0 element and 0 batch size
      Tensor* dst_tensor = nullptr;
      if (tf_shape_src.num_elements() == 0) {
        HandleEmptyInput(context, tf_shape_src, scale_tensor.shape(),
                         &dst_tensor);
        return;
      }

      if (dnn_shape_src.IsMklTensor())
        depth_ = dnn_shape_src.DimSize(MklDnnDims::Dim_C);
      else
        ExtractParams(context);

      // Indices of output tensors
      const size_t kDstIndex = 0;

      // allocate 4 output TF tensors
      Tensor* batch_mean_tensor = nullptr;
      Tensor* batch_variance_tensor = nullptr;
      Tensor* saved_mean_tensor = nullptr;
      Tensor* saved_variance_tensor = nullptr;
      AllocateTFOutputs(context, scale_tensor.shape(), &batch_mean_tensor,
                        &batch_variance_tensor, &saved_mean_tensor,
                        &saved_variance_tensor);

      if (is_training_)
        SetMeanVariance(*batch_mean_tensor, *batch_variance_tensor);
      else
        SetMeanVariance(est_mean_tensor, est_variance_tensor);

      MklDnnData<T> src(&cpu_engine);
      MklDnnData<T> weights(&cpu_engine);

      memory::format format_m;
      if (dnn_shape_src.IsMklTensor()) {
        if (dnn_shape_src.IsTensorInNCHWFormat()) {
          format_m = memory::format::nchw;
        } else {
          format_m = memory::format::nhwc;
        }
      } else {
        format_m = TFDataFormatToMklDnnDataFormat(tensor_format_);
      }

      // set src primitive
      memory::dims src_dims = dnn_shape_src.IsMklTensor()
            ? dnn_shape_src.GetSizesAsMklDnnDims()
            : TFShapeToMklDnnDimsInNCHW(src_tensor.shape(), tensor_format_);

      auto src_md = dnn_shape_src.IsMklTensor()
                        ? dnn_shape_src.GetMklLayout()
                        : memory::desc(src_dims, MklDnnType<T>(), format_m);

      // MKL-DNN packs scale & shift as "weights":
      // <scale>...<scale><shift>...<shift>
      weights.AllocateBuffer(2 * depth_ * sizeof (T));
      T* weights_data =  reinterpret_cast<T*>(weights.GetAllocatedBuffer());
      const T* scale_tf = scale_tensor.flat<T>().data();
      const T* shift_tf = shift_tensor.flat<T>().data();

      std::memcpy(weights_data, scale_tf, depth_ * sizeof(T));
      std::memcpy(weights_data + depth_, shift_tf, depth_ * sizeof(T));
      char* saved_mean_data_tf =
          reinterpret_cast<char*>(saved_mean_tensor->flat<T>().data());
      std::memcpy(saved_mean_data_tf, reinterpret_cast<char*>(mean_values_),
                  depth_ * sizeof(T));

      char* saved_variance_data_tf =
          reinterpret_cast<char*>(saved_variance_tensor->flat<T>().data());
      std::memcpy(saved_variance_data_tf,
                  reinterpret_cast<char*>(variance_values_),
                  depth_ * sizeof(T));

      // get batchnorm op from the pool
      MklBatchNormFwdParams fwdParams(src_dims, depth_, epsilon_, is_training_);
      MklFusedBatchNormFwdPrimitive<T> *bn_fwd =
          MklFusedBatchNormFwdPrimitiveFactory<T>::Get(fwdParams);

      // check if reorder is needed for src, weights, mean, variance
      const T* src_data = src_tensor.flat<T>().data();
      if (src_md.data.format != bn_fwd->GetSrcFmt()) {
         src.SetUsrMem(src_md, &src_tensor);
         auto src_target = memory::primitive_desc({{src_dims}, MklDnnType<T>(),
              static_cast<memory::format>(bn_fwd->GetSrcFmt())}, cpu_engine);
         src.CheckReorderToOpMem(src_target);
         src_data = const_cast<T*>(
             reinterpret_cast<T*>(src.GetOpMem().get_data_handle()));
      }

      // allocate output (dst) tensor; always set it as MKL-DNN layout
      MklDnnShape dnn_shape_dst;
      TensorShape tf_shape_dst;
      dnn_shape_dst.SetMklTensor(true);
      auto dst_pd = bn_fwd->GetDstPd();
      dnn_shape_dst.SetMklLayout(&dst_pd);
      dnn_shape_dst.SetElemType(MklDnnType<T>());
      auto ndims = dnn_shape_src.IsMklTensor()
                 ? dnn_shape_src.GetDimension()
                 : src_tensor.shape().dims();
      dnn_shape_dst.SetTfLayout(ndims, src_dims, format_m);
      tf_shape_dst.AddDim(dst_pd.get_size() / sizeof(T));
      AllocateOutputSetMklShape(context, kDstIndex, &dst_tensor,
                                tf_shape_dst, dnn_shape_dst);

      T* weights_op_data = weights_data;
      T* mean_op_data = saved_mean_tensor->flat<T>().data();
      T* variance_op_data = saved_variance_tensor->flat<T>().data();
      T* dst_data = dst_tensor->flat<T>().data();

      // execution
      bn_fwd->Execute(src_data, weights_op_data, dst_data,
                      mean_op_data, variance_op_data);

      // copy batch_mean data
      T* batch_mean_data_tf = batch_mean_tensor->flat<T>().data();
      std::memcpy(reinterpret_cast<char*>(batch_mean_data_tf),
                  reinterpret_cast<char*>(saved_mean_data_tf),
                  depth_ * sizeof(T));
      // TODO(yli135): OpMem is same as usr mem since
      // since its format is hard-coded as nc when primitive is created.

      // copy batch_variance data with Bessel's correction
      float adjust_factor = 1.0;
      if (is_training_) {
        size_t orig_size = src_dims[0] * src_dims[2] * src_dims[3];
        size_t adjust_size = orig_size - 1;
        adjust_factor = (static_cast<float>(orig_size)) / adjust_size;
      }

      auto variance_data = reinterpret_cast<T*>(saved_variance_data_tf);
      auto batch_variance_data = batch_variance_tensor->flat<T>().data();
      if (is_training_) {
        for (int k = 0; k < depth_; k++) {
            batch_variance_data[k] = variance_data[k] * adjust_factor;
        }
      } else {
        std::memcpy(batch_variance_data, variance_data, depth_ * sizeof(T));
      }
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 private:
  T epsilon_;
  TensorFormat tensor_format_;
  bool is_training_;
  T* mean_values_;
  T* variance_values_;
  size_t depth_;          // batch normalization is done for per channel.
  engine cpu_engine = engine(engine::cpu, 0);

  void ExtractParams(OpKernelContext* context) {
    const Tensor& input = MklGetInput(context, 0);
    depth_ = static_cast<int>(GetTensorDim(input, tensor_format_, 'C'));
  }

  void SetMeanVariance(const Tensor& mean, const Tensor& variance) {
    mean_values_ = reinterpret_cast<T*>(const_cast<T*>(mean.flat<T>().data()));
    variance_values_ =
        reinterpret_cast<T*>(const_cast<T*>(variance.flat<T>().data()));
  }

  void HandleEmptyInput(OpKernelContext* context, TensorShape tf_shape_src,
                        TensorShape tf_shape_scale, Tensor** dst_tensor) {
    CHECK_NOTNULL(dst_tensor);

    const size_t kDstIndex = 0;
    MklDnnShape dnn_shape_dst;
    dnn_shape_dst.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kDstIndex, dst_tensor, tf_shape_src,
                              dnn_shape_dst);
    CHECK_NOTNULL(*dst_tensor);
    memset(const_cast<char*>((*dst_tensor)->tensor_data().data()), 0,
           (*dst_tensor)->tensor_data().size());

    Tensor* batch_mean_tensor = nullptr;
    Tensor* batch_variance_tensor = nullptr;
    Tensor* saved_mean_tensor = nullptr;
    Tensor* saved_variance_tensor = nullptr;
    AllocateTFOutputs(context, tf_shape_scale, &batch_mean_tensor,
                      &batch_variance_tensor, &saved_mean_tensor,
                      &saved_variance_tensor);
  }

  void AllocateTFOutputs(OpKernelContext* context, TensorShape tf_shape_scale,
                         Tensor** batch_mean_tensor,
                         Tensor** batch_variance_tensor,
                         Tensor** saved_mean_tensor,
                         Tensor** saved_variance_tensor) {
    CHECK_NOTNULL(batch_mean_tensor);
    CHECK_NOTNULL(batch_variance_tensor);
    CHECK_NOTNULL(saved_mean_tensor);
    CHECK_NOTNULL(saved_variance_tensor);

    const size_t kBatchMeanIndex = 1;
    const size_t kBatchVarianceIndex = 2;
    const size_t kSavedMeanIndex = 3;
    const size_t kSavedVarianceIndex = 4;

    // allocate batch mean output tensor
    MklDnnShape mkl_shape_batch_mean;
    mkl_shape_batch_mean.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kBatchMeanIndex, batch_mean_tensor,
                              tf_shape_scale, mkl_shape_batch_mean);
    CHECK_NOTNULL(*batch_mean_tensor);
    // set NAN mean value in case of empty input tensor
    int num_elements = tf_shape_scale.num_elements();
    auto batch_mean_data =  (*batch_mean_tensor)->flat<T>().data();
    for (int k = 0; k < num_elements; k++)
      batch_mean_data[k] = NAN;

    // allocate batch variance output tensor
    MklDnnShape mkl_shape_batch_variance;
    mkl_shape_batch_variance.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kBatchVarianceIndex,
                              batch_variance_tensor, tf_shape_scale,
                              mkl_shape_batch_variance);
    CHECK_NOTNULL(*batch_variance_tensor);
    // set NAN variance value in case of empty input tensor
    auto batch_variance_data = (*batch_variance_tensor)->flat<T>().data();
    for (int k = 0; k < num_elements; k++)
        batch_variance_data[k] = NAN;

    // Mean and variance (without Bessel's correction) saved for backward
    // computation to serve as pre-computed mean and variance.
    MklDnnShape mkl_shape_saved_mean;
    mkl_shape_saved_mean.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kSavedMeanIndex, saved_mean_tensor,
                              tf_shape_scale, mkl_shape_saved_mean);
    CHECK_NOTNULL(*saved_mean_tensor);
    // set NAN mean value in case of empty input tensor
    auto saved_mean_data = (*saved_mean_tensor)->flat<T>().data();
    for (int k = 0; k < num_elements; k++)
        saved_mean_data[k] = NAN;

    MklDnnShape mkl_shape_saved_variance;
    mkl_shape_saved_variance.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kSavedVarianceIndex,
                              saved_variance_tensor, tf_shape_scale,
                              mkl_shape_saved_variance);
    CHECK_NOTNULL(*saved_variance_tensor);
    // set NAN variance value in case of empty input tensor
    auto saved_variance_data = (*saved_variance_tensor)->flat<T>().data();
    for (int k = 0; k < num_elements; k++)
      saved_variance_data[k] = NAN;
  }
};

template <typename Device, typename T>
class MklFusedBatchNormGradOp : public OpKernel {
 public:
  explicit MklFusedBatchNormGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    float epsilon;
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
    epsilon_ = T(epsilon);
    string tensor_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &tensor_format));
    OP_REQUIRES(context, FormatFromString(tensor_format, &tensor_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));
  }

  void Compute(OpKernelContext* context) override {
    try {
      const size_t kDiffDstIndex = 0;   // index of diff_dst tensor
      const size_t kSrcIndex = 1;       // index of src input tensor
      const size_t kScaleIndex = 2;     // index of scale tensor
      const size_t kMeanIndex = 3;      // index of saved_mean tensor
      const size_t kVarianceIndex = 4;  // index of saved_variance tensor

      const Tensor& diff_dst_tensor = MklGetInput(context, kDiffDstIndex);
      const Tensor& src_tensor = MklGetInput(context, kSrcIndex);
      const Tensor& scale_tensor = MklGetInput(context, kScaleIndex);
      const Tensor& saved_mean_tensor = MklGetInput(context, kMeanIndex);
      const Tensor& saved_variance_tensor =
          MklGetInput(context, kVarianceIndex);

      MklDnnShape dnn_shape_src, dnn_shape_diff_dst;
      GetMklShape(context, kSrcIndex, &dnn_shape_src);
      GetMklShape(context, kDiffDstIndex, &dnn_shape_diff_dst);

      TensorShape tf_shape_src, tf_shape_diff_dst;
      if (dnn_shape_diff_dst.IsMklTensor()) {
        tf_shape_diff_dst = dnn_shape_diff_dst.GetTfShape();
        OP_REQUIRES(
            context, dnn_shape_diff_dst.GetDimension() == 4,
            errors::InvalidArgument("input must be 4-dimensional",
                                    diff_dst_tensor.shape().DebugString()));
      } else {
        tf_shape_diff_dst = diff_dst_tensor.shape();
        OP_REQUIRES(
            context, diff_dst_tensor.dims() == 4,
            errors::InvalidArgument("input must be 4-dimensional",
                                    diff_dst_tensor.shape().DebugString()));
      }

      if (dnn_shape_src.IsMklTensor()) {
        tf_shape_src = dnn_shape_src.GetTfShape();
        OP_REQUIRES(context, dnn_shape_src.GetDimension() == 4,
                    errors::InvalidArgument("input must be 4-dimensional",
                                            src_tensor.shape().DebugString()));
      } else {
        tf_shape_src = src_tensor.shape();
        OP_REQUIRES(context, src_tensor.dims() == 4,
                    errors::InvalidArgument("input must be 4-dimensional",
                                            src_tensor.shape().DebugString()));
      }

      OP_REQUIRES(context, scale_tensor.dims() == 1,
                  errors::InvalidArgument("scale must be 1-dimensional",
                                          scale_tensor.shape().DebugString()));
      OP_REQUIRES(
          context, saved_mean_tensor.dims() == 1,
          errors::InvalidArgument("saved mean must be 1-dimensional",
                                  saved_mean_tensor.shape().DebugString()));

      OP_REQUIRES(
          context, saved_variance_tensor.dims() == 1,
          errors::InvalidArgument("saved variance must be 1-dimensional",
                                  saved_variance_tensor.shape().DebugString()));

      Tensor* diff_src_tensor = nullptr;
      // special case: input with 0 element and 0 batch size
      if (tf_shape_src.num_elements() == 0 ||
          tf_shape_diff_dst.num_elements() == 0) {
        HandleEmptyInput(context, tf_shape_src, scale_tensor.shape(),
                         &diff_src_tensor);
        return;
      }

      if (dnn_shape_src.IsMklTensor()) {
        depth_ = dnn_shape_src.DimSize(MklDnnDims::Dim_C);
      } else if (dnn_shape_diff_dst.IsMklTensor()) {
        depth_ = dnn_shape_diff_dst.DimSize(MklDnnDims::Dim_C);
      } else {
        ExtractParams(context);
      }

      memory::format format_m;
      if (dnn_shape_src.IsMklTensor()) {
        if (dnn_shape_src.IsTensorInNCHWFormat())
          format_m = memory::format::nchw;
        else
          format_m = memory::format::nhwc;
      } else {
        format_m = TFDataFormatToMklDnnDataFormat(tensor_format_);
      }

      MklDnnData<T> src(&cpu_engine);
      MklDnnData<T> diff_dst(&cpu_engine);
      MklDnnData<T> weights(&cpu_engine);
      MklDnnData<T> diff_weights(&cpu_engine);

      memory::dims src_dims = dnn_shape_src.IsMklTensor()
          ? dnn_shape_src.GetSizesAsMklDnnDims()
          : TFShapeToMklDnnDimsInNCHW(src_tensor.shape(), tensor_format_);
      memory::dims diff_dst_dims = dnn_shape_diff_dst.IsMklTensor()
          ? dnn_shape_diff_dst.GetSizesAsMklDnnDims()
          : TFShapeToMklDnnDimsInNCHW(diff_dst_tensor.shape(), tensor_format_);

      // set src and diff_dst primitive descriptors
      memory::desc src_md = dnn_shape_src.IsMklTensor()
          ? dnn_shape_src.GetMklLayout()
          : memory::desc(src_dims, MklDnnType<T>(), format_m);
      memory::desc diff_dst_md = dnn_shape_diff_dst.IsMklTensor()
          ? dnn_shape_diff_dst.GetMklLayout()
          : memory::desc(diff_dst_dims, MklDnnType<T>(), format_m);

      // weights -- MKL DNN packs scales/ shifts as weights in order
      // of scale, ..., scale, shift, ...., shift
      weights.AllocateBuffer(2 * depth_ * sizeof(T));
      T* weights_data_tf = reinterpret_cast<T*>(weights.GetAllocatedBuffer());
      const T* scale_tf = scale_tensor.flat<T>().data();
      for (int k = 0; k < depth_; k++) {
        weights_data_tf[k] = scale_tf[k];
        weights_data_tf[k + depth_] = 0;
      }

      diff_weights.AllocateBuffer(2 * depth_ * sizeof(T));

      MklBatchNormBwdParams bwdParams(src_dims, diff_dst_dims,
          depth_, epsilon_, is_training_);
      MklFusedBatchNormBwdPrimitive<T> *bn_bwd =
          MklFusedBatchNormBwdPrimitiveFactory<T>::Get(bwdParams);

      // check if src/diff_dst need to be reordered
      const T* src_data = src_tensor.flat<T>().data();
      if (src_md.data.format != bn_bwd->GetSrcFmt()) {
        src.SetUsrMem(src_md, &src_tensor);
        auto src_target = memory::primitive_desc({{src_dims}, MklDnnType<T>(),
            static_cast<memory::format>(bn_bwd->GetSrcFmt())}, cpu_engine);
        src.CheckReorderToOpMem(src_target);
        src_data = const_cast<T*>(
            reinterpret_cast<T*>(src.GetOpMem().get_data_handle()));
      }

      const T* diff_dst_data = diff_dst_tensor.flat<T>().data();
      if (diff_dst_md.data.format != bn_bwd->GetDiffDstFmt()) {
        diff_dst.SetUsrMem(diff_dst_md, &diff_dst_tensor);
        auto diff_dst_target = memory::primitive_desc({{diff_dst_dims},
            MklDnnType<T>(), static_cast<memory::format>(
            bn_bwd->GetDiffDstFmt())}, cpu_engine);
        diff_dst.CheckReorderToOpMem(diff_dst_target);
        diff_dst_data = const_cast<T*>(
            reinterpret_cast<T*>(diff_dst.GetOpMem().get_data_handle()));
      }

      // Indices of output tensors
      const size_t kDiffSrcIndex = 0;  // index of diff_src tensor

      // allocate output tensor: diff_src, always set as MKL-DNN layout
      MklDnnShape dnn_shape_diff_src;
      TensorShape tf_shape_diff_src;
      dnn_shape_diff_src.SetMklTensor(true);
      auto diff_src_pd = bn_bwd->GetDiffSrcPd();
      dnn_shape_diff_src.SetMklLayout(&diff_src_pd);
      dnn_shape_diff_src.SetElemType(MklDnnType<T>());
      dnn_shape_diff_src.SetTfLayout(src_dims.size(), src_dims,
                                     format_m);
      dnn_shape_diff_src.SetTfDimOrder(src_dims.size(), tensor_format_);
      tf_shape_diff_src.AddDim(diff_src_pd.get_size() / sizeof(T));
      AllocateOutputSetMklShape(context, kDiffSrcIndex, &diff_src_tensor,
                                tf_shape_diff_src, dnn_shape_diff_src);

      T* mean_data = static_cast<T*>(const_cast<T*>(
                         saved_mean_tensor.flat<T>().data()));
      T* variance_data = static_cast<T*>(const_cast<T*>(
                         saved_variance_tensor.flat<T>().data()));
      T* weights_data = weights_data_tf;
      T* diff_src_data = static_cast<T*>(
                             diff_src_tensor->flat<T>().data());
      T* diff_weights_data = static_cast<T*>(
                             diff_weights.GetAllocatedBuffer());
      // Execute
      bn_bwd->Execute(src_data, mean_data, variance_data, diff_dst_data,
          weights_data, diff_src_data, diff_weights_data);

      // allocate output TF tensors: diff_scale and diff_shift
      Tensor* diff_scale_tensor = nullptr;
      Tensor* diff_shift_tensor = nullptr;
      AllocateTFOutputs(context, scale_tensor.shape(), &diff_scale_tensor,
                        &diff_shift_tensor);

      // copy data: diff_scale and diff_shift
      auto diff_scale_data = diff_scale_tensor->flat<T>().data();
      auto diff_shift_data = diff_shift_tensor->flat<T>().data();
      std::memcpy(reinterpret_cast<char*>(diff_scale_data),
                  reinterpret_cast<char*>(diff_weights_data),
                  depth_ * sizeof(T));
      std::memcpy(reinterpret_cast<char*>(diff_shift_data),
                  reinterpret_cast<char*>(diff_weights_data + depth_),
                  depth_ * sizeof(T));
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 private:
  T epsilon_;
  TensorFormat tensor_format_;
  int depth_;  // batch normalization is done for per channel.
  bool is_training_;
  engine cpu_engine = engine(engine::cpu, 0);

  void ExtractParams(OpKernelContext* context) {
    const Tensor& input = MklGetInput(context, 0);
    depth_ = static_cast<int>(GetTensorDim(input, tensor_format_, 'C'));
  }

  void HandleEmptyInput(OpKernelContext* context, TensorShape tf_shape_src,
                        TensorShape tf_shape_scale_shift,
                        Tensor** diff_src_tensor) {
    const size_t kDiffSrcIndex = 0;

    MklDnnShape dnn_shape_diff_src;
    dnn_shape_diff_src.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kDiffSrcIndex, diff_src_tensor,
                              tf_shape_src, dnn_shape_diff_src);
    int num_elements = (*diff_src_tensor)->shape().num_elements();
    auto diff_src_data = (*diff_src_tensor)->flat<T>().data();
    for (size_t i = 0; i < num_elements; i++)
        diff_src_data[i] = 0;

    Tensor* diff_scale_tensor = nullptr;
    Tensor* diff_shift_tensor = nullptr;
    AllocateTFOutputs(context, tf_shape_scale_shift, &diff_scale_tensor,
                      &diff_shift_tensor);
  }

  void AllocateTFOutputs(OpKernelContext* context,
                         TensorShape tf_shape_scale_shift,
                         Tensor** diff_scale_tensor,
                         Tensor** diff_shift_tensor) {
    CHECK_NOTNULL(diff_scale_tensor);
    CHECK_NOTNULL(diff_shift_tensor);

    const size_t kDiffScaleIndex = 1;
    const size_t kDiffShiftIndex = 2;
    const size_t kP1Index = 3;
    const size_t kP2Index = 4;

    // separate out scale and shift grad and copy to individual tensors
    MklDnnShape mkl_shape_diff_scale;
    mkl_shape_diff_scale.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kDiffScaleIndex, diff_scale_tensor,
                              tf_shape_scale_shift, mkl_shape_diff_scale);
    CHECK_NOTNULL(*diff_scale_tensor);
    int diff_scale_num_elements = (*diff_scale_tensor)->shape().num_elements();
    auto diff_scale_data =  (*diff_scale_tensor)->flat<T>().data();
    for (size_t i = 0; i < diff_scale_num_elements; i++)
      diff_scale_data[i] = 0;

    MklDnnShape mkl_shape_diff_shift;
    mkl_shape_diff_shift.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kDiffShiftIndex, diff_shift_tensor,
                              tf_shape_scale_shift, mkl_shape_diff_shift);
    CHECK_NOTNULL(*diff_shift_tensor);
    int diff_shift_num_elements = (*diff_shift_tensor)->shape().num_elements();
    auto diff_shift_data = (*diff_shift_tensor)->flat<T>().data();
    for (size_t i = 0; i < diff_shift_num_elements; i++)
      diff_shift_data[i] = 0;

    // Placeholders for estimated_mean and estimated_variance, which are
    // used for inference and thus not needed here for gradient computation.
    Tensor *p1_tensor = nullptr, *p2_tensor = nullptr;
    MklDnnShape mkl_shape_p;
    mkl_shape_p.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kP1Index, &p1_tensor, TensorShape({}),
                              mkl_shape_p);
    AllocateOutputSetMklShape(context, kP2Index, &p2_tensor, TensorShape({}),
                              mkl_shape_p);
  }

  memory::dims GetMeanVarianceDims() { return memory::dims({1, depth_}); }
};

#endif

#define REGISTER_MKL_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("_MklFusedBatchNorm")                \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T")               \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklFusedBatchNormOp<CPUDevice, T>);
TF_CALL_float(REGISTER_MKL_CPU);
#undef REGISTER_MKL_CPU

#define REGISTER_MKL_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("_MklFusedBatchNormGrad")            \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T")               \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklFusedBatchNormGradOp<CPUDevice, T>);
TF_CALL_float(REGISTER_MKL_CPU);
#undef REGISTER_MKL_CPU
}  // namespace tensorflow

#endif  // INTEL_MKL
