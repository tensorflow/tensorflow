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

#include "mkl_dnn.h"
#include "mkl_dnn_types.h"
#include "tensorflow/core/util/mkl_util.h"

// TODO(inteltf) Address comments from PR 8968.

namespace tensorflow {
using CPUDevice = Eigen::ThreadPoolDevice;
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

#define REGISTER_MKL_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("_MklFusedBatchNorm")                \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T")               \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklFusedBatchNormOp<CPUDevice, T>);
TF_CALL_float(REGISTER_MKL_CPU);
#undef REGISTER_MKL_CPU

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
