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

// See docs in ../ops/nn_ops.cc.
#ifdef INTEL_MKL
#define EIGEN_USE_THREADS

#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/mkl_pooling_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/padding.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

// An implementation of MaxPooling (forward).
template <typename Device, typename T>
class MklMaxPoolingOp : public OpKernel {
 public:
  explicit MklMaxPoolingOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;

    pooling_fwd_ = nullptr;
    lt_user_input_fwd_ = nullptr;
    lt_workspace_ = nullptr;

    workspace_ = nullptr;

    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented("Pooling is not yet supported on the "
                                      "batch dimension."));

    workspace_enabled_ = false;
    // We may not get this attribute for this node if it does not go through
    // graph rewrite pass. So we do not check for error while retrieving this
    // attribute value.
    context->GetAttr("workspace_enabled", &workspace_enabled_);

    mkl_params_.in_dim = 4;
  }

  void Compute(OpKernelContext* context) override {
    // Get the input tensor
    const Tensor& tensor_in = MklGetInput(context, 0);
    GetMklShape(context, 0, &mkl_input_shape);
    input_in_mkl_format_ = mkl_input_shape.IsMklTensor();

    MklPoolParameters params;
    if (input_in_mkl_format_ == false) {
      params.Init(context, ksize_, stride_, padding_, data_format_,
                  tensor_in.shape());
      OP_REQUIRES(context, (params.depth_window == 1),
                  errors::Unimplemented(
                    "Depthwise max pooling not supported by MKL"));

    } else {
      params.Init(context, ksize_, stride_, padding_, data_format_,
                   &mkl_input_shape);
    }

    // Extract the parameters for the op from the pooling specs
    ExtractMklOpParams(context, data_format_, params, &mkl_params_);

    MklCreateLayoutsAndPrimitives(context);

    // Declare output tensor
    TensorShape tensor_out_shape;
    MklShape mkl_out_shape;
    mkl_out_shape.SetMklTensor(true);
    mkl_out_shape.SetMklLayout(pooling_fwd_, dnnResourceDst);
    mkl_out_shape.SetTfLayout(mkl_params_.in_dim,
                              mkl_params_.out_sizes,
                              mkl_params_.out_strides);

    Tensor* output_tensor = nullptr;
    tensor_out_shape.AddDim(dnnLayoutGetMemorySize_F32(
        static_cast<dnnLayout_t>(mkl_out_shape.GetMklLayout())) / sizeof(T));
    AllocateOutputSetMklshape(context,
                              0,
                              &output_tensor,
                              tensor_out_shape,
                              mkl_out_shape);

    // For allocating temporary buffer
    Tensor workspace_tensor;

    if (workspace_enabled_) {
      Tensor *workspace_tensor;
      TensorShape workspace_shape;
      workspace_shape.AddDim(dnnLayoutGetMemorySize_F32(
        static_cast<dnnLayout_t>(lt_workspace_))/ sizeof(T));
      AllocateOutputSetMklshape(context, 1, &workspace_tensor,
                                workspace_shape, mkl_out_shape);
      pooling_res_[dnnResourceWorkspace] = const_cast<void*>(
        static_cast<const void*>(workspace_tensor->flat<T>().data()));
    } else {
      AllocTmpBuffer(context, &workspace_tensor, lt_workspace_, &workspace_);
      pooling_res_[dnnResourceWorkspace] = workspace_;
    }

    pooling_res_[dnnResourceSrc] =
      const_cast<void*>(
        static_cast<const void*>(tensor_in.flat<T>().data()));
    pooling_res_[dnnResourceDst] =
      const_cast<void*>(
        static_cast<const void*>(output_tensor->flat<T>().data()));

    CHECK_EQ(dnnExecute_F32(pooling_fwd_, pooling_res_),
             E_SUCCESS);

    if (workspace_enabled_ == false) {
      workspace_ = nullptr;
    }

    MklCleanup();
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  MklShape mkl_input_shape;

  bool workspace_enabled_;
  bool input_in_mkl_format_;

  void* workspace_;
  void* pooling_res_[dnnResourceNumber];

  dnnPrimitive_t pooling_fwd_;
  dnnLayout_t lt_user_input_fwd_;
  dnnLayout_t lt_workspace_;

  MklPoolingOpParams mkl_params_;

  void MklCreateLayoutsAndPrimitives(OpKernelContext* context) {
    // Create or use existing DNN user layout
    if (input_in_mkl_format_ == false) {
      CHECK_EQ(dnnLayoutCreate_F32(&lt_user_input_fwd_,
                                   mkl_params_.in_dim,
                                   mkl_params_.in_sizes,
                                   mkl_params_.in_strides),
               E_SUCCESS);
    } else {
      lt_user_input_fwd_ = (dnnLayout_t)mkl_input_shape.GetCurLayout();
    }

    dnnAlgorithm_t algorithm = dnnAlgorithmPoolingMax;
    dnnPrimitiveAttributes_t primAttr = nullptr;

    // Create DNN primitives
    CHECK_EQ(dnnPoolingCreateForward_F32(&pooling_fwd_,
                                         primAttr,
                                         algorithm,
                                         lt_user_input_fwd_,
                                         mkl_params_.kernel_size,
                                         mkl_params_.kernel_stride,
                                         mkl_params_.in_offset,
                                         dnnBorderZerosAsymm),
             E_SUCCESS);

    // Creates layout for the workspace
    CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&lt_workspace_,
                                              pooling_fwd_,
                                              dnnResourceWorkspace),
             E_SUCCESS);
  }

  void MklCleanup() {
    CHECK_EQ(dnnDelete_F32(pooling_fwd_), E_SUCCESS);
    pooling_fwd_ = nullptr;

    if (input_in_mkl_format_) {
      CHECK_EQ(dnnLayoutDelete_F32(lt_user_input_fwd_), E_SUCCESS);
      lt_user_input_fwd_ = nullptr;
    }

    CHECK_EQ(dnnLayoutDelete_F32(lt_workspace_), E_SUCCESS);
    lt_workspace_ = nullptr;
  }
};

// The operation to compute MaxPool gradients.
// It takes three inputs:
//   - The original input tensor
//   - The original output tensor
//   - Backprop tensor for output
// It produces one output: backprop tensor for input.
template <class Device, class T>
class MklMaxPoolingGradOp : public OpKernel {
 public:
  explicit MklMaxPoolingGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;

    pooling_fwd_ = nullptr;
    pooling_bwd_ = nullptr;

    lt_outbackprop_user_ = nullptr;
    lt_outbackprop_prim_ = nullptr;
    lt_input_user_ = nullptr;
    lt_input_prim_ = nullptr;

    convert_outbackprop_ = nullptr;
    convert_input_ = nullptr;

    input_buf_ = nullptr;
    outbackprop_buf_ = nullptr;

    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    workspace_enabled_ = false;
    // We may not get this attribute for this node if it does not go through
    // graph rewrite pass. So we do not check for error while retrieving this
    // attribute value.
    context->GetAttr("workspace_enabled", &workspace_enabled_);
  }

  void Compute(OpKernelContext* context) override {
    // Input - The original input tensor
    const Tensor& tensor_in = MklGetInput(context, 0);

    // Output - Backprop tensor for input.
    Tensor* output_tensor = nullptr;

    GetMklShape(context, 0, &mkl_input_shape);
    input_in_mkl_format_ = mkl_input_shape.IsMklTensor();

    MklShape mkl_output_backprop_shape;
    GetMklShape(context, 2, &mkl_output_backprop_shape);
    outbackprop_in_mkl_format_ = mkl_output_backprop_shape.IsMklTensor();

    if (input_in_mkl_format_ == false)
      mkl_params_.in_dim = tensor_in.dims();
    else
      mkl_params_.in_dim = mkl_input_shape.GetDimension();

    MklPoolParameters params;
    if (input_in_mkl_format_ == false) {
      params.Init(context, ksize_, stride_, padding_, data_format_,
                  tensor_in.shape());
      OP_REQUIRES(context, (params.depth_window == 1),
                  errors::Unimplemented(
                    "Depthwise max pooling not supported by MKL"));

    } else {
      params.Init(context, ksize_, stride_, padding_, data_format_,
                   &mkl_input_shape);
    }

    // Extract the parameters for the op from the pooling specs
    ExtractMklOpParams(context, data_format_, params, &mkl_params_);

    // mkldnn
    MklCreateLayouts(context);
    MklCreatePrimitives(context);
    MklPrepareInputs(context);

    // Create shape for the input back prop output
    TensorShape mkl_input_backprop;
    MklShape mklOutputShape;
    mklOutputShape.SetMklTensor(true);
    mklOutputShape.SetMklLayout(pooling_bwd_, dnnResourceDiffSrc);
    mklOutputShape.SetTfLayout(mkl_params_.in_dim,
                               mkl_params_.in_sizes,
                               mkl_params_.in_strides);

    mkl_input_backprop.AddDim(
        dnnLayoutGetMemorySize_F32(
            static_cast<dnnLayout_t>(mklOutputShape.GetMklLayout())) /
        sizeof(T));
    AllocateOutputSetMklshape(context,
                              0,
                              &output_tensor,
                              mkl_input_backprop,
                              mklOutputShape);
    pooling_res_[dnnResourceDiffSrc] =
      static_cast<void*>(const_cast<float*>(output_tensor->flat<T>().data()));

    int64 output_size = output_tensor->NumElements();
    for (int64 i = 0; i < output_size; ++i) {
      (static_cast<float*>(pooling_res_[dnnResourceDiffSrc]))[i] = 0;
    }

    CHECK_EQ(dnnExecute_F32(pooling_bwd_, pooling_res_), E_SUCCESS);

    MklCleanup();
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  MklShape mkl_input_shape;

  bool workspace_enabled_;
  bool input_in_mkl_format_;
  bool outbackprop_in_mkl_format_;

  void* input_buf_;
  void* outbackprop_buf_;
  void* pooling_res_fwd_[dnnResourceNumber];  // Pooling resource array for fwd
  void* pooling_res_[dnnResourceNumber];      // Pooling resource array

  dnnPrimitive_t pooling_fwd_;
  dnnPrimitive_t pooling_bwd_;
  dnnPrimitive_t convert_input_;
  dnnPrimitive_t convert_outbackprop_;

  dnnLayout_t lt_outbackprop_user_;
  dnnLayout_t lt_outbackprop_prim_;
  dnnLayout_t lt_input_user_;
  dnnLayout_t lt_input_prim_;

  MklPoolingOpParams mkl_params_;

  void MklCreateLayouts(OpKernelContext* context) {
    // Create DNN user layout for input and outbackprop or get existing layout
    if (input_in_mkl_format_ == false) {
      CHECK_EQ(dnnLayoutCreate_F32(&lt_input_user_,
                                   mkl_params_.in_dim,
                                   mkl_params_.in_sizes,
                                   mkl_params_.in_strides),
               E_SUCCESS);
    } else {
      lt_input_user_ = (dnnLayout_t)mkl_input_shape.GetCurLayout();
    }

    MklShape mkl_output_backprop_shape;
    GetMklShape(context, 2, &mkl_output_backprop_shape);

    // We dont care about the output layout for now as we can create it from
    // primitives for the max pooling fwd prop
    if (outbackprop_in_mkl_format_ == false) {
      CHECK_EQ(dnnLayoutCreate_F32(&lt_outbackprop_user_,
                                   mkl_params_.in_dim,
                                   mkl_params_.out_sizes,
                                   mkl_params_.out_strides),
               E_SUCCESS);
    } else {
      lt_outbackprop_user_ =
        (dnnLayout_t)mkl_output_backprop_shape.GetCurLayout();
    }
  }

  // Create DNN primitives
  void MklCreatePrimitives(OpKernelContext* context) {
    dnnAlgorithm_t algorithm = dnnAlgorithmPoolingMax;
    dnnPrimitiveAttributes_t primAttr = nullptr;

    if (workspace_enabled_ == false) {
      CHECK_EQ(dnnPoolingCreateForward_F32(&pooling_fwd_,
                                           primAttr,
                                           algorithm,
                                           lt_input_user_,
                                           mkl_params_.kernel_size,
                                           mkl_params_.kernel_stride,
                                           mkl_params_.in_offset,
                                           dnnBorderZerosAsymm),
               E_SUCCESS);
    }

    CHECK_EQ(dnnPoolingCreateBackward_F32(&pooling_bwd_,
                                          primAttr,
                                          algorithm,
                                          lt_input_user_,
                                          mkl_params_.kernel_size,
                                          mkl_params_.kernel_stride,
                                          mkl_params_.in_offset,
                                          dnnBorderZerosAsymm),
             E_SUCCESS);

    // Creates conversions
    CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&lt_outbackprop_prim_,
                                              pooling_bwd_,
                                              dnnResourceDiffDst),
             E_SUCCESS);

    // Tensors needed to create temporary buffers
    Tensor input_buf_tensor, outbackprop_buf_tensor;

    if (workspace_enabled_ == false) {
      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&lt_input_prim_,
                                                pooling_fwd_,
                                                dnnResourceSrc),
               E_SUCCESS);
      if (!dnnLayoutCompare_F32(lt_input_user_, lt_input_prim_)) {
        CHECK_EQ(dnnConversionCreate_F32(&convert_input_,
                                         lt_input_user_,
                                         lt_input_prim_),
                 E_SUCCESS);
        AllocTmpBuffer(context,
                       &input_buf_tensor,
                       lt_input_prim_,
                       &input_buf_);
      }
    }

    if (!dnnLayoutCompare_F32(lt_outbackprop_user_, lt_outbackprop_prim_)) {
      CHECK_EQ(dnnConversionCreate_F32(&convert_outbackprop_,
                                       lt_outbackprop_user_,
                                       lt_outbackprop_prim_),
               E_SUCCESS);
      AllocTmpBuffer(context,
                     &outbackprop_buf_tensor,
                     lt_outbackprop_prim_,
                     &outbackprop_buf_);
    }
  }

  // Compare incoming tensor layouts with MKL preferred layouts and convert
  // data to the preferred layout if necessary
  void MklPrepareInputs(OpKernelContext* context) {
    // Input - The original input tensor
    const Tensor& tensor_in = MklGetInput(context, 0);
    // Backprop tensor for output
    const Tensor& out_backprop = MklGetInput(context, 2);

    MklShape mkl_input_shape;
    GetMklShape(context, 0, &mkl_input_shape);

    void* tmp_output_buf;
    Tensor tmp_output_buf_tensor;

    void* workspace_buf;
    Tensor workspace_buf_tensor;

    if (workspace_enabled_ == false) {
      if (convert_input_ != nullptr) {
        if (input_in_mkl_format_ == false) {
          CHECK_EQ(
            dnnConversionExecute_F32(
                convert_input_,
                const_cast<void*>(
                  static_cast<const void*>(tensor_in.flat<T>().data())),
                input_buf_),
            E_SUCCESS);
          CHECK_EQ(dnnDelete_F32(convert_input_), E_SUCCESS);
          convert_input_ = nullptr;
        } else {
          mkl_input_shape.GetConvertedFlatData(
            lt_input_prim_,
            const_cast<void*>(
              static_cast<const void*>(tensor_in.flat<T>().data())),
            input_buf_);
        }
        pooling_res_fwd_[dnnResourceSrc] = input_buf_;
        input_buf_ = nullptr;
      } else {
        pooling_res_fwd_[dnnResourceSrc] =
          const_cast<void*>(
            static_cast<const void*>(tensor_in.flat<T>().data()));
      }

      dnnLayout_t lt_workspace;
      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&lt_workspace,
                                                pooling_fwd_,
                                                dnnResourceWorkspace),
               E_SUCCESS);
      AllocTmpBuffer(context,
                     &workspace_buf_tensor,
                     lt_workspace, &workspace_buf);
      pooling_res_fwd_[dnnResourceWorkspace] = workspace_buf;

      dnnLayoutDelete_F32(lt_workspace);

      // We create the layout for max pooling fwd prop tmp output here
      AllocTmpBuffer(context, &tmp_output_buf_tensor,
                     lt_outbackprop_prim_, &tmp_output_buf);
      pooling_res_fwd_[dnnResourceDst] = tmp_output_buf;

      CHECK_EQ(dnnExecute_F32(pooling_fwd_, pooling_res_fwd_), E_SUCCESS);
      pooling_res_[dnnResourceWorkspace] =
        pooling_res_fwd_[dnnResourceWorkspace];
    } else {
      const Tensor& workspace = MklGetInput(context, 3);
      pooling_res_[dnnResourceWorkspace] = const_cast<void*>(
        static_cast<const void*>(workspace.flat<T>().data()));
    }

    // Out backprop conversions if needed
    if (convert_outbackprop_ != nullptr) {
      if (outbackprop_in_mkl_format_ == false) {
        CHECK_EQ(dnnConversionExecute_F32(
                   convert_outbackprop_,
                   const_cast<void*>(
                     static_cast<const void*>(out_backprop.flat<T>().data())),
                   outbackprop_buf_),
                 E_SUCCESS);
        CHECK_EQ(dnnDelete_F32(convert_outbackprop_), E_SUCCESS);
        convert_outbackprop_ = nullptr;
      } else {
        MklShape mkl_output_backprop_shape;
        GetMklShape(context, 2, &mkl_output_backprop_shape);
        mkl_output_backprop_shape.GetConvertedFlatData(
            lt_outbackprop_prim_,
            const_cast<void*>(
              static_cast<const void*>(out_backprop.flat<T>().data())),
            outbackprop_buf_);
      }
      pooling_res_[dnnResourceDiffDst] = outbackprop_buf_;
      outbackprop_buf_ = nullptr;
    } else {
      pooling_res_[dnnResourceDiffDst] =
        const_cast<void*>(
          static_cast<const void*>(out_backprop.flat<T>().data()));
    }
  }

  void MklCleanup() {
    if (workspace_enabled_ == false) {
      CHECK_EQ(dnnDelete_F32(pooling_fwd_), E_SUCCESS);
      pooling_fwd_ = nullptr;
    }

    CHECK_EQ(dnnDelete_F32(pooling_bwd_), E_SUCCESS);
    pooling_bwd_ = nullptr;

    if (outbackprop_in_mkl_format_ == false) {
      CHECK_EQ(dnnLayoutDelete_F32(lt_outbackprop_user_), E_SUCCESS);
      lt_outbackprop_user_ = nullptr;
    }

    CHECK_EQ(dnnLayoutDelete_F32(lt_outbackprop_prim_), E_SUCCESS);
    lt_outbackprop_prim_ = nullptr;

    if (input_in_mkl_format_ == false) {
      CHECK_EQ(dnnLayoutDelete_F32(lt_input_user_), E_SUCCESS);
      lt_input_user_ = nullptr;
    }

    if (workspace_enabled_ == false) {
      CHECK_EQ(dnnLayoutDelete_F32(lt_input_prim_), E_SUCCESS);
      lt_input_prim_ = nullptr;
    }
  }
};

REGISTER_KERNEL_BUILDER(
  Name("MklMaxPool").Device(DEVICE_CPU).TypeConstraint<float>("T")
  .Label(mkl_layer_registry::kMklLayerLabel),
  MklMaxPoolingOp<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(
  Name("MklMaxPoolGrad").Device(DEVICE_CPU).TypeConstraint<float>("T")
  .Label(mkl_layer_registry::kMklLayerLabel),
  MklMaxPoolingGradOp<CPUDevice, float>);

}       // namespace tensorflow
#endif  // INTEL_MKL
