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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/mkl_pooling_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/padding.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

// An implementation of MaxPooling (forward).
template <typename Device, typename T>
class MklMaxPoolingOp : public OpKernel {
 public:
  explicit MklMaxPoolingOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;

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
  }

  void Compute(OpKernelContext* context) override {
    MklMaxPoolingOpContext mkl_context;
    // Get the input tensor
    const Tensor& tensor_in = MklGetInput(context, 0);
    GetMklShape(context, 0, &mkl_context.input_shape);
    bool input_in_mkl_format = mkl_context.input_shape.IsMklTensor();

    mkl_context.params.in_dim = 4;
    MklPoolParameters pool_params;
    if (input_in_mkl_format == false) {
      pool_params.Init(context, ksize_, stride_, padding_, data_format_,
                       tensor_in.shape());
      OP_REQUIRES(
          context, (pool_params.depth_window == 1),
          errors::Unimplemented("Depthwise max pooling not supported by MKL"));

    } else {
      pool_params.Init(context, ksize_, stride_, padding_, data_format_,
                       &mkl_context.input_shape);
    }

    // Extract the parameters for the op from the pooling specs

    ExtractMklOpParams(context, data_format_, pool_params, &mkl_context.params);

    mkl_context.MklCreateLayoutsAndPrimitives(context);
    OP_REQUIRES_OK(context, context->status());

    // Declare output tensor
    TensorShape tensor_out_shape;
    MklShape mkl_out_shape, mkl_workspace_shape;
    mkl_out_shape.SetMklTensor(true);
    mkl_out_shape.SetMklLayout(mkl_context.prim_pooling_fwd, dnnResourceDst);
    mkl_out_shape.SetTfLayout(mkl_context.params.in_dim,
                              mkl_context.params.out_sizes,
                              mkl_context.params.out_strides);
    mkl_out_shape.SetTfDimOrder(mkl_context.params.in_dim, data_format_);

    Tensor* output_tensor = nullptr;
    tensor_out_shape.AddDim(dnnLayoutGetMemorySize_F32(static_cast<dnnLayout_t>(
                                mkl_out_shape.GetMklLayout())) /
                            sizeof(T));
    AllocateOutputSetMklShape(context, 0, &output_tensor, tensor_out_shape,
                              mkl_out_shape);

    Tensor* workspace_tensor;
    void* workspace_buf = nullptr;

    TensorShape workspace_shape;
    mkl_workspace_shape.SetMklTensor(false);
    workspace_shape.AddDim(dnnLayoutGetMemorySize_F32(static_cast<dnnLayout_t>(
                               mkl_context.lt_workspace)) /
                           sizeof(T));
    AllocateOutputSetMklShape(context, 1, &workspace_tensor, workspace_shape,
                              mkl_workspace_shape);

    mkl_context.pooling_res[dnnResourceWorkspace] = const_cast<void*>(
        static_cast<const void*>(workspace_tensor->flat<T>().data()));
    mkl_context.pooling_res[dnnResourceSrc] =
        const_cast<void*>(static_cast<const void*>(tensor_in.flat<T>().data()));
    mkl_context.pooling_res[dnnResourceDst] = const_cast<void*>(
        static_cast<const void*>(output_tensor->flat<T>().data()));

    CHECK_EQ(
        dnnExecute_F32(mkl_context.prim_pooling_fwd, mkl_context.pooling_res),
        E_SUCCESS);

    mkl_context.MklCleanup();
  }

 private:
  typedef struct {
    MklPoolingOpParams params;
    MklShape input_shape;
    void* pooling_res[dnnResourceNumber];
    dnnPrimitive_t prim_pooling_fwd = nullptr;
    dnnLayout_t lt_user_input = nullptr, lt_workspace = nullptr;

    void MklCreateLayoutsAndPrimitives(OpKernelContext* context) {
      bool input_in_mkl_format = input_shape.IsMklTensor();
      // Create or use existing DNN user layout
      if (input_in_mkl_format == false) {
        CHECK_EQ(dnnLayoutCreate_F32(&lt_user_input, params.in_dim,
                                     params.in_sizes, params.in_strides),
                 E_SUCCESS);
      } else {
        lt_user_input = (dnnLayout_t)input_shape.GetCurLayout();
      }

      dnnAlgorithm_t algorithm = dnnAlgorithmPoolingMax;
      dnnPrimitiveAttributes_t primAttr = nullptr;

      // Create DNN primitives
      CHECK_EQ(dnnPoolingCreateForward_F32(
                   &prim_pooling_fwd, primAttr, algorithm, lt_user_input,
                   params.kernel_size, params.kernel_stride, params.in_offset,
                   dnnBorderZerosAsymm),
               E_SUCCESS);

      // Creates layout for the workspace
      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&lt_workspace, prim_pooling_fwd,
                                                dnnResourceWorkspace),
               E_SUCCESS);
    }

    void MklCleanup() {
      bool input_in_mkl_format = input_shape.IsMklTensor();
      CHECK_EQ(dnnDelete_F32(prim_pooling_fwd), E_SUCCESS);
      if (!input_in_mkl_format) {
        CHECK_EQ(dnnLayoutDelete_F32(lt_user_input), E_SUCCESS);
      }
      CHECK_EQ(dnnLayoutDelete_F32(lt_workspace), E_SUCCESS);
    }
  } MklMaxPoolingOpContext;

  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool workspace_enabled_;
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
    MklMaxPoolingGradOpContext mkl_context;
    // Input - The original input tensor
    const Tensor& tensor_in = MklGetInput(context, 0);

    // Output - Backprop tensor for input.
    Tensor* output_tensor = nullptr;

    GetMklShape(context, 0, &mkl_context.input_shape);
    GetMklShape(context, 2, &mkl_context.output_backprop_shape);
    bool input_in_mkl_format = mkl_context.input_shape.IsMklTensor();

    if (input_in_mkl_format == false)
      mkl_context.params.in_dim = tensor_in.dims();
    else
      mkl_context.params.in_dim = mkl_context.input_shape.GetDimension();

    MklPoolParameters pool_params;
    if (input_in_mkl_format == false) {
      pool_params.Init(context, ksize_, stride_, padding_, data_format_,
                       tensor_in.shape());
      OP_REQUIRES(
          context, (pool_params.depth_window == 1),
          errors::Unimplemented("Depthwise max pooling not supported by MKL"));

    } else {
      pool_params.Init(context, ksize_, stride_, padding_, data_format_,
                       &mkl_context.input_shape);
    }

    // Extract the parameters for the op from the pooling specs
    ExtractMklOpParams(context, data_format_, pool_params, &mkl_context.params);

    mkl_context.MklCreateLayouts(context);
    OP_REQUIRES_OK(context, context->status());

    mkl_context.MklCreatePrimitives(context, workspace_enabled_);
    OP_REQUIRES_OK(context, context->status());

    mkl_context.MklPrepareInputs(context, workspace_enabled_);
    OP_REQUIRES_OK(context, context->status());

    // Create shape for the input back prop output
    TensorShape mkl_input_backprop;
    MklShape mkl_output_shape;
    mkl_output_shape.SetMklTensor(true);
    mkl_output_shape.SetMklLayout(mkl_context.prim_pooling_bwd,
                                  dnnResourceDiffSrc);
    mkl_output_shape.SetTfLayout(mkl_context.params.in_dim,
                                 mkl_context.params.in_sizes,
                                 mkl_context.params.in_strides);
    mkl_output_shape.SetTfDimOrder(mkl_context.params.in_dim, data_format_);

    mkl_input_backprop.AddDim(
        dnnLayoutGetMemorySize_F32(
            static_cast<dnnLayout_t>(mkl_output_shape.GetMklLayout())) /
        sizeof(T));
    AllocateOutputSetMklShape(context, 0, &output_tensor, mkl_input_backprop,
                              mkl_output_shape);
    mkl_context.pooling_res[dnnResourceDiffSrc] = const_cast<void*>(
        static_cast<const void*>(output_tensor->flat<T>().data()));

    int64 output_size = output_tensor->NumElements();
    for (int64 i = 0; i < output_size; ++i) {
      (static_cast<float*>(mkl_context.pooling_res[dnnResourceDiffSrc]))[i] = 0;
    }

    CHECK_EQ(
        dnnExecute_F32(mkl_context.prim_pooling_bwd, mkl_context.pooling_res),
        E_SUCCESS);

    mkl_context.MklCleanup(workspace_enabled_);
  }

 private:
  typedef struct {
    MklPoolingOpParams params;
    MklShape input_shape, output_backprop_shape;
    void* pooling_resfwd[dnnResourceNumber];
    void* pooling_res[dnnResourceNumber];
    dnnPrimitive_t prim_pooling_fwd = nullptr, prim_pooling_bwd = nullptr,
                   convert_input = nullptr, convert_outbackprop = nullptr;
    dnnLayout_t lt_outbackprop_user = nullptr, lt_outbackprop_prim = nullptr,
                lt_input_user = nullptr, lt_input_prim = nullptr;
    void* input_buf;
    void* outbackprop_buf;
    Tensor tmp_output_buf_tensor;
    Tensor workspace_buf_tensor;
    Tensor input_buf_tensor, outbackprop_buf_tensor;

    void MklCreateLayouts(OpKernelContext* context) {
      bool input_in_mkl_format = input_shape.IsMklTensor();
      bool outbackprop_in_mkl_format = output_backprop_shape.IsMklTensor();
      // Create DNN user layout for input and outbackprop or get existing layout
      if (input_in_mkl_format == false) {
        CHECK_EQ(dnnLayoutCreate_F32(&lt_input_user, params.in_dim,
                                     params.in_sizes, params.in_strides),
                 E_SUCCESS);
      } else {
        lt_input_user = (dnnLayout_t)input_shape.GetCurLayout();
      }

      // We dont care about the output layout for now as we can create it from
      // primitives for the max pooling fwd prop
      if (outbackprop_in_mkl_format == false) {
        CHECK_EQ(dnnLayoutCreate_F32(&lt_outbackprop_user, params.in_dim,
                                     params.out_sizes, params.out_strides),
                 E_SUCCESS);
      } else {
        lt_outbackprop_user = (dnnLayout_t)output_backprop_shape.GetCurLayout();
      }
    }

    // Create DNN primitives
    void MklCreatePrimitives(OpKernelContext* context, bool workspace_enabled) {
      dnnAlgorithm_t algorithm = dnnAlgorithmPoolingMax;
      dnnPrimitiveAttributes_t primAttr = nullptr;

      if (workspace_enabled == false) {
        CHECK_EQ(dnnPoolingCreateForward_F32(
                     &prim_pooling_fwd, primAttr, algorithm, lt_input_user,
                     params.kernel_size, params.kernel_stride, params.in_offset,
                     dnnBorderZerosAsymm),
                 E_SUCCESS);
      }

      CHECK_EQ(dnnPoolingCreateBackward_F32(
                   &prim_pooling_bwd, primAttr, algorithm, lt_input_user,
                   params.kernel_size, params.kernel_stride, params.in_offset,
                   dnnBorderZerosAsymm),
               E_SUCCESS);

      // Creates conversions
      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(
                   &lt_outbackprop_prim, prim_pooling_bwd, dnnResourceDiffDst),
               E_SUCCESS);

      if (workspace_enabled == false) {
        CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(
                     &lt_input_prim, prim_pooling_fwd, dnnResourceSrc),
                 E_SUCCESS);
        if (!dnnLayoutCompare_F32(lt_input_user, lt_input_prim)) {
          CHECK_EQ(dnnConversionCreate_F32(&convert_input, lt_input_user,
                                           lt_input_prim),
                   E_SUCCESS);
          AllocTmpBuffer(context, &input_buf_tensor, lt_input_prim, &input_buf);
        }
      }

      if (!dnnLayoutCompare_F32(lt_outbackprop_user, lt_outbackprop_prim)) {
        CHECK_EQ(
            dnnConversionCreate_F32(&convert_outbackprop, lt_outbackprop_user,
                                    lt_outbackprop_prim),
            E_SUCCESS);
        AllocTmpBuffer(context, &outbackprop_buf_tensor, lt_outbackprop_prim,
                       &outbackprop_buf);
      }
    }

    // Compare incoming tensor layouts with MKL preferred layouts and convert
    // data to the preferred layout if necessary
    void MklPrepareInputs(OpKernelContext* context, bool workspace_enabled) {
      const Tensor& tensor_in = MklGetInput(context, 0);
      const Tensor& out_backprop = MklGetInput(context, 2);
      bool input_in_mkl_format = input_shape.IsMklTensor();
      bool outbackprop_in_mkl_format = output_backprop_shape.IsMklTensor();

      void* tmp_output_buf = nullptr;
      void* workspace_buf = nullptr;

      if (workspace_enabled == false) {
        if (convert_input != nullptr) {
          if (input_in_mkl_format == false) {
            CHECK_EQ(
                dnnConversionExecute_F32(
                    convert_input, const_cast<void*>(static_cast<const void*>(
                                       tensor_in.flat<T>().data())),
                    input_buf),
                E_SUCCESS);
            CHECK_EQ(dnnDelete_F32(convert_input), E_SUCCESS);
            convert_input = nullptr;
          } else {
            input_shape.GetConvertedFlatData(
                lt_input_prim, const_cast<void*>(static_cast<const void*>(
                                   tensor_in.flat<T>().data())),
                input_buf);
          }
          pooling_resfwd[dnnResourceSrc] = input_buf;
        } else {
          pooling_resfwd[dnnResourceSrc] = const_cast<void*>(
              static_cast<const void*>(tensor_in.flat<T>().data()));
        }

        dnnLayout_t lt_workspace;
        CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(
                     &lt_workspace, prim_pooling_fwd, dnnResourceWorkspace),
                 E_SUCCESS);
        AllocTmpBuffer(context, &workspace_buf_tensor, lt_workspace,
                       &workspace_buf);
        pooling_resfwd[dnnResourceWorkspace] = workspace_buf;

        dnnLayoutDelete_F32(lt_workspace);

        // We create the layout for max pooling fwd prop tmp output here
        AllocTmpBuffer(context, &tmp_output_buf_tensor, lt_outbackprop_prim,
                       &tmp_output_buf);
        pooling_resfwd[dnnResourceDst] = tmp_output_buf;

        CHECK_EQ(dnnExecute_F32(prim_pooling_fwd, pooling_resfwd), E_SUCCESS);
        pooling_res[dnnResourceWorkspace] =
            pooling_resfwd[dnnResourceWorkspace];
      } else {
        const Tensor& workspace = MklGetInput(context, 3);
        pooling_res[dnnResourceWorkspace] = const_cast<void*>(
            static_cast<const void*>(workspace.flat<T>().data()));
      }

      // Out backprop conversions if needed
      if (convert_outbackprop != nullptr) {
        if (outbackprop_in_mkl_format == false) {
          CHECK_EQ(dnnConversionExecute_F32(
                       convert_outbackprop,
                       const_cast<void*>(static_cast<const void*>(
                           out_backprop.flat<T>().data())),
                       outbackprop_buf),
                   E_SUCCESS);
          CHECK_EQ(dnnDelete_F32(convert_outbackprop), E_SUCCESS);
        } else {
          output_backprop_shape.GetConvertedFlatData(
              lt_outbackprop_prim, const_cast<void*>(static_cast<const void*>(
                                       out_backprop.flat<T>().data())),
              outbackprop_buf);
        }
        pooling_res[dnnResourceDiffDst] = outbackprop_buf;
      } else {
        pooling_res[dnnResourceDiffDst] = const_cast<void*>(
            static_cast<const void*>(out_backprop.flat<T>().data()));
      }
    }

    void MklCleanup(bool workspace_enabled) {
      bool input_in_mkl_format = input_shape.IsMklTensor();
      bool outbackprop_in_mkl_format = output_backprop_shape.IsMklTensor();
      if (workspace_enabled == false) {
        CHECK_EQ(dnnDelete_F32(prim_pooling_fwd), E_SUCCESS);
      }
      CHECK_EQ(dnnDelete_F32(prim_pooling_bwd), E_SUCCESS);
      if (outbackprop_in_mkl_format == false) {
        CHECK_EQ(dnnLayoutDelete_F32(lt_outbackprop_user), E_SUCCESS);
      }
      CHECK_EQ(dnnLayoutDelete_F32(lt_outbackprop_prim), E_SUCCESS);
      if (input_in_mkl_format == false) {
        CHECK_EQ(dnnLayoutDelete_F32(lt_input_user), E_SUCCESS);
      }
      if (workspace_enabled == false) {
        CHECK_EQ(dnnLayoutDelete_F32(lt_input_prim), E_SUCCESS);
      }
    }
  } MklMaxPoolingGradOpContext;

  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;

  bool workspace_enabled_;
};

REGISTER_KERNEL_BUILDER(Name("_MklMaxPool")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T")
                            .Label(mkl_op_registry::kMklOpLabel),
                        MklMaxPoolingOp<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("_MklMaxPoolGrad")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T")
                            .Label(mkl_op_registry::kMklOpLabel),
                        MklMaxPoolingGradOp<CPUDevice, float>);

}  // namespace tensorflow
#endif  // INTEL_MKL
