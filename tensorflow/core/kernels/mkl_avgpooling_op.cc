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

#ifdef INTEL_MKL
#define EIGEN_USE_THREADS

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/mkl_util.h"

#include "tensorflow/core/kernels/mkl_pooling_ops_common.h"

#ifndef INTEL_MKL_ML
#include "mkldnn.hpp"
using mkldnn::algorithm;
using mkldnn::engine;
using mkldnn::error;
using mkldnn::memory;
using mkldnn::padding_kind;
using mkldnn::pooling_backward;
using mkldnn::pooling_forward;
using mkldnn::prop_kind;
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

#ifdef INTEL_MKL_ML

template <typename Device, typename T>
class MklAvgPoolingOp : public OpKernel {
 public:
  explicit MklAvgPoolingOp(OpKernelConstruction* context) : OpKernel(context) {
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
  }

  void Compute(OpKernelContext* context) override {
    MklAvgPoolingOpContext mkl_context;
    const Tensor& tensor_in = MklGetInput(context, 0);
    GetMklShape(context, 0, &mkl_context.input_shape);
    bool input_in_mkl_format = mkl_context.input_shape.IsMklTensor();

    if (!input_in_mkl_format)
      mkl_context.params.in_dim = tensor_in.dims();
    else
      mkl_context.params.in_dim = mkl_context.input_shape.GetDimension();

    MklPoolParameters pool_params;
    if (!input_in_mkl_format) {
      pool_params.Init(context, ksize_, stride_, padding_, data_format_,
                       tensor_in.shape());
    } else {
      pool_params.Init(context, ksize_, stride_, padding_, data_format_,
                       &mkl_context.input_shape);
    }

    // Extract the parameters for the op from the pooling specs
    ExtractMklOpParams(context, data_format_, pool_params, &mkl_context.params);

    Tensor mkl_tmp_input_buf_tensor_;
    mkl_context.MklCreateLayoutsAndPrimitives(context,
                                              &mkl_tmp_input_buf_tensor_);
    OP_REQUIRES_OK(context, context->status());

    Tensor workspace_tensor;
    void* workspace_buf;
    AllocTmpBuffer(context, &workspace_tensor, mkl_context.lt_workspace,
                   &workspace_buf);

    if (mkl_context.convert_input != nullptr) {
      if (input_in_mkl_format == false) {
        CHECK_EQ(
            dnnConversionExecute_F32(
                mkl_context.convert_input,
                static_cast<void*>(const_cast<T*>(tensor_in.flat<T>().data())),
                mkl_context.input_buf),
            E_SUCCESS);
        CHECK_EQ(dnnDelete_F32(mkl_context.convert_input), E_SUCCESS);
      } else {
        mkl_context.input_shape.GetConvertedFlatData(
            mkl_context.lt_prim_input,
            static_cast<void*>(const_cast<T*>(tensor_in.flat<T>().data())),
            mkl_context.input_buf);
      }
      mkl_context.pooling_res[dnnResourceSrc] = mkl_context.input_buf;
    } else {
      mkl_context.pooling_res[dnnResourceSrc] =
          static_cast<void*>(const_cast<T*>(tensor_in.flat<T>().data()));
    }

    // Declare output tensor and allocate memory
    Tensor* output = nullptr;
    TensorShape tensor_out_shape;
    MklShape mkl_out_shape;
    mkl_out_shape.SetMklTensor(true);
    mkl_out_shape.SetMklLayout(mkl_context.prim_pooling_fwd, dnnResourceDst);
    mkl_out_shape.SetTfLayout(mkl_context.params.in_dim,
                              mkl_context.params.out_sizes,
                              mkl_context.params.out_strides);
    mkl_out_shape.SetTfDimOrder(mkl_context.params.in_dim, data_format_);

    tensor_out_shape.AddDim(dnnLayoutGetMemorySize_F32(static_cast<dnnLayout_t>(
                                mkl_out_shape.GetMklLayout())) /
                            sizeof(T));

    AllocateOutputSetMklShape(context, 0, &output, tensor_out_shape,
                              mkl_out_shape);
    mkl_context.pooling_res[dnnResourceDst] =
        static_cast<void*>(output->flat<T>().data());

    mkl_context.pooling_res[dnnResourceWorkspace] = workspace_buf;

    CHECK_EQ(
        dnnExecute_F32(mkl_context.prim_pooling_fwd, mkl_context.pooling_res),
        E_SUCCESS);

    mkl_context.MklCleanup();
  }  // Compute

 private:
  typedef struct {
    MklPoolingOpParams params;
    MklShape input_shape;
    dnnPrimitive_t prim_pooling_fwd = nullptr, convert_input = nullptr;
    dnnLayout_t lt_user_input = nullptr, lt_prim_input = nullptr,
                lt_workspace = nullptr;
    void* input_buf = nullptr;
    void* pooling_res[dnnResourceNumber];

    void MklCreateLayoutsAndPrimitives(OpKernelContext* context,
                                       Tensor* mkl_tmp_input_buf_tensor) {
      bool input_in_mkl_format = input_shape.IsMklTensor();

      if (!input_in_mkl_format) {
        CHECK_EQ(dnnLayoutCreate_F32(&lt_user_input, params.in_dim,
                                     params.in_sizes, params.in_strides),
                 E_SUCCESS);
      } else {
        lt_user_input = (dnnLayout_t)input_shape.GetCurLayout();
      }

      dnnAlgorithm_t algorithm = dnnAlgorithmPoolingAvg;
      dnnPrimitiveAttributes_t primAttr = nullptr;

      // Create DNN primitives
      CHECK_EQ(dnnPoolingCreateForward_F32(
                   &prim_pooling_fwd, primAttr, algorithm, lt_user_input,
                   params.kernel_size, params.kernel_stride, params.in_offset,
                   dnnBorderZerosAsymm),
               E_SUCCESS);

      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(
                   &lt_prim_input, prim_pooling_fwd, dnnResourceSrc),
               E_SUCCESS);
      if (!dnnLayoutCompare_F32(lt_user_input, lt_prim_input)) {
        CHECK_EQ(dnnConversionCreate_F32(&convert_input, lt_user_input,
                                         lt_prim_input),
                 E_SUCCESS);

        AllocTmpBuffer(context, mkl_tmp_input_buf_tensor, lt_prim_input,
                       &input_buf);
      }

      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&lt_workspace, prim_pooling_fwd,
                                                dnnResourceWorkspace),
               E_SUCCESS);
    }

    void MklCleanup() {
      bool input_in_mkl_format = input_shape.IsMklTensor();
      if (!input_in_mkl_format) {
        CHECK_EQ(dnnLayoutDelete_F32(lt_user_input), E_SUCCESS);
      }

      CHECK_EQ(dnnDelete_F32(prim_pooling_fwd), E_SUCCESS);
      CHECK_EQ(dnnLayoutDelete_F32(lt_prim_input), E_SUCCESS);
    }
  } MklAvgPoolingOpContext;

  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

//-----------------------------------------------------------------------------

template <class Device, class T>
class MklAvgPoolingGradOp : public OpKernel {
 public:
  explicit MklAvgPoolingGradOp(OpKernelConstruction* context)
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
                errors::Unimplemented("Pooling is not yet supported on the "
                                      "batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    MklAvgPoolingGradOpContext mkl_context;
    const Tensor& tensor_in_shape = MklGetInput(context, 0);
    const Tensor& out_backprop = MklGetInput(context, 1);
    GetMklShape(context, 1, &mkl_context.out_backprop_shape);
    bool outbackprop_in_mkl_format =
        mkl_context.out_backprop_shape.IsMklTensor();

    TensorShape output_shape;
    auto shape_vec = tensor_in_shape.vec<int32>();
    for (int64 i = 0; i < tensor_in_shape.NumElements(); ++i) {
      output_shape.AddDim(shape_vec(i));
    }

    MklPoolParameters pool_params;
    pool_params.Init(context, ksize_, stride_, padding_, data_format_,
                     output_shape);

    if (outbackprop_in_mkl_format == false)
      mkl_context.params.in_dim = out_backprop.dims();
    else
      mkl_context.params.in_dim = mkl_context.out_backprop_shape.GetDimension();

    // Extract the parameters for the op from the pooling specs
    ExtractMklOpParams(context, data_format_, pool_params, &mkl_context.params);

    // Tensors needed to create temporary buffers
    Tensor outbackprop_buf_tensor;
    void* outbackprop_buf;
    mkl_context.MklCreateLayoutsAndPrimitives(context);
    OP_REQUIRES_OK(context, context->status());

    // Check if outbackprop layout requires conversion.
    if (!dnnLayoutCompare_F32(mkl_context.lt_user_outbackprop,
                              mkl_context.lt_prim_outbackprop)) {
      CHECK_EQ(dnnConversionCreate_F32(&mkl_context.convert_outbackprop,
                                       mkl_context.lt_user_outbackprop,
                                       mkl_context.lt_prim_outbackprop),
               E_SUCCESS);

      AllocTmpBuffer(context, &outbackprop_buf_tensor,
                     mkl_context.lt_prim_outbackprop, &outbackprop_buf);

      if (!outbackprop_in_mkl_format) {
        CHECK_EQ(dnnConversionExecute_F32(mkl_context.convert_outbackprop,
                                          static_cast<void*>(const_cast<T*>(
                                              out_backprop.flat<T>().data())),
                                          outbackprop_buf),
                 E_SUCCESS);
        CHECK_EQ(dnnDelete_F32(mkl_context.convert_outbackprop), E_SUCCESS);
      } else {
        mkl_context.out_backprop_shape.GetConvertedFlatData(
            mkl_context.lt_prim_outbackprop,
            static_cast<void*>(const_cast<T*>(out_backprop.flat<T>().data())),
            outbackprop_buf);
      }
      mkl_context.pooling_res[dnnResourceDiffDst] = outbackprop_buf;
    } else {
      mkl_context.pooling_res[dnnResourceDiffDst] =
          static_cast<void*>(const_cast<T*>(out_backprop.flat<T>().data()));
    }

    // Handle workspace requirements.
    Tensor workspace_buf_tensor;
    void* workspace_buf;
    AllocTmpBuffer(context, &workspace_buf_tensor, mkl_context.lt_workspace,
                   &workspace_buf);
    mkl_context.pooling_res[dnnResourceWorkspace] = workspace_buf;

    // Handle MKL output tensor setup.
    Tensor* output = nullptr;
    TensorShape tensor_out_shape;
    MklShape mkl_out_shape;
    mkl_out_shape.SetMklTensor(true);
    mkl_out_shape.SetMklLayout(mkl_context.prim_pooling_bwd,
                               dnnResourceDiffSrc);
    mkl_out_shape.SetTfLayout(mkl_context.params.in_dim,
                              mkl_context.params.in_sizes,
                              mkl_context.params.in_strides);
    mkl_out_shape.SetTfDimOrder(mkl_context.params.in_dim, data_format_);

    tensor_out_shape.AddDim(dnnLayoutGetMemorySize_F32(static_cast<dnnLayout_t>(
                                mkl_out_shape.GetMklLayout())) /
                            sizeof(T));

    AllocateOutputSetMklShape(context, 0, &output, tensor_out_shape,
                              mkl_out_shape);

    // Set output tensor.
    mkl_context.pooling_res[dnnResourceDiffSrc] =
        static_cast<void*>(output->flat<T>().data());

    // Execute primitive.
    CHECK_EQ(
        dnnExecute_F32(mkl_context.prim_pooling_bwd, mkl_context.pooling_res),
        E_SUCCESS);

    mkl_context.MklCleanup();
  }

 private:
  typedef struct {
    MklPoolingOpParams params;
    MklShape out_backprop_shape;
    dnnPrimitive_t prim_pooling_bwd = nullptr, convert_outbackprop = nullptr;
    void* pooling_res[dnnResourceNumber];
    dnnLayout_t lt_user_input = nullptr, lt_user_outbackprop = nullptr,
                lt_prim_outbackprop = nullptr, lt_workspace = nullptr;

    void MklCreateLayoutsAndPrimitives(OpKernelContext* context) {
      const Tensor& tensor_in_shape = MklGetInput(context, 0);
      const Tensor& out_backprop = MklGetInput(context, 1);
      bool outbackprop_in_mkl_format = out_backprop_shape.IsMklTensor();

      if (!outbackprop_in_mkl_format) {
        // For avgpooling, tensor_in_shape should have 1 dimension, and 4
        // elements.
        OP_REQUIRES(
            context,
            tensor_in_shape.dims() == 1 && tensor_in_shape.NumElements() == 4,
            errors::InvalidArgument("original input shape must be "
                                    "1-dimensional and 4 elements"));

        // For avgpooling, out_backprop should have 4 dimensions.
        OP_REQUIRES(context, out_backprop.dims() == 4,
                    errors::InvalidArgument("out_backprop must be "
                                            "4-dimensional"));
      } else {
        // Input in MKL format.
        // For avgpooling, out_backprop should have 4 dimensions.
        OP_REQUIRES(context, out_backprop_shape.GetDimension() == 4,
                    errors::InvalidArgument("out_backprop must be "
                                            "4-dimensional"));
      }

      // TODO(inteltf): Get outbackprop layout.
      // Do we need to create layout in every invocation?
      if (!outbackprop_in_mkl_format) {
        CHECK_EQ(dnnLayoutCreate_F32(&lt_user_outbackprop, params.in_dim,
                                     params.out_sizes, params.out_strides),
                 E_SUCCESS);
      } else {
        lt_user_outbackprop = (dnnLayout_t)out_backprop_shape.GetCurLayout();
      }

      // Create the backward primitive
      // Create DNN user layout
      CHECK_EQ(dnnLayoutCreate_F32(&lt_user_input, params.in_dim,
                                   params.in_sizes, params.in_strides),
               E_SUCCESS);

      // Create PoolingBackward primitive
      dnnAlgorithm_t algorithm = dnnAlgorithmPoolingAvg;
      dnnPrimitiveAttributes_t primAttr = nullptr;
      CHECK_EQ(dnnPoolingCreateBackward_F32(
                   &prim_pooling_bwd, primAttr, algorithm, lt_user_input,
                   params.kernel_size, params.kernel_stride, params.in_offset,
                   dnnBorderZerosAsymm),
               E_SUCCESS);

      // Create expected outbackprop layout from the primitive.
      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(
                   &lt_prim_outbackprop, prim_pooling_bwd, dnnResourceDiffDst),
               E_SUCCESS);

      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&lt_workspace, prim_pooling_bwd,
                                                dnnResourceWorkspace),
               E_SUCCESS);
    }

    void MklCleanup() {
      bool outbackprop_in_mkl_format = out_backprop_shape.IsMklTensor();
      CHECK_EQ(dnnDelete_F32(prim_pooling_bwd), E_SUCCESS);
      CHECK_EQ(dnnLayoutDelete_F32(lt_user_input), E_SUCCESS);
      if (!outbackprop_in_mkl_format) {
        CHECK_EQ(dnnLayoutDelete_F32(lt_user_outbackprop), E_SUCCESS);
      }
      CHECK_EQ(dnnLayoutDelete_F32(lt_prim_outbackprop), E_SUCCESS);
      CHECK_EQ(dnnLayoutDelete_F32(lt_workspace), E_SUCCESS);
    }
  } MklAvgPoolingGradOpContext;

  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};  // MklAvgPoolingGradOp



#else


template <typename Device, typename T>
class MklAvgPoolingOp : public MklPoolingForwardOpBase<T> {
 public:
  explicit MklAvgPoolingOp(OpKernelConstruction* context)
      : MklPoolingForwardOpBase<T>(context) {
    // Workspace is an MKLDNN construct that is only used in Max Pooling.
    // So set workspace_enabled_ to false.
    this->workspace_enabled_ = false;
  }

  void Compute(OpKernelContext* context) override {
    try {
      auto cpu_engine = engine(engine::cpu, 0);
      const Tensor& input_tensor =
          MklGetInput(context, this->kInputTensorIndexInput);
      MklDnnShape dnn_shape_input;
      GetMklShape(context, this->kInputTensorIndexInput, &dnn_shape_input);
      this->SanityCheckInput(context, input_tensor, dnn_shape_input);
      if (!context->status().ok()) return;

      MklDnnData<T> dnn_data_input(&cpu_engine);
      MklDnnData<T> dnn_data_output(&cpu_engine);

      // initialize variables for the pooling op
      MklPoolParameters pool_params;
      // Get the input tensor and initialize the pooling parameters
      this->ConfigureInput(context, dnn_shape_input, input_tensor, &pool_params,
                           &dnn_data_input);
      OP_REQUIRES_OK(context, context->status());

      // Declare output tensor
      Tensor* output_tensor = nullptr;
      memory::dims output_dims_mkl_order;
      this->GetOutputDims(pool_params, &output_dims_mkl_order);

      // If input is an empty tensor, allocate an empty output tensor and return
      if (input_tensor.NumElements() == 0) {
        MklDnnShape output_mkl_shape;
        output_mkl_shape.SetMklTensor(false);
        TensorShape output_tf_shape;
        if (pool_params.data_format == TensorFormat::FORMAT_NCHW) {
          output_tf_shape = MklDnnDimsToTFShape(output_dims_mkl_order);
        } else {
          memory::dims output_dims_NHWC_order;
          output_dims_NHWC_order = {pool_params.tensor_in_batch,
                                    static_cast<int>(pool_params.out_height),
                                    static_cast<int>(pool_params.out_width),
                                    pool_params.out_depth};
          output_tf_shape = MklDnnDimsToTFShape(output_dims_NHWC_order);
        }
        const int kOutputIndex = 0;
        AllocateOutputSetMklShape(context, kOutputIndex, &output_tensor,
                                    output_tf_shape, output_mkl_shape);
        CHECK_NOTNULL(output_tensor);
        return;
      }

      // If input is in Mkl layout, then just get the memory format from it
      // directly, instead of using input data_format to AvgPool.
      if (dnn_shape_input.IsMklTensor()) {
        dnn_data_output.SetUsrMem(
            output_dims_mkl_order,
            static_cast<memory::format>(
                dnn_data_input.GetUsrMemDesc().data.format));

      } else {
        dnn_data_output.SetUsrMem(output_dims_mkl_order,
                                  this->data_format_mkldnn_);
      }

      // describe the memory layout
      dnn_data_output.SetOpMemDesc(output_dims_mkl_order, memory::format::any);

      // 3. create a pooling primitive descriptor
      auto pool_desc = pooling_forward::desc(
          prop_kind::forward, algorithm::pooling_avg_exclude_padding,
          dnn_data_input.GetUsrMemDesc(), dnn_data_output.GetUsrMemDesc(),
          memory::dims({pool_params.row_stride, pool_params.col_stride}),
          memory::dims({pool_params.window_rows, pool_params.window_cols}),
          memory::dims({static_cast<int>(pool_params.pad_top),
                        static_cast<int>(pool_params.pad_left)}),
          memory::dims({static_cast<int>(pool_params.pad_bottom),
                        static_cast<int>(pool_params.pad_right)}),
          TFPaddingToMklDnnPadding(this->padding_));
      auto pool_prim_desc =
          pooling_forward::primitive_desc(pool_desc, cpu_engine);

      this->AllocateOutputTensor(context, pool_prim_desc, output_dims_mkl_order,
                                 this->data_format_mkldnn_, &output_tensor);
      CHECK_NOTNULL(output_tensor);

      OP_REQUIRES_OK(context, context->status());
      dnn_data_output.SetUsrMemDataHandle(output_tensor);

      this->PrepareAndExecuteNet(pool_prim_desc, &dnn_data_input,
                                 &dnn_data_output);
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }  // Compute
};   // MklAvgPoolingOp

//-----------------------------------------------------------------------------

template <class Device, class T>
class MklAvgPoolingGradOp : public MklPoolingBackwardOpBase<T> {
 public:
  explicit MklAvgPoolingGradOp(OpKernelConstruction* context)
      : MklPoolingBackwardOpBase<T>(context) {}

  void Compute(OpKernelContext* context) override {
    try {
      auto cpu_engine = engine(engine::cpu, 0);
      MklDnnShape original_input_mkl_shape, input_gradient_mkl_shape;
      const Tensor& tensor_in_shape =
          MklGetInput(context, kInputTensorIndexInputShape);
      const Tensor& input_gradient_tensor =
          MklGetInput(context, kInputTensorIndexInputGradient);
      GetMklShape(context, kInputTensorIndexInputShape,
                  &original_input_mkl_shape);
      GetMklShape(context, kInputTensorIndexInputGradient,
                  &input_gradient_mkl_shape);

      SanityCheckInputs(context, tensor_in_shape, input_gradient_tensor,
                        original_input_mkl_shape, input_gradient_mkl_shape);
      if (!context->status().ok()) return;

      // Used to allocate output_diff_src/diff_src
      // and create pool_fwd mdm desc
      // 0. Input("orig_input_shape: int32") //NOT a T Tensor!
      // 1. Input("grad: T")

      MklDnnData<T> input_gradient_diff_dst(&cpu_engine);
      MklDnnData<T> output_diff_src(&cpu_engine);
      Tensor* output_tensor_diff_src = nullptr;
      TensorShape original_input_shape;
      MklPoolParameters pool_params;
      memory::dims output_dims_mkl_order, original_input_dims_nchw;
      // Configure the original input memory descriptor
      memory::desc original_input_md = ConfigureOriginalInput(
          context, tensor_in_shape, original_input_mkl_shape,
          &original_input_dims_nchw, &pool_params, &original_input_shape);

      // configure the original output memory descriptor
      // by definition, the shape of the original output is the same
      // as the shape of the gradient diff_dst
      memory::desc original_output_md = this->ConfigureOriginalOutput(
          pool_params, input_gradient_mkl_shape, output_dims_mkl_order);

      memory::desc target_diff_dst_md = this->ConfigureInputGradient(
          input_gradient_mkl_shape, input_gradient_tensor,
          &input_gradient_diff_dst, original_output_md);
      // The shape of the output diff src needs to be the same shape as the
      // original input. But we will set its format to be same as the format of
      // input gradient. We won't use format of original input since it will
      // always be in Tensorflow layout (given that AvgPoolGrad gets shape of
      // the input rather than actual input).
      output_diff_src.SetUsrMem(
          original_input_dims_nchw,
          static_cast<memory::format>(target_diff_dst_md.data.format));

      // Create the forward pooling primitive descriptor so we can reference it
      // in the backward pooling primitive descriptor
      auto pool_fwd_desc = pooling_forward::desc(
          prop_kind::forward, algorithm::pooling_avg_exclude_padding,
          original_input_md, original_output_md,
          memory::dims({pool_params.row_stride, pool_params.col_stride}),
          memory::dims({pool_params.window_rows, pool_params.window_cols}),
          memory::dims({static_cast<int>(pool_params.pad_top),
                        static_cast<int>(pool_params.pad_left)}),
          memory::dims({static_cast<int>(pool_params.pad_bottom),
                        static_cast<int>(pool_params.pad_right)}),
          TFPaddingToMklDnnPadding(this->padding_));
      auto pool_fwd_prim_desc =
          pooling_forward::primitive_desc(pool_fwd_desc, cpu_engine);

      auto pool_bkwd_desc = pooling_backward::desc(
          algorithm::pooling_avg_exclude_padding,
          output_diff_src.GetUsrMemDesc(), target_diff_dst_md,
          memory::dims({pool_params.row_stride, pool_params.col_stride}),
          memory::dims({pool_params.window_rows, pool_params.window_cols}),
          memory::dims({static_cast<int>(pool_params.pad_top),
                        static_cast<int>(pool_params.pad_left)}),
          memory::dims({static_cast<int>(pool_params.pad_bottom),
                        static_cast<int>(pool_params.pad_right)}),
          TFPaddingToMklDnnPadding(this->padding_));
      auto pool_bkwd_prim_desc = pooling_backward::primitive_desc(
          pool_bkwd_desc, cpu_engine, pool_fwd_prim_desc);
      this->AllocateOutputTensor(
          context, pool_bkwd_prim_desc, original_input_dims_nchw,
          this->data_format_mkldnn_, &output_tensor_diff_src);

      output_diff_src.SetUsrMemDataHandle(output_tensor_diff_src);

      this->PrepareAndExecuteNet(
          pool_bkwd_prim_desc, &input_gradient_diff_dst, &output_diff_src,
          memory::primitive_desc(target_diff_dst_md, cpu_engine));
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(context, errors::Aborted("Compute received an exception:",
                                              error_msg));
    }
  }  // Compute

 private:
  // 0. Input("orig_input_shape: int32")
  // 1. Input("grad: T")
  const int kInputTensorIndexInputShape = 0;
  const int kInputTensorIndexInputGradient = 1;

  memory::desc ConfigureOriginalInput(
      OpKernelContext* context, const Tensor& tensor_original_input_shape,
      const MklDnnShape& original_input_mkl_shape,
      memory::dims* original_input_dims_mkl_order,
      MklPoolParameters* pool_params, TensorShape* input_tensor_shape) {
    CHECK_NOTNULL(original_input_dims_mkl_order);
    CHECK_NOTNULL(pool_params);
    CHECK_NOTNULL(input_tensor_shape);
    // For AvgPoolGrad, we only get the size of the original input because
    // The original data is irrelvant.
    auto shape_vec = tensor_original_input_shape.vec<int32>();
    for (int64 i = 0; i < tensor_original_input_shape.NumElements(); ++i) {
      input_tensor_shape->AddDim(shape_vec(i));
    }

    return MklPoolingBackwardOpBase<T>::ConfigureOriginalInput(
        context, tensor_original_input_shape, original_input_mkl_shape,
        original_input_dims_mkl_order, pool_params, *input_tensor_shape);
  }

  void SanityCheckInputs(OpKernelContext* context,
                         const Tensor& tensor_in_shape,
                         const Tensor& input_gradient_tensor,
                         const MklDnnShape& original_input_mkl_shape,
                         const MklDnnShape& input_gradient_mkl_shape) {
    if (!original_input_mkl_shape.IsMklTensor()) {
      OP_REQUIRES(
          context,
          tensor_in_shape.dims() == 1 && tensor_in_shape.NumElements() == 4,
          errors::InvalidArgument("original input shape must be "
                                  "1-dimensional and 4 elements"));
    } else {
      OP_REQUIRES(context,
                  original_input_mkl_shape.GetDimension() == 1 &&
                      original_input_mkl_shape.DimSize(0) == 4,
                  errors::InvalidArgument("original input shape must be "
                                          "1-dimensional and 4 elements"));
    }

    if (!input_gradient_mkl_shape.IsMklTensor()) {
      // For avgpooling, input_gradient_diff_dst should have 4 dimensions.
      OP_REQUIRES(context, input_gradient_tensor.dims() == 4,
                  errors::InvalidArgument("Gradient shape must be "
                                          "4-dimensional"));
    } else {
      OP_REQUIRES(context, input_gradient_mkl_shape.GetDimension() == 4,
                  errors::InvalidArgument("Gradient shape must be "
                                          "4-dimensional"));
    }
  }
};  // MklAvgPoolingGradOp




#endif  // INTEL_MKL_ML


REGISTER_KERNEL_BUILDER(Name("_MklAvgPool")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T")
                            .Label(mkl_op_registry::kMklOpLabel),
                        MklAvgPoolingOp<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("_MklAvgPoolGrad")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T")
                            .Label(mkl_op_registry::kMklOpLabel),
                        MklAvgPoolingGradOp<CPUDevice, float>);

}  // namespace tensorflow
#endif  // INTEL_MKL
