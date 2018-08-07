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
      const Tensor& input_tensor =
          MklGetInput(context, this->kInputTensorIndexInput);
      MklDnnShape dnn_shape_input;
      GetMklShape(context, this->kInputTensorIndexInput, &dnn_shape_input);
      this->SanityCheckInput(context, input_tensor, dnn_shape_input);
      if (!context->status().ok()) return;

      MklDnnData<T> dnn_data_input(&cpu_engine_);

      // initialize variables for the pooling op
      MklPoolParameters pool_params;
      // Get the input tensor and initialize the pooling parameters
      TensorShape input_tensor_shape = input_tensor.shape();
      this->InitMklPoolParameters(context, &pool_params, dnn_shape_input,
                                  input_tensor_shape);
      OP_REQUIRES_OK(context, context->status());

      // Declare output tensor
      Tensor* output_tensor = nullptr;
      memory::dims output_dims_mkl_order;
      this->GetOutputDims(pool_params, &output_dims_mkl_order);

      // If input is an empty tensor, allocate an empty output tensor and return
      if (input_tensor.NumElements() == 0) {
        const int kOutputIndex = 0;
        this->AllocateEmptyOutputTensor(context, kOutputIndex, &pool_params,
                                        output_dims_mkl_order, &output_tensor);
        return;
      }

      memory::dims filter_dims, strides, padding_left, padding_right;
      this->PoolParamsToDims(&pool_params, &filter_dims, &strides,
                             &padding_left, &padding_right);

      // Get the input memory descriptor
      memory::desc input_md =
          dnn_shape_input.IsMklTensor()
              ? dnn_shape_input.GetMklLayout()
              : memory::desc(TFShapeToMklDnnDimsInNCHW(input_tensor_shape,
                                                       this->data_format_tf_),
                             MklDnnType<T>(), this->data_format_mkldnn_);

      // Get src/filter/stride/padding information
      memory::dims src_dims =
          dnn_shape_input.IsMklTensor()
              ? dnn_shape_input.GetSizesAsMklDnnDims()
              : TFShapeToMklDnnDimsInNCHW(input_tensor.shape(),
                                          this->data_format_tf_);

      // Get an average pooling primitive from the op pool
      MklPoolingFwdPrimitive<T>* pooling_fwd = nullptr;
      MklPoolingParams fwdParams(src_dims, output_dims_mkl_order, filter_dims,
                                 strides, padding_left, padding_right,
                                 algorithm::pooling_avg_exclude_padding);
      pooling_fwd = MklPoolingFwdPrimitiveFactory<T>::Get(fwdParams);

      // allocate output tensor
      this->AllocateOutputTensor(context, *(pooling_fwd->GetPoolingFwdPd()),
                                 output_dims_mkl_order,
                                 this->data_format_mkldnn_, &output_tensor);
      CHECK_NOTNULL(output_tensor);

      OP_REQUIRES_OK(context, context->status());

      // check whether we need to reorder src
      const T* src_data = input_tensor.flat<T>().data();
      if (input_md.data.format != pooling_fwd->GetSrcMemoryFormat()) {
        dnn_data_input.SetUsrMem(input_md, &input_tensor);
        auto src_target_primitive_desc = memory::primitive_desc(
            {{src_dims}, MklDnnType<T>(), pooling_fwd->GetSrcMemoryFormat()},
            cpu_engine_);
        dnn_data_input.CheckReorderToOpMem(src_target_primitive_desc);
        src_data = const_cast<T*>(
            reinterpret_cast<T*>(dnn_data_input.GetOpMem().get_data_handle()));
      }

      T* dst_data = output_tensor->flat<T>().data();

      // execute pooling
      pooling_fwd->Execute(src_data, dst_data);
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }  // Compute

 private:
  engine cpu_engine_ = engine(engine::cpu, 0);
};  // MklAvgPoolingOp

template <class Device, class T>
class MklAvgPoolingGradOp : public MklPoolingBackwardOpBase<T> {
 public:
  explicit MklAvgPoolingGradOp(OpKernelConstruction* context)
      : MklPoolingBackwardOpBase<T>(context) {}

  void Compute(OpKernelContext* context) override {
    try {
      const Tensor& orig_input_tensor =
          MklGetInput(context, kInputTensorIndexInputShape);
      const Tensor& grad_tensor =
          MklGetInput(context, kInputTensorIndexInputGradient);

      MklDnnShape orig_input_mkl_shape, grad_mkl_shape;
      GetMklShape(context, kInputTensorIndexInputShape, &orig_input_mkl_shape);
      GetMklShape(context, kInputTensorIndexInputGradient, &grad_mkl_shape);
      if (!context->status().ok()) return;

      // Used to allocate output_diff_src/diff_src
      MklDnnData<T> grad_dnn_data(&cpu_engine_);
      MklPoolParameters pool_params;
      auto shape_vec = orig_input_tensor.vec<int32>();
      TensorShape orig_input_shape;
      for (int i = 0; i < orig_input_tensor.NumElements(); i++) {
        orig_input_shape.AddDim(shape_vec(i));
      }
      this->InitMklPoolParameters(context, &pool_params, orig_input_mkl_shape,
                                  orig_input_shape);

      memory::dims filter_dims, strides, padding_left, padding_right;
      this->PoolParamsToDims(&pool_params, &filter_dims, &strides,
                             &padding_left, &padding_right);

      memory::dims orig_input_dims_mkl_order =
          orig_input_mkl_shape.IsMklTensor()
              ? orig_input_mkl_shape.GetSizesAsMklDnnDims()
              : TFShapeToMklDnnDimsInNCHW(orig_input_shape,
                                          this->data_format_tf_);

      memory::dims diff_dst_dims =
          grad_mkl_shape.IsMklTensor()
              ? grad_mkl_shape.GetSizesAsMklDnnDims()
              : TFShapeToMklDnnDimsInNCHW(grad_tensor.shape(),
                                          this->data_format_tf_);
      memory::dims output_dims_mkl_order;
      this->GetOutputDims(pool_params, &output_dims_mkl_order);

      MklPoolingParams bwdParams(orig_input_dims_mkl_order,
                                 output_dims_mkl_order, filter_dims, strides,
                                 padding_left, padding_right,
                                 algorithm::pooling_avg_exclude_padding);
      MklPoolingBwdPrimitive<T>* pooling_bwd =
          MklPoolingBwdPrimitiveFactory<T>::Get(bwdParams);

      Tensor* output_tensor = nullptr;
      this->AllocateOutputTensor(context, *(pooling_bwd->GetPoolingBwdPd()),
                                 orig_input_dims_mkl_order,
                                 this->data_format_mkldnn_, &output_tensor);
      // get diff_dst memory::desc
      memory::desc diff_dst_md =
          grad_mkl_shape.IsMklTensor()
              ? grad_mkl_shape.GetMklLayout()
              : memory::desc(diff_dst_dims, MklDnnType<T>(),
                             this->data_format_mkldnn_);
      // Check whether we need to reorder diff_dst
      const T* diff_dst_data = grad_tensor.flat<T>().data();
      if (diff_dst_md.data.format != pooling_bwd->GetDiffDstFormat()) {
        auto target_diff_dst = memory::primitive_desc(
            {{diff_dst_dims}, MklDnnType<T>(), pooling_bwd->GetDiffDstFormat()},
            cpu_engine_);
        grad_dnn_data.SetUsrMem(diff_dst_md, &grad_tensor);
        grad_dnn_data.CheckReorderToOpMem(target_diff_dst);
        diff_dst_data = const_cast<T*>(
            reinterpret_cast<T*>(grad_dnn_data.GetOpMem().get_data_handle()));
      }

      T* diff_src_data = output_tensor->flat<T>().data();

      // execute pooling op
      pooling_bwd->Execute(diff_dst_data, diff_src_data);
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(context, errors::Aborted("Compute received an exception:",
                                              error_msg));
    }
  }

 private:
  // 0. Input("orig_input_shape: int32")
  // 1. Input("grad: T")
  const int kInputTensorIndexInputShape = 0;
  const int kInputTensorIndexInputGradient = 1;
  engine cpu_engine_ = engine(engine::cpu, 0);

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
