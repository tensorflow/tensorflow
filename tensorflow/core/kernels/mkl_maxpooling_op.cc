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

#include <algorithm>

#include "mkldnn.hpp"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/mkl_pooling_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/padding.h"

using mkldnn::algorithm;
using mkldnn::engine;
using mkldnn::error;
using mkldnn::memory;
using mkldnn::padding_kind;
using mkldnn::pooling_backward;
using mkldnn::pooling_forward;
using mkldnn::prop_kind;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

// An implementation of MaxPooling (forward).
template <typename Device, typename T>
class MklMaxPoolingOp : public MklPoolingForwardOpBase<T> {
 public:
  explicit MklMaxPoolingOp(OpKernelConstruction* context)
      : MklPoolingForwardOpBase<T>(context) {
    // In Max Pooling, MKLDNN does not allow passing workspace as NULL.
    // So we set workspace_enabled_ to true.
    this->workspace_enabled_ = true;
  }

  void Compute(OpKernelContext* context) override {
    try {
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
      // check whether pooling is 2D or 3D
      bool is_pool2d = (this->ksize_.size() == 4);
      // Get the input tensor and initialize the pooling parameters
      TensorShape input_tensor_shape = input_tensor.shape();
      this->InitMklPoolParameters(context, &pool_params, dnn_shape_input,
                                  input_tensor_shape);
      OP_REQUIRES_OK(context, context->status());

      // Declare output tensor
      Tensor* output_tensor = nullptr;
      // Declare output workspace tensor
      Tensor* output_ws_tensor = nullptr;
      memory::dims output_dims_mkl_order;
      this->GetOutputDims(pool_params, &output_dims_mkl_order);

      // If input is an empty tensor, allocate an empty output tensor and return
      if (input_tensor.NumElements() == 0) {
        const int kOutputIndex = 0;
        this->AllocateEmptyOutputTensor(context, kOutputIndex, &pool_params,
                                        output_dims_mkl_order, &output_tensor);
        bool int8_forward_inference =
            std::is_same<T, qint8>::value || std::is_same<T, quint8>::value;

        // Allocate an empty workspace tensor if not Quantized MaxPooling
        // Because Quantized MaxPooling does not have backward pass
        // Therefore no workspace, which is used to help backward pass in MKL
        if (!int8_forward_inference) {
          const int kOutputWorkspaceIndex = 1;
          // output_ws_tensor is not really used, so using output_dims_mkl_order
          this->AllocateEmptyOutputTensor(context, kOutputWorkspaceIndex,
                                          &pool_params, output_dims_mkl_order,
                                          &output_ws_tensor);
        }
        return;
      }

      // Get the input memory descriptor
      memory::desc input_md =
          dnn_shape_input.IsMklTensor()
              ? dnn_shape_input.GetMklLayout()
              : is_pool2d ? memory::desc(
                                TFShapeToMklDnnDimsInNCHW(
                                    input_tensor_shape, this->data_format_tf_),
                                MklDnnType<T>(), this->data_format_mkldnn_)
                          : memory::desc(
                                TFShapeToMklDnnDimsInNCDHW(
                                    input_tensor_shape, this->data_format_tf_),
                                MklDnnType<T>(), this->data_format_mkldnn_);

      // Get src/filter/stride/padding information
      memory::dims src_dims =
          dnn_shape_input.IsMklTensor()
              ? dnn_shape_input.GetSizesAsMklDnnDims()
              : is_pool2d ? TFShapeToMklDnnDimsInNCHW(input_tensor.shape(),
                                                      this->data_format_tf_)
                          : TFShapeToMklDnnDimsInNCDHW(input_tensor.shape(),
                                                       this->data_format_tf_);
      memory::dims filter_dims, strides, padding_left, padding_right;
      this->PoolParamsToDims(&pool_params, &filter_dims, &strides,
                             &padding_left, &padding_right, is_pool2d);

      // Get a pooling op from the cached pool
      MklPoolingFwdPrimitive<T>* pooling_fwd = nullptr;
      prop_kind pooling_prop_kind;
      bool int8_forward_inference =
          std::is_same<T, qint8>::value || std::is_same<T, quint8>::value;
      if (int8_forward_inference)
        pooling_prop_kind = prop_kind::forward_inference;
      else
        pooling_prop_kind = prop_kind::forward_training;
      MklPoolingParams fwdParams(src_dims, output_dims_mkl_order, filter_dims,
                                 strides, padding_left, padding_right,
                                 algorithm::pooling_max, pooling_prop_kind);
      pooling_fwd = MklPoolingFwdPrimitiveFactory<T>::Get(fwdParams);

      // allocate output tensor
      this->AllocateOutputTensor(context, *(pooling_fwd->GetPoolingFwdPd()),
                                 output_dims_mkl_order,
                                 this->data_format_mkldnn_, &output_tensor);
      OP_REQUIRES_OK(context, context->status());
      dnn_data_output.SetUsrMem(output_dims_mkl_order,
                                pooling_fwd->GetDstMemoryFormat(),
                                output_tensor);

      // check wehther we need to reorder src
      const T* src_data = input_tensor.flat<T>().data();
      if (input_md.data.format != pooling_fwd->GetSrcMemoryFormat()) {
        dnn_data_input.SetUsrMem(input_md, &input_tensor);
        auto src_target_primitive_desc = memory::primitive_desc(
            {{src_dims}, MklDnnType<T>(), pooling_fwd->GetSrcMemoryFormat()},
            cpu_engine);
        dnn_data_input.CheckReorderToOpMem(src_target_primitive_desc);
        src_data = const_cast<T*>(
            reinterpret_cast<T*>(dnn_data_input.GetOpMem().get_data_handle()));
      }

      T* dst_data = output_tensor->flat<T>().data();

      if (int8_forward_inference) {
        // Execute pooling op
        pooling_fwd->Execute(src_data, dst_data);

        // pass min, max from input to output
        const Tensor& min_input_t = MklGetInput(context, 1);
        const Tensor& max_input_t = MklGetInput(context, 2);
        const float min_input = min_input_t.flat<float>()(0);
        const float max_input = max_input_t.flat<float>()(0);

        Tensor* output_min = nullptr;
        Tensor* output_max = nullptr;
        MklDnnShape output_min_mkl_shape, output_max_mkl_shape;
        output_min_mkl_shape.SetMklTensor(false);
        output_max_mkl_shape.SetMklTensor(false);
        AllocateOutputSetMklShape(context, 1, &output_min, {},
                                  output_min_mkl_shape);
        AllocateOutputSetMklShape(context, 2, &output_max, {},
                                  output_max_mkl_shape);
        output_min->flat<float>()(0) = min_input;
        output_max->flat<float>()(0) = max_input;
      } else {
        MklDnnData<uint8> dnn_data_wksp(&cpu_engine);
        AllocateWorkspaceTensor(context, *(pooling_fwd->GetPoolingFwdPd()),
                                &dnn_data_wksp);
        OP_REQUIRES_OK(context, context->status());
        T* ws_data =
            static_cast<T*>(dnn_data_wksp.GetOpMem().get_data_handle());

        // execute pooling op
        pooling_fwd->Execute(src_data, dst_data, ws_data);
      }
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(context, errors::Aborted("Compute received an exception:",
                                              error_msg));
    }
  }

 private:
  const int kOutputTensorIndexWorkspace = 1;
  engine cpu_engine = engine(engine::cpu, 0);

  void AllocateWorkspaceTensor(
      OpKernelContext* context,
      const pooling_forward::primitive_desc& pool_fwd_prim_desc,
      MklDnnData<uint8>* dnn_data_wksp) {
    CHECK_NOTNULL(dnn_data_wksp);
    Tensor* workspace_tensor = nullptr;
    memory::primitive_desc workspace_pd =
        pool_fwd_prim_desc.workspace_primitive_desc();
    size_t workspace_bytes = workspace_pd.get_size();
    MklDnnShape workspace_mkl_shape;
    workspace_mkl_shape.SetMklTensor(false);
    TensorShape workspace_tf_shape;
    workspace_tf_shape.AddDim(workspace_bytes);
    AllocateOutputSetMklShape(context, kOutputTensorIndexWorkspace,
                              &workspace_tensor, workspace_tf_shape,
                              workspace_mkl_shape);
    CHECK_NOTNULL(workspace_tensor);
    dnn_data_wksp->SetUsrMem(workspace_pd, workspace_tensor);
  }
};

// The operation to compute MaxPool gradients.
// It takes three inputs:
//   - The original input tensor
//   - The original output tensor
//   - Backprop tensor for output
// It produces one output: backprop tensor for input.
template <class Device, class T>
class MklMaxPoolingGradOp : public MklPoolingBackwardOpBase<T> {
 public:
  explicit MklMaxPoolingGradOp(OpKernelConstruction* context)
      : MklPoolingBackwardOpBase<T>(context) {}
  void Compute(OpKernelContext* context) override {
    try {
      auto cpu_engine = engine(engine::cpu, 0);
      const Tensor& orig_input_tensor =
          MklGetInput(context, kInputTensorIndexOrigInput);
      const Tensor& grad_tensor =
          MklGetInput(context, kInputTensorIndexGradient);
      const Tensor& workspace_tensor =
          MklGetInput(context, kInputTensorIndexWorkspace);
      MklDnnShape orig_input_mkl_shape, grad_mkl_shape;
      GetMklShape(context, kInputTensorIndexOrigInput, &orig_input_mkl_shape);
      GetMklShape(context, kInputTensorIndexGradient, &grad_mkl_shape);
      if (!context->status().ok()) return;

      MklDnnData<T> grad_dnn_data(&cpu_engine);
      MklDnnData<uint8> workspace_dnn_data(&cpu_engine);

      MklPoolParameters pool_params;
      TensorShape orig_input_shape = orig_input_tensor.shape();

      bool is_pool2d = (this->ksize_.size() == 4);
      this->InitMklPoolParameters(context, &pool_params, orig_input_mkl_shape,
                                  orig_input_shape);

      memory::dims filter_dims, strides, padding_left, padding_right;
      this->PoolParamsToDims(&pool_params, &filter_dims, &strides,
                             &padding_left, &padding_right, is_pool2d);

      memory::dims orig_input_dims_mkl_order =
          orig_input_mkl_shape.IsMklTensor()
              ? orig_input_mkl_shape.GetSizesAsMklDnnDims()
              : is_pool2d ? TFShapeToMklDnnDimsInNCHW(orig_input_shape,
                                                      this->data_format_tf_)
                          : TFShapeToMklDnnDimsInNCDHW(orig_input_shape,
                                                       this->data_format_tf_);

      memory::dims diff_dst_dims =
          grad_mkl_shape.IsMklTensor()
              ? grad_mkl_shape.GetSizesAsMklDnnDims()
              : is_pool2d ? TFShapeToMklDnnDimsInNCHW(grad_tensor.shape(),
                                                      this->data_format_tf_)
                          : TFShapeToMklDnnDimsInNCDHW(grad_tensor.shape(),
                                                       this->data_format_tf_);

      memory::dims output_dims_mkl_order;
      this->GetOutputDims(pool_params, &output_dims_mkl_order);

      MklPoolingParams bwdParams(
          orig_input_dims_mkl_order, output_dims_mkl_order, filter_dims,
          strides, padding_left, padding_right, algorithm::pooling_max,
          prop_kind::forward_training);
      MklPoolingBwdPrimitive<T>* pooling_bwd =
          MklPoolingBwdPrimitiveFactory<T>::Get(bwdParams);

      // allocate output tensor and memory primitive
      Tensor* output_tensor = nullptr;
      this->AllocateOutputTensor(context, *(pooling_bwd->GetPoolingBwdPd()),
                                 orig_input_dims_mkl_order,
                                 this->data_format_mkldnn_, &output_tensor);
      // get diff_dst mem desc
      memory::desc diff_dst_md =
          grad_mkl_shape.IsMklTensor()
              ? grad_mkl_shape.GetMklLayout()
              : memory::desc(diff_dst_dims, MklDnnType<T>(),
                             this->data_format_mkldnn_);
      // check if diff_dst needs to be reordered
      const T* diff_dst_data = grad_tensor.flat<T>().data();
      if (diff_dst_md.data.format != pooling_bwd->GetDiffDstFormat()) {
        auto target_diff_dst = memory::primitive_desc(
            {{diff_dst_dims}, MklDnnType<T>(), pooling_bwd->GetDiffDstFormat()},
            cpu_engine);
        grad_dnn_data.SetUsrMem(diff_dst_md, &grad_tensor);
        grad_dnn_data.CheckReorderToOpMem(target_diff_dst);
        diff_dst_data = const_cast<T*>(
            reinterpret_cast<T*>(grad_dnn_data.GetOpMem().get_data_handle()));
      }

      void* ws_data = static_cast<void*>(
          const_cast<uint8*>(workspace_tensor.flat<uint8>().data()));

      auto ws_md =
          pooling_bwd->GetPoolingFwdPd()->workspace_primitive_desc().desc();
      if (ws_md.data.format != pooling_bwd->GetWorkspaceFormat()) {
        memory::dims ws_dims;
        ws_dims.assign(ws_md.data.dims, ws_md.data.dims + ws_md.data.ndims);
        auto target_ws =
            memory::primitive_desc({{ws_dims},
                                    pooling_bwd->GetWorkspaceDataType(),
                                    pooling_bwd->GetWorkspaceFormat()},
                                   cpu_engine);
        workspace_dnn_data.SetUsrMem(ws_md, &workspace_tensor);
        workspace_dnn_data.CheckReorderToOpMem(target_ws);
        ws_data = workspace_dnn_data.GetOpMem().get_data_handle();
      }

      T* diff_src_data = output_tensor->flat<T>().data();

      // execute pooling
      pooling_bwd->Execute(diff_dst_data, diff_src_data, ws_data);
    } catch (mkldnn::error& e) {
      string error_msg = "Status:" + std::to_string(e.status) +
                         ", message: " + string(e.message) + ". in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(context, errors::Aborted("Compute received an exception:",
                                              error_msg));
    }
  }

 private:
  // .Input("orig_input: T")
  // .Input("orig_output: T")
  // .Input("grad: T")
  // .Input("workspace: T")
  const int kInputTensorIndexOrigInput = 0;
  const int kInputTensorIndexOrigOutput = 1;
  const int kInputTensorIndexGradient = 2;
  const int kInputTensorIndexWorkspace = 3;

  void ConfigureWorkspace(const Tensor& workspace_tensor,
                          memory::primitive_desc workspace_pd,
                          MklDnnData<uint8>* workspace_dnn_data) {
    CHECK_NOTNULL(workspace_dnn_data);

    workspace_dnn_data->SetUsrMem(workspace_pd, &workspace_tensor);
  }

  void SanityCheckInputs(OpKernelContext* context,
                         const Tensor& orig_input_tensor,
                         const Tensor& orig_output_tensor,
                         const Tensor& grad_tensor,
                         const Tensor& workspace_tensor,
                         const MklDnnShape& orig_input_mkl_shape,
                         const MklDnnShape& orig_output_mkl_shape,
                         const MklDnnShape& grad_mkl_shape,
                         const MklDnnShape& workspace_mkl_shape) {
    if (!orig_input_mkl_shape.IsMklTensor()) {
      OP_REQUIRES(context, orig_input_tensor.dims() == 4,
                  errors::InvalidArgument(
                      "Original input shape must be 4-dimensional"));
    } else {
      OP_REQUIRES(context, orig_input_mkl_shape.GetDimension() == 4,
                  errors::InvalidArgument(
                      "Original input shape must be 4-dimensional"));
    }
    if (!orig_output_mkl_shape.IsMklTensor()) {
      OP_REQUIRES(
          context, orig_output_tensor.dims() == 4,
          errors::InvalidArgument("Original output must be 4-dimensional"));
    } else {
      OP_REQUIRES(
          context, orig_output_mkl_shape.GetDimension() == 4,
          errors::InvalidArgument("Original output must be 4-dimensional"));
    }
    if (!grad_mkl_shape.IsMklTensor()) {
      OP_REQUIRES(context, grad_tensor.dims() == 4,
                  errors::InvalidArgument("Gradient must be 4-dimensional"));
    } else {
      OP_REQUIRES(context, grad_mkl_shape.GetDimension() == 4,
                  errors::InvalidArgument("Gradient must be 4-dimensional"));
    }
    if (this->workspace_enabled_) {
      // The workspace should not be an MKL tensor
      OP_REQUIRES(context, workspace_mkl_shape.IsMklTensor() == false,
                  errors::InvalidArgument(
                      "Workspace tensor should not be an MKL Tensor."));
      // It should only have one dimension
      OP_REQUIRES(
          context, workspace_tensor.dims() == 1,
          errors::InvalidArgument("Workspace tensor must be 1-dimensional"));
    } else {
      OP_REQUIRES(
          context, this->workspace_enabled_,
          errors::Unimplemented("MKL-DNN Max Pooling does not "
                                "yet support the use case "
                                "where MaxPoolGrad is called without first"
                                " calling MaxPool."));
    }
  }
};  // MklMaxPoolingGradOp

#define REGISTER_MKL_MAXPOOL3D_KERNELS(T)                      \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklMaxPool3D")                                    \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklMaxPoolingOp<CPUDevice, T>);                          \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklMaxPool3DGrad")                                \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklMaxPoolingGradOp<CPUDevice, T>);

TF_CALL_float(REGISTER_MKL_MAXPOOL3D_KERNELS);
TF_CALL_bfloat16(REGISTER_MKL_MAXPOOL3D_KERNELS);

#define REGISTER_MKL_MAXPOOL_KERNELS(T)                        \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklMaxPool")                                      \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklMaxPoolingOp<CPUDevice, T>);                          \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklMaxPoolGrad")                                  \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklMaxPoolingGradOp<CPUDevice, T>);

TF_CALL_float(REGISTER_MKL_MAXPOOL_KERNELS);
TF_CALL_bfloat16(REGISTER_MKL_MAXPOOL_KERNELS);

REGISTER_KERNEL_BUILDER(Name("_MklQuantizedMaxPool")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklMaxPoolingOp<CPUDevice, quint8>);

REGISTER_KERNEL_BUILDER(Name("_MklQuantizedMaxPool")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("T")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklMaxPoolingOp<CPUDevice, qint8>);

}  // namespace tensorflow
#endif  // INTEL_MKL
