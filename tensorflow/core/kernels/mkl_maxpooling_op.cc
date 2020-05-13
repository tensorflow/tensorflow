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
#include "tensorflow/core/util/mkl_types.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/padding.h"

using mkldnn::algorithm;
using mkldnn::engine;
using mkldnn::error;
using mkldnn::memory;
#ifndef ENABLE_MKLDNN_V1
using mkldnn::padding_kind;
#endif
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
    // In Max Pooling, MKL-DNN does not allow passing workspace as nullptr.
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

      MklDnnData<T> dnn_data_input(&cpu_engine_);
      MklDnnData<T> dnn_data_output(&cpu_engine_);

      // Initialize variables for the pooling op.
      MklPoolParameters pool_params;
      // Check whether pooling is 2D or 3D.
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
#ifdef ENABLE_MKLDNN_V1
      // TODO(DNNL): Figure out what should be used for input_md.data.format
      MklPoolingParams fwdParams(
          src_dims, output_dims_mkl_order, filter_dims, strides, padding_left,
          padding_right, ALGORITHM::pooling_max, pooling_prop_kind,
          static_cast<MEMORY_FORMAT>(this->data_format_mkldnn_), input_md);
#else
      MklPoolingParams fwdParams(
          src_dims, output_dims_mkl_order, filter_dims, strides, padding_left,
          padding_right, ALGORITHM::pooling_max, pooling_prop_kind,
          static_cast<MEMORY_FORMAT>(input_md.data.format), input_md);
#endif
      pooling_fwd = MklPoolingFwdPrimitiveFactory<T>::Get(fwdParams);
      // Allocate output tensor.
      this->AllocateOutputTensor(context, *(pooling_fwd->GetPoolingFwdPd()),
                                 output_dims_mkl_order,
                                 this->tensor_format_mkldnn_, &output_tensor);
      OP_REQUIRES_OK(context, context->status());
#ifndef ENABLE_MKLDNN_V1
      dnn_data_output.SetUsrMem(output_dims_mkl_order,
                                this->data_format_mkldnn_, output_tensor);
#else
      dnn_data_output.SetUsrMem(
          GET_DST_DESC_FROM_OP_PD(pooling_fwd->GetPoolingFwdPd()),
          output_tensor);
#endif  // !ENABLE_MKLDNN_V1
      const T* src_data = input_tensor.flat<T>().data();

      T* dst_data = output_tensor->flat<T>().data();
      std::shared_ptr<stream> fwd_cpu_stream;
      fwd_cpu_stream.reset(CreateStream(context, pooling_fwd->GetEngine()));

      if (int8_forward_inference) {
        // Execute pooling op
        pooling_fwd->Execute(src_data, dst_data, nullptr, fwd_cpu_stream);

        // Pass min, max from input to output.
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
        MklDnnData<uint8> dnn_data_wksp(&cpu_engine_);
        AllocateWorkspaceTensor(context, *(pooling_fwd->GetPoolingFwdPd()),
                                &dnn_data_wksp);
        OP_REQUIRES_OK(context, context->status());
        T* ws_data =
            static_cast<T*>(dnn_data_wksp.GetOpMem().get_data_handle());
        // Execute pooling op.
        pooling_fwd->Execute(src_data, dst_data, ws_data, fwd_cpu_stream);
      }
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) + ", message: " +
                         string(e.message) + ", in file " + string(__FILE__) +
                         ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(context, errors::Aborted("Compute received an exception:",
                                              error_msg));
    }
  }

 private:
  const int kOutputTensorIndexWorkspace = 1;
  engine cpu_engine_ = engine(ENGINE_CPU, 0);

  void AllocateWorkspaceTensor(
      OpKernelContext* context,
      const pooling_forward::primitive_desc& pool_fwd_prim_desc,
      MklDnnData<uint8>* dnn_data_wksp) {
    DCHECK(dnn_data_wksp);
    Tensor* workspace_tensor = nullptr;
    MEMORY_PRIMITIVE_DESC workspace_pd =
        pool_fwd_prim_desc.PRIMITIVE_DESC_WORKSPACE;
    size_t workspace_bytes = workspace_pd.get_size();
    MklDnnShape workspace_mkl_shape;
    workspace_mkl_shape.SetMklTensor(false);
    TensorShape workspace_tf_shape;
    workspace_tf_shape.AddDim(workspace_bytes);
    AllocateOutputSetMklShape(context, kOutputTensorIndexWorkspace,
                              &workspace_tensor, workspace_tf_shape,
                              workspace_mkl_shape);
    DCHECK(workspace_tensor);
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

      MklDnnData<T> grad_dnn_data(&cpu_engine_);
      MklDnnData<uint8> workspace_dnn_data(&cpu_engine_);

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

      // get src mem desc
      memory::desc src_md =
          orig_input_mkl_shape.IsMklTensor()
              ? orig_input_mkl_shape.GetMklLayout()
              : memory::desc(orig_input_dims_mkl_order, MklDnnType<T>(),
                             this->data_format_mkldnn_);

      // Get diff_dst memory descriptor.
      memory::desc diff_dst_md =
          grad_mkl_shape.IsMklTensor()
              ? grad_mkl_shape.GetMklLayout()
              : memory::desc(diff_dst_dims, MklDnnType<T>(),
                             this->data_format_mkldnn_);

#ifdef ENABLE_MKLDNN_V1
      // TODO(DNNL): Find out what should be used for src_md.data.format.
      MklPoolingParams bwdParams(
          orig_input_dims_mkl_order, output_dims_mkl_order, filter_dims,
          strides, padding_left, padding_right, ALGORITHM::pooling_max,
          prop_kind::forward_training,
          static_cast<MEMORY_FORMAT>(this->data_format_mkldnn_), src_md,
          diff_dst_md);
#else
      MklPoolingParams bwdParams(
          orig_input_dims_mkl_order, output_dims_mkl_order, filter_dims,
          strides, padding_left, padding_right, ALGORITHM::pooling_max,
          prop_kind::forward_training,
          static_cast<MEMORY_FORMAT>(src_md.data.format), src_md);
#endif
      MklPoolingBwdPrimitive<T>* pooling_bwd =
          MklPoolingBwdPrimitiveFactory<T>::Get(bwdParams);

      std::shared_ptr<stream> bwd_cpu_stream;
      bwd_cpu_stream.reset(CreateStream(context, pooling_bwd->GetEngine()));
      // Allocate output tensor and memory primitive.
      Tensor* output_tensor = nullptr;
      this->AllocateOutputTensor(context, *(pooling_bwd->GetPoolingBwdPd()),
                                 orig_input_dims_mkl_order,
                                 this->tensor_format_mkldnn_, &output_tensor);

      // Check if diff_dst needs to be reordered.
      std::shared_ptr<PoolingBwdPd> pooling_bwd_pd =
          pooling_bwd->GetPoolingBwdPd();
      T* diff_dst_data = nullptr;
      if (IS_DIFF_DST_REORDER_NEEDED(diff_dst_md, pooling_bwd_pd,
                                     pooling_bwd)) {
        grad_dnn_data.SetUsrMem(diff_dst_md, &grad_tensor);
        grad_dnn_data.CheckReorderToOpMem(
            MEMORY_PD_WITHOUT_DATA(GET_DIFF_DST_DESC_FROM_OP_PD(pooling_bwd_pd),
                                   cpu_engine_),
            context);
        diff_dst_data =
            static_cast<T*>(grad_dnn_data.GetOpMem().get_data_handle());
      } else {
        diff_dst_data =
            static_cast<T*>(const_cast<T*>(grad_tensor.flat<T>().data()));
      }

      void* ws_data = static_cast<void*>(
          const_cast<uint8*>(workspace_tensor.flat<uint8>().data()));

#ifndef ENABLE_MKLDNN_V1
      auto ws_md =
          pooling_bwd->GetPoolingFwdPd()->PRIMITIVE_DESC_WORKSPACE.desc();
      if (ws_md.data.format != pooling_bwd->GetWorkspaceMemoryFormat()) {
        workspace_dnn_data.SetUsrMem(ws_md, &workspace_tensor);
        workspace_dnn_data.CheckReorderToOpMem(MEMORY_PD_WITHOUT_DATA(
            GET_WORKSPACE_DESC_FROM_OP_PD(pooling_bwd_pd), cpu_engine_));
        ws_data = workspace_dnn_data.GetOpMem().get_data_handle();
      }
#endif  // ENABLE_MKLDNN_V1

      T* diff_src_data = output_tensor->flat<T>().data();

      // Execute pooling op.
      pooling_bwd->Execute(diff_dst_data, diff_src_data, ws_data,
                           bwd_cpu_stream);
    } catch (mkldnn::error& e) {
      string error_msg = "Status:" + std::to_string(e.status) + ", message: " +
                         string(e.message) + ". in file " + string(__FILE__) +
                         ":" + std::to_string(__LINE__);
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
  engine cpu_engine_ = engine(ENGINE_CPU, 0);
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
