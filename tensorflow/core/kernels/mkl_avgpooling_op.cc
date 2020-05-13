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

#include "mkldnn.hpp"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/mkl_pooling_ops_common.h"
#include "tensorflow/core/util/mkl_types.h"
#include "tensorflow/core/util/mkl_util.h"

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

      // Initialize variables for the pooling op.
      MklPoolParameters pool_params;
      // Check whether pooling is 2D or 3D.
      bool is_pool2d = (this->ksize_.size() == 4);
      // Get the input tensor and initialize the pooling parameters.
      TensorShape input_tensor_shape = input_tensor.shape();
      this->InitMklPoolParameters(context, &pool_params, dnn_shape_input,
                                  input_tensor_shape);
      OP_REQUIRES_OK(context, context->status());

      Tensor* output_tensor = nullptr;
      memory::dims output_dims_mkl_order;
      this->GetOutputDims(pool_params, &output_dims_mkl_order);

      // If input is an empty tensor, allocate an empty output tensor.
      if (input_tensor.NumElements() == 0) {
        const int kOutputIndex = 0;
        this->AllocateEmptyOutputTensor(context, kOutputIndex, &pool_params,
                                        output_dims_mkl_order, &output_tensor);
        return;
      }

      memory::dims filter_dims, strides, padding_left, padding_right;
      // Get src/filter/stride/padding information.
      this->PoolParamsToDims(&pool_params, &filter_dims, &strides,
                             &padding_left, &padding_right, is_pool2d);

      // Get the input memory descriptor.
      memory::dims src_dims =
          dnn_shape_input.IsMklTensor()
              ? dnn_shape_input.GetSizesAsMklDnnDims()
              : is_pool2d ? TFShapeToMklDnnDimsInNCHW(input_tensor.shape(),
                                                      this->data_format_tf_)
                          : TFShapeToMklDnnDimsInNCDHW(input_tensor.shape(),
                                                       this->data_format_tf_);
      memory::desc input_md = dnn_shape_input.IsMklTensor()
                                  ? dnn_shape_input.GetMklLayout()
                                  : memory::desc(src_dims, MklDnnType<T>(),
                                                 this->data_format_mkldnn_);

      // Get an average pooling primitive from the op pool.
      MklPoolingFwdPrimitive<T>* pooling_fwd = nullptr;
      prop_kind pooling_prop_kind;
      bool int8_forward_inference =
          std::is_same<T, qint8>::value || std::is_same<T, quint8>::value;
      if (int8_forward_inference)
        pooling_prop_kind = prop_kind::forward_inference;
      else
        pooling_prop_kind = prop_kind::forward_training;
#ifdef ENABLE_MKLDNN_V1
      // TODO(DNNL): Find out what should we use input_md.data.format.
      MklPoolingParams fwdParams(
          src_dims, output_dims_mkl_order, filter_dims, strides, padding_left,
          padding_right, ALGORITHM::pooling_avg_exclude_padding,
          pooling_prop_kind,
          static_cast<MEMORY_FORMAT>(this->data_format_mkldnn_), input_md);
#else
      MklPoolingParams fwdParams(
          src_dims, output_dims_mkl_order, filter_dims, strides, padding_left,
          padding_right, ALGORITHM::pooling_avg_exclude_padding,
          pooling_prop_kind, static_cast<MEMORY_FORMAT>(input_md.data.format),
          input_md);
#endif
      pooling_fwd = MklPoolingFwdPrimitiveFactory<T>::Get(fwdParams);

      // Allocate output tensor.
      this->AllocateOutputTensor(context, *(pooling_fwd->GetPoolingFwdPd()),
                                 output_dims_mkl_order,
                                 this->tensor_format_mkldnn_, &output_tensor);
      DCHECK(output_tensor);
      OP_REQUIRES_OK(context, context->status());

      const T* src_data = input_tensor.flat<T>().data();

      T* dst_data = output_tensor->flat<T>().data();
      std::shared_ptr<stream> fwd_cpu_stream;
      fwd_cpu_stream.reset(CreateStream(context, pooling_fwd->GetEngine()));
      // Execute pooling op.
      pooling_fwd->Execute(src_data, dst_data, nullptr, fwd_cpu_stream);

      // Pass min, max from input to output.
      if (int8_forward_inference) {
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
      }
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) + ", message: " +
                         string(e.message) + ", in file " + string(__FILE__) +
                         ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }  // Compute

 private:
  engine cpu_engine_ = engine(ENGINE_CPU, 0);
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

      // Used to allocate output_diff_src/diff_src.
      MklDnnData<T> grad_dnn_data(&cpu_engine_);
      MklPoolParameters pool_params;
      auto shape_vec = orig_input_tensor.vec<int32>();
      TensorShape orig_input_shape;
      for (int i = 0; i < orig_input_tensor.NumElements(); i++) {
        orig_input_shape.AddDim(shape_vec(i));
      }

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

      // get src memory::desc
      memory::desc src_md =
          orig_input_mkl_shape.IsMklTensor()
              ? orig_input_mkl_shape.GetMklLayout()
              : memory::desc(orig_input_dims_mkl_order, MklDnnType<T>(),
                             this->data_format_mkldnn_);

      // Get diff_dst memory::desc.
      memory::desc diff_dst_md =
          grad_mkl_shape.IsMklTensor()
              ? grad_mkl_shape.GetMklLayout()
              : memory::desc(diff_dst_dims, MklDnnType<T>(),
                             this->data_format_mkldnn_);

// Pass prop_kind::forward_training to create a forward primitive
// that is used in the backward pass.
#ifdef ENABLE_MKLDNN_V1
      // TODO(DNNL): Find out what should we use src_md.data.format.
      MklPoolingParams bwdParams(
          orig_input_dims_mkl_order, output_dims_mkl_order, filter_dims,
          strides, padding_left, padding_right,
          ALGORITHM::pooling_avg_exclude_padding, prop_kind::forward_training,
          static_cast<MEMORY_FORMAT>(this->data_format_mkldnn_), src_md,
          diff_dst_md);
#else
      MklPoolingParams bwdParams(
          orig_input_dims_mkl_order, output_dims_mkl_order, filter_dims,
          strides, padding_left, padding_right,
          ALGORITHM::pooling_avg_exclude_padding, prop_kind::forward_training,
          static_cast<MEMORY_FORMAT>(src_md.data.format), src_md);
#endif
      MklPoolingBwdPrimitive<T>* pooling_bwd =
          MklPoolingBwdPrimitiveFactory<T>::Get(bwdParams);

      std::shared_ptr<stream> bwd_cpu_stream;
      bwd_cpu_stream.reset(CreateStream(context, pooling_bwd->GetEngine()));
      Tensor* output_tensor = nullptr;
      this->AllocateOutputTensor(context, *(pooling_bwd->GetPoolingBwdPd()),
                                 orig_input_dims_mkl_order,
                                 this->tensor_format_mkldnn_, &output_tensor);

      // TODO(nammbash): Refactor (lines 249-262) common code for
      // max & avg pooling into superclass or common utils function.
      // Check whether we need to reorder diff_dst.
      std::shared_ptr<PoolingBwdPd> pooling_bwd_pd =
          pooling_bwd->GetPoolingBwdPd();
      T* diff_dst_data = nullptr;
      if (IS_DIFF_DST_REORDER_NEEDED(diff_dst_md, pooling_bwd_pd,
                                     pooling_bwd)) {
        grad_dnn_data.SetUsrMem(diff_dst_md, &grad_tensor);
        grad_dnn_data.CheckReorderToOpMem(MEMORY_PD_WITHOUT_DATA(
            GET_DIFF_DST_DESC_FROM_OP_PD(pooling_bwd_pd), cpu_engine_));
        diff_dst_data =
            static_cast<T*>(grad_dnn_data.GetOpMem().get_data_handle());
      } else {
        diff_dst_data =
            static_cast<T*>(const_cast<T*>(grad_tensor.flat<T>().data()));
      }

      T* diff_src_data = output_tensor->flat<T>().data();

      // Execute pooling op.
      pooling_bwd->Execute(diff_dst_data, diff_src_data, nullptr,
                           bwd_cpu_stream);
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) + ", message: " +
                         string(e.message) + ", in file " + string(__FILE__) +
                         ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(context, errors::Aborted("Compute received an exception:",
                                              error_msg));
    }
  }

 private:
  // 0. Input("orig_input_shape: int32")
  // 1. Input("grad: T")
  const int kInputTensorIndexInputShape = 0;
  const int kInputTensorIndexInputGradient = 1;
  engine cpu_engine_ = engine(ENGINE_CPU, 0);
};  // MklAvgPoolingGradOp

#define REGISTER_MKL_AVGPOOL3D_KERNELS(T)                      \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklAvgPool3D")                                    \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklAvgPoolingOp<CPUDevice, T>);                          \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklAvgPool3DGrad")                                \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklAvgPoolingGradOp<CPUDevice, T>);

TF_CALL_float(REGISTER_MKL_AVGPOOL3D_KERNELS);
TF_CALL_bfloat16(REGISTER_MKL_AVGPOOL3D_KERNELS);

#define REGISTER_MKL_AVGPOOL_KERNELS(T)                        \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklAvgPool")                                      \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklAvgPoolingOp<CPUDevice, T>);                          \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklAvgPoolGrad")                                  \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklAvgPoolingGradOp<CPUDevice, T>);

TF_CALL_float(REGISTER_MKL_AVGPOOL_KERNELS);
TF_CALL_bfloat16(REGISTER_MKL_AVGPOOL_KERNELS);

REGISTER_KERNEL_BUILDER(Name("_MklQuantizedAvgPool")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklAvgPoolingOp<CPUDevice, quint8>);

REGISTER_KERNEL_BUILDER(Name("_MklQuantizedAvgPool")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("T")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklAvgPoolingOp<CPUDevice, qint8>);

}  // namespace tensorflow

#endif  // INTEL_MKL
