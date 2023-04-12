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
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"

using dnnl::algorithm;
using dnnl::engine;
using dnnl::error;
using dnnl::memory;
using dnnl::pooling_backward;
using dnnl::pooling_forward;
using dnnl::prop_kind;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

// An implementation of MaxPooling (forward).
template <typename Device, typename T, bool native_format = false>
class MklMaxPoolingOp : public MklPoolingForwardOpBase<T> {
 public:
  explicit MklMaxPoolingOp(OpKernelConstruction* context)
      : MklPoolingForwardOpBase<T>(context) {
    // In Max Pooling, oneDNN does not allow passing workspace as nullptr.
    // So we set workspace_enabled_ to true.
    this->workspace_enabled_ = true;
    this->native_format_ = native_format;
  }

  void Compute(OpKernelContext* context) override {
    try {
      const Tensor& input_tensor =
          MklGetInput(context, this->kInputTensorIndexInput);
      MklDnnShape dnn_shape_input;
      GetMklShape(context, this->kInputTensorIndexInput, &dnn_shape_input,
                  this->native_format_);
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
      MklPoolingParams fwdParams(
          src_dims, output_dims_mkl_order, filter_dims, strides, padding_left,
          padding_right, dnnl::algorithm::pooling_max, pooling_prop_kind,
          static_cast<memory::format_tag>(this->data_format_mkldnn_), input_md,
          this->native_format_);
      MklDnnThreadPool eigen_tp(context);
      pooling_fwd = MklPoolingFwdPrimitiveFactory<T>::Get(fwdParams);
      // Allocate output tensor.
      this->AllocateOutputTensor(context, *(pooling_fwd->GetPoolingFwdPd()),
                                 output_dims_mkl_order,
                                 this->tensor_format_mkldnn_, &output_tensor);
      OP_REQUIRES_OK(context, context->status());
      dnn_data_output.SetUsrMem(pooling_fwd->GetPoolingFwdPd()->dst_desc(),
                                output_tensor);
      const T* src_data = input_tensor.flat<T>().data();

      T* dst_data = output_tensor->flat<T>().data();
      std::shared_ptr<stream> fwd_cpu_stream;

      fwd_cpu_stream.reset(CreateStream(&eigen_tp, pooling_fwd->GetEngine()));

      if (int8_forward_inference) {
        // Execute pooling op
        pooling_fwd->Execute(src_data, dst_data, nullptr, fwd_cpu_stream);

        // Pass min, max from input to output.
        const Tensor& min_input_t = MklGetInput(context, 1);
        const Tensor& max_input_t = MklGetInput(context, 2);
        OP_REQUIRES(
            context, TensorShapeUtils::IsScalar(min_input_t.shape()),
            errors::InvalidArgument(
                "min_input shape must be rank 0 but is rank ",
                min_input_t.dims(), ", received shape: ", min_input_t.shape()));
        OP_REQUIRES(
            context, TensorShapeUtils::IsScalar(max_input_t.shape()),
            errors::InvalidArgument(
                "max_input shape must be rank 0 but is rank ",
                max_input_t.dims(), ", received shape: ", max_input_t.shape()));
        const float min_input = min_input_t.scalar<float>()();
        const float max_input = max_input_t.scalar<float>()();

        Tensor* output_min = nullptr;
        Tensor* output_max = nullptr;
        MklDnnShape output_min_mkl_shape, output_max_mkl_shape;
        output_min_mkl_shape.SetMklTensor(false);
        output_max_mkl_shape.SetMklTensor(false);
        AllocateOutputSetMklShape(context, 1, &output_min, {},
                                  output_min_mkl_shape, this->native_format_);
        AllocateOutputSetMklShape(context, 2, &output_max, {},
                                  output_max_mkl_shape, this->native_format_);
        output_min->scalar<float>()() = min_input;
        output_max->scalar<float>()() = max_input;
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
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(context, errors::Aborted("Compute received an exception:",
                                              error_msg));
    }
  }

 private:
  const int kOutputTensorIndexWorkspace = 1;
  engine cpu_engine_ = engine(engine::kind::cpu, 0);

  void AllocateWorkspaceTensor(
      OpKernelContext* context,
      const pooling_forward::primitive_desc& pool_fwd_prim_desc,
      MklDnnData<uint8>* dnn_data_wksp) {
    DCHECK(dnn_data_wksp);
    Tensor* workspace_tensor = nullptr;
    memory::desc workspace_pd = pool_fwd_prim_desc.workspace_desc();
    size_t workspace_bytes = workspace_pd.get_size();
    MklDnnShape workspace_mkl_shape;
    workspace_mkl_shape.SetMklTensor(false);
    TensorShape workspace_tf_shape;
    workspace_tf_shape.AddDim(workspace_bytes);
    AllocateOutputSetMklShape(context, kOutputTensorIndexWorkspace,
                              &workspace_tensor, workspace_tf_shape,
                              workspace_mkl_shape, this->native_format_);
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
template <class Device, class T, bool native_format = false>
class MklMaxPoolingGradOp : public MklPoolingBackwardOpBase<T> {
 public:
  explicit MklMaxPoolingGradOp(OpKernelConstruction* context)
      : MklPoolingBackwardOpBase<T>(context) {
    this->native_format_ = native_format;
  }
  void Compute(OpKernelContext* context) override {
    try {
      const Tensor& orig_input_tensor =
          MklGetInput(context, kInputTensorIndexOrigInput);
      const Tensor& grad_tensor =
          MklGetInput(context, kInputTensorIndexGradient);
      const Tensor& workspace_tensor =
          MklGetInput(context, kInputTensorIndexWorkspace);
      MklDnnShape orig_input_mkl_shape, grad_mkl_shape;
      GetMklShape(context, kInputTensorIndexOrigInput, &orig_input_mkl_shape,
                  this->native_format_);
      GetMklShape(context, kInputTensorIndexGradient, &grad_mkl_shape,
                  this->native_format_);
      if (!context->status().ok()) return;

      MklDnnData<T> grad_dnn_data(&cpu_engine_);
      MklDnnData<uint8> workspace_dnn_data(&cpu_engine_);

      MklPoolParameters pool_params;
      TensorShape orig_input_shape = orig_input_tensor.shape();

      if (orig_input_tensor.NumElements() == 0 ||
          grad_tensor.NumElements() == 0) {
        Tensor* output = nullptr;
        TensorShape output_shape;
        auto shape_vec = orig_input_tensor.vec<int32>();
        for (int64_t i = 0; i < orig_input_tensor.NumElements(); ++i) {
          OP_REQUIRES_OK(context, output_shape.AddDimWithStatus(shape_vec(i)));
        }
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, output_shape, &output));
        output->flat<T>().setZero();
        return;
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

      // Get src mem desc
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

      MklPoolingParams bwdParams(
          orig_input_dims_mkl_order, output_dims_mkl_order, filter_dims,
          strides, padding_left, padding_right, dnnl::algorithm::pooling_max,
          prop_kind::forward_training,
          static_cast<memory::format_tag>(this->data_format_mkldnn_), src_md,
          this->native_format_);
      MklDnnThreadPool eigen_tp(context);
      MklPoolingBwdPrimitive<T>* pooling_bwd =
          MklPoolingBwdPrimitiveFactory<T>::Get(bwdParams);

      std::shared_ptr<stream> bwd_cpu_stream;

      bwd_cpu_stream.reset(CreateStream(&eigen_tp, pooling_bwd->GetEngine()));
      // Allocate output tensor and memory primitive.
      Tensor* output_tensor = nullptr;
      this->AllocateOutputTensor(context, *(pooling_bwd->GetPoolingBwdPd()),
                                 orig_input_dims_mkl_order,
                                 this->tensor_format_mkldnn_, &output_tensor);

      // Check if diff_dst needs to be reordered.
      std::shared_ptr<PoolingBwdPd> pooling_bwd_pd =
          pooling_bwd->GetPoolingBwdPd();
      T* diff_dst_data = nullptr;
      if (!this->native_format_ &&
          (diff_dst_md != pooling_bwd_pd->diff_dst_desc())) {
        grad_dnn_data.SetUsrMem(diff_dst_md, &grad_tensor);
        grad_dnn_data.CheckReorderToOpMem(pooling_bwd_pd->diff_dst_desc(),
                                          cpu_engine_, context);
        diff_dst_data =
            static_cast<T*>(grad_dnn_data.GetOpMem().get_data_handle());
      } else {
        diff_dst_data =
            static_cast<T*>(const_cast<T*>(grad_tensor.flat<T>().data()));
      }

      void* ws_data = static_cast<void*>(
          const_cast<uint8*>(workspace_tensor.flat<uint8>().data()));

      T* diff_src_data = output_tensor->flat<T>().data();

      // Execute pooling op.
      pooling_bwd->Execute(diff_dst_data, diff_src_data, ws_data,
                           bwd_cpu_stream);
    } catch (dnnl::error& e) {
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
  engine cpu_engine_ = engine(engine::kind::cpu, 0);
};  // MklMaxPoolingGradOp

#define REGISTER_MKL_MAXPOOL3D_KERNELS(T)                                     \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_MklMaxPool3D")                                                   \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),                \
      MklMaxPoolingOp<CPUDevice, T>);                                         \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_MklMaxPool3DGrad")                                               \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),                \
      MklMaxPoolingGradOp<CPUDevice, T>);                                     \
  REGISTER_KERNEL_BUILDER(Name("_MklNativeMaxPool3D")                         \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          MklMaxPoolingOp<CPUDevice, T, true>);               \
  REGISTER_KERNEL_BUILDER(Name("_MklNativeMaxPool3DGrad")                     \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          MklMaxPoolingGradOp<CPUDevice, T, true>);

TF_CALL_float(REGISTER_MKL_MAXPOOL3D_KERNELS);
TF_CALL_bfloat16(REGISTER_MKL_MAXPOOL3D_KERNELS);

#define REGISTER_MKL_MAXPOOL_KERNELS(T)                                       \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_MklMaxPool")                                                     \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),                \
      MklMaxPoolingOp<CPUDevice, T>);                                         \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_MklMaxPoolGrad")                                                 \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),                \
      MklMaxPoolingGradOp<CPUDevice, T>);                                     \
  REGISTER_KERNEL_BUILDER(Name("_MklNativeMaxPool")                           \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          MklMaxPoolingOp<CPUDevice, T, true>);               \
  REGISTER_KERNEL_BUILDER(Name("_MklNativeMaxPoolGrad")                       \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          MklMaxPoolingGradOp<CPUDevice, T, true>);

TF_CALL_float(REGISTER_MKL_MAXPOOL_KERNELS);
TF_CALL_bfloat16(REGISTER_MKL_MAXPOOL_KERNELS);

REGISTER_KERNEL_BUILDER(Name("_MklQuantizedMaxPool")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklMaxPoolingOp<CPUDevice, quint8, true>);

REGISTER_KERNEL_BUILDER(Name("_MklQuantizedMaxPool")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("T")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklMaxPoolingOp<CPUDevice, qint8, true>);

REGISTER_KERNEL_BUILDER(
    Name("_QuantizedMaxPool3D").Device(DEVICE_CPU).TypeConstraint<quint8>("T"),
    MklMaxPoolingOp<CPUDevice, quint8, true>);
REGISTER_KERNEL_BUILDER(
    Name("_QuantizedMaxPool3D").Device(DEVICE_CPU).TypeConstraint<qint8>("T"),
    MklMaxPoolingOp<CPUDevice, qint8, true>);
}  // namespace tensorflow

#endif  // INTEL_MKL
