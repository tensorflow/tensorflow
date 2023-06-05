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

#if defined(INTEL_MKL) && !defined(ENABLE_ONEDNN_V3)
#define EIGEN_USE_THREADS

#include "dnnl.hpp"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h"

using dnnl::algorithm;
using dnnl::engine;
using dnnl::error;
using dnnl::memory;
using dnnl::pooling_backward;
using dnnl::pooling_forward;
using dnnl::prop_kind;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T, bool native_format = false>
class MklAvgPoolingOp : public MklPoolingForwardOpBase<T> {
 public:
  explicit MklAvgPoolingOp(OpKernelConstruction* context)
      : MklPoolingForwardOpBase<T>(context) {
    // Workspace is an oneDNN construct that is only used in Max Pooling.
    // So set workspace_enabled_ to false.
    this->workspace_enabled_ = false;
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
      // Check for corner case - if output is an empty tensor, return.
      TensorShape out_tf_shape = MklDnnDimsToTFShape(output_dims_mkl_order);

      // If input is an empty tensor, allocate an empty output tensor.
      if (input_tensor.NumElements() == 0 || out_tf_shape.num_elements() == 0) {
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

      MklPoolingParams fwdParams(
          src_dims, output_dims_mkl_order, filter_dims, strides, padding_left,
          padding_right, dnnl::algorithm::pooling_avg_exclude_padding,
          pooling_prop_kind,
          static_cast<memory::format_tag>(this->data_format_mkldnn_), input_md,
          this->native_format_);
      MklDnnThreadPool eigen_tp(context);
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
      fwd_cpu_stream.reset(CreateStream(&eigen_tp, pooling_fwd->GetEngine()));
      // Execute pooling op.
      pooling_fwd->Execute(src_data, dst_data, nullptr, fwd_cpu_stream);

      // Pass min, max from input to output.
      if (int8_forward_inference) {
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
      }
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }  // Compute

 private:
  engine cpu_engine_ = engine(engine::kind::cpu, 0);
};  // MklAvgPoolingOp

template <class Device, class T, bool native_format = false>
class MklAvgPoolingGradOp : public MklPoolingBackwardOpBase<T> {
 public:
  explicit MklAvgPoolingGradOp(OpKernelConstruction* context)
      : MklPoolingBackwardOpBase<T>(context) {
    this->native_format_ = native_format;
  }

  void Compute(OpKernelContext* context) override {
    try {
      const Tensor& orig_input_tensor =
          MklGetInput(context, kInputTensorIndexInputShape);
      const Tensor& grad_tensor =
          MklGetInput(context, kInputTensorIndexInputGradient);

      // For empty tensor, avg_pool_3d_grad in oneDNN doesn't handle this case.
      // Follow what native TF does in this case.

      TensorShape output_shape;
      auto shape_vec = orig_input_tensor.vec<int32>();
      for (int64_t i = 0; i < orig_input_tensor.NumElements(); ++i) {
        OP_REQUIRES_OK(context, output_shape.AddDimWithStatus(shape_vec(i)));
      }
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, output_shape, &output_tensor));
      output_tensor->flat<T>().setZero();

      bool is_pool2d = (this->ksize_.size() == 4);

      // out-of-memory boundary index check for output_tensor in 2D case.
      const int depth_window = this->ksize_[3];
      if (is_pool2d && depth_window == 1) {
        const int window_rows = this->ksize_[1];
        const int window_cols = this->ksize_[2];
        const int row_stride = this->stride_[1];
        const int col_stride = this->stride_[2];
        const int64_t in_rows = output_shape.dim_size(1);
        const int64_t in_cols = output_shape.dim_size(2);
        const int64_t out_backprop_batch = grad_tensor.dim_size(0);
        const int64_t out_backprop_rows = grad_tensor.dim_size(1);
        const int64_t out_backprop_cols = grad_tensor.dim_size(2);
        const int64_t out_backprop_depth = grad_tensor.dim_size(3);
        int64_t out_height, out_width, pad_rows, pad_cols;
        OP_REQUIRES_OK(context, GetWindowedOutputSize(
                                    in_rows, window_rows, row_stride,
                                    this->padding_, &out_height, &pad_rows));

        OP_REQUIRES_OK(context, GetWindowedOutputSize(
                                    in_cols, window_cols, col_stride,
                                    this->padding_, &out_width, &pad_cols));

        for (int64_t r = 0; r < out_backprop_rows; ++r) {
          int rindex, rsize;
          OP_REQUIRES_OK(context,
                         GetBroadcastSize(r, in_rows, window_rows, row_stride,
                                          pad_rows, &rindex, &rsize));
          for (int64_t c = 0; c < out_backprop_cols; ++c) {
            int cindex, csize;
            OP_REQUIRES_OK(context,
                           GetBroadcastSize(c, in_cols, window_cols, col_stride,
                                            pad_cols, &cindex, &csize));
            int64_t input_max =
                ((out_backprop_batch - 1) * in_rows + rindex + rsize - 1) *
                    in_cols +
                cindex + csize - 1;
            OP_REQUIRES(context, input_max < output_tensor->NumElements(),
                        errors::InvalidArgument(
                            "Output only has ", output_tensor->NumElements(),
                            " elements but computation requested"
                            " would use element with index=",
                            input_max));
          }
        }
      }

      if (output_shape.num_elements() == 0 || grad_tensor.NumElements() == 0) {
        return;
      }
      MklDnnShape orig_input_mkl_shape, grad_mkl_shape;
      GetMklShape(context, kInputTensorIndexInputShape, &orig_input_mkl_shape,
                  this->native_format_);
      GetMklShape(context, kInputTensorIndexInputGradient, &grad_mkl_shape,
                  this->native_format_);
      if (!context->status().ok()) return;

      // Used to allocate output_diff_src/diff_src.
      MklDnnData<T> grad_dnn_data(&cpu_engine_);
      MklPoolParameters pool_params;
      this->InitMklPoolParameters(context, &pool_params, orig_input_mkl_shape,
                                  output_shape);

      memory::dims filter_dims, strides, padding_left, padding_right;
      this->PoolParamsToDims(&pool_params, &filter_dims, &strides,
                             &padding_left, &padding_right, is_pool2d);

      memory::dims orig_input_dims_mkl_order =
          orig_input_mkl_shape.IsMklTensor()
              ? orig_input_mkl_shape.GetSizesAsMklDnnDims()
          : is_pool2d
              ? TFShapeToMklDnnDimsInNCHW(output_shape, this->data_format_tf_)
              : TFShapeToMklDnnDimsInNCDHW(output_shape, this->data_format_tf_);

      memory::dims diff_dst_dims =
          grad_mkl_shape.IsMklTensor()
              ? grad_mkl_shape.GetSizesAsMklDnnDims()
              : is_pool2d ? TFShapeToMklDnnDimsInNCHW(grad_tensor.shape(),
                                                      this->data_format_tf_)
                          : TFShapeToMklDnnDimsInNCDHW(grad_tensor.shape(),
                                                       this->data_format_tf_);

      OP_REQUIRES(
          context, orig_input_dims_mkl_order[0] == diff_dst_dims[0],
          errors::InvalidArgument(
              "Expected first dimension of orig_input and diff_dst to match, "
              "got ",
              orig_input_dims_mkl_order[0], " and ", diff_dst_dims[0]));

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
      MklPoolingParams bwdParams(
          orig_input_dims_mkl_order, output_dims_mkl_order, filter_dims,
          strides, padding_left, padding_right,
          dnnl::algorithm::pooling_avg_exclude_padding,
          prop_kind::forward_training,
          static_cast<memory::format_tag>(this->data_format_mkldnn_), src_md,
          this->native_format_);
      MklDnnThreadPool eigen_tp(context);
      MklPoolingBwdPrimitive<T>* pooling_bwd =
          MklPoolingBwdPrimitiveFactory<T>::Get(bwdParams);

      std::shared_ptr<stream> bwd_cpu_stream;
      bwd_cpu_stream.reset(CreateStream(&eigen_tp, pooling_bwd->GetEngine()));
      // TODO(intel-tf): Refactor (lines 249-262) common code for
      // max & avg pooling into superclass or common utils function.
      // Check whether we need to reorder diff_dst.
      std::shared_ptr<PoolingBwdPd> pooling_bwd_pd =
          pooling_bwd->GetPoolingBwdPd();
      T* diff_dst_data = nullptr;
      if (!this->native_format_ &&
          (diff_dst_md != pooling_bwd_pd->diff_dst_desc())) {
        grad_dnn_data.SetUsrMem(diff_dst_md, &grad_tensor);
        grad_dnn_data.CheckReorderToOpMem(pooling_bwd_pd->diff_dst_desc(),
                                          cpu_engine_);
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
    } catch (dnnl::error& e) {
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
  engine cpu_engine_ = engine(engine::kind::cpu, 0);
};  // MklAvgPoolingGradOp

#define REGISTER_MKL_AVGPOOL3D_KERNELS(T)                                     \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_MklAvgPool3D")                                                   \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),                \
      MklAvgPoolingOp<CPUDevice, T>);                                         \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_MklAvgPool3DGrad")                                               \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),                \
      MklAvgPoolingGradOp<CPUDevice, T>);                                     \
  REGISTER_KERNEL_BUILDER(Name("_MklNativeAvgPool3D")                         \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          MklAvgPoolingOp<CPUDevice, T, true>);               \
  REGISTER_KERNEL_BUILDER(Name("_MklNativeAvgPool3DGrad")                     \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          MklAvgPoolingGradOp<CPUDevice, T, true>);

TF_CALL_float(REGISTER_MKL_AVGPOOL3D_KERNELS);
TF_CALL_bfloat16(REGISTER_MKL_AVGPOOL3D_KERNELS);
#undef REGISTER_MKL_AVGPOOL3D_KERNELS

#define REGISTER_MKL_AVGPOOL_KERNELS(T)                                       \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_MklAvgPool")                                                     \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),                \
      MklAvgPoolingOp<CPUDevice, T>);                                         \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_MklAvgPoolGrad")                                                 \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),                \
      MklAvgPoolingGradOp<CPUDevice, T>);                                     \
  REGISTER_KERNEL_BUILDER(Name("_MklNativeAvgPool")                           \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          MklAvgPoolingOp<CPUDevice, T, true>);               \
  REGISTER_KERNEL_BUILDER(Name("_MklNativeAvgPoolGrad")                       \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          MklAvgPoolingGradOp<CPUDevice, T, true>);

TF_CALL_float(REGISTER_MKL_AVGPOOL_KERNELS);
TF_CALL_bfloat16(REGISTER_MKL_AVGPOOL_KERNELS);
#undef REGISTER_MKL_AVGPOOL_KERNELS

REGISTER_KERNEL_BUILDER(Name("_MklQuantizedAvgPool")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklAvgPoolingOp<CPUDevice, quint8, true>);

REGISTER_KERNEL_BUILDER(Name("_MklQuantizedAvgPool")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("T")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklAvgPoolingOp<CPUDevice, qint8, true>);

}  // namespace tensorflow

#endif  // INTEL_MKL && !ENABLE_ONEDNN_V3
