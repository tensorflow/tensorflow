/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "dnnl.hpp"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/tensor_format.h"

using namespace dnnl;
using dnnl::batch_normalization_forward;
using dnnl::prop_kind;
using dnnl::stream;
using CPUDevice = Eigen::ThreadPoolDevice;

namespace tensorflow {
template <typename Device, typename T>
class MklFusedInstanceNormOp : public OpKernel {
 public:
  explicit MklFusedInstanceNormOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
    std::vector<int> mean_reduction_axes;
    OP_REQUIRES_OK(context,
                   context->GetAttr("reduction_axes", &mean_reduction_axes));
    OP_REQUIRES(context, InferDataFormat(mean_reduction_axes),
                absl::InvalidArgumentError(
                    "Failed to infer data format from reduction axes"));
    CheckFusedActivation(context);
  }

  void Compute(OpKernelContext* ctx) override {
    try {
      const Tensor& src_tensor = ctx->input(kSrcIndex);
      const Tensor& scale_tensor = ctx->input(kScaleIndex);
      const Tensor& shift_tensor = ctx->input(kShiftIndex);

      OP_REQUIRES(ctx,
                  (src_tensor.dims() == 4 && data_format_ == "NHWC") ||
                      (src_tensor.dims() == 4 && data_format_ == "NCHW") ||
                      (src_tensor.dims() == 5 && data_format_ == "NDHWC") ||
                      (src_tensor.dims() == 5 && data_format_ == "NCDHW"),
                  absl::InvalidArgumentError(absl::StrCat(
                      "Unsupported input: ", src_tensor.shape().DebugString(),
                      ", ", data_format_)));
      size_t num_elements_scale = scale_tensor.NumElements();
      size_t num_elements_shift = shift_tensor.NumElements();
      OP_REQUIRES(ctx, num_elements_scale == num_elements_shift,
                  absl::InvalidArgumentError(
                      absl::StrCat("Number of elements in scale and shift",
                                   "tensors are not same.")));

      TensorFormat tensor_format;
      OP_REQUIRES(ctx, FormatFromString(data_format_, &tensor_format),
                  absl::InvalidArgumentError("Invalid data format"));

      // Create the oneDNN wrapper over Eigen threadpool and set max threads
      // in oneDNN.
      Eigen::ThreadPoolInterface* eigen_interface =
          EigenThreadPoolFromTfContext(ctx);
      tsl::OneDnnThreadPool eigen_tp(eigen_interface,
                                     ThreadPoolUseCallerThread());
      std::shared_ptr<stream> engine_stream_ptr;
      engine_stream_ptr.reset(CreateStream(&eigen_tp, cpu_engine_));

      const int64_t batch_size = src_tensor.shape().dim_size(0);
      const int64_t elems_per_batch =
          src_tensor.shape().num_elements() / batch_size;

      memory::dims src_dims =
          (src_tensor.dims() == 5)
              ? TFShapeToMklDnnDimsInNCDHW(src_tensor.shape(), tensor_format)
              : TFShapeToMklDnnDimsInNCHW(src_tensor.shape(), tensor_format);

      // oneDNN has no direct support for instancenorm, use a workaround
      // with performing multiple batchnorm computations for each sample
      // in the batched input.
      src_dims[0] = 1;

      memory::format_tag tag;
      if (data_format_ == "NCHW" || data_format_ == "NCDHW") {
        tag = (src_dims.size() == 5) ? memory::format_tag::ncdhw
                                     : memory::format_tag::nchw;
      } else {
        tag = (src_dims.size() == 5) ? memory::format_tag::ndhwc
                                     : memory::format_tag::nhwc;
      }
      auto src_md = memory::desc(src_dims, MklDnnType<T>(), tag);

#ifndef ENABLE_ONEDNN_V3
#define NUM_DUPLICATE 2
#else
#define NUM_DUPLICATE 1
#endif  // !ENABLE_ONEDNN_V3
      memory::dims scale_shift_dims = {
          static_cast<dnnl_dim_t>(NUM_DUPLICATE * num_elements_scale)};
      auto scale_shift_md = memory::desc(scale_shift_dims, MklDnnType<float>(),
                                         memory::format_tag::x);
      int64_t tensor_shape = scale_shift_md.get_size() / sizeof(float);
#undef NUM_DUPLICATE

#ifndef ENABLE_ONEDNN_V3
      Tensor scale_shift_tensor;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<float>::v(), {tensor_shape},
                                  &scale_shift_tensor));
      void* scale_shift_buf =
          static_cast<void*>(scale_shift_tensor.flat<float>().data());
      SetupScaleShiftBuffer(ctx, scale_tensor, shift_tensor, engine_stream_ptr,
                            num_elements_scale, scale_shift_buf);
      auto scale_shift_mem =
          memory(scale_shift_md, cpu_engine_, scale_shift_buf);

      // Create oneDNN primitive
      auto bnorm_desc = batch_normalization_forward::desc(
          prop_kind::forward_inference, src_md, epsilon_,
          normalization_flags::use_scale_shift);
#else
      Tensor scale_fp32_tensor;
      Tensor shift_fp32_tensor;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<float>::v(), {tensor_shape},
                                  &scale_fp32_tensor));
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<float>::v(), {tensor_shape},
                                  &shift_fp32_tensor));
      void* scale_fp32_buf =
          static_cast<void*>(scale_fp32_tensor.flat<float>().data());
      void* shift_fp32_buf =
          static_cast<void*>(shift_fp32_tensor.flat<float>().data());

      SetupScaleShiftBuffer(ctx, scale_tensor, shift_tensor, engine_stream_ptr,
                            num_elements_scale, scale_fp32_buf, shift_fp32_buf);
      auto scale_mem = memory(scale_shift_md, cpu_engine_, scale_fp32_buf);
      auto shift_mem = memory(scale_shift_md, cpu_engine_, shift_fp32_buf);
#endif  // !ENABLE_ONEDNN_V3
      batch_normalization_forward::primitive_desc bnorm_pd;
      if (fuse_activation_) {
        dnnl::post_ops post_ops;
        dnnl::primitive_attr post_ops_attr;
#ifndef ENABLE_ONEDNN_V3
        post_ops.append_eltwise(1.0, dnnl::algorithm::eltwise_relu,
                                leakyrelu_alpha_, 0.0);
        post_ops_attr.set_post_ops(post_ops);
        bnorm_pd = batch_normalization_forward::primitive_desc(
            bnorm_desc, post_ops_attr, cpu_engine_);
#else
        post_ops.append_eltwise(dnnl::algorithm::eltwise_relu, leakyrelu_alpha_,
                                0.0);
        post_ops_attr.set_post_ops(post_ops);
        bnorm_pd = batch_normalization_forward::primitive_desc(
            cpu_engine_, prop_kind::forward_inference, src_md, src_md, epsilon_,
            normalization_flags::use_scale | normalization_flags::use_shift,
            post_ops_attr);
#endif  // !ENABLE_ONEDNN_V3
      } else {
#ifndef ENABLE_ONEDNN_V3
        bnorm_pd = batch_normalization_forward::primitive_desc(bnorm_desc,
                                                               cpu_engine_);
#else
        bnorm_pd = batch_normalization_forward::primitive_desc(
            cpu_engine_, prop_kind::forward_inference, src_md, src_md, epsilon_,
            normalization_flags::use_scale | normalization_flags::use_shift);
#endif  // !ENABLE_ONEDNN_V3
      }
      auto bnorm_prim = batch_normalization_forward(bnorm_pd);

      // dst memory
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                              {0}, 0, src_tensor.shape(), &output_tensor));

      std::unique_ptr<dnnl::memory> dst_mem_ptr(
          new memory(src_md, cpu_engine_, (char*)nullptr));
      std::unique_ptr<dnnl::memory> src_mem_ptr(
          new memory(src_md, cpu_engine_, (char*)nullptr));

      const T* src_buf_batch = const_cast<T*>(src_tensor.flat<T>().data());
      const T* dst_buf_batch = const_cast<T*>(output_tensor->flat<T>().data());

      std::unordered_map<int, memory> bnorm_args;
      bnorm_args.insert({DNNL_ARG_SRC, *src_mem_ptr});
      bnorm_args.insert({DNNL_ARG_DST, *dst_mem_ptr});
#ifndef ENABLE_ONEDNN_V3
      bnorm_args.insert({DNNL_ARG_SCALE_SHIFT, scale_shift_mem});
#else
      bnorm_args.insert({DNNL_ARG_SCALE, scale_mem});
      bnorm_args.insert({DNNL_ARG_SHIFT, shift_mem});
#endif  // !ENABLE_ONEDNN_V3

      // Perform batchnorm computation for each batch in input
      for (int i = 0; i < batch_size; i++) {
        src_mem_ptr->set_data_handle(static_cast<void*>(
            const_cast<T*>(src_buf_batch + i * elems_per_batch)));
        dst_mem_ptr->set_data_handle(static_cast<void*>(
            const_cast<T*>(dst_buf_batch + i * elems_per_batch)));
        bnorm_prim.execute(*engine_stream_ptr, bnorm_args);
      }
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(ctx, absl::AbortedError(absl::StrCat(
                              "Operation received an exception:", error_msg)));
    }
  }

 private:
  engine cpu_engine_ = engine(engine::kind::cpu, 0);
  float epsilon_ = 0.0001f;
  float leakyrelu_alpha_ = 0.2f;
  string data_format_ = "";
  const int kSrcIndex = 0;
  const int kScaleIndex = 1;
  const int kShiftIndex = 2;
  bool fuse_activation_ = false;

  void CheckFusedActivation(OpKernelConstruction* context) {
    std::vector<string> fused_ops;
    OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops));

    if (fused_ops.empty()) return;

    if (fused_ops == std::vector<string>{"Relu"}) {
      fuse_activation_ = true;
      leakyrelu_alpha_ = 0.0f;
    } else if (fused_ops == std::vector<string>{"LeakyRelu"}) {
      fuse_activation_ = true;
      OP_REQUIRES_OK(context,
                     context->GetAttr("leakyrelu_alpha", &leakyrelu_alpha_));
    } else {
      OP_REQUIRES(context, false,
                  absl::UnimplementedError(
                      absl::StrCat("Fusion is not implemented: [",
                                   absl::StrJoin(fused_ops, ","), "]")));
    }
  }

  // Given the reduction axes of mean computation, infer data format of input
  bool InferDataFormat(const std::vector<int>& mean_reduction_axes) {
    bool valid = true;
    if (mean_reduction_axes == std::vector<int>{1, 2}) {
      data_format_ = "NHWC";
    } else if (mean_reduction_axes == std::vector<int>{2, 3}) {
      data_format_ = "NCHW";
    } else if (mean_reduction_axes == std::vector<int>{1, 2, 3}) {
      data_format_ = "NDHWC";
    } else if (mean_reduction_axes == std::vector<int>{2, 3, 4}) {
      data_format_ = "NCDHW";
    } else {
      valid = false;
    }
    return valid;
  }

  // Helper function to prepare scale and shift data in float type as
  // required by oneDNN library. Prior to oneDNN 3.x version, the library
  // requires the final scale and shift data to be passed in the same buffer
  // whereas the 3.x version requires separate buffers for scale and shift
  // data.
  void SetupScaleShiftBuffer(OpKernelContext* ctx, const Tensor& scale_tensor,
                             const Tensor& shift_tensor,
                             std::shared_ptr<stream> engine_stream_ptr,
                             int num_elements, void* fp32_scale_or_combine_buf,
                             void* fp32_shift_buf = nullptr) {
    void* scale_buf_src =
        static_cast<void*>(const_cast<T*>(scale_tensor.flat<T>().data()));
    void* shift_buf_src =
        static_cast<void*>(const_cast<T*>(shift_tensor.flat<T>().data()));
    auto data_size = sizeof(float) * num_elements;
    void* scale_buf_dst = fp32_scale_or_combine_buf;
    void* shift_buf_dst = nullptr;
#ifndef ENABLE_ONEDNN_V3
    shift_buf_dst = static_cast<char*>(fp32_scale_or_combine_buf) + data_size;
    (void)fp32_shift_buf;
#else
    OP_REQUIRES(ctx, (fp32_shift_buf != nullptr),
                absl::InvalidArgumentError("Invalid shift buffer"));
    shift_buf_dst = fp32_shift_buf;
#endif  // !ENABLE_ONEDNN_V3

    if (std::is_same<T, float>::value) {
      memcpy(scale_buf_dst, scale_buf_src, data_size);
      memcpy(shift_buf_dst, shift_buf_src, data_size);
    } else {
      // oneDNN requires float type for scale_shift, need to convert to float
      // type
      auto scale_mem_src =
          memory({{num_elements}, MklDnnType<T>(), memory::format_tag::x},
                 cpu_engine_, scale_buf_src);
      auto scale_mem_dst =
          memory({{num_elements}, MklDnnType<float>(), memory::format_tag::x},
                 cpu_engine_, scale_buf_dst);
      auto scale_reorder_prim = reorder(scale_mem_src, scale_mem_dst);
      std::unordered_map<int, memory> scale_reorder_args;
      scale_reorder_args.insert({DNNL_ARG_FROM, scale_mem_src});
      scale_reorder_args.insert({DNNL_ARG_TO, scale_mem_dst});
      scale_reorder_prim.execute(*engine_stream_ptr, scale_reorder_args);

      auto shift_mem_src =
          memory({{num_elements}, MklDnnType<T>(), memory::format_tag::x},
                 cpu_engine_, shift_buf_src);
      auto shift_mem_dst =
          memory({{num_elements}, MklDnnType<float>(), memory::format_tag::x},
                 cpu_engine_, shift_buf_dst);
      auto shift_reorder_prim = reorder(shift_mem_src, shift_mem_dst);
      std::unordered_map<int, memory> shift_reorder_args;
      shift_reorder_args.insert({DNNL_ARG_FROM, shift_mem_src});
      shift_reorder_args.insert({DNNL_ARG_TO, shift_mem_dst});
      shift_reorder_prim.execute(*engine_stream_ptr, shift_reorder_args);
    }
  }
};

#define REGISTER_FUSED_INSTANCE_NORM_CPU(T)                                    \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_MklFusedInstanceNorm").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      MklFusedInstanceNormOp<CPUDevice, T>);
TF_CALL_float(REGISTER_FUSED_INSTANCE_NORM_CPU)
    TF_CALL_bfloat16(REGISTER_FUSED_INSTANCE_NORM_CPU)
        TF_CALL_half(REGISTER_FUSED_INSTANCE_NORM_CPU)

}  // namespace tensorflow

#endif  // INTEL_MKL
