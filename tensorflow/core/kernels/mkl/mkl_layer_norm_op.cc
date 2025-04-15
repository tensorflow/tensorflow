/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

using CPUDevice = Eigen::ThreadPoolDevice;
using dnnl::layer_normalization_forward;
using dnnl::normalization_flags;
using dnnl::prop_kind;

namespace tensorflow {

template <typename Device, typename T>
class MklLayerNormOp : public OpKernel {
 public:
  explicit MklLayerNormOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
  }

  void Compute(OpKernelContext* ctx) override {
    try {
      const Tensor& src_tensor = MklGetInput(ctx, kSrcIndex);
      const Tensor& scale_tensor = MklGetInput(ctx, kScaleIndex);
      const Tensor& shift_tensor = MklGetInput(ctx, kShiftIndex);

      OP_REQUIRES(
          ctx, src_tensor.dims() == 2 || src_tensor.dims() == 3,
          absl::InvalidArgumentError(absl::StrCat(
              "input must be 2D or 3D", src_tensor.shape().DebugString())));
      OP_REQUIRES(
          ctx, scale_tensor.dims() == 1,
          absl::InvalidArgumentError(absl::StrCat(
              "scale must be 1D tensor", scale_tensor.shape().DebugString())));
      OP_REQUIRES(
          ctx, shift_tensor.dims() == 1,
          absl::InvalidArgumentError(absl::StrCat(
              "offset must be 1D tensor", shift_tensor.shape().DebugString())));
      int64_t num_elements_scale = scale_tensor.dim_size(0);
      int64_t num_elements_shift = shift_tensor.dim_size(0);
      OP_REQUIRES(ctx, num_elements_scale == num_elements_shift,
                  absl::InvalidArgumentError(
                      absl::StrCat("Number of elements in scale and shift",
                                   "tensors are not same.")));

      auto cpu_engine = engine(engine::kind::cpu, 0);
      // Create the oneDNN wrapper over Eigen threadpool and set max threads
      // in oneDNN.
      Eigen::ThreadPoolInterface* eigen_interface =
          EigenThreadPoolFromTfContext(ctx);
      tsl::OneDnnThreadPool eigen_tp(eigen_interface,
                                     ThreadPoolUseCallerThread());
      auto cpu_stream =
          std::unique_ptr<stream>(CreateStream(&eigen_tp, cpu_engine));

      memory::dims src_dims = TFShapeToMklDnnDims(src_tensor.shape());
      auto src_md =
          memory::desc(src_dims, MklDnnType<T>(),
                       (src_dims.size() == 3) ? memory::format_tag::tnc
                                              : memory::format_tag::nc);
      void* src_buf =
          static_cast<void*>(const_cast<T*>(src_tensor.flat<T>().data()));
      auto src_mem = memory(src_md, cpu_engine, src_buf);

#ifndef ENABLE_ONEDNN_V3
      // oneDNN v2.x requires scale-shift as a combined array in float32 type.
      memory::dims scale_shift_dims = {
          2, static_cast<dnnl_dim_t>(num_elements_scale)};
      auto scale_shift_md =
          memory::desc(static_cast<memory::dims>(scale_shift_dims),
                       MklDnnType<float>(), memory::format_tag::nc);
      Tensor scale_shift_tensor;
      int64_t tensor_shape = scale_shift_md.get_size() / sizeof(float);
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<float>::v(), {tensor_shape},
                                  &scale_shift_tensor));
      void* scale_shift_buf =
          static_cast<void*>(scale_shift_tensor.flat<float>().data());
      auto scale_shift_mem =
          memory(scale_shift_md, cpu_engine, scale_shift_buf);
      void* scale_buf_dst = scale_shift_buf;
      void* shift_buf_dst = static_cast<char*>(scale_shift_buf) +
                            sizeof(float) * num_elements_scale;
#else
      // oneDNN v3.x requires scale and shift as separate float32 arrays
      memory::dims scale_shift_dims = {
          static_cast<dnnl_dim_t>(num_elements_scale)};
      auto scale_shift_md =
          memory::desc(static_cast<memory::dims>(scale_shift_dims),
                       MklDnnType<float>(), memory::format_tag::x);

      const int64_t scale_shift_tensor_shape =
          scale_shift_md.get_size() / sizeof(float);

      Tensor scale_buf_tensor;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<float>::v(),
                                             {scale_shift_tensor_shape},
                                             &scale_buf_tensor));
      void* scale_buf_dst =
          static_cast<void*>(scale_buf_tensor.flat<float>().data());
      auto scale_mem = memory(scale_shift_md, cpu_engine, scale_buf_dst);

      Tensor shift_buf_tensor;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<float>::v(),
                                             {scale_shift_tensor_shape},
                                             &shift_buf_tensor));
      void* shift_buf_dst =
          static_cast<void*>(shift_buf_tensor.flat<float>().data());
      auto shift_mem = memory(scale_shift_md, cpu_engine, shift_buf_dst);
#endif  // !ENABLE_ONEDNN_V3

      void* scale_buf_src =
          static_cast<void*>(const_cast<T*>(scale_tensor.flat<T>().data()));
      auto scale_mem_src = memory({{static_cast<ptrdiff_t>(num_elements_scale)},
                                   MklDnnType<T>(),
                                   memory::format_tag::x},
                                  cpu_engine, scale_buf_src);
      auto scale_mem_dst = memory({{static_cast<ptrdiff_t>(num_elements_scale)},
                                   MklDnnType<float>(),
                                   memory::format_tag::x},
                                  cpu_engine, scale_buf_dst);
      auto scale_reorder_prim = reorder(scale_mem_src, scale_mem_dst);
      std::unordered_map<int, memory> scale_reorder_args;
      scale_reorder_args.insert({DNNL_ARG_FROM, scale_mem_src});
      scale_reorder_args.insert({DNNL_ARG_TO, scale_mem_dst});
      scale_reorder_prim.execute(*cpu_stream, scale_reorder_args);

      void* shift_buf_src =
          static_cast<void*>(const_cast<T*>(shift_tensor.flat<T>().data()));
      auto shift_mem_src = memory({{static_cast<ptrdiff_t>(num_elements_shift)},
                                   MklDnnType<T>(),
                                   memory::format_tag::x},
                                  cpu_engine, shift_buf_src);
      auto shift_mem_dst = memory({{static_cast<ptrdiff_t>(num_elements_shift)},
                                   MklDnnType<float>(),
                                   memory::format_tag::x},
                                  cpu_engine, shift_buf_dst);
      auto shift_reorder_prim = reorder(shift_mem_src, shift_mem_dst);
      std::unordered_map<int, memory> shift_reorder_args;
      shift_reorder_args.insert({DNNL_ARG_FROM, shift_mem_src});
      shift_reorder_args.insert({DNNL_ARG_TO, shift_mem_dst});
      shift_reorder_prim.execute(*cpu_stream, shift_reorder_args);

      // Create layer_normalization primitive
#ifndef ENABLE_ONEDNN_V3
      auto lnorm_desc = layer_normalization_forward::desc(
          prop_kind::forward_inference, src_md, epsilon_,
          normalization_flags::use_scale_shift);
      auto lnorm_pd =
          layer_normalization_forward::primitive_desc(lnorm_desc, cpu_engine);
#else
      auto dst_md = src_md;
      auto lnorm_pd = layer_normalization_forward::primitive_desc(
          cpu_engine, prop_kind::forward_inference, src_md, dst_md, epsilon_,
          normalization_flags::use_scale | normalization_flags::use_shift);
#endif  // !ENABLE_ONEDNN_V3
      auto lnorm_prim = layer_normalization_forward(lnorm_pd);

      // mean and variance memory
      auto mean_mem = memory(lnorm_pd.mean_desc(), cpu_engine);
      auto variance_mem = memory(lnorm_pd.variance_desc(), cpu_engine);

      // dst memory
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                              {0}, 0, src_tensor.shape(), &output_tensor));
      void* dst_buf =
          static_cast<void*>(const_cast<T*>(output_tensor->flat<T>().data()));
      auto dst_mem = memory(src_md, cpu_engine, dst_buf);

      std::unordered_map<int, memory> lnorm_args;
      lnorm_args.insert({DNNL_ARG_SRC, src_mem});
      lnorm_args.insert({DNNL_ARG_MEAN, mean_mem});
      lnorm_args.insert({DNNL_ARG_VARIANCE, variance_mem});
#ifndef ENABLE_ONEDNN_V3
      lnorm_args.insert({DNNL_ARG_SCALE_SHIFT, scale_shift_mem});
#else
      lnorm_args.insert({DNNL_ARG_SCALE, scale_mem});
      lnorm_args.insert({DNNL_ARG_SHIFT, shift_mem});
#endif  // !ENABLE_ONEDNN_V3
      lnorm_args.insert({DNNL_ARG_DST, dst_mem});
      lnorm_prim.execute(*cpu_stream, lnorm_args);
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(ctx, absl::AbortedError(absl::StrCat(
                              "Operation received an exception:", error_msg)));
    }
  }

 private:
  float epsilon_;
  const int kSrcIndex = 0;
  const int kScaleIndex = 1;
  const int kShiftIndex = 2;
};

REGISTER_KERNEL_BUILDER(
    Name("_MklLayerNorm").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    MklLayerNormOp<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("_MklLayerNorm").Device(DEVICE_CPU).TypeConstraint<bfloat16>("T"),
    MklLayerNormOp<CPUDevice, bfloat16>);

REGISTER_KERNEL_BUILDER(
    Name("_MklLayerNorm").Device(DEVICE_CPU).TypeConstraint<Eigen::half>("T"),
    MklLayerNormOp<CPUDevice, Eigen::half>);

}  // namespace tensorflow

#endif  // INTEL_MKL
