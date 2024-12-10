/* Copyright 2024 The OpenXLA Authors.

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
#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#include "xla/service/cpu/onednn_util.h"

#define EIGEN_USE_THREADS

namespace xla {
namespace cpu {

std::unique_ptr<tsl::OneDnnThreadPool> CreateOneDnnThreadPool(
    const Eigen::ThreadPoolDevice* threadpool_device) {
#ifndef ENABLE_ONEDNN_OPENMP
  if (threadpool_device != nullptr) {
    return std::make_unique<tsl::OneDnnThreadPool>(threadpool_device->getPool(),
                                                   false);
  }
#endif  // !ENABLE_ONEDNN_OPENMP
  return nullptr;
}

dnnl::stream MakeOneDnnStream(
    const dnnl::engine& cpu_engine,
    dnnl::threadpool_interop::threadpool_iface* thread_pool) {
  return (thread_pool != nullptr)
             ? dnnl::threadpool_interop::make_stream(cpu_engine, thread_pool)
             : dnnl::stream(cpu_engine);
}

dnnl::post_ops PopulateOneDnnPostOps(
    const dnnl::engine& cpu_engine,
    const std::vector<dnnl::memory::desc>& fused_mds,
    const OneDnnFusionConfig* fusion_config,
    FusedOperandsRef* fused_operands_ref, dnnl::memory::desc* bias_md) {
  dnnl::post_ops post_ops;
  int fused_operand_idx = 0;
  for (auto& fused_op : fusion_config->ops()) {
    switch (fused_op) {
      case OneDnnFusionConfig::RELU:
        post_ops.append_eltwise(dnnl::algorithm::eltwise_relu, 0.f, 0.f);
        break;
      case OneDnnFusionConfig::TANH:
        post_ops.append_eltwise(dnnl::algorithm::eltwise_tanh, 0.f, 0.f);
        break;
      case OneDnnFusionConfig::GELU_TANH:
        post_ops.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, 0.f, 0.f);
        break;
      case OneDnnFusionConfig::GELU_ERF:
        post_ops.append_eltwise(dnnl::algorithm::eltwise_gelu_erf, 0.f, 0.f);
        break;
      case OneDnnFusionConfig::RELU6:
        post_ops.append_eltwise(dnnl::algorithm::eltwise_clip_v2, 0.f, 6.0f);
        break;
      case OneDnnFusionConfig::SIGMOID:
        post_ops.append_eltwise(dnnl::algorithm::eltwise_logistic, 0.f, 0.f);
        break;
      case OneDnnFusionConfig::BIAS: {
        *bias_md = fused_mds.at(fused_operand_idx);
        if (fused_operands_ref) {
          fused_operands_ref->postop_args.emplace_back(
              DNNL_ARG_BIAS,
              dnnl::memory(*bias_md, cpu_engine,
                           fused_operands_ref->bufs[fused_operand_idx]));
        }
        fused_operand_idx++;
      } break;
      case OneDnnFusionConfig::ELU:
        post_ops.append_eltwise(dnnl::algorithm::eltwise_elu, 1.0f, 0.0f);
        break;
      case OneDnnFusionConfig::BINARY_ADD: {
        auto binary_md = fused_mds.at(fused_operand_idx);
        if (fused_operands_ref) {
          auto arg_idx =
              DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_ops.len()) | DNNL_ARG_SRC_1;
          fused_operands_ref->postop_args.emplace_back(
              arg_idx,
              dnnl::memory(binary_md, cpu_engine,
                           fused_operands_ref->bufs[fused_operand_idx]));
        }
        post_ops.append_binary(dnnl::algorithm::binary_add, binary_md);
        fused_operand_idx++;
      } break;
      case OneDnnFusionConfig::LINEAR: {
        float const_float;
        *(reinterpret_cast<int32_t*>(&const_float)) =
            fusion_config->alpha_typecast();
        post_ops.append_eltwise(dnnl::algorithm::eltwise_linear, const_float,
                                0.f);
      } break;
      default:
        LOG(FATAL) << __FILE__ << ":" << __LINE__
                   << " Attempt to call OneDNN runtime library with "
                      "unsupported post op."
                   << std::endl;
    }
  }
  return post_ops;
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
