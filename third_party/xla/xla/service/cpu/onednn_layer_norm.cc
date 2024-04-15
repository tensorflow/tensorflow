/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/cpu/onednn_layer_norm.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <vector>

#define EIGEN_USE_THREADS

#include "dnnl.hpp"
#include "absl/base/dynamic_annotations.h"
#include "xla/executable_run_options.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/onednn_memory_util.h"
#include "xla/service/cpu/runtime_lightweight_check.h"
#include "xla/tsl/util/onednn_threadpool.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla {
namespace cpu {
namespace {
using dnnl::engine;
using dnnl::layer_normalization_forward;
using dnnl::memory;
using dnnl::normalization_flags;
using dnnl::prop_kind;
using dnnl::stream;
}  // namespace

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_OneDnnLayerNorm(
    void* result, void** args) {
  // args[0]: ptr to nargs. We don't use nargs here.
  // args[1]: ptr to ExecutableRunOptions
  // args[2]: ptr to OneDnnLayerNormConfig
  // args[3...]: ptrs to operands
  int arg_indx = 1;
  const xla::ExecutableRunOptions* run_options =
      static_cast<const xla::ExecutableRunOptions*>(args[arg_indx++]);
  XLA_LIGHTWEIGHT_CHECK(run_options != nullptr);
  XLA_LIGHTWEIGHT_CHECK(run_options->intra_op_thread_pool() != nullptr);
  tsl::OneDnnThreadPool thread_pool(
      run_options->intra_op_thread_pool()->getPool(), false);
  engine cpu_engine(engine::kind::cpu, 0);
#ifndef ENABLE_ONEDNN_OPENMP
  auto onednn_stream =
      stream(dnnl::threadpool_interop::make_stream(cpu_engine, &thread_pool));
#else
  auto onednn_stream = stream(cpu_engine);
#endif  // ENABLE_ONEDNN_OPENMP
  std::string config_str(static_cast<const char*>(args[arg_indx++]));
  OneDnnLayerNormConfig ln_config;
  ln_config.ParseFromString(config_str);

  MemrefInfo layer_minfo(args[arg_indx++]);
  MemrefInfo gamma_minfo(args[arg_indx++]);
  MemrefInfo beta_minfo(args[arg_indx++]);
  MemrefInfo result_minfo(result);

  auto src_md = layer_minfo.GetOneDnnMemDesc();
  auto dst_md = result_minfo.GetOneDnnMemDesc();
  auto scaleshift_md = beta_minfo.GetOneDnnMemDesc();

  auto src_mem = memory(src_md, cpu_engine, layer_minfo.Data());
  auto dst_mem = memory(dst_md, cpu_engine, result_minfo.Data());
  auto scale_mem = memory(scaleshift_md, cpu_engine, gamma_minfo.Data());
  auto shift_mem = memory(scaleshift_md, cpu_engine, beta_minfo.Data());

  // TODO(intel-tf): Move epsilon to OneDnnLayerNormConfig.
  float epsilon;
  *(reinterpret_cast<int32_t*>(&epsilon)) = ln_config.epsilon_typecast();

  auto lnorm_pd = layer_normalization_forward::primitive_desc(
      cpu_engine, prop_kind::forward_inference, src_md, dst_md, epsilon,
      normalization_flags::use_scale | normalization_flags::use_shift);

  auto lnorm_prim = layer_normalization_forward(lnorm_pd);

  std::unordered_map<int, memory> ln_args;
  ln_args.insert({DNNL_ARG_SRC, src_mem});
  ln_args.insert({DNNL_ARG_SCALE, scale_mem});
  ln_args.insert({DNNL_ARG_SHIFT, shift_mem});
  ln_args.insert({DNNL_ARG_DST, dst_mem});

  lnorm_prim.execute(onednn_stream, ln_args);
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
