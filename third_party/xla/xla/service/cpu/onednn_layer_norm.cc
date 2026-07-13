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

#include "xla/service/cpu/onednn_layer_norm.h"

#include <cstdint>
#include <string>
#include <unordered_map>

#define EIGEN_USE_THREADS

#include "absl/base/attributes.h"
#include "oneapi/dnnl/dnnl_threadpool.hpp"
#include "oneapi/dnnl/dnnl_types.h"
#include "xla/executable_run_options.h"
#include "xla/service/cpu/backend_config.pb.h"

// Eigen Tensor must come after `onednn_threadpool.h`
#include "unsupported/Eigen/CXX11/Tensor"  // NOLINT

namespace xla {
namespace cpu {
namespace {

using dnnl::engine;
using dnnl::layer_normalization_forward;
using dnnl::memory;
using dnnl::normalization_flags;
using dnnl::primitive;
using dnnl::prop_kind;
using dnnl::stream;

}  // namespace

void ExecuteOneDnnLayerNorm(OneDnnNormConfig ln_config,
                            const dnnl::engine& cpu_engine,
                            dnnl::stream& onednn_stream,
                            OneDnnPrimResources& resources) {
  MemrefInfo layer_minfo(resources.arg_memrefs[0].get());
  MemrefInfo gamma_minfo(resources.arg_memrefs[1].get());
  MemrefInfo beta_minfo(resources.arg_memrefs[2].get());
  MemrefInfo result_minfo(resources.result_memrefs[0].get());

  auto src_md = layer_minfo.GetOneDnnMemDesc();
  auto dst_md = result_minfo.GetOneDnnMemDesc();
  auto scaleshift_md = beta_minfo.GetOneDnnMemDesc();

  resources.src_mem = memory(src_md, cpu_engine, layer_minfo.Data());
  resources.dst_mem = memory(dst_md, cpu_engine, result_minfo.Data());
  resources.scale_mem = memory(scaleshift_md, cpu_engine, gamma_minfo.Data());
  resources.shift_mem = memory(scaleshift_md, cpu_engine, beta_minfo.Data());

  float epsilon = absl::bit_cast<float>(ln_config.epsilon_typecast());

  auto lnorm_pd = layer_normalization_forward::primitive_desc(
      cpu_engine, prop_kind::forward_inference, src_md, dst_md, epsilon,
      normalization_flags::use_scale | normalization_flags::use_shift);

  resources.primitive = primitive(lnorm_pd);

  std::unordered_map<int, memory> ln_args = {
      {DNNL_ARG_SRC, resources.src_mem},
      {DNNL_ARG_SCALE, resources.scale_mem},
      {DNNL_ARG_SHIFT, resources.shift_mem},
      {DNNL_ARG_DST, resources.dst_mem},
  };

  resources.primitive.execute(onednn_stream, ln_args);
}

}  // namespace cpu
}  // namespace xla
