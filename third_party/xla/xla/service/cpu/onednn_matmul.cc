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

#include "xla/service/cpu/onednn_matmul.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <utility>
#include <vector>

#define EIGEN_USE_THREADS

#include "dnnl.hpp"
#include "absl/base/dynamic_annotations.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "xla/executable_run_options.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/onednn_memory_util.h"
#include "xla/service/cpu/runtime_lightweight_check.h"
#include "tsl/platform/logging.h"
#include "tsl/util/onednn_threadpool.h"

namespace xla {
namespace cpu {
namespace {
using dnnl::engine;
using dnnl::matmul;
using dnnl::memory;
using dnnl::stream;
}  // namespace

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_OneDnnMatMul(
    void* result, void** args) {
  // args[0]: ptr to nargs
  // args[1]: ptr to ExecutableRunOptions
  // args[2]: ptr to OneDnnMatMulConfig
  // args[3...]: ptrs to operands
  int arg_indx = 0;
  const int64_t num_args = *(static_cast<int64_t*>(args[arg_indx++]));

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
  OneDnnMatMulConfig matmul_config;
  matmul_config.ParseFromString(config_str);

  MemrefInfo lhs_minfo(args[arg_indx++]);
  MemrefInfo rhs_minfo(args[arg_indx++]);
  MemrefInfo result_minfo(result);

  auto lhs_md = lhs_minfo.GetOneDnnMemDesc();
  auto rhs_md = rhs_minfo.GetOneDnnMemDesc();
  auto bias_md = memory::desc();
  auto result_md = result_minfo.GetOneDnnMemDesc();

  // Update dims and strides for transposed inputs.
  bool transpose_a = matmul_config.transpose_a();
  if (transpose_a) {
    int64_t ndims = lhs_md.get_ndims();
    std::vector<int> permutation(ndims);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::swap(permutation[ndims - 1], permutation[ndims - 2]);
    lhs_md = lhs_md.permute_axes(permutation);
  }
  bool transpose_b = matmul_config.transpose_b();
  if (transpose_b) {
    int64_t ndims = rhs_md.get_ndims();
    std::vector<int> permutation(ndims);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::swap(permutation[ndims - 1], permutation[ndims - 2]);
    rhs_md = rhs_md.permute_axes(permutation);
  }
  auto lhs_mem = memory(lhs_md, cpu_engine, lhs_minfo.Data());
  auto rhs_mem = memory(rhs_md, cpu_engine, rhs_minfo.Data());
  auto bias_mem = memory(nullptr);
  auto result_mem = memory(result_md, cpu_engine, result_minfo.Data());
  std::vector<std::pair<int, dnnl::memory>> postop_args;

  // Currently, GELU/ReLU only fusion is supported.
  dnnl::post_ops post_ops;
  for (auto& fused_op : matmul_config.fused_ops()) {
    switch (fused_op) {
      case OneDnnMatMulConfig::RELU:
        post_ops.append_eltwise(dnnl::algorithm::eltwise_relu, 0.f, 0.f);
        break;
      case OneDnnMatMulConfig::TANH:
        post_ops.append_eltwise(dnnl::algorithm::eltwise_tanh, 0.f, 0.f);
        break;
      case OneDnnMatMulConfig::GELU_TANH:
        post_ops.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, 0.f, 0.f);
        break;
      case OneDnnMatMulConfig::GELU_ERF:
        post_ops.append_eltwise(dnnl::algorithm::eltwise_gelu_erf, 0.f, 0.f);
        break;
      case OneDnnMatMulConfig::BIAS: {
        MemrefInfo bias_minfo(args[arg_indx++]);
        bias_md = bias_minfo.GetOneDnnMemDesc();

        // Extend bias rank to match result rank.
        auto missed_rank = result_md.get_ndims() - bias_md.get_ndims();
        XLA_LIGHTWEIGHT_CHECK(missed_rank >= 0);
        if (missed_rank > 0) {
          auto bias_dims = bias_md.get_dims();
          bias_dims.insert(bias_dims.begin(), missed_rank, 1);
          bias_md = bias_md.reshape(bias_dims);
        }
        bias_mem = memory(bias_md, cpu_engine, bias_minfo.Data());
      } break;
      case OneDnnMatMulConfig::BINARY_ADD: {
        MemrefInfo binary_minfo(args[arg_indx++]);
        auto binary_md = binary_minfo.GetOneDnnMemDesc();
        auto arg_idx =
            DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_ops.len()) | DNNL_ARG_SRC_1;
        post_ops.append_binary(dnnl::algorithm::binary_add, binary_md);
        postop_args.emplace_back(
            arg_idx, dnnl::memory(binary_md, cpu_engine, binary_minfo.Data()));
      } break;
      default:
        LOG(FATAL) << __FILE__ << ":" << __LINE__
                   << " Attempt to call OneDNN MatMul runtime library with "
                      "unsupported post op."
                   << std::endl;
    }
  }

  XLA_LIGHTWEIGHT_CHECK(num_args == arg_indx);

  dnnl::primitive_attr attrs;
  if (post_ops.len() > 0) {
    attrs.set_post_ops(post_ops);
  }

  auto matmul_pd = matmul::primitive_desc(cpu_engine, lhs_md, rhs_md, bias_md,
                                          result_md, attrs);

  if (std::strstr(matmul_pd.impl_info_str(), "ref") != nullptr) {
    LOG(WARNING) << "[Perf]: MatMul reference implementation being executed";
  }

  auto matmul_prim = matmul(matmul_pd);

  std::unordered_map<int, memory> matmul_args{{DNNL_ARG_SRC, lhs_mem},
                                              {DNNL_ARG_WEIGHTS, rhs_mem},
                                              {DNNL_ARG_BIAS, bias_mem},
                                              {DNNL_ARG_DST, result_mem}};

  matmul_args.insert(postop_args.begin(), postop_args.end());

  matmul_prim.execute(onednn_stream, matmul_args);
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
