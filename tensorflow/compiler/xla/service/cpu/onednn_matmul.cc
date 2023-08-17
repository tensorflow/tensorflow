/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/onednn_matmul.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <vector>

#define EIGEN_USE_THREADS

#include "dnnl.hpp"
#include "absl/base/dynamic_annotations.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/service/cpu/backend_config.pb.h"
#include "tensorflow/compiler/xla/service/cpu/onednn_memory_util.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_lightweight_check.h"
#include "tensorflow/tsl/util/onednn_threadpool.h"

namespace xla {
namespace cpu {
namespace {
using dnnl::engine;
using dnnl::matmul;
using dnnl::memory;
using dnnl::stream;
}  // namespace

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_OneDnnMatMul(
    const void* run_options_ptr, void* lhs, void* rhs, void* result,
    void* config) {
  const xla::ExecutableRunOptions* run_options =
      static_cast<const xla::ExecutableRunOptions*>(run_options_ptr);
  XLA_LIGHTWEIGHT_CHECK(run_options != nullptr);
  XLA_LIGHTWEIGHT_CHECK(run_options->intra_op_thread_pool() != nullptr);
  tsl::OneDnnThreadPool thread_pool(
      run_options->intra_op_thread_pool()->getPool(), false);
  engine cpu_engine(engine::kind::cpu, 0);
  auto tp_stream =
      stream(dnnl::threadpool_interop::make_stream(cpu_engine, &thread_pool));

  MemrefInfo lhs_minfo(lhs);
  MemrefInfo rhs_minfo(rhs);
  MemrefInfo result_minfo(result);

  std::string config_str(static_cast<const char*>(config));
  OneDnnMatMulConfig matmul_config;
  matmul_config.ParseFromString(config_str);

  // Currently, no fusion is supported.
  XLA_LIGHTWEIGHT_CHECK(matmul_config.fused_ops().empty());

  auto src_md = lhs_minfo.GetOneDnnMemDesc();
  auto weights_md = rhs_minfo.GetOneDnnMemDesc();
  auto dst_md = result_minfo.GetOneDnnMemDesc();

  auto src_mem = memory(src_md, cpu_engine, lhs_minfo.Data());
  auto weights_mem = memory(weights_md, cpu_engine, rhs_minfo.Data());
  auto dst_mem = memory(dst_md, cpu_engine, result_minfo.Data());

  auto matmul_pd =
      matmul::primitive_desc(cpu_engine, src_md, weights_md, dst_md);

  auto matmul_prim = matmul(matmul_pd);

  std::unordered_map<int, memory> matmul_args;
  matmul_args.insert({DNNL_ARG_SRC, src_mem});
  matmul_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
  matmul_args.insert({DNNL_ARG_DST, dst_mem});

  matmul_prim.execute(tp_stream, matmul_args);
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
