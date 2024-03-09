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
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/logging.h"
#include "tsl/util/onednn_threadpool.h"

namespace xla {
namespace cpu {
namespace {
using dnnl::engine;
using dnnl::matmul;
using dnnl::memory;
using dnnl::stream;

dnnl::memory::desc Transpose(const dnnl::memory::desc& md) {
  int64_t ndims = md.get_ndims();
  // Do not transpose 1D
  if (ndims == 1) {
    return md;
  }

  std::vector<int> permutation(ndims);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::swap(permutation[ndims - 1], permutation[ndims - 2]);
  return md.permute_axes(permutation);
}

dnnl::memory::desc ShapeToMemDesc(const Shape& shape, bool transpose = false) {
  auto dimensions = shape.dimensions();
  if (dimensions.size() == 0) {
    return dnnl::memory::desc{};
  }

  auto dims = dnnl::memory::dims(dimensions.begin(), dimensions.end());

  dnnl::memory::dims strides(dims.size());
  dnnl::memory::dim stride = 1;
  for (auto i : shape.layout().minor_to_major()) {
    strides.at(i) = stride;
    stride *= dims.at(i);
  }

  auto dt = ToOneDnnDataType(static_cast<PrimitiveType>(shape.element_type()));

  return transpose ? Transpose(dnnl::memory::desc(dims, dt, strides))
                   : dnnl::memory::desc(dims, dt, strides);
}

dnnl::memory::desc OneDnnMatMulOptWeightsDesc(
    const dnnl::engine& engine, const dnnl::memory::desc& input_md,
    const dnnl::memory::desc& weights_md, const dnnl::memory::desc& bias_md,
    const dnnl::memory::desc& output_md) {
  auto weights_any_md =
      memory::desc(weights_md.get_dims(), weights_md.get_data_type(),
                   dnnl::memory::format_tag::any);

  auto matmul_pd = matmul::primitive_desc(engine, input_md, weights_any_md,
                                          bias_md, output_md);

  return matmul_pd.weights_desc();
}

dnnl::memory::desc OneDnnMatMulOptWeightsDesc(
    const dnnl::engine& engine, const Shape& input_shape,
    const Shape& weights_shape, const Shape& bias_shape,
    const Shape& output_shape, const OneDnnMatMulConfig* matmul_config) {
  auto input_md = ShapeToMemDesc(input_shape, matmul_config->transpose_a());
  auto weights_md = ShapeToMemDesc(weights_shape, matmul_config->transpose_b());
  auto bias_md =
      absl::c_count(matmul_config->fused_ops(), OneDnnMatMulConfig::BIAS) > 0
          ? ShapeToMemDesc(bias_shape)
          : dnnl::memory::desc{};
  auto output_md = ShapeToMemDesc(output_shape);

  // extend bias rank to match result rank
  auto missed_rank = output_md.get_ndims() - bias_md.get_ndims();
  XLA_LIGHTWEIGHT_CHECK(missed_rank >= 0);
  if (!bias_md.is_zero() && missed_rank > 0) {
    auto bias_dims = bias_md.get_dims();
    bias_dims.insert(bias_dims.begin(), missed_rank, 1);
    bias_md = bias_md.reshape(bias_dims);
  }

  return OneDnnMatMulOptWeightsDesc(engine, input_md, weights_md, bias_md,
                                    output_md);
}

Shape MemDescToXlaShape(const dnnl::memory::desc& md) {
  auto dtype = md.get_data_type();
  auto element_size = dnnl::memory::data_type_size(dtype);
  int64_t bytes_num = md.get_size();
  XLA_LIGHTWEIGHT_CHECK(bytes_num % element_size == 0);
  int64_t elements_num = static_cast<int64_t>(bytes_num / element_size);
  return ShapeUtil::MakeShape(ToXlaPrimitiveType(dtype), {elements_num});
}

std::unique_ptr<tsl::OneDnnThreadPool> CreateOneDnnThreadPool(
    const xla::ExecutableRunOptions* run_options) {
#ifndef ENABLE_ONEDNN_OPENMP
  if (run_options != nullptr &&
      run_options->intra_op_thread_pool() != nullptr) {
    return std::make_unique<tsl::OneDnnThreadPool>(
        run_options->intra_op_thread_pool()->getPool(), false);
  } else {
    return nullptr;
  }
#else
  return nullptr;
#endif  // ENABLE_ONEDNN_OPENMP
}

dnnl::stream MakeOneDnnStream(
    const dnnl::engine& cpu_engine,
    dnnl::threadpool_interop::threadpool_iface* thread_pool) {
  if (thread_pool != nullptr) {
    return dnnl::threadpool_interop::make_stream(cpu_engine, thread_pool);
  } else {
    return dnnl::stream(cpu_engine);
  }
}

}  // namespace

Shape OneDnnMatMulOptWeightsShape(const Shape& input_shape,
                                  const Shape& weights_shape,
                                  const Shape& bias_shape,
                                  const Shape& output_shape,
                                  const OneDnnMatMulConfig* matmul_config) {
  engine cpu_engine(engine::kind::cpu, 0);
  auto optimized_weights_md =
      OneDnnMatMulOptWeightsDesc(cpu_engine, input_shape, weights_shape,
                                 bias_shape, output_shape, matmul_config);
  return MemDescToXlaShape(optimized_weights_md);
}

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

  auto thread_pool = CreateOneDnnThreadPool(run_options);
  engine cpu_engine(engine::kind::cpu, 0);
  auto onednn_stream = MakeOneDnnStream(cpu_engine, thread_pool.get());

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
  if (matmul_config.transpose_a()) {
    lhs_md = Transpose(lhs_md);
  }

  if (matmul_config.transpose_b()) {
    rhs_md = Transpose(rhs_md);
  }
  auto bias_mem = memory(nullptr);
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
      case OneDnnMatMulConfig::LINEAR: {
        float const_float;
        *(reinterpret_cast<int32_t*>(&const_float)) =
            matmul_config.alpha_typecast();
        post_ops.append_eltwise(dnnl::algorithm::eltwise_linear, const_float,
                                0.f);
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

  bool weights_packed = rhs_md.get_ndims() == 1 &&
                        rhs_md.get_dims().front() != lhs_md.get_dims().back();
  if (weights_packed) {
    // expected 2D buffer with last dim of input and last dim of output
    auto rhs_any_md =
        memory::desc({lhs_md.get_dims().back(), result_md.get_dims().back()},
                     rhs_md.get_data_type(), memory::format_tag::any);

    rhs_md = OneDnnMatMulOptWeightsDesc(cpu_engine, lhs_md, rhs_any_md, bias_md,
                                        result_md);
  }

  auto lhs_mem = memory(lhs_md, cpu_engine, lhs_minfo.Data());
  auto rhs_mem = memory(rhs_md, cpu_engine, rhs_minfo.Data());
  auto result_mem = memory(result_md, cpu_engine, result_minfo.Data());

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

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_OneDnnMatMulReorder(
    void* result, void** args) {
  // args[0]: ptr to nargs
  // args[1]: ptr to ExecutableRunOptions
  // args[2]: ptr to OneDnnMatMulConfig
  // args[3...]: ptrs to operands
  int arg_indx = 0;
  const int64_t num_args = *(static_cast<int64_t*>(args[arg_indx++]));

  const xla::ExecutableRunOptions* run_options =
      static_cast<const xla::ExecutableRunOptions*>(args[arg_indx++]);

  auto thread_pool = CreateOneDnnThreadPool(run_options);
  engine cpu_engine(engine::kind::cpu, 0);
  auto onednn_stream = MakeOneDnnStream(cpu_engine, thread_pool.get());

  std::string config_str(static_cast<const char*>(args[arg_indx++]));
  OneDnnMatMulConfig matmul_config;
  matmul_config.ParseFromString(config_str);

  MemrefInfo input_minfo(args[arg_indx++]);
  MemrefInfo weight_minfo(args[arg_indx++]);
  MemrefInfo output_minfo(args[arg_indx++]);
  MemrefInfo result_minfo(result);

  auto input_md = input_minfo.GetOneDnnMemDesc();
  auto weight_md = weight_minfo.GetOneDnnMemDesc();
  auto output_md = output_minfo.GetOneDnnMemDesc();

  auto bias_md = dnnl::memory::desc{};
  if (absl::c_count(matmul_config.fused_ops(), OneDnnMatMulConfig::BIAS) > 0) {
    MemrefInfo bias_minfo(args[arg_indx++]);
    bias_md = bias_minfo.GetOneDnnMemDesc();
  }

  XLA_LIGHTWEIGHT_CHECK(num_args >= arg_indx);

  // Update dims and strides for transposed inputs.
  bool transpose_a = matmul_config.transpose_a();
  if (transpose_a) {
    input_md = Transpose(input_md);
  }
  bool transpose_b = matmul_config.transpose_b();
  if (transpose_b) {
    weight_md = Transpose(weight_md);
  }

  // extend bias rank to match result rank
  if (!bias_md.is_zero()) {
    auto missed_rank = output_md.get_ndims() - bias_md.get_ndims();
    XLA_LIGHTWEIGHT_CHECK(missed_rank >= 0);
    if (missed_rank > 0) {
      auto bias_dims = bias_md.get_dims();
      bias_dims.insert(bias_dims.begin(), missed_rank, 1);
      bias_md = bias_md.reshape(bias_dims);
    }
  }

  auto result_md = OneDnnMatMulOptWeightsDesc(cpu_engine, input_md, weight_md,
                                              bias_md, output_md);

  XLA_LIGHTWEIGHT_CHECK(result_minfo.GetOneDnnMemDesc().get_size() ==
                        result_md.get_size());

  auto weight_mem = dnnl::memory{weight_md, cpu_engine, weight_minfo.Data()};
  auto result_mem = dnnl::memory{result_md, cpu_engine, result_minfo.Data()};

  dnnl::reorder rdr{weight_mem, result_mem};
  rdr.execute(onednn_stream, weight_mem, result_mem);
  onednn_stream.wait();
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
