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
#if defined(INTEL_MKL)
#include "xla/service/cpu/onednn_matmul.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/base/dynamic_annotations.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "dnnl.hpp"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/onednn_config.pb.h"
#include "xla/service/cpu/onednn_memory_util.h"
#include "xla/service/cpu/onednn_util.h"
#include "xla/service/cpu/runtime_lightweight_check.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/util/onednn_threadpool.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/logging.h"

#define EIGEN_USE_THREADS

namespace xla {
namespace cpu {
namespace {
using dnnl::engine;
using dnnl::matmul;
using dnnl::memory;
using dnnl::stream;

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
  auto input_md = ShapeToMemDesc(input_shape);
  auto weights_md = ShapeToMemDesc(weights_shape);
  TRANSPOSE_LAST_TWO_DIMS_IF(matmul_config->transpose_a(), input_md);
  TRANSPOSE_LAST_TWO_DIMS_IF(matmul_config->transpose_b(), weights_md);
  auto bias_md = absl::c_count(matmul_config->fusions().ops(),
                               OneDnnFusionConfig::BIAS) > 0
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
  return MemDescToXlaShapeFlattened(optimized_weights_md);
}

std::unique_ptr<matmul::primitive_desc> CreateMatMulPrimDesc(
    const engine& cpu_engine, const memory::desc& input_md,
    const memory::desc& plain_weights_md, const memory::desc& output_md,
    const std::vector<memory::desc>& fused_mds,
    const OneDnnMatMulConfig& matmul_config,
    FusedOperandsRef* fused_operands_ref = nullptr) {
  auto bias_md = memory::desc();
  bool weights_packed = matmul_config.optimization_config().weights_prepacked();
  auto weights_md = plain_weights_md;
  if (weights_packed) {
    weights_md = memory::desc(weights_md.get_dims(), weights_md.get_data_type(),
                              memory::format_tag::any);
  }

  dnnl::post_ops post_ops =
      PopulateOneDnnPostOps(cpu_engine, fused_mds, &matmul_config.fusions(),
                            fused_operands_ref, &bias_md);

  dnnl::primitive_attr attrs;
  if (matmul_config.optimization_config().user_scratchpad()) {
    attrs.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  }
  if (post_ops.len() > 0) {
    attrs.set_post_ops(post_ops);
  }
  return std::make_unique<matmul::primitive_desc>(
      cpu_engine, input_md, weights_md, bias_md, output_md, attrs);
}

std::unique_ptr<matmul::primitive_desc> CreateMatMulPrimDesc(
    const Shape& input_shape, const Shape& weights_shape,
    const Shape& output_shape, const std::vector<Shape>& fused_shapes,
    const OneDnnMatMulConfig& matmul_config) {
  auto input_md = ShapeToMemDesc(input_shape);
  auto weights_md = ShapeToMemDesc(weights_shape);
  TRANSPOSE_LAST_TWO_DIMS_IF(matmul_config.transpose_a(), input_md);
  TRANSPOSE_LAST_TWO_DIMS_IF(matmul_config.transpose_b(), weights_md);
  auto output_md = ShapeToMemDesc(output_shape);
  std::vector<memory::desc> fused_mds;
  std::transform(fused_shapes.begin(), fused_shapes.end(),
                 std::back_inserter(fused_mds),
                 [](const Shape& shape) { return ShapeToMemDesc(shape); });
  return CreateMatMulPrimDesc(engine(engine::kind::cpu, 0), input_md,
                              weights_md, output_md, fused_mds, matmul_config);
}

template <>
typename PrimitiveTrait<kOnednnMatmulConfig>::pointer_type
GetKernelConfig<kOnednnMatmulConfig>(
    absl::StatusOr<BackendConfig>* backend_config) {
  return (*backend_config)->mutable_onednn_matmul_config();
}

template <>
std::unique_ptr<dnnl::matmul::primitive_desc>
CreateOneDnnPrimDesc<dnnl::matmul::primitive_desc>(HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kCustomCall) {
    return nullptr;
  }
  auto custom_call = Cast<xla::HloCustomCallInstruction>(instr);
  auto backend_config = custom_call->backend_config<BackendConfig>();
  if (!backend_config.ok()) {
    return nullptr;
  }
  auto& matmul_config = backend_config.value().onednn_matmul_config();
  auto operands = custom_call->operands();
  auto input = operands[0];
  auto weight = operands[1];  // assuming weights is the second operand
  auto input_shape = input->shape();
  auto weight_shape = weight->shape();
  auto output_shape = custom_call->shape().IsTuple()
                          ? custom_call->shape().tuple_shapes(0)
                          : custom_call->shape();

  auto fused_operands =
      HloInstruction::InstructionVector(operands.begin() + 2, operands.end());
  std::vector<Shape> fused_shapes;
  std::transform(fused_operands.begin(), fused_operands.end(),
                 std::back_inserter(fused_shapes),
                 [](const HloInstruction* instr) { return instr->shape(); });

  return CreateMatMulPrimDesc(input_shape, weight_shape, output_shape,
                              fused_shapes, matmul_config);
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_OneDnnMatMul(
    void* result, void* scratch, void** args) {
  // args[0]: ptr to nargs
  // args[1]: ptr to ExecutableRunOptions
  // args[2]: ptr to OneDnnMatMulConfig
  // args[3...]: ptrs to operands
  int arg_indx = 0;
  const int64_t num_args = *(static_cast<int64_t*>(args[arg_indx++]));

  const xla::ExecutableRunOptions* run_options =
      static_cast<const xla::ExecutableRunOptions*>(args[arg_indx++]);
  auto thread_pool = CreateOneDnnThreadPool(
      run_options ? run_options->intra_op_thread_pool() : nullptr);
  engine cpu_engine(engine::kind::cpu, 0);
  auto onednn_stream = MakeOneDnnStream(cpu_engine, thread_pool.get());

  std::string config_str(static_cast<const char*>(args[arg_indx++]));
  OneDnnMatMulConfig matmul_config;
  matmul_config.ParseFromString(config_str);

  MemrefInfo input_minfo(args[arg_indx++]);
  MemrefInfo weights_minfo(args[arg_indx++]);
  MemrefInfo output_minfo(result);

  auto input_md = input_minfo.GetOneDnnMemDesc();
  auto weights_md = weights_minfo.GetOneDnnMemDesc();
  // Input and weights memory::desc need to be in correct layout before matmul
  // primitive descriptor is created.
  TRANSPOSE_LAST_TWO_DIMS_IF(
      matmul_config.transpose_a() && input_md.get_ndims() > 1, input_md);
  TRANSPOSE_LAST_TWO_DIMS_IF(
      matmul_config.transpose_b() && weights_md.get_ndims() > 1, weights_md);
  auto output_md = output_minfo.GetOneDnnMemDesc();

  Literal* reordered_weights_literal = nullptr;
  void* rhs_data = weights_minfo.Data();

  auto weight_format = tsl::port::IsAarch64CPU() ? memory::format_tag::any
                                                 : memory::format_tag::ab;
  if (matmul_config.optimization_config().weights_prepacked()) {
    // Weight pre-packing is supported for 2D weights only.
    // Since prepacked weights array is flattened, try to infer the dims from
    // input and output.
    // TODO(intel-tf): Add support for prepacked weights for higher then 2D
    // array.
    weights_md =
        memory::desc({input_md.get_dims().back(), output_md.get_dims().back()},
                     weights_md.get_data_type(), weight_format);
  } else if (tsl::port::IsAarch64CPU()) {
    // Weights are not pre-packed, and this scenario requires
    // weights reordering on ARM64 platform
    auto weights_mem =
        dnnl::memory{weights_md, cpu_engine, weights_minfo.Data()};

    auto bias_md = dnnl::memory::desc{};

    if (absl::c_count(matmul_config.fusions().ops(), OneDnnFusionConfig::BIAS) >
        0) {
      MemrefInfo bias_minfo(args[arg_indx]);
      bias_md = bias_minfo.GetOneDnnMemDesc();
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
    auto reordered_weights_md = OneDnnMatMulOptWeightsDesc(
        cpu_engine, input_md, weights_md, bias_md, output_md);

    auto reordered_weights_shape =
        MemDescToXlaShapeFlattened(reordered_weights_md);
    reordered_weights_literal = new Literal(reordered_weights_shape);

    rhs_data = reordered_weights_literal->untyped_data();
    auto reordered_weights_mem =
        dnnl::memory{reordered_weights_md, cpu_engine, rhs_data};

    dnnl::reorder rdr{weights_mem, reordered_weights_mem};
    rdr.execute(onednn_stream, weights_mem, reordered_weights_mem);
    onednn_stream.wait();
    weights_md = reordered_weights_md;
  }

  const int64_t num_fused_operands = num_args - arg_indx;
  std::vector<memory::desc> fused_mds;
  std::vector<void*> fused_bufs;
  for (int64_t i = 0; i < num_fused_operands; ++i) {
    // Skip the MemrefInfo object for the SUM operand, as oneDNN does not
    // require an input and performs in-place accumulation.
    if (matmul_config.fusions().ops(i) == OneDnnFusionConfig::SUM) {
      arg_indx++;
      continue;
    }
    MemrefInfo operand_minfo(args[arg_indx++]);
    fused_mds.push_back(operand_minfo.GetOneDnnMemDesc());
    fused_bufs.push_back(operand_minfo.Data());
  }

  std::vector<std::pair<int, dnnl::memory>> postop_args;
  FusedOperandsRef fused_operands_ref{fused_bufs, postop_args};
  auto matmul_pd =
      CreateMatMulPrimDesc(cpu_engine, input_md, weights_md, output_md,
                           fused_mds, matmul_config, &fused_operands_ref);

  XLA_LIGHTWEIGHT_CHECK(num_args == arg_indx);

  auto lhs_mem = memory(input_md, cpu_engine, input_minfo.Data());
  auto rhs_mem = memory(matmul_pd->weights_desc(), cpu_engine, rhs_data);
  auto result_mem = memory(output_md, cpu_engine, output_minfo.Data());

  if (std::strstr(matmul_pd->impl_info_str(), "ref") != nullptr) {
    LOG(WARNING) << "[Perf]: MatMul reference implementation being executed";
  }

  auto matmul_prim = matmul(*matmul_pd);

  std::unordered_map<int, memory> matmul_args{{DNNL_ARG_SRC, lhs_mem},
                                              {DNNL_ARG_WEIGHTS, rhs_mem},
                                              {DNNL_ARG_DST, result_mem}};

  if (matmul_config.optimization_config().user_scratchpad()) {
    XLA_LIGHTWEIGHT_CHECK(scratch != nullptr);
    MemrefInfo scratch_minfo(scratch);
    auto scratchpad_md = matmul_pd->scratchpad_desc();
    auto scratch_mem = memory(scratchpad_md, cpu_engine, scratch_minfo.Data());
    matmul_args.insert({DNNL_ARG_SCRATCHPAD, scratch_mem});
  }

  matmul_args.insert(postop_args.begin(), postop_args.end());

  matmul_prim.execute(onednn_stream, matmul_args);

  if (reordered_weights_literal != nullptr) {
    delete reordered_weights_literal;
    reordered_weights_literal = nullptr;
  }
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

  auto thread_pool = CreateOneDnnThreadPool(
      run_options ? run_options->intra_op_thread_pool() : nullptr);
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
  if (absl::c_count(matmul_config.fusions().ops(), OneDnnFusionConfig::BIAS) >
      0) {
    MemrefInfo bias_minfo(args[arg_indx++]);
    bias_md = bias_minfo.GetOneDnnMemDesc();
  }

  XLA_LIGHTWEIGHT_CHECK(num_args >= arg_indx);

  // Update dims and strides for transposed inputs.
  TRANSPOSE_LAST_TWO_DIMS_IF(matmul_config.transpose_a(), input_md);
  TRANSPOSE_LAST_TWO_DIMS_IF(matmul_config.transpose_b(), weight_md);

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

#endif  // INTEL_MKL
