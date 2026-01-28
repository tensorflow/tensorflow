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

#include "xla/service/cpu/onednn_matmul.h"

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_common.hpp"
#include "oneapi/dnnl/dnnl_types.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/onednn_config.pb.h"
#include "xla/shape.h"
#include "tsl/platform/cpu_info.h"

#define EIGEN_USE_THREADS

namespace xla {
namespace cpu {
namespace {

using dnnl::engine;
using dnnl::matmul;
using dnnl::memory;
using dnnl::primitive;
using dnnl::stream;

void TransposeIfNecessary(
    const tsl::protobuf::RepeatedField<uint64_t> dimensions,
    bool transpose_last_2_dims, dnnl::memory::desc& mem_desc) {
  if (mem_desc.get_ndims() < 2) {
    return;
  }
  std::vector<int> permutation(mem_desc.get_ndims());
  absl::c_iota(permutation, 0);
  int counter = 0;
  for (auto it = dimensions.begin(); it != dimensions.end(); it++) {
    permutation[*it - 1] = counter++;
  }
  mem_desc = mem_desc.permute_axes(permutation);
  TRANSPOSE_LAST_TWO_DIMS_IF(transpose_last_2_dims, mem_desc);
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
  auto input_md = ShapeToMemDesc(input_shape);
  auto weights_md = ShapeToMemDesc(weights_shape);
  TransposeIfNecessary(matmul_config->lhs().tensor().dimensions(),
                       matmul_config->transpose_a(), input_md);
  TransposeIfNecessary(matmul_config->rhs().tensor().dimensions(),
                       matmul_config->transpose_b(), weights_md);
  auto bias_md = absl::c_count(matmul_config->fusions().ops(),
                               OneDnnFusionConfig::BIAS) > 0
                     ? ShapeToMemDesc(bias_shape)
                     : dnnl::memory::desc{};
  auto output_md = ShapeToMemDesc(output_shape);
  TransposeIfNecessary(matmul_config->result().tensor().dimensions(), false,
                       output_md);

  // extend bias rank to match result rank
  auto missed_rank = output_md.get_ndims() - bias_md.get_ndims();
  CHECK_GE(missed_rank, 0);
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
  TransposeIfNecessary(matmul_config.lhs().tensor().dimensions(),
                       matmul_config.transpose_a(), input_md);
  TransposeIfNecessary(matmul_config.rhs().tensor().dimensions(),
                       matmul_config.transpose_b(), weights_md);
  auto output_md = ShapeToMemDesc(output_shape);
  TransposeIfNecessary(matmul_config.result().tensor().dimensions(), false,
                       output_md);
  std::vector<memory::desc> fused_mds;
  absl::c_transform(fused_shapes, std::back_inserter(fused_mds),
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
  absl::c_transform(fused_operands, std::back_inserter(fused_shapes),
                    [](const HloInstruction* instr) { return instr->shape(); });

  return CreateMatMulPrimDesc(input_shape, weight_shape, output_shape,
                              fused_shapes, matmul_config);
}

void ExecuteOneDnnMatMul(absl::Span<MemrefInfoHandler> arguments,
                         absl::Span<MemrefInfoHandler> results,
                         OneDnnMatMulConfig matmul_config,
                         const dnnl::engine& cpu_engine,
                         dnnl::stream& onednn_stream,
                         OneDnnResources& resources) {
  MemrefInfo input_minfo(arguments[0].get());
  MemrefInfo weights_minfo(arguments[1].get());
  MemrefInfo output_minfo(results[0].get());

  auto input_md = input_minfo.GetOneDnnMemDesc();
  auto weights_md = weights_minfo.GetOneDnnMemDesc();
  auto output_md = output_minfo.GetOneDnnMemDesc();

  // Input and weights memory::desc need to be in correct layout before matmul
  // primitive descriptor is created.
  TransposeIfNecessary(matmul_config.lhs().tensor().dimensions(),
                       matmul_config.transpose_a(), input_md);
  TransposeIfNecessary(matmul_config.rhs().tensor().dimensions(),
                       matmul_config.transpose_b(), weights_md);
  TransposeIfNecessary(matmul_config.result().tensor().dimensions(), false,
                       output_md);

  auto weight_format = memory::format_tag::ab;
  if (matmul_config.optimization_config().weights_prepacked()) {
    // Weight pre-packing is supported for 2D weights only.
    // Since prepacked weights array is flattened, try to infer the dims from
    // input and output.
    // TODO(intel-tf): Add support for prepacked weights for higher than 2D
    // array.
    weights_md =
        memory::desc({input_md.get_dims().back(), output_md.get_dims().back()},
                     weights_md.get_data_type(), weight_format);
  }

  // Excluding input and weight operands.
  const int64_t num_fused_operands = arguments.size() - 2;
  std::vector<memory::desc> fused_mds;
  std::vector<void*> fused_bufs;
  for (int64_t i = 0; i < num_fused_operands; ++i) {
    MemrefInfo operand_minfo(arguments[i + 2].get());
    fused_mds.push_back(operand_minfo.GetOneDnnMemDesc());
    fused_bufs.push_back(operand_minfo.Data());
  }

  FusedOperandsRef fused_operands_ref{fused_bufs, resources.postop_args};
  auto matmul_pd =
      CreateMatMulPrimDesc(cpu_engine, input_md, weights_md, output_md,
                           fused_mds, matmul_config, &fused_operands_ref);

  resources.src_mem = memory(input_md, cpu_engine, input_minfo.Data());
  resources.wei_mem =
      memory(matmul_pd->weights_desc(), cpu_engine, weights_minfo.Data());
  resources.dst_mem = memory(output_md, cpu_engine, output_minfo.Data());

  if (std::strstr(matmul_pd->impl_info_str(), "ref") != nullptr) {
    LOG(WARNING) << "[Perf]: MatMul reference implementation being executed";
  }

  resources.primitive = primitive(*matmul_pd);

  std::unordered_map<int, memory> matmul_args{
      {DNNL_ARG_SRC, resources.src_mem},
      {DNNL_ARG_WEIGHTS, resources.wei_mem},
      {DNNL_ARG_DST, resources.dst_mem}};

  if (matmul_config.optimization_config().user_scratchpad()) {
    MemrefInfo scratch_minfo(results[1].get());
    auto scratchpad_md = matmul_pd->scratchpad_desc();
    resources.scratch_mem =
        memory(scratchpad_md, cpu_engine, scratch_minfo.Data());
    matmul_args.insert({DNNL_ARG_SCRATCHPAD, resources.scratch_mem});
  }

  matmul_args.insert(resources.postop_args.begin(),
                     resources.postop_args.end());

  resources.primitive.execute(onednn_stream, matmul_args);
}

}  // namespace cpu
}  // namespace xla
