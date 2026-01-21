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

#include "xla/service/cpu/onednn_convolution.h"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_common.hpp"
#include "oneapi/dnnl/dnnl_threadpool.hpp"
#include "oneapi/dnnl/dnnl_types.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/onednn_config.pb.h"
#include "xla/shape.h"

#define EIGEN_USE_THREADS

namespace xla {
namespace cpu {
namespace {

using dnnl::algorithm;
using dnnl::convolution_forward;
using dnnl::memory;
using dnnl::primitive;
using dnnl::prop_kind;
using dnnl::stream;

memory::dims GetPrimitiveParameter(
    const tsl::protobuf::RepeatedField<uint64_t>& field, int offset = 0) {
  memory::dims param_field(field.begin(), field.end());
  // Subtract the offset so that values are interpreted accurately
  for (int64_t& n : param_field) {
    n -= offset;
  }
  return param_field;
}

std::vector<int> ComputePermutations(
    uint64_t dims, uint64_t dim0, uint64_t dim1,
    const tsl::protobuf::RepeatedField<uint64_t>& spatial_dims) {
  std::vector<int> perm_axes(dims);
  perm_axes[dim0] = 0;
  perm_axes[dim1] = 1;
  int index = 2;
  for (uint64_t n : spatial_dims) {
    perm_axes[n] = index++;
  }
  return perm_axes;
}

}  // namespace

dnnl::memory ReorderMemory(const dnnl::engine& engine,
                           const dnnl::memory::desc& dest_md,
                           dnnl::memory& src_mem,
                           const dnnl::stream& onednn_stream) {
  auto dest_mem = memory(dest_md, engine);
  dnnl::reorder(src_mem, dest_mem).execute(onednn_stream, src_mem, dest_mem);
  return dest_mem;
}

dnnl::memory::format_tag GetFormatTag(const int dims) {
  return (dims == 3)   ? dnnl::memory::format_tag::nwc
         : (dims == 4) ? dnnl::memory::format_tag::nhwc
         : (dims == 5) ? dnnl::memory::format_tag::ndhwc
                       : dnnl::memory::format_tag::any;
}

template <>
typename PrimitiveTrait<kOnednnConvConfig>::pointer_type
GetKernelConfig<kOnednnConvConfig>(
    absl::StatusOr<BackendConfig>* backend_config) {
  return (*backend_config)->mutable_onednn_conv_config();
}

template <>
std::unique_ptr<dnnl::convolution_forward::primitive_desc>
CreateOneDnnPrimDesc<dnnl::convolution_forward::primitive_desc>(
    HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kCustomCall) {
    return nullptr;
  }
  xla::HloCustomCallInstruction* custom_call =
      Cast<xla::HloCustomCallInstruction>(instr);
  absl::StatusOr<BackendConfig> backend_config =
      custom_call->backend_config<BackendConfig>();
  if (!backend_config.ok()) {
    return nullptr;
  }
  const auto& conv_config = backend_config->onednn_conv_config();
  const HloInstruction::InstructionVector& operands = custom_call->operands();
  const Shape& input_shape = operands[0]->shape();
  const Shape& weight_shape = operands[1]->shape();
  const Shape& output_shape = custom_call->shape().IsTuple()
                                  ? custom_call->shape().tuple_shapes(0)
                                  : custom_call->shape();

  std::vector<Shape> fused_shapes;
  for (int i = 2; i < operands.size(); ++i) {
    fused_shapes.push_back(operands[i]->shape());
  }

  memory::desc input_md = ShapeToMemDesc(input_shape);
  memory::desc weights_md = ShapeToMemDesc(weight_shape);
  memory::desc output_md = ShapeToMemDesc(output_shape);

  memory::dims strides = GetPrimitiveParameter(conv_config.window().strides());
  memory::dims pad_left =
      GetPrimitiveParameter(conv_config.window().pad_left());
  memory::dims pad_right =
      GetPrimitiveParameter(conv_config.window().pad_right());
  memory::dims rhs_dilations =
      GetPrimitiveParameter(conv_config.window().window_dilations(), 1);

  uint64_t groups = conv_config.feature_groups();

  memory::desc new_inp_md = input_md.permute_axes(ComputePermutations(
      conv_config.dims(), conv_config.input().data().batch_dim(),
      conv_config.input().data().feature_dim(),
      conv_config.input().data().spatial_dims()));
  memory::desc new_ker_md = weights_md.permute_axes(ComputePermutations(
      conv_config.dims(), conv_config.kernel().filter().output_feature_dim(),
      conv_config.kernel().filter().input_feature_dim(),
      conv_config.kernel().filter().spatial_dims()));
  memory::desc new_out_md = output_md.permute_axes(ComputePermutations(
      conv_config.dims(), conv_config.output().data().batch_dim(),
      conv_config.output().data().feature_dim(),
      conv_config.output().data().spatial_dims()));

  if (groups > 1) {
    memory::dims corr_dims = new_ker_md.get_dims();
    corr_dims.insert(corr_dims.begin(), 1, groups);
    corr_dims[1] = corr_dims[1] / groups;
    new_ker_md = new_ker_md.reshape(corr_dims);
  }

  std::vector<memory::desc> fused_mds;
  for (const Shape& shape : fused_shapes) {
    memory::desc mem_desc = ShapeToMemDesc(shape);
    // The post-op argument must be oriented consistently with the output memory
    // descriptor.
    // Bias inputs are one-dimensional, as required by oneDNN. The broadcast
    // dimensions of all binary post-op inputs are expanded in the rewriter.
    // Hence, this condition should hold for all non-bias inputs.
    if (mem_desc.get_ndims() == new_out_md.get_ndims()) {
      mem_desc = mem_desc.permute_axes(ComputePermutations(
          conv_config.dims(), conv_config.output().data().batch_dim(),
          conv_config.output().data().feature_dim(),
          conv_config.output().data().spatial_dims()));
    }
    fused_mds.push_back(mem_desc);
  }

  memory::desc bias_md = memory::desc();

  dnnl::post_ops post_ops =
      PopulateOneDnnPostOps(dnnl::engine(dnnl::engine::kind::cpu, 0), fused_mds,
                            &conv_config.fusions(), nullptr, &bias_md);

  memory::desc any_ker_md =
      memory::desc(new_ker_md.get_dims(), new_ker_md.get_data_type(),
                   dnnl::memory::format_tag::any);
  memory::desc any_inp_md =
      memory::desc(new_inp_md.get_dims(), new_inp_md.get_data_type(),
                   GetFormatTag(new_inp_md.get_ndims()));
  memory::desc any_res_md =
      memory::desc(new_out_md.get_dims(), new_out_md.get_data_type(),
                   GetFormatTag(new_out_md.get_ndims()));

  dnnl::primitive_attr attrs;

  if (conv_config.optimization_config().user_scratchpad()) {
    attrs.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  }

  if (post_ops.len() > 0) {
    attrs.set_post_ops(post_ops);
  }

  return std::make_unique<convolution_forward::primitive_desc>(
      dnnl::engine(dnnl::engine::kind::cpu, 0), prop_kind::forward_inference,
      algorithm::convolution_direct, any_inp_md, any_ker_md, bias_md,
      any_res_md, strides, rhs_dilations, pad_left, pad_right, attrs);
}

void ExecuteOneDnnConvolution(absl::Span<MemrefInfoHandler> arguments,
                              absl::Span<MemrefInfoHandler> results,
                              OneDnnConvolutionConfig conv_config,
                              const dnnl::engine& cpu_engine,
                              dnnl::stream& onednn_stream,
                              OneDnnResources& resources) {
  MemrefInfo inp_minfo(arguments[0].get());
  MemrefInfo ker_minfo(arguments[1].get());
  MemrefInfo res_minfo(results[0].get());

  memory::desc inp_md = inp_minfo.GetOneDnnMemDesc();
  memory::desc ker_md = ker_minfo.GetOneDnnMemDesc();
  memory::desc res_md = res_minfo.GetOneDnnMemDesc();

  memory::desc new_inp_md = inp_md.permute_axes(ComputePermutations(
      conv_config.dims(), conv_config.input().data().batch_dim(),
      conv_config.input().data().feature_dim(),
      conv_config.input().data().spatial_dims()));
  memory::desc new_ker_md = ker_md.permute_axes(ComputePermutations(
      conv_config.dims(), conv_config.kernel().filter().output_feature_dim(),
      conv_config.kernel().filter().input_feature_dim(),
      conv_config.kernel().filter().spatial_dims()));
  memory::desc new_res_md = res_md.permute_axes(ComputePermutations(
      conv_config.dims(), conv_config.output().data().batch_dim(),
      conv_config.output().data().feature_dim(),
      conv_config.output().data().spatial_dims()));

  memory::dims strides = GetPrimitiveParameter(conv_config.window().strides());
  memory::dims pad_left =
      GetPrimitiveParameter(conv_config.window().pad_left());
  memory::dims pad_right =
      GetPrimitiveParameter(conv_config.window().pad_right());
  memory::dims rhs_dilations =
      GetPrimitiveParameter(conv_config.window().window_dilations(), 1);

  uint64_t groups = conv_config.feature_groups();

  if (groups > 1) {
    auto corr_dims = new_ker_md.get_dims();
    corr_dims.insert(corr_dims.begin(), 1, groups);
    corr_dims[1] = corr_dims[1] / groups;
    new_ker_md = new_ker_md.reshape(corr_dims);
  }

  const int64_t num_fused_operands = arguments.size() - 2;
  std::vector<memory::desc> fused_mds;
  std::vector<void*> fused_bufs;
  for (int64_t i = 0; i < num_fused_operands; ++i) {
    MemrefInfo operand_minfo(arguments[i + 2].get());
    memory::desc mem_desc = operand_minfo.GetOneDnnMemDesc();
    if (mem_desc.get_ndims() == new_res_md.get_ndims()) {
      mem_desc = mem_desc.permute_axes(ComputePermutations(
          conv_config.dims(), conv_config.output().data().batch_dim(),
          conv_config.output().data().feature_dim(),
          conv_config.output().data().spatial_dims()));
    }
    fused_mds.push_back(mem_desc);
    fused_bufs.push_back(operand_minfo.Data());
  }

  FusedOperandsRef fused_operands_ref{fused_bufs, resources.postop_args};

  memory::desc bias_md = memory::desc();

  dnnl::post_ops post_ops =
      PopulateOneDnnPostOps(cpu_engine, fused_mds, &conv_config.fusions(),
                            &fused_operands_ref, &bias_md);

  auto any_ker_md =
      memory::desc(new_ker_md.get_dims(), new_ker_md.get_data_type(),
                   dnnl::memory::format_tag::any);
  auto any_inp_md =
      memory::desc(new_inp_md.get_dims(), new_inp_md.get_data_type(),
                   GetFormatTag(new_inp_md.get_ndims()));
  auto any_res_md =
      memory::desc(new_res_md.get_dims(), new_res_md.get_data_type(),
                   GetFormatTag(new_res_md.get_ndims()));

  dnnl::primitive_attr attrs;

  if (conv_config.optimization_config().user_scratchpad()) {
    attrs.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  }

  if (post_ops.len() > 0) {
    attrs.set_post_ops(post_ops);
  }

  auto conv_pd = std::make_unique<convolution_forward::primitive_desc>(
      cpu_engine, prop_kind::forward_inference, algorithm::convolution_direct,
      any_inp_md, any_ker_md, bias_md, any_res_md, strides, rhs_dilations,
      pad_left, pad_right, attrs);

  auto inp_mem = memory(new_inp_md, cpu_engine, inp_minfo.Data());
  auto ker_mem = memory(new_ker_md, cpu_engine, ker_minfo.Data());
  auto res_mem = memory(new_res_md, cpu_engine, res_minfo.Data());

  resources.src_mem = (conv_pd->src_desc() == inp_mem.get_desc())
                          ? inp_mem
                          : ReorderMemory(cpu_engine, conv_pd->src_desc(),
                                          inp_mem, onednn_stream);
  resources.wei_mem = (conv_pd->weights_desc() == ker_mem.get_desc())
                          ? ker_mem
                          : ReorderMemory(cpu_engine, conv_pd->weights_desc(),
                                          ker_mem, onednn_stream);
  resources.dst_mem = (conv_pd->dst_desc() == res_mem.get_desc())
                          ? res_mem
                          : memory(conv_pd->dst_desc(), cpu_engine);

  resources.primitive = primitive(*conv_pd);

  std::unordered_map<int, memory> conv_args{
      {DNNL_ARG_SRC, resources.src_mem},
      {DNNL_ARG_WEIGHTS, resources.wei_mem},
      {DNNL_ARG_DST, resources.dst_mem}};

  if (conv_config.optimization_config().user_scratchpad()) {
    CHECK_GT(results.size(), 1);
    MemrefInfo scratch_minfo(results[1].get());

    size_t required_size = conv_pd->scratchpad_desc().get_size();
    size_t provided_size = scratch_minfo.GetOneDnnDims()[0];  // bytes (u8)
    CHECK_LE(required_size, provided_size);

    resources.scratch_mem =
        memory(conv_pd->scratchpad_desc(), cpu_engine, scratch_minfo.Data());
    conv_args.insert({DNNL_ARG_SCRATCHPAD, resources.scratch_mem});
  }

  conv_args.insert(resources.postop_args.begin(), resources.postop_args.end());
  resources.primitive.execute(onednn_stream, conv_args);

  if (conv_pd->dst_desc() == res_mem.get_desc()) {
    res_mem = resources.dst_mem;
  } else {
    dnnl::reorder(resources.dst_mem, res_mem)
        .execute(onednn_stream, resources.dst_mem, res_mem);
  }
}

}  // namespace cpu
}  // namespace xla
