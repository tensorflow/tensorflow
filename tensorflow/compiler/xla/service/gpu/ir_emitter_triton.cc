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

#include "tensorflow/compiler/xla/service/gpu/ir_emitter_triton.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "llvm/Linker/Linker.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter_triton.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/logging.h"
#include "triton/codegen/pass.h"
#include "triton/codegen/target.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/context.h"
#include "triton/ir/instructions.h"
#include "triton/ir/module.h"
#include "triton/ir/type.h"
#include "triton/ir/value.h"

namespace xla {
namespace gpu {

namespace ir = triton::ir;

// XLA -> Triton type conversions.
ir::type* TritonType(ir::builder& b, PrimitiveType t) {
  switch (t) {
    case F64:
      return b.get_double_ty();
    case F32:
      return b.get_float_ty();
    case F16:
      return b.get_half_ty();
    case BF16:
      return b.get_bf16_ty();
    case S64:
      return b.get_int64_ty();
    case S32:
      return b.get_int32_ty();
    case S16:
      return b.get_int16_ty();
    case PRED:
      // Treat PRED as S8.
    case S8:
      return b.get_int8_ty();
    default:
      LOG(FATAL) << "This type is not supported yet: "
                 << primitive_util::LowercasePrimitiveTypeName(t);
  }
}

// Annotation for Triton compiler.
void SetMultipleOf(ir::value* instruction, std::vector<unsigned> value) {
  dynamic_cast<ir::instruction*>(instruction)
      ->set_metadata(ir::metadata::multiple_of, value);
}

// Annotation for Triton compiler.
void SetMaxContiguous(ir::value* instruction, std::vector<unsigned> value) {
  dynamic_cast<ir::instruction*>(instruction)
      ->set_metadata(ir::metadata::max_contiguous, value);
}

// Triton type conversions.
ir::value* Cast(ir::builder& b, ir::value* value, ir::type* dst_ty) {
  ir::type* src_ty = value->get_type();
  if (src_ty->is_block_ty() && !dst_ty->is_block_ty()) {
    dst_ty = ir::block_type::get_same_shapes(dst_ty, src_ty);
  }
  if (src_ty == dst_ty) {
    return value;
  }
  ir::type* src_scalar_ty = src_ty->get_scalar_ty();
  ir::type* dst_scalar_ty = dst_ty->get_scalar_ty();

  // bf16 <=> (non-fp32).
  if ((src_scalar_ty->is_bf16_ty() && !dst_scalar_ty->is_fp32_ty()) ||
      (dst_scalar_ty->is_bf16_ty() && !src_scalar_ty->is_fp32_ty())) {
    return Cast(b, Cast(b, value, b.get_float_ty()), dst_scalar_ty);
  }

  // FP Truncation.
  if (src_scalar_ty->is_floating_point_ty() &&
      dst_scalar_ty->is_floating_point_ty() &&
      src_scalar_ty->get_fp_mantissa_width() >
          dst_scalar_ty->get_fp_mantissa_width()) {
    return b.create_fp_trunc(value, dst_ty);
  }

  // FP Extension.
  if (src_scalar_ty->is_floating_point_ty() &&
      dst_scalar_ty->is_floating_point_ty() &&
      src_scalar_ty->get_fp_mantissa_width() <
          dst_scalar_ty->get_fp_mantissa_width()) {
    return b.create_fp_ext(value, dst_ty);
  }

  // int => float.
  if (src_scalar_ty->is_integer_ty() && dst_scalar_ty->is_floating_point_ty()) {
    // TODO(b/266862493): Support unsigned integer types.
    return b.create_si_to_fp(value, dst_ty);
  }

  // float => int.
  if (src_scalar_ty->is_floating_point_ty() && dst_scalar_ty->is_integer_ty()) {
    // TODO(b/266862493): Support unsigned integer types.
    return b.create_fp_to_si(value, dst_ty);
  }

  LOG(FATAL) << "Type conversion not supported: "
             << src_scalar_ty->get_type_id() << " -> "
             << dst_scalar_ty->get_type_id();
}

// Create a scalar constant of type 'ty' with value 'value'.
ir::value* ScalarConst(ir::builder& b, ir::type* ty, double value) {
  if (ty->is_floating_point_ty()) {
    return ir::constant_fp::get(ty, value);
  }
  return ir::constant_int::get(ty, value);
}

// Do 'x' + 'y' using float or integer instruction according to the data type.
ir::value* Add(ir::builder& b, ir::value* x, ir::value* y) {
  if (x->get_type()->get_scalar_ty()->is_floating_point_ty()) {
    CHECK(y->get_type()->get_scalar_ty()->is_floating_point_ty());
    return b.create_fadd(x, y);
  }
  return b.create_add(x, y);
}

// Variable naming: lhs [m, k] x rhs [k, n] -> out [m, n].
std::optional<LaunchDimensions> MatMul(
    ir::builder& b, const HloDotInstruction* dot_instr, ir::function* fn,
    tensorflow::AutotuneResult::TritonGemmKey& config, int shmem_budget) {
  const DotDimensionNumbers& dims = dot_instr->dot_dimension_numbers();
  const DotFusionAnalysis analysis(dot_instr);
  const HloInstruction* hlo_lhs_param = analysis.OperandToParameter(0);
  const HloInstruction* hlo_rhs_param = analysis.OperandToParameter(1);

  ir::type* lhs_ty = TritonType(b, hlo_lhs_param->shape().element_type());
  ir::type* rhs_ty = TritonType(b, hlo_rhs_param->shape().element_type());

  // Rely on dot decomposer: there is just one contracting and one
  // non-contracting dimension on each side + one batch optionally.
  CHECK_EQ(dims.lhs_contracting_dimensions_size(), 1);
  CHECK_EQ(dims.rhs_contracting_dimensions_size(), 1);
  CHECK_LE(dims.lhs_batch_dimensions_size(), 1);
  const bool batch = !dims.lhs_batch_dimensions().empty();
  CHECK_EQ(dot_instr->operand(0)->shape().rank(), 2 + batch);
  const int lhs_noncontracting_dim_idx =
      NoncontractingDimensionIndex(dims.lhs_contracting_dimensions(0),
                                   batch ? dims.lhs_batch_dimensions(0) : -1);
  const int rhs_noncontracting_dim_idx =
      NoncontractingDimensionIndex(dims.rhs_contracting_dimensions(0),
                                   batch ? dims.rhs_batch_dimensions(0) : -1);

  // Non-contracting dimension lengths.
  // Just the fastest-varying part of it if the dimension is split.
  const int m = analysis.IterSpec(0, lhs_noncontracting_dim_idx)[0].count;
  const int n = analysis.IterSpec(1, rhs_noncontracting_dim_idx)[0].count;

  // Contracting dimension length.
  const int k = dot_instr->operand(0)->shape().dimensions(
      dims.lhs_contracting_dimensions(0));

  // LHS non-contracting can be split into two.
  const bool lhs_nc_split =
      (analysis.IterSpec(0, lhs_noncontracting_dim_idx).size() > 1);
  CHECK_EQ(analysis.IterSpec(0, lhs_noncontracting_dim_idx).size(),
           1 + lhs_nc_split);
  // For now split and batch are not supported simultaneously because they
  // are implemented via same mechanism.
  CHECK_LE(batch + lhs_nc_split, 1);
  // Splitting of the other ones is not supported yet.
  CHECK_EQ(analysis.IterSpec(1, rhs_noncontracting_dim_idx).size(), 1);
  CHECK_EQ(analysis.IterSpec(0, dims.lhs_contracting_dimensions(0)).size(), 1);

  const int stride_lhs_m =
      analysis.IterSpec(0, lhs_noncontracting_dim_idx)[0].stride;
  const int stride_lhs_k =
      analysis.IterSpec(0, dims.lhs_contracting_dimensions(0))[0].stride;
  const int stride_rhs_k =
      analysis.IterSpec(1, dims.rhs_contracting_dimensions(0))[0].stride;
  const int stride_rhs_n =
      analysis.IterSpec(1, rhs_noncontracting_dim_idx)[0].stride;

  // Either batch size or upper part of the length of a split nc dimension.
  int batch_size = 1;
  int stride_batch_lhs = 0;
  int stride_batch_rhs = 0;
  if (lhs_nc_split) {
    batch_size = analysis.IterSpec(0, lhs_noncontracting_dim_idx)[1].count;
    stride_batch_lhs =
        analysis.IterSpec(0, lhs_noncontracting_dim_idx)[1].stride;
    stride_batch_rhs = 0;
  } else if (batch) {
    // Batch dimension should have same length left and right.
    CHECK_EQ(analysis.IterSpec(0, dims.lhs_batch_dimensions(0))[0].count,
             analysis.IterSpec(1, dims.rhs_batch_dimensions(0))[0].count);
    batch_size = analysis.IterSpec(0, dims.lhs_batch_dimensions(0))[0].count;
    stride_batch_lhs =
        analysis.IterSpec(0, dims.lhs_batch_dimensions(0))[0].stride;
    stride_batch_rhs =
        analysis.IterSpec(1, dims.rhs_batch_dimensions(0))[0].stride;
  }

  constexpr int64_t group_m = 8;

  bool transpose_output =
      !LayoutUtil::IsMonotonicWithDim0Major(dot_instr->shape().layout());
  const int stride_out_m = transpose_output ? 1 : n;
  const int stride_out_n = transpose_output ? m : 1;

  const unsigned int block_m = config.block_m();
  const unsigned int block_k = config.block_k();
  const unsigned int block_n = config.block_n();
  CHECK_GE(block_m, 32);
  CHECK_GE(block_k, 32);
  CHECK_GE(block_n, 32);

  VLOG(3) << block_m << " " << block_k << " " << block_n << " "
          << config.num_warps() << " " << config.num_stages();

  const int grid_m = ceil(1.0 * m / block_m);
  const int grid_n = ceil(1.0 * n / block_n);
  const int width = group_m * grid_n;

  // TODO(b/266863137): handle atomic add for split_k > 1.
  // This also requires output zero-init.
  const unsigned int split_k = config.split_k();
  CHECK_EQ(split_k, 1);
  const LaunchDimensions launch_dimensions{
      {grid_m * grid_n, split_k, batch_size},
      {config.num_warps() * WarpSize(), 1, 1}};
  ir::type* root_ty = TritonType(b, dot_instr->shape().element_type());
  // Data type to which dot() inputs are converted.
  ir::type* dot_ty = b.get_float_ty();
  if (lhs_ty->is_fp32_ty() || rhs_ty->is_fp32_ty()) {
    dot_ty = b.get_float_ty();
  } else if (lhs_ty->is_bf16_ty() || rhs_ty->is_bf16_ty()) {
    dot_ty = b.get_bf16_ty();
  } else if (lhs_ty->is_fp16_ty() || rhs_ty->is_fp16_ty()) {
    dot_ty = b.get_half_ty();
  }

  const int required_shmem_size =
      (block_m * lhs_ty->get_primitive_size_in_bits() +
       block_n * rhs_ty->get_primitive_size_in_bits()) *
      block_k * config.num_stages() / 8;
  // TODO(b/266857785): Add dynamic shared memory size.
  if (required_shmem_size > shmem_budget) {
    VLOG(2) << "Requires too much shared memory: " << required_shmem_size
            << " B.";
    return std::nullopt;
  }

  // TODO(b/266862493): Accumulator can be integer too.
  // Otherwise only f64 x f64 -> f64 uses f64 accumulator.
  ir::type* acc_ty = (root_ty->is_fp64_ty() && dot_ty->is_fp64_ty())
                         ? b.get_double_ty()
                         : b.get_float_ty();
  ir::value* lhs = fn->args()[hlo_lhs_param->parameter_number()];
  ir::value* rhs = fn->args()[hlo_rhs_param->parameter_number()];
  ir::value* out = fn->args().back();

  ir::value* pid0 = b.create_get_program_id(0);
  ir::value* pid1 = b.create_get_program_id(1);
  ir::value* pid2 = b.create_get_program_id(2);
  ir::value* group_id = b.create_sdiv(pid0, b.get_int32(width));
  ir::value* first_pid_m = b.create_mul(group_id, b.get_int32(group_m));
  ir::value* sub0 = b.create_sub(b.get_int32(grid_m), first_pid_m);
  ir::value* group_size = b.create_select(
      b.create_icmpSLT(sub0, b.get_int32(group_m)), sub0, b.get_int32(group_m));

  auto range_m = [&]() {
    ir::value* pid_m =
        b.create_add(first_pid_m, b.create_srem(pid0, group_size));
    ir::value* pid_m_stride = b.create_mul(pid_m, b.get_int32(block_m));
    return b.create_add(b.create_splat(pid_m_stride, {block_m}),
                        b.get_range(0, block_m));
  };

  auto range_n = [&]() {
    ir::value* pid_n =
        b.create_sdiv(b.create_srem(pid0, b.get_int32(width)), group_size);
    ir::value* pid_n_stride = b.create_mul(pid_n, b.get_int32(block_n));
    return b.create_add(b.create_splat(pid_n_stride, {block_n}),
                        b.get_range(0, block_n));
  };

  ir::value* range_k = b.create_add(
      b.create_splat(b.create_mul(pid1, b.get_int32(block_k)), {block_k}),
      b.get_range(0, block_k));

  std::vector<unsigned int> shape_m_1{block_m, 1};
  ir::value* range_lhs_m =
      b.create_srem(range_m(), b.create_splat(b.get_int32(m), {block_m}));
  SetMultipleOf(range_lhs_m, {block_m});
  SetMaxContiguous(range_lhs_m, {block_m});
  ir::value* lhs_offset_m =
      b.create_mul(b.create_reshape(range_lhs_m, shape_m_1),
                   b.create_splat(b.get_int32(stride_lhs_m), shape_m_1));
  std::vector<unsigned int> shape_1_k{1, block_k};
  ir::value* lhs_offset_k =
      b.create_mul(b.create_reshape(range_k, shape_1_k),
                   b.create_splat(b.get_int32(stride_lhs_k), shape_1_k));
  std::vector<unsigned int> shape_m_k{block_m, block_k};
  ir::value* lhs_offset =
      b.create_add(b.create_broadcast(lhs_offset_m, shape_m_k),
                   b.create_broadcast(lhs_offset_k, shape_m_k));
  ir::value* lhs_offset_batch =
      b.create_mul(pid2, b.get_int32(stride_batch_lhs));
  ir::value* lhs_ptrs_base = b.create_gep(lhs, {lhs_offset_batch});
  lhs_ptrs_base =
      b.create_gep(b.create_splat(lhs_ptrs_base, shape_m_k), {lhs_offset});

  std::vector<unsigned int> shape_k_1{block_k, 1};
  ir::value* rhs_off_k =
      b.create_mul(b.create_reshape(range_k, shape_k_1),
                   b.create_splat(b.get_int32(stride_rhs_k), shape_k_1));
  std::vector<unsigned int> shape_1_n{1, block_n};
  ir::value* range_rhs_n =
      b.create_srem(range_n(), b.create_splat(b.get_int32(n), {block_n}));
  SetMultipleOf(range_rhs_n, {block_n});
  SetMaxContiguous(range_rhs_n, {block_n});
  ir::value* rhs_offset_n =
      b.create_mul(b.create_reshape(range_rhs_n, shape_1_n),
                   b.create_splat(b.get_int32(stride_rhs_n), shape_1_n));
  std::vector<unsigned int> shape_k_n{block_k, block_n};
  ir::value* rhs_offset =
      b.create_add(b.create_broadcast(rhs_off_k, shape_k_n),
                   b.create_broadcast(rhs_offset_n, shape_k_n));
  ir::value* rhs_offset_batch =
      b.create_mul(pid2, b.get_int32(stride_batch_rhs));
  ir::value* rhs_ptrs_base = b.create_gep(rhs, {rhs_offset_batch});
  rhs_ptrs_base =
      b.create_gep(b.create_splat(rhs_ptrs_base, shape_k_n), {rhs_offset});

  std::vector<unsigned int> shape_m_n{block_m, block_n};
  ir::value* acc_init =
      b.create_splat(ir::constant_fp::get(acc_ty, 0), shape_m_n);

  ir::basic_block* entry_bb = b.get_insert_block();
  ir::basic_block* post_loop_bb =
      ir::basic_block::create(b.get_context(), "post_loop", fn);
  ir::basic_block* loop_bb =
      ir::basic_block::create(b.get_context(), "loop", fn, post_loop_bb);

  // - loop: for (int ki = K; ki > 0; ki -= BLOCK_K * SPLIT_K)

  // Triton compiler relies on last instruction before loop
  // being a conditional jump despite it doesn't really have to be.
  b.create_cond_br(b.get_int1(true), loop_bb, post_loop_bb);
  b.set_insert_point(loop_bb);

  ir::phi_node* ki = b.create_phi(b.get_int32_ty(), /*num_reserved=*/2);
  ir::phi_node* lhs_ptrs =
      b.create_phi(lhs_ptrs_base->get_type(), /*num_reserved=*/2);
  ir::phi_node* rhs_ptrs =
      b.create_phi(rhs_ptrs_base->get_type(), /*num_reserved=*/2);
  ir::phi_node* acc = b.create_phi(acc_init->get_type(), /*num_reserved=*/2);

  ki->add_incoming(b.get_int32(k), entry_bb);
  lhs_ptrs->add_incoming(lhs_ptrs_base, entry_bb);
  rhs_ptrs->add_incoming(rhs_ptrs_base, entry_bb);
  acc->add_incoming(acc_init, entry_bb);

  ir::value* lhs_tile;
  ir::value* rhs_tile;
  ir::value* zeros_like_lhs =
      b.create_splat(ScalarConst(b, lhs_ty, 0), shape_m_k);
  ir::value* zeros_like_rhs =
      b.create_splat(ScalarConst(b, rhs_ty, 0), shape_k_n);
  if (k % (block_k * split_k) == 0) {
    // Unmasked loads for even K.
    lhs_tile = b.create_load(lhs_ptrs, /*cache=*/ir::load_inst::NONE,
                             /*eviction=*/ir::load_inst::NORMAL,
                             /*is_volatile=*/false);
    rhs_tile = b.create_load(rhs_ptrs, /*cache=*/ir::load_inst::NONE,
                             /*eviction=*/ir::load_inst::NORMAL,
                             /*is_volatile=*/false);
  } else {
    // Masked loads.
    ir::value* lhs_mask = b.create_broadcast(
        b.create_icmpSLT(b.create_reshape(range_k, shape_1_k),
                         b.create_splat(ki, shape_1_k)),
        shape_m_k);
    lhs_tile = b.create_masked_load(lhs_ptrs, lhs_mask, zeros_like_lhs,
                                    /*cache=*/ir::load_inst::NONE,
                                    /*eviction=*/ir::load_inst::NORMAL,
                                    /*is_volatile=*/false);
    ir::value* rhs_mask = b.create_broadcast(
        b.create_icmpSLT(b.create_reshape(range_k, shape_k_1),
                         b.create_splat(ki, shape_k_1)),
        shape_k_n);
    rhs_tile = b.create_masked_load(rhs_ptrs, rhs_mask, zeros_like_rhs,
                                    /*cache=*/ir::load_inst::NONE,
                                    /*eviction=*/ir::load_inst::NORMAL,
                                    /*is_volatile=*/false);
  }
  // TODO(b/266857788): this + 0 is a workaround for
  // mixed-type pipelining bug in triton.
  lhs_tile = Add(b, lhs_tile, zeros_like_lhs);
  rhs_tile = Add(b, rhs_tile, zeros_like_rhs);

  lhs_tile = Cast(b, lhs_tile, dot_ty);
  rhs_tile = Cast(b, rhs_tile, dot_ty);

  ir::value* dot =
      b.create_dot(lhs_tile, rhs_tile,
                   b.create_splat(ir::constant_fp::get(acc_ty, 0), shape_m_n),
                   /*trans_a=*/false, /*trans_b=*/false,
                   /*allow_tf32=*/true);
  ir::value* acc_next = b.create_fadd(acc, dot);
  acc->add_incoming(acc_next, loop_bb);

  ir::value* lhs_ptrs_inc =
      b.create_splat(b.get_int32(block_k * split_k * stride_lhs_k), shape_m_k);
  lhs_ptrs->add_incoming(b.create_gep(lhs_ptrs, {lhs_ptrs_inc}), loop_bb);

  ir::value* rhs_ptrs_inc =
      b.create_splat(b.get_int32(block_k * split_k * stride_rhs_k), shape_k_n);
  rhs_ptrs->add_incoming(b.create_gep(rhs_ptrs, {rhs_ptrs_inc}), loop_bb);

  ir::value* k_next = b.create_add(ki, b.get_int32(-block_k * split_k));
  ki->add_incoming(k_next, loop_bb);
  b.create_cond_br(b.create_icmpSGT(k_next, b.get_int32(0)), loop_bb,
                   post_loop_bb);

  // - loop end

  b.set_insert_point(post_loop_bb);

  ir::phi_node* acc_final = b.create_phi(acc_next->get_type(), 2);
  acc_final->add_incoming(acc_next, loop_bb);
  acc_final->add_incoming(acc_init, entry_bb);

  // Output tile offsets.
  ir::value* out_offset_batch = b.create_mul(pid2, b.get_int32(m * n));
  ir::value* out_ptrs = b.create_gep(out, {out_offset_batch});
  ir::value* out_offset_m =
      b.create_mul(b.create_reshape(range_m(), shape_m_1),
                   b.create_splat(b.get_int32(stride_out_m), shape_m_1));
  out_ptrs = b.create_gep(b.create_splat(out_ptrs, shape_m_1), {out_offset_m});
  ir::value* out_offset_n =
      b.create_mul(b.create_reshape(range_n(), shape_1_n),
                   b.create_splat(b.get_int32(stride_out_n), shape_1_n));
  out_ptrs = b.create_gep(b.create_broadcast(out_ptrs, shape_m_n),
                          {b.create_broadcast(out_offset_n, shape_m_n)});

  // Output tile store mask: check that the indices are within [M, N].
  ir::value* rm_cmp =
      b.create_icmpSLT(b.create_reshape(range_m(), shape_m_1),
                       b.create_splat(b.get_int32(m), shape_m_1));
  ir::value* rn_cmp =
      b.create_icmpSLT(b.create_reshape(range_n(), shape_1_n),
                       b.create_splat(b.get_int32(n), shape_1_n));
  ir::value* mask = b.create_and(b.create_broadcast(rm_cmp, shape_m_n),
                                 b.create_broadcast(rn_cmp, shape_m_n));

  b.create_masked_store(out_ptrs, Cast(b, acc_final, root_ty), mask,
                        /*eviction=*/ir::store_inst::NORMAL);
  return launch_dimensions;
}

std::optional<LaunchDimensions> TritonWrapper(
    absl::string_view fn_name, const HloComputation* hlo_computation,
    const se::CudaComputeCapability& cc, const GpuDeviceInfo& device_info,
    AutotuneResult::TritonGemmKey& config, llvm::Module* llvm_module,
    std::function<std::optional<LaunchDimensions>(
        ir::builder&, const HloDotInstruction*, ir::function*,
        AutotuneResult::TritonGemmKey&, int)>
        generator) {
  ir::context triton_context;
  ir::builder b(triton_context);
  ir::module triton_module("", b);

  const HloInstruction* root =
      (hlo_computation->root_instruction()->opcode() == HloOpcode::kBitcast)
          ? hlo_computation->root_instruction()->operand(0)
          : hlo_computation->root_instruction();
  CHECK_EQ(root->opcode(), HloOpcode::kDot);
  VLOG(3) << root->parent()->ToString();

  VLOG(2) << config.DebugString();

  ir::type* root_ty = TritonType(
      b, hlo_computation->root_instruction()->shape().element_type());
  std::vector<ir::type*> fn_arg_types;
  for (HloInstruction* p : hlo_computation->parameter_instructions()) {
    fn_arg_types.push_back(
        ir::pointer_type::get(TritonType(b, p->shape().element_type()),
                              /*address_space=*/0));
  }
  fn_arg_types.push_back(ir::pointer_type::get(root_ty, /*address_space=*/0));
  ir::function* fn = triton_module.get_or_insert_function(
      std::string(fn_name),
      ir::function_type::get(b.get_void_ty(), fn_arg_types));
  for (int i = 0; i < fn->get_num_operands(); ++i) {
    fn->add_attr(i + 1, ir::attribute(ir::aligned, 16));
  }
  fn->set_is_kernel(true);
  b.set_insert_point(ir::basic_block::create(triton_context, "entry", fn));

  std::optional<LaunchDimensions> launch_dimensions =
      MatMul(b, ::xla::Cast<HloDotInstruction>(root), fn, config,
             device_info.shared_memory_per_block);
  if (!launch_dimensions.has_value()) {
    return std::nullopt;
  }
  b.create_ret_void();

  if (VLOG_IS_ON(4)) {
    std::ostringstream ttir;
    triton_module.print(ttir);
    XLA_VLOG_LINES(4, ttir.str());
  }

  triton::codegen::nvidia_cu_target target(cc.major * 10 + cc.minor);
  int shared_mem_bytes = 0;
  std::unique_ptr<llvm::Module> ll_triton_module =
      triton::codegen::add_passes_to_emit_bin(
          triton_module, llvm_module->getContext(), &target, config.num_warps(),
          config.num_stages(), shared_mem_bytes, /*extern_libs=*/{});
  VLOG(2) << "Shared memory usage: " << shared_mem_bytes << " B";
  // TODO(b/266857785): Add dynamic shared memory size.
  if (shared_mem_bytes > device_info.shared_memory_per_block) {
    LOG(WARNING) << "Shared memory size limit exceeded.";
    return std::nullopt;
  }
  launch_dimensions->SetSharedMemBytes(shared_mem_bytes);

  LogAndVerify(ll_triton_module.get());

  for (auto& metadata :
       llvm::make_early_inc_range(ll_triton_module->named_metadata())) {
    ll_triton_module->eraseNamedMDNode(&metadata);
  }
  ll_triton_module->setDataLayout(llvm_module->getDataLayout());
  CHECK(!llvm::Linker::linkModules(*llvm_module, std::move(ll_triton_module)));

  LogAndVerify(llvm_module);

  return launch_dimensions;
}

}  // namespace gpu
}  // namespace xla
