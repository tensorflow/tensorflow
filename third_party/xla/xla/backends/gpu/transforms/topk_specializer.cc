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

#include "xla/backends/gpu/transforms/topk_specializer.h"

#include <stddef.h>

#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/tuple_util.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

// Broadcast a 32-bit scalar to a target shape.
HloInstruction* BroadcastU32(HloComputation* comp, const Shape& target_shape,
                             uint32_t value) {
  HloInstruction* constant = comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint32_t>(value)));
  return comp->AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::ChangeElementType(target_shape, U32), constant, {}));
}

// Broadcast a 64-bit scalar to a target shape.
HloInstruction* BroadcastU64(HloComputation* comp, const Shape& target_shape,
                             uint64_t value) {
  HloInstruction* constant = comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint64_t>(value)));
  return comp->AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::ChangeElementType(target_shape, U64), constant, {}));
}

// Emits HLO to pack F32 values and Iota indices into U64 priority keys.
HloInstruction* BuildPackF32ToU64(HloInstruction* data, HloComputation* comp) {
  const Shape& shape = data->shape();
  int64_t iota_dim = shape.dimensions().size() - 1;

  Shape u32_shape = ShapeUtil::ChangeElementType(shape, U32);
  Shape s32_shape = ShapeUtil::ChangeElementType(shape, S32);
  Shape u64_shape = ShapeUtil::ChangeElementType(shape, U64);

  // 1. Generate reversed indices: 0xFFFFFFFF - iota
  HloInstruction* iota_u32 =
      comp->AddInstruction(HloInstruction::CreateIota(u32_shape, iota_dim));
  HloInstruction* broadcast_ff = BroadcastU32(comp, shape, 0xFFFFFFFF);
  HloInstruction* iota_neg = comp->AddInstruction(HloInstruction::CreateBinary(
      u32_shape, HloOpcode::kSubtract, broadcast_ff, iota_u32));

  // 2. Pure Bitwise Radix Float Flip (F32 -> U32)
  // s32_val = bitcast(data)
  HloInstruction* s32_val = comp->AddInstruction(
      HloInstruction::CreateBitcastConvert(s32_shape, data));

  // sign_smeared = s32_val >> 31 (arithmetic shift)
  HloInstruction* const_31_s32 = comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(31)));
  HloInstruction* broadcast_31_s32 = comp->AddInstruction(
      HloInstruction::CreateBroadcast(s32_shape, const_31_s32, {}));
  HloInstruction* sign_smeared = comp->AddInstruction(
      HloInstruction::CreateBinary(s32_shape, HloOpcode::kShiftRightArithmetic,
                                   s32_val, broadcast_31_s32));

  // mask = bitcast(sign_smeared) | 0x80000000
  HloInstruction* sign_smeared_u32 = comp->AddInstruction(
      HloInstruction::CreateBitcastConvert(u32_shape, sign_smeared));
  HloInstruction* broadcast_8 = BroadcastU32(comp, shape, 0x80000000);
  HloInstruction* mask = comp->AddInstruction(HloInstruction::CreateBinary(
      u32_shape, HloOpcode::kOr, sign_smeared_u32, broadcast_8));

  // radix_key = bitcast(data) ^ mask
  HloInstruction* u32_val = comp->AddInstruction(
      HloInstruction::CreateBitcastConvert(u32_shape, data));
  HloInstruction* radix_key = comp->AddInstruction(
      HloInstruction::CreateBinary(u32_shape, HloOpcode::kXor, u32_val, mask));

  // 3. Pack into U64: (radix_key << 32) | iota_neg
  HloInstruction* val_u64 =
      comp->AddInstruction(HloInstruction::CreateConvert(u64_shape, radix_key));
  HloInstruction* broadcast_32_u64 = BroadcastU64(comp, shape, 32);
  HloInstruction* val_u64_top =
      comp->AddInstruction(HloInstruction::CreateBinary(
          u64_shape, HloOpcode::kShiftLeft, val_u64, broadcast_32_u64));

  HloInstruction* iota_neg_u64 =
      comp->AddInstruction(HloInstruction::CreateConvert(u64_shape, iota_neg));

  return comp->AddInstruction(HloInstruction::CreateBinary(
      u64_shape, HloOpcode::kOr, val_u64_top, iota_neg_u64));
}

// Emits HLO to unpack U64 priority keys back to F32 values.
HloInstruction* BuildUnpackU64ToF32(HloInstruction* u64_values,
                                    HloComputation* comp) {
  const Shape& shape = u64_values->shape();
  Shape u32_shape = ShapeUtil::ChangeElementType(shape, U32);
  Shape f32_shape = ShapeUtil::ChangeElementType(shape, F32);

  // 1. Shift right by 32 and cast to U32 to extract the radix_key
  HloInstruction* broadcast_32_u64 = BroadcastU64(comp, shape, 32);
  HloInstruction* rshift = comp->AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kShiftRightLogical, u64_values, broadcast_32_u64));
  HloInstruction* radix_key =
      comp->AddInstruction(HloInstruction::CreateConvert(u32_shape, rshift));

  // 2. Isolate the MSB: msb = radix_key >> 31 (Logical shift)
  HloInstruction* const_31_u32 = comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint32_t>(31)));
  HloInstruction* broadcast_31_u32 = comp->AddInstruction(
      HloInstruction::CreateBroadcast(u32_shape, const_31_u32, {}));
  HloInstruction* msb = comp->AddInstruction(HloInstruction::CreateBinary(
      u32_shape, HloOpcode::kShiftRightLogical, radix_key, broadcast_31_u32));

  // 3. Reconstruct mask: msb_minus_one = msb - 1
  // If MSB was 1: 1 - 1 = 0x00000000.
  // If MSB was 0: 0 - 1 = 0xFFFFFFFF (unsigned underflow).
  HloInstruction* broadcast_1 = BroadcastU32(comp, shape, 1);
  HloInstruction* msb_minus_one =
      comp->AddInstruction(HloInstruction::CreateBinary(
          u32_shape, HloOpcode::kSubtract, msb, broadcast_1));

  // 4. unmask = msb_minus_one | 0x80000000
  HloInstruction* broadcast_8 = BroadcastU32(comp, shape, 0x80000000);
  HloInstruction* unmask = comp->AddInstruction(HloInstruction::CreateBinary(
      u32_shape, HloOpcode::kOr, msb_minus_one, broadcast_8));

  // 5. Unflip the bits: original_u32 = radix_key ^ unmask
  HloInstruction* original_u32 =
      comp->AddInstruction(HloInstruction::CreateBinary(
          u32_shape, HloOpcode::kXor, radix_key, unmask));

  // 6. Bitcast back to F32
  return comp->AddInstruction(
      HloInstruction::CreateBitcastConvert(f32_shape, original_u32));
}

// Checks if we can safely route stable TopK to RAFT using the Uint64 adapter.
bool ShouldRewriteStableTopKToUint64(HloCustomCallInstruction* topk) {
  if (!hlo_instruction_utils::IsTopKStable(topk)) {
    return false;
  }

  Shape data_shape = topk->operand(0)->shape();
  PrimitiveType dtype = data_shape.element_type();
  if (!(dtype == F32 || dtype == BF16)) {
    return false;  // Only F32 and BF16 are supported for now.
  }

  bool has_batch = data_shape.dimensions().size() == 2;
  size_t batch = has_batch ? data_shape.dimensions(0) : 1;
  size_t n = data_shape.dimensions(has_batch ? 1 : 0);
  size_t k = topk->shape().tuple_shapes(0).dimensions(has_batch ? 1 : 0);

  double ratio = static_cast<double>(k) / n;
  if (ratio >= 0.85) {
    return false;
  }

  // Use built-in stable XLA GPU TopK kernel if n/k ranges are supported
  if (n >= 1024 && k <= 16) {
    return false;
  }

  // Upper bounds for using RAFT select_k.
  // The heuristic for deciding when to use Raft select_k versus Sort + Slice
  // was developed as part of the initial research in b/409009349
  size_t max_k = 128;
  if (dtype == F32) {
    max_k = 128;
    if (batch >= 64 && n >= 16384) {
      max_k = 256;
    }
  } else if (dtype == BF16) {
    max_k = 128;
    if (batch >= 16 && n >= 65536) {
      max_k = 256;
    }
    if (batch >= 64 && batch <= 128 && n >= 8192 && n <= 32768) {
      max_k = 64;
    }
  }
  if (k > max_k) {
    return false;
  }
  return true;
}

absl::StatusOr<HloInstruction*> RewriteStableTopKToUint64(
    HloCustomCallInstruction* topk) {
  HloComputation* comp = topk->parent();
  HloInstruction* data = topk->mutable_operand(0);
  PrimitiveType original_type = data->shape().element_type();

  // 1. Pack
  // If BF16, upcast to F32 first to match F32 bitmath logic.
  if (original_type == BF16) {
    data = comp->AddInstruction(HloInstruction::CreateConvert(
        ShapeUtil::ChangeElementType(data->shape(), F32), data));
  }
  HloInstruction* packed_u64 = BuildPackF32ToU64(data, comp);

  // 2. Create the specialized __gpu$TopK custom call
  Shape k_shape =
      ShapeUtil::ChangeElementType(topk->shape().tuple_shapes(0), U64);
  Shape idx_shape = topk->shape().tuple_shapes(1);
  Shape new_cc_shape = ShapeUtil::MakeTupleShape({k_shape, idx_shape});

  HloInstruction* new_topk =
      comp->AddInstruction(HloInstruction::CreateCustomCall(
          new_cc_shape, {packed_u64}, topk->to_apply(), "__gpu$TopK", "",
          CustomCallApiVersion::API_VERSION_TYPED_FFI));

  // The packed U64 keys guarantee uniqueness, making ties impossible.
  // Therefore, the inner TopK operation no longer requires stability to
  // produce a stable overall result. We clear the is_stable flag so the
  // backend can freely route this to the fast unstable topk kernel (RAFT lib).
  new_topk->set_raw_backend_config_string("{is_stable = false}");

  // 3. Unpack values and retain indices
  HloInstruction* u64_vals = comp->AddInstruction(
      HloInstruction::CreateGetTupleElement(k_shape, new_topk, 0));
  HloInstruction* indices = comp->AddInstruction(
      HloInstruction::CreateGetTupleElement(idx_shape, new_topk, 1));

  // Unpack to F32.
  HloInstruction* unpacked_data = BuildUnpackU64ToF32(u64_vals, comp);

  // If the original input was BF16, downcast the F32 result back to BF16.
  if (original_type == BF16) {
    unpacked_data = comp->AddInstruction(HloInstruction::CreateConvert(
        ShapeUtil::ChangeElementType(unpacked_data->shape(), BF16),
        unpacked_data));
  }

  return comp->AddInstruction(
      HloInstruction::CreateTuple({unpacked_data, indices}));
}

absl::StatusOr<HloInstruction*> SmallBufferOptimization(
    HloCustomCallInstruction* topk, bool is_cuda) {
  Shape data_shape = topk->operand(0)->shape();
  auto dtype = data_shape.element_type();
  auto supported_dtypes = {F32, BF16};
  if (!absl::c_linear_search(supported_dtypes, dtype)) {
    return InvalidArgument("Invalid Dtype: %s",
                           primitive_util::LowercasePrimitiveTypeName(dtype));
  }
  // We only support topk of the shape [x] or [batch, x].
  if (data_shape.dimensions().size() > 2) {
    return InvalidArgument("Invalid input dimensions: %s",
                           data_shape.ToString());
  }
  bool has_batch = data_shape.dimensions().size() == 2;
  size_t max_k = 16;  // CustomCall TopK requires k <= 16 and n >= 1024
  size_t min_n = 1024;
  size_t batch = 0;
  if (has_batch) {
    batch = data_shape.dimensions(0);
  }
  size_t n = data_shape.dimensions(has_batch ? 1 : 0);
  size_t k = topk->shape().tuple_shapes(0).dimensions(has_batch ? 1 : 0);
  double ratio = static_cast<double>(k) / n;
  if (ratio >= 0.85) {
    return InvalidArgument(
        "k/n ratio (%f) is too high for TopK. Falling back to sort + slice.",
        ratio);
  }
  // Enable RAFT if TopK is_stable = false.
  bool use_raft = !hlo_instruction_utils::IsTopKStable(topk);

  if (is_cuda && use_raft) {
    // The heuristic for deciding when to use Raft select_k versus Sort + Slice
    // was developed as part of the initial research in b/409009349
    if (dtype == F32) {
      min_n = 1;
      max_k = 128;
      if (batch >= 64 && n >= 16384) {
        max_k = 256;
      }
    } else if (dtype == BF16) {
      min_n = 1;
      max_k = 128;
      if (batch >= 16 && n >= 65536) {
        max_k = 256;
      }
      if (batch >= 64 && batch <= 128 && n >= 8192 && n <= 32768) {
        max_k = 64;
      }
    }
  }

  if (k > max_k) {
    return InvalidArgument("k too large (%d), must be <= %d", k, max_k);
  }
  if (n < min_n) {
    return InvalidArgument("Input too small (n=%d, min_n=%d)", n, min_n);
  }
  HloComputation* comp = topk->parent();
  HloInstruction* new_topk =
      comp->AddInstruction(HloInstruction::CreateCustomCall(
          topk->shape(), topk->operands(),
          // We don't need the original to_apply, but keeping it around allows
          // us to round-trip this CustomCall on tests.
          topk->to_apply(), "__gpu$TopK",
          /*opaque=*/"", CustomCallApiVersion::API_VERSION_TYPED_FFI));
  new_topk->set_raw_backend_config_string(topk->raw_backend_config_string());
  return TupleUtil::ExtractPrefix(new_topk, 2);
}

class SpecializeTopkVisitor : public DfsHloRewriteVisitor {
 public:
  explicit SpecializeTopkVisitor(se::GpuComputeCapability compute_capability)
      : compute_capability_(std::move(compute_capability)) {}

  absl::Status HandleCustomCall(HloInstruction* inst) override {
    HloCustomCallInstruction* topk = DynCast<HloCustomCallInstruction>(inst);
    if (topk == nullptr || topk->custom_call_target() != "TopK" ||
        compute_capability_.IsOneAPI()) {
      return absl::OkStatus();
    }
    TF_RET_CHECK(topk->operand_count() == 1);
    bool is_cuda = compute_capability_.IsCuda();
    // Route stable TopK to RAFT select_k via Uint64 adapter
    if (is_cuda && ShouldRewriteStableTopKToUint64(topk)) {
      ABSL_ASSIGN_OR_RETURN(HloInstruction * new_topk,
                       RewriteStableTopKToUint64(topk));
      return ReplaceInstruction(topk, new_topk);
    }

    if (auto small_topk = SmallBufferOptimization(topk, is_cuda);
        small_topk.ok()) {
      return ReplaceInstruction(topk, *small_topk);
    } else {  // NOLINT(readability-else-after-return)
      VLOG(2) << "Small TopK optimization doesn't match: "
              << small_topk.status();
    }

    return absl::OkStatus();
  }

 private:
  se::GpuComputeCapability compute_capability_;
};

}  // namespace

absl::StatusOr<bool> TopkSpecializer::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return SpecializeTopkVisitor(compute_capability_)
      .RunOnModule(module, execution_threads);
}

}  // namespace xla::gpu
