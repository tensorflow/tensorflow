/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/dynamic_slice_fusion.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

using Offset = DynamicSliceFusion::Offset;
using OffsetResolution = DynamicSliceFusion::OffsetResolution;

// Single-element arrays (e.g. `s32[1]`) are treated as scalars in offset
// expressions, because reshapes and bitcasts between them are no-ops.
static bool IsScalarInteger(const Shape& shape) {
  return ShapeUtil::IsEffectiveScalar(shape) &&
         primitive_util::IsIntegralType(shape.element_type());
}

static bool IsScalarIntegerOrPred(const Shape& shape) {
  return IsScalarInteger(shape) ||
         (ShapeUtil::IsEffectiveScalar(shape) && shape.element_type() == PRED);
}

bool Offset::IsExpr(const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kParameter:
    case HloOpcode::kConstant:
      return IsScalarIntegerOrPred(instr->shape());
    case HloOpcode::kBitcast:
    case HloOpcode::kReshape:
      // HLO bitcasts can change the element type (e.g. `s32[] bitcast(f32[])`),
      // only bitcasts and reshapes between single-element arrays of the same
      // element type are no-ops.
      return IsScalarIntegerOrPred(instr->shape()) &&
             ShapeUtil::SameElementType(instr->shape(),
                                        instr->operand(0)->shape());
    case HloOpcode::kAdd:
    case HloOpcode::kSubtract:
    case HloOpcode::kMultiply:
      return IsScalarInteger(instr->shape());
    case HloOpcode::kDivide:
    case HloOpcode::kRemainder:
    case HloOpcode::kMinimum:
    case HloOpcode::kMaximum:
    case HloOpcode::kClamp:
      // TODO(ezhulenev): Remove the flag check once the runtime support is
      // outside of the GPU forward compatibility window.
      return IsScalarInteger(instr->shape()) && instr->GetModule() &&
             instr->GetModule()
                 ->config()
                 .debug_options()
                 .xla_gpu_experimental_enable_dynamic_slice_extended_offsets();
    case HloOpcode::kConvert:
      // Offset verification reads offset parameters only as 32- or 64-bit
      // integers, so we don't convert from narrower types, which would make
      // them offset expression parameters.
      return IsScalarInteger(instr->shape()) &&
             IsScalarInteger(instr->operand(0)->shape()) &&
             primitive_util::BitWidth(
                 instr->operand(0)->shape().element_type()) >= 32;
    case HloOpcode::kCompare:
      return IsScalarInteger(instr->operand(0)->shape()) &&
             IsScalarInteger(instr->operand(1)->shape());
    case HloOpcode::kSelect:
      return IsScalarIntegerOrPred(instr->shape());
    default:
      return false;
  }
}

Offset::Expr Offset::Constant(int64_t value) {
  return {Offset::Expr::Constant{value}};
}

Offset::Expr Offset::Parameter(int64_t parameter_number) {
  return {Offset::Expr::Parameter{parameter_number}};
}

Offset::Expr Offset::Add(Offset::Expr lhs, Offset::Expr rhs) {
  return {Offset::Expr::Add{{std::move(lhs), std::move(rhs)}}};
}

Offset::Expr Offset::Subtract(Offset::Expr lhs, Offset::Expr rhs) {
  return {Offset::Expr::Subtract{{std::move(lhs), std::move(rhs)}}};
}

Offset::Expr Offset::Multiply(Offset::Expr lhs, Offset::Expr rhs) {
  return {Offset::Expr::Multiply{{std::move(lhs), std::move(rhs)}}};
}

Offset::Expr Offset::Divide(Offset::Expr lhs, Offset::Expr rhs) {
  return {Offset::Expr::Divide{{std::move(lhs), std::move(rhs)}}};
}

Offset::Expr Offset::Remainder(Offset::Expr lhs, Offset::Expr rhs) {
  return {Offset::Expr::Remainder{{std::move(lhs), std::move(rhs)}}};
}

Offset::Expr Offset::Minimum(Offset::Expr lhs, Offset::Expr rhs) {
  return {Offset::Expr::Minimum{{std::move(lhs), std::move(rhs)}}};
}

Offset::Expr Offset::Maximum(Offset::Expr lhs, Offset::Expr rhs) {
  return {Offset::Expr::Maximum{{std::move(lhs), std::move(rhs)}}};
}

Offset::Expr Offset::Compare(ComparisonDirection direction, Offset::Expr lhs,
                             Offset::Expr rhs) {
  return {Offset::Expr::Compare{direction, {std::move(lhs), std::move(rhs)}}};
}

Offset::Expr Offset::Select(Offset::Expr pred, Offset::Expr on_true,
                            Offset::Expr on_false) {
  return {Offset::Expr::Select{
      {std::move(pred), std::move(on_true), std::move(on_false)}}};
}

static bool IsBitcastOrReshape(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kBitcast ||
         instr->opcode() == HloOpcode::kReshape;
}

static const HloInstruction* WalkThroughBitcastsAndReshapes(
    const HloInstruction* instr) {
  while (IsBitcastOrReshape(instr)) {
    instr = instr->operand(0);
  }
  return instr;
}

static bool IsNoOpInstruction(const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kParameter:
    case HloOpcode::kConstant:
    case HloOpcode::kBitcast:
    case HloOpcode::kReshape:
    case HloOpcode::kTuple:
    case HloOpcode::kGetTupleElement:
      return true;
    default:
      return false;
  }
}

static bool IsSlicingInstruction(const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kSlice:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
      return true;
    default:
      return false;
  }
}

static const HloInstruction* WalkThroughBitcasts(const HloInstruction* instr) {
  return WalkThroughBitcastsAndReshapes(instr);
}

static absl::StatusOr<int64_t> GetScalarIntegerLiteral(
    const HloConstantInstruction* constant) {
  if (!ShapeUtil::IsEffectiveScalar(constant->shape())) {
    return Internal(
        "DynamicSliceFusion: expected scalar offset constant, got %s",
        constant->ToString());
  }

  switch (constant->shape().element_type()) {
    case PRED:
      return constant->literal().GetFirstElement<bool>() ? 1 : 0;
    default:
      if (std::optional<int64_t> value = constant->literal().GetFirstInteger();
          value.has_value()) {
        return *value;
      }
      return Internal(
          "DynamicSliceFusion: expected integer or pred offset constant, "
          "got %s",
          constant->ToString());
  }
}

static absl::StatusOr<Offset::Expr> BuildOffsetExpr(
    const HloInstruction* instr) {
  if (!Offset::IsExpr(instr)) {
    return Internal(
        "DynamicSliceFusion: expected DS/DUS offset to be a scalar "
        "expression of fusion parameters and constants, got %s",
        instr->ToString());
  }

  if (auto* parameter = DynCast<HloParameterInstruction>(instr)) {
    return Offset::Parameter(parameter->parameter_number());
  }

  if (auto* constant = DynCast<HloConstantInstruction>(instr)) {
    ABSL_ASSIGN_OR_RETURN(int64_t value, GetScalarIntegerLiteral(constant));
    return Offset::Constant(value);
  }

  switch (instr->opcode()) {
    case HloOpcode::kAdd: {
      ABSL_ASSIGN_OR_RETURN(auto lhs, BuildOffsetExpr(instr->operand(0)));
      ABSL_ASSIGN_OR_RETURN(auto rhs, BuildOffsetExpr(instr->operand(1)));
      return Offset::Add(std::move(lhs), std::move(rhs));
    }
    case HloOpcode::kSubtract: {
      ABSL_ASSIGN_OR_RETURN(auto lhs, BuildOffsetExpr(instr->operand(0)));
      ABSL_ASSIGN_OR_RETURN(auto rhs, BuildOffsetExpr(instr->operand(1)));
      return Offset::Subtract(std::move(lhs), std::move(rhs));
    }
    case HloOpcode::kMultiply: {
      ABSL_ASSIGN_OR_RETURN(auto lhs, BuildOffsetExpr(instr->operand(0)));
      ABSL_ASSIGN_OR_RETURN(auto rhs, BuildOffsetExpr(instr->operand(1)));
      return Offset::Multiply(std::move(lhs), std::move(rhs));
    }
    case HloOpcode::kDivide: {
      ABSL_ASSIGN_OR_RETURN(auto lhs, BuildOffsetExpr(instr->operand(0)));
      ABSL_ASSIGN_OR_RETURN(auto rhs, BuildOffsetExpr(instr->operand(1)));
      return Offset::Divide(std::move(lhs), std::move(rhs));
    }
    case HloOpcode::kRemainder: {
      ABSL_ASSIGN_OR_RETURN(auto lhs, BuildOffsetExpr(instr->operand(0)));
      ABSL_ASSIGN_OR_RETURN(auto rhs, BuildOffsetExpr(instr->operand(1)));
      return Offset::Remainder(std::move(lhs), std::move(rhs));
    }
    case HloOpcode::kMinimum: {
      ABSL_ASSIGN_OR_RETURN(auto lhs, BuildOffsetExpr(instr->operand(0)));
      ABSL_ASSIGN_OR_RETURN(auto rhs, BuildOffsetExpr(instr->operand(1)));
      return Offset::Minimum(std::move(lhs), std::move(rhs));
    }
    case HloOpcode::kMaximum: {
      ABSL_ASSIGN_OR_RETURN(auto lhs, BuildOffsetExpr(instr->operand(0)));
      ABSL_ASSIGN_OR_RETURN(auto rhs, BuildOffsetExpr(instr->operand(1)));
      return Offset::Maximum(std::move(lhs), std::move(rhs));
    }
    case HloOpcode::kClamp: {
      // clamp(lo, x, hi) = min(max(x, lo), hi)
      ABSL_ASSIGN_OR_RETURN(auto lo, BuildOffsetExpr(instr->operand(0)));
      ABSL_ASSIGN_OR_RETURN(auto x, BuildOffsetExpr(instr->operand(1)));
      ABSL_ASSIGN_OR_RETURN(auto hi, BuildOffsetExpr(instr->operand(2)));
      return Offset::Minimum(Offset::Maximum(std::move(x), std::move(lo)),
                             std::move(hi));
    }
    case HloOpcode::kBitcast:
    case HloOpcode::kReshape:
      // Bitcasts and reshapes between single-element arrays are no-ops.
      return BuildOffsetExpr(instr->operand(0));
    case HloOpcode::kConvert:
      // Offset expressions are evaluated in int64_t, so integer converts are
      // no-ops (assuming offsets do not overflow the narrower type).
      return BuildOffsetExpr(instr->operand(0));
    case HloOpcode::kCompare: {
      auto* compare = Cast<HloCompareInstruction>(instr);
      ABSL_ASSIGN_OR_RETURN(auto lhs, BuildOffsetExpr(compare->operand(0)));
      ABSL_ASSIGN_OR_RETURN(auto rhs, BuildOffsetExpr(compare->operand(1)));
      return Offset::Compare(compare->direction(), std::move(lhs),
                             std::move(rhs));
    }
    case HloOpcode::kSelect: {
      ABSL_ASSIGN_OR_RETURN(auto pred, BuildOffsetExpr(instr->operand(0)));
      ABSL_ASSIGN_OR_RETURN(auto on_true, BuildOffsetExpr(instr->operand(1)));
      ABSL_ASSIGN_OR_RETURN(auto on_false, BuildOffsetExpr(instr->operand(2)));
      return Offset::Select(std::move(pred), std::move(on_true),
                            std::move(on_false));
    }
    default:
      return Internal("Unsupported offset expression opcode: %s",
                      instr->ToString());
  }
}

static std::optional<DynamicSliceConfig> ExtractDynamicSliceConfig(
    const HloInstruction* instr) {
  auto config = instr->backend_config<GpuBackendConfig>();
  if (!config.ok() || !config->has_dynamic_slice_config()) {
    return std::nullopt;
  }
  return config->dynamic_slice_config();
}

static absl::Status VerifyArgCount(const Offset::Expr& expr,
                                   absl::Span<const Offset::Expr> args,
                                   size_t expected) {
  if (args.size() != expected) {
    return InvalidArgument(
        "Expected offset expression %s to have %d args, got %d",
        absl::StrCat(expr), expected, args.size());
  }
  return absl::OkStatus();
}

static absl::StatusOr<int64_t> GetParameterValue(
    int64_t parameter_number,
    absl::Span<const std::pair<int64_t, int64_t>> parameters) {
  for (const auto& [candidate, value] : parameters) {
    if (candidate == parameter_number) {
      return value;
    }
  }
  return InvalidArgument("Missing value for offset parameter %d",
                         parameter_number);
}

static absl::Span<const Offset::Expr> GetArgs(const Offset::Expr& expr) {
  return std::visit(
      [](const auto& e) -> absl::Span<const Offset::Expr> {
        using T = std::decay_t<decltype(e)>;
        if constexpr (std::is_same_v<T, Offset::Expr::Constant> ||
                      std::is_same_v<T, Offset::Expr::Parameter>) {
          return {};
        } else {
          return e.args;
        }
      },
      expr.value);
}

// Evaluates HLO integer division and remainder. Division by zero and signed
// overflow follow HLO semantics: `x / 0 = -1`, `x % 0 = x`,
// `INT_MIN / -1 = INT_MIN` and `INT_MIN % -1 = 0`.
static int64_t EvaluateDivide(int64_t lhs, int64_t rhs) {
  if (rhs == 0) {
    return -1;
  }
  if (lhs == std::numeric_limits<int64_t>::min() && rhs == -1) {
    return lhs;
  }
  return lhs / rhs;
}

static int64_t EvaluateRemainder(int64_t lhs, int64_t rhs) {
  if (rhs == 0) {
    return lhs;
  }
  if (lhs == std::numeric_limits<int64_t>::min() && rhs == -1) {
    return 0;
  }
  return lhs % rhs;
}

absl::StatusOr<int64_t> DynamicSliceFusion::Evaluate(
    const Offset::Expr& expr,
    absl::Span<const std::pair<int64_t, int64_t>> parameters) {
  auto evaluate_compare = [](ComparisonDirection direction, int64_t lhs,
                             int64_t rhs) -> absl::StatusOr<int64_t> {
    switch (direction) {
      case ComparisonDirection::kEq:
        return lhs == rhs ? 1 : 0;
      case ComparisonDirection::kNe:
        return lhs != rhs ? 1 : 0;
      case ComparisonDirection::kGe:
        return lhs >= rhs ? 1 : 0;
      case ComparisonDirection::kGt:
        return lhs > rhs ? 1 : 0;
      case ComparisonDirection::kLe:
        return lhs <= rhs ? 1 : 0;
      case ComparisonDirection::kLt:
        return lhs < rhs ? 1 : 0;
    }
    return Internal("Unsupported comparison direction in offset expression");
  };

  return std::visit(
      [&](const auto& e) -> absl::StatusOr<int64_t> {
        using T = std::decay_t<decltype(e)>;
        if constexpr (std::is_same_v<T, Offset::Expr::Constant>) {
          return e.value;
        } else if constexpr (std::is_same_v<T, Offset::Expr::Parameter>) {
          return GetParameterValue(e.parameter_number, parameters);
        } else if constexpr (std::is_same_v<T, Offset::Expr::Add>) {
          ABSL_RETURN_IF_ERROR(VerifyArgCount(expr, e.args, 2));
          ABSL_ASSIGN_OR_RETURN(int64_t lhs, Evaluate(e.args[0], parameters));
          ABSL_ASSIGN_OR_RETURN(int64_t rhs, Evaluate(e.args[1], parameters));
          return lhs + rhs;
        } else if constexpr (std::is_same_v<T, Offset::Expr::Subtract>) {
          ABSL_RETURN_IF_ERROR(VerifyArgCount(expr, e.args, 2));
          ABSL_ASSIGN_OR_RETURN(int64_t lhs, Evaluate(e.args[0], parameters));
          ABSL_ASSIGN_OR_RETURN(int64_t rhs, Evaluate(e.args[1], parameters));
          return lhs - rhs;
        } else if constexpr (std::is_same_v<T, Offset::Expr::Multiply>) {
          ABSL_RETURN_IF_ERROR(VerifyArgCount(expr, e.args, 2));
          ABSL_ASSIGN_OR_RETURN(int64_t lhs, Evaluate(e.args[0], parameters));
          ABSL_ASSIGN_OR_RETURN(int64_t rhs, Evaluate(e.args[1], parameters));
          return lhs * rhs;
        } else if constexpr (std::is_same_v<T, Offset::Expr::Divide>) {
          ABSL_RETURN_IF_ERROR(VerifyArgCount(expr, e.args, 2));
          ABSL_ASSIGN_OR_RETURN(int64_t lhs, Evaluate(e.args[0], parameters));
          ABSL_ASSIGN_OR_RETURN(int64_t rhs, Evaluate(e.args[1], parameters));
          return EvaluateDivide(lhs, rhs);
        } else if constexpr (std::is_same_v<T, Offset::Expr::Remainder>) {
          ABSL_RETURN_IF_ERROR(VerifyArgCount(expr, e.args, 2));
          ABSL_ASSIGN_OR_RETURN(int64_t lhs, Evaluate(e.args[0], parameters));
          ABSL_ASSIGN_OR_RETURN(int64_t rhs, Evaluate(e.args[1], parameters));
          return EvaluateRemainder(lhs, rhs);
        } else if constexpr (std::is_same_v<T, Offset::Expr::Minimum>) {
          ABSL_RETURN_IF_ERROR(VerifyArgCount(expr, e.args, 2));
          ABSL_ASSIGN_OR_RETURN(int64_t lhs, Evaluate(e.args[0], parameters));
          ABSL_ASSIGN_OR_RETURN(int64_t rhs, Evaluate(e.args[1], parameters));
          return std::min(lhs, rhs);
        } else if constexpr (std::is_same_v<T, Offset::Expr::Maximum>) {
          ABSL_RETURN_IF_ERROR(VerifyArgCount(expr, e.args, 2));
          ABSL_ASSIGN_OR_RETURN(int64_t lhs, Evaluate(e.args[0], parameters));
          ABSL_ASSIGN_OR_RETURN(int64_t rhs, Evaluate(e.args[1], parameters));
          return std::max(lhs, rhs);
        } else if constexpr (std::is_same_v<T, Offset::Expr::Compare>) {
          ABSL_RETURN_IF_ERROR(VerifyArgCount(expr, e.args, 2));
          ABSL_ASSIGN_OR_RETURN(int64_t lhs, Evaluate(e.args[0], parameters));
          ABSL_ASSIGN_OR_RETURN(int64_t rhs, Evaluate(e.args[1], parameters));
          return evaluate_compare(e.direction, lhs, rhs);
        } else if constexpr (std::is_same_v<T, Offset::Expr::Select>) {
          ABSL_RETURN_IF_ERROR(VerifyArgCount(expr, e.args, 3));
          ABSL_ASSIGN_OR_RETURN(int64_t pred, Evaluate(e.args[0], parameters));
          return Evaluate(e.args[pred != 0 ? 1 : 2], parameters);
        } else {
          return Internal("Unsupported offset expression kind");
        }
      },
      expr.value);
}

std::vector<int64_t> DynamicSliceFusion::CollectOffsetParameters(
    const Offset::Expr& expr) {
  std::vector<int64_t> parameter_numbers;
  auto collect = [&](auto& self, const Offset::Expr& expr) -> void {
    if (auto* e = std::get_if<Offset::Expr::Parameter>(&expr.value);
        e && !absl::c_contains(parameter_numbers, e->parameter_number)) {
      parameter_numbers.push_back(e->parameter_number);
    }
    for (const Offset::Expr& arg : GetArgs(expr)) {
      self(self, arg);
    }
  };
  collect(collect, expr);
  return parameter_numbers;
}

static absl::flat_hash_set<const HloInstruction*> CollectOffsetInstructions(
    const HloComputation* body) {
  absl::flat_hash_set<const HloInstruction*> offsets;

  auto collect = [&](auto& self, const HloInstruction* instr) -> void {
    instr = WalkThroughBitcasts(instr);
    if (!offsets.insert(instr).second) {
      return;
    }
    if (HloPredicateIsOp<HloOpcode::kParameter, HloOpcode::kConstant>(instr)) {
      return;
    }
    for (const HloInstruction* operand : instr->operands()) {
      self(self, operand);
    }
  };

  for (const HloInstruction* hlo : body->instructions()) {
    if (auto* ds = DynCast<HloDynamicSliceInstruction>(hlo)) {
      for (int64_t i = 1; i < ds->operand_count(); ++i) {
        collect(collect, ds->operand(i));
      }
    } else if (auto* dus = DynCast<HloDynamicUpdateSliceInstruction>(hlo)) {
      for (int64_t i = 2; i < dus->operand_count(); ++i) {
        collect(collect, dus->operand(i));
      }
    }
  }

  return offsets;
}

const HloInstruction* DynamicSliceFusion::FindHero(const HloComputation* body) {
  absl::flat_hash_set<const HloInstruction*> offsets =
      CollectOffsetInstructions(body);
  for (const HloInstruction* hlo : body->instructions()) {
    if (!offsets.contains(hlo) && !IsNoOpInstruction(hlo) &&
        !IsSlicingInstruction(hlo)) {
      return hlo;
    }
  }
  return nullptr;
}

static std::optional<DynamicSliceConfig> ComputeStaticSliceConfig(
    const HloSliceInstruction* slice) {
  auto byte_strides = ShapeUtil::ByteStrides(slice->operand(0)->shape());
  if (!byte_strides.has_value()) {
    return std::nullopt;
  }
  int64_t byte_offset = 0;
  for (int64_t dim = 0; dim < slice->shape().dimensions().size(); ++dim) {
    byte_offset += slice->slice_starts(dim) * (*byte_strides)[dim];
  }
  DynamicSliceConfig config;
  config.mutable_linear()->set_byte_offset(byte_offset);
  config.mutable_linear()->set_byte_stride(0);
  return config;
}

// Resolves per-dimension offset expressions of a DS/DUS. Returns one
// expression per dimension, or std::nullopt if offsets are not representable as
// Offset::Expr and offset resolution is optional.
static absl::StatusOr<std::optional<std::vector<Offset>>> ResolveOffsets(
    const HloDynamicIndexInstruction* instr,
    OffsetResolution offset_resolution) {
  absl::Span<HloInstruction* const> index_operands = instr->index_operands();
  std::vector<Offset> offsets;
  offsets.reserve(index_operands.size());
  for (int64_t dim = 0; dim < index_operands.size(); ++dim) {
    absl::StatusOr<Offset::Expr> expr = BuildOffsetExpr(index_operands[dim]);
    if (!expr.ok()) {
      if (offset_resolution == OffsetResolution::kOptional) {
        VLOG(3) << "Failed to resolve offset expression of " << instr->name()
                << ": " << expr.status();
        return std::nullopt;
      }
      return expr.status();
    }
    offsets.push_back(Offset{dim, *std::move(expr)});
  }
  return offsets;
}

// Resolves one result chain: walks from a hero output through bitcasts to a
// DUS, extracts the config, and finds the target parameter.
static absl::StatusOr<std::optional<DynamicSliceFusion::Result>>
ResolveOneResultChain(const HloInstruction* start, const Shape& hero_shape,
                      int64_t result_number,
                      OffsetResolution offset_resolution) {
  const HloInstruction* walk = start;
  while (walk->opcode() == HloOpcode::kBitcast) {
    if (walk->user_count() != 1) {
      break;
    }
    walk = walk->users().front();
  }

  auto* dus = DynCast<HloDynamicUpdateSliceInstruction>(walk);
  if (dus == nullptr) {
    return std::nullopt;
  }

  std::optional<DynamicSliceConfig> config = ExtractDynamicSliceConfig(dus);

  const HloInstruction* target = WalkThroughBitcasts(dus->operand(0));
  auto* target_param = DynCast<HloParameterInstruction>(target);
  if (target_param == nullptr) {
    return Internal(
        "DynamicSliceFusion: DUS target must be a fusion parameter, got %s",
        target->ToString());
  }

  ABSL_ASSIGN_OR_RETURN(auto offsets, ResolveOffsets(dus, offset_resolution));

  return DynamicSliceFusion::Result{
      std::optional<int64_t>(target_param->parameter_number()),
      result_number,
      dus->operand(0)->shape(),
      dus->operand(1)->shape(),
      config,
      std::move(offsets),
  };
}

// Walks a GTE chain from `instr` following the given ShapeIndex path. For
// example, ShapeIndex{0,1} means: find the GTE with index 0 among instr's
// users, then find the GTE with index 1 among that GTE's users. Returns
// nullptr if the chain does not exist.
static const HloInstruction* WalkGteChain(const HloInstruction* instr,
                                          const ShapeIndex& index) {
  const HloInstruction* current = instr;
  for (int64_t i = 0; i < index.size(); ++i) {
    const HloInstruction* found = nullptr;
    for (const HloInstruction* user : current->users()) {
      auto* gte = DynCast<HloGetTupleElementInstruction>(user);
      if (gte != nullptr && gte->tuple_index() == index[i]) {
        found = gte;
        break;
      }
    }
    if (found == nullptr) {
      return nullptr;
    }
    current = found;
  }
  return current;
}

absl::StatusOr<DynamicSliceFusion::Parameter>
DynamicSliceFusion::ResolveParameter(const HloInstruction* operand,
                                     OffsetResolution offset_resolution) {
  const HloInstruction* walk = WalkThroughBitcasts(operand);

  std::optional<DynamicSliceConfig> config;
  std::optional<std::vector<Offset>> offsets;
  const HloInstruction* source = walk;
  Shape slice_shape = operand->shape();

  if (auto* ds = DynCast<HloDynamicSliceInstruction>(walk)) {
    config = ExtractDynamicSliceConfig(ds);
    ABSL_ASSIGN_OR_RETURN(offsets, ResolveOffsets(ds, offset_resolution));
    slice_shape = ds->shape();
    source = ds->operand(0);
  } else if (auto* slice = DynCast<HloSliceInstruction>(walk)) {
    config = ComputeStaticSliceConfig(slice);
    slice_shape = slice->shape();
    source = slice->operand(0);
  }

  Shape source_shape = source->shape();
  source = WalkThroughBitcasts(source);
  auto* parameter = DynCast<HloParameterInstruction>(source);
  if (parameter == nullptr) {
    return Internal(
        "DynamicSliceFusion: expected fusion parameter backing hero operand, "
        "got %s",
        source->ToString());
  }

  return DynamicSliceFusion::Parameter{
      parameter->parameter_number(),
      std::move(source_shape),
      slice_shape,
      config,
      std::move(offsets),
  };
}

absl::StatusOr<std::vector<DynamicSliceFusion::Parameter>>
DynamicSliceFusion::ResolveParameters(const HloInstruction* hero) {
  std::vector<DynamicSliceFusion::Parameter> result;
  result.reserve(hero->operand_count());

  for (const HloInstruction* operand : hero->operands()) {
    ABSL_ASSIGN_OR_RETURN(DynamicSliceFusion::Parameter parameter,
                     ResolveParameter(operand));
    result.push_back(std::move(parameter));
  }

  return result;
}

absl::StatusOr<std::vector<DynamicSliceFusion::Result>>
DynamicSliceFusion::ResolveResults(const HloInstruction* hero,
                                   OffsetResolution offset_resolution) {
  if (hero->shape().IsTuple()) {
    auto leaves = ShapeUtil::GetLeafShapes(hero->shape());
    int64_t n = leaves.size();
    std::vector<DynamicSliceFusion::Result> results(n);

    for (int64_t i = 0; i < n; ++i) {
      const Shape& leaf = leaves[i].shape;
      results[i] = Result{std::nullopt, i, leaf, leaf};
    }

    for (int64_t i = 0; i < n; ++i) {
      const ShapeIndex& index = leaves[i].index;
      const HloInstruction* leaf_gte = WalkGteChain(hero, index);
      if (leaf_gte == nullptr) {
        continue;
      }

      for (const HloInstruction* user : leaf_gte->users()) {
        ABSL_ASSIGN_OR_RETURN(auto rs, ResolveOneResultChain(user, leaves[i].shape,
                                                        i, offset_resolution));
        if (rs.has_value()) {
          results[i] = *std::move(rs);
        }
      }
    }
    return results;
  }

  // Non-tuple hero: single result.
  for (const HloInstruction* user : hero->users()) {
    ABSL_ASSIGN_OR_RETURN(auto rs, ResolveOneResultChain(user, hero->shape(), 0,
                                                    offset_resolution));
    if (rs.has_value()) {
      return std::vector{*std::move(rs)};
    }
  }

  return std::vector{Result{std::nullopt, 0, hero->shape(), hero->shape()}};
}

}  // namespace xla::gpu
