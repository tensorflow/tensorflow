// Copyright 2024 The TensorFlow Authors
//
// Refactored OuterDimensionPropagationPass (v8):
// - Encapsulates propagation state & logic into OutDimPropagator class.
// - Preserves previous multiplier rules and tuple/call/while handling.
// - Adds callsite index and small caches to reduce repeated module scans.
// - Keeps conservative semantics: on conflict, mark conflicted and skip
// metadata.
//
// Patch: algorithm/worklist optimizations:
// - avoid duplicate enqueueing with in_queue_ set
// - detect whether Merge would actually change mapping before performing merge
//   and only serialize/write metadata when mapping truly changed
// - only enqueue when a merge actually changed the mapping
//
// These changes reduce unnecessary work and serialization overhead.

#include "xla/service/outer_dimension_propagation.h"

#include <deque>
#include <queue>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/layout_util.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

// Path and mapping types
using Path = std::string;  // e.g., "0" or "1.2"
using PathMulMap = absl::flat_hash_map<Path, int64_t>;

// Utility functions (previous logic, kept local)
static std::string MakePath(const Path& prefix, int64_t idx) {
  if (prefix.empty()) return absl::StrCat(idx);
  return absl::StrCat(prefix, ".", idx);
}

static bool MatchAndStripPrefix(const Path& key, const Path& candidate,
                                std::string* remainder) {
  if (candidate == key) {
    *remainder = "";
    return true;
  }
  if (candidate.size() > key.size() + 1 &&
      candidate.substr(0, key.size()) == key && candidate[key.size()] == '.') {
    *remainder = candidate.substr(key.size() + 1);
    return true;
  }
  return false;
}

static bool MergePathMulMap(PathMulMap& dest, const PathMulMap& src) {
  for (const auto& kv : src) {
    auto it = dest.find(kv.first);
    if (it == dest.end()) {
      dest.emplace(kv.first, kv.second);
    } else {
      if (it->second != kv.second) {
        LOG(ERROR) << "OuterDimensionPropagation conflict at path " << kv.first
                   << ": existing=" << it->second << " new=" << kv.second;
        return false;
      }
    }
  }
  return true;
}

static bool InsertPath(PathMulMap& dest, const Path& key, int64_t mul) {
  auto it = dest.find(key);
  if (it == dest.end()) {
    dest.emplace(key, mul);
    return true;
  }
  if (it->second != mul) {
    LOG(ERROR) << "OuterDimensionPropagation conflict at path " << key
               << ": existing=" << it->second << " new=" << mul;
    return false;
  }
  return true;
}

// Serialize mapping according to shape: nested arrays for tuples, scalar or
// "null" for non-tuple.
static std::pair<bool, std::string> SerializeMappingForShape(
    const Shape& shape, const PathMulMap& map, const Path& prefix = "") {
  if (shape.IsTuple()) {
    int64_t arity = shape.tuple_shapes_size();
    std::string out;
    absl::StrAppend(&out, "[");
    for (int64_t i = 0; i < arity; ++i) {
      if (i) absl::StrAppend(&out, ",");
      std::string child_prefix = MakePath(prefix, i);
      auto it_exact = map.find(child_prefix);
      bool has_exact = (it_exact != map.end());
      bool has_children = false;
      for (const auto& kv : map) {
        const Path& k = kv.first;
        if (k.size() > child_prefix.size() + 1 &&
            k.substr(0, child_prefix.size()) == child_prefix &&
            k[child_prefix.size()] == '.') {
          has_children = true;
          break;
        }
      }
      if (has_exact && has_children) {
        LOG(ERROR) << "Outer-dim propagation structural conflict: both leaf "
                      "and children for path "
                   << child_prefix;
        return {false, ""};
      }
      if (has_exact) {
        absl::StrAppend(&out, absl::StrCat(it_exact->second));
      } else if (has_children) {
        const Shape& child_shape = shape.tuple_shapes(i);
        auto sub = SerializeMappingForShape(child_shape, map, child_prefix);
        if (!sub.first) return {false, ""};
        absl::StrAppend(&out, sub.second);
      } else {
        absl::StrAppend(&out, "null");
      }
    }
    absl::StrAppend(&out, "]");
    return {true, out};
  } else {
    Path key = prefix.empty() ? std::string("0") : prefix;
    auto it = map.find(key);
    if (it != map.end()) {
      return {true, absl::StrCat(it->second)};
    }
    return {true, "null"};
  }
}

// Simplified recursive parser for the serialized mapping format produced by
// SerializeMappingForShape. 
static bool ParseSerializedMappingForShapeRec(const Shape& shape,
                                             absl::string_view s, size_t& pos,
                                             const Path& prefix, PathMulMap& out) {
  auto skip_ws = [&](void) {
    while (pos < s.size() && isspace(static_cast<unsigned char>(s[pos]))) ++pos;
  };
  skip_ws();
  if (!shape.IsTuple()) {
    // parse a number or 'null'
    if (pos >= s.size()) return true;  // empty -> null
    if (s.substr(pos, 4) == "null") {
      pos += 4;
      return true;
    }
    // parse optional sign and digits
    size_t start = pos;
    if (s[pos] == '+' || s[pos] == '-') ++pos;
    while (pos < s.size() && isdigit(static_cast<unsigned char>(s[pos]))) ++pos;
    if (start == pos) return false;  // no number
    int64_t v = 0;
    if (!absl::SimpleAtoi<int64_t>(std::string(s.substr(start, pos - start)), &v)) return false;
    Path key = prefix.empty() ? std::string("0") : prefix;
    out.emplace(key, v);
    skip_ws();
    return true;
  }

  // shape is tuple: expect '['
  skip_ws();
  if (pos >= s.size() || s[pos] != '[') return false;
  ++pos;
  skip_ws();
  int64_t arity = shape.tuple_shapes_size();
  for (int64_t i = 0; i < arity; ++i) {
    const Shape& child_shape = shape.tuple_shapes(i);
    // parse child value (could be nested array, number, or null)
    if (child_shape.IsTuple()) {
      if (!ParseSerializedMappingForShapeRec(child_shape, s, pos, MakePath(prefix, i), out)) return false;
    } else {
      // Non-tuple child: parse number or 'null'
      skip_ws();
      if (pos < s.size() && s.substr(pos, 4) == "null") {
        pos += 4;
      } else {
        size_t start = pos;
        if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) ++pos;
        while (pos < s.size() && isdigit(static_cast<unsigned char>(s[pos]))) ++pos;
        if (start == pos) return false;
        int64_t v = 0;
        if (!absl::SimpleAtoi<int64_t>(std::string(s.substr(start, pos - start)), &v)) return false;
        out.emplace(MakePath(prefix, i), v);
      }
    }
    skip_ws();
    // after each child expect ',' or ']' (if last)
    if (i < arity - 1) {
      if (pos >= s.size() || s[pos] != ',') return false;
      ++pos;
      skip_ws();
    }
  }
  skip_ws();
  if (pos >= s.size() || s[pos] != ']') return false;
  ++pos;
  skip_ws();
  return true;
}

static std::pair<bool, PathMulMap> ParseSerializedMappingForShape(
    const Shape& shape, absl::string_view s, const Path& prefix = "") {
  PathMulMap out;
  size_t pos = 0;
  if (!ParseSerializedMappingForShapeRec(shape, s, pos, prefix, out)) {
    return {false, out};
  }
  // allow trailing spaces but nothing else
  while (pos < s.size() && isspace(static_cast<unsigned char>(s[pos]))) ++pos;
  if (pos != s.size()) return {false, out};
  return {true, out};
}

static void UpdateOuterMulInMetadata(HloInstruction* instr,
                                     const std::string& serialized) {
  OpMetadata md = instr->metadata();
  std::string op_name = md.op_name();
  static const char kMarker[] = "|tf_outer_marker=";
  size_t pos = op_name.rfind(kMarker);
  if (pos != std::string::npos) op_name = op_name.substr(0, pos);
  absl::StrAppend(&op_name, kMarker, serialized);
  md.set_op_name(op_name);
  instr->set_metadata(md);
}

static int64_t GetLeadingDimOrMinusOne(const Shape& shape) {
  if (shape.dimensions_size() == 0) return -1;
  int64_t d0 = shape.dimensions(0);
  if (d0 <= 0) return -1;
  return d0;
}

static absl::optional<int64_t> ComputeMultiplierForUserScalar(HloInstruction* operand,
                                                 HloInstruction* user, int64_t operand_mul) {
  if (!operand || !user) return absl::nullopt;

  const Shape& in_shape = operand->shape();
  const Shape& out_shape = user->shape();
  int64_t flag_unsupport = -2;
  int64_t flag_error = -1;

  // 1. Basic Shape Checks
  // todo: process tuple shape
  if (in_shape.IsTuple() || out_shape.IsTuple()) {
    return flag_unsupport;
  }
  if (in_shape.dimensions_size() == 0 || out_shape.dimensions_size() == 0) {
    return absl::nullopt;
  }

  // 2. Pre-calculate Dimensions and Potential Multiplier
  int64_t in_lead = in_shape.dimensions(0);
  int64_t out_lead = out_shape.dimensions(0);

  // Filter dynamic dimensions represented by negative values
  if (in_lead < 0 || out_lead < 0) {
    return flag_error;
  }

  // Calculate generic arithmetic relationship
  // operand_mul_val is the potential multiplier IF the relationship holds.
  int64_t operand_mul_val = -1;
  // todo 1: 
  if ((in_lead > 0) && ((operand_mul * out_lead) % in_lead == 0)) {
    operand_mul_val = operand_mul * out_lead / in_lead;
  }

  // Helper: check if 'operand' is actually an input to 'user' at specific index
  auto is_operand_at = [&](int index) {
    return index >= 0 && index < user->operand_count() &&
           user->operand(index) == operand;
  };

  bool is_any_operand = false;
  for (int i = 0; i < user->operand_count(); ++i) {
    if (user->operand(i) == operand) {
      is_any_operand = true;
      break;
    }
  }
  if (!is_any_operand) return flag_error;

  // 3. Comprehensive Opcode Switch
  switch (user->opcode()) {
    // -------------------------------------------------------------------------
    // Group 1: Opaque / Unknown / Complex Layouts
    // -------------------------------------------------------------------------
    case HloOpcode::kCustomCall:
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kRaggedAllToAll:
    case HloOpcode::kRaggedDot:
      return flag_unsupport;
    case HloOpcode::kGather: {
      // Gather is complex; output depends on indices when index_vector_dim not zero
      if (is_operand_at(1)) {
        const auto& dnums = user->gather_dimension_numbers();
        if (dnums.index_vector_dim() != 0) {
          return operand_mul;
        }
        return flag_error;
      }
      return absl::nullopt;
    }

    // -------------------------------------------------------------------------
    // Group 2: Dot (Matrix Multiplication)
    // -------------------------------------------------------------------------
    case HloOpcode::kDot: {
      // LHS (operand 0): Leading dim is preserved (Batch). operand_mul.
      if (is_operand_at(0)) return operand_mul;
      // RHS (operand 1): Leading dim is contracted or transposed.
      return absl::nullopt;
    }

    // -------------------------------------------------------------------------
    // Group 3: Strict Identity / Pass-Through on Operand 0
    // -------------------------------------------------------------------------
    // For these ops, the output leading dimension is strictly tied to
    // operand 0. Other operands (kernels, updates, indices) do not dictate
    // the output batch size in a broadcast/mult sense.
    case HloOpcode::kConvolution:
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kScatter:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kSetDimensionSize:
      if (is_operand_at(0)) return operand_mul;
      return absl::nullopt;

    // -------------------------------------------------------------------------
    // Group 4: Concatenate
    // -------------------------------------------------------------------------
    case HloOpcode::kConcatenate: {
      int64_t concat_dim = user->concatenate_dimension();
      if (concat_dim == 0) {
        return operand_mul_val;
      } else {
        if (in_lead == out_lead) return operand_mul;
        return flag_error;
      }
    }

    // -------------------------------------------------------------------------
    // Group 5: Collectives (Preserving or Multiplicative)
    // -------------------------------------------------------------------------
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
      if (is_operand_at(0)) return operand_mul;
      return absl::nullopt;

    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart: {
      int64_t all_gather_dim = Cast<HloAllGatherInstruction>(user)->all_gather_dimension();
      if (all_gather_dim == 0) {
        return operand_mul_val;
      }
      if (in_lead == out_lead) return operand_mul;
      return flag_error;
    }

    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kCollectiveBroadcast: // Broadcasts data to other replicas
      // Usually preserves shape on local device, or effectively 1:1 mapping
      if (in_lead == out_lead) return operand_mul;
      return flag_error;

    case HloOpcode::kAllToAll:
    case HloOpcode::kReduceScatter:
      // These change shapes (split/concat). If it's a clean multiple, we return it.
      return operand_mul_val;

    // -------------------------------------------------------------------------
    // Group 6: Reshape / Transpose / Slice (Scrambling)
    // -------------------------------------------------------------------------
    case HloOpcode::kTranspose: {
       if (!user->dimensions().empty() && user->dimensions(0) == 0) {
        return operand_mul_val;
      }
      return flag_error;
    }
    case HloOpcode::kReshape:
    case HloOpcode::kDynamicReshape:
    case HloOpcode::kBitcast:
      // Dimensions are scrambled.
      return operand_mul_val;

    // -------------------------------------------------------------------------
    // Group 7: Elementwise / Unary / Binary / Shape Preserving
    // -------------------------------------------------------------------------
    // For all these, if inputs and outputs are arrays, the generic check
    // (out_lead % in_lead == 0) usually suffices.
    case HloOpcode::kAbs:
    case HloOpcode::kAdd:
    case HloOpcode::kAddDependency:
    case HloOpcode::kAnd:
    case HloOpcode::kAtan2:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCbrt:
    case HloOpcode::kCeil:
    case HloOpcode::kClamp:
    case HloOpcode::kClz:
    case HloOpcode::kCompare:
    case HloOpcode::kComplex:
    case HloOpcode::kConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kCos:
    case HloOpcode::kDivide:
    case HloOpcode::kDomain:
    case HloOpcode::kErf:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFft:
    case HloOpcode::kFloor:
    case HloOpcode::kFusion: // Black box, but often preserves if elementwise
    case HloOpcode::kImag:
    case HloOpcode::kIota:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kLogistic:
    case HloOpcode::kMap:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kOr:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kPower:
    case HloOpcode::kReal:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kRemainder:
    case HloOpcode::kReverse:
    case HloOpcode::kRng:
    case HloOpcode::kRngBitGenerator:
    case HloOpcode::kRngGetAndUpdateState:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSelect:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSlice: // Changes size, operand_mul_val catches ratio
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kSort:
    case HloOpcode::kSqrt:
    case HloOpcode::kStochasticConvert:
    case HloOpcode::kSubtract:
    case HloOpcode::kTan:
    case HloOpcode::kTanh:
    case HloOpcode::kTriangularSolve:
    case HloOpcode::kCholesky:
    case HloOpcode::kWhile: // Loop body usually preserves shape
    case HloOpcode::kXor:
      return operand_mul_val;

    case HloOpcode::kReduce: {
      if (!is_operand_at(0)) return absl::nullopt;
      for (int64_t d : user->dimensions()) {
        if (d == 0) {
          return absl::nullopt;
        }
      }
      return operand_mul_val;
    }

    // TODO:fix kSelectAndScatter
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kReduceWindow: {
      if (!is_operand_at(0)) return absl::nullopt;
      const Window& w = user->window();
      if (w.dimensions_size() == 0) return operand_mul_val;
      const WindowDimension& dim0 = w.dimensions(0);
      bool inert_on_dim0 =
          dim0.size() == 1 && dim0.stride() == 1 && dim0.padding_low() == 0 &&
          dim0.padding_high() == 0 && dim0.base_dilation() == 1 &&
          dim0.window_dilation() == 1;
      if (inert_on_dim0) {
        return operand_mul_val;
      }
      return absl::nullopt;
    }

    case HloOpcode::kBroadcast: {
      if (!user->dimensions().empty() && user->dimensions(0) == 0) {
        return operand_mul_val;
      }
      return flag_error;
    }
    case HloOpcode::kPad: {
      if (!is_operand_at(0))
        return absl::nullopt;
      const auto& pad_config = user->padding_config();
      if (pad_config.dimensions_size() > 0) {
        const auto& dim0_config = pad_config.dimensions(0);
        if (dim0_config.edge_padding_low() == 0 &&
            dim0_config.edge_padding_high() == 0 &&
            dim0_config.interior_padding() == 0) {
          return operand_mul;
        }
      }
      return flag_error;
    }
    case HloOpcode::kTuple:
    case HloOpcode::kGetTupleElement:
      return flag_unsupport;
    // -------------------------------------------------------------------------
    // Group 8: Token/Side-Effect (Mostly unreachable due to early checks)
    // -------------------------------------------------------------------------
    case HloOpcode::kParameter:
    case HloOpcode::kConstant:
    case HloOpcode::kReplicaId:
    case HloOpcode::kPartitionId:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
    case HloOpcode::kCopyStart:
    case HloOpcode::kCopyDone:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kTopK: // Returns tuple
    case HloOpcode::kAfterAll:
    case HloOpcode::kGetDimensionSize: // Scalar output
    case HloOpcode::kBatchNormTraining: // Tuple output
    case HloOpcode::kBatchNormGrad: // Tuple output
    case HloOpcode::kAllReduceDone: // Async done wrapper
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kCollectivePermuteDone:
      // Fallback for these is operand_mul_val, but effectively blocked by
      // is_tuple checks or rank checks earlier.
      return operand_mul_val;
  }

  return absl::nullopt;
}

// -----------------------------------------------------------------------------

// OutDimPropagator: encapsulates state & logic for the pass.
class OutDimPropagator {
 public:
  explicit OutDimPropagator(HloModule* m) : module_(m), changed_(false) {
    BuildCallsiteIndex();
  }

  absl::StatusOr<bool> Run() {
    SeedParameters();
    while (!work_queue_.empty()) {
      HloInstruction* cur = work_queue_.front();
      work_queue_.pop_front();
      // Mark not in queue anymore
      in_queue_.erase(cur);
      if (conflicted_.contains(cur)) continue;
      ProcessInstructionUsers(cur);
    }
    UpdateMultiplierToShape();
    return changed_;
  }

 private:
  void ApplyPathMulMapToShape(Shape* shape, const PathMulMap& map,
                                     const Path& prefix = "") {
    if (!shape) return;
    if (shape->IsTuple()) {
      // For tuple shapes, iterate children and recursively apply any mapping
      // relevant to each child. If there is an exact mapping for the child
      // prefix use it (leaf), otherwise look for nested keys.
      int64_t arity = shape->tuple_shapes_size();
      for (int64_t i = 0; i < arity; ++i) {
        std::string child_prefix = MakePath(prefix, i);
        // First check exact leaf mapping for child_prefix.
        auto it_exact = map.find(child_prefix);
        if (it_exact != map.end()) {
          // Set the multiplier for the child shape (leaf mapping).
          Shape* child = shape->mutable_tuple_shapes(i);
          child->set_outer_multiplier(it_exact->second);
        } else {
          // See if there are any keys that are children of this prefix.
          bool has_children = false;
          for (const auto& kv : map) {
            const Path& k = kv.first;
            if (k.size() > child_prefix.size() + 1 &&
                k.substr(0, child_prefix.size()) == child_prefix &&
                k[child_prefix.size()] == '.') {
              has_children = true;
              break;
            }
          }
          if (has_children) {
            // Recurse into child shape with the same map and the new prefix.
            Shape* child = shape->mutable_tuple_shapes(i);
            ApplyPathMulMapToShape(child, map, child_prefix);
          } else {
            // No mapping for this child: leave multiplier as-is (-1).
          }
        }
      }
    } else {
      // Non-tuple shape: its key is either prefix (if prefix non-empty) or "0".
      Path key = prefix.empty() ? std::string("0") : prefix;
      auto it = map.find(key);
      if (it != map.end()) {
        shape->set_outer_multiplier(it->second);
      } else {
        // No mapping for this scalar/array shape: leave multiplier unchanged.
      }
    }
  }

  void UpdateMultiplierToShape() {
    for (auto it : element_mul_map_) {
      Shape* shape = it.first->mutable_shape();
      if (!shape)
        LOG(ERROR) << "OuterDimensionPropagation ERROR: No shape for " << it.first->ToString();
      ApplyPathMulMapToShape(shape, it.second, "");
    }
  }
  // Build callee -> callsites map once to avoid scanning module repeatedly.
  void BuildCallsiteIndex() {
    for (HloComputation* comp : module_->computations()) {
      for (HloInstruction* instr : comp->instructions()) {
        if (instr->opcode() == HloOpcode::kCall) {
          HloComputation* callee = instr->to_apply();
          if (callee) callsite_index_[callee].push_back(instr);
        } else if (instr->opcode() == HloOpcode::kFusion) {
          HloComputation* fused = instr->fused_instructions_computation();
          if (fused) callsite_index_[fused].push_back(instr);
        }
      }
    }
  }

  // Seed parameters by parsing tf_outer_marker metadata if present.
  void SeedParameters() {
    static const char kMarker[] = "|tf_outer_marker=";
    HloComputation* entry = module_->entry_computation();
    if (entry == nullptr) return;
    for (HloInstruction* instr : entry->instructions()) {
      if (instr->opcode() == HloOpcode::kParameter) {
        std::string opname = instr->metadata().op_name();
        // If op_name contains the tf_outer_marker marker, try to parse a
        // serialized mapping.
        size_t pos = opname.rfind(kMarker);
        if (pos != std::string::npos) {
          // Extract content after marker to end of string.
          std::string serialized = opname.substr(pos + strlen(kMarker));
          // Try parse according to the parameter's shape.
          auto parsed =
              ParseSerializedMappingForShape(instr->shape(), serialized, "");
          if (!parsed.first) {
            // Failed to parse: log and fallback to a safe default mapping
            // ("0"->1).
            LOG(ERROR)
                << "Failed to parse tf_outer_marker serialized mapping for "
                << instr->ToString() << ", falling back to {\"0\":1}";
            element_mul_map_[instr].emplace("0", 1);
            UpdateOuterMulInMetadata(instr, "1");
            changed_ = true;
            Enqueue(instr);
            continue;
          }
          // Merge parsed mapping into global map
          for (const auto& kv : parsed.second) {
            element_mul_map_[instr].emplace(kv.first, kv.second);
          }
          Enqueue(instr);
        } else {
          // Fallback: legacy marker without explicit serialized mapping.
          std::string optype = instr->metadata().op_type();
          if (absl::StrContains(optype, "XLA_Arg_dyn")) {
            element_mul_map_[instr].emplace("0", 1);
            UpdateOuterMulInMetadata(instr, "1");
            changed_ = true;
            Enqueue(instr);
          }
        }
      }
    }
  }

  void Enqueue(HloInstruction* instr) {
    if (!in_queue_.contains(instr) && !conflicted_.contains(instr)) {
      work_queue_.push_back(instr);
      in_queue_.insert(instr);
    }
  }

  enum MergeResult { kNoChange, kMerged, kConflict };

  // Merge add_map into element_mul_map_[instr]; on success enqueue instr.
  // Optimization: test whether merge would change mapping before doing
  // insertions and before serializing. Only serialize/write when mapping
  // actually changes.
  MergeResult MergeAndEnqueue(HloInstruction* instr,
                              const PathMulMap& add_map) {
    if (conflicted_.contains(instr)) return kNoChange;
    PathMulMap& cur = element_mul_map_[instr];  // creates if absent

    // Fast-path: if add_map empty -> no change
    if (add_map.empty()) return kNoChange;

    // First pass: detect conflicts and whether any new entries / changed values
    // exist.
    bool would_change = false;
    for (const auto& kv : add_map) {
      auto it = cur.find(kv.first);
      if (it == cur.end()) {
        would_change = true;
      } else if (it->second != kv.second) {
        // conflict
        conflicted_.insert(instr);
        element_mul_map_.erase(instr);
        LOG(ERROR) << "OuterDimensionPropagation: conflict merging for "
                   << instr->ToString();
        return kConflict;
      }
    }
    if (!would_change) {
      // Nothing to do.
      return kNoChange;
    }

    // Perform actual merge (we already checked conflicts).
    for (const auto& kv : add_map) {
      auto it = cur.find(kv.first);
      if (it == cur.end()) {
        cur.emplace(kv.first, kv.second);
      }  // else already equal
    }

    // Serialize and write metadata only after mapping changed.
    auto ser = SerializeMappingForShape(instr->shape(), cur, "");
    if (!ser.first) {
      conflicted_.insert(instr);
      element_mul_map_.erase(instr);
      LOG(ERROR) << "OuterDimensionPropagation: serialization conflict for "
                 << instr->ToString();
      return kConflict;
    }

    if (instr->shape().IsTuple() || ser.second != "null") {
      // Avoid writing metadata if the serialized suffix is identical to what is
      // already present.
      std::string existing_suffix;
      OpMetadata md = instr->metadata();
      std::string op_name = md.op_name();
      static const char kMarker[] = "|tf_outer_marker=";
      size_t pos = op_name.rfind(kMarker);
      if (pos != std::string::npos)
        existing_suffix = op_name.substr(pos + strlen(kMarker));
      if (ser.second != existing_suffix) {
        UpdateOuterMulInMetadata(instr, ser.second);
        changed_ = true;
      }
    }

    // Propagate callee root -> callsites if instr is a computation root.
    if (instr->IsRoot()) {
      HloComputation* comp = instr->parent();
      auto cs_it = callsite_index_.find(comp);
      if (cs_it != callsite_index_.end()) {
        for (HloInstruction* potential_call : cs_it->second) {
          // map callee root's mapping into call instruction
          PathMulMap call_map = cur;  // copy
          // merge into call (may enqueue)
          MergeAndEnqueue(potential_call, call_map);
        }
      }
    }

    Enqueue(instr);
    return kMerged;
  }

  // Process users of cur instruction
  void ProcessInstructionUsers(HloInstruction* cur) {
    auto it_find = element_mul_map_.find(cur);
    PathMulMap cur_map;
    if (it_find != element_mul_map_.end()) cur_map = it_find->second;

    // iterate over users
    for (HloInstruction* user : cur->users()) {
      if (conflicted_.contains(user)) continue;

      switch (user->opcode()) {
        case HloOpcode::kCall: {
          HloComputation* callee = user->to_apply();
          PropagateToCallLike(user, callee, cur, cur_map);
          break;
        }
        case HloOpcode::kFusion: {
          HloComputation* fused = user->fused_instructions_computation();
          PropagateToCallLike(user, fused, cur, cur_map);
          break;
        }
        case HloOpcode::kWhile:
          PropagateToWhile(user, cur, cur_map);
          break;
        case HloOpcode::kTuple:
          HandleTupleOperand(user, cur, cur_map);
          break;
        case HloOpcode::kGetTupleElement:
          HandleGetTupleElement(user);
          break;
        default:
          HandleDefaultPropagation(user, cur, cur_map);
          break;
      }
    }
  }

  // Propagate operand cur -> callee param and if callee root already has
  // mapping, map it to call.
  void PropagateToCallLike(HloInstruction* call_instr, HloComputation* callee,
                           HloInstruction* cur, const PathMulMap& cur_map) {
    if (!callee) return;
    // find operand positions equal to cur
    for (int j = 0; j < call_instr->operand_count(); ++j) {
      if (call_instr->operand(j) != cur) continue;
      if (j < callee->num_parameters()) {
        HloInstruction* callee_param = callee->parameter_instruction(j);
        PathMulMap param_map = cur_map;  // copy
        MergeAndEnqueue(callee_param, param_map);
      }
    }
    // If callee root already has mapping, apply it to call instruction (merged
    // via MergeAndEnqueue)
    auto it_root = element_mul_map_.find(callee->root_instruction());
    if (it_root != element_mul_map_.end()) {
      PathMulMap call_map = it_root->second;
      MergeAndEnqueue(call_instr, call_map);
    }
  }
  // Handle while: fixed-point propagation within body.
  void PropagateToWhile(HloInstruction* while_instr, HloInstruction* cur,
                        const PathMulMap& cur_map) {
    HloComputation* body = while_instr->while_body();
    if (!body) return;
    // We only seed/use the while if init operand (operand 0) has mapping.
    const HloInstruction* init_state = while_instr->operand(0);
    auto it_init = element_mul_map_.find(init_state);
    if (it_init == element_mul_map_.end()) return;

    // Conservative: skip if body contains Calls (too complex for now)
    for (HloInstruction* i : body->instructions()) {
      if (i->opcode() == HloOpcode::kCall) {
        LOG(ERROR) << "Skipping while propagation because body contains Call: "
                  << while_instr->ToString();
        return;
      }
    }

    PathMulMap current = it_init->second;
    // todo: no need set kMaxIters
    const int kMaxIters = 2;
    for (int iter = 0; iter < kMaxIters; ++iter) {
      absl::flat_hash_map<const HloInstruction*, PathMulMap> seeds;
      seeds.emplace(body->parameter_instruction(0), current);
      absl::StatusOr<PathMulMap> maybe_body_map =
          PropagateWithinComputation(body, seeds);
      if (!maybe_body_map.ok()) {
        // Mark the while instruction conflicted and stop further processing of it.
        std::string reason = absl::StrCat(
            "PropagateWithinComputation failed: ", maybe_body_map.status().ToString());
        MarkConflicted(while_instr, reason);
        return;
      }
      PathMulMap body_root_map = std::move(maybe_body_map.value());

      // Compare body_root_map vs current_param_map
      if (body_root_map == current) {
        // Fixed point reached: map body_root_map back to while_instr
        MergeAndEnqueue(while_instr, body_root_map);
        return;
      }
      // If different, adopt body_root_map as next param map (conservative).
      current = std::move(body_root_map);
    }
    MarkConflicted(while_instr, "while fixed-point did not converge");
  }

  // Tuple operand -> element i mapping
  void HandleTupleOperand(HloInstruction* tuple_instr, HloInstruction* cur,
                          const PathMulMap& cur_map) {
    PathMulMap prefixed;
    for (int i = 0; i < tuple_instr->operand_count(); ++i) {
      if (tuple_instr->operand(i) != cur) continue;
      // For each occurrence, prefix all keys with the element index.
      for (const auto& kv : cur_map) {
        if (kv.first.empty() || kv.first == "0") {
          prefixed.emplace(absl::StrCat(i), kv.second);
        } else {
          prefixed.emplace(absl::StrCat(i, ".", kv.first), kv.second);
        }
      }
    }
    if (!prefixed.empty()) {
      MergeAndEnqueue(tuple_instr, prefixed);
    }
  }

  // GTE: extract child's map for the element index and merge as user's "0"
  void HandleGetTupleElement(HloInstruction* gte) {
    int64_t tuple_index = gte->tuple_index();
    HloInstruction* tuple_operand = gte->mutable_operand(0);
    auto it = element_mul_map_.find(tuple_operand);
    if (it == element_mul_map_.end()) return;
    const PathMulMap& tuple_map = it->second;
    PathMulMap user_map;
    Path key_prefix = absl::StrCat(tuple_index);
    for (const auto& kv : tuple_map) {
      std::string remainder;
      if (MatchAndStripPrefix(key_prefix, kv.first, &remainder)) {
        if (remainder.empty()) {
          if (!InsertPath(user_map, "0", kv.second)) {
            MarkConflicted(gte, "GTE conflict leaf");
            return;
          }
        } else {
          if (!InsertPath(user_map, remainder, kv.second)) {
            MarkConflicted(gte, "GTE conflict nested");
            return;
          }
        }
      }
    }
    if (!user_map.empty()) MergeAndEnqueue(gte, user_map);
  }

  // Default propagation for non-tuple user ops using scalar-path "0"
  void HandleDefaultPropagation(HloInstruction* user, HloInstruction* cur,
                                const PathMulMap& cur_map) {
    if (cur->shape().IsTuple()) {
      LOG(ERROR) << "Can't handle tuple input in HandleDefaultPropagation:" << user->ToString();
      return;
    }
    auto it0 = cur_map.find("0");
    if (it0 == cur_map.end()) return;
    int64_t operand_mul = it0->second;
    absl::optional<int64_t> opt_mul =
        ComputeMultiplierForUserScalar(cur, user, operand_mul);
    if (opt_mul.has_value()) {
      if (opt_mul.value() < 0) {
        LOG(ERROR) <<"HandleDefaultPropagation error, opt_mul can't be negative";
        return;
      }
      PathMulMap add;
      add.emplace("0", opt_mul.value());
      MergeAndEnqueue(user, add);
    }
  }

  // Local propagation inside a computation (no metadata writes). Returns root
  // map.
  absl::StatusOr<PathMulMap> PropagateWithinComputation(
      HloComputation* comp,
      const absl::flat_hash_map<const HloInstruction*, PathMulMap>& seeds) {
    // local state
    absl::flat_hash_map<const HloInstruction*, PathMulMap> local_map = seeds;
    absl::flat_hash_set<const HloInstruction*> local_conflicted;
    std::deque<const HloInstruction*> local_q;
    // seed queue
    for (const auto& kv : seeds) {
      if (kv.first->parent() == comp) local_q.push_back(kv.first);
    }

    auto LocalMerge = [&](const HloInstruction* instr,
                          const PathMulMap& add) -> bool {
      if (local_conflicted.contains(instr)) return true;
      PathMulMap& cur = local_map[instr];
      if (!MergePathMulMap(cur, add)) {
        local_conflicted.insert(instr);
        return false;
      }
      local_q.push_back(instr);
      return true;
    };

    while (!local_q.empty()) {
      const HloInstruction* cur = local_q.front();
      auto it_find = local_map.find(cur);
      PathMulMap cur_map;
      if (it_find != local_map.end())
        cur_map = it_find->second;
      local_q.pop_front();
      if (local_conflicted.contains(cur)) continue;
      for (const HloInstruction* user : cur->users()) {
        if (user->parent() != comp) continue;
        if (local_conflicted.contains(user)) continue;
        if (user->opcode() == HloOpcode::kCall) {
          return xla::Internal(
              "PropagateWithinComputation: encountered Call inside body; "
              "aborting");
        }
        if (user->opcode() == HloOpcode::kTuple) {
          PathMulMap prefixed;
          for (int i = 0; i < user->operand_count(); ++i) {
            if (user->operand(i) != cur) continue;
            // For each occurrence, prefix all keys with the element index.
            for (const auto& kv : cur_map) {
              if (kv.first.empty() || kv.first == "0") {
                prefixed.emplace(absl::StrCat(i), kv.second);
              } else {
                prefixed.emplace(absl::StrCat(i, ".", kv.first), kv.second);
              }
            }
          }
          if (!LocalMerge(user, prefixed))
            return xla::Internal("local tuple merge conflict");
          continue;
        }
        if (user->opcode() == HloOpcode::kGetTupleElement) {
          int64_t tuple_index = user->tuple_index();
          const HloInstruction* tuple_operand = user->operand(0);
          PathMulMap user_map;
          Path key_prefix = absl::StrCat(tuple_index);
          for (const auto& kv : cur_map) {
            std::string remainder;
            if (MatchAndStripPrefix(key_prefix, kv.first, &remainder)) {
              if (remainder.empty()) {
                if (!InsertPath(user_map, "0", kv.second)) {
                  return xla::Internal("local GTE conflict leaf");
                }
              } else {
                if (!InsertPath(user_map, remainder, kv.second)) {
                  return xla::Internal("local GTE conflict nested");
                }
              }
            }
          }
          if (!user_map.empty()) {
            if (!LocalMerge(user, user_map))
              return xla::Internal("local GTE merge conflict");
          }
          continue;
        }

        // default scalar propagation
        auto it0 = local_map[cur].find("0");
        if (it0 != local_map[cur].end()) {
          int64_t operand_mul = it0->second;
          absl::optional<int64_t> opt_mul = ComputeMultiplierForUserScalar(
              const_cast<HloInstruction*>(cur),
              const_cast<HloInstruction*>(user), operand_mul);
          if (opt_mul.has_value()) {
            PathMulMap add;
            add.emplace("0", opt_mul.value());
            if (!LocalMerge(user, add))
              return xla::Internal("local scalar merge conflict");
          }
        }
      }  // for users
    }    // while local_q

    // Merge local_map results back into global element_mul_map_ so that
    // instructions inside 'comp' get metadata written and global state reflects
    // the propagation done within the body computation.
    for (const auto& kv : local_map) {
      const HloInstruction* instr = kv.first;
      // Only merge back entries for instructions that belong to this computation.
      if (instr->parent() != comp) continue;
      // Convert const HloInstruction* to mutable for MergeAndEnqueue.
      HloInstruction* mutable_instr = const_cast<HloInstruction*>(instr);
      MergeResult mr = MergeAndEnqueue(mutable_instr, kv.second);
      if (mr == kConflict) {
        return xla::Internal("PropagateWithinComputation: conflict when merging local results to global state");
      }
    }

    HloInstruction* root = comp->root_instruction();
    PathMulMap res;
    auto it_root = local_map.find(root);
    if (it_root != local_map.end()) res = it_root->second;
    return res;
  }

  void MarkConflicted(HloInstruction* instr, absl::string_view reason) {
    conflicted_.insert(instr);
    element_mul_map_.erase(instr);
    LOG(ERROR) << "OuterDimensionPropagation: marking conflicted instr "
               << instr->ToString() << " reason: " << reason;
  }

  // Members
  HloModule* module_;
  absl::flat_hash_map<HloInstruction*, PathMulMap> element_mul_map_;
  absl::flat_hash_set<HloInstruction*> conflicted_;
  std::deque<HloInstruction*> work_queue_;
  absl::flat_hash_set<const HloInstruction*>
      in_queue_;  // avoid duplicates in work_queue_
  absl::flat_hash_map<const HloComputation*, std::vector<HloInstruction*>>
      callsite_index_;
  bool changed_;
};

}  // namespace


// -----------------------------------------------------------------------------
// Public pass entrypoint: wraps OutDimPropagator
// -----------------------------------------------------------------------------
absl::StatusOr<bool> OuterDimensionPropagationPass::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  OutDimPropagator p(module);
  return p.Run();
}

}  // namespace xla
