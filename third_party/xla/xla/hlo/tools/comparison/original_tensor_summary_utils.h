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

#ifndef XLA_HLO_TOOLS_COMPARISON_ORIGINAL_TENSOR_SUMMARY_UTILS_H_
#define XLA_HLO_TOOLS_COMPARISON_ORIGINAL_TENSOR_SUMMARY_UTILS_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/tools/comparison/comparison_result.pb.h"
#include "xla/hlo/tools/comparison/tensor_summary_util.h"
#include "xla/shape_util.h"
#include "xla/tools/debug_event.pb.h"

namespace xla::numerics::comparison {

// A helper class to log progress of a long running task. It logs the progress
// to stderr at a specified interval. This class follows the RAII pattern,
// logging the final progress upon destruction.
class ProgressReporter {
 public:
  // Constructs a new ProgressReporter.
  //
  // - `message_prefix`: A prefix for the progress message.
  // - `total_count`: The total number of items to process.
  // - `use_percent`: If true, display progress as a percentage; otherwise,
  //   display as processed/total.
  // - `log_interval`: The minimum interval between log messages.
  ProgressReporter(::absl::string_view message_prefix, int64_t total_count,
                   bool use_percent = false,
                   std::optional<::absl::Duration> log_interval = std::nullopt);

  ~ProgressReporter();

  // Reports progress.
  //
  // - `new_processed_count`: If non-negative, sets the processed count to
  //    this value; otherwise, increments the processed count by 1.
  // - `new_total_count`: If non-negative, updates the total count to this
  //   value.
  void Report(int64_t new_processed_count = -1, int64_t new_total_count = -1);

 private:
  std::string message_prefix_;
  int64_t total_count_;
  ::absl::Duration log_interval_;
  int64_t processed_count_;
  ::absl::Time last_log_time_;
  bool use_percent_;
};

enum class ComparisonVariant {
  kBaseline,
  kTarget,
};

inline std::string ToString(ComparisonVariant variant) {
  switch (variant) {
    case ComparisonVariant::kBaseline:
      return "baseline";
    case ComparisonVariant::kTarget:
      return "target";
  }
}

namespace tensor_transformation {
struct Reshape;
struct Broadcast;
struct Unshard;
using TensorTransformation = std::variant<Reshape, Broadcast, Unshard>;

struct Reshape {
  std::shared_ptr<const TensorTransformation> continuation;
  // The output shape dimensions of the tensor after this reshape. Note that
  // for recovery transformations, this reshape is to undo a reshape applied
  // during XLA optimization. So in that case,these dimensions are the
  // dimensions of the tensor before the optimization reshape.
  std::vector<int64_t> output_dimensions;

  Reshape() = default;
  // NOLINTNEXTLINE
  Reshape(std::vector<int64_t> output_dimensions)
      : continuation(nullptr),
        output_dimensions(std::move(output_dimensions)) {}
  Reshape(std::shared_ptr<const TensorTransformation> continuation,
          std::vector<int64_t> output_dimensions)
      : continuation(std::move(continuation)),
        output_dimensions(std::move(output_dimensions)) {}

  bool EqualsWithoutContinuation(const Reshape& other) const {
    return output_dimensions == other.output_dimensions;
  }
};

// Note: transpose can be modeled as a broadcast where the broadcast_dimensions
// are the inverse permutation of the permutation specified in the transpose.
struct Broadcast {
  std::shared_ptr<const TensorTransformation> continuation;
  // The output shape dimensions of the tensor after this broadcast.
  std::vector<int64_t> output_dimensions;
  // The dimensions that are broadcasted. This vector maps input dimensions to
  // output dimensions. For example, if the input tensor has dimensions
  // {x, y, z} and the output tensor has dimensions {a, x, b, y, c, z}, then
  // the broadcast dimensions are {1, 3, 5}.
  // Note that the broadcast dimensions also support dropping dimensions to undo
  // a broadcast operation with -1 as the output dimension number. See the
  // comment of `ApplyBroadcastToSummary` for more details.
  std::vector<int64_t> broadcast_dimensions;

  Broadcast() = default;
  Broadcast(std::vector<int64_t> output_dimensions,
            std::vector<int64_t> broadcast_dimensions)
      : continuation(nullptr),
        output_dimensions(std::move(output_dimensions)),
        broadcast_dimensions(std::move(broadcast_dimensions)) {}
  Broadcast(std::shared_ptr<const TensorTransformation> continuation,
            std::vector<int64_t> output_dimensions,
            std::vector<int64_t> broadcast_dimensions)
      : continuation(std::move(continuation)),
        output_dimensions(std::move(output_dimensions)),
        broadcast_dimensions(std::move(broadcast_dimensions)) {}

  bool EqualsWithoutContinuation(const Broadcast& other) const {
    return output_dimensions == other.output_dimensions &&
           broadcast_dimensions == other.broadcast_dimensions;
  }
};

struct Unshard {
  std::shared_ptr<const TensorTransformation> continuation;
  // The dimensions of the original tensor before sharding.
  std::vector<int64_t> original_dimensions;
  HloSharding sharding;

  Unshard() = default;
  Unshard(std::vector<int64_t> original_dimensions, HloSharding sharding)
      : continuation(nullptr),
        original_dimensions(std::move(original_dimensions)),
        sharding(std::move(sharding)) {}
  Unshard(std::shared_ptr<const TensorTransformation> continuation,
          std::vector<int64_t> original_dimensions, HloSharding sharding)
      : continuation(std::move(continuation)),
        original_dimensions(std::move(original_dimensions)),
        sharding(std::move(sharding)) {}

  bool EqualsWithoutContinuation(const Unshard& other) const {
    return original_dimensions == other.original_dimensions &&
           sharding == other.sharding;
  }
};

inline bool EqualsWithoutContinuation(const TensorTransformation& a,
                                      const TensorTransformation& b) {
  if (a.index() != b.index()) {
    return false;
  }
  if (a.valueless_by_exception()) {
    return true;
  }
  if (std::holds_alternative<Reshape>(a)) {
    return std::get<Reshape>(a).EqualsWithoutContinuation(std::get<Reshape>(b));
  }
  if (std::holds_alternative<Broadcast>(a)) {
    return std::get<Broadcast>(a).EqualsWithoutContinuation(
        std::get<Broadcast>(b));
  }
  if (std::holds_alternative<Unshard>(a)) {
    return std::get<Unshard>(a).EqualsWithoutContinuation(std::get<Unshard>(b));
  }
  return false;
}

inline std::shared_ptr<const TensorTransformation> GetContinuation(
    const TensorTransformation* transformation) {
  if (transformation == nullptr) {
    return nullptr;
  }
  return std::visit([](const auto& arg) { return arg.continuation; },
                    *transformation);
}

// Appends a continuation to a transformation and returns the new
// transformation. Note this function simply returns a new object without
// modifying the original transformation.
std::shared_ptr<const TensorTransformation> AppendContinuation(
    std::shared_ptr<const TensorTransformation> curr,
    std::shared_ptr<const TensorTransformation> to_append);

inline bool operator==(const TensorTransformation& a,
                       const TensorTransformation& b);

inline bool operator==(const Reshape& a, const Reshape& b) {
  bool continuation_equal;
  if (a.continuation == nullptr && b.continuation == nullptr) {
    continuation_equal = true;
  } else if (a.continuation != nullptr && b.continuation != nullptr) {
    continuation_equal = *a.continuation == *b.continuation;
  } else {
    continuation_equal = false;
  }
  return a.EqualsWithoutContinuation(b) && continuation_equal;
}

inline bool operator!=(const Reshape& a, const Reshape& b) { return !(a == b); }

inline bool operator==(const Unshard& a, const Unshard& b) {
  bool continuation_equal;
  if (a.continuation == nullptr && b.continuation == nullptr) {
    continuation_equal = true;
  } else if (a.continuation != nullptr && b.continuation != nullptr) {
    continuation_equal = *a.continuation == *b.continuation;
  } else {
    continuation_equal = false;
  }
  return a.EqualsWithoutContinuation(b) && continuation_equal;
}

inline bool operator!=(const Unshard& a, const Unshard& b) { return !(a == b); }

inline bool operator==(const Broadcast& a, const Broadcast& b) {
  bool continuation_equal;
  if (a.continuation == nullptr && b.continuation == nullptr) {
    continuation_equal = true;
  } else if (a.continuation != nullptr && b.continuation != nullptr) {
    continuation_equal = *a.continuation == *b.continuation;
  } else {
    continuation_equal = false;
  }
  return a.EqualsWithoutContinuation(b) && continuation_equal;
}

inline bool operator!=(const Broadcast& a, const Broadcast& b) {
  return !(a == b);
}

inline bool operator==(const TensorTransformation& a,
                       const TensorTransformation& b) {
  if (a.index() != b.index()) {
    return false;
  }
  if (a.valueless_by_exception()) {
    return true;
  }
  if (std::holds_alternative<Reshape>(a)) {
    return std::get<Reshape>(a) == std::get<Reshape>(b);
  }
  if (std::holds_alternative<Broadcast>(a)) {
    return std::get<Broadcast>(a) == std::get<Broadcast>(b);
  }
  if (std::holds_alternative<Unshard>(a)) {
    return std::get<Unshard>(a) == std::get<Unshard>(b);
  }
  return false;
}

inline bool operator!=(const TensorTransformation& a,
                       const TensorTransformation& b) {
  return !(a == b);
}

// Forward declarations due to recursion.
template <typename Sink>
void AbslStringify(Sink& sink, const TensorTransformation& rt);
template <typename H>
H AbslHashValue(H h, const TensorTransformation& rt);

template <typename Sink>
void AbslStringify(Sink& sink, const Reshape& r) {
  ::absl::Format(&sink, "Reshape{dimensions=[%s], continuation=",
                 ::absl::StrJoin(r.output_dimensions, ", "));
  if (r.continuation) {
    AbslStringify(sink, *r.continuation);
  } else {
    sink.Append("nullptr");
  }
  sink.Append("}");
}

template <typename Sink>
void AbslStringify(Sink& sink, const Unshard& u) {
  ::absl::Format(&sink, "Unshard{dimensions=[%s], sharding=%s, continuation=",
                 ::absl::StrJoin(u.original_dimensions, ", "),
                 u.sharding.ToString());
  if (u.continuation) {
    AbslStringify(sink, *u.continuation);
  } else {
    sink.Append("nullptr");
  }
  sink.Append("}");
}

template <typename Sink>
void AbslStringify(Sink& sink, const Broadcast& b) {
  ::absl::Format(&sink,
                 "Broadcast{dimensions=[%s], broadcast_dimensions=[%s], "
                 "continuation=",
                 ::absl::StrJoin(b.output_dimensions, ", "),
                 ::absl::StrJoin(b.broadcast_dimensions, ", "));
  if (b.continuation) {
    AbslStringify(sink, *b.continuation);
  } else {
    sink.Append("nullptr");
  }
  sink.Append("}");
}

template <typename Sink>
void AbslStringify(Sink& sink, const TensorTransformation& rt) {
  std::visit([&](const auto& v) { AbslStringify(sink, v); }, rt);
}

template <typename H>
H AbslHashValue(H h, const Reshape& r) {
  return H::combine(std::move(h), r.output_dimensions, r.continuation);
}

template <typename H>
H AbslHashValue(H h, const Unshard& u) {
  return H::combine(std::move(h), u.original_dimensions, u.sharding,
                    u.continuation);
}

template <typename H>
H AbslHashValue(H h, const Broadcast& b) {
  return H::combine(std::move(h), b.output_dimensions, b.broadcast_dimensions,
                    b.continuation);
}

template <typename H>
H AbslHashValue(H h, const TensorTransformation& rt) {
  return std::visit(
      [&](const auto& v) { return H::combine(std::move(h), rt.index(), v); },
      rt);
}

std::string ToString(const TensorTransformation* transformation);

void ToProto(const TensorTransformation* transformation,
             google::protobuf::RepeatedPtrField<TensorTransformationProto>* proto_field);

::absl::StatusOr<std::shared_ptr<const TensorTransformation>> FromProto(
    const google::protobuf::RepeatedPtrField<TensorTransformationProto>& proto_field);

}  // namespace tensor_transformation

struct ScopeInstruction {
  // The name of the call or loop instruction
  std::string instruction_name;
  // Only set if this instruction is a loop instruction.
  //
  // -1 is reserved to
  // mean this matches any iteration index. This is useful when a loop
  // instruction is hoisted out of the loop body. In string format this is
  // encoded as `*`.
  //
  // The wildcard index (-1) is necessary to handle transformations like
  // loop-invariant code motion. When an expression within a loop is found to be
  // invariant across iterations, compilers can optimize by moving this
  // expression outside the loop. This means an instruction that originally
  // executed inside each iteration (producing a tensor per iteration) now
  // executes only once before the loop. To maintain the correspondence, the
  // single tensor produced by the hoisted instruction in the optimized version
  // needs to be matched with all the tensors produced by the original
  // instruction in each iteration of the unoptimized version. The -1 index
  // acts as a wildcard, indicating that this ScopeInstruction instance
  // represents the hoisted instruction and should match any iteration index of
  // its counterpart within the loop.
  //
  // For example, consider a loop that runs 5 times. If an instruction `A` is
  // inside the loop in the baseline, we will have 5 distinct tensors
  // (A#0, A#1, A#2, A#3, A#4). If, in the target, instruction `A` is hoisted
  // out of the loop, there will be only one tensor. We would represent the
  // hoisted tensor's scope with iteration_index = -1 (e.g., A#*). This allows
  // the single hoisted tensor (A#*) in the target to be correctly compared
  // against all 5 original tensors (A#0 through A#4) from the baseline.
  //
  // -2 is reserved to mean this iteration index should be replaced by the
  // actual iteration index at runtime. In string format this is encoded as `$`.
  // This is used by original call-like instruction tracking. For example,
  // consider the following program with nested loops written in pseudocode:
  //
  // ```
  // while.1 (i; i < 5; i++) {
  //   constant = 1
  //   while.2 (j; j < 3; j++) {
  //     constant += j
  //   }
  //   a = constant + i
  // }
  // ```
  // Here while.2 can be hoisted out of while.1 by invariant code motion. So
  // after optimiazation this becomes
  //
  // ```
  // constant' = 1
  // while.2' (j; j < 3; j++) {
  //   constant' += j
  // }
  // while.1' (i; i < 5; i++) {
  //   a' = constant' + i
  // }
  // ```
  //
  // Now, the original call-like instruction of `while.2'` should be
  // `whiile.1#*/while.2#$`. This indicates that at runtime when the scope
  // instruction is `while.2'#3`, the recovered original scope instruction
  // should be `while.1#*/while.2#3`, note `$` is replaced by the actual
  // iteration index 3.
  int64_t iteration_index = 0;

  // Returns true if this scope instruction matches any iteration index.
  bool MatchesAnyIteration() const { return iteration_index == -1; }

  std::string ToString() const {
    std::string result = instruction_name;
    if (iteration_index == -1) {
      ::absl::StrAppend(&result, "#*");
    } else if (iteration_index == -2) {
      ::absl::StrAppend(&result, "#$");
    } else if (iteration_index != 0) {
      ::absl::StrAppend(&result, "#", iteration_index);
    }
    return result;
  }

  static ScopeInstruction Create(::absl::string_view instruction_name,
                                 int64_t iteration_index = 0) {
    return ScopeInstruction{/*instruction_name=*/std::string(instruction_name),
                            /*iteration_index=*/iteration_index};
  }

  static ScopeInstruction FromString(::absl::string_view instruction_name) {
    if (::absl::EndsWith(instruction_name, "#*")) {
      return Create(instruction_name.substr(0, instruction_name.size() - 2),
                    -1);
    }
    if (::absl::EndsWith(instruction_name, "#$")) {
      return Create(instruction_name.substr(0, instruction_name.size() - 2),
                    -2);
    }
    if (::absl::StrContains(instruction_name, "#")) {
      std::vector<std::string> parts = ::absl::StrSplit(instruction_name, '#');
      int64_t iteration_index;
      CHECK(::absl::SimpleAtoi(parts.back(), &iteration_index));
      return Create(parts.front(), iteration_index);
    }
    return Create(instruction_name);
  }

  ScopeInstructionProto ToProto() const {
    ScopeInstructionProto proto;
    proto.set_instruction_name(instruction_name);
    proto.set_iteration_index(iteration_index);
    return proto;
  }

  static ScopeInstruction FromProto(const ScopeInstructionProto& proto) {
    return Create(proto.instruction_name(), proto.iteration_index());
  }

  friend bool operator==(const ScopeInstruction& a, const ScopeInstruction& b) {
    return a.instruction_name == b.instruction_name &&
           a.iteration_index == b.iteration_index;
  }
  friend bool operator!=(const ScopeInstruction& a, const ScopeInstruction& b) {
    return !(a == b);
  }
  friend bool operator<(const ScopeInstruction& a, const ScopeInstruction& b) {
    return std::tie(a.instruction_name, a.iteration_index) <
           std::tie(b.instruction_name, b.iteration_index);
  }
  friend bool operator>(const ScopeInstruction& a, const ScopeInstruction& b) {
    return b < a;
  }
  friend bool operator<=(const ScopeInstruction& a, const ScopeInstruction& b) {
    return !(b < a);
  }
  friend bool operator>=(const ScopeInstruction& a, const ScopeInstruction& b) {
    return !(a < b);
  }
  template <typename H>
  friend H AbslHashValue(H h, const ScopeInstruction& s) {
    return H::combine(std::move(h), s.instruction_name, s.iteration_index);
  }
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ScopeInstruction& s) {
    sink.Append(s.ToString());
  }
  friend std::ostream& operator<<(std::ostream& os, const ScopeInstruction& s) {
    return os << s.ToString();
  }
};

struct TensorKey {
  // The name of the leaf instruction.
  std::string instruction_name;
  // The shape index of the leaf instruction.
  xla::ShapeIndex shape_index;

  TensorKey() = default;
  // NOLINTNEXTLINE
  TensorKey(std::string instruction_name, xla::ShapeIndex shape_index = {})
      : instruction_name(std::move(instruction_name)),
        shape_index(std::move(shape_index)) {}

  std::string ToString() const {
    std::string result = instruction_name;
    if (!shape_index.empty()) {
      ::absl::StrAppend(&result, "@", shape_index.ToString());
    }
    return result;
  }

  static TensorKey Create(::absl::string_view instruction_name,
                          xla::ShapeIndex shape_index = {}) {
    return TensorKey(/*instruction_name=*/std::string(instruction_name),
                     /*shape_index=*/std::move(shape_index));
  }

  TensorKeyProto ToProto() const {
    TensorKeyProto proto;
    proto.set_instruction_name(instruction_name);
    for (int64_t index : shape_index) {
      proto.add_shape_index(index);
    }
    return proto;
  }

  static TensorKey FromProto(const TensorKeyProto& proto) {
    return Create(
        proto.instruction_name(),
        ShapeIndex(proto.shape_index().begin(), proto.shape_index().end()));
  }

  friend bool operator==(const TensorKey& a, const TensorKey& b) {
    return a.instruction_name == b.instruction_name &&
           a.shape_index == b.shape_index;
  }
  friend bool operator!=(const TensorKey& a, const TensorKey& b) {
    return !(a == b);
  }
  friend bool operator<(const TensorKey& a, const TensorKey& b) {
    return std::tie(a.instruction_name, a.shape_index) <
           std::tie(b.instruction_name, b.shape_index);
  }
  friend bool operator>(const TensorKey& a, const TensorKey& b) {
    return b < a;
  }
  friend bool operator<=(const TensorKey& a, const TensorKey& b) {
    return !(b < a);
  }
  friend bool operator>=(const TensorKey& a, const TensorKey& b) {
    return !(a < b);
  }
  template <typename H>
  friend H AbslHashValue(H h, const TensorKey& s) {
    return H::combine(std::move(h), s.instruction_name, s.shape_index);
  }
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const TensorKey& s) {
    sink.Append(s.ToString());
  }
  friend std::ostream& operator<<(std::ostream& os, const TensorKey& s) {
    return os << s.ToString();
  }
};

struct ScopedTensorKey {
  // The scope instructions that are used to locate the call or loop
  // instruction for this tensor during execution. They are like stack frames
  // in a stack trace. They are ordered from the outermost to the innermost.
  // That is, the last instruction is the immediate call or loop instruction
  // that contains this tensor.
  std::vector<ScopeInstruction> scope_instructions;
  TensorKey tensor_key;

  ScopedTensorKey() = default;
  ScopedTensorKey(std::vector<ScopeInstruction> scope_instructions,
                  TensorKey tensor_key)
      : scope_instructions(std::move(scope_instructions)),
        tensor_key(std::move(tensor_key)) {}
  // NOLINTNEXTLINE
  ScopedTensorKey(TensorKey tensor_key) : tensor_key(std::move(tensor_key)) {}

  // Returns a string representation of the ScopedTensorKey.
  // Scope instructions are separated by `/`. Each scope instruction includes
  // its name, followed by `#` and the iteration index if the index is non-zero.
  // The TensorKey's instruction name is appended after the scopes, separated by
  // `/`. The shape index is appended to the TensorKey's instruction name,
  // separated by `@`, but only if the shape index is not empty.
  //
  // Examples:
  // - {scope_instructions: [{name: "loop", index: 3}], tensor_key: {name:
  // "hlo", shape_index: {0, 1}}} -> "loop#3/hlo@{0,1}"
  // - {scope_instructions: [{name: "call", index: 0}], tensor_key: {name:
  // "hlo", shape_index: {}}} -> "call/hlo"
  // - {scope_instructions: [], tensor_key: {name: "hlo", shape_index: {2}}} ->
  // "hlo@{2}"
  std::string ToString() const {
    std::string result;
    if (!scope_instructions.empty()) {
      std::vector<std::string> scope_parts;
      scope_parts.reserve(scope_instructions.size());
      for (const auto& scope : scope_instructions) {
        scope_parts.push_back(scope.ToString());
      }
      ::absl::StrAppend(&result, ::absl::StrJoin(scope_parts, "/"), "/");
    }
    ::absl::StrAppend(&result, tensor_key.ToString());
    return result;
  }

  static ScopedTensorKey Create(
      TensorKey tensor_key,
      std::vector<ScopeInstruction> scope_instructions = {}) {
    return ScopedTensorKey(/*scope_instructions=*/std::move(scope_instructions),
                           /*tensor_key=*/std::move(tensor_key));
  }

  static ScopedTensorKey FromString(std::string str,
                                    xla::ShapeIndex shape_index = {}) {
    std::vector<::absl::string_view> parts = ::absl::StrSplit(str, '/');
    std::vector<ScopeInstruction> scope_instructions;
    scope_instructions.reserve(parts.size() - 1);
    for (int i = 0; i < parts.size() - 1; ++i) {
      scope_instructions.push_back(ScopeInstruction::FromString(parts[i]));
    }
    return Create(TensorKey::Create(parts.back(), shape_index),
                  std::move(scope_instructions));
  }

  AbsoluteScopedTensorKeyProto ToProto() const {
    AbsoluteScopedTensorKeyProto proto;
    *proto.mutable_tensor_key() = tensor_key.ToProto();
    for (const auto& scope : scope_instructions) {
      *proto.add_scope_instructions() = scope.ToProto();
    }
    return proto;
  }

  static ScopedTensorKey FromProto(const AbsoluteScopedTensorKeyProto& proto) {
    std::vector<ScopeInstruction> scope_instructions;
    scope_instructions.reserve(proto.scope_instructions_size());
    for (const auto& scope_proto : proto.scope_instructions()) {
      scope_instructions.push_back(ScopeInstruction::FromProto(scope_proto));
    }
    return Create(TensorKey::FromProto(proto.tensor_key()),
                  std::move(scope_instructions));
  }

  friend bool operator==(const ScopedTensorKey& a, const ScopedTensorKey& b) {
    return a.scope_instructions == b.scope_instructions &&
           a.tensor_key == b.tensor_key;
  }
  friend bool operator!=(const ScopedTensorKey& a, const ScopedTensorKey& b) {
    return !(a == b);
  }
  friend bool operator<(const ScopedTensorKey& a, const ScopedTensorKey& b) {
    return std::tie(a.scope_instructions, a.tensor_key) <
           std::tie(b.scope_instructions, b.tensor_key);
  }
  friend bool operator>(const ScopedTensorKey& a, const ScopedTensorKey& b) {
    return b < a;
  }
  friend bool operator<=(const ScopedTensorKey& a, const ScopedTensorKey& b) {
    return !(b < a);
  }
  friend bool operator>=(const ScopedTensorKey& a, const ScopedTensorKey& b) {
    return !(a < b);
  }
  template <typename H>
  friend H AbslHashValue(H h, const ScopedTensorKey& s) {
    return H::combine(std::move(h), s.scope_instructions, s.tensor_key);
  }
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ScopedTensorKey& s) {
    sink.Append(s.ToString());
  }
  friend std::ostream& operator<<(std::ostream& os, const ScopedTensorKey& s) {
    return os << s.ToString();
  }
};
// AbsoluteScopedTensorKey has the full scopes.
using AbsoluteScopedTensorKey = ScopedTensorKey;
// RelativeScopedTensorKey does not have the full scopes. This is typical for
// instructions that partially inlined or partially unrolled. For example,
// a nested computation may be inlined for one level, leaving the outer call
// untouched. In this case, the original value would contain such a relative
// key.
using RelativeScopedTensorKey = ScopedTensorKey;

struct OriginalTensorSummary {
  // The dimensions of the original tensor.
  std::vector<int64_t> dimensions;
  // The summaries of the original tensor. The vector may contain multiple
  // summaries if the original tensor has manual sharding, in which case there
  // are multiple "global" views for the original tensor, one for each manual
  // subgroup. For more details about the concept of manual sharding, see
  // https://docs.jax.dev/en/latest/notebooks/shard_map.html
  std::vector<xla::comparison::FloatSummary> summaries;

  RecoveredTensorSummaryProto::OriginalTensorSummaryProto ToProto() const;

  static OriginalTensorSummary FromProto(
      const RecoveredTensorSummaryProto::OriginalTensorSummaryProto& proto);

  // Returns a human-readable string representation of the
  // OriginalTensorSummary.
  std::string ToDebugString() const {
    std::string result = "OriginalTensorSummary{\n";
    ::absl::StrAppend(&result, "  dimensions: [",
                      ::absl::StrJoin(dimensions, ", "), "]\n");
    ::absl::StrAppend(&result, "  summaries:\n");
    for (const auto& summary : summaries) {
      ::absl::StrAppend(&result, "    split_spec:\n");
      for (const auto& spec : summary.split_spec) {
        ::absl::StrAppend(&result, "      {dim_index: ", spec.dim_index,
                          ", block_count: ", spec.block_count, "}\n");
      }

      ::absl::StrAppend(&result, "    block_summaries:\n");
      for (const auto& block : summary.block_summaries) {
        ::absl::StrAppend(&result, "      {block_indices: [",
                          ::absl::StrJoin(block.block_indices, ", "),
                          "], min: ", block.min, ", max: ", block.max,
                          ", mean: ", block.mean, ", stddev: ", block.stddev,
                          ", count: ", block.count,
                          ", nan_count: ", block.nan_count,
                          ", pos_inf_count: ", block.pos_inf_count,
                          ", neg_inf_count: ", block.neg_inf_count,
                          ", zero_count: ", block.zero_count, "}\n");
      }
    }
    ::absl::StrAppend(&result, "}\n");
    return result;
  }
};

// The C++ struct version of `RecoveredTensorSummaryProto`.
struct RecoveredTensorSummary {
  AbsoluteScopedTensorKey original_tensor_key;
  std::shared_ptr<const tensor_transformation::TensorTransformation>
      pending_transformation;
  OriginalTensorSummary original_tensor_summary;
};

// `original_tensor_key`: The key of the original tensor.
// `pending_transformation`: The recovering transformations that are pending
// to be applied to the optimized tensor to recover the original tensor. These
// transformations are delayed to be applied during comparison instead. This
// is so that the comparison tool can apply recovering transformations only
// until the common suffix of the two sequences of pending transformations.
// This is desirable because transforming summaries can be lossy.
// `original_tensor_summary`: The summary of the original tensor.
using OriginalTensorSummaryCallback = std::function<::absl::Status(
    const AbsoluteScopedTensorKey& original_tensor_key,
    std::shared_ptr<const tensor_transformation::TensorTransformation>
        pending_transformation,
    const OriginalTensorSummary& original_tensor_summary)>;

using IsOriginalTensorAlreadyRecoveredCallback =
    std::function<bool(const AbsoluteScopedTensorKey& tensor_key)>;

// Applies a reshape transformation to a tensor summary. This function tries to
// preserve the block structure as much as possible. If a dimension that is
// split into blocks is not affected by the reshape (i.e., it is in the common
// prefix or suffix of the dimensions before and after the reshape), the block
// structure along that dimension is maintained. Blocks are only merged when
// the split dimension is part of the reshaped portion of the tensor.
//
// For example, reshaping a tensor from [2, 3, 4, 5] to [2, 12, 5]. The
// dimensions 0 and 3 of the original shape are preserved (as dimensions 0 and
// 2 of the new shape). If the summary has splits along these dimensions, the
// splits will be carried over to the new summary. Any splits along the reshaped
// dimensions (1 and 2 of the original shape) will be merged.
xla::comparison::FloatSummary ApplyReshapeToSummary(
    const xla::comparison::FloatSummary& summary,
    ::absl::Span<const int64_t> current_shape,
    ::absl::Span<const int64_t> new_shape);

// Applies a transpose transformation to a tensor summary. The transpose
// operation preserves all blocks, but permutes dimensions in split_spec
// and block_indices.
// The permutation follows HLO's transpose convention:
// `output_dimensions[i] = input_dimensions[permutation[i]]`.
// This means input dimension `permutation[i]` becomes output dimension `i`.
// Therefore, a split on input dimension `d` becomes a split on output dimension
// `k` such that `permutation[k]=d`, which means
// `k=InversePermutation(permutation)[d]`.
//
// Example:
//   If current_shape = [10, 20, 30], permutation = {2, 0, 1},
//   it means input dimension 2 becomes output dimension 0, input dimension 0
//   becomes output dimension 1, and input dimension 1 becomes output
//   dimension 2. The inverse permutation is {1, 2, 0}.
//   If summary.split_spec = [{0, 2}, {1, 4}], indicating input dimension 0
//   is split 2 ways and input dimension 1 is split 4 ways,
//   ApplyTransposeToSummary will produce a new summary where output
//   dimension 1 is split 2 ways (from input dim 0) and output dimension 2 is
//   split 4 ways (from input dim 1). The resulting split_spec will be
//   [{1, 2}, {2, 4}] (sorted by dimension index).
//   A block with `block_indices = [i, j]` in the input summary corresponds to
//   the i-th block of dim 0 and j-th block of dim 1. In the output summary,
//   this will correspond to i-th block of dim 1 and j-th block of dim 2,
//   so `block_indices` will remain `[i, j]` because `new_split_spec = [{1,
//   2}, {2, 4}]` is sorted by dimension index.
xla::comparison::FloatSummary ApplyTransposeToSummary(
    const xla::comparison::FloatSummary& summary,
    ::absl::Span<const int64_t> current_shape,
    ::absl::Span<const int64_t> permutation);

// Applies a broadcast transformation to a tensor summary. This simulates the
// effect of an HLO broadcast operation, where dimensions are added or operand
// dimensions are mapped to different output dimensions. The
// `broadcast_dimensions` parameter maps input dimensions to output dimensions.
//
// This function also supports inverse broadcast operations where dimensions are
// dropped. If `broadcast_dimensions[i] == -1`, it signifies that dimension `i`
// of `current_shape` is being dropped in `new_shape`. If dimension `i` is
// split in `summary.split_spec`, blocks along this dimension will be merged.
//
// Blocks remain logically the same after broadcasting, but `split_spec` and
// `block_indices` are updated to reflect dimension changes. Specifically,
// dimensions in `split_spec` are mapped to new dimensions in the output shape
// according to `broadcast_dimensions`, and `block_indices` are reordered to
// match the sorted order of the new `split_spec`. If dimensions are dropped,
// they are removed from `split_spec`, and corresponding blocks are merged.
//
// The min, max, mean, and stddev statistics of each block remain unchanged by
// broadcast replication, but merging blocks will combine their statistics. All
// the counts of each block is multiplied by `new_elements / current_elements`
// to reflect the change in tensor size.
//
// Example:
//   If `current_shape = [2, 3]`, `new_shape = [4, 2, 3]`, and
//   `broadcast_dimensions = {1, 2}` (meaning input dim 0 maps to output dim 1,
//   and input dim 1 maps to output dim 2), and if `summary.split_spec = [{0,
//   2}]`, then `ApplyBroadcastToSummary` will produce a new summary where
//   `split_spec = [{1, 2}]`.
//   If a block in `summary` has `block_indices = [i]` and `count = c`,
//   the corresponding block in `new_summary` will have `block_indices = [i]`
//   and `count = c * 4` because `new_elements / current_elements = 24 / 6 = 4`.
//
// Example 2:
//   If `current_shape = [2, 3]`, `new_shape = [4, 2, 3]`,
//   `broadcast_dimensions = {1, 2}`, and `summary.split_spec = [{0, 2}, {1,
//   3}]`, then `new_summary.split_spec` will be `[{1, 2}, {2, 3}]`. A block
//   with `block_indices = [i, j]` in `summary` corresponds to the i-th block of
//   dim 0 and j-th block of dim 1. In `new_summary`, this block will correspond
//   to the i-th block of dim 1 and j-th block of dim 2, so `block_indices` will
//   remain `[i, j]` because `new_split_spec = [{1, 2}, {2, 3}]` is sorted by
//   dimension index.
//
// Example 3 (Inverse Broadcast):
//   If `current_shape = [4, 2, 3]`, `new_shape = [2, 3]`,
//   `broadcast_dimensions = {-1, 0, 1}`, and `summary.split_spec = [{0, 2}, {1,
//   2}]` (dim 0 split 2 ways, dim 1 split 2 ways),
//   then dimension 0 of `current_shape` is dropped. The split on dim 0 will be
//   removed, and blocks along dim 0 will be merged. The split on dim 1 of
//   `current_shape` becomes split on dim 0 of `new_shape`.
//   `new_summary.split_spec` will be `[{0, 2}]`.
xla::comparison::FloatSummary ApplyBroadcastToSummary(
    const xla::comparison::FloatSummary& summary,
    ::absl::Span<const int64_t> current_shape,
    ::absl::Span<const int64_t> new_shape,
    ::absl::Span<const int64_t> broadcast_dimensions);

// Applies a chain of recovering transformations to a tensor summary.
//
// **Important:** The provided `transformation` chain must *not* include any
// `Unshard` transformations. An error will be returned if an `Unshard` is
// encountered.
//
// The `stopping_transformation` parameter defines an optional end point for
// the application. Transformations are applied starting from the given
// `transformation` and continue up to, but *not including*, the
// `stopping_transformation`. If `stopping_transformation` is `nullptr`, the
// entire chain of transformations is applied.
::absl::StatusOr<OriginalTensorSummary>
ApplyNonUnshardTensorTransformationToSummary(
    const OriginalTensorSummary& original_tensor_summary,
    tensor_transformation::TensorTransformation const* transformation,
    tensor_transformation::TensorTransformation const* stopping_transformation);

// Aligns the block structures of two `OriginalTensorSummary` objects to make
// them compatible for comparison. This is achieved by adjusting their
// `split_spec` to a common, coarser block structure. The aligned summaries
// are returned as a pair of <baseline, target> summaries.
//
// The alignment rules are as follows:
// 1.  **Dimensions Split in Both Summaries:** If a dimension is split in both
//     `baseline_tensor_summary` and `target_tensor_summary`, the new number of
//     splits for that dimension in both summaries will be the greatest common
//     divisor (GCD) of their original split counts. This effectively merges
//     blocks to the largest common block size.
// 2.  **Dimensions Split in Only One Summary:** If a dimension is split in
//     one summary but not the other, the summary where the dimension was split
//     will have its blocks along that dimension merged into a single block.
//     The other summary remains unchanged for this dimension (as it was already
//     a single block).
//
// Examples:
// Let DimSplitSpec be represented as {dim_index: block_count}.
//
// Example 1: Shared Split Dimensions
//   - baseline: split_spec = [{0: 4}, {2: 6}]
//   - target:   split_spec = [{0: 6}, {2: 4}]
//   - Aligned:
//     - Dimension 0: GCD(4, 6) = 2. Both will be split into 2 blocks.
//     - Dimension 2: GCD(6, 4) = 2. Both will be split into 2 blocks.
//     - Result: baseline' and target' both have split_spec = [{0: 2}, {2: 2}]
//
// Example 2: Unique Split Dimensions
//   - baseline: split_spec = [{1: 8}]
//   - target:   split_spec = [{1: 8}, {3: 5}]
//   - Aligned:
//     - Dimension 1: Split in both, GCD(8, 8) = 8. Remains 8 blocks.
//     - Dimension 3: Split only in target. Baseline has no split here.
//       Target's dimension 3 blocks are merged.
//     - Result:
//       - baseline': split_spec = [{1: 8}]
//       - target':   split_spec = [{1: 8}] (Dim 3 merged)
//
// Example 3: Combined Scenario
//   - baseline: split_spec = [{0: 10}, {1: 4}]
//   - target:   split_spec = [{0: 4}, {2: 5}]
//   - Aligned:
//     - Dimension 0: Split in both, GCD(10, 4) = 2. Both become 2 blocks.
//     - Dimension 1: Split only in baseline. Baseline's Dim 1 blocks are
//     merged.
//     - Dimension 2: Split only in target. Target's Dim 2 blocks are merged.
//     - Result:
//       - baseline': split_spec = [{0: 2}]
//       - target':   split_spec = [{0: 2}]
::absl::StatusOr<std::pair<OriginalTensorSummary, OriginalTensorSummary>>
AlignTensorSummaries(const OriginalTensorSummary& baseline_tensor_summary,
                     const OriginalTensorSummary& target_tensor_summary);

// Extracts the absolute scoped tensor key from the log hlo output metadata.
AbsoluteScopedTensorKey GetAbsoluteScopedTensorKey(
    xla::LogHloOutputMetadata log_hlo_output_metadata);

// Creates a `RecoveredTensorSummaryProto` from the arguments of
// `OriginalTensorSummaryCallback`.
RecoveredTensorSummaryProto CreateRecoveredTensorSummaryProto(
    const AbsoluteScopedTensorKey& original_tensor_key,
    std::shared_ptr<const tensor_transformation::TensorTransformation>
        pending_transformation,
    const OriginalTensorSummary& original_tensor_summary);

// Creates a `RecoveredTensorSummary` from a `RecoveredTensorSummaryProto`.
::absl::StatusOr<RecoveredTensorSummary> RecoveredTensorSummaryFromProto(
    const RecoveredTensorSummaryProto& proto);

bool IsCallLike(const HloInstruction& instr);

}  // namespace xla::numerics::comparison

#endif  // XLA_HLO_TOOLS_COMPARISON_ORIGINAL_TENSOR_SUMMARY_UTILS_H_
