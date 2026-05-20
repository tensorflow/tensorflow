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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_DYNAMIC_SLICE_FUSION_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_DYNAMIC_SLICE_FUSION_H_

#include <cstdint>
#include <optional>
#include <ostream>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla::gpu {

// Utilities for analyzing dynamic-slice fusion instructions. These fusions are
// created by the dynamic-slice fusion rewriter when it can prove that DS/DUS
// offsets depend only on the while-loop induction variable or are fully static.
//
// A dynamic-slice fusion wraps a hero computation (e.g. a custom call or a
// kernel fusion) with dynamic-slice (DS) inputs and dynamic-update-slice (DUS)
// outputs. At run time, the DynamicSliceFusionThunk adjusts buffer offsets
// per while-loop iteration so the hero operates on a different slice each time.
//
// Example HLO (a custom call reading and writing one row per iteration of a
// 4-iteration while loop). The induction variable `ivar` is passed as a fusion
// parameter so the runtime can verify the annotated offset.
//
//   %dsf_computation {
//     %p0 = f32[4,8,8] parameter(0)         // input buffer
//     %p1 = s32[] parameter(1)              // ivar (runtime offset)
//     %p2 = f32[4,8,8] parameter(2)         // output buffer
//     %c0 = s32[] constant(0)
//     %ds = f32[1,8,8] dynamic-slice(%p0, %p1, %c0, %c0),
//       dynamic_slice_sizes={1,8,8},
//       backend_config={"dynamic_slice_config":{
//         "loop_index":0, "byte_offset":0, "byte_stride":256}}
//     %bc_in = f32[8,8] bitcast(%ds)
//     %hero = f32[8,8] custom-call(%bc_in),
//       custom_call_target="fake_target"
//     %bc_out = f32[1,8,8] bitcast(%hero)
//     ROOT %dus = f32[4,8,8] dynamic-update-slice(%p2, %bc_out, %p1, %c0, %c0),
//       backend_config={"dynamic_slice_config":{
//         "loop_index":0, "byte_offset":0, "byte_stride":256}}
//   }
//
//   body {
//     %param = (s32[], f32[4,8,8], f32[4,8,8]) parameter(0)
//     %ivar = s32[] get-tuple-element(%param), index=0
//     %input = f32[4,8,8] get-tuple-element(%param), index=1
//     %output = f32[4,8,8] get-tuple-element(%param), index=2
//     %updated = f32[4,8,8] fusion(%input, %ivar, %output),
//       kind=kCustom, calls=%dsf_computation,
//       backend_config={"fusion_backend_config":{
//         "kind":"__custom_fusion",
//         "custom_fusion_config":{"name":"dynamic_slice_fusion"}}}
//     %one = s32[] constant(1)
//     %next_ivar = s32[] add(%ivar, %one)
//     ROOT %result = (s32[], f32[4,8,8], f32[4,8,8])
//       tuple(%next_ivar, %input, %updated)
//   }
//
// This library extracts the hero, resolves parameter/result slices, and returns
// DynamicSliceConfig protos that the ThunkEmitter converts into runtime thunks.

// Analysis result for a dynamic-slice fusion. Describes how each hero
// parameter and result maps to fusion parameters and how offsets are computed
// at runtime.
struct DynamicSliceFusion {
  // A DS/DUS offset for one dimension that is a compile-time constant (e.g.
  // `s32[] constant(0)` inside the fusion body). The actual byte contribution
  // for this dimension is `offset * byte_stride_for_dimension`.
  struct ConstantOffset {
    int64_t offset;            // Constant index value for this dimension.
    int64_t dimension_number;  // Which dimension of the DS/DUS this applies to.
  };

  // A DS/DUS offset for one dimension that comes from a fusion parameter
  // holding a runtime scalar (e.g. the loop induction variable passed as a
  // fusion operand). At emit time the thunk emitter maps `parameter_number`
  // to a BufferAllocation::Slice; at runtime the thunk D2H-copies the scalar
  // to verify that the annotated DynamicSliceConfig offset matches the actual
  // XLA-computed offset.
  struct RuntimeOffset {
    int64_t parameter_number;  // Fusion parameter index of the offset scalar.
    int64_t dimension_number;  // Which dimension of the DS/DUS this applies to.
  };

  // Per-dimension offset for a DS or DUS instruction: either a compile-time
  // constant (literal inside the fusion) or a reference to a runtime scalar
  // fusion parameter.
  using Offset = std::variant<ConstantOffset, RuntimeOffset>;

  // Parameter numbers in the structs below correspond to
  // HloParameterInstruction::parameter_number() inside the fusion body. Note
  // that a fusion parameter can be a tuple, which at runtime is flattened into
  // multiple buffers by the buffer assignment. The thunk emitter is responsible
  // for mapping these HLO-level parameter numbers to buffer slices.

  // A resolved hero operand: describes which fusion parameter backs the hero
  // input and how the buffer is sliced at runtime.
  struct Parameter {
    // Fusion parameter number of the buffer backing this hero operand
    // (the DS source or direct parameter).
    int64_t parameter_number;

    // Shape of the fusion parameter (the full, unsliced buffer).
    Shape parameter_shape;

    // Shape of the slice fed to the hero (DS output shape, or same as
    // parameter_shape when not sliced).
    Shape slice_shape;

    // DynamicSliceConfig from the DS backend_config. Encodes the host-side
    // offset formula: `offset + loop_iteration[loop_index] * stride`.
    // Absent when the operand is not sliced (direct parameter pass-through).
    std::optional<DynamicSliceConfig> slice_config;

    // Per-dimension offset info from the DS index operands. Absent when the
    // operand is not sliced (no DS between the parameter and the hero).
    std::optional<std::vector<Offset>> slice_offsets;
  };

  // A resolved hero result: describes which fusion parameter holds the DUS
  // target buffer and how the result is sliced back at runtime.
  struct Result {
    // Fusion parameter number of the DUS target buffer. Absent when the
    // result does not flow through a DUS.
    std::optional<int64_t> parameter_number;

    // Flat leaf index (DFS order) within the hero's output shape. 0 for a
    // single non-tuple result. For nested tuples like ((f32, f32), f32)
    // the leaves are numbered 0, 1, 2.
    int64_t result_number = 0;

    // Shape of the DUS target buffer (the full output buffer).
    Shape result_shape;

    // Shape of the DUS update (the hero output slice inserted into the
    // result buffer).
    Shape update_shape;

    // DynamicSliceConfig from the DUS backend_config. Absent when the
    // result does not flow through a DUS.
    std::optional<DynamicSliceConfig> update_config;

    // Per-dimension offset info from the DUS index operands. Absent when
    // the result does not flow through a DUS.
    std::optional<std::vector<Offset>> update_offsets;
  };

  // Finds the "hero" instruction inside a dynamic-slice fusion body.
  static const HloInstruction* FindHero(const HloComputation* body);

  // Resolves parameters for the hero instruction. Returns an entry for each
  // operand of the hero; the same fusion parameter may appear more than once
  // if it feeds multiple hero operands.
  static absl::StatusOr<std::vector<Parameter>> ResolveParameters(
      const HloInstruction* hero);

  // Resolves results for the hero instruction. Returns an entry for all results
  // of the dynamic slice fusion (root of the hero, or for each tuple entry).
  static absl::StatusOr<std::vector<Result>> ResolveResults(
      const HloInstruction* hero);
};

//===----------------------------------------------------------------------===//
// DynamicSliceFusion comparison and stringification
//===----------------------------------------------------------------------===//

inline bool operator==(const DynamicSliceFusion::ConstantOffset& a,
                       const DynamicSliceFusion::ConstantOffset& b) {
  return a.offset == b.offset && a.dimension_number == b.dimension_number;
}

inline bool operator==(const DynamicSliceFusion::RuntimeOffset& a,
                       const DynamicSliceFusion::RuntimeOffset& b) {
  return a.parameter_number == b.parameter_number &&
         a.dimension_number == b.dimension_number;
}

inline bool ConfigsEqual(const std::optional<DynamicSliceConfig>& a,
                         const std::optional<DynamicSliceConfig>& b) {
  if (a.has_value() != b.has_value()) {
    return false;
  }
  if (!a.has_value()) {
    return true;
  }
  return a->has_loop_index() == b->has_loop_index() &&
         a->loop_index() == b->loop_index() &&
         a->byte_offset() == b->byte_offset() &&
         a->byte_stride() == b->byte_stride();
}

inline bool operator==(const DynamicSliceFusion::Parameter& a,
                       const DynamicSliceFusion::Parameter& b) {
  return a.parameter_number == b.parameter_number &&
         a.parameter_shape == b.parameter_shape &&
         a.slice_shape == b.slice_shape &&
         ConfigsEqual(a.slice_config, b.slice_config) &&
         a.slice_offsets == b.slice_offsets;
}

inline bool operator==(const DynamicSliceFusion::Result& a,
                       const DynamicSliceFusion::Result& b) {
  return a.parameter_number == b.parameter_number &&
         a.result_number == b.result_number &&
         a.result_shape == b.result_shape && a.update_shape == b.update_shape &&
         ConfigsEqual(a.update_config, b.update_config) &&
         a.update_offsets == b.update_offsets;
}

template <typename Sink>
void AbslStringify(Sink& sink, const DynamicSliceFusion::ConstantOffset& o) {
  absl::Format(&sink, "c(d%d,%d)", o.dimension_number, o.offset);
}

template <typename Sink>
void AbslStringify(Sink& sink, const DynamicSliceFusion::RuntimeOffset& o) {
  absl::Format(&sink, "r(d%d,p%d)", o.dimension_number, o.parameter_number);
}

template <typename Sink>
void AbslStringify(Sink& sink, const DynamicSliceFusion::Offset& o) {
  std::visit([&sink](const auto& v) { AbslStringify(sink, v); }, o);
}

template <typename Sink>
void StringifyConfig(Sink& sink, const std::optional<DynamicSliceConfig>& c) {
  if (!c.has_value()) {
    absl::Format(&sink, "config{}");
    return;
  }
  absl::Format(&sink, "config{loop=%d, offset=%d, stride=%d}", c->loop_index(),
               c->byte_offset(), c->byte_stride());
}

template <typename Sink>
void AbslStringify(Sink& sink, const DynamicSliceFusion::Parameter& p) {
  absl::Format(&sink, "Parameter{param=%d %s->%s, ", p.parameter_number,
               ShapeUtil::HumanString(p.parameter_shape),
               ShapeUtil::HumanString(p.slice_shape));
  StringifyConfig(sink, p.slice_config);
  if (p.slice_offsets.has_value()) {
    absl::Format(&sink, ", offsets=[%s]",
                 absl::StrJoin(*p.slice_offsets, ", "));
  } else {
    absl::Format(&sink, ", offsets=none");
  }
  absl::Format(&sink, "}");
}

template <typename Sink>
void AbslStringify(Sink& sink, const DynamicSliceFusion::Result& r) {
  if (r.parameter_number.has_value()) {
    absl::Format(&sink, "Result{param=%d", *r.parameter_number);
  } else {
    absl::Format(&sink, "Result{param=none");
  }
  absl::Format(&sink, ", result=%d %s->%s, ", r.result_number,
               ShapeUtil::HumanString(r.update_shape),
               ShapeUtil::HumanString(r.result_shape));
  StringifyConfig(sink, r.update_config);
  if (r.update_offsets.has_value()) {
    absl::Format(&sink, ", offsets=[%s]",
                 absl::StrJoin(*r.update_offsets, ", "));
  } else {
    absl::Format(&sink, ", offsets=none");
  }
  absl::Format(&sink, "}");
}

inline std::ostream& operator<<(std::ostream& os,
                                const DynamicSliceFusion::ConstantOffset& o) {
  return os << absl::StrCat(o);
}

inline std::ostream& operator<<(std::ostream& os,
                                const DynamicSliceFusion::RuntimeOffset& o) {
  return os << absl::StrCat(o);
}

inline std::ostream& operator<<(std::ostream& os,
                                const DynamicSliceFusion::Offset& o) {
  return os << absl::StrCat(o);
}

inline std::ostream& operator<<(std::ostream& os,
                                const DynamicSliceFusion::Parameter& p) {
  return os << absl::StrCat(p);
}

inline std::ostream& operator<<(std::ostream& os,
                                const DynamicSliceFusion::Result& r) {
  return os << absl::StrCat(r);
}

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_DYNAMIC_SLICE_FUSION_H_
