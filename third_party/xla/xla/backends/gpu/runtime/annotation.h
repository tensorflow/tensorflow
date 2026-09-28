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

#ifndef XLA_BACKENDS_GPU_RUNTIME_ANNOTATION_H_
#define XLA_BACKENDS_GPU_RUNTIME_ANNOTATION_H_

#include <cstdint>
#include <optional>
#include <string>
#include <variant>

#include "absl/base/macros.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "tsl/profiler/lib/nvtx_utils.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// Prepared annotation payloads
//===----------------------------------------------------------------------===//

// Trace annotation levels are ordered: each level includes the metadata from
// lower levels.
enum class TraceAnnotationLevel : int32_t {
  kBasic = 0,
  kDetailed = 1,
};

// Prepared information for the top level NVTX/profiler range covering an
// HloModule
class ModuleAnnotation {
 public:
  explicit ModuleAnnotation(absl::string_view module_name);
  explicit ModuleAnnotation(const HloModule& mod);

  absl::string_view longest_op_name_prefix() const { return longest_prefix_; }
  explicit operator absl::string_view() const { return title_str_; }
  tsl::profiler::StringHandle title() const { return title_; }
  static uint64_t NvtxSchemaId();
  int32_t common_stack_frames() const { return common_stack_frames_; }

 private:
  friend void RangePush(tsl::profiler::ProfilerDomainHandle domain,
                        const ModuleAnnotation& annotation);

  std::string longest_prefix_;
  std::string title_str_;
  tsl::profiler::StringHandle title_;
  tsl::profiler::StringHandle module_name_;
  tsl::profiler::StringHandle common_src_locations_;
  int32_t module_id_;
  int32_t common_stack_frames_;
};

// Prepared NVTX and XProf information for an HLO instruction within a module.
class InstructionAnnotation {
 public:
  InstructionAnnotation(
      const ModuleAnnotation& module_annotation, const HloInstruction& inst,
      TraceAnnotationLevel annotation_level = TraceAnnotationLevel::kBasic);

  // NVTX uses a compact registered name because detailed metadata is carried
  // separately in the structured payload.
  absl::string_view nvtx_name() const { return nvtx_name_str_; }

  // XProf records a string annotation, so at the detailed level this name also
  // contains the metadata as parseable key-value fields.
  absl::string_view xprof_name() const { return xprof_name_str_; }

  bool has_detailed_annotations() const {
    return !std::holds_alternative<Basic>(payload_);
  }
  bool is_collective_annotation() const {
    return std::holds_alternative<Collective>(payload_);
  }

 private:
  struct Basic {
    static uint64_t NvtxSchemaId();

    tsl::profiler::StringHandle hlo_dump;
    tsl::profiler::StringHandle src_locations;
    tsl::profiler::StringHandle called_hlo_dump;
  };

  struct Detailed {
    static uint64_t NvtxSchemaId();

    Basic basic;
    tsl::profiler::StringHandle hlo_op_name;
    int64_t hlo_op_id;
    tsl::profiler::StringHandle op_type;
    tsl::profiler::StringHandle op_name;
    tsl::profiler::StringHandle source_file;
    int32_t source_line;
    tsl::profiler::StringHandle output_shape;
  };

  struct Collective {
    static uint64_t NvtxSchemaId();

    Detailed detailed;
    tsl::profiler::StringHandle replica_groups;
    uint8_t is_pipelined;
    uint8_t is_spmd_generated;
    tsl::profiler::StringHandle collective_group_key;
    tsl::profiler::StringHandle combiner_key;
    tsl::profiler::StringHandle scheduling_group_id;
    tsl::profiler::StringHandle stream_annotation;
  };

  friend void RangePush(tsl::profiler::ProfilerDomainHandle domain,
                        const InstructionAnnotation& annotation);

  std::string nvtx_name_str_;
  std::string xprof_name_str_;
  tsl::profiler::StringHandle nvtx_name_;
  std::variant<Basic, Detailed, Collective> payload_;
};

//===----------------------------------------------------------------------===//
// Per-module annotation collection
//===----------------------------------------------------------------------===//

// Parsed/prepared information for an HloModule that gets propagated to NVTX
// ranges/profilers/... at execution time.
struct ModuleAnnotations {
  explicit ModuleAnnotations(
      absl::string_view module_name,
      TraceAnnotationLevel annotation_level = TraceAnnotationLevel::kBasic);
  explicit ModuleAnnotations(
      const HloModule&,
      TraceAnnotationLevel annotation_level = TraceAnnotationLevel::kBasic);

  ModuleAnnotation top_level;
  absl::flat_hash_map<absl::string_view, InstructionAnnotation> instructions;
};

//===----------------------------------------------------------------------===//
// Scoped RAII helper to set and restore thread local module annotations
//===----------------------------------------------------------------------===//

class ScopedModuleAnnotations {
 public:
  explicit ScopedModuleAnnotations(const ModuleAnnotations* annotations);
  ~ScopedModuleAnnotations();

 private:
  const ModuleAnnotations* restore_;
};

const ModuleAnnotations* GetCurrentModuleAnnotations();

std::optional<tsl::profiler::ScopedAnnotation> GetInstructionAnnotation(
    absl::string_view profile_annotation);

ABSL_DEPRECATE_AND_INLINE()
inline std::optional<tsl::profiler::ScopedAnnotation> GetKernelAnnotation(
    absl::string_view profile_annotation) {
  return GetInstructionAnnotation(profile_annotation);
}

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_ANNOTATION_H_
