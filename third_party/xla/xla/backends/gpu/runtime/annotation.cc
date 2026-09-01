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

#include "xla/backends/gpu/runtime/annotation.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/utils/hlo_longest_prefix.h"
#include "xla/printer.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/side_effect_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/nvtx_utils.h"
#include "tsl/profiler/lib/scoped_annotation.h"

#if GOOGLE_CUDA
#include "nvtx3/nvToolsExt.h"
#include "nvtx3/nvToolsExtPayload.h"
#endif

namespace xla::gpu {

using ::tsl::profiler::ScopedAnnotation;
using ::tsl::profiler::StringHandle;
using ::xla::hlo_longest_prefix::GetLongestOpNamePrefix;
using ::xla::hlo_longest_prefix::VisitInstAndCalledButNotOperands;

namespace {

StringHandle RegisterString(const std::string& str) {
  if (auto domain = tsl::profiler::DefaultProfilerDomain(); domain) {
    return tsl::profiler::RegisterString(domain, str);
  }
  return {};
}

StringHandle RegisterOptionalString(const std::string& str) {
  return str.empty() ? nullptr : RegisterString(str);
}

// Nsight Systems supports some basic HTML markup in annotation strings. This
// escaping stops things like <module> from disappearing.
std::ostream& PrintEscaped(std::ostream& os, absl::string_view str) {
  for (char c : str) {
    switch (c) {
      case '<':
        os << "&lt;";
        break;
      case '>':
        os << "&gt;";
        break;
      default:
        os << c;
    }
  }
  return os;
}

// Print options for profiler annotations.
HloPrintOptions PrintOptions() {
  auto opts = HloPrintOptions::ShortParsable();
  opts.set_print_large_constants(false);
  opts.set_print_control_dependencies(true);
  opts.set_print_operand_index_annotation_interval(5);
  opts.set_print_backend_config(true);
  opts.set_print_metadata(true);
  opts.set_print_name_after_closing_brace(true);
  return opts;
}

// Sortable struct representing a frame in the Python stacktrace attached to a
// given instruction.
struct StackFrame {
  absl::string_view file_name, function_name, op_name;
  int line, column;

 private:
  auto tied() const {
    return std::tie(file_name, line, column, function_name, op_name);
  }
  friend bool operator==(StackFrame const& lhs, StackFrame const& rhs) {
    return lhs.tied() == rhs.tied();
  }
  friend bool operator<(StackFrame const& lhs, StackFrame const& rhs) {
    return lhs.tied() < rhs.tied();
  }
};

// Walk through the HLO graph from an instruction and collect the source
// file/line information we see along the way. This allows us to generate an
// annotation for each instruction that shows the (merged) Python stacktraces of
// the operations that are represented by this instruction. For example:
//
// - /opt/jax/examples/mnist_vae.py:143[<module>]
// -- /opt/jax/examples/mnist_vae.py:127[run_epoch]
// --- /opt/jax/examples/mnist_vae.py:125[body_fun]
// ---- /opt/jax/examples/mnist_vae.py:124[<lambda>]
// ----- /opt/jax/examples/mnist_vae.py:122[body_fun] transpose[permutation=(1,
// 0)]
// --- /opt/jax/examples/mnist_vae.py:126[body_fun] add
// --- /opt/jax/examples/mnist_vae.py:126[body_fun] mul
// --- /opt/jax/examples/mnist_vae.py:126[body_fun] sub
//
// shows four merged stacktraces (3 of depth 3, 1 of depth 5).
class SourceLocationVisitor : public ConstDfsHloVisitorWithDefault {
 public:
  explicit SourceLocationVisitor(
      absl::string_view op_name_prefix_to_remove__ = {})
      : op_name_prefix_to_remove_{op_name_prefix_to_remove__} {}

  std::string AsString(int32_t common_prefix) const {
    // Format the call stacks we've collected; if call stack collection was not
    // enabled then each "stack" just has depth 1 and no column/function name
    // information. Skip the first `common_prefix` elements of each stack trace
    if (common_prefix < 0) {
      return "[invalid common_prefix]";
    }
    std::ostringstream oss{};
    oss << '\n';
    std::vector<StackFrame> current_state{};
    for (auto const& call_stack : location_set_) {
      for (auto depth = 0; depth < call_stack.size() - common_prefix; ++depth) {
        auto const& frame = call_stack[common_prefix + depth];
        if (depth < current_state.size() && current_state[depth] == frame) {
          continue;
        }
        current_state.resize(depth + 1);
        current_state[depth] = frame;
        FormatFrame(oss, frame, depth);
      }
    }
    return std::move(oss).str();
  }

  absl::Status DefaultAction(HloInstruction const* inst) final {
    OpMetadata const& meta = inst->metadata();
    // The full op_name is split across three places: the module-level
    // annotation shows the prefix that is common to the whole module, the
    // instruction-level annotation removes that prefix and shows whatever
    // middle sections of the name are common to all operations in the
    // instruction, and the individual call stack frames in the
    // instruction-level annotation show the final parts of the op_name that
    // have not already been shown.
    absl::string_view op_name = meta.op_name();
    if (!op_name.empty()) {
      op_name = op_name.substr(op_name_prefix_to_remove_.size());
    }
    if (!op_name.empty() && op_name.front() == '/') {
      op_name = op_name.substr(1);
    }
    if (StackFrameId frame_id{meta.stack_frame_id()}; frame_id.valid()) {
      std::vector<StackFrame> call_stack{};
      HloModule const* const hlo_module = inst->parent()->parent();
      while (frame_id.valid()) {
        HloModule::StackFrame frame = hlo_module->get_stack_frame(frame_id);
        if (frame.empty()) {
          break;
        }
        frame_id = frame.parent_frame_id;
        call_stack.emplace_back(StackFrame{frame.file_name, frame.function_name,
                                           op_name, frame.line, frame.column});
        // only attach the op_name to the most-nested frame
        op_name = {};
      }
      // re-order to be [caller, callee, ...]
      std::reverse(call_stack.begin(), call_stack.end());
      location_set_.emplace(call_stack);
    } else if (!meta.source_file().empty() && meta.source_line() != 0) {
      location_set_.emplace(1, StackFrame{meta.source_file(),
                                          {/* function_name */},
                                          op_name,
                                          meta.source_line()});
    }
    return absl::OkStatus();
  }

  std::pair<StringHandle, int32_t> LongestSourceLocationPrefix() const {
    // Find the longest common prefix along the members of location_set_ and
    // return a formatted version of that prefix, along with its length. As
    // location_set_ is sorted, that just means looking for the longest common
    // prefix of the first and last elements.
    if (location_set_.size() < 2) {
      // Only extract a prefix if there are enough stack traces.
      return {};
    }
    const auto& first_loc = *location_set_.begin();
    const auto common_end = std::mismatch(first_loc.begin(), first_loc.end(),
                                          location_set_.rbegin()->begin(),
                                          location_set_.rbegin()->end())
                                .first;
    std::ostringstream oss{};
    oss << '\n';
    std::for_each(first_loc.begin(), common_end,
                  [&oss](const StackFrame& frame) { FormatFrame(oss, frame); });
    const int32_t prefix_frames = std::distance(first_loc.begin(), common_end);
    return {RegisterString(std::move(oss).str()), prefix_frames};
  }

 private:
  static void FormatFrame(std::ostringstream& oss, const StackFrame& frame,
                          int depth = -1) {
    if (depth >= 0) {
      oss << std::string(depth + 1, '-') << ' ';
    }
    PrintEscaped(oss, frame.file_name) << ':' << frame.line;
    if (frame.column) {
      oss << ':' << frame.column;
    }
    if (!frame.function_name.empty()) {
      PrintEscaped(oss << '[', frame.function_name) << ']';
    }
    if (!frame.op_name.empty()) {
      PrintEscaped(oss << ' ', frame.op_name);
    }
    oss << '\n';
  }
  absl::string_view op_name_prefix_to_remove_{};
  std::set<std::vector<StackFrame>> location_set_{};
};

std::string MakeTitle(const HloModule& mod, absl::string_view longest_prefix) {
  if (longest_prefix.empty()) {
    return absl::StrCat("XlaModule:#hlo_module=", mod.name(),
                        ",program_id=", mod.unique_id(), "#");
  }
  return absl::StrCat("XlaModule:#prefix=", longest_prefix,
                      ",hlo_module=", mod.name(),
                      ",program_id=", mod.unique_id(), "#");
}

std::string FormatSourceLocations(HloInstruction const& inst,
                                  int32_t common_frames) {
  // Inside the source location/backtrace report the op_name too, but remove the
  // instruction-wide prefix for brevity
  SourceLocationVisitor visitor{GetLongestOpNamePrefix(inst)};
  // Visit the given instruction, and the things it calls, but not its operands
  // -- we don't want to collect the source code locations that produced the
  // inputs to this instruction, just those corresponding to the instruction
  // itself.
  if (!VisitInstAndCalledButNotOperands(visitor, inst).ok()) {
    return "[error]";
  }
  return visitor.AsString(common_frames);
}

std::pair<std::string, int32_t> ResolveSourceLocation(
    const HloInstruction& inst) {
  const OpMetadata& metadata = inst.metadata();
  if (!metadata.source_file().empty()) {
    return {metadata.source_file(), metadata.source_line()};
  }

  StackFrameId frame_id{metadata.stack_frame_id()};
  if (!frame_id.valid() || inst.GetModule() == nullptr) {
    return {};
  }
  HloModule::StackFrame frame = inst.GetModule()->get_stack_frame(frame_id);
  if (frame.empty()) {
    return {};
  }
  return {std::string(frame.file_name), frame.line};
}

std::string GetFrontendAttribute(const HloInstruction& inst,
                                 absl::string_view key) {
  if (std::optional<std::string> value = inst.get_frontend_attribute(key)) {
    return std::move(*value);
  }
  return {};
}

// Get the string representation of this instruction as an std::string.
std::string InstructionAsString(HloInstruction const& inst) {
  StringPrinter printer;
  inst.Print(&printer, PrintOptions());
  return std::move(printer).ToString();
}

// Get the string representation of the HLO code called by this instruction,
// but not the instruction itself. The typical example is a fusion instruction,
// where InstructionAsString(fusion_inst) would be something like
//   fusion.N = ... fusion(...), calls=fused_computation.N ...
// and CalledInstructionsAsString(fusion_inst) would be something like
//   fused_computation.N { ... }
std::string CalledInstructionsAsString(HloInstruction const& inst) {
  StringPrinter printer;
  auto const opts = PrintOptions();
  for (HloComputation const* called : inst.called_computations()) {
    called->Print(&printer, opts);
  }
  return std::move(printer).ToString();
}

// Get a string representing the longest common prefix of source locations in
// this module, and the number of frames that that represents.
std::pair<StringHandle, int32_t> GetLongestSourceLocationPrefix(
    const HloModule& mod) {
  // In the presence of (at least) debug callbacks, calling Accept on the root
  // instruction of the module may not reach all instructions in the module.
  SourceLocationVisitor visitor{};
  for (const HloComputation* computation : mod.computations()) {
    for (const HloInstruction* inst : computation->instructions()) {
      if (!visitor.DefaultAction(inst).ok()) {
        return {};
      }
    }
  }
  return visitor.LongestSourceLocationPrefix();
}

struct InstructionAnnotationMetadata {
  // Basic annotation payload fields are prepared directly from the HLO
  // instruction and do not need an intermediate representation here.

  // Detailed annotation payload.
  std::string hlo_op_name;
  int64_t hlo_op_id = -1;
  std::string op_type;
  std::string op_name;
  std::string source_file;
  int32_t source_line = 0;
  std::string output_shape;

  // Collective annotation payload.
  bool is_collective = false;
  std::string replica_groups;
  bool is_pipelined = false;
  bool is_spmd_generated = false;
  std::string collective_group_key;
  std::string combiner_key;
  std::string scheduling_group_id;
  std::string stream_annotation;
};

InstructionAnnotationMetadata GetInstructionAnnotationMetadata(
    const HloInstruction& inst) {
  InstructionAnnotationMetadata metadata;
  metadata.hlo_op_name = inst.name();
  metadata.hlo_op_id = inst.unique_id();
  metadata.op_type = inst.metadata().op_type();
  metadata.op_name = inst.metadata().op_name();
  if (metadata.op_name.empty()) {
    metadata.op_name = GetLongestOpNamePrefix(inst);
  }
  std::tie(metadata.source_file, metadata.source_line) =
      ResolveSourceLocation(inst);
  metadata.output_shape = inst.shape().ToString();

  const auto* collective = DynCast<HloCollectiveInstruction>(&inst);
  if (collective == nullptr) {
    return metadata;
  }

  metadata.is_collective = true;
  metadata.replica_groups = collective->device_list()->ToString();
  if (auto backend_config = collective->backend_config<GpuBackendConfig>();
      backend_config.ok()) {
    const auto& collective_config = backend_config->collective_backend_config();
    metadata.is_pipelined = collective_config.is_pipelined();
    metadata.is_spmd_generated = collective_config.is_spmd_generated();
  }

  metadata.collective_group_key =
      GetFrontendAttribute(inst, kCollectiveGroupKeyAttr);
  metadata.combiner_key = GetFrontendAttribute(inst, kCombinerKeyAttr);
  metadata.scheduling_group_id =
      GetFrontendAttribute(inst, kXlaSchedulingGroupIdAttr);
  metadata.stream_annotation =
      GetFrontendAttribute(inst, kXlaStreamAnnotationAttr);

  return metadata;
}
}  // namespace

ModuleAnnotation::ModuleAnnotation(absl::string_view module_name_)
    : title_str_(absl::StrCat("XlaModule:#hlo_module=", module_name_, "#")),
      title_(RegisterString(title_str_)),
      module_name_(RegisterString(std::string{module_name_})),
      common_src_locations_(nullptr),
      module_id_(-1),
      common_stack_frames_(0) {}

ModuleAnnotation::ModuleAnnotation(const HloModule& mod)
    : longest_prefix_(GetLongestOpNamePrefix(mod)),
      title_str_(MakeTitle(mod, longest_prefix_)),
      title_(RegisterString(title_str_)),
      module_name_(RegisterString(mod.name())),
      common_src_locations_(nullptr),
      module_id_(mod.unique_id()),
      common_stack_frames_(0) {
  std::tie(common_src_locations_, common_stack_frames_) =
      GetLongestSourceLocationPrefix(mod);
}

#if GOOGLE_CUDA
static nvtxPayloadSchemaEntry_t SchemaEntry(uint64_t type, const char* name,
                                            uint64_t offset) {
  nvtxPayloadSchemaEntry_t r{};
  r.type = type;
  r.name = name;
  r.offset = offset;
  return r;
}

static uint64_t RegisterStaticSchema(
    const char* name, absl::Span<const nvtxPayloadSchemaEntry_t> entries,
    size_t payload_size) {
  auto domain = tsl::profiler::DefaultProfilerDomain();
  if (!domain) {
    return 0;
  }
  const nvtxPayloadSchemaAttr_t schema_attr = {
#if defined(NVTX_PAYLOAD_SCHEMA_ATTR_NAME)
      /* .fieldMask = */ NVTX_PAYLOAD_SCHEMA_ATTR_NAME |
          NVTX_PAYLOAD_SCHEMA_ATTR_TYPE | NVTX_PAYLOAD_SCHEMA_ATTR_ENTRIES |
          NVTX_PAYLOAD_SCHEMA_ATTR_NUM_ENTRIES |
          NVTX_PAYLOAD_SCHEMA_ATTR_STATIC_SIZE,
#elif defined(NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NAME)
      /* .fieldMask = */ NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NAME |
          NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_TYPE |
          NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_ENTRIES |
          NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NUM_ENTRIES |
          NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_STATIC_SIZE,
#else
#error Unknown NVTX variant.
#endif
      /* .name = */ name,
      /* .type = */ NVTX_PAYLOAD_SCHEMA_TYPE_STATIC,
      /* .flags = */ NVTX_PAYLOAD_SCHEMA_FLAG_NONE,
      /* .entries = */ entries.data(),
      /* .numEntries = */ entries.size(),
      /* .payloadStaticSize = */ payload_size};
  const uint64_t schema_id = RegisterSchema(domain, &schema_attr);
  VLOG(1) << "Registered structured NVTX schema: name=" << name
          << " id=" << schema_id << " payload_size=" << payload_size;
  return schema_id;
}
#endif

uint64_t ModuleAnnotation::NvtxSchemaId() {
  static std::uint64_t schema_id = []() -> std::uint64_t {
#if GOOGLE_CUDA
    auto domain = tsl::profiler::DefaultProfilerDomain();
    if (!domain) {
      return 0;
    }
    const std::array<nvtxPayloadSchemaEntry_t, 3> schema = {
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "Name", offsetof(ModuleAnnotation, module_name_)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_INT32, "Unique ID",
                    offsetof(ModuleAnnotation, module_id_)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "Common source locations",
                    offsetof(ModuleAnnotation, common_src_locations_))};
    const nvtxPayloadSchemaAttr_t schemaAttr = {
#if defined(NVTX_PAYLOAD_SCHEMA_ATTR_NAME)
        /* .fieldMask = */ NVTX_PAYLOAD_SCHEMA_ATTR_NAME |
            NVTX_PAYLOAD_SCHEMA_ATTR_TYPE | NVTX_PAYLOAD_SCHEMA_ATTR_ENTRIES |
            NVTX_PAYLOAD_SCHEMA_ATTR_NUM_ENTRIES |
            NVTX_PAYLOAD_SCHEMA_ATTR_STATIC_SIZE,
#elif defined(NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NAME)
        /* .fieldMask = */ NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NAME |
            NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_TYPE |
            NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_ENTRIES |
            NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NUM_ENTRIES |
            NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_STATIC_SIZE,
#else
#error Unknown NVTX variant.
#endif
        /* .name = */ "XlaModule",
        /* .type = */ NVTX_PAYLOAD_SCHEMA_TYPE_STATIC,
        /* .flags = */ NVTX_PAYLOAD_SCHEMA_FLAG_NONE,
        /* .entries = */ schema.data(),
        /* .numEntries = */ schema.size(),
        /* .payloadStaticSize = */ sizeof(ModuleAnnotation)};
    return RegisterSchema(domain, &schemaAttr);
#else
    return 0;
#endif
  }();
  return schema_id;
}

static std::string MakeInstructionTitle(absl::string_view prefix,
                                        const HloInstruction& inst) {
  // Sometimes an instruction doesn't have metadata, but the computations that
  // it calls do have metadata. Consider all of those metadata op_name entries
  // and attach the longest prefix to this launch.
  absl::string_view op_name = GetLongestOpNamePrefix(inst);

  std::string title;
  if (op_name.empty()) {
    title = absl::StrCat("Thunk:#hlo_op=", inst.name(),
                         ",unique_hlo_op_id=", inst.unique_id());
  } else if (op_name.substr(0, prefix.size()) != prefix) {
    // The op_name for this instruction does not start with the prefix that was
    // common to the other instructions in the module.
    title = absl::StrCat("Thunk:#name=", op_name, ",hlo_op=", inst.name(),
                         ",unique_hlo_op_id=", inst.unique_id());
  } else {
    auto short_name = op_name.substr(prefix.size());
    if (!short_name.empty() && short_name.front() == '/') {
      short_name = short_name.substr(1);
    }
    title = absl::StrCat("Thunk:#name=", short_name, ",hlo_op=", inst.name(),
                         ",unique_hlo_op_id=", inst.unique_id());
  }

  title.push_back('#');
  return title;
}

static std::string MakeInstructionDetails(const HloInstruction& inst) {
  // Collect instruction metadata as a key-value suffix that can be parsed by
  // XProf.
  InstructionAnnotationMetadata metadata =
      GetInstructionAnnotationMetadata(inst);

  std::string details;
  auto append = [&](absl::string_view key, std::string value) {
    if (!value.empty()) {
      value = absl::StrReplaceAll(value, {{",", ";"}});
      absl::StrAppend(&details, ",", key, "=", value);
    }
  };

  append("op_type", metadata.op_type);
  append("op_name", metadata.op_name);
  if (!metadata.source_file.empty()) {
    append("source_file", metadata.source_file);
    append("source_line", absl::StrCat(metadata.source_line));
  }
  append("shape", metadata.output_shape);

  if (metadata.is_collective) {
    append("replica_groups", metadata.replica_groups);
    append("is_pipelined", absl::StrCat(metadata.is_pipelined));
    append("is_spmd_generated", absl::StrCat(metadata.is_spmd_generated));
    append("collective_group_key", metadata.collective_group_key);
    append("combiner_key", metadata.combiner_key);
    append("scheduling_group_id", metadata.scheduling_group_id);
    append("stream_annotation", metadata.stream_annotation);
  }

  return details;
}

static std::string MakeInstructionName(absl::string_view prefix,
                                       const HloInstruction& inst,
                                       TraceAnnotationLevel annotation_level) {
  std::string name = MakeInstructionTitle(prefix, inst);
  if (annotation_level < TraceAnnotationLevel::kDetailed) {
    return name;
  }

  name.pop_back();
  absl::StrAppend(&name, MakeInstructionDetails(inst), "#");
  return name;
}

InstructionAnnotation::InstructionAnnotation(
    const ModuleAnnotation& module_annotation, const HloInstruction& inst,
    TraceAnnotationLevel annotation_level)
    : nvtx_name_str_(MakeInstructionTitle(
          module_annotation.longest_op_name_prefix(), inst)),
      xprof_name_str_(MakeInstructionName(
          module_annotation.longest_op_name_prefix(), inst, annotation_level)),
      nvtx_name_(RegisterString(nvtx_name_str_)) {
  payload_ = Basic{
      RegisterString(InstructionAsString(inst)),
      RegisterString(
          FormatSourceLocations(inst, module_annotation.common_stack_frames())),
      RegisterString("\n" + CalledInstructionsAsString(inst)),
  };
  if (annotation_level < TraceAnnotationLevel::kDetailed) {
    return;
  }

  InstructionAnnotationMetadata metadata =
      GetInstructionAnnotationMetadata(inst);

  payload_ = Detailed{
      std::move(std::get<Basic>(payload_)),
      RegisterOptionalString(metadata.hlo_op_name),
      metadata.hlo_op_id,
      RegisterOptionalString(metadata.op_type),
      RegisterOptionalString(metadata.op_name),
      RegisterOptionalString(metadata.source_file),
      metadata.source_line,
      RegisterOptionalString(metadata.output_shape),
  };
  if (!metadata.is_collective) {
    return;
  }

  payload_ = Collective{
      std::move(std::get<Detailed>(payload_)),
      RegisterOptionalString(metadata.replica_groups),
      static_cast<uint8_t>(metadata.is_pipelined),
      static_cast<uint8_t>(metadata.is_spmd_generated),
      RegisterOptionalString(metadata.collective_group_key),
      RegisterOptionalString(metadata.combiner_key),
      RegisterOptionalString(metadata.scheduling_group_id),
      RegisterOptionalString(metadata.stream_annotation),
  };
}

void RangePush(tsl::profiler::ProfilerDomainHandle domain,
               const ModuleAnnotation& annotation) {
  tsl::profiler::RangePush(domain, annotation.title(), annotation);
}

void RangePush(tsl::profiler::ProfilerDomainHandle domain,
               const InstructionAnnotation& annotation) {
  std::visit(
      [&](const auto& payload) {
        tsl::profiler::RangePush(domain, annotation.nvtx_name_, payload);
      },
      annotation.payload_);
}

ModuleAnnotations::ModuleAnnotations(absl::string_view module_name,
                                     TraceAnnotationLevel /*annotation_level*/)
    : top_level(module_name) {}

uint64_t InstructionAnnotation::Basic::NvtxSchemaId() {
  static std::uint64_t schema_id = []() -> std::uint64_t {
#if GOOGLE_CUDA
    const std::array<nvtxPayloadSchemaEntry_t, 3> schema = {
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "Source locations", offsetof(Basic, src_locations)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "HLO", offsetof(Basic, hlo_dump)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "Called HLO", offsetof(Basic, called_hlo_dump))};
    return RegisterStaticSchema("XlaInstruction", schema, sizeof(Basic));
#else
    return 0;
#endif
  }();
  return schema_id;
}

uint64_t InstructionAnnotation::Detailed::NvtxSchemaId() {
  static std::uint64_t schema_id = []() -> std::uint64_t {
#if GOOGLE_CUDA
    constexpr uint64_t kBasicOffset = offsetof(Detailed, basic);
    const std::array<nvtxPayloadSchemaEntry_t, 10> schema = {
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "Source locations",
                    kBasicOffset + offsetof(Basic, src_locations)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "HLO", kBasicOffset + offsetof(Basic, hlo_dump)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "Called HLO",
                    kBasicOffset + offsetof(Basic, called_hlo_dump)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "HLO name", offsetof(Detailed, hlo_op_name)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_INT64, "HLO unique ID",
                    offsetof(Detailed, hlo_op_id)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "Framework op type", offsetof(Detailed, op_type)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "Framework op name", offsetof(Detailed, op_name)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "Source file", offsetof(Detailed, source_file)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_INT32, "Source line",
                    offsetof(Detailed, source_line)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "Output shape", offsetof(Detailed, output_shape))};
    return RegisterStaticSchema("XlaInstructionDetailed", schema,
                                sizeof(Detailed));
#else
    return 0;
#endif
  }();
  return schema_id;
}

uint64_t InstructionAnnotation::Collective::NvtxSchemaId() {
  static std::uint64_t schema_id = []() -> std::uint64_t {
#if GOOGLE_CUDA
    constexpr uint64_t kDetailedOffset = offsetof(Collective, detailed);
    constexpr uint64_t kBasicOffset =
        kDetailedOffset + offsetof(Detailed, basic);
    const std::array<nvtxPayloadSchemaEntry_t, 17> schema = {
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "Source locations",
                    kBasicOffset + offsetof(Basic, src_locations)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "HLO", kBasicOffset + offsetof(Basic, hlo_dump)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "Called HLO",
                    kBasicOffset + offsetof(Basic, called_hlo_dump)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "HLO name",
                    kDetailedOffset + offsetof(Detailed, hlo_op_name)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_INT64, "HLO unique ID",
                    kDetailedOffset + offsetof(Detailed, hlo_op_id)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "Framework op type",
                    kDetailedOffset + offsetof(Detailed, op_type)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "Framework op name",
                    kDetailedOffset + offsetof(Detailed, op_name)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "Source file",
                    kDetailedOffset + offsetof(Detailed, source_file)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_INT32, "Source line",
                    kDetailedOffset + offsetof(Detailed, source_line)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "Output shape",
                    kDetailedOffset + offsetof(Detailed, output_shape)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "Replica groups", offsetof(Collective, replica_groups)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_UINT8, "Is pipelined",
                    offsetof(Collective, is_pipelined)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_UINT8, "Is SPMD generated",
                    offsetof(Collective, is_spmd_generated)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "Collective group key",
                    offsetof(Collective, collective_group_key)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "Combiner key", offsetof(Collective, combiner_key)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "Scheduling group ID",
                    offsetof(Collective, scheduling_group_id)),
        SchemaEntry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                    "Stream annotation",
                    offsetof(Collective, stream_annotation))};
    return RegisterStaticSchema("XlaCollectiveDetailed", schema,
                                sizeof(Collective));
#else
    return 0;
#endif
  }();
  return schema_id;
}

ModuleAnnotations::ModuleAnnotations(const HloModule& mod,
                                     TraceAnnotationLevel annotation_level)
    : top_level{mod} {
  VLOG(1) << "Preparing GPU trace module annotations: module=" << mod.name()
          << " annotation_level=" << static_cast<int32_t>(annotation_level);

  // Loop through `mod` and populate `instructions` with the information we
  // want to attach to individual instruction ranges.
  for (const HloComputation* computation : mod.computations()) {
    for (const HloInstruction* inst : computation->instructions()) {
      // e.g. inst.name is "fusion.6", inst.opcode is "kFusion" and called
      // is ["fused_computation.5"], in which case the content of
      // "fused_computation.5" ends up under a profiler range called
      // "fusion.6". We want to construct a useful annotation for that range
      // based on the content of `inst`, including `called` etc.
      // FIXME: using try_emplace here was sensitive to
      // https://github.com/abseil/abseil-cpp/issues/388.
      instructions.insert(
          {inst->name(),
           InstructionAnnotation{top_level, *inst, annotation_level}});
    }
  }
}

//===----------------------------------------------------------------------===//
// Scoped RAII helper to set and restore thread local module annotations
//===----------------------------------------------------------------------===//

namespace {
thread_local const ModuleAnnotations* current_annotations = nullptr;
}  // namespace

ScopedModuleAnnotations::ScopedModuleAnnotations(
    const ModuleAnnotations* annotations)
    : restore_(std::exchange(current_annotations, annotations)) {}

ScopedModuleAnnotations::~ScopedModuleAnnotations() {
  current_annotations = restore_;
}

std::optional<ScopedAnnotation> GetInstructionAnnotation(
    absl::string_view profile_annotation) {
  if (profile_annotation.empty()) {
    return {};
  }
  if (current_annotations) {
    // Have a set of pre-prepared instruction annotations to use
    const auto iter =
        current_annotations->instructions.find(profile_annotation);
    if (iter != current_annotations->instructions.end()) {
      // Have a pre-prepared annotation, use it
      return std::optional<ScopedAnnotation>{
          std::in_place, [&] { return iter->second.xprof_name(); },
          [&]() -> const InstructionAnnotation& { return iter->second; }};
    }
  }
  return std::optional<ScopedAnnotation>{
      [&] { return absl::StrFormat("Thunk:#hlo_op=%s#", profile_annotation); }};
}

}  // namespace xla::gpu
