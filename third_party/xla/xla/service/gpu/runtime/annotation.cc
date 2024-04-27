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

#include "xla/service/gpu/runtime/annotation.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <optional>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/printer.h"
#include "xla/status.h"
#include "tsl/platform/errors.h"
#include "tsl/profiler/lib/nvtx_utils.h"
#include "tsl/profiler/lib/scoped_annotation.h"

#if GOOGLE_CUDA
#include "nvtx3/nvToolsExt.h"
#include "nvtx3/nvToolsExtPayload.h"
#endif

namespace xla::gpu {

using ::tsl::profiler::ScopedAnnotation;
using ::tsl::profiler::StringHandle;
namespace {

StringHandle RegisterString(const std::string& str) {
  if (auto domain = tsl::profiler::DefaultProfilerDomain(); domain) {
    return tsl::profiler::RegisterString(domain, str);
  }
  return {};
}

// Nsight Systems supports some basic HTML markup in annotation strings. This
// escaping stops things like <module> from disappearing.
std::ostream& PrintEscaped(std::ostream& os, std::string_view str) {
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
  std::string_view file_name, function_name, op_name;
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
// annotation for each kernel that shows the (merged) Python stacktraces of the
// operations that were traced and compiled int this kernel. For example:
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
      std::string_view op_name_prefix_to_remove__ = {})
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

  Status DefaultAction(HloInstruction const* inst) final {
    OpMetadata const& meta = inst->metadata();
    // The full op_name is split across three places: the module-level
    // annotation shows the prefix that is common to the whole module, the
    // kernel-level annotation removes that prefix and shows whatever middle
    // sections of the name are common to all operations in the kernel, and the
    // individual call stack frames in the kernel-level annotation show the
    // final parts of the op_name that have not already been shown.
    std::string_view op_name = meta.op_name();
    if (!op_name.empty()) {
      op_name = op_name.substr(op_name_prefix_to_remove_.size());
    }
    if (!op_name.empty() && op_name.front() == '/') {
      op_name = op_name.substr(1);
    }
    if (int frame_id = meta.stack_frame_id(); frame_id != 0) {
      std::vector<StackFrame> call_stack{};
      HloModule const* const hlo_module = inst->parent()->parent();
      while (frame_id != 0) {
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
    return OkStatus();
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
  std::string_view op_name_prefix_to_remove_{};
  std::set<std::vector<StackFrame>> location_set_{};
};

template <typename Visitor>
absl::Status VisitInstAndCalledButNotOperands(Visitor& visitor,
                                              const HloInstruction& inst) {
  // Visit the given instruction, and the things it calls, but not its operands.
  TF_RETURN_IF_ERROR(visitor.DefaultAction(&inst));
  for (const HloComputation* called : inst.called_computations()) {
    const HloInstruction* const root = called->root_instruction();
    TF_RETURN_IF_ERROR(root->Accept(&visitor, false /* call_finish_visit */,
                                    true /* ignore_control_predecessors */,
                                    true /* cross_computation */));
  }
  return absl::OkStatus();
}

// Split `a` and `b` by `delim` into two lists of possibly-empty tokens, then
// rejoin the first N of those lists that match by `delim`. Note: it is
// unspecified which argument the return value points into.
std::string_view LongestPrefix(std::string_view a, std::string_view b,
                               char delim = '/') {
  auto split_a = absl::StrSplit(a, delim);
  auto split_b = absl::StrSplit(b, delim);

  size_t common_prefix_len = 0;

  for (auto a_it = split_a.begin(), b_it = split_b.begin();
       a_it != split_a.end() && b_it != split_b.end(); ++a_it, ++b_it) {
    if (*a_it != *b_it) break;

    if (common_prefix_len) ++common_prefix_len;  // account for delimiter
    common_prefix_len += a_it->size();           // length of a matching token
  }

  return std::string_view(a.data(), common_prefix_len);
}

// Find the longest prefix among instructions' op_name metadata
// Chunk this by delimiting slashes, i.e. given a/b/cat and a/b/cabbage, the
// longest prefix is a/b not a/b/ca
class OpNamePrefixVisitor : public ConstDfsHloVisitorWithDefault {
 public:
  absl::Status DefaultAction(const HloInstruction* inst) final {
    auto const& op_name = inst->metadata().op_name();
    if (!op_name.empty()) {
      prefix_ = prefix_ ? LongestPrefix(*prefix_, op_name) : op_name;
    }
    return absl::OkStatus();
  }

  std::string_view longest_op_name_prefix() const {
    return prefix_.value_or("");
  }

 private:
  std::optional<std::string_view> prefix_;
};

std::string_view GetLongestOpNamePrefix(const HloModule& mod) {
  // In the presence of (at least) debug callbacks, calling Accept on the root
  // instruction of the module may not reach all instructions in the module.
  OpNamePrefixVisitor visitor{};
  for (const HloComputation* computation : mod.computations()) {
    for (const HloInstruction* inst : computation->instructions()) {
      if (!visitor.DefaultAction(inst).ok()) {
        return {};
      }
    }
  }
  return visitor.longest_op_name_prefix();
}

std::string_view GetLongestOpNamePrefix(const HloInstruction& inst) {
  OpNamePrefixVisitor visitor{};
  if (!VisitInstAndCalledButNotOperands(visitor, inst).ok()) {
    return {};
  }
  return visitor.longest_op_name_prefix();
}

std::string MakeTitle(const HloModule& mod, std::string_view longest_prefix) {
  if (longest_prefix.empty()) {
    return absl::StrFormat("XlaModule:#hlo_module=%s,program_id=%d#",
                           mod.name(), mod.unique_id());
  }
  return absl::StrFormat("XlaModule:#prefix=%s,hlo_module=%s,program_id=%d#",
                         longest_prefix, mod.name(), mod.unique_id());
}

std::string FormatSourceLocations(HloInstruction const& inst,
                                  int32_t common_frames) {
  // Inside the source location/backtrace report the op_name too, but remove the
  // kernel-wide prefix for brevity
  SourceLocationVisitor visitor{GetLongestOpNamePrefix(inst)};
  // Visit the given instruction, and the things it calls, but not its operands
  // -- we don't want to collect the source code locations that produced the
  // inputs to this kernel, just those corresponding to the kernel itself.
  if (!VisitInstAndCalledButNotOperands(visitor, inst).ok()) {
    return "[error]";
  }
  return visitor.AsString(common_frames);
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
}  // namespace

ModuleAnnotation::ModuleAnnotation(std::string_view module_name_)
    : title_str_(absl::StrFormat("XlaModule:#hlo_module=%s#", module_name_)),
      title_(RegisterString(title_str_)),
      module_name_(RegisterString(std::string{module_name_})) {}

ModuleAnnotation::ModuleAnnotation(const HloModule& mod)
    : longest_prefix_(GetLongestOpNamePrefix(mod)),
      title_str_(MakeTitle(mod, longest_prefix_)),
      title_(RegisterString(title_str_)),
      module_name_(RegisterString(mod.name())),
      module_id_(mod.unique_id()) {
  std::tie(common_src_locations_, common_stack_frames_) =
      GetLongestSourceLocationPrefix(mod);
}

#if GOOGLE_CUDA
namespace {
auto schema_entry(uint64_t type, const char* name, uint64_t offset) {
  nvtxPayloadSchemaEntry_t r{};
  r.type = type;
  r.name = name;
  r.offset = offset;
  return r;
}
}  // namespace
#endif

uint64_t ModuleAnnotation::NvtxSchemaId() {
  static std::uint64_t schema_id = []() -> std::uint64_t {
#if GOOGLE_CUDA
    auto domain = tsl::profiler::DefaultProfilerDomain();
    if (!domain) {
      return 0;
    }
    const nvtxPayloadSchemaEntry_t schema[] = {
        schema_entry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                     "Name", offsetof(ModuleAnnotation, module_name_)),
        schema_entry(NVTX_PAYLOAD_ENTRY_TYPE_INT32, "Unique ID",
                     offsetof(ModuleAnnotation, module_id_)),
        schema_entry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                     "Common source locations",
                     offsetof(ModuleAnnotation, common_src_locations_))};
    const nvtxPayloadSchemaAttr_t schemaAttr = {
        /* .fieldMask = */ NVTX_PAYLOAD_SCHEMA_ATTR_NAME |
            NVTX_PAYLOAD_SCHEMA_ATTR_TYPE | NVTX_PAYLOAD_SCHEMA_ATTR_ENTRIES |
            NVTX_PAYLOAD_SCHEMA_ATTR_NUM_ENTRIES |
            NVTX_PAYLOAD_SCHEMA_ATTR_STATIC_SIZE,
        /* .name = */ "XlaModule",
        /* .type = */ NVTX_PAYLOAD_SCHEMA_TYPE_STATIC,
        /* .flags = */ NVTX_PAYLOAD_SCHEMA_FLAG_NONE,
        /* .entries = */ schema,
        /* .numEntries = */ sizeof(schema) / sizeof(schema[0]),
        /* .payloadStaticSize = */ sizeof(ModuleAnnotation)};
    return RegisterSchema(domain, &schemaAttr);
#else
    return 0;
#endif
  }();
  return schema_id;
}

namespace {
std::string MakeKernelName(std::string_view prefix,
                           const HloInstruction& inst) {
  // Sometimes an instruction doesn't have metadata, but the computations that
  // it calls do have metadata. Consider all of those metadata op_name entries
  // and attach the longest prefix to this launch.
  std::string_view op_name = GetLongestOpNamePrefix(inst);
  if (op_name.empty()) {
    return absl::StrFormat("Thunk:#hlo_op=%s#", inst.name());
  } else if (op_name.substr(0, prefix.size()) != prefix) {
    // the op_name we got for this instruction does not start with the prefix
    // that we thought was common to all instructions in the module
    return absl::StrFormat("Thunk:#name=%s,hlo_op=%s#", op_name, inst.name());
  } else {
    // remove the prefix that's in the parent module annotation
    auto short_name = op_name.substr(prefix.size());
    // remove the leading / if there is one (prefix might be an empty string)
    if (!short_name.empty() && short_name.front() == '/') {
      short_name = short_name.substr(1);
    }
    return absl::StrFormat("Thunk:#name=%s,hlo_op=%s#", short_name,
                           inst.name());
  }
}
}  // namespace

KernelAnnotation::KernelAnnotation(const ModuleAnnotation& module_annotation,
                                   const HloInstruction& inst)
    : title_str(
          MakeKernelName(module_annotation.longest_op_name_prefix(), inst)),
      title(RegisterString(title_str)),
      hlo_dump(RegisterString(InstructionAsString(inst))),
      src_locations(RegisterString(FormatSourceLocations(
          inst, module_annotation.common_stack_frames()))),
      called_hlo_dump(RegisterString("\n" + CalledInstructionsAsString(inst))) {
}

ModuleAnnotations::ModuleAnnotations(std::string_view module_name)
    : top_level(module_name) {}

uint64_t KernelAnnotation::NvtxSchemaId() {
  static std::uint64_t schema_id = []() -> std::uint64_t {
#if GOOGLE_CUDA
    auto domain = tsl::profiler::DefaultProfilerDomain();
    if (!domain) {
      return 0;
    }
    const nvtxPayloadSchemaEntry_t schema[] = {
        schema_entry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                     "Source locations",
                     offsetof(KernelAnnotation, src_locations)),
        schema_entry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                     "HLO", offsetof(KernelAnnotation, hlo_dump)),
        schema_entry(NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE,
                     "Called HLO",
                     offsetof(KernelAnnotation, called_hlo_dump))};
    const nvtxPayloadSchemaAttr_t schemaAttr = {
        /* .fieldMask = */ NVTX_PAYLOAD_SCHEMA_ATTR_NAME |
            NVTX_PAYLOAD_SCHEMA_ATTR_TYPE | NVTX_PAYLOAD_SCHEMA_ATTR_ENTRIES |
            NVTX_PAYLOAD_SCHEMA_ATTR_NUM_ENTRIES |
            NVTX_PAYLOAD_SCHEMA_ATTR_STATIC_SIZE,
        /* .name = */ "XlaKernel",
        /* .type = */ NVTX_PAYLOAD_SCHEMA_TYPE_STATIC,
        /* .flags = */ NVTX_PAYLOAD_SCHEMA_FLAG_NONE,
        /* .entries = */ schema,
        /* .numEntries = */ sizeof(schema) / sizeof(schema[0]),
        /* .payloadStaticSize = */ sizeof(KernelAnnotation)};
    return RegisterSchema(domain, &schemaAttr);
#else
    return 0;
#endif
  }();
  return schema_id;
}

ModuleAnnotations::ModuleAnnotations(const HloModule& mod) : top_level{mod} {
  // loop through `mod` and populate `kernels` (string -> KernelAnnotation map)
  // with the information we want to attach to individual kernels.
  for (const HloComputation* computation : mod.computations()) {
    for (const HloInstruction* inst : computation->instructions()) {
      // e.g. inst.name is "fusion.6", inst.opcode is "kFusion" and called
      // is ["fused_computation.5"], in which case the content of
      // "fused_computation.5" ends up under an NVTX range called
      // "fusion.6". We want to construct a useful annotation for that NVTX
      // range based on the content of `inst`, including `called` etc.
      // FIXME: using try_emplace here was sensitive to
      // https://github.com/abseil/abseil-cpp/issues/388.
      kernels.insert({inst->name(), {top_level, *inst}});
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
  std::exchange(current_annotations, restore_);
}

const ModuleAnnotations* GetCurrentModuleAnnotations() {
  return current_annotations;
}

std::optional<ScopedAnnotation> GetKernelAnnotation(
    const ModuleAnnotations* annotations, std::string_view profile_annotation) {
  if (profile_annotation.empty()) {
    return {};
  }
  if (annotations) {
    // Have a set of pre-prepared thunk/kernel annotations to use
    const auto iter = annotations->kernels.find(profile_annotation);
    if (iter != annotations->kernels.end()) {
      // Have a pre-prepared annotation, use it
      return std::optional<ScopedAnnotation>{[&] { return iter->second; }};
    }
  }
  return std::optional<ScopedAnnotation>{
      [&] { return absl::StrFormat("Thunk:#hlo_op=%s#", profile_annotation); }};
}

}  // namespace xla::gpu
