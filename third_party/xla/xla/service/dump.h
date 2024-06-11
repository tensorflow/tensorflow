/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_DUMP_H_
#define XLA_SERVICE_DUMP_H_

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/xla.pb.h"

// Consolidated utilities for logging information during compilation, usually
// based on the options specified in the DebugOptions proto.
//
// Most functions here take an HloModule and read the DebugOptions from the
// module's config.

namespace xla {

// Argument used when calling DumpHloModuleIfEnabled before optimizations are
// performed on an HloModule.
constexpr char kBeforeOptimizationsDumpName[] = "before_optimizations";
constexpr char kAfterOptimizationsDumpName[] = "after_optimizations";

class BufferAssignment;
class HloSnapshot;

// Get a timestamp which we can use as a filename prefix specific to this
// module.
std::string TimestampFor(const HloModule& module);

// Create the filename we will use to dump in DumpToFileInDir.
std::string FilenameFor(int unique_id, absl::string_view module_name,
                        absl::string_view prefix, absl::string_view suffix);
std::string FilenameFor(const HloModule& module, absl::string_view prefix,
                        absl::string_view suffix);

// Writes the given string to a file in the xla_dump_to directory specified by
// module's DebugOptions.
//
// If module doesn't have an xla_dump_to directory, does nothing.
void DumpToFileInDir(const HloModule& module, absl::string_view file_prefix,
                     absl::string_view file_suffix, absl::string_view contents);
void DumpToFileInDir(const DebugOptions& debug_options,
                     absl::string_view filename, absl::string_view contents);

// Like DumpToFileInDir, except if module doesn't have an xla_dump_to directory
// specified, or if that directory is equal to "-", writes to stdout instead.
void DumpToFileInDirOrStdout(const HloModule& module,
                             absl::string_view file_prefix,
                             absl::string_view file_suffix,
                             absl::string_view contents);

// Like DumpToFileInDir, except if debug_options doesn't have an xla_dump_to
// directory specified, or if that directory is equal to "-", writes to stdout
// instead.
void DumpToFileInDirOrStdout(const DebugOptions& debug_options, int unique_id,
                             absl::string_view module_name,
                             absl::string_view file_prefix,
                             absl::string_view file_suffix,
                             absl::string_view contents);

// Writes the given op to a file in the xla_dump_to directory specified by
// module's DebugOptions. Sets the op's source locations to that file.
//
// If module doesn't have an xla_dump_to directory, does nothing.
void DumpToFileInDirOrStdout(const HloModule& module,
                             absl::string_view file_prefix,
                             mlir::Operation* op);

// Dumps the given protobuf to the given filename if dumping is enabled.
// Exactly where and in what formats it's dumped is determined by the debug
// options. Allows for an optional custom serialization function to be used for
// added customization.
void DumpProtobufToFile(const tsl::protobuf::Message& proto,
                        const DebugOptions& debug_options,
                        absl::string_view filename,
                        absl::AnyInvocable<absl::StatusOr<std::string>(
                            tsl::Env*, const tsl::protobuf::Message&)>
                            text_formatter = nullptr);

// Render graph in a given format.
std::string RenderGraph(absl::string_view label, const HloModule& module,
                        RenderedGraphFormat format,
                        bool show_fusion_subcomputations = true);

// Similar to above, but the filename depends on module's information and the
// given name. Also allows for the optional serialization function.
void DumpPerModuleProtobufToFile(const HloModule& module,
                                 const tsl::protobuf::Message& proto,
                                 const DebugOptions& debug_options,
                                 absl::string_view name,
                                 absl::AnyInvocable<absl::StatusOr<std::string>(
                                     tsl::Env*, const tsl::protobuf::Message&)>
                                     text_formatter = nullptr);

// Dumps the given HLO module if dumping is enabled for the module. Exactly
// where and in what formats it's dumped is determined by the module's config.
void DumpHloModuleIfEnabled(const HloModule& module, absl::string_view name);
void DumpHloModuleIfEnabled(const HloModule& module,
                            const BufferAssignment& buffer_assn,
                            absl::string_view name);

// Dumps the given HLO module after running one HLO pass and before running
// another, if that's enabled. Returns the full file paths of all dumps of the
// module, or an empty vector if nothing was dumped.
std::vector<std::string> DumpHloModuleBetweenPassesIfEnabled(
    absl::string_view pipeline_name, absl::string_view before_pass_name,
    absl::string_view after_pass_name, const HloModule& module);

// Dumps the given HLO module during the given HLO pass, if that's enabled.
//
// "step" is a human-readable description of where we are in the middle of this
// pass.  For example, "before-assigning-layouts".
void DumpHloModuleDuringPassIfEnabled(absl::string_view pass_name,
                                      absl::string_view step,
                                      const HloModule& module);

// Dumps the given HloSnapshot to the module's xla_dump_dir, if this is enabled.
//
// Prefer the first overload below, as this will give filenames that are
// consistent with the other methods here.  The second overload (which doesn't
// take an HloModule) is useful in the cases when you're dumping an HloSnapshot
// and simply don't have an HloModule.
void DumpHloSnapshotIfEnabled(const HloModule& module,
                              const HloSnapshot& snapshot);
void DumpHloSnapshotIfEnabled(const HloSnapshot& snapshot,
                              const DebugOptions& opts);

void DumpHloModuleMetadataIfEnabled(const std::vector<HloModule*>& modules);

// Returns true if we should dump data for an HloModule.  This is useful if you
// want to check if DumpToFileInDir{,OrStdout} will do anything before
// generating an expensive string.
bool DumpingEnabledForHloModule(absl::string_view hlo_module_name,
                                const DebugOptions& opts);

// Returns true if we should dump data for an HLO pass
bool DumpingEnabledForHloPass(absl::string_view hlo_pass_name,
                              const DebugOptions& opts);

inline bool DumpingEnabledForHloModule(const HloModule& module) {
  return DumpingEnabledForHloModule(module.name(),
                                    module.config().debug_options());
}

// Returns true if DumpToFileInDirOrStdout and DumpHloModuleIfEnabled will write
// to stdout, rather than to a file on disk.
//
// This is useful if you want to do something different when writing to stdout.
// For example, maybe you have (almost-)duplicate data that you wouldn't mind
// writing to two files, but you don't want to print twice.
bool DumpingToStdout(const DebugOptions& opts);

}  // namespace xla

#endif  // XLA_SERVICE_DUMP_H_
