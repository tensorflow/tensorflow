/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_DUMP_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_DUMP_H_

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/xla.pb.h"

// Consolidated utilities for logging information during compilation, usually
// based on the options specified in the DebugOptions proto.
//
// Most functions here take an HloModule and read the DebugOptions from the
// module's config.

namespace xla {

class BufferAssignment;
class HloExecutionProfile;
class HloSnapshot;

// Get a timestamp which we can use as a filename prefix specific to this
// module.
string TimestampFor(const HloModule& module);

// Create the filename we will use to dump in DumpToFileInDir.
string FilenameFor(const HloModule& module, absl::string_view prefix,
                   absl::string_view suffix);

// Writes the given string to a file in the xla_dump_to directory specified by
// module's DebugOptions.
//
// If module doesn't have an xla_dump_to directory, does nothing.
void DumpToFileInDir(const HloModule& module, absl::string_view file_prefix,
                     absl::string_view file_suffix, absl::string_view contents);

// Like DumpToFileInDir, except if module doesn't have an xla_dump_to directory
// specified, or if that directory is equal to "-", writes to stdout instead.
void DumpToFileInDirOrStdout(const HloModule& module,
                             absl::string_view file_prefix,
                             absl::string_view file_suffix,
                             absl::string_view contents);

// Dumps the given execution options if dumping is enabled. Exactly
// where and in what formats it's dumped is determined by the debug options.
void DumpExecutionOptions(const ExecutionOptions& execution_options,
                          const DebugOptions& debug_options);

// Dumps the given HLO module if dumping is enabled for the module.  Exactly
// where and in what formats it's dumped is determined by the module's config.
//
// If you pass an HloExecutionProfile, note that currently only DOT-based output
// formats (i.e. --xla_dump_as_{dot,html,url}) are able to incorporate it into
// their output.  Other formats will just ignore the profile.
void DumpHloModuleIfEnabled(const HloModule& module, absl::string_view name);
void DumpHloModuleIfEnabled(const HloModule& module,
                            const BufferAssignment& buffer_assn,
                            absl::string_view name);
void DumpHloModuleIfEnabled(const HloModule& module,
                            const HloExecutionProfile& profile,
                            absl::string_view name);

// Dumps the given HLO module after running one HLO pass and before running
// another, if that's enabled.
void DumpHloModuleBetweenPassesIfEnabled(absl::string_view pipeline_name,
                                         absl::string_view before_pass_name,
                                         absl::string_view after_pass_name,
                                         const HloModule& module);

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

// Returns true if we should dump data for an HloModule.  This is useful if you
// want to check if DumpToFileInDir{,OrStdout} will do anything before
// generating an expensive string.
bool DumpingEnabledForHloModule(absl::string_view hlo_module_name,
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

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_DUMP_H_
