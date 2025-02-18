/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_HLO_MODULE_UTIL_H_
#define XLA_SERVICE_HLO_MODULE_UTIL_H_

#include <functional>
#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/compiler.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"

namespace xla {

// Converts an HloModule from the given hlo textual IR string (in
// HloModule::ToString format).
absl::StatusOr<std::unique_ptr<HloModule>> CreateModuleFromString(
    absl::string_view hlo_string, const DebugOptions& debug_options);

// Creates an HloModule from the given proto.
absl::StatusOr<std::unique_ptr<HloModule>> CreateModuleFromProto(
    const HloModuleProto& proto,
    const DebugOptions& debug_options = DebugOptions::default_instance());

// Create an HLO state from serialized representation. In addition to
// creating the proto with HloModule::CreateFromProto(...) it also
// uses HloVerifier to ensure basic invariants are held.
// The HLO module could be a pre-optimizations (default) or post-optimizations
// module, which affects how the HLO module is verified, e.g., mixed-precision
// is allowed in post-optimizations HLOs.
absl::StatusOr<std::unique_ptr<HloModule>> CreateModuleFromProto(
    const HloModuleProto& proto, const HloModuleConfig& module_config,
    bool is_module_post_optimizations = false);

// Reads the proto file in xla.HloProto format, creates and returns the
// HloModule.
absl::StatusOr<std::unique_ptr<HloModule>> ReadModuleFromBinaryProtoFile(
    absl::string_view filename,
    const DebugOptions& debug_options = DebugOptions::default_instance());

// Reads the proto file in xla.HloModule format, creates and returns the
// HloModule.
absl::StatusOr<std::unique_ptr<HloModule>> ReadModuleFromModuleBinaryProtofile(
    const std::string& filename, const DebugOptions& debug_options);

// Reads the HLO text dump file in HloModule::ToString format, creates and
// returns the HloModule.
absl::StatusOr<std::unique_ptr<HloModule>> ReadModuleFromHloTextFile(
    absl::string_view filename,
    const DebugOptions& debug_options = DebugOptions::default_instance(),
    const HloParserOptions& options = HloParserOptions());

absl::StatusOr<std::unique_ptr<HloModule>> ReadModuleFromTextProtoFile(
    absl::string_view hlo_file);

enum class InputFormat {
  kText,                 // Text format returned by HloModule::ToString().
  kProtoText,            // Protobuf text format of an xla::HloProto message.
  kProtoBinary,          // Protobuf binary format of an xla::HloProto message.
  kSnapshotProtoBinary,  // HloSnapshot protobuf binary format. Can be dumped by
                         // TensorFlow by setting the environment variable
                         // xla_dump_hlo_snapshots.
  kUnoptimizedSnapshotProtoBinary,  // HloUnoptimizedSnapshot protobuf binary
                                    // format. Can be dumped by
                                    // setting the flag
                                    // xla_dump_hlo_snapshots in conjunction
                                    // with xla_dump_as_text.
  kUnoptimizedSnapshotProtoText,    // HloUnoptimizedSnapshot protobuf text
                                    // format. Can be dumped by TensorFlow by
                                    // setting the flag xla_dump_hlo_snapshots
                                    // in conjunction with xla_dump_as_text.
};

bool AbslParseFlag(absl::string_view text, InputFormat* input_format,
                   std::string* error);

std::string AbslUnparseFlag(InputFormat input_format);

struct HloModuleAndArguments {
  std::unique_ptr<HloModule> hlo_module;

  // The outer `std::vector` represents the list of shards. The inner
  // `std::vector<Literal>` represents a list of arguments for a single shard
  // partition.
  std::vector<std::vector<Literal>> arguments;
};

absl::StatusOr<HloModuleAndArguments> LoadHloModuleAndArguments(
    absl::string_view hlo_file, InputFormat input_format);

absl::StatusOr<HloModuleAndArguments> ReadModuleFromSnapshotBinaryProtoFile(
    absl::string_view hlo_file);
absl::StatusOr<HloModuleAndArguments>
ReadModuleFromUnoptimizedSnapshotBinaryProtoFile(absl::string_view hlo_file);
absl::StatusOr<HloModuleAndArguments>
ReadModuleFromUnoptimizedSnapshotTextProtoFile(absl::string_view hlo_file);

// Creates an HloModuleConfig for a given program shape and arguments.
// If execution_options does not set num_replicas, default_num_replicas is used.
// num_threads is optional; if not given, intra_op_parallelism_threads not set.
// aot_options is optional; if not given a default is used.
absl::StatusOr<std::unique_ptr<HloModuleConfig>> CreateModuleConfig(
    const ProgramShape& program_shape,
    absl::Span<const Shape* const> argument_shapes,
    const ExecutionOptions* execution_options, int default_num_replicas,
    std::optional<int> num_threads = std::nullopt,
    const AotCompilationOptions* aot_options = nullptr);

typedef std::function<Shape(const Shape&)> DeviceShapeRepresentationFn;

// Update entry computation's computation layout by translating each shape
// with shape_representation_fn(shape). It can be used for example to add
// tiling info for each shape.
void UpdateEntryComputationLayout(
    HloModule* module, DeviceShapeRepresentationFn shape_representation_fn,
    bool empty_tiles_only = true);
}  // namespace xla

#endif  // XLA_SERVICE_HLO_MODULE_UTIL_H_
