// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TOOLS_APPLY_PLUGIN_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TOOLS_APPLY_PLUGIN_H_

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/compiler/plugin/compiler_flags.h"
#include "tensorflow/lite/experimental/litert/tools/outstream.h"

namespace litert::tools {

using ::litert::internal::CompilerFlags;

struct ApplyPluginRun {
  // NOTE: All StrFlagT are expected to have static storage duration.
  using Ptr = std::unique_ptr<ApplyPluginRun>;

  // A specific command implemented by the tool to run.
  enum class Cmd {
    // Displays info about all plugins found in given search paths.
    //
    // FLAG SEMANTICS:
    // "lib_search_paths": Required, at least one.
    // "model": Ignored.
    // "soc_manufacturer": Optional, filters plugins to display.
    // "soc_models": Ignored.
    // "outs": Required, must be size one.
    // "dump_out": Optional.
    INFO,

    // Does nothing and simply de-serializes and re-serializes the given model.
    // This is intended for testing and internal debugging only.
    //
    // FLAG SEMANTICS:
    // "lib_search_paths": Ignored.
    // "model": Required.
    // "soc_manufacturer": Ignored.
    // "soc_models": Ignored.
    // "outs": Required, must be size one.
    // "dump_out": Optional.
    NOOP,

    // Runs the entire end to end flow. This is the standard compiler plugin
    // usage. A seperate compilation step will occur for each sco_model tag that
    // is supported by the loaded plugin, and a new output model will be
    // generated for each. Partitioning is invariant accross different soc_model
    // targets from the same manufacturer, so only one compilation step will
    // occur even if multiple targest are requested.
    //
    // FLAG SEMANTICS:
    // "lib_search_paths": Required, at least one.
    // "model": Required.
    // "soc_manufacturer": Required.
    // "soc_models": Required, at least one.
    // "outs": Required, must be size equal to "soc_models".
    // "dump_out": Optional.
    //
    // TODO: Support multi target compilation.
    APPLY,

    // Only run the partiion step and skip compilation. Writes a ".tflite" model
    // to "out" where selected partitions are manifested as new standard
    // flatbuffer subgraphs added to the input model.
    // The partitions original locations are replaced with a single custom op
    // the contains an identifier to the corresponding partition (new subgraph).
    // This is intended for testing and development.
    //
    // FLAG SEMANTICS:
    // "lib_search_paths": Required, at least one.
    // "model": Required.
    // "soc_manufacturer": Required.
    // "soc_models": Ignored.
    // "outs": Required, must be size one.
    // "dump_out": Optional.
    PARTITION,

    // Skip partitioning and run the entire input model through compilation
    // directly. Fails if any ops in the input model are unsupported by the
    // plugin. Writes the raw compiled result to the "out" stream without any
    // wrapping flatbuffer. Runs multi-target compilation as in "APPLY",
    // Intended for testing and development.
    //
    // FLAG SEMANTICS:
    // "lib_search_paths": Required, at least one.
    // "model": Required.
    // "soc_manufacturer": Required.
    // "soc_models": Required, at least one.
    // "out": Required, must be size equal to "soc_models".
    // "dump_out": Optional.
    //
    // TODO: Support multi target compilation.
    COMPILE,
  };

  // A command to run, see above.
  Cmd cmd;

  // Collection of paths on local files system dictating where the tool should
  // look for suitable LiteRtCompilerPlugin shared libraries. The tool will
  // select the first ".so" file found with prefix "libLiteRtPlugin" that has
  // the "soc_manufacturer" tag passed. Providing more than one plugin shared
  // library for the same manufacturer results in an error.
  std::vector<absl::string_view> lib_search_paths = {};

  // Path to ".tflite" model the tool should operated on.
  std::optional<absl::string_view> model = {};

  // A tag representing a manufacturer the tool should target for compilation.
  // This is used to select the appropriate plugin if multiple plugins are found
  // in "lib_search_paths".
  std::optional<absl::string_view> soc_manufacturer = {};

  // Collection of soc models tags the tool should target for compilation.
  std::vector<absl::string_view> soc_models = {};

  // Where the tool should write its result file(s) to. If the command runs
  // compilation, an "out" stream should be passed for each "soc_model" target
  // requested for compilation. Output for the "ith" target will be written to
  // the "ith" outs stream.
  std::vector<OutStream> outs = {std::cout};

  // Where to direct logging for this run. Passing nullopt here indicates
  // "silent" behavior and should only be used when this tool is part of a
  // larger pipeline like an end2end test.
  UserStream dump_out;

  // Compiler flags to pass to the plugin. Only relevant for "APPLY" and
  // "COMPILE" commands.
  CompilerFlags compiler_flags;

  // If provided, only the subgraphs with the given indices are applied with the
  // plugin.
  absl::flat_hash_set<uint32_t> subgraphs = {};
};

LiteRtStatus ApplyPlugin(ApplyPluginRun::Ptr run);

}  // namespace litert::tools

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TOOLS_APPLY_PLUGIN_H_
