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

#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_detail.h"
#include "tensorflow/lite/experimental/litert/core/byte_code_util.h"
#include "tensorflow/lite/experimental/litert/tools/outstream.h"

namespace litert::tools {

using ::litert::internal::Serialization;

// TODO remove these usings other than Ptr and outStraemT

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
    // "serialization": Ignored.
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
    // "serialization": Ignored.
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
    // "serialization": Required.
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
    // "serialization": Ignored.
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
    // "serialization": Ignored.
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

  // Dictates how the final model with compiled assets should be serialized.
  // Only relevant to the "apply" function.
  //
  // [METADATA] Write the compiled module into a metadata buffer using the
  // soc_manufacturer as a key. This is for testing and debugging as it allows
  // the contents of the byte code to be rendered by exisitng flatbuffer
  // tooling. Custom op options will contain only a string identifying the
  // respective entry point.
  //
  // [APPEND] Appends the compiled byte code to the end of the ".tflite" file.
  // Custom options will contain both an entry point name, and an optional
  // metadata lookup key. This facilitates per-op metadata while allowing
  // multiple ops to share the same metadata if needed. Any instances of this
  // metadata are pairs indicating the offset into the file where the byte code
  // starts as well as the size of the byte code.
  Serialization serialization = Serialization::kMetadata;
};

LiteRtStatus ApplyPlugin(ApplyPluginRun::Ptr run);

}  // namespace litert::tools

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TOOLS_APPLY_PLUGIN_H_
