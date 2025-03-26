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
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either expruns or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CommandLine.h"
#include "tensorflow/lite/experimental/litert/compiler/plugin/compiler_flags.h"
#include "tensorflow/lite/experimental/litert/tools/apply_plugin.h"
#include "tensorflow/lite/experimental/litert/tools/outstream.h"

using ::litert::tools::ApplyPlugin;
using ::litert::tools::ApplyPluginRun;
using ::litert::tools::UserStream;

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> cmd(
    llvm::cl::Positional,
    llvm::cl::desc("Routine to run (apply, partition, compile, info, noop)."),
    llvm::cl::init("partition"));

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> model(
    "model", llvm::cl::desc("Path to flatbuffer file."), llvm::cl::init(""));

// TODO: b/366821557 - Support path to pre-compiled plugin in flags.
// NOLINTNEXTLINE
static llvm::cl::opt<std::string> soc_manufacturer(
    "soc_man",
    llvm::cl::desc("String identifier of SoC manufacturer (e.g., GoogleTensor, "
                   "Qualcomm)."),
    llvm::cl::init("ExampleSocManufacturer"));

// TODO: Support multi target compilation.
// NOLINTNEXTLINE
static llvm::cl::opt<std::string> soc_model("soc_model",
                                            llvm::cl::desc("Target SoC model."),
                                            llvm::cl::init("ExampleSocModel"));

// NOLINTNEXTLINE
static llvm::cl::list<std::string> libs(
    "libs",
    llvm::cl::desc("List of directories in which to search for suitable "
                   "compiler plugin shared libraries."),
    llvm::cl::list_init(llvm::ArrayRef<std::string>{
        "third_party/tensorflow/lite/experimental/litert/vendors/examples",
        "third_party/tensorflow/lite/experimental/litert/vendors/qualcomm/"
        "compiler",
        "third_party/tensorflow/lite/experimental/litert/vendors/mediatek/"
        "compiler",
        "third_party/tensorflow/lite/experimental/litert/vendors/"
        "google_tensor/compiler"}));

// NOLINTNEXTLINE
static llvm::cl::list<std::string> outs(
    "o",
    llvm::cl::desc("Path to files for output, \"-\" indicates standard out, "
                   "\"--\" for standard err, \"none\" for null stream."),
    llvm::cl::list_init(llvm::ArrayRef<std::string>{"-"}));

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> err(
    "err",
    llvm::cl::desc("Path to file for err output, \"-\" indicates standard out, "
                   "\"--\" for standard err, \"none\" for null stream."),
    llvm::cl::init("--"));

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> compiler_flags(
    "compiler-flags",
    llvm::cl::desc("List of comma separated (no space) compiler flags. Flags "
                   "may be key-value pairs "
                   "in the format of \"key=value\", or just \"key\". E.g. "
                   "\"--compiler-flags=key1=value1,key2\""));

// NOLINTNEXTLINE
static llvm::cl::list<uint32_t> subgraphs(
    "subgraphs",
    llvm::cl::desc("If provides, only the subgraphs with the given indices "
                   "are applied with the plugin."),
    llvm::cl::list_init(llvm::ArrayRef<uint32_t>{}));

ApplyPluginRun::Ptr ParseFlags() {
  auto res = std::make_unique<ApplyPluginRun>();

  if (!model.empty()) {
    res->model = model;
  }

  res->compiler_flags = *litert::internal::ParseCompilerFlags(compiler_flags);

  res->soc_manufacturer = soc_manufacturer;
  res->soc_models.push_back(soc_model);

  res->lib_search_paths.assign(libs.begin(), libs.end());

  if (cmd == "apply") {
    res->cmd = ApplyPluginRun::Cmd::APPLY;
  } else if (cmd == "partition") {
    res->cmd = ApplyPluginRun::Cmd::PARTITION;
  } else if (cmd == "compile") {
    res->cmd = ApplyPluginRun::Cmd::COMPILE;
  } else if (cmd == "info") {
    res->cmd = ApplyPluginRun::Cmd::INFO;
  } else if (cmd == "noop") {
    res->cmd = ApplyPluginRun::Cmd::NOOP;
  } else {
    return nullptr;
  }

  for (auto subgraph_idx : subgraphs) {
    res->subgraphs.insert(subgraph_idx);
  }

  return res;
}

int main(int argc, char* argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  auto run = ParseFlags();
  if (run == nullptr) {
    return 1;
  }

  run->outs.clear();
  std::vector<std::unique_ptr<litert::tools::UserStream>> oss;
  for (const auto& out : outs) {
    oss.push_back(std::make_unique<litert::tools::UserStream>(
        UserStream::MakeFromFlag(out)));
    run->outs.push_back(oss.back()->Get());
  }

  run->dump_out = UserStream::MakeFromFlag(err);

  run->dump_out.Get() << absl::StreamFormat(
      "CMD: %s\nMODEL: %s\nSOC_MANUFACTURER: %s\nSOC_MODEL: %s\n", cmd, model,
      soc_manufacturer, soc_model);

  return ApplyPlugin(std::move(run));
}
