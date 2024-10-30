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

#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CommandLine.h"
#include "tensorflow/lite/experimental/litert/tools/apply_plugin.h"

using ::litert::tools::ApplyPlugin;
using ::litert::tools::ApplyPluginRun;

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
        "third_party/tensorflow/lite/experimental/litert/vendors/examples"}));

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> out(
    "o",
    llvm::cl::desc("Path to file for output, \"-\" indicates standard out."),
    llvm::cl::init("-"));

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> err(
    "err",
    llvm::cl::desc("Path to file for error output, \"-\" indicates stdandard "
                   "error and \"none\" indicates silent."),
    llvm::cl::init("-"));

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> serialization(
    "serialization", llvm::cl::desc("Serialization strategy to use."),
    llvm::cl::init("METADATA"));

ApplyPluginRun::Ptr ParseFlags() {
  auto res = std::make_unique<ApplyPluginRun>();

  std::ofstream file_out;
  if (out != "-") {
    file_out.open(out);
    res->outs.clear();
    res->outs.push_back(file_out);
  }

  std::ofstream file_err;
  if (err != "-") {
    file_err.open(err);
    res->dump_out.emplace(file_err);
  }

  if (!model.empty()) {
    res->model = model;
  }

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
  }

  return res;
}

int main(int argc, char* argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  auto run = ParseFlags();
  if (run == nullptr) {
    return 1;
  }

  return ApplyPlugin(std::move(run));
}
