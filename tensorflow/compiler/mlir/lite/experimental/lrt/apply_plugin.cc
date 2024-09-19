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

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "llvm/Support/CommandLine.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_compiler_plugin.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_support.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/core/algo.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/core/lite_rt_model_init.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/core/model.h"

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> model_path(
    "model_path", llvm::cl::desc("Path to flatbuffer file."),
    llvm::cl::init(""));

// TODO: b/366821557 - Support path to pre-compiled plugin in flags.
// NOLINTNEXTLINE
static llvm::cl::opt<std::string> soc_manufacturer(
    "soc_man",
    llvm::cl::desc("String identifier of SoC backend (pixel, qcc, darwinn)."),
    llvm::cl::init("ExampleSocManufacturer"));

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> soc_model(
    "soc_model",
    llvm::cl::desc("Compilation configuration identifier (chip type)."),
    llvm::cl::init("DummyMulOp"));

// TODO swap "dry_run" for optional "don't delete partitioned subgraphs".
// NOLINTNEXTLINE
static llvm::cl::opt<bool> dry_run(
    "dry_run",
    llvm::cl::desc(
        "Only run \"partition\" phase and output the spliced out subgraphs."),
    llvm::cl::init(true));

#define EXIT_IF_NULL(val, msg) \
  if (!val) {                  \
    std::cerr << msg << "\n";  \
    return 1;                  \
  }

void DumpSubgraph(const LrtSubgraphT& subgraph, std::string_view label) {
#ifndef NDEBUG
  std::cerr << "===== " << label << " =====\n";
  for (auto op : subgraph.ops) {
    debug::DumpOp(*op);
  }
  for (auto tensor : subgraph.inputs) {
    std::cerr << "SG_IN " << tensor << "\n";
  }

  for (auto tensor : subgraph.outputs) {
    std::cerr << "SG_OUT " << tensor << "\n";
  }
#endif
}

bool IsSocModelSupported(LrtCompilerPlugin plugin,
                         std::string_view requested_soc_model) {
  const auto num_supported_configs = LrtPluginNumSupportedSocModels(plugin);
  for (int i = 0; i < num_supported_configs; ++i) {
    const char* config;
    LRT_RETURN_VAL_IF_NOT_OK(
        LrtPluginGetSupportedSocModelId(plugin, i, &config), false);
    if (requested_soc_model == config) {
      return true;
    }
  }

  return false;
}

// TODO: b/366821557 - Replace loading pre-compiled plugin.
UniqueLrtCompilerPlugin LoadPlugin() {
  if (soc_manufacturer != LrtPluginSocManufacturer()) {
    std::cerr << "Only ExampleSocManufacturer currently supported";
    return nullptr;
  }

  LrtCompilerPlugin plugin;
  LRT_RETURN_VAL_IF_NOT_OK(LrtPluginInit(&plugin), nullptr);
  auto result = UniqueLrtCompilerPlugin(plugin);

  if (!IsSocModelSupported(result.get(), soc_model)) {
    std::cerr << "Only DummyMulOp currently supported\n";
    return nullptr;
  }

  return result;
}

UniqueLrtModel LoadModel(std::string_view filename) {
  LrtModel model;
  LRT_RETURN_VAL_IF_NOT_OK(LoadModelFromFile(filename.data(), &model), nullptr);
  return UniqueLrtModel(model);
}

LrtStatus ApplyPlugin(LrtModel model, LrtCompilerPlugin plugin) {
  LRT_RETURN_STATUS_IF_NOT_OK(
      RegisterCustomOpCode(model, LrtPluginSocManufacturer()));

  LrtOpListT selected_ops;
  LRT_RETURN_STATUS_IF_NOT_OK(
      LrtPluginPartitionModel(plugin, model, &selected_ops));

  auto partitions =
      algo::DisjointSets::GetPartitionsFromFlatList(selected_ops.ops);

  // TODO: b/366821557 - Support multiple subgraphs in plugin application.
  auto& main_subgraph = model->subgraphs.front();
  DumpSubgraph(main_subgraph, "Main subgraph before partioning.");

  std::vector<LrtSubgraph> slices;
  std::vector<LrtOp> custom_ops;
  slices.reserve(partitions.size());
  custom_ops.reserve(partitions.size());

  for (auto& partition : partitions) {
    LrtSubgraph new_subgraph = &model->subgraphs.emplace_back();

    LrtOp custom_op = algo::GraphSlicer::SlicePartitionFromGraph(
        main_subgraph, new_subgraph, partition);
    custom_ops.push_back(custom_op);
    slices.push_back(new_subgraph);

    DumpSubgraph(*new_subgraph, "New subgraph");
  }

  DumpSubgraph(main_subgraph, "Main subgraph after partioning.");

  if (dry_run) {
    return StatusOk();
  }

  LrtCompiledResult compiled_result;
  LRT_RETURN_STATUS_IF_NOT_OK(
      LrtPluginCompile(plugin, slices.data(), slices.size(), &compiled_result));

  lrt_param_index_t num_calls_compiled;
  LRT_RETURN_STATUS_IF_NOT_OK(
      LrtCompiledResultGetNumCalls(compiled_result, &num_calls_compiled));

  if (num_calls_compiled != slices.size()) {
    std::cerr
        << "Plugin must provide and entry point for each compiled partition\n";
    return StatusCreate(kLrtStatusErrorNotFound);
  }

  for (int i = 0; i < num_calls_compiled; ++i) {
    const void* call_info;
    size_t call_info_size;

    LRT_RETURN_STATUS_IF_NOT_OK(LrtCompiledResultGetCallInfo(
        compiled_result, i, &call_info, &call_info_size));

    auto* custom_op = custom_ops.at(i);
    custom_op->custom_options.assign(reinterpret_cast<const char*>(call_info),
                                     call_info_size);
  }

  const void* byte_code;
  size_t byte_code_size;

  LRT_RETURN_STATUS_IF_NOT_OK(LrtCompiledResultGetByteCode(
      compiled_result, &byte_code, &byte_code_size));

  LRT_RETURN_STATUS_IF_NOT_OK(AppendMetadata(model, byte_code, byte_code_size,
                                             LrtPluginSocManufacturer()));

  return StatusOk();
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  auto model = LoadModel(model_path);
  EXIT_IF_NULL(model, "Failed to load model");

  auto plugin = LoadPlugin();
  EXIT_IF_NULL(plugin, "Failed to load plugin.");

  LRT_RETURN_VAL_IF_NOT_OK(ApplyPlugin(model.get(), plugin.get()), 1);

  uint8_t* buf;
  size_t buf_size;
  size_t buf_offset;

  LRT_RETURN_VAL_IF_NOT_OK(
      SerializeModel(model.release(), &buf, &buf_size, &buf_offset), 1);

  std::string out(reinterpret_cast<const char*>(buf) + buf_offset,
                  buf_size - buf_offset);
  std::cout << out;

  delete[] buf;

  return 0;
}
