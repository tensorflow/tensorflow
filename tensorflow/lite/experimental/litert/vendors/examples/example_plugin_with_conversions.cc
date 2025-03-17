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

#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_format.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/convert_graph.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/partition_with_capabilities.h"
#include "tensorflow/lite/experimental/litert/vendors/examples/example_conversion_impl.h"
#include "tensorflow/lite/experimental/litert/vendors/examples/example_ir.h"
#include "tensorflow/lite/experimental/litert/vendors/examples/example_plugin_common.h"

using ::litert::PartitionWithCapabilities;
using ::litert::example::ExampleGraphBuilder;
using ::litert::example::ExampleOpAllocator;
using ::litert::example::ExampleOpType;
using ::litert::example::ExampleTensorAllocator;
using ::litert::example::ExampleTypes;
using ::litert::example::MakeAllLegalizations;
using ::litert::example::MakeTensorConverter;

// Example plugin implementations that leverage the pluggable conversion
// infrastructure. Implementations of common interfaces are provided in
// example_conversion_impl.h. These are passed to higher-level litert functions
// to perform the actual conversion.
// The primary benifit of this approach is the re-use of conversion logic
// between the partition and compile phases.

// Plugins can hold state.
struct LiteRtCompilerPluginT {
  ExampleTypes::Legalizations legalizations;
};

namespace {

bool MulCapability(const ExampleTypes::Op* op) {
  return op->op_code == ExampleOpType::MUL;
}

}  // namespace

// Initialize example plugin and register legalizations.
LiteRtStatus LiteRtCreateCompilerPlugin(LiteRtCompilerPlugin* compiler_plugin) {
  auto* plugin = new LiteRtCompilerPluginT;
  plugin->legalizations = MakeAllLegalizations();
  *compiler_plugin = plugin;
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin) {
  delete compiler_plugin;
}

// Leverage the convert_type PartitionViaCapabilties algorithm for partitioning
// implementation.
LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                           const char* soc_model,
                                           LiteRtSubgraph subgraph,
                                           LiteRtOpList selected_ops) {
  ExampleTensorAllocator tensor_alloc;
  ExampleOpAllocator op_alloc;

  auto ops = PartitionWithCapabilities<ExampleTypes>(
      compiler_plugin->legalizations, MulCapability, MakeTensorConverter,
      tensor_alloc, op_alloc, ::litert::Subgraph(subgraph));
  if (!ops) {
    return ops.Error().Status();
  }

  for (auto* op : *ops) {
    LITERT_RETURN_IF_ERROR(LiteRtPushOp(selected_ops, op, 0));
  }

  return kLiteRtStatusOk;
}

namespace {

LiteRtStatus CompileSinglePartition(
    const ExampleTypes::Legalizations& legalizations, std::string name,
    LiteRtSubgraph subgraph, LiteRtCompiledResultT& result) {
  ::litert::Subgraph litert_subgraph(subgraph);

  ExampleTensorAllocator tensor_alloc;
  ExampleOpAllocator op_alloc;

  ExampleGraphBuilder builder;

  LITERT_RETURN_IF_ERROR(::litert::ConvertGraph<ExampleTypes>(
      litert_subgraph, name, MakeTensorConverter, tensor_alloc, op_alloc,
      legalizations, builder));

  // This example plugin only supports a single byte code module.
  result.byte_code[0].append(builder.Serialize());
  result.per_op_data.push_back(std::move(name));

  return kLiteRtStatusOk;
}

}  // namespace

// Plugin compiler implementation that leverages the pluggable convert_types
// infrastructure.
LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result) {
  auto model = litert::Model::CreateFromNonOwnedHandle(partitions);
  const auto num_partitions = model.NumSubgraphs();
  auto result = std::make_unique<LiteRtCompiledResultT>();
  result->byte_code.resize(num_partitions);
  for (auto i = 0; i < num_partitions; ++i) {
    auto name = absl::StrFormat("partition_%lu", i);
    LITERT_RETURN_IF_ERROR(
        CompileSinglePartition(compiler_plugin->legalizations, std::move(name),
                               model.Subgraph(i)->Get(), *result));
  }

  *compiled_result = result.release();

  return kLiteRtStatusOk;
}
