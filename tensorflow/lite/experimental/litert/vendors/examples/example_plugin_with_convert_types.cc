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
#include <sstream>
#include <utility>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/litert_convert.h"
#include "tensorflow/lite/experimental/litert/vendors/examples/example_convert_types_impl.h"
#include "tensorflow/lite/experimental/litert/vendors/examples/example_plugin_common.h"

// A simple compiler plugin example that implements the interface using the
// litert_convert_types utilities. The basic idea is that users implement a
// small set of core functionality which is factored into generic algorithms for
// partitioning/graph construction. Implementations for said functionality is
// found in example_plugin_convert_types_impl. This plugin matches on mul ops,
// and emits "byte code" that is simply a string representative of the ops
// consumed.

using ::litert::Model;
using ::litert::PartitionViaCapabilities;
using ::litert::Subgraph;
using ::litert::example::ExampleCapability;
using ::litert::example::ExampleConvertTensor;
using ::litert::example::ExampleGraphContext;
using ::litert::example::ExampleGraphConverter;
using ::litert::example::ExampleLegalizer;
using ::litert::example::ExampleOp;
using ::litert::example::MakeExampleGraphFinalizer;
using ::litert::example::MakeExampleGraphInitializer;
using ::litert::example::MakeExampleOpFinalizer;
using ::litert::example::MakeExampleTensorFinalizer;
using Legalisation =
    ::litert::example::ExampleOpLegalization<kLiteRtOpCodeTflMul>;
using ::litert::Expected;
using ::litert::example::ExampleTensor;

// Plugins can hold state.
struct LiteRtCompilerPluginT {};

namespace {

Legalisation::PtrVec MakeLegalizations() {
  Legalisation::PtrVec res;
  res.emplace_back(Legalisation::Create());
  return res;
}

Expected<ExampleLegalizer::Ptr> MakeExampleLegalizer() {
  auto legalizer = std::make_unique<ExampleLegalizer>();
  for (auto& legalization : MakeLegalizations()) {
    LITERT_EXPECT_OK(legalizer->Register(std::move(legalization)));
  }
  return legalizer;
}

}  // namespace

// Initialize example plugin and register legalizations.
LiteRtStatus LiteRtCreateCompilerPlugin(LiteRtCompilerPlugin* compiler_plugin) {
  *compiler_plugin = new LiteRtCompilerPluginT;
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin) {
  delete compiler_plugin;
}

// Leverage the convert_type PartitionViaCapabilties algorithm for partitioning
// implementation.
LiteRtStatus LiteRtCompilerPluginPartitionModel(
    LiteRtCompilerPlugin compiler_plugin, LiteRtModel model,
    LiteRtOpList selected_ops) {
  auto legalizer = MakeExampleLegalizer();
  if (!legalizer) {
    return legalizer.Error().Status();
  }
  auto model_wrapper = Model::CreateFromNonOwnedHandle(model);
  return PartitionViaCapabilities<ExampleOp, ExampleTensor>(
      **legalizer, ExampleCapability, model_wrapper,
      [&](auto op) { return LiteRtPushOp(selected_ops, op); });
}

namespace {

// "Compile" a string representation of given example op.
std::string CompileOp(const ExampleOp& example_op) {
  static constexpr absl::string_view kInputFmt = "inputs[%s]";
  static constexpr absl::string_view kOutputFmt = "outputs[%s]";
  static constexpr absl::string_view kOpCodeFmt = "op_code[%d]";
  static constexpr absl::string_view kFmt = "%s%s%s";
  static constexpr absl::string_view kDelim = ",";

  auto join = [](const auto& strs) {
    std::stringstream s;
    for (auto it = strs.begin(); it < strs.end(); ++it) {
      if (it->empty()) {
        s << "<empty>";
      } else {
        s << *it;
      }
      if (it != strs.end() - 1) {
        s << kDelim;
      }
    }
    return s.str();
  };

  const auto input_str =
      absl::StrFormat(kInputFmt, join(example_op.input_types));
  const auto output_str =
      absl::StrFormat(kOutputFmt, join(example_op.output_types));
  const auto opcode_str =
      absl::StrFormat(kOpCodeFmt, static_cast<int>(example_op.code));

  return absl::StrFormat(kFmt, opcode_str, input_str, output_str);
}

// "Compile" the example results from a single subgraph conversion into
// a representative string.
std::string CompileSubgraph(const ExampleGraphContext::Info& info) {
  std::string res;
  for (const auto& backend_op : info.backend_ops) {
    res += CompileOp(backend_op);
  }
  return res;
}

// Make a subgraph name from its index.
std::string MakeGraphName(LiteRtParamIndex graph_idx) {
  return absl::StrFormat("partition_%lu", graph_idx);
}

}  // namespace

// Plugin compiler implementation that leverages the pluggable convert_types
// infrastructure.
LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtSubgraphArray partitions, LiteRtParamIndex num_partitions,
    LiteRtCompiledResult* compiled_result) {
  ExampleGraphContext graph_context;
  ExampleGraphConverter converter(ExampleConvertTensor,
                                  MakeExampleTensorFinalizer(graph_context),
                                  MakeExampleOpFinalizer(graph_context),
                                  MakeExampleGraphInitializer(graph_context),
                                  MakeExampleGraphFinalizer(graph_context));

  for (auto& legalization : MakeLegalizations()) {
    LITERT_RETURN_STATUS_IF_NOT_OK(converter.Register(std::move(legalization)));
  }

  auto graphs = absl::MakeSpan(partitions, num_partitions);
  for (auto i = 0; i < graphs.size(); ++i) {
    LITERT_RETURN_STATUS_IF_NOT_OK(
        converter.ConvertGraph(Subgraph(graphs[i]), MakeGraphName(i)));
  }

  if (graph_context.infos.size() != num_partitions) {
    return kLiteRtStatusErrorNotFound;
  }

  auto result = new LiteRtCompiledResultT;
  for (const auto& info : graph_context.infos) {
    if (!info.finalized) {
      return kLiteRtStatusErrorNotFound;
    }
    result->per_op_data.push_back(info.name);
    result->byte_code.append(CompileSubgraph(info));
  }

  *compiled_result = result;

  return kLiteRtStatusOk;
}
