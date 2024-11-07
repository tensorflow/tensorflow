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

#include "tensorflow/lite/experimental/litert/tools/apply_plugin.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_support.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_support.h"
#include "tensorflow/lite/experimental/litert/compiler/plugin/algo.h"
#include "tensorflow/lite/experimental/litert/compiler/plugin/compiler_plugin.h"
#include "tensorflow/lite/experimental/litert/core/byte_code_util.h"
#include "tensorflow/lite/experimental/litert/core/model/model_load.h"
#include "tensorflow/lite/experimental/litert/core/model/model_serialize.h"
#include "tensorflow/lite/experimental/litert/core/util/buffer_ref.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/experimental/litert/tools/dump.h"
#include "tensorflow/lite/experimental/litert/tools/tool_display.h"

namespace litert::tools {

using ::litert::BufferRef;
using ::litert::OwningBufferRef;
using ::litert::internal::CompilerPlugin;
using ::litert::internal::Dump;
using ::litert::internal::FinishByteCodePlaceholders;
using ::litert::internal::GroupPartitions;
using ::litert::internal::kByteCodeMetadataKey;
using ::litert::internal::kLiteRtBuildStampKey;
using ::litert::internal::kLiteRtDispatchOpCustomCode;
using ::litert::internal::LoadModelFromFile;
using ::litert::internal::MakeBuildStamp;
using ::litert::internal::MakeByteCodePlaceholder;
using ::litert::internal::MakeExecInfo;
using ::litert::internal::OutlinePartition;
using ::litert::internal::Serialization;
using ::litert::internal::VerifyFlatbuffer;
using ::litert::tools::ApplyPluginRun;

#define LITERT_ENSURE_CONFIG(expr)              \
  if (!(expr)) {                                \
    return kLiteRtStatusErrorInvalidToolConfig; \
  }

namespace {

static constexpr absl::string_view kArt = R"(
    __    _ __       ____  __
   / /   (_/ /____  / __ \/ /_
  / /   / / __/ _ \/ /_/ / __/
 / /___/ / /_/  __/ _, _/ /_
/_____/_/\__/\___/_/ |_|\__/
)";

class Context {
 public:
  using Ptr = std::unique_ptr<Context>;
  using ResultT = LiteRtResult<Context>;

  explicit Context(ApplyPluginRun::Ptr run)
      : run_(std::move(run)),
        display_(ToolDisplay(run_->dump_out, Context::CmdStr(run_->cmd))) {}

  ApplyPluginRun::Cmd Cmd() const { return run_->cmd; }

  absl::Span<const absl::string_view> LibSearchPaths() const {
    return absl::MakeConstSpan(run_->lib_search_paths.data(),
                               run_->lib_search_paths.size());
  }

  absl::string_view SocModelTarget() const {
    ABSL_CHECK_EQ(run_->soc_models.size(), 1);
    return run_->soc_models.front();
  }

  absl::string_view SocManufacturer() const {
    return run_->soc_manufacturer.value();
  }

  std::ostream& Out() {
    ABSL_CHECK_EQ(run_->outs.size(), 1);
    return run_->outs.front();
  }

  ApplyPluginRun::OutStreamT SwapOut(ApplyPluginRun::OutStreamT out) {
    ABSL_CHECK_EQ(run_->outs.size(), 1);
    auto res = run_->outs.front();
    run_->outs.at(0) = out;
    return res;
  }

  Serialization Serialization() const { return run_->serialization; }

  const ApplyPluginRun& Run() const { return *run_; }
  ApplyPluginRun& Run() { return *run_; }

  ToolDisplay& Dump() { return display_; }

  void DumpPrelude();

  static absl::string_view CmdStr(ApplyPluginRun::Cmd cmd);

 private:
  ApplyPluginRun::Ptr run_;
  ToolDisplay display_;
};

absl::string_view Context::CmdStr(ApplyPluginRun::Cmd cmd) {
  switch (cmd) {
    case ApplyPluginRun::Cmd::INFO:
      return "INFO";
    case ApplyPluginRun::Cmd::NOOP:
      return "NOOP";
    case ApplyPluginRun::Cmd::PARTITION:
      return "PARTITION";
    case ApplyPluginRun::Cmd::COMPILE:
      return "COMPILE";
    case ApplyPluginRun::Cmd::APPLY:
      return "APPLY";
  }
}

void Context::DumpPrelude() {
  Dump().Display() << kArt << "\n";
  // TODO pretty print run struct.
}

CompilerPlugin::ResultVecT LoadAllPlugins(Context* ctx) {
  ctx->Dump().Start("Load Plugins");
  ctx->Dump().Labeled() << "Loading plugins from: ";
  const auto paths = ctx->LibSearchPaths();
  for (auto it = paths.begin(); it < paths.end(); ++it) {
    ctx->Dump().Display() << *it;
    if (it < paths.end() - 1) {
      ctx->Dump().Display() << ", ";
    }
  }
  ctx->Dump().Display() << "\n";

  auto plugins = CompilerPlugin::LoadPlugins(ctx->LibSearchPaths());
  if (!plugins.HasValue()) {
    ctx->Dump().Fail();
    return plugins;
  }
  ctx->Dump().Labeled() << "Found plugins\n";
  ctx->Dump().Labeled() << absl::StreamFormat("Loaded %lu plugins\n",
                                              plugins.Value().size());

  ctx->Dump().Done();
  return plugins;
}

CompilerPlugin::ResultT LoadPlugin(Context* ctx) {
  LITERT_MOVE_OR_RETURN_RESULT(auto plugins, LoadAllPlugins(ctx),
                               CompilerPlugin);
  ctx->Dump().Start("Select Plugin");

  for (auto& plugin : plugins) {
    if (plugin.SocManufacturer() == ctx->Run().soc_manufacturer) {
      ctx->Dump().Labeled() << absl::StreamFormat("Selected plugin for: %s\n",
                                                  plugin.SocManufacturer());
      ctx->Dump().Done();
      return CompilerPlugin::ResultT::TakeValue(std::move(plugin));
    }
  }

  ctx->Dump().Fail();
  return CompilerPlugin::ResultT::FromStatus(kLiteRtStatusErrorNotFound);
}

LiteRtResult<Model> LoadModel(Context* ctx) {
  ctx->Dump().Start("Load Model");
  ctx->Dump().Labeled() << absl::StreamFormat("Loading model from: %s\n",
                                              ctx->Run().model.value());
  auto model_result = LoadModelFromFile(ctx->Run().model->data());
  if (!model_result.HasValue()) {
    ctx->Dump().Labeled() << "Failed to load model from file.";
    ctx->Dump().Fail();
    return model_result;
  }

  ctx->Dump().Labeled();
  Dump(*model_result.Value().Get(), ctx->Dump().Display());
  ctx->Dump().Done();

  return model_result;
}

std::vector<LiteRtOp> ApplyPartition(Context* ctx, const Model& model,
                                     CompilerPlugin& plugin) {
  ctx->Dump().Start("Partition Model");
  model.Get()->custom_op_code = kLiteRtDispatchOpCustomCode;

  ctx->Dump().Labeled() << "Input model: \n";
  for (auto it = model.Get()->subgraphs.begin();
       it < model.Get()->subgraphs.end(); ++it) {
    ctx->Dump().Labeled();
    ctx->Dump().Indented() << "(input graph) ";
    Dump(*it, ctx->Dump().Display());
  }

  auto partiion = plugin.PartitionModel(model);
  if (!partiion.HasValue()) {
    return {};
  }
  auto grouped_partitions = GroupPartitions(partiion.Value());
  if (grouped_partitions.empty()) {
    return {};
  }
  ctx->Dump().Labeled() << absl::StreamFormat(
      "Plugin selected %lu ops, yielding %lu partitions\n",
      partiion.Value().size(), grouped_partitions.size());

  std::vector<LiteRtOp> res;
  for (auto& partition : grouped_partitions) {
    LiteRtOp custom_op =
        OutlinePartition(model.Get()->subgraphs.front(),
                         &model.Get()->subgraphs.emplace_back(), partition);
    res.push_back(custom_op);
  }

  ctx->Dump().Labeled() << "Partitioned model: \n";
  ctx->Dump().Labeled();
  ctx->Dump().Indented() << "(initial graph) ";
  Dump(model.Get()->subgraphs.front(), ctx->Dump().Display());
  for (auto it = model.Get()->subgraphs.begin() + 1;
       it < model.Get()->subgraphs.end(); ++it) {
    ctx->Dump().Labeled();
    ctx->Dump().Indented() << "(new graph) ";
    Dump(*it, ctx->Dump().Display());
  }

  ctx->Dump().Done();
  return res;
}

LiteRtResult<Model> PartitionModel(Context* ctx, Model&& model,
                                   CompilerPlugin& plugin) {
  auto custom_ops = ApplyPartition(ctx, model, plugin);
  if (custom_ops.empty()) {
    return LiteRtResult<Model>::FromStatus(kLiteRtStatusErrorGraphModification);
  }
  return LiteRtResult<Model>::TakeValue(std::move(model));
}

LiteRtResult<std::vector<std::string>> CompilePartitions(
    Context* ctx, std::vector<LiteRtSubgraph>& partitions,
    CompilerPlugin& plugin) {
  ctx->Dump().Start("Compile Model");
  ctx->Dump().Labeled() << absl::StreamFormat(
      "Requesting compilation for target \"%s\" on %lu subgraphs\n",
      ctx->SocModelTarget(), partitions.size());

  std::vector<std::string> call_info_out;
  if (plugin.Compile(ctx->SocModelTarget(), partitions, ctx->Out(),
                     call_info_out) != kLiteRtStatusOk) {
    ctx->Dump().Fail();
    return LiteRtResult<std::vector<std::string>>::FromStatus(
        kLiteRtStatusErrorCompilationr);
  }

  ctx->Dump().Labeled() << "Entry point info: ";
  for (auto it = call_info_out.begin(); it < call_info_out.end(); ++it) {
    ctx->Dump().Display() << absl::StreamFormat("\"%s\"", *it);
    if (it < call_info_out.end() - 1) {
      ctx->Dump().Display() << ", ";
    }
  }
  ctx->Dump().Display() << "\n";

  ctx->Dump().Done();
  return LiteRtResult<std::vector<std::string>>::TakeValue(
      std::move(call_info_out));
}

//
// INFO Command
//

LiteRtStatus ValidateInfoRun(const ApplyPluginRun& run) {
  LITERT_ENSURE_CONFIG(!run.lib_search_paths.empty());
  LITERT_ENSURE_CONFIG(run.outs.size() == 1);
  return kLiteRtStatusOk;
}

LiteRtStatus Info(Context* ctx) {
  LITERT_MOVE_OR_RETURN_STATUS(auto plugins, LoadAllPlugins(ctx));
  for (auto& plugin : plugins) {
    ctx->Out() << absl::StreamFormat("< LiteRtCompilerPlugin > \"%s\" | ",
                                     plugin.SocManufacturer());
    const auto& models = plugin.SocModels();
    for (auto it = models.begin(); it < models.end(); ++it) {
      ctx->Out() << absl::StreamFormat("\"%s\"", *it);
      if (it < models.end() - 1) {
        ctx->Out() << ", ";
      }
    }
    ctx->Out() << "\n";
  }
  return kLiteRtStatusOk;
}

//
// NOOP Command
//

LiteRtStatus ValidateNoopRun(const ApplyPluginRun& run) {
  LITERT_ENSURE_CONFIG(run.model.has_value());
  LITERT_ENSURE_CONFIG(run.outs.size() == 1);
  return kLiteRtStatusOk;
}

LiteRtStatus Noop(Context* ctx) {
  LITERT_MOVE_OR_RETURN_STATUS(auto model, LoadModel(ctx));
  LITERT_MOVE_OR_RETURN_STATUS(auto serialized,
                               SerializeModel(std::move(model)));
  LITERT_ENSURE(VerifyFlatbuffer(serialized.Span()),
                kLiteRtStatusErrorInvalidFlatbuffer,
                "Failed to invalidate flatbuffer");
  serialized.WriteStr(ctx->Out());
  return kLiteRtStatusOk;
}

//
// PARTITION Command
//

LiteRtStatus ValidatePartitionRun(const ApplyPluginRun& run) {
  LITERT_ENSURE_CONFIG(!run.lib_search_paths.empty());
  LITERT_ENSURE_CONFIG(run.model.has_value());
  LITERT_ENSURE_CONFIG(run.soc_manufacturer.has_value());
  LITERT_ENSURE_CONFIG(!run.outs.empty());
  return kLiteRtStatusOk;
}

LiteRtStatus Partition(Context* ctx) {
  LITERT_MOVE_OR_RETURN_STATUS(auto plugin, LoadPlugin(ctx));
  LITERT_MOVE_OR_RETURN_STATUS(auto model, LoadModel(ctx));

  LITERT_MOVE_OR_RETURN_STATUS(auto new_model,
                               PartitionModel(ctx, std::move(model), plugin));
  LITERT_MOVE_OR_RETURN_STATUS(auto serialized,
                               SerializeModel(std::move(new_model)));
  LITERT_ENSURE(VerifyFlatbuffer(serialized.Span()),
                kLiteRtStatusErrorInvalidFlatbuffer,
                "Failed to invalidate flatbuffer");
  serialized.WriteStr(ctx->Out());
  return kLiteRtStatusOk;
}

//
// COMPILE Command
//

LiteRtStatus ValidateCompileRun(const ApplyPluginRun& run) {
  LITERT_ENSURE_CONFIG(!run.lib_search_paths.empty());
  LITERT_ENSURE_CONFIG(run.model.has_value());
  LITERT_ENSURE_CONFIG(run.soc_manufacturer.has_value());
  LITERT_ENSURE_CONFIG(run.outs.size() == run.soc_models.size());
  // TODO: implement multi target compilation.
  LITERT_ENSURE_SUPPORTED(run.soc_models.size() == 1,
                          "Multi target compilation not implemented.");
  return kLiteRtStatusOk;
}

LiteRtStatus Compile(Context* ctx) {
  LITERT_MOVE_OR_RETURN_STATUS(auto model, LoadModel(ctx));
  LITERT_MOVE_OR_RETURN_STATUS(auto plugin, LoadPlugin(ctx));

  std::vector<LiteRtSubgraph> compilation_input;
  compilation_input.reserve(model.Get()->subgraphs.size());
  for (auto& subgraph : model.Get()->subgraphs) {
    compilation_input.push_back(&subgraph);
  }
  LITERT_MOVE_OR_RETURN_STATUS(
      auto entry_point_info, CompilePartitions(ctx, compilation_input, plugin));

  return kLiteRtStatusOk;
}

//
// APPLY Command
//

LiteRtStatus StampModel(Context* ctx, LiteRtModel model) {
  LITERT_ASSIGN_OR_RETURN_STATUS(
      auto stamp, MakeBuildStamp(ctx->SocManufacturer(), ctx->SocModelTarget(),
                                 ctx->Serialization()));
  ctx->Dump().Labeled() << absl::StreamFormat("Stamping model: %s\n",
                                              stamp.StrView());
  return model->PushMetadata(kLiteRtBuildStampKey, stamp);
}

LiteRtResult<OwningBufferRef<uint8_t>> DoMetadataSerialization(
    Context* ctx, std::vector<LiteRtOp>& custom_ops,
    std::vector<std::string>& call_info, BufferRef<uint8_t> compilation_out,
    Model&& model) {
  using ResT = OwningBufferRef<uint8_t>;

  ctx->Dump().Start("Serializing with bytecode in METADATA");

  {
    auto call_it = call_info.begin();
    auto custom_op_it = custom_ops.begin();
    for (; call_it < call_info.end() && custom_op_it < custom_ops.end();) {
      (*custom_op_it)->custom_options =
          OwningBufferRef<uint8_t>((*call_it).c_str());
      ++call_it;
      ++custom_op_it;
    }
  }

  {
    ctx->Dump().Labeled() << absl::StreamFormat(
        "Adding metadata byte code of size: %lu bytes\n",
        compilation_out.Size());

    LITERT_RETURN_RESULT_IF_NOT_OK(
        model.Get()->PushMetadata(kByteCodeMetadataKey, compilation_out), ResT);
  }

  LITERT_ASSIGN_OR_RETURN_RESULT(auto serialized,
                                 SerializeModel(std::move(model)),
                                 OwningBufferRef<uint8_t>);
  ctx->Dump().Labeled() << absl::StreamFormat(
      "Serialized model of size: %lu bytes\n", serialized.Size());
  if (!VerifyFlatbuffer(serialized.Span())) {
    ctx->Dump().Fail();
    return LiteRtResult<ResT>::FromStatus(kLiteRtStatusErrorInvalidFlatbuffer);
  }
  ctx->Dump().Done();

  return LiteRtResult<ResT>::TakeValue(std::move(serialized));
}

LiteRtResult<OwningBufferRef<uint8_t>> DoAppendSerialization(
    Context* ctx, std::vector<LiteRtOp>& custom_ops,
    std::vector<std::string>& call_info, BufferRef<uint8_t> compilation_out,
    Model&& model) {
  using ResT = OwningBufferRef<uint8_t>;
  ctx->Dump().Start("Serializing with bytecode APPEND");

  // This need not be the same for all custom ops.
  static constexpr absl::string_view kSharedByteCodePlaceholderName =
      kByteCodeMetadataKey;
  LITERT_RETURN_RESULT_IF_NOT_OK(
      model.Get()->PushMetadata(kSharedByteCodePlaceholderName,
                                std::move(MakeByteCodePlaceholder())),
      ResT);

  {
    auto call_it = call_info.begin();
    auto custom_op_it = custom_ops.begin();
    for (; call_it < call_info.end() && custom_op_it < custom_ops.end();) {
      LITERT_ASSIGN_OR_RETURN_RESULT(
          auto exec_info,
          MakeExecInfo(*call_it, kSharedByteCodePlaceholderName), ResT);
      (*custom_op_it)->custom_options = std::move(exec_info);
      ++call_it;
      ++custom_op_it;
    }
  }

  LITERT_ASSIGN_OR_RETURN_RESULT(auto serialized,
                                 SerializeModel(std::move(model)), ResT);
  ctx->Dump().Labeled() << absl::StreamFormat(
      "Serialized model of size: %lu bytes\n", serialized.Size());
  LITERT_RETURN_RESULT_IF_NOT_OK(
      FinishByteCodePlaceholders(serialized, compilation_out.Size()), ResT);

  OwningBufferRef<uint8_t> with_append(serialized.Size() +
                                       compilation_out.Size());

  uint8_t* write = with_append.Data();
  std::memcpy(write, serialized.Data(), serialized.Size());
  write += serialized.Size();
  std::memcpy(write, compilation_out.Data(), compilation_out.Size());

  ctx->Dump().Labeled() << absl::StreamFormat(
      "Appended byte code of size %lu\n", compilation_out.Size());

  ctx->Dump().Done();
  return LiteRtResult<ResT>::TakeValue(std::move(with_append));
}

LiteRtStatus ValidateApplyRun(const ApplyPluginRun& run) {
  LITERT_ENSURE_CONFIG(!run.lib_search_paths.empty());
  LITERT_ENSURE_CONFIG(run.model.has_value());
  LITERT_ENSURE_CONFIG(run.soc_manufacturer.has_value());
  LITERT_ENSURE_CONFIG(run.outs.size() == run.soc_models.size());
  // TODO: implement multi target compilation.
  LITERT_ENSURE_SUPPORTED(run.soc_models.size() == 1,
                          "Multi target compilation not implemented.");
  LITERT_ENSURE_SUPPORTED(run.serialization != Serialization::kUnknown,
                          "No serialization strategy supported.");
  return kLiteRtStatusOk;
}

LiteRtStatus Apply(Context* ctx) {
  LITERT_MOVE_OR_RETURN_STATUS(auto model, LoadModel(ctx));
  LITERT_MOVE_OR_RETURN_STATUS(auto plugin, LoadPlugin(ctx));
  static constexpr size_t kNumInputSubgraphs = 1;
  LITERT_ENSURE_SUPPORTED(model.Get()->subgraphs.size() == kNumInputSubgraphs,
                          "Only single subgraph models currently supported.");

  // Query plugin for compilable ops and slice partitions out of the graph,
  // replacing use with single custom op..
  auto custom_ops = ApplyPartition(ctx, model, plugin);
  LITERT_ENSURE(!custom_ops.empty(), kLiteRtStatusErrorGraphModification,
                "Failed to partiion graph.");
  // All new subgraphs to be compiled are appended to the model's subgraphs.
  std::vector<LiteRtSubgraph> compilation_input;
  for (auto it = model.Get()->subgraphs.begin() + kNumInputSubgraphs;
       it < model.Get()->subgraphs.end(); ++it) {
    compilation_input.push_back(&*it);
  }

  // Call compilation method on the plugin.
  std::stringstream compilation_out;
  ApplyPluginRun::OutStreamT out = ctx->SwapOut(compilation_out);

  LITERT_MOVE_OR_RETURN_STATUS(
      auto call_info, CompilePartitions(ctx, compilation_input, plugin));

  // Update custom op info the it's respective entry point info from the plugin.
  LITERT_ENSURE(call_info.size() == custom_ops.size(),
                kLiteRtStatusErrorCompilationr,
                "Failed to verify entry point information.");

  model.Get()->subgraphs.resize(kNumInputSubgraphs);
  LITERT_RETURN_STATUS_IF_NOT_OK(StampModel(ctx, model.Get()));

  BufferRef<uint8_t> compiled_buffer(compilation_out.view().data(),
                                     compilation_out.view().size());

  ctx->SwapOut(out);
  if (ctx->Serialization() == Serialization::kMetadata) {
    LITERT_ASSIGN_OR_RETURN_STATUS(
        auto serialized,
        DoMetadataSerialization(ctx, custom_ops, call_info, compiled_buffer,
                                std::move(model)));
    serialized.WriteStr(ctx->Out());

  } else if (ctx->Serialization() == Serialization::kAppend) {
    LITERT_ASSIGN_OR_RETURN_STATUS(
        auto serialized,
        DoAppendSerialization(ctx, custom_ops, call_info, compiled_buffer,
                              std::move(model)));
    serialized.WriteStr(ctx->Out());
  } else {
    return kLiteRtStatusErrorUnsupported;
  }
  return kLiteRtStatusOk;
}

}  // namespace

LiteRtStatus ApplyPlugin(ApplyPluginRun::Ptr run) {
  Context context(std::move(run));
  context.DumpPrelude();

  switch (context.Cmd()) {
    case ApplyPluginRun::Cmd::INFO:
      LITERT_RETURN_STATUS_IF_NOT_OK(ValidateInfoRun(context.Run()));
      return Info(&context);

    case ApplyPluginRun::Cmd::PARTITION:
      LITERT_RETURN_STATUS_IF_NOT_OK(ValidatePartitionRun(context.Run()));
      return Partition(&context);

    case ApplyPluginRun::Cmd::COMPILE:
      LITERT_RETURN_STATUS_IF_NOT_OK(ValidateCompileRun(context.Run()));
      return Compile(&context);

    case ApplyPluginRun::Cmd::APPLY:
      LITERT_RETURN_STATUS_IF_NOT_OK(ValidateApplyRun(context.Run()));
      return Apply(&context);

    case ApplyPluginRun::Cmd::NOOP:
      LITERT_RETURN_STATUS_IF_NOT_OK(ValidateNoopRun(context.Run()));
      return Noop(&context);

    default:
      return kLiteRtStatusErrorInvalidArgument;
  }

  return kLiteRtStatusOk;
}

}  // namespace litert::tools
