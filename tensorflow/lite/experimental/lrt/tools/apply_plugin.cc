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

#include "tensorflow/lite/experimental/lrt/tools/apply_plugin.h"

#include <cstddef>
#include <cstdint>
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
#include "tensorflow/lite/experimental/lrt/c/litert_common.h"
#include "tensorflow/lite/experimental/lrt/c/litert_model.h"
#include "tensorflow/lite/experimental/lrt/c/litert_support.h"
#include "tensorflow/lite/experimental/lrt/cc/litert_support.h"
#include "tensorflow/lite/experimental/lrt/core/compiler_plugin/algo.h"
#include "tensorflow/lite/experimental/lrt/core/compiler_plugin/compiler_plugin.h"
#include "tensorflow/lite/experimental/lrt/core/litert_model_init.h"
#include "tensorflow/lite/experimental/lrt/test/common.h"
#include "tensorflow/lite/experimental/lrt/tools/dump.h"
#include "tensorflow/lite/experimental/lrt/tools/tool_display.h"

namespace litert::tools {

using ::litert::internal::CompilerPlugin;
using ::litert::internal::Dump;
using ::litert::internal::GroupPartitions;
using ::litert::internal::OutlinePartition;
using ::litert::testing::VerifyFlatbuffer;
using ::litert::tools::ApplyPluginRun;

#define _ENSURE_CONFIG(expr)                    \
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
      ctx->Dump().Done();
      return CompilerPlugin::ResultT::TakeValue(std::move(plugin));
    }
  }

  ctx->Dump().Fail();
  return CompilerPlugin::ResultT::FromStatus(kLiteRtStatusErrorNotFound);
}

LiteRtResult<UniqueLiteRtModel> LoadModel(Context* ctx) {
  ctx->Dump().Start("Load Model");
  ctx->Dump().Labeled() << absl::StreamFormat("Loading model from: %s\n",
                                              ctx->Run().model.value());

  LiteRtModel model;
  if (LoadModelFromFile(ctx->Run().model->data(), &model) != kLiteRtStatusOk) {
    ctx->Dump().Fail();
    return LiteRtResult<UniqueLiteRtModel>::FromStatus(
        kLiteRtStatusErrorFileIO);
  }

  ctx->Dump().Labeled();
  Dump(*model, ctx->Dump().Display());

  ctx->Dump().Done();
  return LiteRtResult<UniqueLiteRtModel>::TakeValue(UniqueLiteRtModel(model));
}

LiteRtStatus SerializeModel(Context* ctx, UniqueLiteRtModel model) {
  ctx->Dump().Start("Serialize Model");

  uint8_t* buf;
  size_t size;
  size_t offset;
  if (SerializeModel(model.release(), &buf, &size, &offset) !=
      kLiteRtStatusOk) {
    delete[] buf;
    ctx->Dump().Fail();
    return kLiteRtStatusErrorSerialization;
  }

  auto out_buf = buf + offset;
  const size_t out_size = size - offset;
  if (!VerifyFlatbuffer(out_buf, out_size)) {
    ctx->Dump().Labeled() << "Failed to verify flatbuffer\n";
    ctx->Dump().Fail();
    delete[] buf;
    return kLiteRtStatusErrorInvalidFlatbuffer;
  }

  ctx->Out().write(reinterpret_cast<const char*>(out_buf), out_size);
  ctx->Dump().Labeled() << absl::StreamFormat(
      "Serialized a model of size: %lu\n", out_size);

  delete[] buf;

  ctx->Dump().Done();
  return kLiteRtStatusOk;
}

std::vector<LiteRtOp> ApplyPartition(Context* ctx, LiteRtModelT& model,
                                     CompilerPlugin& plugin) {
  ctx->Dump().Start("Partition Model");
  LITERT_RETURN_VAL_IF_NOT_OK(
      RegisterCustomOpCode(&model, ctx->Run().soc_manufacturer->data()), {});

  ctx->Dump().Labeled() << "Input model: \n";
  for (auto it = model.subgraphs.begin(); it < model.subgraphs.end(); ++it) {
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
    LiteRtOp custom_op = OutlinePartition(
        model.subgraphs.front(), &model.subgraphs.emplace_back(), partition);
    res.push_back(custom_op);
  }

  ctx->Dump().Labeled() << "Partitioned model: \n";
  ctx->Dump().Labeled();
  ctx->Dump().Indented() << "(initial graph) ";
  Dump(model.subgraphs.front(), ctx->Dump().Display());
  for (auto it = model.subgraphs.begin() + 1; it < model.subgraphs.end();
       ++it) {
    ctx->Dump().Labeled();
    ctx->Dump().Indented() << "(new graph) ";
    Dump(*it, ctx->Dump().Display());
  }

  ctx->Dump().Done();
  return res;
}

LiteRtResult<UniqueLiteRtModel> PartitionModel(Context* ctx,
                                               UniqueLiteRtModel model,
                                               CompilerPlugin& plugin) {
  auto custom_ops = ApplyPartition(ctx, *model, plugin);
  if (custom_ops.empty()) {
    return LiteRtResult<UniqueLiteRtModel>::FromStatus(
        kLiteRtStatusErrorGraphModification);
  }
  return LiteRtResult<UniqueLiteRtModel>::TakeValue(std::move(model));
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
  _ENSURE_CONFIG(!run.lib_search_paths.empty());
  _ENSURE_CONFIG(run.outs.size() == 1);
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
  }
  return kLiteRtStatusOk;
}

//
// NOOP Command
//

LiteRtStatus ValidateNoopRun(const ApplyPluginRun& run) {
  _ENSURE_CONFIG(run.model.has_value());
  _ENSURE_CONFIG(run.outs.size() == 1);
  return kLiteRtStatusOk;
}

LiteRtStatus Noop(Context* ctx) {
  LITERT_MOVE_OR_RETURN_STATUS(auto model, LoadModel(ctx));
  LITERT_RETURN_STATUS_IF_NOT_OK(SerializeModel(ctx, std::move(model)));
  return kLiteRtStatusOk;
}

//
// PARTITION Command
//

LiteRtStatus ValidatePartitionRun(const ApplyPluginRun& run) {
  _ENSURE_CONFIG(!run.lib_search_paths.empty());
  _ENSURE_CONFIG(run.model.has_value());
  _ENSURE_CONFIG(run.soc_manufacturer.has_value());
  _ENSURE_CONFIG(!run.outs.empty());
  return kLiteRtStatusOk;
}

LiteRtStatus Partition(Context* ctx) {
  LITERT_MOVE_OR_RETURN_STATUS(auto plugin, LoadPlugin(ctx));
  LITERT_MOVE_OR_RETURN_STATUS(auto model, LoadModel(ctx));

  LITERT_MOVE_OR_RETURN_STATUS(auto new_model,
                               PartitionModel(ctx, std::move(model), plugin));
  LITERT_RETURN_STATUS_IF_NOT_OK(SerializeModel(ctx, std::move(new_model)));
  return kLiteRtStatusOk;
}

//
// COMPILE Command
//

LiteRtStatus ValidateCompileRun(const ApplyPluginRun& run) {
  _ENSURE_CONFIG(!run.lib_search_paths.empty());
  _ENSURE_CONFIG(run.model.has_value());
  _ENSURE_CONFIG(run.soc_manufacturer.has_value());
  _ENSURE_CONFIG(run.outs.size() == run.soc_models.size());
  // TODO: implement multi target compilation.
  LITERT_ENSURE_SUPPORTED(run.soc_models.size() == 1,
                          "Multi target compilation not implemented.");
  // TODO: implement append serialization.
  LITERT_ENSURE_SUPPORTED(
      run.serialization == ApplyPluginRun::Serialization::METADATA,
      "Only metadata serialization currently supported.");
  return kLiteRtStatusOk;
}

LiteRtStatus Compile(Context* ctx) {
  LITERT_MOVE_OR_RETURN_STATUS(auto model, LoadModel(ctx));
  LITERT_MOVE_OR_RETURN_STATUS(auto plugin, LoadPlugin(ctx));

  std::vector<LiteRtSubgraph> compilation_input;
  compilation_input.reserve(model->subgraphs.size());
  for (auto& subgraph : model->subgraphs) {
    compilation_input.push_back(&subgraph);
  }
  LITERT_MOVE_OR_RETURN_STATUS(
      auto entry_point_info, CompilePartitions(ctx, compilation_input, plugin));

  return kLiteRtStatusOk;
}

//
// APPLY Command
//

LiteRtStatus ValidateApplyRun(const ApplyPluginRun& run) {
  _ENSURE_CONFIG(!run.lib_search_paths.empty());
  _ENSURE_CONFIG(run.model.has_value());
  _ENSURE_CONFIG(run.soc_manufacturer.has_value());
  _ENSURE_CONFIG(run.outs.size() == run.soc_models.size());
  // TODO: implement multi target compilation.
  LITERT_ENSURE_SUPPORTED(run.soc_models.size() == 1,
                          "Multi target compilation not implemented.");
  // TODO: implement append serialization.
  LITERT_ENSURE_SUPPORTED(
      run.serialization == ApplyPluginRun::Serialization::METADATA,
      "Only metadata serialization currently supported.");
  return kLiteRtStatusOk;
}

LiteRtStatus Apply(Context* ctx) {
  LITERT_MOVE_OR_RETURN_STATUS(auto model, LoadModel(ctx));
  LITERT_MOVE_OR_RETURN_STATUS(auto plugin, LoadPlugin(ctx));
  ctx->Dump().Labeled() << "Loaded assets\n";
  static constexpr size_t kNumInputSubgraphs = 1;
  LITERT_ENSURE_SUPPORTED(model->subgraphs.size() == kNumInputSubgraphs,
                          "Only single subgraph models currently supported.");

  auto custom_ops = ApplyPartition(ctx, *model, plugin);
  LITERT_ENSURE(!custom_ops.empty(), kLiteRtStatusErrorGraphModification,
                "Failed to partiion graph.");

  std::vector<LiteRtSubgraph> compilation_input;
  for (auto it = model->subgraphs.begin() + kNumInputSubgraphs;
       it < model->subgraphs.end(); ++it) {
    compilation_input.push_back(&*it);
  }

  std::stringstream compilation_out;
  ApplyPluginRun::OutStreamT out = ctx->SwapOut(compilation_out);
  LITERT_MOVE_OR_RETURN_STATUS(
      auto call_info, CompilePartitions(ctx, compilation_input, plugin));
  LITERT_ENSURE(call_info.size() == custom_ops.size(),
                kLiteRtStatusErrorCompilationr,
                "Failed to verify entry point information.");

  auto call_it = call_info.begin();
  auto custom_op_it = custom_ops.begin();
  for (; call_it < call_info.end() && custom_op_it < custom_ops.end();) {
    (*custom_op_it)->custom_options.swap(*call_it);
    ++call_it;
    ++custom_op_it;
  }

  model->subgraphs.resize(kNumInputSubgraphs);

  LITERT_RETURN_STATUS_IF_NOT_OK(AppendMetadata(
      model.get(), compilation_out.str().data(), compilation_out.str().size(),
      plugin.SocManufacturer().data()));

  ctx->SwapOut(out);
  LITERT_RETURN_STATUS_IF_NOT_OK(SerializeModel(ctx, std::move(model)));

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
