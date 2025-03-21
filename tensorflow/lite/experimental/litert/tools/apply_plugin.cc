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
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/compiler/plugin/compiler_flags.h"
#include "tensorflow/lite/experimental/litert/compiler/plugin/compiler_plugin.h"
#include "tensorflow/lite/experimental/litert/core/model/model_serialize.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/experimental/litert/tools/dump.h"
#include "tensorflow/lite/experimental/litert/tools/tool_display.h"

namespace litert::tools {

using ::litert::BufferRef;
using ::litert::internal::CompilerFlags;
using ::litert::internal::CompilerPlugin;
using ::litert::internal::Dump;
using ::litert::internal::PartitionResult;
using ::litert::internal::SerializeModel;
using ::litert::internal::VerifyFlatbuffer;
using ::litert::tools::ApplyPluginRun;

#define LITERT_ENSURE_CONFIG(expr)              \
  if (!(expr)) {                                \
    return kLiteRtStatusErrorInvalidToolConfig; \
  }

namespace {

class Context {
 public:
  using Ptr = std::unique_ptr<Context>;

  explicit Context(ApplyPluginRun::Ptr run)
      : run_(std::move(run)),
        display_(ToolDisplay(std::move(run_->dump_out),
                             Context::CmdStr(run_->cmd))) {}

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

  std::ostream& Out(size_t out_ind = 0) {
    ABSL_CHECK_GE(run_->outs.size(), 1);
    return run_->outs.at(out_ind);
  }

  const CompilerFlags& Flags() const { return run_->compiler_flags; }

  OutStream SwapOut(OutStream out) {
    ABSL_CHECK_EQ(run_->outs.size(), 1);
    auto res = run_->outs.front();
    run_->outs.at(0) = out;
    return res;
  }

  uint32_t NumOuts() const { return run_->outs.size(); }

  const ApplyPluginRun& Run() const { return *run_; }
  ApplyPluginRun& Run() { return *run_; }

  ToolDisplay& Dump() { return display_; }

  static absl::string_view CmdStr(ApplyPluginRun::Cmd cmd);

 private:
  ApplyPluginRun::Ptr run_;
  ToolDisplay display_;
};

void DumpSubgraphs(ToolDisplay& display, absl::string_view label,
                   absl::Span<LiteRtSubgraph> subgraphs) {
  for (auto* subgraph : subgraphs) {
    display.Labeled();
    display.Indented() << absl::StreamFormat("(%s graph)", label);
    Dump(*subgraph, display.Display());
  }
}

void DumpCompilationRequest(ToolDisplay& display, absl::string_view soc_model,
                            size_t num_subgraphs, const CompilerFlags& flags) {
  display.Labeled() << absl::StreamFormat(
                           "Requesting compilation for target `%s` on %lu "
                           "partitions with flags: ",
                           soc_model, num_subgraphs)
                    << flags << "\n";
}

void DumpCompilationResult(ToolDisplay& display, size_t byte_code_size,
                           size_t num_entry_points) {
  display.Labeled() << absl::StreamFormat(
      "Compiled %lu partitions into %lu bytes\n", num_entry_points,
      byte_code_size);
}

void DumpModelStats(ToolDisplay& display, BufferRef<uint8_t> buf) {
  display.Labeled() << absl::StreamFormat(
      "Serialized a model of size %lu bytes\n", buf.Size());
}

void DumpPartitionResult(ToolDisplay& display, const PartitionResult& result) {
  display.Labeled() << absl::StreamFormat(
      "Partitioning yielded %lu new subgraphs\n", result.second.Size());

  DumpSubgraphs(display, "new subgraphs", result.second.Elements());
}

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

Expected<std::vector<CompilerPlugin>> LoadAllPlugins(Context& ctx) {
  ctx.Dump().Start("Load Plugins");
  ctx.Dump().Labeled() << "Loading plugins from: ";
  const auto paths = ctx.LibSearchPaths();
  for (auto it = paths.begin(); it < paths.end(); ++it) {
    ctx.Dump().Display() << *it;
    if (it < paths.end() - 1) {
      ctx.Dump().Display() << ", ";
    }
  }
  ctx.Dump().Display() << "\n";

  auto plugins = CompilerPlugin::LoadPlugins(ctx.LibSearchPaths());
  if (!plugins.HasValue()) {
    ctx.Dump().Fail();
    return plugins;
  }
  ctx.Dump().Labeled() << "Found plugins\n";
  ctx.Dump().Labeled() << absl::StreamFormat("Loaded %lu plugins\n",
                                             plugins.Value().size());

  ctx.Dump().Done();
  return plugins;
}

Expected<CompilerPlugin> LoadPlugin(Context& ctx) {
  auto plugins = LoadAllPlugins(ctx);
  if (!plugins) {
    return plugins.Error();
  }

  ctx.Dump().Start("Select Plugin");

  for (auto& plugin : *plugins) {
    if (plugin.SocManufacturer() == ctx.Run().soc_manufacturer) {
      ctx.Dump().Labeled() << absl::StreamFormat("Selected plugin for: %s\n",
                                                 plugin.SocManufacturer());
      ctx.Dump().Done();
      return std::move(plugin);
    }
  }

  ctx.Dump().Fail();
  return Unexpected(kLiteRtStatusErrorNotFound);
}

Expected<Model> LoadModel(Context& ctx) {
  ctx.Dump().Start("Load Model");
  ctx.Dump().Labeled() << absl::StreamFormat("Loading model from: %s\n",
                                             ctx.Run().model.value());
  auto model_result = Model::CreateFromFile(ctx.Run().model->data());
  if (!model_result.HasValue()) {
    ctx.Dump().Labeled() << "Failed to load model from file.";
    ctx.Dump().Fail();
    return model_result;
  }

  ctx.Dump().Labeled();
  Dump(*model_result.Value().Get(), ctx.Dump().Display());
  ctx.Dump().Done();

  return model_result;
}

//
// INFO Command
//

LiteRtStatus ValidateInfoRun(const ApplyPluginRun& run) {
  LITERT_ENSURE_CONFIG(!run.lib_search_paths.empty());
  LITERT_ENSURE_CONFIG(run.outs.size() == 1);
  return kLiteRtStatusOk;
}

LiteRtStatus Info(Context& ctx) {
  auto plugins = LoadAllPlugins(ctx);
  if (!plugins) {
    return plugins.Error().Status();
  }

  for (auto& plugin : *plugins) {
    ctx.Out() << absl::StreamFormat("< LiteRtCompilerPlugin > \"%s\" | ",
                                    plugin.SocManufacturer());
    const auto& models = plugin.SocModels();
    for (auto it = models.begin(); it < models.end(); ++it) {
      ctx.Out() << absl::StreamFormat("\"%s\"", *it);
      if (it < models.end() - 1) {
        ctx.Out() << ", ";
      }
    }
    ctx.Out() << "\n";
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

LiteRtStatus Noop(Context& ctx) {
  auto model = LoadModel(ctx);
  if (!model) {
    return model.Error().Status();
  }

  auto serialized = SerializeModel(std::move(*model->Get()));
  if (!serialized) {
    return serialized.Error().Status();
  }
  LITERT_ENSURE(VerifyFlatbuffer(serialized->Span()),
                kLiteRtStatusErrorInvalidFlatbuffer,
                "Failed to invalidate flatbuffer");
  serialized->WriteStr(ctx.Out());
  return kLiteRtStatusOk;
}

//
// PARTITION Command
//

LiteRtStatus ValidatePartitionRun(const ApplyPluginRun& run) {
  LITERT_ENSURE_CONFIG(!run.lib_search_paths.empty());
  LITERT_ENSURE_CONFIG(run.model.has_value() && !run.model.value().empty());
  LITERT_ENSURE_CONFIG(run.soc_manufacturer.has_value());
  LITERT_ENSURE_CONFIG(!run.outs.empty());
  return kLiteRtStatusOk;
}

LiteRtStatus Partition(Context& ctx) {
  auto plugin = LoadPlugin(ctx);
  if (!plugin) {
    return plugin.Error().Status();
  }

  auto model_wrap = LoadModel(ctx);
  if (!model_wrap) {
    return model_wrap.Error().Status();
  }
  auto& model = *model_wrap->Get();

  ctx.Dump().Start("Partitioning model");
  auto partition_result = PartitionModel(*plugin, model, ctx.Run().subgraphs);
  if (!partition_result) {
    return partition_result.Error().Status();
  }
  ctx.Dump().Done();
  DumpPartitionResult(ctx.Dump(), *partition_result);

  auto& new_subgraphs = partition_result->second;
  model.TransferSubgraphsFrom(std::move(new_subgraphs));

  ctx.Dump().Start("Serializing model");
  auto serialized = SerializeModel(std::move(model));
  DumpModelStats(ctx.Dump(), *serialized);
  ctx.Dump().Done();

  ctx.Dump().Start("Verifying flatbuffer");
  LITERT_ENSURE(VerifyFlatbuffer(serialized->Span()),
                kLiteRtStatusErrorInvalidFlatbuffer,
                "Failed to invalidate flatbuffer");
  ctx.Dump().Done();

  ctx.Dump().Start("Writing to out");
  serialized->WriteStr(ctx.Out());
  ctx.Dump().Done();

  return kLiteRtStatusOk;
}

//
// COMPILE Command
//

LiteRtStatus ValidateCompileRun(const ApplyPluginRun& run) {
  LITERT_ENSURE_CONFIG(!run.lib_search_paths.empty());
  LITERT_ENSURE_CONFIG(run.model.has_value());
  LITERT_ENSURE_CONFIG(run.soc_manufacturer.has_value());
  // TODO: implement multi target compilation.
  LITERT_ENSURE_SUPPORTED(run.soc_models.size() == 1,
                          "Multi target compilation not implemented.");
  return kLiteRtStatusOk;
}

LiteRtStatus Compile(Context& ctx) {
  auto model_wrap = LoadModel(ctx);
  if (!model_wrap) {
    return model_wrap.Error().Status();
  }
  auto& model = *model_wrap->Get();

  auto plugin = LoadPlugin(ctx);
  if (!plugin) {
    return plugin.Error().Status();
  }

  ctx.Dump().Start("Compiling");
  DumpCompilationRequest(ctx.Dump(), ctx.SocModelTarget(), model.NumSubgraphs(),
                         ctx.Flags());
  plugin->SetFlags(ctx.Flags());
  auto compilation_result = plugin->Compile(&model, ctx.SocModelTarget());
  if (!compilation_result) {
    ctx.Dump().Fail();
    return compilation_result.Error().Status();
  }

  auto num_byte_code = compilation_result->NumByteCodeModules();
  if (*num_byte_code < 1) {
    ctx.Dump().Fail();
    return compilation_result.Error().Status();
  }
  if (!num_byte_code) {
    ctx.Dump().Fail();
    return compilation_result.Error().Status();
  }
  for (int i = 0; i < ctx.NumOuts(); ++i) {
    auto byte_code = compilation_result->ByteCode(i);
    if (!byte_code) {
      ctx.Dump().Fail();
      return compilation_result.Error().Status();
    }
    auto num_calls = compilation_result->NumCalls();
    if (!num_calls) {
      ctx.Dump().Fail();
      return compilation_result.Error().Status();
    }

    DumpCompilationResult(ctx.Dump(), byte_code->Size(), *num_calls);
    byte_code->WriteStr(ctx.Out(i));
  }
  ctx.Dump().Done();

  return kLiteRtStatusOk;
}

//
// APPLY Command
//

LiteRtStatus ValidateApplyRun(const ApplyPluginRun& run) {
  LITERT_ENSURE_CONFIG(!run.lib_search_paths.empty());
  LITERT_ENSURE_CONFIG(run.model.has_value());
  LITERT_ENSURE_CONFIG(run.soc_manufacturer.has_value());
  LITERT_ENSURE_CONFIG(run.outs.size() == run.soc_models.size());
  // TODO: implement multi target compilation.
  LITERT_ENSURE_SUPPORTED(run.soc_models.size() == 1,
                          "Multi target compilation not implemented.");
  return kLiteRtStatusOk;
}

LiteRtStatus Apply(Context& ctx) {
  auto model_wrap = LoadModel(ctx);
  if (!model_wrap) {
    return model_wrap.Error().Status();
  }
  auto& model = *model_wrap->Get();

  auto plugin = LoadPlugin(ctx);
  if (!plugin) {
    return plugin.Error().Status();
  }

  ctx.Dump().Start("Applying plugin");
  plugin->SetFlags(ctx.Flags());
  if (auto status = litert::internal::ApplyPlugin(
          *plugin, model, ctx.SocModelTarget(), ctx.Run().subgraphs);
      !status) {
    LITERT_LOG(LITERT_ERROR, "%s", status.Error().Message().c_str());
    return status.Error().Status();
  }
  ctx.Dump().Done();

  ctx.Dump().Start("Serializing model");
  auto serialized = SerializeModel(std::move(model));
  DumpModelStats(ctx.Dump(), *serialized);
  ctx.Dump().Done();

  ctx.Dump().Start("Verifying flatbuffer");
  LITERT_ENSURE(VerifyFlatbuffer(serialized->Span()),
                kLiteRtStatusErrorInvalidFlatbuffer,
                "Failed to invalidate flatbuffer");
  ctx.Dump().Done();

  ctx.Dump().Start("Writing to out");
  serialized->WriteStr(ctx.Out());
  ctx.Dump().Done();

  return kLiteRtStatusOk;
}

}  // namespace

LiteRtStatus ApplyPlugin(ApplyPluginRun::Ptr run) {
  Context context(std::move(run));
  DumpPreamble(context.Dump());

  switch (context.Cmd()) {
    case ApplyPluginRun::Cmd::INFO:
      if (auto stat = ValidateInfoRun(context.Run()); stat != kLiteRtStatusOk) {
        context.Dump().Labeled() << "Invalid arguments for INFO command\n";
        return stat;
      }
      return Info(context);

    case ApplyPluginRun::Cmd::PARTITION:
      if (auto stat = ValidatePartitionRun(context.Run());
          stat != kLiteRtStatusOk) {
        context.Dump().Labeled() << "Invalid arguments for PARTITION command\n";
        return stat;
      }
      return Partition(context);

    case ApplyPluginRun::Cmd::COMPILE:
      if (auto stat = ValidateCompileRun(context.Run());
          stat != kLiteRtStatusOk) {
        context.Dump().Labeled() << "Invalid arguments for COMPILE command\n";
        return stat;
      }
      return Compile(context);

    case ApplyPluginRun::Cmd::APPLY:
      if (auto stat = ValidateApplyRun(context.Run());
          stat != kLiteRtStatusOk) {
        context.Dump().Labeled() << "Invalid arguments for APPLY command\n";
        return stat;
      }
      return Apply(context);

    case ApplyPluginRun::Cmd::NOOP:

      if (auto stat = ValidateNoopRun(context.Run()); stat != kLiteRtStatusOk) {
        context.Dump().Labeled() << "Invalid arguments for NOP command\n";
        return stat;
      }
      return Noop(context);

    default:
      return kLiteRtStatusErrorInvalidArgument;
  }

  return kLiteRtStatusOk;
}

}  // namespace litert::tools
