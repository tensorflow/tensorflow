/* Copyright 2019 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Emits an HLO module in a text form suitable for diffing.

#include "xla/tools/hlo_module_loader.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "re2/re2.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tools/run_hlo_module.pb.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

absl::Status OverrideConfig(const hlo_module_loader_details::Config& ovr_config,
                            HloModuleConfig* config) {
  config->set_replica_count(ovr_config.num_replicas);
  config->set_num_partitions(ovr_config.num_partitions);
  return absl::OkStatus();
}

}  // namespace

std::string StripLogHeaders(absl::string_view hlo_string) {
  // I0521 12:04:45.883483    1509 service.cc:186] ...
  static RE2* const matcher = new RE2(
      "[IWEF]\\d{4} "
      "\\d{2}:\\d{2}:\\d{2}\\.\\d+\\s+\\d+\\s+[^:]+:\\d+\\]\\s?(.*)");
  absl::string_view matches[4];
  std::vector<std::string> lines = absl::StrSplit(hlo_string, '\n');
  for (auto& line : lines) {
    if (matcher->Match(line, 0, line.size(), RE2::ANCHOR_START, matches, 4)) {
      line = std::string(matches[1]);
    }
  }
  return absl::StrJoin(lines, "\n",
                       [](std::string* out, const std::string& line) {
                         absl::StrAppend(out, line);
                       });
}

absl::StatusOr<std::unique_ptr<HloModule>> LoadModuleFromData(
    const std::string& data, absl::string_view format,
    const hlo_module_loader_details::Config& ovr_config,
    const std::function<void(HloModuleConfig*)>& config_modifier_hook,
    BufferAssignmentProto* buffer_assignment_proto, bool fill_missing_layouts) {
  DebugOptions debug_options = GetDebugOptionsFromFlags();
  std::unique_ptr<HloModule> module;
  if (format == "hlo" || format == "txt") {
    std::string hlo_string = StripLogHeaders(data);
    HloModuleConfig config;
    config.set_debug_options(debug_options);
    TF_RETURN_IF_ERROR(OverrideConfig(ovr_config, &config));
    if (config_modifier_hook) {
      config_modifier_hook(&config);
    }
    HloParserOptions options;
    options.set_fill_missing_layouts(fill_missing_layouts);
    TF_ASSIGN_OR_RETURN(
        module, ParseAndReturnUnverifiedModule(hlo_string, config, options));
  } else {
    HloSnapshot proto;
    if (format == "pb") {
      if (!proto.ParseFromString(data) &&
          !proto.mutable_hlo()->ParseFromString(data) &&
          !proto.mutable_hlo()->mutable_hlo_module()->ParseFromString(data)) {
        return InvalidArgument("Failed to parse input as HLO protobuf binary");
      }
      if (buffer_assignment_proto != nullptr) {
        if (proto.hlo().has_buffer_assignment()) {
          *buffer_assignment_proto = proto.hlo().buffer_assignment();
        } else {
          return InvalidArgument(
              "Expected buffer assignment in HLO protobuf binary.");
        }
      }
    } else if (format == "pbtxt") {
      if (!tsl::protobuf::TextFormat::ParseFromString(data, &proto) &&
          !tsl::protobuf::TextFormat::ParseFromString(data,
                                                      proto.mutable_hlo()) &&
          !tsl::protobuf::TextFormat::ParseFromString(
              data, proto.mutable_hlo()->mutable_hlo_module())) {
        return InvalidArgument("Failed to parse input as HLO protobuf text");
      }
    } else {
      return InvalidArgument(
          "Invalid format from file extension: '%s'. Expected: hlo, txt, pb, "
          "or pbtxt",
          format);
    }
    TF_ASSIGN_OR_RETURN(HloModuleConfig config,
                        HloModule::CreateModuleConfigFromProto(
                            proto.hlo().hlo_module(), debug_options));
    TF_RETURN_IF_ERROR(OverrideConfig(ovr_config, &config));
    if (config_modifier_hook) {
      config_modifier_hook(&config);
    }
    TF_ASSIGN_OR_RETURN(
        module, HloModule::CreateFromProto(proto.hlo().hlo_module(), config));
  }
  return std::move(module);
}

absl::StatusOr<std::unique_ptr<HloModule>> LoadModuleFromFile(
    const std::string& path, std::string format,
    const hlo_module_loader_details::Config& ovr_config,
    const std::function<void(HloModuleConfig*)>& config_modifier_hook,
    BufferAssignmentProto* buffer_assignment_proto, bool fill_missing_layouts) {
  std::string data;
  if (format.empty()) {
    format = std::string(tsl::io::Extension(path));
  }
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(tsl::Env::Default(), path, &data));
  return LoadModuleFromData(data, format, ovr_config, config_modifier_hook,
                            buffer_assignment_proto, fill_missing_layouts);
}

absl::StatusOr<std::unique_ptr<RunHloModuleIterationLiterals>>
LoadInputFromData(const std::string& data, absl::string_view format) {
  HloSnapshot proto;
  if (format == "pb") {
    if (!proto.ParseFromString(data) &&
        !proto.mutable_hlo()->ParseFromString(data) &&
        !proto.mutable_hlo()->mutable_hlo_module()->ParseFromString(data)) {
      return InvalidArgument("Failed to parse input as HLO protobuf binary");
    }
  } else if (format == "pbtxt") {
    if (!tsl::protobuf::TextFormat::ParseFromString(data, &proto) &&
        !tsl::protobuf::TextFormat::ParseFromString(data,
                                                    proto.mutable_hlo()) &&
        !tsl::protobuf::TextFormat::ParseFromString(
            data, proto.mutable_hlo()->mutable_hlo_module())) {
      return InvalidArgument("Failed to parse input as HLO protobuf text");
    }
  } else {
    return InvalidArgument(
        "Invalid format from file extension: '%s'. Expected: pb, "
        "or pbtxt",
        format);
  }

  auto iteration_literals_proto =
      std::make_unique<RunHloModuleIterationLiterals>();
  for (const auto& i : proto.arguments()) {
    *iteration_literals_proto->add_arguments() = i;
  }
  return std::move(iteration_literals_proto);
}

absl::StatusOr<std::unique_ptr<RunHloModuleIterationLiterals>>
LoadInputFromFile(const std::string& path, std::string format) {
  std::string data;
  if (format.empty()) {
    format = std::string(tsl::io::Extension(path));
  }
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(tsl::Env::Default(), path, &data));
  return LoadInputFromData(data, format);
}

}  // namespace xla
