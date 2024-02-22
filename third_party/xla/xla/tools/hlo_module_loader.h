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

#ifndef XLA_TOOLS_HLO_MODULE_LOADER_H_
#define XLA_TOOLS_HLO_MODULE_LOADER_H_

#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/statusor.h"
#include "xla/tools/run_hlo_module.pb.h"

namespace xla {
namespace hlo_module_loader_details {

struct Config {
  Config() {}
  int64_t num_replicas = 1;
  int64_t num_partitions = 1;
};

}  // namespace hlo_module_loader_details

// Given a string composed by multiple lines, strip the log headers, if present
// at the beginning of each line.
std::string StripLogHeaders(std::string_view hlo_string);

// Loads an HLO module from a string.
// The data can have the followings formats:
// 1) A binary of text proto file, the proto should be in xla.HloProto type. It
//    can be a binary proto (format must be "pb"), or a text proto (format must
//    be "pbtxt").
// 2) A hlo text dump, the string should be in HloModule::ToString() format
//    (format must be "txt" or "hlo"). The input data can also contain log
//    headers, which will be stripped.
// The ovr_config data can be used to override certain fields of the
// HloModuleConfig.
// The HloModuleConfig is passed to config_modifier_hook for custom
// modifications before use. If the buffer assignment proto pointer is not null
// and the hlo module format is proto, it loads buffer assignment from the
// proto.
absl::StatusOr<std::unique_ptr<HloModule>> LoadModuleFromData(
    const std::string& data, std::string_view format,
    const hlo_module_loader_details::Config& ovr_config =
        hlo_module_loader_details::Config(),
    const std::function<void(HloModuleConfig*)>& config_modifier_hook = {},
    BufferAssignmentProto* buffer_assignment_proto = nullptr);

// Loads an HLO module from file.
// The file can be one of the followings:
// 1) A binary of text proto file, the proto should be in xla.HloProto type. It
//    can be a binary proto (with .pb extension), or a text proto (with a .pbtxt
//    extension).
// 2) A hlo text dump, the string should be in HloModule::ToString() format
//    (with a .hlo or .txt extension). A text file can also contain log headers,
//    which will be stripped.
// If the format is specified (not empty), it overrides the one guessed from the
// file extension. The ovr_config data can be used to override certain fields of
// the HloModuleConfig.
// The HloModuleConfig is passed to config_modifier_hook for custom
// modifications before use. If the buffer assignment proto pointer is not null
// and the hlo module format is proto, it loads buffer assignment from the
// proto.
absl::StatusOr<std::unique_ptr<HloModule>> LoadModuleFromFile(
    const std::string& path, std::string format = "",
    const hlo_module_loader_details::Config& ovr_config =
        hlo_module_loader_details::Config(),
    const std::function<void(HloModuleConfig*)>& config_modifier_hook = {},
    BufferAssignmentProto* buffer_assignment_proto = nullptr);

// Loads an HLO snapshot from a string, only for its inputs
// The data format must be one of the following:
// 1) A binary proto (format "pb")
// 2) A text proto (format "pbtxt")
absl::StatusOr<std::unique_ptr<RunHloModuleIterationLiterals>>
LoadInputFromData(const std::string& data, std::string_view format);

// Loads an HLO snapshot from file, only for its inputs
// The file must be one of the following:
// 1) A binary proto (with .pb extension)
// 2) A text proto (with a .pbtxt extension)
// If the format is specified (not empty), it overrides the one guessed from the
// file extension.
absl::StatusOr<std::unique_ptr<RunHloModuleIterationLiterals>>
LoadInputFromFile(const std::string& path, std::string format = "");

}  // namespace xla

#endif  // XLA_TOOLS_HLO_MODULE_LOADER_H_
