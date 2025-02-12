/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/python/ifrt/ir/ifrt_ir_program.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/Support/Casting.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

char IfrtIRProgram::ID = 0;
char SerializeIfrtIRProgramOptions::ID = 0;
char DeserializeIfrtIRProgramOptions::ID = 0;
char IfrtIRCompileOptions::ID = 0;

absl::StatusOr<std::unique_ptr<IfrtIRCompileOptions>> GetIfrtIRCompileOptions(
    std::unique_ptr<CompileOptions> options) {
  if (!llvm::isa<IfrtIRCompileOptions>(options.get())) {
    return absl::InvalidArgumentError("options must be IfrtIRCompileOptions");
  }
  return std::unique_ptr<IfrtIRCompileOptions>(
      static_cast<IfrtIRCompileOptions*>(options.release()));
}

absl::StatusOr<std::unique_ptr<IfrtIRCompileOptions>>
IfrtIRCompileOptions::FromProto(const IfrtIrCompileOptionsProto& proto) {
  auto compile_options_overrides = std::make_unique<absl::flat_hash_map<
      std::string, std::unique_ptr<xla::ifrt::CompileOptions>>>();
  compile_options_overrides->reserve(proto.compile_option_overrides_size());

  std::vector<DeviceId> device_ids;
  device_ids.reserve(proto.device_ids_size());
  for (int64_t device_id : proto.device_ids()) {
    device_ids.push_back(DeviceId(device_id));
  }

  for (const auto& [key, value] : proto.compile_option_overrides()) {
    TF_ASSIGN_OR_RETURN(xla::CompileOptions compile_options,
                        xla::CompileOptions::FromProto(value));
    compile_options_overrides->insert(
        {key, std::make_unique<XlaCompileOptions>(compile_options)});
  }
  return std::make_unique<IfrtIRCompileOptions>(
      std::move(device_ids),
      absl::flat_hash_map<std::string, std::shared_ptr<LoadedExecutable>>(),
      std::move(compile_options_overrides), proto.propagate_shardings());
}

absl::StatusOr<IfrtIrCompileOptionsProto> IfrtIRCompileOptions::ToProto()
    const {
  IfrtIrCompileOptionsProto proto;
  proto.mutable_device_ids()->Reserve(device_assignments.size());
  for (const DeviceId& device_id : device_assignments) {
    proto.add_device_ids(device_id.value());
  }
  if (compile_options_overrides != nullptr) {
    for (const auto& [id, compile_options] : *compile_options_overrides) {
      if (!llvm::isa<XlaCompileOptions>(compile_options)) {
        return absl::InvalidArgumentError(
            "compile_options must be XlaCompileOptions");
      }

      TF_ASSIGN_OR_RETURN(
          CompileOptionsProto compile_options_proto,
          static_cast<xla::ifrt::XlaCompileOptions*>(compile_options.get())
              ->compile_options.ToProto());
      proto.mutable_compile_option_overrides()->insert(
          {id, compile_options_proto});
    }
  }
  proto.set_propagate_shardings(propagate_shardings);
  return proto;
}

}  // namespace ifrt
}  // namespace xla
