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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/ifrt_ir_compile_options.pb.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/human_readable_json.h"

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
  const SerDesVersionNumber version_number(proto.version_number());
  if (version_number != SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unsupported ", version_number,
                     " for IfrtIRCompileOptions deserialization"));
  }

  if (proto.propagate_shardings()) {
    return absl::InvalidArgumentError(
        "IfrtIrCompileOptionsProto.propagate_shardings is deprecated");
  }

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
    // TODO(emilyaf): XlaCompileOptions should be built with the correct
    // devices. Pass `ifrt::Client*` to `IfrtIRCompileOptions::FromProto` and
    // look up the IFRT devices corresponding to `device_ids`.
    DeviceListRef devices = BasicDeviceList::Create({});
    compile_options_overrides->insert(
        {key, std::make_unique<XlaCompileOptions>(compile_options, devices)});
  }
  return std::make_unique<IfrtIRCompileOptions>(
      std::move(device_ids),
      absl::flat_hash_map<std::string, LoadedExecutableRef>(),
      std::move(compile_options_overrides), proto.mlir_dump_to(),
      proto.mlir_dump_pass_re(), proto.mlir_dump_func_re(),
      proto.mlir_enable_timing(), proto.dot_graph_dump_to(),
      proto.dot_graph_min_executable_peak_memory_bytes(),
      proto.dot_graph_min_executable_flops(),
      proto.dot_graph_min_per_device_transfer_size_bytes());
}

absl::Status IfrtIRCompileOptions::ToProto(IfrtIrCompileOptionsProto& proto,
                                           SerDesVersion version) const {
  if (version.version_number() < SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unsupported ", version.version_number(),
                     " for IfrtIRCompileOptions serialization"));
  }

  proto.Clear();
  proto.set_version_number(SerDesVersionNumber(0).value());
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
  proto.set_mlir_dump_to(mlir_dump_to);
  proto.set_mlir_dump_pass_re(mlir_dump_pass_re);
  proto.set_mlir_dump_func_re(mlir_dump_func_re);
  proto.set_mlir_enable_timing(mlir_enable_timing);
  proto.set_dot_graph_dump_to(dot_graph_dump_to);
  proto.set_dot_graph_min_executable_peak_memory_bytes(
      dot_graph_min_executable_peak_memory_bytes);
  proto.set_dot_graph_min_executable_flops(dot_graph_min_executable_flops);
  proto.set_dot_graph_min_per_device_transfer_size_bytes(
      dot_graph_min_per_device_transfer_size_bytes);
  return absl::OkStatus();
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              const IfrtIRCompileOptions& options) {
  absl::StatusOr<xla::ifrt::IfrtIrCompileOptionsProto> proto_or =
      options.ToProto();
  if (!proto_or.ok()) {
    os << "Failed to convert IfrtIRCompileOptions to proto: "
       << proto_or.status().ToString();
  } else {
    os << absl::StrCat(proto_or.value());
  }
  return os;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              std::shared_ptr<IfrtIRCompileOptions> options) {
  os << *options;
  return os;
}

}  // namespace ifrt
}  // namespace xla

namespace llvm::cl {

using ::xla::ifrt::IfrtIRCompileOptions;

//===----------------------------------------------------------------------===//
// IfrtIRCompileOptions
//===----------------------------------------------------------------------===//

template class basic_parser<std::shared_ptr<IfrtIRCompileOptions>>;

bool parser<std::shared_ptr<IfrtIRCompileOptions>>::parse(
    Option& opt, StringRef, StringRef arg,
    std::shared_ptr<IfrtIRCompileOptions>& value) {
  auto proto = std::make_unique<xla::ifrt::IfrtIrCompileOptionsProto>();
  absl::Status decode_json_status =
      tsl::HumanReadableJsonToProto(arg.str(), proto.get());
  if (!decode_json_status.ok()) {
    return opt.error(
        "Failed to parse IfrtIRCompileOptions from JSON "
        "string.\n\nParsing error: " +
        decode_json_status.ToString() + ".\n\n String input: " + arg);
  }
  absl::StatusOr<std::unique_ptr<IfrtIRCompileOptions>> options_or =
      IfrtIRCompileOptions::FromProto(*proto);
  if (!options_or.ok()) {
    return opt.error("Failed to create IfrtIRCompileOptions from proto: " +
                     options_or.status().ToString());
  }

  value = absl::ShareUniquePtr(std::move(*options_or));
  return false;
}

void parser<std::shared_ptr<IfrtIRCompileOptions>>::printOptionDiff(
    const Option& opt, const std::shared_ptr<IfrtIRCompileOptions>& value,
    const OptVal& defaultValue, size_t globalWidth) const {
  printOptionName(opt, globalWidth);
  outs() << "= " << value;
  if (defaultValue.hasValue()) {
    outs().indent(2) << " (default: " << defaultValue.getValue() << ")";
  }
  outs() << "\n";
}

void parser<std::shared_ptr<IfrtIRCompileOptions>>::anchor() {}

}  // namespace llvm::cl
