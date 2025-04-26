/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/python/pjrt_ifrt/xla_compiler.h"

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/pjrt_ifrt/xla_compiler.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

namespace {

class XlaCompileOptionsSerDes
    : public llvm::RTTIExtends<XlaCompileOptionsSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::XlaCompileOptions";
  }

  absl::StatusOr<std::string> Serialize(
      const Serializable& serializable,
      std::unique_ptr<SerializeOptions>) override {
    const auto& options = llvm::cast<XlaCompileOptions>(serializable);

    XlaCompileOptionsProto proto;
    TF_ASSIGN_OR_RETURN(*proto.mutable_compile_options(),
                        options.compile_options.ToProto());
    if (!options.loaded_host_callbacks.empty()) {
      return absl::UnimplementedError(
          "xla::ifrt::XlaCompileOptions with loaded_host_callbacks is not "
          "serializable");
    }
    return proto.SerializeAsString();
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions>) override {
    XlaCompileOptionsProto proto;
    if (!proto.ParseFromString(serialized)) {
      return absl::DataLossError(
          "Unable to parse serialized XlaCompileOptionsProto");
    }

    auto options = std::make_unique<XlaCompileOptions>();
    TF_ASSIGN_OR_RETURN(
        options->compile_options,
        xla::CompileOptions::FromProto(proto.compile_options()));
    return options;
  }

  static char ID;  // NOLINT
};

char XlaCompileOptionsSerDes::ID = 0;  // NOLINT

bool register_xla_compile_options_serdes = ([]{
  RegisterSerDes<XlaCompileOptions>(
      std::make_unique<XlaCompileOptionsSerDes>());
}(), true);

}  // namespace

char XlaCompileOptions::ID = 0;
char XlaDeserializeExecutableOptions::ID = 0;

absl::StatusOr<std::unique_ptr<XlaCompileOptions>> GetXlaCompileOptions(
    std::unique_ptr<CompileOptions> options) {
  if (!llvm::isa<XlaCompileOptions>(options.get())) {
    return xla::InvalidArgument("options must be XlaCompileOptions");
  }
  return std::unique_ptr<XlaCompileOptions>(
      static_cast<XlaCompileOptions*>(options.release()));
}

absl::StatusOr<std::unique_ptr<XlaDeserializeExecutableOptions>>
GetXlaDeserializeExecutableOptions(
    std::unique_ptr<DeserializeExecutableOptions> options) {
  if (!llvm::isa<XlaDeserializeExecutableOptions>(options.get())) {
    return xla::InvalidArgument(
        "options must be XlaDeserializeExecutableOptions");
  }
  return std::unique_ptr<XlaDeserializeExecutableOptions>(
      static_cast<XlaDeserializeExecutableOptions*>(options.release()));
}

}  // namespace ifrt
}  // namespace xla
