/* Copyright 2025 The OpenXLA Authors.

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

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/pjrt_ifrt/executable_metadata.pb.h"
#include "xla/python/pjrt_ifrt/xla_executable_version.h"

namespace xla {
namespace ifrt {
namespace {

class XlaExecutableVersionSerDes
    : public llvm::RTTIExtends<XlaExecutableVersionSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::XlaExecutableVersion";
  }

  absl::StatusOr<std::string> Serialize(
      const Serializable& serializable,
      std::unique_ptr<SerializeOptions> options) override {
    const SerDesVersion version = GetRequestedSerDesVersion(options.get());

    const XlaExecutableVersion& executable_version =
        llvm::cast<XlaExecutableVersion>(serializable);

    absl::StatusOr<SerializedXlaExecutableVersion> executable_version_proto =
        executable_version.ToProto(version);
    if (!executable_version_proto.ok()) {
      return executable_version_proto.status();
    }

    return executable_version_proto.value().SerializeAsString();
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    SerializedXlaExecutableVersion executable_version_proto;
    if (!executable_version_proto.ParseFromString(serialized)) {
      return absl::InvalidArgumentError(
          "Failed to parse ExecutableVersionProto");
    }
    return XlaExecutableVersion::FromProto(executable_version_proto);
  }

  XlaExecutableVersionSerDes() = default;
  ~XlaExecutableVersionSerDes() override = default;

  static char ID;  // NOLINT
};

}  // namespace

[[maybe_unused]] char XlaExecutableVersionSerDes::ID = 0;

bool register_executable_version_serdes = ([]{
    RegisterSerDes<XlaExecutableVersion>(
      std::make_unique<XlaExecutableVersionSerDes>());
}(), true);

}  // namespace ifrt
}  // namespace xla
