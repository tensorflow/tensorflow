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
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/layout_serdes.pb.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes_version.h"

namespace xla {
namespace ifrt {
namespace {

// Serialization/deserialization for `CompactLayout`.
class CompactLayoutSerDes
    : public llvm::RTTIExtends<CompactLayoutSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::CompactLayout";
  }

  absl::StatusOr<std::string> Serialize(
      const Serializable& serializable,
      std::unique_ptr<SerializeOptions> options) override {
    const SerDesVersion version = GetRequestedSerDesVersion(options.get());
    if (version.version_number() < SerDesVersionNumber(0)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Unsupported ", version.version_number(),
                       " for CompactLayout serialization"));
    }

    const auto* compact_layout = llvm::cast<CompactLayout>(&serializable);
    const auto& major_to_minor = compact_layout->major_to_minor();
    CompactLayoutProto proto;
    proto.set_version_number(SerDesVersionNumber(0).value());
    proto.mutable_major_to_minor()->Reserve(major_to_minor.size());
    proto.mutable_major_to_minor()->Add(major_to_minor.begin(),
                                        major_to_minor.end());
    return proto.SerializeAsString();
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    CompactLayoutProto proto;
    if (!proto.ParseFromString(serialized)) {
      return absl::InvalidArgumentError(
          "Failed to parse serialized CompactLayout");
    }
    const SerDesVersionNumber version_number(proto.version_number());
    if (version_number != SerDesVersionNumber(0)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Unsupported ", version_number,
                       " for CompactLayout deserialization"));
    }
    return CompactLayout::Create(proto.major_to_minor());
  }

  static char ID;  // NOLINT
};

[[maybe_unused]] char CompactLayoutSerDes::ID = 0;  // NOLINT

// clang-format off
bool register_compact_layout_serdes = ([]{
  RegisterSerDes<CompactLayout>(
      std::make_unique<CompactLayoutSerDes>());
}(), true);
// clang-format on

}  // namespace
}  // namespace ifrt
}  // namespace xla
