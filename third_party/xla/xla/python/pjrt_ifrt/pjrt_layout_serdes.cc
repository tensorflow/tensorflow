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
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/pjrt_ifrt/pjrt_layout.h"
#include "xla/python/pjrt_ifrt/pjrt_layout_serdes.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace {

// Serialization/deserialization for `PjRtLayout`.
class PjRtLayoutSerDes : public llvm::RTTIExtends<PjRtLayoutSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::PjRtLayout";
  }

  absl::StatusOr<std::string> Serialize(
      const Serializable& serializable,
      std::unique_ptr<SerializeOptions> options) override {
    const SerDesVersion version = GetRequestedSerDesVersion(options.get());
    if (version.version_number() < SerDesVersionNumber(0)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Unsupported ", version.version_number(),
                       " for PjRtLayout serialization"));
    }
    const auto* pjrt_layout = llvm::cast<PjRtLayout>(&serializable);
    PjRtLayoutProto proto;
    proto.set_version_number(SerDesVersionNumber(0).value());
    // Use `xla::Layout` proto serialization, which is currently faster than
    // `xla::PjRtLayout` human-readable serialization, and reasonably stable for
    // the features used via `xla::PjRtLayout`.
    *proto.mutable_xla_layout() =
        pjrt_layout->pjrt_layout()->xla_layout().ToProto();
    return proto.SerializeAsString();
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    PjRtLayoutProto proto;
    if (!proto.ParseFromString(serialized)) {
      return absl::InvalidArgumentError(
          "Failed to parse serialized PjRtLayout");
    }
    const SerDesVersionNumber version_number(proto.version_number());
    if (version_number != SerDesVersionNumber(0)) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Unsupported ", version_number, " for PjRtLayout deserialization"));
    }
    TF_ASSIGN_OR_RETURN(auto xla_layout,
                        xla::Layout::FromProto(proto.xla_layout()));
    return PjRtLayout::Create(
        std::make_unique<xla::PjRtLayout>(std::move(xla_layout)));
  }

  static char ID;  // NOLINT
};

[[maybe_unused]] char PjRtLayoutSerDes::ID = 0;  // NOLINT

// clang-format off
bool register_pjrt_layout_serdes = ([]{
  RegisterSerDes<PjRtLayout>(
      std::make_unique<PjRtLayoutSerDes>());
}(), true);
// clang-format on

}  // namespace
}  // namespace ifrt
}  // namespace xla
