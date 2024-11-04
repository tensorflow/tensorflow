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

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/tools/dump.h"

#include <ostream>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "third_party/qairt/latest/include/QNN/QnnInterface.h"
#include "third_party/qairt/latest/include/QNN/System/QnnSystemInterface.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/qnn_manager.h"

namespace litert::qnn::internal {
namespace {

static constexpr absl::string_view kNullDumpTpl = "%s : nullptr\n";

void Dump(const QnnInterface_t* interface, std::ostream& out) {
  static constexpr absl::string_view kQnnInterfaceHeader = "< QnnInterface_t >";
  // NOLINTBEGIN
  static constexpr absl::string_view kQnnInterfaceDumpTpl =
      "\
  %s\n\
  name: %s\n\
  backend_id: %u\n\
  core_api_version: %u.%u.%u\n\
  backend_api_version: %u.%u.%u\n";
  // NOLINTEND

  if (interface == nullptr) {
    out << absl::StreamFormat(kNullDumpTpl, kQnnInterfaceHeader);
    return;
  }

  const auto core_version = interface->apiVersion.coreApiVersion;
  const auto backend_version = interface->apiVersion.backendApiVersion;

  out << absl::StreamFormat(kQnnInterfaceDumpTpl, kQnnInterfaceHeader,
                            interface->providerName, interface->backendId,
                            core_version.major, core_version.minor,
                            core_version.patch, backend_version.major,
                            backend_version.minor, backend_version.patch);
}

void Dump(const QnnSystemInterface_t* interface, std::ostream& out) {
  static constexpr absl::string_view kQnnSystemInterfaceHeader =
      "< QnnSystemInterface_t >";
  // NOLINTBEGIN
  static constexpr absl::string_view kQnnSystemInterfaceDumpTpl =
      "\
  %s\n\
  name: %s\n\
  backend_id: %u\n\
  system_api_version: %u.%u.%u\n";
  // NOLINTEND

  if (interface == nullptr) {
    out << absl::StreamFormat(kNullDumpTpl, kQnnSystemInterfaceHeader);
    return;
  }

  const auto system_version = interface->systemApiVersion;

  out << absl::StreamFormat(kQnnSystemInterfaceDumpTpl,
                            kQnnSystemInterfaceHeader, interface->providerName,
                            interface->backendId, system_version.major,
                            system_version.minor, system_version.patch);
}

}  // namespace

void Dump(const QnnManager& qnn, std::ostream& out) {
  Dump(qnn.interface_, out);
  Dump(qnn.system_interface_, out);
}
}  // namespace litert::qnn::internal
