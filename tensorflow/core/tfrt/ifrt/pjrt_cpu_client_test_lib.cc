/* Copyright 2022 The OpenXLA Authors.

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
#include <utility>

#include "absl/status/statusor.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

const bool kUnused =
    (test_util::RegisterClientFactory(
         []() -> absl::StatusOr<std::shared_ptr<Client>> {
           xla::CpuClientOptions options;
           options.cpu_device_count = 8;
           TF_ASSIGN_OR_RETURN(auto pjrt_client,
                               xla::GetXlaPjrtCpuClient(std::move(options)));
           return std::shared_ptr<Client>(
               PjRtClient::Create(std::move(pjrt_client)));
         }),
     true);

}  // namespace
}  // namespace ifrt
}  // namespace xla
