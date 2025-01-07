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

#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
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
           options.cpu_device_count = 4;
           TF_ASSIGN_OR_RETURN(auto pjrt_client,
                               xla::GetXlaPjrtCpuClient(std::move(options)));

           // Creates a client with two global processes. The local process acts
           // as task 1, and any attempt to use non-addressable devices on task
           // 0 will fail.
           PjRtClient::CreateOptions pjrt_client_options;
           pjrt_client_options.pjrt_client = std::move(pjrt_client);

           PjRtClient::CreateOptions::GlobalDeviceMapping&
               global_device_mapping =
                   pjrt_client_options.global_device_mapping.emplace();
           global_device_mapping.addressable_device_ids = {
               DeviceId(4), DeviceId(5), DeviceId(6), DeviceId(7)};
           global_device_mapping.device_id_to_process_index = {
               {DeviceId(0), 0}, {DeviceId(1), 0},
               {DeviceId(2), 0}, {DeviceId(3), 0},  //
               {DeviceId(4), 1}, {DeviceId(5), 1},
               {DeviceId(6), 1}, {DeviceId(7), 1},  //
           };
           return PjRtClient::Create(std::move(pjrt_client_options));
         }),
     true);

}  // namespace
}  // namespace ifrt
}  // namespace xla
