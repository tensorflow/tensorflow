/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/megascale/c_api_client/megascale_c_api_client.h"

#include <cstdint>
#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/megascale/c_api_client/megascale_types.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_megascale_extension.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/plugin_names.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/casts.h"

namespace xla {
namespace megascale {
namespace c_api_client {
namespace {

absl::StatusOr<PJRT_Megascale_Extension*> GetMegascaleExtension(
    const PJRT_Api* c_api) {
  PJRT_Megascale_Extension* extension =
      pjrt::FindExtension<PJRT_Megascale_Extension>(
          c_api, PJRT_Extension_Type_Megascale);
  if (extension == nullptr) {
    return absl::InternalError("Megascale extension is not available.");
  }
  return extension;
}

}  // namespace

absl::StatusOr<std::unique_ptr<xla::MultiSliceConfig>> CreateAoTMegascaleConfig(
    const xla::PjRtTopologyDescription& topology_description, int num_slices) {
  TF_ASSIGN_OR_RETURN(const PJRT_Api* c_api, pjrt::PjrtApi(kTpuPjrtName));
  TF_ASSIGN_OR_RETURN(PJRT_Megascale_Extension * extension,
                      GetMegascaleExtension(c_api));

  PJRT_Megascale_CreateAoTConfig_Args args;
  args.struct_size = PJRT_Megascale_CreateAoTConfig_Args_STRUCT_SIZE;
  args.topology = tsl::down_cast<const xla::PjRtCApiTopologyDescription&>(
                      topology_description)
                      .c_topology();
  args.num_slices = num_slices;
  args.multi_slice_config = nullptr;

  RETURN_STATUS_IF_PJRT_ERROR(extension->create_aot_config(&args), c_api);

  return std::make_unique<PjRtCApiMultiSliceConfig>(args.multi_slice_config,
                                                    c_api, extension);
}

absl::StatusOr<std::unique_ptr<const xla::MultiSliceConfig>>
CreateMultiSliceMegascaleConfig(
    const xla::PjRtTopologyDescription& topology_description, int num_slices,
    int32_t local_slice_id, int32_t local_host_id,
    const xla::megascale::runtime::EndpointAddresses& endpoint_addresses,
    const xla::megascale::runtime::DCNTopology& dcn_topology,
    std::shared_ptr<CApiPjRtClientContext> megascale_client_ctx) {
  TF_ASSIGN_OR_RETURN(const PJRT_Api* c_api, pjrt::PjrtApi(kTpuPjrtName));
  TF_ASSIGN_OR_RETURN(PJRT_Megascale_Extension * extension,
                      GetMegascaleExtension(c_api));

  std::string endpoint_addresses_str = endpoint_addresses.SerializeAsString();
  std::string dcn_topology_str = dcn_topology.SerializeAsString();

  PJRT_Megascale_CreateMultiSliceConfig_Args args;
  args.struct_size = PJRT_Megascale_CreateMultiSliceConfig_Args_STRUCT_SIZE;
  args.topology = tsl::down_cast<const xla::PjRtCApiTopologyDescription&>(
                      topology_description)
                      .c_topology();
  args.num_slices = num_slices;
  args.local_slice_id = local_slice_id;
  args.local_host_id = local_host_id;
  args.endpoint_addresses = endpoint_addresses_str.data();
  args.endpoint_addresses_size = endpoint_addresses_str.size();
  args.dcn_topology = dcn_topology_str.data();
  args.dcn_topology_size = dcn_topology_str.size();
  args.client_context = megascale_client_ctx->get();
  args.multi_slice_config = nullptr;

  RETURN_STATUS_IF_PJRT_ERROR(extension->create_multi_slice_config(&args),
                              c_api);

  CHECK(args.multi_slice_config != nullptr);
  return std::make_unique<PjRtCApiMultiSliceConfig>(args.multi_slice_config,
                                                    c_api, extension);
}

absl::StatusOr<std::shared_ptr<CApiPjRtClientContext>>
MegaScaleClientContextFromClient(xla::PjRtClient* client) {
  TF_ASSIGN_OR_RETURN(const PJRT_Api* c_api, pjrt::PjrtApi(kTpuPjrtName));
  TF_ASSIGN_OR_RETURN(PJRT_Megascale_Extension * extension,
                      GetMegascaleExtension(c_api));

  PJRT_Megascale_CreateClientContextFromPjRtClient_Args args;
  args.struct_size =
      PJRT_Megascale_CreateClientContextFromPjRtClient_Args_STRUCT_SIZE;

  args.client = tsl::down_cast<xla::PjRtCApiClient*>(client)->pjrt_c_client();
  args.client_context = nullptr;

  RETURN_STATUS_IF_PJRT_ERROR(
      extension->create_client_context_from_pjrt_client(&args), c_api);

  CHECK(args.client_context != nullptr);
  return std::make_shared<CApiPjRtClientContext>(args.client_context, c_api,
                                                 extension);
}

absl::StatusOr<std::shared_ptr<CApiPjRtClientContext>>
CreateDefaultMegaScaleClientContext() {
  TF_ASSIGN_OR_RETURN(const PJRT_Api* c_api, pjrt::PjrtApi(kTpuPjrtName));
  TF_ASSIGN_OR_RETURN(PJRT_Megascale_Extension * extension,
                      GetMegascaleExtension(c_api));

  PJRT_Megascale_CreateDefaultClientContext_Args args;
  args.struct_size = PJRT_Megascale_CreateDefaultClientContext_Args_STRUCT_SIZE;
  args.client_context = nullptr;

  RETURN_STATUS_IF_PJRT_ERROR(extension->create_default_client_context(&args),
                              c_api);

  return std::make_shared<CApiPjRtClientContext>(args.client_context, c_api,
                                                 extension);
}

}  // namespace c_api_client
}  // namespace megascale
}  // namespace xla
