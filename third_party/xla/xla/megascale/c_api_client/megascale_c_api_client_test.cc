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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/megascale/addresses.pb.h"
#include "xla/megascale/dcn_topology.pb.h"
#include "xla/megascale/megascale_runtime_error_overlay.pb.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_megascale_extension.h"
#include "xla/pjrt/c/pjrt_c_api_multi_slice_internal_types.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/plugin_names.h"
#include "xla/pjrt/plugin/xla_tpu/xla_tpu_pjrt_client.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/statusor.h"

namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;
using ::xla::GetXlaPjrtTpuClient;
using ::xla::megascale::c_api_client::CApiPjRtClientContext;
using ::xla::megascale::c_api_client::CreateAoTMegascaleConfig;
using ::xla::megascale::c_api_client::CreateDefaultMegaScaleClientContext;
using ::xla::megascale::c_api_client::CreateMegascaleErrorAggregator;
using ::xla::megascale::c_api_client::CreateMultiSliceMegascaleConfig;
using ::xla::megascale::c_api_client::DiscoverSlices;
using ::xla::megascale::c_api_client::DiscoverSlicesFromClient;
using ::xla::megascale::c_api_client::GetIncarnationId;
using ::xla::megascale::c_api_client::GetInterfaceAddressesHelper;
using ::xla::megascale::c_api_client::GetLocalHostId;
using ::xla::megascale::c_api_client::GetNumHosts;
using ::xla::megascale::c_api_client::GetOrCreateRuntimeError;
using ::xla::megascale::c_api_client::GlobalBarrier;
using ::xla::megascale::c_api_client::MegaScaleClientContextFromClient;
using ::xla::megascale::c_api_client::RegisterMegascaleErrorHandler;
using ::xla::megascale::c_api_client::UnregisterMegascaleErrorHandler;
using ::xla::megascale::runtime::DCNTopology;
using ::xla::megascale::runtime::EndpointAddresses;
using ::xla::megascale::runtime::external::MegaScaleRuntimeError;

TEST(MegaScaleCApiClientTest, CreateDefaultMegaScaleClientContext) {
  TF_ASSERT_OK_AND_ASSIGN(auto client_context,
                          CreateDefaultMegaScaleClientContext());
  EXPECT_NE(client_context, nullptr);
}

TEST(MegaScaleCApiClientTest, MegaScaleClientContextFromClient) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetXlaPjrtTpuClient());
  TF_ASSERT_OK_AND_ASSIGN(auto client_context,
                          MegaScaleClientContextFromClient(client.get()));
  EXPECT_NE(client_context, nullptr);
  ASSERT_OK(client_context->Initialize());
  EXPECT_THAT(client_context->megascale_port(), IsOkAndHolds(-1));
}

TEST(MegaScaleCApiClientTest, CreateAoTMegascaleConfig) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetXlaPjrtTpuClient());
  TF_ASSERT_OK_AND_ASSIGN(auto topology, client->GetTopologyDescription());
  TF_ASSERT_OK_AND_ASSIGN(
      auto config, CreateAoTMegascaleConfig(*topology, /*num_slices=*/2));
  EXPECT_EQ(config->NumSlices(), 2);
}

TEST(MegaScaleCApiClientTest, CreateMultiSliceMegascaleConfig) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetXlaPjrtTpuClient());
  TF_ASSERT_OK_AND_ASSIGN(auto topology, client->GetTopologyDescription());
  TF_ASSERT_OK_AND_ASSIGN(auto client_context,
                          CreateDefaultMegaScaleClientContext());
  EndpointAddresses endpoint_addresses;
  DCNTopology dcn_topology;
  TF_ASSERT_OK_AND_ASSIGN(
      auto config, CreateMultiSliceMegascaleConfig(
                       *topology, /*num_slices=*/1,
                       /*local_slice_id=*/0, /*local_host_id=*/0,
                       endpoint_addresses, dcn_topology, client_context));
  EXPECT_EQ(config->NumSlices(), 1);
  EXPECT_EQ(config->SliceId(), 0);
  EXPECT_THAT(config->NumDevicesPerSlice(), UnorderedElementsAre(Pair(0, 2)));
  EXPECT_GT(config->Serialize().size(), 0);
}

TEST(MegaScaleCApiClientTest, CreateMegascaleErrorAggregator) {
  TF_ASSERT_OK_AND_ASSIGN(auto aggregator,
                          CreateMegascaleErrorAggregator("test_app"));
  EXPECT_NE(aggregator, nullptr);
  MegaScaleRuntimeError error;
  aggregator->AddError("worker1", error);
  EXPECT_EQ(aggregator->size(), 1);
  EXPECT_TRUE(aggregator->active());
  auto digest = aggregator->ProcessAndShutdown();
  aggregator->LogErrorDigest(digest);
  EXPECT_FALSE(aggregator->active());
}

TEST(MegaScaleCApiClientTest, RegisterAndUnregisterMegascaleErrorHandler) {
  auto handler = [](const MegaScaleRuntimeError& error) {};
  EXPECT_OK(RegisterMegascaleErrorHandler("test_handler", handler));
  EXPECT_OK(UnregisterMegascaleErrorHandler("test_handler"));
}

TEST(MegaScaleCApiClientTest, GetInterfaceAddressesHelper) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto addresses,
      GetInterfaceAddressesHelper("test_port", -1, {}, true, false));
  // The exact addresses returned depend on the environment,
  // but we can assert the C API plumbing works successfully.
  EXPECT_GE(addresses.size(), 0);
}

TEST(MegaScaleCApiClientTest, GetOrCreateRuntimeError) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      GetOrCreateRuntimeError(MegaScaleRuntimeError::UNRECOVERABLE_ERROR,
                              absl::Now(), absl::OkStatus(), /*launch_id=*/0,
                              MegaScaleRuntimeError::HOST_TO_DEVICE_ERROR));

  const auto& [error, is_new] = result;
  EXPECT_EQ(error.error_type(), MegaScaleRuntimeError::UNRECOVERABLE_ERROR);
  EXPECT_EQ(error.unrecoverable_error_type(),
            MegaScaleRuntimeError::HOST_TO_DEVICE_ERROR);
  EXPECT_TRUE(is_new);
}

TEST(MegaScaleCApiClientTest, DiscoverSlicesNullContext) {
  auto result = DiscoverSlices(nullptr);
  EXPECT_TRUE(absl::IsInvalidArgument(result.status()));
}

TEST(MegaScaleCApiClientTest, DiscoverSlicesOlderOrUnimplementedExtension) {
  TF_ASSERT_OK_AND_ASSIGN(const PJRT_Api* c_api, pjrt::PjrtApi(kTpuPjrtName));
  const PJRT_Megascale_Extension* ext =
      pjrt::FindExtension<PJRT_Megascale_Extension>(
          c_api, PJRT_Extension_Type_Megascale);
  ASSERT_NE(ext, nullptr);
  PJRT_Megascale_Extension older_extension = *ext;
  older_extension.base.struct_size = sizeof(PJRT_Extension_Base);
  older_extension.discover_slices = nullptr;

  auto client_context = std::make_shared<CApiPjRtClientContext>(
      /*client_context=*/nullptr, c_api, &older_extension);

  auto result = DiscoverSlices(client_context);
  EXPECT_TRUE(absl::IsUnimplemented(result.status()));
}

TEST(MegaScaleCApiClientTest, MultiSliceConfigAccessorsAndGlobalBarrier) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetXlaPjrtTpuClient());
  TF_ASSERT_OK_AND_ASSIGN(auto topology, client->GetTopologyDescription());
  TF_ASSERT_OK_AND_ASSIGN(auto client_context,
                          CreateDefaultMegaScaleClientContext());
  EndpointAddresses endpoint_addresses;
  DCNTopology dcn_topology;
  TF_ASSERT_OK_AND_ASSIGN(
      auto config, CreateMultiSliceMegascaleConfig(
                       *topology, /*num_slices=*/1,
                       /*local_slice_id=*/0, /*local_host_id=*/0,
                       endpoint_addresses, dcn_topology, client_context));

  TF_ASSERT_OK_AND_ASSIGN(int64_t incarnation_id, GetIncarnationId(*config));
  EXPECT_EQ(incarnation_id, 0);

  TF_ASSERT_OK_AND_ASSIGN(int32_t host_id, GetLocalHostId(*config));
  EXPECT_EQ(host_id, 0);

  TF_ASSERT_OK_AND_ASSIGN(int32_t num_hosts, GetNumHosts(*config));
  EXPECT_EQ(num_hosts, 1);

  // In this unit test without a live communication backend, GlobalBarrier
  // reaches the communicator creation layer and returns an internal error
  // rather than unimplemented or invalid argument.
  EXPECT_TRUE(absl::IsInternal(
      GlobalBarrier(*config, "test_barrier", absl::Seconds(1))));
  EXPECT_TRUE(absl::IsInvalidArgument(
      GlobalBarrier(*config, "test_barrier", absl::Milliseconds(-1))));
}

class NonCApiMultiSliceConfig : public xla::MultiSliceConfig {
 public:
  int32_t NumSlices() const override { return 1; }
  int32_t SliceId() const override { return 0; }
  absl::flat_hash_map<int32_t, int32_t> NumDevicesPerSlice() const override {
    return {{0, 1}};
  }
  std::string Serialize() const override { return ""; }
};

class TrackedMultiSliceConfig : public xla::MultiSliceConfig {
 public:
  explicit TrackedMultiSliceConfig(bool* destroyed) : destroyed_(destroyed) {}
  ~TrackedMultiSliceConfig() override { *destroyed_ = true; }

  int32_t NumSlices() const override { return 4; }
  int32_t SliceId() const override { return 2; }
  absl::flat_hash_map<int32_t, int32_t> NumDevicesPerSlice() const override {
    return {{0, 8}, {1, 8}, {2, 8}, {3, 8}};
  }
  std::string Serialize() const override { return "stub_discovered_slices"; }

 private:
  bool* destroyed_;
};

TEST(MegaScaleCApiClientTest, DiscoverSlicesReturnsRefCountedConfig) {
  ASSERT_OK_AND_ASSIGN(const PJRT_Api* c_api, pjrt::PjrtApi(kTpuPjrtName));
  const PJRT_Megascale_Extension* ext =
      pjrt::FindExtension<PJRT_Megascale_Extension>(
          c_api, PJRT_Extension_Type_Megascale);
  ASSERT_NE(ext, nullptr);

  bool config_destroyed = false;
  PJRT_Megascale_Extension stub_extension = *ext;
  stub_extension.delete_client_context =
      +[](PJRT_Megascale_DeleteClientContext_Args* /*args*/) -> PJRT_Error* {
    return nullptr;
  };
  stub_extension.discover_slices =
      +[](PJRT_Megascale_DiscoverSlices_Args* args) -> PJRT_Error* {
    bool* destroyed = reinterpret_cast<bool*>(args->client_context);
    args->multi_slice_config = new PJRT_MultiSlice_Config{
        tsl::MakeRef<TrackedMultiSliceConfig>(destroyed)};
    return nullptr;
  };

  auto client_context = std::make_shared<CApiPjRtClientContext>(
      reinterpret_cast<PJRT_Megascale_ClientContext*>(&config_destroyed), c_api,
      &stub_extension);

  xla::MultiSliceConfigRef retained_copy;
  {
    ASSERT_OK_AND_ASSIGN(xla::MultiSliceConfigRef config,
                         DiscoverSlices(client_context));
    ASSERT_NE(config.get(), nullptr);
    EXPECT_EQ(config->NumSlices(), 4);
    EXPECT_EQ(config->SliceId(), 2);
    EXPECT_THAT(
        config->NumDevicesPerSlice(),
        UnorderedElementsAre(Pair(0, 8), Pair(1, 8), Pair(2, 8), Pair(3, 8)));
    EXPECT_EQ(config->Serialize(), "stub_discovered_slices");
    retained_copy = config;
  }
  EXPECT_FALSE(config_destroyed);

  retained_copy.reset();
  EXPECT_TRUE(config_destroyed);
}

TEST(MegaScaleCApiClientTest, MultiSliceConfigAccessorsInvalidConfigType) {
  tsl::RCReference<NonCApiMultiSliceConfig> invalid_config =
      tsl::MakeRef<NonCApiMultiSliceConfig>();
  EXPECT_THAT(GetIncarnationId(*invalid_config),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(GetLocalHostId(*invalid_config),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(GetNumHosts(*invalid_config),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(GlobalBarrier(*invalid_config, "test_barrier", absl::Seconds(1)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
