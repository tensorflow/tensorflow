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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/megascale/addresses.pb.h"
#include "xla/megascale/dcn_topology.pb.h"
#include "xla/pjrt/plugin/xla_tpu/xla_tpu_pjrt_client.h"
#include "xla/tsl/platform/statusor.h"

namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;
using ::xla::GetXlaPjrtTpuClient;
using ::xla::megascale::c_api_client::CreateAoTMegascaleConfig;
using ::xla::megascale::c_api_client::CreateDefaultMegaScaleClientContext;
using ::xla::megascale::c_api_client::CreateMultiSliceMegascaleConfig;
using ::xla::megascale::c_api_client::MegaScaleClientContextFromClient;
using ::xla::megascale::runtime::DCNTopology;
using ::xla::megascale::runtime::EndpointAddresses;

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

}  // namespace
