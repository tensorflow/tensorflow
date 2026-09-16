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

#include "xla/ffi/collectives_ffi.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/collectives_api.h"
#include "xla/ffi/api/collectives_c_api.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/invoke.h"

namespace xla::ffi {
namespace {

using ::testing::ElementsAre;

static const XLA_FFI_Api* Api() { return GetXlaFfiApi(); }

static XLA_FFI_Communicator* const kFakeCommunicator =
    reinterpret_cast<XLA_FFI_Communicator*>(0xC0FFEE);

// Fake backend: implements the extension callbacks and records what the handler
// requested so the test can check the C++ wrapper builds the C args correctly.
struct FakeBackend {
  bool request_called = false;
  XLA_FFI_CollectiveGroupMode requested_group_mode =
      XLA_FFI_GROUP_CROSS_REPLICA;
  size_t requested_num_groups = 0;
  int64_t requested_communication_id = -1;
  std::vector<int64_t> requested_ids;
};

static XLA_FFI_Error* FakeRequestCommunicator(
    const XLA_FFI_Collectives_Extension* self,
    XLA_FFI_Communicator_Request_Args* args) {
  auto* backend = reinterpret_cast<FakeBackend*>(self->state);
  backend->request_called = true;
  backend->requested_group_mode = args->group_mode;
  backend->requested_num_groups = args->num_groups;
  backend->requested_communication_id = args->communication_id;
  backend->requested_ids.clear();
  for (size_t i = 0; i < args->num_groups; ++i) {
    for (size_t j = 0; j < args->groups[i].size; ++j) {
      backend->requested_ids.push_back(args->groups[i].ids[j]);
    }
  }
  return nullptr;
}

static XLA_FFI_Error* FakeGetCommunicator(
    const XLA_FFI_Collectives_Extension* self,
    XLA_FFI_Communicator_Get_Args* args) {
  args->communicator = kFakeCommunicator;
  return nullptr;
}

// Builds a collectives extension for the fake backend. Mirrors the small
// builder each backend uses to publish the extension (see the GPU backend).
static XLA_FFI_Collectives_Extension MakeFakeCollectivesExtension(
    XLA_FFI_CollectivesState* state,
    XLA_FFI_Communicator_Request* request_communicator,
    XLA_FFI_Communicator_Get* get_communicator) {
  XLA_FFI_Collectives_Extension ext;
  ext.extension_base =
      MakeExtensionHeader<internal::CollectivesExtensionBase<void>>();
  ext.state = state;
  ext.request_communicator = request_communicator;
  ext.get_communicator = get_communicator;
  return ext;
}

TEST(CollectivesFfiTest, RequestAndGetCommunicator) {
  bool called = false;
  XLA_FFI_Communicator* got = nullptr;

  auto handler = Ffi::Bind().Ctx<Extension<Collectives>>().To(
      [&](Communicator comm) -> absl::Status {
        called = true;
        if (absl::Status status = comm.RequestCommunicator(
                GroupMode::kFlattenedId, {{0, 1}}, /*communication_id=*/7);
            !status.ok()) {
          return status;
        }
        absl::StatusOr<XLA_FFI_Communicator*> comm_or = comm.GetCommunicator(
            GroupMode::kFlattenedId, {{0, 1}}, /*communication_id=*/7);
        if (!comm_or.ok()) {
          return comm_or.status();
        }
        got = *comm_or;
        return absl::OkStatus();
      });

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  auto call_frame = builder.Build();

  FakeBackend backend;
  XLA_FFI_Collectives_Extension ext = MakeFakeCollectivesExtension(
      reinterpret_cast<XLA_FFI_CollectivesState*>(&backend),
      FakeRequestCommunicator, FakeGetCommunicator);

  InvokeContext context;
  context.extension_start = &ext.extension_base;

  auto status =
      Invoke(Api(), *handler, call_frame, context, ExecutionStage::kExecute);

  ASSERT_OK(status);
  EXPECT_TRUE(called);
  EXPECT_EQ(got, kFakeCommunicator);
  EXPECT_TRUE(backend.request_called);
  EXPECT_EQ(backend.requested_group_mode, XLA_FFI_GROUP_FLATTENED_ID);
  EXPECT_EQ(backend.requested_num_groups, 1u);
  EXPECT_EQ(backend.requested_communication_id, 7);
  EXPECT_THAT(backend.requested_ids, ElementsAre(0, 1));
}

}  // namespace
}  // namespace xla::ffi
