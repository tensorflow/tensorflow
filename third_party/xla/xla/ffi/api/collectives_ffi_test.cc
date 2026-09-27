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

#include "xla/ffi/api/collectives_ffi.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/collectives_api.h"
#include "xla/ffi/api/collectives_c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/invoke.h"

namespace xla::ffi {
namespace {

using ::testing::ElementsAre;

static const XLA_FFI_Api* Api() { return GetXlaFfiApi(); }

static XLA_FFI_Communicator* const kFakeCommunicator =
    reinterpret_cast<XLA_FFI_Communicator*>(0xC0FFEE);
static XLA_FFI_Window* const kFakeWindow =
    reinterpret_cast<XLA_FFI_Window*>(0xDEC0DE);
static constexpr size_t kFakeWindowOffset = 42;

// Fake backend: implements the extension callbacks and records what the handler
// requested so the test can check the C++ wrapper builds the C args correctly.
struct FakeBackend {
  bool request_called = false;
  XLA_FFI_CollectiveGroupMode requested_group_mode =
      XLA_FFI_GROUP_CROSS_REPLICA;
  size_t requested_num_groups = 0;
  int64_t requested_communication_id = -1;
  std::vector<int64_t> requested_ids;

  bool request_window_called = false;
  XLA_FFI_CollectiveGroupMode requested_window_group_mode =
      XLA_FFI_GROUP_CROSS_REPLICA;
  size_t requested_window_num_groups = 0;
  int64_t requested_window_communication_id = -1;
  std::vector<int64_t> requested_window_ids;
  std::vector<XLA_FFI_CollectiveMemoryRegion> requested_regions;
  const void* get_window_buffer = nullptr;
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

static XLA_FFI_Error* FakeRequestWindow(
    const XLA_FFI_Collectives_Extension* self,
    XLA_FFI_Window_Request_Args* args) {
  auto* backend = reinterpret_cast<FakeBackend*>(self->state);
  backend->request_window_called = true;
  backend->requested_window_group_mode = args->group_mode;
  backend->requested_window_num_groups = args->num_groups;
  backend->requested_window_communication_id = args->communication_id;
  backend->requested_window_ids.clear();
  for (size_t i = 0; i < args->num_groups; ++i) {
    for (size_t j = 0; j < args->groups[i].size; ++j) {
      backend->requested_window_ids.push_back(args->groups[i].ids[j]);
    }
  }
  backend->requested_regions.clear();
  for (size_t i = 0; i < args->num_regions; ++i) {
    backend->requested_regions.push_back(args->regions[i]);
  }
  return nullptr;
}

static XLA_FFI_Error* FakeGetWindow(const XLA_FFI_Collectives_Extension* self,
                                    XLA_FFI_Window_Get_Args* args) {
  auto* backend = reinterpret_cast<FakeBackend*>(self->state);
  backend->get_window_buffer = args->buffer;
  args->window = kFakeWindow;
  args->window_offset = kFakeWindowOffset;
  return nullptr;
}

// Builds a collectives extension for the fake backend. Mirrors the small
// builder each backend uses to publish the extension (see the GPU backend).
static XLA_FFI_Collectives_Extension MakeFakeCollectivesExtension(
    XLA_FFI_CollectivesState* state,
    XLA_FFI_Communicator_Request* request_communicator,
    XLA_FFI_Communicator_Get* get_communicator,
    XLA_FFI_Window_Request* request_window = nullptr,
    XLA_FFI_Window_Get* get_window = nullptr) {
  XLA_FFI_Collectives_Extension ext = {};
  ext.extension_base =
      MakeExtensionHeader<internal::CollectivesExtensionBase<void>>();
  ext.state = state;
  ext.request_communicator = request_communicator;
  ext.get_communicator = get_communicator;
  ext.request_window = request_window;
  ext.get_window = get_window;
  return ext;
}

TEST(CollectivesFfiTest, RequestAndGetCommunicator) {
  bool called = false;
  XLA_FFI_Communicator* got = nullptr;

  auto handler = Ffi::Bind().Ctx<Extension<Collectives>>().To(
      [&](Communicator comm) -> Error {
        called = true;
        if (Error status = comm.RequestCommunicator(
                GroupMode::kFlattenedId, {{0, 1}}, /*communication_id=*/7);
            status.failure()) {
          return status;
        }
        ErrorOr<XLA_FFI_Communicator*> comm_or = comm.GetCommunicator(
            GroupMode::kFlattenedId, {{0, 1}}, /*communication_id=*/7);
        if (comm_or.has_error()) {
          return comm_or.error();
        }
        got = comm_or.value();
        return Error::Success();
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

TEST(CollectivesFfiTest, RequestAndGetWindow) {
  static uint32_t fake_buffer0 = 0;
  static uint32_t fake_buffer1 = 0;
  const std::vector<CollectiveMemoryRegion> regions = {
      {&fake_buffer0, sizeof(fake_buffer0)},
      {&fake_buffer1, sizeof(fake_buffer1)},
  };

  bool called = false;
  WindowLookup got_lookup = {nullptr, 0};

  auto handler = Ffi::Bind().Ctx<Extension<Collectives>>().To(
      [&](Communicator comm) -> Error {
        called = true;
        if (Error status = comm.RequestWindow(GroupMode::kFlattenedId, {{0, 1}},
                                              /*communication_id=*/9, regions);
            status.failure()) {
          return status;
        }
        ErrorOr<WindowLookup> lookup_or =
            comm.GetWindow(GroupMode::kFlattenedId, {{0, 1}},
                           /*communication_id=*/9, &fake_buffer0);
        if (lookup_or.has_error()) {
          return lookup_or.error();
        }
        got_lookup = lookup_or.value();
        return Error::Success();
      });

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  auto call_frame = builder.Build();

  FakeBackend backend;
  XLA_FFI_Collectives_Extension ext = MakeFakeCollectivesExtension(
      reinterpret_cast<XLA_FFI_CollectivesState*>(&backend),
      FakeRequestCommunicator, FakeGetCommunicator, FakeRequestWindow,
      FakeGetWindow);

  InvokeContext context;
  context.extension_start = &ext.extension_base;

  auto status =
      Invoke(Api(), *handler, call_frame, context, ExecutionStage::kExecute);

  ASSERT_OK(status);
  EXPECT_TRUE(called);
  EXPECT_TRUE(backend.request_window_called);
  EXPECT_EQ(backend.requested_window_group_mode, XLA_FFI_GROUP_FLATTENED_ID);
  EXPECT_EQ(backend.requested_window_num_groups, 1u);
  EXPECT_EQ(backend.requested_window_communication_id, 9);
  EXPECT_THAT(backend.requested_window_ids, ElementsAre(0, 1));
  ASSERT_EQ(backend.requested_regions.size(), 2u);
  EXPECT_EQ(backend.requested_regions[0].buffer, &fake_buffer0);
  EXPECT_EQ(backend.requested_regions[0].byte_size, sizeof(fake_buffer0));
  EXPECT_EQ(backend.requested_regions[1].buffer, &fake_buffer1);
  EXPECT_EQ(backend.requested_regions[1].byte_size, sizeof(fake_buffer1));
  EXPECT_EQ(backend.get_window_buffer, &fake_buffer0);
  EXPECT_EQ(got_lookup.window, kFakeWindow);
  EXPECT_EQ(got_lookup.offset, kFakeWindowOffset);
}

}  // namespace
}  // namespace xla::ffi
