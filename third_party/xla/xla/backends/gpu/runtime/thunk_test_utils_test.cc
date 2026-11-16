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

#include "xla/backends/gpu/runtime/thunk_test_utils.h"

#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/command_buffer_requests.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/custom_options.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/execution_context.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/mock_stream_executor.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::Optional;

static_assert(!std::is_copy_constructible_v<TestThunkParams>);
static_assert(!std::is_move_constructible_v<TestThunkParams>);

TEST(ThunkTestUtilsTest, WiresOwnedStateByDefault) {
  se::MockStreamExecutor executor;
  TestThunkParams params(&executor, /*buffer_allocations=*/nullptr);
  params.mutable_run_options()->mutable_run_options()->set_run_id(RunId(3));
  params.mutable_run_options()->mutable_run_options()->set_local_device_count(
      2);

  Thunk::PrepareParams prepare = params.PrepareParams();
  EXPECT_EQ(prepare.executor, &executor);
  EXPECT_NE(prepare.buffer_allocations, nullptr);
  EXPECT_EQ(prepare.command_buffer_requests, &params.command_buffer_requests());
  EXPECT_EQ(prepare.execution_scoped_state, &params.execution_scoped_state());
  EXPECT_EQ(prepare.collective_clique_requests,
            &params.collective_clique_requests());
  EXPECT_EQ(prepare.collective_memory_requests,
            &params.collective_memory_requests());

  Thunk::InitializeParams init = params.InitializeParams();
  EXPECT_EQ(init.executor, &executor);
  EXPECT_EQ(init.buffer_allocations, nullptr);
  EXPECT_EQ(init.local_device_count, 2);
  EXPECT_EQ(init.command_buffer_requests, &params.command_buffer_requests());
  EXPECT_EQ(init.execution_scoped_state, &params.execution_scoped_state());

  Thunk::ExecuteParams execute = params.ExecuteParams();
  EXPECT_EQ(execute.buffer_allocations, prepare.buffer_allocations);
  EXPECT_EQ(execute.execution_scoped_state, &params.execution_scoped_state());
  EXPECT_EQ(execute.execution_id, 3);
}

TEST(ThunkTestUtilsTest, PropagatesOverrides) {
  se::MockStreamExecutor executor;
  se::MockStream stream, trace_stream;
  BufferAllocations allocations({}, 0, nullptr);
  CustomOptions custom_options;
  ffi::ExecutionContext ffi_ctx;
  CollectiveCliques cliques;
  CollectiveCliqueRequests clique_requests;
  CollectiveMemoryRequests memory_requests(allocations);
  CommandBufferRequests cb_requests;
  Thunk::ExecutionScopedState scoped_state;
  ExecutableRunOptions exec_run_options;
  exec_run_options.set_run_id(RunId(7));
  exec_run_options.set_local_device_count(2);
  ServiceExecutableRunOptions run_options(exec_run_options);

  TestThunkParams params;
  params.WithExecutor(&executor)
      .WithBufferAllocations(&allocations)
      .WithStream(&stream)
      .WithCommandBufferTraceStream(&trace_stream)
      .WithCollectiveCliqueRequests(&clique_requests)
      .WithCollectiveMemoryRequests(&memory_requests)
      .WithCollectiveCliques(&cliques)
      .WithCommandBufferRequests(&cb_requests)
      .WithExecutionScopedState(&scoped_state)
      .WithCustomOptions(&custom_options)
      .WithFfiExecutionContext(&ffi_ctx)
      .WithRunOptions(&run_options)
      .WithLocalDeviceCount(4)
      .WithPersistentAllocIndices({0, 2});

  const TestThunkParams& const_params = params;
  EXPECT_EQ(&const_params.command_buffer_requests(), &cb_requests);
  EXPECT_EQ(&params.collective_clique_requests(), &clique_requests);
  EXPECT_EQ(&params.collective_memory_requests(), &memory_requests);
  EXPECT_EQ(&params.execution_scoped_state(), &scoped_state);

  Thunk::PrepareParams prepare = params.PrepareParams();
  EXPECT_EQ(prepare.executor, &executor);
  EXPECT_EQ(prepare.buffer_allocations, &allocations);
  EXPECT_EQ(prepare.collective_clique_requests, &clique_requests);
  EXPECT_EQ(prepare.collective_memory_requests, &memory_requests);
  EXPECT_EQ(prepare.command_buffer_requests, &cb_requests);
  EXPECT_EQ(prepare.execution_scoped_state, &scoped_state);
  EXPECT_EQ(prepare.custom_options, &custom_options);

  Thunk::InitializeParams init = params.InitializeParams();
  EXPECT_EQ(init.executor, &executor);
  EXPECT_EQ(init.buffer_allocations, &allocations);
  EXPECT_EQ(init.stream, &stream);
  EXPECT_EQ(init.command_buffer_trace_stream, &trace_stream);
  EXPECT_EQ(init.collective_cliques, &cliques);
  EXPECT_EQ(init.command_buffer_requests, &cb_requests);
  EXPECT_EQ(init.execution_scoped_state, &scoped_state);
  EXPECT_EQ(init.custom_options, &custom_options);
  EXPECT_EQ(init.ffi_execution_context, &ffi_ctx);
  EXPECT_EQ(init.local_device_count, 4);
  EXPECT_THAT(init.persistent_alloc_indices, Optional(ElementsAre(0, 2)));

  Thunk::ExecuteParams execute = params.ExecuteParams();
  EXPECT_EQ(execute.buffer_allocations, &allocations);
  EXPECT_EQ(execute.stream, &stream);
  EXPECT_EQ(execute.command_buffer_trace_stream, &trace_stream);
  EXPECT_EQ(execute.collective_cliques, &cliques);
  EXPECT_EQ(execute.execution_scoped_state, &scoped_state);
  EXPECT_EQ(execute.custom_options, &custom_options);
  EXPECT_EQ(execute.ffi_execution_context, &ffi_ctx);
  EXPECT_EQ(execute.execution_id, 7);
  EXPECT_THAT(execute.persistent_alloc_indices, Optional(ElementsAre(0, 2)));
}

TEST(ThunkTestUtilsTest, AllowsExplicitNullptrOverrides) {
  se::MockStreamExecutor executor;
  TestThunkParams params(&executor, /*buffer_allocations=*/nullptr);
  params.WithCollectiveCliqueRequests(nullptr)
      .WithCollectiveMemoryRequests(nullptr)
      .WithExecutionScopedState(nullptr);

  Thunk::PrepareParams prepare = params.PrepareParams();
  EXPECT_EQ(prepare.collective_clique_requests, nullptr);
  EXPECT_EQ(prepare.collective_memory_requests, nullptr);
  EXPECT_EQ(prepare.execution_scoped_state, nullptr);
  EXPECT_EQ(params.InitializeParams().execution_scoped_state, nullptr);
  EXPECT_EQ(params.ExecuteParams().execution_scoped_state, nullptr);
}

TEST(ThunkTestUtilsTest, ResetsOwnedCollectiveMemoryRequestsOnAllocChange) {
  se::MockStreamExecutor executor;
  BufferAllocations alloc1({}, 0, nullptr), alloc2({}, 0, nullptr);
  TestThunkParams params(&executor, &alloc1);
  CollectiveMemoryRequests* initial_reqs = &params.collective_memory_requests();
  EXPECT_EQ(&initial_reqs->buffers(), &alloc1);

  params.WithBufferAllocations(&alloc1);
  EXPECT_EQ(&params.collective_memory_requests(), initial_reqs);

  params.WithBufferAllocations(&alloc2);
  EXPECT_EQ(&params.collective_memory_requests().buffers(), &alloc2);
}

}  // namespace
}  // namespace xla::gpu
