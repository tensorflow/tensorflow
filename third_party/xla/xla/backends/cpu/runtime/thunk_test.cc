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

#include "xla/backends/cpu/runtime/thunk.h"

#include <utility>

#include "xla/executable_run_options.h"
#include "xla/service/cpu/collectives_interface.h"
#include "xla/service/cpu/cpu_executable_run_options.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::cpu {
namespace {

TEST(ThunkTest, OkExecuteEventSingleton) {
  auto event = Thunk::OkExecuteEventSingleton();
  ASSERT_TRUE(event.IsConcrete());
}

TEST(ThunkTest, ExecuteSession) {
  Thunk::ExecuteSession session(/*max_workers=*/2, /*split_threshold=*/2);
  EXPECT_EQ(session.num_workers(), 0);

  {  // Test that destructor releases the lock.
    Thunk::ExecuteSession::Lock lock = session.Join();
    EXPECT_TRUE(lock);
    EXPECT_EQ(session.num_workers(), 1);
  }

  EXPECT_EQ(session.num_workers(), 0);

  // Test that we can join the session multiple times.
  Thunk::ExecuteSession::Lock lock0 = session.TryJoin();
  Thunk::ExecuteSession::Lock lock1 = session.TryJoin();

  EXPECT_TRUE(lock0);
  EXPECT_TRUE(lock1);

  EXPECT_EQ(session.num_workers(), 2);

  // At this point we have reached the maximum number of workers.
  Thunk::ExecuteSession::Lock lock2 = session.TryJoin();
  EXPECT_FALSE(lock2);

  EXPECT_EQ(session.num_workers(), 2);

  // Test that `Join` always returns a valid lock.
  Thunk::ExecuteSession::Lock lock3 = session.Join();
  EXPECT_TRUE(lock3);
  EXPECT_EQ(session.num_workers(), 3);

  // Test that we can move the lock and safely destroy it.
  auto sink = [](Thunk::ExecuteSession::Lock lock) {};
  sink(std::move(lock0));
  sink(std::move(lock1));
  sink(std::move(lock3));

  EXPECT_EQ(session.num_workers(), 0);

  // Test that lock is copyable.
  Thunk::ExecuteSession::Lock lock4 = session.Join();
  Thunk::ExecuteSession::Lock lock5 = lock4;

  EXPECT_TRUE(lock4);
  EXPECT_TRUE(lock5);

  EXPECT_EQ(session.num_workers(), 2);
}

TEST(ThunkTest, CollectiveExecuteParams) {
  ExecutableRunOptions run_options;
  run_options.set_device_ordinal(0);

  // Collectives interface initialized with a default implementation.
  TF_ASSERT_OK_AND_ASSIGN(auto params,
                          Thunk::CollectiveExecuteParams::Create(&run_options));
  EXPECT_NE(params.collectives, nullptr);

  // Test forwarding collectives interface from CpuExecutableRunOptions.
  CpuExecutableRunOptions cpu_run_options;
  cpu_run_options.set_collectives(
      reinterpret_cast<CollectivesInterface*>(0x12345678));
  run_options.set_cpu_executable_run_options(&cpu_run_options);

  TF_ASSERT_OK_AND_ASSIGN(params,
                          Thunk::CollectiveExecuteParams::Create(&run_options));
  EXPECT_EQ(params.collectives,
            reinterpret_cast<CollectivesInterface*>(0x12345678));
}

}  // namespace
}  // namespace xla::cpu
