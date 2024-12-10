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

#include "xla/stream_executor/gpu/scoped_activate_context.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/stream_executor/gpu/mock_context.h"
#include "tsl/platform/env.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"

using testing::Return;

namespace stream_executor::gpu {
namespace {

TEST(ScopedActivateContextTest, SetsActiveOnceForSameContextWorks) {
  MockContext context;
  EXPECT_CALL(context, SetActive).Times(1);
  EXPECT_CALL(context, device_ordinal).WillRepeatedly(Return(1));
  EXPECT_CALL(context, IsActive).WillRepeatedly(Return(true));
  {
    ScopedActivateContext scoped_activate_context1(&context);
    { ScopedActivateContext scoped_activate_context2(&context); }
  }
}

TEST(ScopedActivateContextTest, TwoDifferentContextsWorks) {
  MockContext context1;
  EXPECT_CALL(context1, SetActive).Times(2);
  EXPECT_CALL(context1, device_ordinal).WillRepeatedly(Return(1));
  EXPECT_CALL(context1, IsActive).WillRepeatedly(Return(true));
  MockContext context2;
  EXPECT_CALL(context2, SetActive).Times(1);
  EXPECT_CALL(context2, device_ordinal).WillRepeatedly(Return(2));
  EXPECT_CALL(context2, IsActive).WillRepeatedly(Return(true));
  {
    ScopedActivateContext scoped_activate_context1(&context1);
    { ScopedActivateContext scoped_activate_context2(&context2); }
  }
}

TEST(ScopedActivateContextTest, TwoThreadsBothSetActiveButDontRestore) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test", 2);
  thread_pool.Schedule([&]() {
    MockContext context1;
    EXPECT_CALL(context1, SetActive).Times(1);
    EXPECT_CALL(context1, device_ordinal).WillRepeatedly(Return(1));
    EXPECT_CALL(context1, IsActive).Times(0);
    ScopedActivateContext scoped_activate_context1(&context1);
  });
  thread_pool.Schedule([&]() {
    MockContext context2;
    EXPECT_CALL(context2, SetActive).Times(1);
    EXPECT_CALL(context2, device_ordinal).WillRepeatedly(Return(1));
    EXPECT_CALL(context2, IsActive).Times(0);
    ScopedActivateContext scoped_activate_context2(&context2);
  });
}

}  // namespace
}  // namespace stream_executor::gpu
