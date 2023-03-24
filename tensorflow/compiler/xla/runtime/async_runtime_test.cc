/*
 * Copyright 2022 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorflow/compiler/xla/runtime/async_runtime.h"

#include <cstddef>
#include <cstdint>
#include <memory>

#include "tensorflow/tsl/platform/test.h"
#include "tfrt/concurrency/async_value_ref.h"  // from @tf_runtime
#include "tfrt/concurrency/chain.h"  // from @tf_runtime

namespace xla {
namespace runtime {
constexpr int kDefaultNumOfThreads = 4;

class AsyncRuntimeTest : public ::testing::Test {
 protected:
  AsyncRuntimeTest() {
    thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
        tsl::Env::Default(), "test", kDefaultNumOfThreads);
    async_task_runner_ =
        std::make_unique<ThreadPoolAsyncTaskRunner>(thread_pool_.get());
    AsyncRuntime::Set(AsyncRuntime(async_task_runner_.get()));
  }
  std::unique_ptr<tsl::thread::ThreadPool> thread_pool_;
  std::unique_ptr<AsyncTaskRunner> async_task_runner_;
};

TEST_F(AsyncRuntimeTest, SetTokenError) {
  AsyncRuntime::Token *token = AsyncRuntime::CreateToken();
  AsyncRuntime::SetError(token);
  EXPECT_EQ(AsyncRuntime::IsError(token), true);

  AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(token));
}

TEST_F(AsyncRuntimeTest, SetValueError) {
  AsyncRuntime::Value *value =
      AsyncRuntime::CreateValue(sizeof(int32_t), alignof(std::max_align_t));
  AsyncRuntime::SetError(value);
  EXPECT_EQ(AsyncRuntime::IsError(value), true);

  AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(value));
}

TEST_F(AsyncRuntimeTest, IsGroupError) {
  AsyncRuntime::Group *group = AsyncRuntime::CreateGroup(1);
  AsyncRuntime::Token *token = AsyncRuntime::CreateToken();
  AsyncRuntime::SetError(token);
  AsyncRuntime::AddTokenToGroup(group, token);
  EXPECT_EQ(AsyncRuntime::IsError(group), true);

  AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(group));
  AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(token));
}

TEST_F(AsyncRuntimeTest, AwaitToken) {
  AsyncRuntime::Token *token = AsyncRuntime::CreateToken();
  AsyncRuntime::Value *value =
      AsyncRuntime::CreateValue(sizeof(int32_t), alignof(std::max_align_t));
  int v = 0;
  AsyncRuntime::AwaitToken(token, [&] {
    v = 42;
    AsyncRuntime::SetAvailable(value);
  });

  AsyncRuntime::SetAvailable(token);
  AsyncRuntime::AwaitValue(value);
  EXPECT_EQ(v, 42);

  AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(token));
  AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(value));
}

TEST_F(AsyncRuntimeTest, AwaitGroup) {
  AsyncRuntime::Group *group = AsyncRuntime::CreateGroup(1);
  AsyncRuntime::Token *token = AsyncRuntime::CreateToken();
  AsyncRuntime::Value *value =
      AsyncRuntime::CreateValue(sizeof(int32_t), alignof(std::max_align_t));
  AsyncRuntime::AddTokenToGroup(group, token);
  int v = 0;
  AsyncRuntime::AwaitGroup(group, [&] {
    v = 42;
    AsyncRuntime::SetAvailable(value);
  });

  AsyncRuntime::SetAvailable(token);
  AsyncRuntime::AwaitValue(value);
  EXPECT_EQ(v, 42);

  AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(group));
  AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(token));
  AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(value));
}

TEST_F(AsyncRuntimeTest, Execute) {
  auto &runtime = AsyncRuntime::GetCurrentRuntime();
  AsyncRuntime::Token *token = AsyncRuntime::CreateToken();
  int v = 0;
  runtime.Execute([&] {
    v = 42;
    AsyncRuntime::SetAvailable(token);
  });
  AsyncRuntime::AwaitToken(token);
  EXPECT_EQ(v, 42);

  AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(token));
}

TEST_F(AsyncRuntimeTest, AsToken) {
  auto chain1 = tsl::MakeAvailableAsyncValueRef<tsl::Chain>();
  auto *token1 = AsyncRuntime::AsToken(chain1);
  EXPECT_EQ(AsyncRuntime::GetAsyncValue(token1)->IsAvailable(), true);

  auto chain2 = tsl::MakeConstructedAsyncValueRef<tsl::Chain>();
  chain2.SetError("error");
  auto *token2 = AsyncRuntime::AsToken(chain2);
  EXPECT_EQ(AsyncRuntime::IsError(token2), true);

  auto chain3 = tsl::MakeConstructedAsyncValueRef<tsl::Chain>();
  auto *token3 = AsyncRuntime::AsToken(chain3);
  EXPECT_EQ(AsyncRuntime::GetAsyncValue(token3)->IsAvailable(), false);
  chain3.SetStateConcrete();
  AsyncRuntime::AwaitToken(token3);
  EXPECT_EQ(AsyncRuntime::GetAsyncValue(token3)->IsAvailable(), true);

  AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(token1));
  AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(token2));
  AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(token3));
}

TEST_F(AsyncRuntimeTest, AsValue) {
  auto async_value1 = tsl::MakeAvailableAsyncValueRef<int32_t>(42);
  auto write = [](auto *v, std::byte *store) {
    int32_t *store_t = reinterpret_cast<int32_t *>(store);
    *store_t = *v;
  };

  auto *value1 = AsyncRuntime::AsValue<int32_t>(
      async_value1, sizeof(int32_t), alignof(std::max_align_t), write);
  auto *storage1 =
      reinterpret_cast<int32_t *>(AsyncRuntime::GetStorage(value1));
  EXPECT_EQ(*storage1, 42);

  auto async_value2 = tsl::MakeConstructedAsyncValueRef<int32_t>();
  async_value2.SetError("error");
  auto *value2 = AsyncRuntime::AsValue<int32_t>(
      async_value2, sizeof(int32_t), alignof(std::max_align_t), write);
  EXPECT_EQ(AsyncRuntime::IsError(value2), true);

  auto async_value3 = tsl::MakeConstructedAsyncValueRef<int32_t>(42);
  auto *value3 = AsyncRuntime::AsValue<int32_t>(
      async_value3, sizeof(int32_t), alignof(std::max_align_t), write);
  EXPECT_EQ(AsyncRuntime::GetAsyncValue(value3)->IsAvailable(), false);
  async_value3.SetStateConcrete();
  AsyncRuntime::AwaitValue(value3);
  auto *storage3 =
      reinterpret_cast<int32_t *>(AsyncRuntime::GetStorage(value3));
  EXPECT_EQ(*storage3, 42);

  AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(value1));
  AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(value2));
  AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(value3));
}

}  // namespace runtime
}  // namespace xla
