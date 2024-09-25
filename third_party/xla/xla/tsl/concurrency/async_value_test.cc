/* Copyright 2022 Google LLC. All Rights Reserved.

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

#include "xla/tsl/concurrency/async_value.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/test.h"

namespace tsl {

TEST(AsyncValueTest, ConstructedToError) {
  AsyncValue* value = MakeConstructedAsyncValueRef<int32_t>(123).release();
  bool callback_triggered = false;

  EXPECT_TRUE(value->IsConstructed());
  EXPECT_FALSE(value->IsConcrete());
  EXPECT_FALSE(value->IsAvailable());

  value->AndThen([&] { callback_triggered = true; });
  EXPECT_FALSE(callback_triggered);
  value->SetError(absl::InternalError("test error"));
  EXPECT_TRUE(callback_triggered);

  EXPECT_TRUE(value->IsAvailable());
  EXPECT_FALSE(value->IsConcrete());
  EXPECT_TRUE(value->IsError());
  value->DropRef();
}

TEST(AsyncValueTest, ConstructedToConcrete) {
  AsyncValue* value = MakeConstructedAsyncValueRef<int32_t>(123).release();

  EXPECT_TRUE(value->IsConstructed());
  EXPECT_FALSE(value->IsConcrete());
  EXPECT_FALSE(value->IsAvailable());

  value->AndThen([] {});
  value->SetStateConcrete();

  EXPECT_TRUE(value->IsAvailable());
  EXPECT_TRUE(value->IsConcrete());
  EXPECT_FALSE(value->IsError());

  EXPECT_EQ(123, value->get<int32_t>());
  value->DropRef();
}

TEST(AsyncValueTest, UnconstructedEmplace) {
  AsyncValue* value = MakeUnconstructedAsyncValueRef<int32_t>().release();

  EXPECT_FALSE(value->IsConstructed());
  EXPECT_FALSE(value->IsConcrete());
  EXPECT_FALSE(value->IsAvailable());

  value->AndThen([] {});

  value->emplace<int32_t>(123);
  EXPECT_FALSE(value->IsConstructed());
  EXPECT_TRUE(value->IsAvailable());
  EXPECT_TRUE(value->IsConcrete());

  EXPECT_EQ(123, value->get<int32_t>());

  value->DropRef();
}

TEST(AsyncValueTest, AddAndDropRef) {
  AsyncValue* value = MakeConstructedAsyncValueRef<int32_t>(123).release();

  value->AndThen([] {});
  value->SetStateConcrete();

  EXPECT_TRUE(value->IsConcrete());

  EXPECT_TRUE(value->IsUnique());
  value->AddRef();
  EXPECT_FALSE(value->IsUnique());

  EXPECT_EQ(123, value->get<int32_t>());

  value->DropRef();
  EXPECT_TRUE(value->IsUnique());

  value->DropRef();
}

TEST(AsyncValueTest, KeepPayloadOnError) {
  int payload_value = 0;

  struct Payload : AsyncPayload::KeepOnError {
    explicit Payload(int* value) : value{value} { *value = 1; }
    ~Payload() { *value = 2; }

    int* value;
  };

  {
    // Test non-error case.
    AsyncValueRef<Payload> value =
        MakeConstructedAsyncValueRef<Payload>(&payload_value);

    EXPECT_EQ(1, *value->value);

    value.SetStateConcrete();

    EXPECT_EQ(1, *value->value);
    EXPECT_TRUE(!value.IsError());
  }
  EXPECT_EQ(2, payload_value);

  {
    // Test error case.
    AsyncValueRef<Payload> value =
        MakeConstructedAsyncValueRef<Payload>(&payload_value);

    EXPECT_TRUE(!value.IsError());

    value.SetError(absl::InternalError("error"));

    EXPECT_EQ(1, *value->value);
    EXPECT_TRUE(value.IsError());
    EXPECT_EQ("error", value.GetError().message());
  }

  EXPECT_EQ(2, payload_value);
}

TEST(AsyncValueTest, StackAllocatedAsyncValue) {
  int32_t counter = 0;

  class Payload {
   public:
    explicit Payload(int32_t& counter) : counter_{counter} { counter_++; }
    ~Payload() { counter_++; }

    int32_t count() const { return counter_; }

   private:
    int32_t& counter_;
  };

  // Stack allocated storage for the async value.
  internal::AsyncValueStorage<Payload> storage;

  // Construct async value in the provided storage.
  AsyncValueOwningRef<Payload> owner =
      MakeConstructedAsyncValueRef<Payload>(storage, counter);

  AsyncValuePtr<Payload> ptr = owner.AsPtr();
  AsyncValue* value = ptr.value();

  EXPECT_TRUE(value->IsConstructed());
  EXPECT_FALSE(value->IsAvailable());

  EXPECT_EQ(1, counter);
  EXPECT_EQ(1, ptr->count());

  ptr.SetStateConcrete();

  EXPECT_TRUE(ptr.IsAvailable());

  // Check that when owner is destructed it calls the payload destructor.
  std::make_unique<AsyncValueOwningRef<Payload>>(std::move(owner));
  EXPECT_EQ(2, counter);
}

}  // namespace tsl
