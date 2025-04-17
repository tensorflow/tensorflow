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

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/logging.h"

namespace tsl {

uint16_t AsyncValue::CreateTypeInfoAndReturnTypeIdImpl(
    const TypeInfo& type_info) {
  size_t type_id = GetTypeInfoTableSingleton()->emplace_back(type_info) + 1;
  DCHECK(type_id < std::numeric_limits<uint16_t>::max())
      << "Too many different AsyncValue types.";
  return type_id;
}

AsyncValue::TypeInfoTable* AsyncValue::GetTypeInfoTableSingleton() {
  constexpr int kInitialCapacity = 64;
  static auto* const type_info_table = new TypeInfoTable(kInitialCapacity);
  return type_info_table;
}

std::atomic<size_t> AsyncValue::total_allocated_async_values_;

// This is called when the value is set into the ConcreteAsyncValue buffer, or
// when the IndirectAsyncValue is forwarded to an available AsyncValue, and we
// need to change our state and clear out the notifications. The current state
// must be unavailable (i.e. kUnconstructed or kConstructed).
void AsyncValue::NotifyAvailable(State available_state) {
  DCHECK((kind() == Kind::kConcrete || kind() == Kind::kIndirect))
      << "Should only be used by ConcreteAsyncValue or IndirectAsyncValue";

  DCHECK(available_state == State::kConcrete ||
         available_state == State::kError);

  // Mark the value as available, ensuring that new queries for the state see
  // the value that got filled in.
  auto waiters_and_state = waiters_and_state_.exchange(
      WaitersAndState(nullptr, available_state), std::memory_order_acq_rel);
  DCHECK(waiters_and_state.state() == State::kUnconstructed ||
         waiters_and_state.state() == State::kConstructed);

  RunWaiters(waiters_and_state.waiter());
}

void AsyncValue::RunWaiters(WaiterListNode* list) {
  while (list) {
    WaiterListNode* node = list;
    (*node)();
    list = node->next;
    delete node;
  }
}

void AsyncValue::EnqueueWaiterListNode(WaiterListNode* waiter,
                                       WaitersAndState waiters_and_state) {
  // Swap the next link in. waiters_and_state.state() must be unavailable when
  // evaluating the loop condition. The acquire barrier on the compare_exchange
  // ensures that prior changes to waiter list are visible here as we may call
  // RunWaiter() on it. The release barrier ensures that prior changes to *node
  // appear to happen before it's added to the list.
  waiter->next = waiters_and_state.waiter();
  while (!waiters_and_state_.compare_exchange_weak(
      waiters_and_state, WaitersAndState(waiter, waiters_and_state.state()),
      std::memory_order_acq_rel, std::memory_order_acquire)) {
    // While swapping in our waiter, the value could have become available. If
    // so, just run the waiter.
    if (waiters_and_state.state() == State::kConcrete ||
        waiters_and_state.state() == State::kError) {
      DCHECK(waiters_and_state.waiter() == nullptr);
      (*waiter)();
      delete waiter;
      return;
    }
    // Update the waiter to point to the new head of the waiter list.
    waiter->next = waiters_and_state.waiter();
  }

  // compare_exchange_weak succeeds. The waiters_and_state must be in either
  // kUnconstructed or kConstructed state.
  DCHECK(waiters_and_state.state() == State::kUnconstructed ||
         waiters_and_state.state() == State::kConstructed);
}

void AsyncValue::SetError(absl::Status status) {
  DCHECK(!status.ok());
  if (kind() == Kind::kConcrete) {
    GetTypeInfo().set_error(this, std::move(status));
  } else {
    DCHECK(kind() == Kind::kIndirect);
    auto error_av = MakeErrorAsyncValueRef(std::move(status));
    static_cast<IndirectAsyncValue*>(this)->ForwardTo(std::move(error_av));
  }
}

// Mark this IndirectAsyncValue as forwarding to the specified value.  This
// gives the IndirectAsyncValue a +1 reference.
void IndirectAsyncValue::ForwardTo(RCReference<AsyncValue> value) {
  DCHECK(IsUnavailable());

  auto s = value->state();
  if (s == State::kConcrete || s == State::kError) {
    DCHECK(!value_) << "IndirectAsyncValue::ForwardTo is called more than once";
    auto* concrete_value = value.release();
    if (concrete_value->kind() == Kind::kIndirect) {
      auto* indirect_value = static_cast<IndirectAsyncValue*>(concrete_value);
      concrete_value = indirect_value->value_;
      DCHECK(concrete_value != nullptr);
      DCHECK(concrete_value->kind() == Kind::kConcrete);
      concrete_value->AddRef();
      indirect_value->DropRef();
    }
    // If indirect async value was created for any particular type id, check
    // that forwarded to value has exactly the same type id or an error.
    DCHECK(type_id_ == kUnknownTypeId || type_id_ == concrete_value->type_id_ ||
           concrete_value->IsType<DummyValueForErrorAsyncValue>())
        << "IndirectAsyncValue::ForwardTo value has an unexpected type id";
    value_ = concrete_value;
    type_id_ = concrete_value->type_id_;
    NotifyAvailable(s);
  } else {
    AsyncValue* av = value.get();
    av->AndThen([self = FormRef(this), value = std::move(value)]() mutable {
      self->ForwardTo(std::move(value));
    });
  }
}

//===----------------------------------------------------------------------===//
// Functions for awaiting on the async values.
//===----------------------------------------------------------------------===//

void BlockUntilReady(AsyncValue* async_value) {
  if (ABSL_PREDICT_TRUE(async_value->IsAvailable())) return;

  absl::BlockingCounter cnt(1);
  async_value->AndThen([&] { cnt.DecrementCount(); });
  cnt.Wait();
}

void RunWhenReady(absl::Span<AsyncValue* const> values,
                  absl::AnyInvocable<void()> callee) {
  // Perform a quick scan of the arguments.  If they are all available,
  // then we can run the callee synchronously.
  absl::InlinedVector<AsyncValue*, 4> unavailable_values;
  for (auto i : values) {
    if (!i->IsAvailable()) unavailable_values.push_back(i);
  }

  // If we can synchronously call 'callee', then do it and we're done.
  if (unavailable_values.empty()) return callee();

  // If there is exactly one unavailable value, then we can just AndThen it.
  if (unavailable_values.size() == 1) {
    unavailable_values[0]->AndThen(
        [callee = std::move(callee)]() mutable { callee(); });
    return;
  }

  struct CounterAndCallee {
    std::atomic<size_t> counter;
    absl::AnyInvocable<void()> callee;
  };

  // Otherwise, we have multiple unavailable values.  Put a counter on the heap
  // and have each unavailable value decrement and test it.
  auto* data =
      new CounterAndCallee{{unavailable_values.size()}, std::move(callee)};

  for (auto* val : unavailable_values) {
    val->AndThen([data]() {
      // Decrement the counter unless we're the last to be here.
      if (data->counter.fetch_sub(1) != 1) return;

      // If we are the last one, then run the callee and free the data.
      data->callee();
      delete data;
    });
  }
}

void RunWhenReady(absl::Span<RCReference<AsyncValue> const> values,
                  absl::AnyInvocable<void()> callee) {
  absl::InlinedVector<AsyncValue*, 8> pointers;
  pointers.reserve(values.size());
  for (const auto& ref : values) {
    pointers.push_back(ref.get());
  }
  RunWhenReady(pointers, std::move(callee));
}

}  // namespace tsl
