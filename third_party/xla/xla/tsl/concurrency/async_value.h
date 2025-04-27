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

#ifndef XLA_TSL_CONCURRENCY_ASYNC_VALUE_H_
#define XLA_TSL_CONCURRENCY_ASYNC_VALUE_H_

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/concurrent_vector.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/logging.h"

namespace tsl {
namespace internal {

template <typename T>
class ConcreteAsyncValue;

template <typename T>
constexpr bool kMaybeBase = std::is_class<T>::value && !std::is_final<T>::value;

}  // namespace internal

// This is a future of the specified value type. Arbitrary C++ types may be used
// here, even non-copyable types and expensive ones like tensors.
//
// An AsyncValue is in one of two states: unavailable or available. If it is in
// the unavailable state, it may have a list of waiters which are notified when
// the value transitions to another state.
//
// The actual payload data is stored in the templated subclass
// ConcreteAsyncValue. This achieves good cache efficiency by storing the meta
// data and the payload data in consecutive memory locations.
class AsyncValue {
 public:
  class Executor;

  ~AsyncValue();

  // Return true if state is kUnconstructed.
  bool IsUnconstructed() const;

  // Return true if state is kConstructed.
  bool IsConstructed() const;

  // Return true if state is kConcrete.
  bool IsConcrete() const;

  // Return true if state is kError.
  bool IsError() const;

  // Return true if this async value is resolved to a concrete value or error.
  bool IsAvailable() const;
  bool IsUnavailable() const { return !IsAvailable(); }

  // Return true if this is an IndirectAsyncValue that hasn't been resolved.
  // Currently an IndirectAsyncValue is available if and only if it is resolved.
  bool IsUnresolvedIndirect() const;

  // Return true if this is an IndirectAsyncValue.
  bool IsIndirect() const;

  // Return reference count. This should be used for testing and debugging only.
  uint32_t NumRef() const { return refcount_.load(std::memory_order_acquire); }

  // Return true if reference count is 1.
  bool IsUnique() const;

  // Add a new reference to this object.
  //
  // Return this object for convenience. e.g. when a call-site wants to feed an
  // AsyncValue object into a function call that takes the object at +1, we can
  // write: foo(av->AddRef());
  AsyncValue* AddRef() { return AddRef(1); }
  AsyncValue* AddRef(uint32_t count);

  // Drop a reference to this object, potentially deallocating it.
  void DropRef() { DropRef(1); }
  void DropRef(uint32_t count);

  // Return the stored value as type T. For the consumer of the AsyncValue, this
  // requires that the state be `kConcrete`. For the producer of the AsyncValue,
  // the state may also be `constructed`, as long as the producer can ensure
  // race-free access to the data (e.g. no concurrent writes and reads and no
  // concurrent changing the state to `kError). For both cases, T must be the
  // exact type or a base type of the payload type of this AsyncValue. When T is
  // a base type of the payload type, the following additional conditions are
  // required:
  // 1) Both the payload type and T are polymorphic (have virtual function) or
  //    neither are.
  // 2) The payload type does not use multiple inheritance.
  // The above conditions are required since we store only the offset of the
  // payload type in AsyncValue as data_traits_.buf_offset. Violation of either
  // 1) or 2) requires additional pointer adjustments to get the proper pointer
  // for the base type, which we do not have sufficient information to perform
  // at runtime.
  template <typename T>
  const T& get() const;

  // Same as the const overload of get(), except for returning a non-const ref.
  template <typename T>
  T& get();

  // Returns the underlying error. IsError() must be true.
  const absl::Status& GetError() const;

  // Returns the underlying error, or nullptr if there is none.
  const absl::Status* GetErrorIfPresent() const;

  template <typename T>
  bool IsType() const {
    return GetTypeId<T>() == type_id_;
  }

  // Change state to kConcrete. Requires that this AsyncValue
  // previously have state constructed.
  void SetStateConcrete();

  // Construct the payload of the AsyncValue in place and change its state to
  // kConcrete. Requires that this AsyncValue previously have state
  // kUnconstructed or kConstructed.
  template <typename T, typename... Args>
  void emplace(Args&&... args);

  void SetError(absl::Status status);

  // If the value is available or becomes available, this invokes the waiter
  // immediately. Otherwise, adds the waiter to the waiter list and calls it
  // when the value becomes available.
  template <typename Waiter>
  void AndThen(Waiter&& waiter);

  // Same as above, but waiter will be invoked on the given executor.
  template <typename Waiter>
  void AndThen(Executor& executor, Waiter&& waiter);

  // Return the total number of async values that are currently live in the
  // process. This is intended for debugging/assertions only, and shouldn't be
  // used for mainline logic in the runtime.
  static size_t GetNumAsyncValueInstances() {
    DCHECK(AsyncValueAllocationTrackingEnabled())
        << "AsyncValue instance tracking disabled!";
    return total_allocated_async_values_.load(std::memory_order_relaxed);
  }

  // Returns true if we track the number of alive AsyncValue instances in
  // total_allocated_async_values_.
  static bool AsyncValueAllocationTrackingEnabled() {
    // For now we track the number of alive AsyncValue instances only in debug
    // builds.
#ifdef NDEBUG
    return false;
#else
    return true;
#endif
  }

  // What sort of AsyncValue this is.
  //
  // We make this an unsigned type so that loading the enum from the bitfield
  // does not sign extend.
  enum class Kind : uint8_t {
    kConcrete = 0,  // ConcreteAsyncValue
    kIndirect = 1,  // IndirectAsyncValue
  };

  // Return the kind of this AsyncValue.
  Kind kind() const { return kind_; }

  class State {
   public:
    // The state of AsyncValue.
    enum StateEnum : int8_t {
      // The underlying value's constructor is not called and the value is not
      // available for consumption. This state can transition to kConstructed,
      // kConcrete and kError.
      kUnconstructed = 0,
      // The underlying value's constructor is called but the value is not
      // available for consumption. This state can transition to
      // kConcrete and kError.
      kConstructed = 1,
      // The underlying value is available for consumption. This state can not
      // transition to any other state.
      kConcrete = 2,
      // This AsyncValue is available and contains an error. This state can not
      // transition to any other state.
      kError = 3,
    };

    State(StateEnum s) : state_(s) {}  // NOLINT

    operator StateEnum() { return state_; }  // NOLINT

    // Return true if state is kUnconstructed.
    bool IsUnconstructed() const { return state_ == kUnconstructed; }

    // Return true if state is kConstructed.
    bool IsConstructed() const { return state_ == kConstructed; }

    // Return true if state is kConcrete.
    bool IsConcrete() const { return state_ == kConcrete; }

    // Return true if state is kError.
    bool IsError() const { return state_ == kError; }

    // Return true if this async value is resolved to a concrete value or error.
    bool IsAvailable() const { return state_ == kConcrete || state_ == kError; }
    bool IsUnavailable() const { return !IsAvailable(); }

    const char* DebugString() const {
      switch (state_) {
        case kUnconstructed:
          return "kUnconstructed";
        case kConstructed:
          return "kConstructed";
        case kConcrete:
          return "kConcrete";
        case kError:
          return "kError";
      }
    }

   private:
    StateEnum state_;
  };

  // Return which state this AsyncValue is in.
  State state() const {
    return waiters_and_state_.load(std::memory_order_acquire).state();
  }

  // AsyncValue executor allows to customize where the waiter callback is
  // executed. By default the waiter callback is executed on the caller thread
  // if async value is already available, or on a thread that sets async value
  // available (emplacing a value or setting an error), which can accidentally
  // lead to executing a very expensive computations on a low-latency thread.
  //
  // IMPORTANT: It's the caller responsibility to ensure that executor passed to
  // all `AndThen` or `Map` function calls stay alive while async values have
  // unresolved waiters waiting to be invoked.
  class Executor {
   public:
    using Task = absl::AnyInvocable<void()>;

    virtual ~Executor() = default;

    virtual void Execute(Task task) = 0;
  };

 protected:
  friend class IndirectAsyncValue;

  struct WaiterListNode;

  static constexpr uint16_t kUnknownTypeId = 0;

  // Utility template for tag dispatching.
  template <typename T>
  struct TypeTag {};

  template <typename T>
  AsyncValue(Kind kind, State state, bool is_refcounted, TypeTag<T>)
      : refcount_(1),
        kind_(kind),
        has_vtable_(std::is_polymorphic<T>()),
        is_refcounted_(is_refcounted),
        type_id_(GetTypeId<T>()),
        waiters_and_state_(WaitersAndState(nullptr, state)) {
    if (AsyncValueAllocationTrackingEnabled() && is_refcounted)
      total_allocated_async_values_.fetch_add(1, std::memory_order_relaxed);
  }

  AsyncValue(Kind kind, State state, bool is_refcounted)
      : refcount_(1),
        kind_(kind),
        has_vtable_(false),
        is_refcounted_(is_refcounted),
        type_id_(kUnknownTypeId),
        waiters_and_state_(WaitersAndState(nullptr, state)) {
    if (AsyncValueAllocationTrackingEnabled() && is_refcounted)
      total_allocated_async_values_.fetch_add(1, std::memory_order_relaxed);
  }

  AsyncValue(const AsyncValue&) = delete;
  AsyncValue& operator=(const AsyncValue&) = delete;

  void NotifyAvailable(State available_state);
  void Destroy();
  void RunWaiters(WaiterListNode* list);

  // IsTypeIdCompatible returns true if the type value stored in this AsyncValue
  // instance can be safely cast to `T`. This is a conservative check. I.e.
  // IsTypeIdCompatible may return true even if the value cannot be safely cast
  // to `T`. However, if it returns false then the value definitely cannot be
  // safely cast to `T`. This means it is useful mainly as a debugging aid for
  // use in assert() etc.

  template <typename T, std::enable_if_t<internal::kMaybeBase<T>>* = nullptr>
  bool IsTypeIdCompatible() const {
    // We can't do a GetTypeId<T> in this case because `T` might be an abstract
    // class.  So we conservatively return true.
    return true;
  }

  template <typename T, std::enable_if_t<!internal::kMaybeBase<T>>* = nullptr>
  bool IsTypeIdCompatible() const {
    return GetTypeId<T>() == type_id_;
  }

  // Return the ID of the given type. Note that at most 2^16-2 (approx. 64K)
  // unique types can be used in AsyncValues, since the ID is 16 bits, and 0 and
  // 2^16-1 are not allowed to be used as type IDs.
  template <typename T>
  static uint16_t GetTypeId() {
    return internal::ConcreteAsyncValue<T>::concrete_type_id_;
  }

  // Creates a AsyncValue::TypeInfo object for `T` and store it in the global
  // TypeInfo table. Returns the "type id" for `T` which currently happens to
  // be one plus the index of this TypeInfo object in the TypeInfo table.
  //
  // This should only be called from the initializer for the static
  // ConcreteAsyncValue concrete_type_id_ field.
  template <typename T>
  static uint16_t CreateTypeInfoAndReturnTypeId() {
    return CreateTypeInfoAndReturnTypeIdImpl(
        MakeTypeInfo<internal::ConcreteAsyncValue<T>>());
  }

  std::atomic<uint32_t> refcount_{1};

  Kind kind_ : 2;
  // has_vtable_ has the same value for a given payload type T. If we want to
  // use the unused bits here for other purpose in the future, we can move
  // has_vtable_ to a global vector<bool> indexed by type_id_.
  const bool has_vtable_ : 1;

  // When is_refcounted_ is false, `AddRef` and `DropRef` have no effect in
  // optimized builds. We always do reference counting in debug builds to verify
  // that async values used correctly and we do not have accidental dangling
  // references.
  const bool is_refcounted_ : 1;

  // This is a 16-bit value that identifies the type.
  uint16_t type_id_ = 0;

  // This is a singly linked list of nodes waiting for notification, hanging off
  // of AsyncValue. When the value becomes available or if an error occurs, the
  // callbacks are informed.
  struct WaiterListNode {
    virtual ~WaiterListNode() = default;
    virtual void operator()() = 0;

    WaiterListNode* next = nullptr;
  };

  // The waiter list and the state are compacted into one single atomic word as
  // accesses to them are tightly related. To change the state from unavailable
  // (i.e. kUnconstructed or kConstructed) to available
  // (i.e. kConcrete or kError), we also need to empty the waiter
  // list. To add a node to the waiter list, we want to make sure the state is
  // unavailable, otherwise we could run the new node immediately.
  //
  // Invariant: If the state is not available, then the waiter list must be
  // nullptr.
  struct WaitersAndState {
    // We rely on the fact that all `WaiterListNode` values are aligned at
    // least to 4 bytes and we can encode state in the lowest 2 bits. We use
    // the conservative estimation of the minimal alignment of pointers returned
    // from memory allocation functions.
    //
    // See: https://en.cppreference.com/w/cpp/types/max_align_t
    static_assert(alignof(std::max_align_t) >= 2);

    static constexpr uintptr_t kStateMask = (1ull << 2) - 1;
    static constexpr uintptr_t kPointerMask = ~kStateMask;

    WaitersAndState(WaiterListNode* ptr, State state) {
      value = (reinterpret_cast<uintptr_t>(ptr) & kPointerMask) |
              (state & kStateMask);
    }

    State state() const {
      return State(static_cast<State::StateEnum>(value & kStateMask));
    }

    WaiterListNode* waiter() const {
      return reinterpret_cast<WaiterListNode*>(value & kPointerMask);
    }

    uintptr_t value;
  };

  std::atomic<WaitersAndState> waiters_and_state_;

  // We assume (and static_assert) that this is the offset of ConcreteAsyncValue
  // data payload so that we can always get a pointer to the start of payload
  // from an async value pointer. We use alignas attribute to guarantee that the
  // data payload stored at exactly this offset. It means that types that have
  // larger alignment requirement are not compatible with AsyncValues.
  static constexpr int kDataOffset = 64;

 private:
  // Information about a ConcreteAsyncValue<T> subclass.
  struct TypeInfo {
    // Destructor returns the size and alignment of the derived AsyncValue to
    // be deallocated.
    using DestructorFn = std::pair<size_t, std::align_val_t> (*)(AsyncValue*);
    using GetErrorFn = const absl::Status& (*)(const AsyncValue*);
    using SetErrorFn = void (*)(AsyncValue*, absl::Status);
    using HasDataFn = bool (*)(const AsyncValue*);

    DestructorFn destructor;
    GetErrorFn get_error;
    SetErrorFn set_error;
    HasDataFn has_data;
  };

  template <typename Derived>
  static TypeInfo MakeTypeInfo() {
    return TypeInfo{
        [](AsyncValue* v) -> std::pair<size_t, std::align_val_t> {
          static_cast<Derived*>(v)->~Derived();
          return {sizeof(Derived), std::align_val_t{alignof(Derived)}};
        },
        [](const AsyncValue* v) -> const absl::Status& {
          return static_cast<const Derived*>(v)->GetError();
        },
        [](AsyncValue* v, absl::Status status) {
          static_cast<Derived*>(v)->SetError(std::move(status));
        },
        [](const AsyncValue* v) {
          return static_cast<const Derived*>(v)->HasData();
        },
    };
  }

  static uint16_t CreateTypeInfoAndReturnTypeIdImpl(const TypeInfo& type_info);

  template <typename T>
  const T& GetConcreteValue() const;

  // Returns the TypeInfoTable instance (there is one per process).
  using TypeInfoTable = internal::ConcurrentVector<TypeInfo>;
  static TypeInfoTable* GetTypeInfoTableSingleton();

  // Get the TypeInfo instance for this AsyncValue.
  const TypeInfo& GetTypeInfo() const {
    TypeInfoTable* type_info_table = AsyncValue::GetTypeInfoTableSingleton();
    DCHECK_NE(type_id_, 0) << "TypeId must be set";
    return (*type_info_table)[type_id_ - 1];
  }

  // Adds a waiter list node to the waiter linked list. If the value is
  // available or becomes available, this calls the waiter immediately.
  // Otherwise, we add waiter to the list where it will be called when the value
  // becomes available.
  void EnqueueWaiterListNode(WaiterListNode* waiter,
                             WaitersAndState waiters_and_state);

  template <typename Waiter>
  void EnqueueWaiter(Waiter&& waiter, WaitersAndState waiters_and_state) {
    static_assert(std::is_invocable_v<Waiter>, "Waiter must be invocable");

    struct Node final : public WaiterListNode {
      explicit Node(Waiter waiter) : waiter(std::move(waiter)) {}
      void operator()() final { std::move(waiter)(); }
      Waiter waiter;
    };

    EnqueueWaiterListNode(new Node{std::forward<Waiter>(waiter)},
                          waiters_and_state);
  }

  // This is a global counter of the number of AsyncValue instances currently
  // live in the process.  This is intended to be used for debugging only, and
  // is only kept in sync if AsyncValueAllocationTrackingEnabled() returns
  // true.
  static std::atomic<size_t> total_allocated_async_values_;
};

// We only optimize the code for 64-bit architectures for now.
static_assert(sizeof(AsyncValue) == 16 || sizeof(void*) != 8,
              "Unexpected size for AsyncValue");

//===----------------------------------------------------------------------===//
// Functions for awaiting on the async values.
//===----------------------------------------------------------------------===//

// Blocks the caller thread until the async value becomes available.
void BlockUntilReady(AsyncValue* async_value);

// Runs the `callee` when all async values become available.
void RunWhenReady(absl::Span<AsyncValue* const> values,
                  absl::AnyInvocable<void() &&> callee);
void RunWhenReady(absl::Span<RCReference<AsyncValue> const> values,
                  absl::AnyInvocable<void() &&> callee);

//===----------------------------------------------------------------------===//

// Traits for customizing AsyncValue behavior for different payload types.
struct AsyncPayload {
  // Under the normal behavior, if an AsyncValue is in kConstructed state (i.e.
  // the payload data is constructed), it will destruct the payload data when
  // the AsyncValue enters the error state (e.g., on AsyncValue::SetError()).
  //
  // However, for the payload types that inherit from `KeepOnError`, AsyncValue
  // exhibits a different behavior: the payload value if constructed, will be
  // kept valid when the AsyncValue goes into the error state.
  struct KeepOnError {};
};

namespace internal {

// Subclass for storing the concrete payload of the AsyncValue.
//
// Async value itself is a container that either holds `absl::Status` (in error
// state) or a concrete value of type `T` (in concrete state). Async value that
// holds an `absl::Status` or `absl::StatusOr<T>` is typically a bad idea, and
// should be expressed as a plain async value of type `T`.
template <typename T>
class ConcreteAsyncValue : public AsyncValue {
 public:
  // Tag type for making a ConcreteAsyncValue without calling underlying value's
  // constructor.
  struct UnconstructedPayload {
    bool is_refcounted = true;
  };

  // Tag type for making a ConcreteAsyncValue with the underlying value
  // constructed but not available for consumption.
  struct ConstructedPayload {
    bool is_refcounted = true;
  };

  // Tag type for making a ConcreteAsyncValue with the underlying value
  // constructed and available for consumption.
  struct ConcretePayload {
    bool is_refcounted = true;
  };

  // Make a ConcreteAsyncValue with kUnconstructed state.
  explicit ConcreteAsyncValue(UnconstructedPayload payload)
      : AsyncValue(Kind::kConcrete, State::kUnconstructed,
                   payload.is_refcounted, TypeTag<T>()) {
    VerifyOffsets();
  }

  // Make a ConcreteAsyncValue with kError state.
  explicit ConcreteAsyncValue(absl::Status status)
      : AsyncValue(Kind::kConcrete, State::kError,
                   /*is_refcounted=*/true, TypeTag<T>()),
        data_store_{std::move(status)} {
    VerifyOffsets();
  }

  // Make a ConcreteAsyncValue with kConstructed state.
  template <typename... Args>
  explicit ConcreteAsyncValue(ConstructedPayload payload, Args&&... args)
      : AsyncValue(Kind::kConcrete, State::kConstructed, payload.is_refcounted,
                   TypeTag<T>()),
        data_store_{TypeTag<T>(), std::forward<Args>(args)...} {
    VerifyOffsets();
  }

  // Make a ConcreteAsyncValue with kConcrete state.
  template <typename... Args>
  explicit ConcreteAsyncValue(ConcretePayload payload, Args&&... args)
      : AsyncValue(Kind::kConcrete, State::kConcrete, payload.is_refcounted,
                   TypeTag<T>()),
        data_store_{TypeTag<T>(), std::forward<Args>(args)...} {
    VerifyOffsets();
  }

  ~ConcreteAsyncValue() { Destroy(); }

  // Return the underlying error. IsError() must return true.
  const absl::Status& GetError() const {
    DCHECK(IsError());
    return data_store_.error();
  }

  void SetError(absl::Status status) {
    data_store_.SetError(state(), std::move(status));
    NotifyAvailable(State::kError);
  }

  const T& get() const {
    DCHECK(HasData());
    return data_store_.data();
  }

  T& get() {
    DCHECK(HasData());
    return data_store_.data();
  }

  // Construct the payload of the AsyncValue in place and change its state to
  // kConcrete. Requires that this AsyncValue previously have state
  // unavailable.
  template <typename... Args>
  void emplace(Args&&... args) {
    data_store_.EmplaceData(std::forward<Args>(args)...);
    NotifyAvailable(State::kConcrete);
  }

  static bool classof(const AsyncValue* v) {
    return v->kind() == AsyncValue::Kind::kConcrete;
  }

 private:
  friend class AsyncValue;

  // Data and error layout when the payload does not inherit from
  // AsyncPayload::KeepOnError. This type destructs the payload value on
  // error. It never keeps both data and error alive at the same time.
  class DataOrError {
   public:
    DataOrError() {}

    explicit DataOrError(absl::Status status)
        : error_{new absl::Status(std::move(status))} {}

    template <typename... Args>
    explicit DataOrError(TypeTag<T>, Args&&... args)
        : data_{std::forward<Args>(args)...} {}

    ~DataOrError() {}

    void Destroy(State s) {
      if (s == State::kError) {
        delete error_;
      } else if (s == State::kConstructed || s == State::kConcrete) {
        data_.~T();
      }
    }

    void SetError(State s, absl::Status status) {
      DCHECK(s == State::kUnconstructed || s == State::kConstructed);
      if (s == State::kConstructed) {
        data_.~T();
      }
      error_ = new absl::Status(std::move(status));
    }

    template <typename... Args>
    void EmplaceData(Args&&... args) {
      new (&data_) T(std::forward<Args>(args)...);
    }

    bool HasData(State s) const {
      return s == State::kConstructed || s == State::kConcrete;
    }

    absl::Status& error() { return *error_; }
    T& data() { return data_; }
    const absl::Status& error() const { return *error_; }
    const T& data() const { return data_; }

   private:
    friend class ConcreteAsyncValue;
    union {
      absl::Status* error_;
      T data_;
    };
  };

  // Data and error layout when the payload inherits from
  // AsyncPayload::KeepOnError. This type does to destruct the payload value
  // on error. It may keep both data and error alive at the same time.
  class DataAndError {
   public:
    explicit DataAndError(absl::Status status) { SetError(std::move(status)); }

    template <typename... Args>
    explicit DataAndError(TypeTag<T>, Args&&... args) {
      EmplaceData(std::forward<Args>(args)...);
    }

    void Destroy(State s) {
      if (HasData()) data().~T();
      error_.reset();
      has_data_ = false;
    }

    void SetError(State s, absl::Status status) {
      DCHECK(!error_);
      error_ = std::make_unique<absl::Status>(std::move(status));
    }

    template <typename... Args>
    void EmplaceData(Args&&... args) {
      DCHECK(!HasData());
      new (&data_) T(std::forward<Args>(args)...);
      has_data_ = true;
    }

    T& data() { return *reinterpret_cast<T*>(&data_); }
    const T& data() const { return *reinterpret_cast<const T*>(&data_); }

    bool HasData(State s) const { return has_data_; }
    bool HasData() const { return has_data_; }
    const absl::Status& error() const { return *error_; }
    absl::Status& error() { return *error_; }

   private:
    friend class ConcreteAsyncValue;

    alignas(T) std::byte data_[sizeof(T)];
    bool has_data_ = false;
    std::unique_ptr<absl::Status> error_;
  };

  using DataStoreT =
      std::conditional_t<std::is_base_of_v<AsyncPayload::KeepOnError, T>,
                         DataAndError, DataOrError>;
  alignas(AsyncValue::kDataOffset) DataStoreT data_store_;

  void Destroy() { data_store_.Destroy(state()); }
  bool HasData() const { return data_store_.HasData(state()); }

  static void VerifyOffsets() {
    static_assert(offsetof(ConcreteAsyncValue<T>, data_store_.data_) ==
                      AsyncValue::kDataOffset,
                  "Offset of ConcreteAsyncValue data payload is assumed to be "
                  "AsyncValue::kDataOffset == 64");
  }

  static const uint16_t concrete_type_id_;
};

template <typename T>
const uint16_t ConcreteAsyncValue<T>::concrete_type_id_ =
    AsyncValue::CreateTypeInfoAndReturnTypeId<T>();
}  // namespace internal

struct DummyValueForErrorAsyncValue {};

class ErrorAsyncValue
    : public internal::ConcreteAsyncValue<DummyValueForErrorAsyncValue> {
 public:
  ErrorAsyncValue(absl::Status status)  // NOLINT
      : internal::ConcreteAsyncValue<DummyValueForErrorAsyncValue>(
            std::move(status)) {}
};

// IndirectAsyncValue represents an un-computed AsyncValue of unspecified kind
// and maybe unknown type. IndirectAsyncValue is used when an AsyncValue must be
// returned, but the value it holds is not ready and the producer of the value
// might not know what type it will ultimately be, or whether it will be an
// error. The purpose of indirect async value is to be eventually forwarded to
// a concrete async value with a constructed payload or an error.
class IndirectAsyncValue : public AsyncValue {
  friend class AsyncValue;

 public:
  IndirectAsyncValue()
      : AsyncValue(Kind::kIndirect, State::kUnconstructed,
                   /*is_refcounted=*/true) {}

  IndirectAsyncValue* AddRef() { return AddRef(1); }
  IndirectAsyncValue* AddRef(uint32_t count) {
    return static_cast<IndirectAsyncValue*>(AsyncValue::AddRef(count));
  }

  // Mark this IndirectAsyncValue as forwarding to the specified value. This
  // gives the IndirectAsyncValue a +1 reference.
  // This method must be called at most once.
  void ForwardTo(RCReference<AsyncValue> value);

  static bool classof(const AsyncValue* v) {
    return v->kind() == AsyncValue::Kind::kIndirect;
  }

  bool IsUnique() const {
    // In addition to checking the refcount of this IndirectAsyncValue, we also
    // need to check the refcount of the underlying value. If the underlying
    // value is not available, we conservatively return false.
    return (refcount_.load(std::memory_order_acquire) == 1) && IsAvailable() &&
           value_->IsUnique();
  }

 protected:
  // Constructor for TypedIndirectAsyncValue (defined below).
  template <typename T>
  explicit IndirectAsyncValue(TypeTag<T>)
      : AsyncValue(Kind::kIndirect, State::kUnconstructed,
                   /*is_refcounted=*/true, TypeTag<T>()) {}

  ~IndirectAsyncValue() { Destroy(); }

 private:
  void Destroy() {
    if (value_) {
      value_->DropRef();
      value_ = nullptr;
    }
  }

  AsyncValue* value_ = nullptr;
};

// TypedIndirectAsyncValue represents an indirect async value of a particular
// type. Indirect async values constructed with a known type can be forwarded
// only to async values of exactly the same type.
template <typename T>
class TypedIndirectAsyncValue : public IndirectAsyncValue {
 public:
  TypedIndirectAsyncValue() : IndirectAsyncValue(TypeTag<T>()) {
    static_assert(sizeof(TypedIndirectAsyncValue) ==
                  sizeof(IndirectAsyncValue));
  }
};

inline AsyncValue::~AsyncValue() {
  DCHECK_EQ(waiters_and_state_.load().waiter(), nullptr)
      << "An async value with waiters should never have refcount of zero";
  if (AsyncValueAllocationTrackingEnabled() && is_refcounted_)
    total_allocated_async_values_.fetch_sub(1, std::memory_order_relaxed);

  // Catch use-after-free errors more eagerly, by triggering the size assertion
  // in the 'get' accessor.
  type_id_ = ~0;
}

inline bool AsyncValue::IsAvailable() const {
  auto s = state();
  return s == State::kConcrete || s == State::kError;
}

inline bool AsyncValue::IsError() const { return state() == State::kError; }

inline bool AsyncValue::IsUnconstructed() const {
  return state() == State::kUnconstructed;
}

inline bool AsyncValue::IsConstructed() const {
  return state() == State::kConstructed;
}

inline bool AsyncValue::IsConcrete() const {
  return state() == State::kConcrete;
}

// Return true if this is an IndirectAsyncValue that hasn't been resolved.
// Currently an IndirectAsyncValue is available if and only if it is resolved.
inline bool AsyncValue::IsUnresolvedIndirect() const {
  return IsUnavailable() && (kind() == Kind::kIndirect);
}

inline bool AsyncValue::IsIndirect() const { return kind() == Kind::kIndirect; }

inline AsyncValue* AsyncValue::AddRef(uint32_t count) {
  // Always enable reference counting in debug builds to verify that the use of
  // async values is "ref count correct". In optimized builds the async value
  // owner is responsible for destructing the non-reference-counted async value.
#if defined(NDEBUG)
  if (!is_refcounted_) return this;
#endif

  if (count > 0) {
    DCHECK_GT(refcount_.load(std::memory_order_relaxed), 0);
    // Increasing the reference counter can always be done with
    // memory_order_relaxed: New references to an object can only be formed from
    // an existing reference, and passing an existing reference from one thread
    // to another must already provide any required synchronization.
    refcount_.fetch_add(count, std::memory_order_relaxed);
  }
  return this;
}

inline void AsyncValue::DropRef(uint32_t count) {
  // Always enable reference counting in debug builds to verify that the use of
  // async values is "ref count correct". In optimized builds the async value
  // owner is responsible for destructing the non-reference-counted async value.
#if defined(NDEBUG)
  if (!is_refcounted_) return;
#endif

  DCHECK_GT(refcount_.load(std::memory_order_relaxed), 0);
  // We expect that `count` argument will often equal the actual reference count
  // here; optimize for that. If `count` == reference count, only an acquire
  // barrier is needed to prevent the effects of the deletion from leaking
  // before this point.
  auto is_last_ref = refcount_.load(std::memory_order_acquire) == count;
  if (!is_last_ref) {
    // If `count` != reference count, a release barrier is needed in
    // addition to an acquire barrier so that prior changes by this thread
    // cannot be seen to occur after this decrement.
    is_last_ref =
        refcount_.fetch_sub(count, std::memory_order_acq_rel) == count;
  }
  // Destroy this value if the refcount drops to zero.
  if (is_last_ref) {
    Destroy();
  }
}

template <typename T>
const T& AsyncValue::GetConcreteValue() const {
  // Make sure both T (the stored type) and BaseT have vtable_ptr or
  // neither have the vtable_ptr.
  DCHECK_EQ(std::is_polymorphic<T>::value, has_vtable_);
  DCHECK(IsTypeIdCompatible<T>()) << "Incorrect accessor";

  const char* this_ptr = reinterpret_cast<const char*>(this);
  return *reinterpret_cast<const T*>(this_ptr + AsyncValue::kDataOffset);
}

template <typename T>
const T& AsyncValue::get() const {
  auto s = state();
  (void)s;

  switch (kind()) {
    case Kind::kConcrete:
#ifndef NDEBUG
      if (!GetTypeInfo().has_data(this)) {
        LOG(FATAL) << "Cannot call get() when ConcreteAsyncValue"
                   << " isn't constructed; state: " << s.DebugString() << ","
                   << " error message: "
                   << (IsError() ? GetError().message() : "None");
      }
#endif  // NDEBUG
      return GetConcreteValue<T>();
    case Kind::kIndirect:
#ifndef NDEBUG
      if (s != State::kConcrete) {
        LOG(FATAL) << "Cannot call get() when IndirectAsyncValue"
                   << " isn't concrete; state: " << s.DebugString() << ","
                   << " error message: "
                   << (IsError() ? GetError().message() : "None");
      }
#endif  // NDEBUG
      auto* iv_value = static_cast<const IndirectAsyncValue*>(this)->value_;
      DCHECK(iv_value) << "Indirect value not resolved";
      return iv_value->get<T>();
  }
}

template <typename T>
T& AsyncValue::get() {
  return const_cast<T&>(static_cast<const AsyncValue*>(this)->get<T>());
}

inline void AsyncValue::SetStateConcrete() {
  DCHECK(IsConstructed() && kind() == Kind::kConcrete);
  NotifyAvailable(State::kConcrete);
}

template <typename T, typename... Args>
void AsyncValue::emplace(Args&&... args) {
  DCHECK_EQ(GetTypeId<T>(), type_id_) << "Incorrect accessor";
  DCHECK(IsUnconstructed() && kind() == Kind::kConcrete);

  static_cast<internal::ConcreteAsyncValue<T>*>(this)->emplace(
      std::forward<Args>(args)...);
}

// Returns the underlying error, or nullptr if there is none.
inline const absl::Status* AsyncValue::GetErrorIfPresent() const {
  switch (kind()) {
    case Kind::kConcrete: {
      if (state() != State::kError) return nullptr;
      return &GetTypeInfo().get_error(this);
    }
    case Kind::kIndirect: {
      auto* iv_value = static_cast<const IndirectAsyncValue*>(this)->value_;
      // Unresolved IndirectAsyncValues are not errors.
      if (!iv_value) return nullptr;

      DCHECK(iv_value->kind() != Kind::kIndirect);
      return iv_value->GetErrorIfPresent();
    }
  }
}

inline const absl::Status& AsyncValue::GetError() const {
  auto* result = GetErrorIfPresent();
  DCHECK(result) << "Cannot call GetError() when error isn't available.";
  return *result;
}

template <typename Waiter>
void AsyncValue::AndThen(Waiter&& waiter) {
  // Clients generally want to use AndThen without them each having to check
  // to see if the value is present. Check for them, and immediately run the
  // waiter if it is already here.
  auto waiters_and_state = waiters_and_state_.load(std::memory_order_acquire);
  if (waiters_and_state.state() == State::kConcrete ||
      waiters_and_state.state() == State::kError) {
    DCHECK_EQ(waiters_and_state.waiter(), nullptr);
    std::forward<Waiter>(waiter)();
    return;
  }

  EnqueueWaiter(std::forward<Waiter>(waiter), waiters_and_state);
}

template <typename Waiter>
void AsyncValue::AndThen(Executor& executor, Waiter&& waiter) {
  // Clients generally want to use AndThen without them each having to check
  // to see if the value is present. Check for them, and immediately run the
  // waiter if it is already here.
  auto waiters_and_state = waiters_and_state_.load(std::memory_order_acquire);
  if (waiters_and_state.state() == State::kConcrete ||
      waiters_and_state.state() == State::kError) {
    DCHECK_EQ(waiters_and_state.waiter(), nullptr);
    executor.Execute(std::forward<Waiter>(waiter));
    return;
  }

  EnqueueWaiter(
      [&executor, waiter = std::forward<Waiter>(waiter)] {
        executor.Execute(std::move(waiter));
      },
      waiters_and_state);
}

inline void AsyncValue::Destroy() {
  // Copy `is_refcounted` flag before destroying the async value object.
  bool was_ref_counted = is_refcounted_;

  if (ABSL_PREDICT_FALSE(kind() == Kind::kIndirect)) {
    // Depending on what the benchmarks say, it might make sense to remove this
    // explicit check and instead make ~IndirectAsyncValue go through the
    // GetTypeInfo().destructor case below.
    static_cast<IndirectAsyncValue*>(this)->~IndirectAsyncValue();
    if (was_ref_counted) {
#if defined(__cpp_sized_deallocation)
      ::operator delete(this, sizeof(IndirectAsyncValue),
                        std::align_val_t{alignof(IndirectAsyncValue)});
#else   // defined(__cpp_sized_deallocation)
      ::operator delete(this, std::align_val_t{alignof(IndirectAsyncValue)});
#endif  // defined(__cpp_sized_deallocation)
    }
    return;
  }

  auto [size, alignment] = GetTypeInfo().destructor(this);
  if (was_ref_counted) {
#if defined(__cpp_sized_deallocation)
    ::operator delete(this, size, alignment);
#else   // defined(__cpp_sized_deallocation)
    ::operator delete(this, alignment);
#endif  // defined(__cpp_sized_deallocation)
  }
}

inline bool AsyncValue::IsUnique() const {
  if (kind() != Kind::kIndirect) {
    return refcount_.load(std::memory_order_acquire) == 1;
  }

  // If it is an IndirectAsyncValue, we also need to check the refcount of the
  // underlying value.
  return static_cast<const IndirectAsyncValue*>(this)->IsUnique();
}

}  // namespace tsl

#endif  // XLA_TSL_CONCURRENCY_ASYNC_VALUE_H_
