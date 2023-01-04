/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_RUNTIME_STATE_H_
#define TENSORFLOW_COMPILER_XLA_RUNTIME_STATE_H_

#include <cstddef>
#include <memory>
#include <type_traits>

#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"

namespace xla {
namespace runtime {

// Forward declare.
template <typename T>
class State;

// StateVector is a container for keeping the state between custom call
// invocations (within same or concurrent calls to `Executable::Execute`).
//
// Every instance of the custom call in the executable must be assigned a unique
// id (dense i64 integer in the [0, num_instances) range), and the state vector
// enables efficient access to value of type `T`, with lazy initialization done
// by the custom call itself.
//
// Custom calls implement special bindings for state arguments:
//
//   CustomCall::Bind("foo")
//     .State<FooState>("id_attr_name")
//     .To([](State<FooState> state) {
//       FooState foo = state.GetOrCreate(...);
//       ...
//     });
//
// State snapshot must be passed via UserData:
//
//   StateVector<FooState>::Snapshot snapshot = ...;
//   CustomCall::UserData user_data(&snapshot, ...);
//
// See `custom_call.h` and `custom_call_test.cc` for more examples.
template <typename T>
class StateVector {
 public:
  explicit StateVector(size_t reserve = 0);

  // Snapshot wraps the read-only state snapshot (potentially obsolete) and
  // falls back on synchronized state access if it can't find an entry in its
  // read-only state snapshot.
  class Snapshot {
   public:
    // Returns a state for the given id if it's already available, or tries to
    // create a new one using the user-provided `create` function. Returns an
    // error if state creation fails. Failures are not cached and always
    // retried.
    template <typename F>
    absl::StatusOr<T*> GetOrCreate(size_t id, F&& create);

    // Returns a state constructed from this snapshot for a given id.
    State<T> state(size_t id) { return State<T>(id, this); }

   private:
    friend StateVector;
    Snapshot(StateVector& state, std::shared_ptr<std::vector<T*>> snapshot);

    // State vector that this snapshot was taken from.
    StateVector& owning_state_;

    // State vector snapshot captured at construction time. It might not contain
    // all the latest data available in the `owning_state_`.
    std::shared_ptr<std::vector<T*>> maybe_obsolete_snapshot_;
  };

  Snapshot snapshot();
  T* operator[](size_t id);

 private:
  mutable absl::Mutex mu_;

  // StateVector owns the values of type `T` indexed by the key.
  std::vector<std::unique_ptr<T>> vector_ ABSL_GUARDED_BY(mu_);

  // A read-only snapshot of `vector_` that is updated every time a new entry
  // is added to the state vector.
  std::shared_ptr<std::vector<T*>> vector_snapshot_ ABSL_GUARDED_BY(mu_);
};

template <typename T>
class State {
  using Snapshot = typename StateVector<T>::Snapshot;

 public:
  template <typename F>
  absl::StatusOr<T*> GetOrCreate(F&& create) {
    return snapshot_->GetOrCreate(id_, std::forward<F>(create));
  }

 private:
  friend Snapshot;
  State(size_t id, Snapshot* snapshot) : id_(id), snapshot_(snapshot) {}

  size_t id_;
  Snapshot* snapshot_;
};

//===----------------------------------------------------------------------===//

template <typename T>
StateVector<T>::StateVector(size_t reserve)
    : vector_(reserve), vector_snapshot_(new std::vector<T*>(reserve)) {}

template <typename T>
StateVector<T>::Snapshot::Snapshot(StateVector& state,
                                   std::shared_ptr<std::vector<T*>> snapshot)
    : owning_state_(state), maybe_obsolete_snapshot_(std::move(snapshot)) {}

template <typename T>
auto StateVector<T>::snapshot() -> Snapshot {
  absl::MutexLock lock(&mu_);
  return Snapshot(*this, vector_snapshot_);
}

template <typename T>
T* StateVector<T>::operator[](size_t id) {
  absl::MutexLock lock(&mu_);
  return id < vector_.size() ? vector_[id].get() : nullptr;
}

template <typename T>
template <typename F>
absl::StatusOr<T*> StateVector<T>::Snapshot::GetOrCreate(size_t id,
                                                         F&& create) {
  static_assert(std::is_invocable_r_v<absl::StatusOr<T>, F>);

  // If snapshot already contains the entry, just return it.
  std::vector<T*>& snapshot = *maybe_obsolete_snapshot_;
  if (id < snapshot.size() && snapshot[id]) return snapshot[id];

  // Otherwise go through the slow synchronized code path.
  absl::MutexLock lock(&owning_state_.mu_);

  // Check if value is present in the state vector, and was not captured in
  // the snapshot that we have.
  std::vector<std::unique_ptr<T>>& state = owning_state_.vector_;
  if (id < state.size() && state[id].get()) return state[id].get();

  // Try to insert a new entry.
  absl::StatusOr<T> value = create();
  if (!value.ok()) return value.status();

  // Update the state vector.
  if (id >= state.size()) state.resize(id + 1);
  state[id] = std::make_unique<T>(*std::move(value));

  // And take the new snapshot.
  auto new_snapshot = std::make_shared<std::vector<T*>>(state.size());
  for (size_t i = 0; i < state.size(); ++i) (*new_snapshot)[i] = state[i].get();

  // Update the snapshot owned by the state.
  owning_state_.vector_snapshot_ = std::move(new_snapshot);

  return state[id].get();
}

}  // namespace runtime
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RUNTIME_STATE_H_
