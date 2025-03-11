/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_ENTRY_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_ENTRY_H_

#include <atomic>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/manual_constructor.h"

namespace tensorflow {

class Tensor;

// An Entry store a single input value for an individual kernel invocation in
// an executor.
//
// Either a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
struct Entry {
  enum class State {
    NO_VALUE = 0,      // The default state for a newly-created Entry.
    HAS_VALUE,         // `this->val` is valid.
    HAS_CONST_TENSOR,  // `this->const_tensor` is valid.
    HAS_REF_TENSOR,    // `this->ref_tensor` is valid.
  };

  Entry() : state(State::NO_VALUE) {}
  Entry(const Entry& other)
      // This shouldn't be copied where it needs to be atomic, atomic access is
      // only required when we have an array of entries shared by multiple
      // threads.
      : state(other.state.load(std::memory_order_relaxed)),
        alloc_attr(other.alloc_attr) {
    switch (state) {
      case State::NO_VALUE:
        break;
      case State::HAS_VALUE:
        val.Init(*other.val);
        break;
      case State::HAS_CONST_TENSOR:
        const_tensor = other.const_tensor;
        break;
      case State::HAS_REF_TENSOR:
        ref_tensor = other.ref_tensor;
        break;
    }
  }

  ~Entry() {
    if (state.load(std::memory_order_acquire) == State::HAS_VALUE) {
      val.Destroy();
    }
  }

  Entry& operator=(const Entry& other) {
    if (state == State::HAS_VALUE) {
      val.Destroy();
    }
    auto new_state = other.state.load(std::memory_order_acquire);
    alloc_attr = other.alloc_attr;
    switch (new_state) {
      case State::NO_VALUE:
        break;
      case State::HAS_VALUE:
        val.Init(*other.val);
        break;
      case State::HAS_CONST_TENSOR:
        const_tensor = other.const_tensor;
        break;
      case State::HAS_REF_TENSOR:
        ref_tensor = other.ref_tensor;
        break;
    }
    state.store(new_state, std::memory_order_release);
    return *this;
  }

  Entry& operator=(Entry&& other) {
    if (state == State::HAS_VALUE) {
      val.Destroy();
    }
    auto new_state = other.state.load(std::memory_order_acquire);
    alloc_attr = other.alloc_attr;
    switch (new_state) {
      case State::NO_VALUE:
        break;
      case State::HAS_VALUE:
        val.Init(std::move(*other.val));
        break;
      case State::HAS_CONST_TENSOR:
        const_tensor = other.const_tensor;
        break;
      case State::HAS_REF_TENSOR:
        ref_tensor = other.ref_tensor;
        break;
    }
    state.store(new_state, std::memory_order_release);
    return *this;
  }

  // Clears the <val> field, and sets this entry to the `NO_VALUE` state.
  void ClearVal() {
    auto old_state = state.exchange(State::NO_VALUE);
    if (old_state == State::HAS_VALUE) {
      val.Destroy();
    }
  }

  union {
    // A tensor value. Valid iff `state_ == HAS_VALUE`.
    ManualConstructor<Tensor> val;

    // A pointer to a constant tensor value. Valid iff `state_ ==
    // HAS_CONST_TENSOR`.
    const Tensor* const_tensor;

    // A tensor reference and associated mutex. Valid iff `state_ ==
    // HAS_REF_TENSOR`.
    struct {
      Tensor* tensor;
      mutex* mu;
    } ref_tensor;
  };

  // The current state of this entry, indicating which member of the above
  // union is active. This can be used to mark the result as ready to another
  // thread waiting for this entry, hence the atomic. If an entry is used this
  // way it must have a stable address (no vector.push_back() while threads are
  // running).
  std::atomic<State> state;

  // The attributes of the allocator that creates the tensor.
  AllocatorAttributes alloc_attr;
};

// TODO(b/152925936): Re-evaluate this constant with current usage patterns.
typedef absl::InlinedVector<Entry, 4UL> EntryVector;

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_ENTRY_H_
