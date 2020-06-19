/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/lite/toco/allocate_transient_arrays.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {
namespace {

// The life span of an array.
struct ArrayLifespan {
  // If true, the array is persistent state (as in a RNN). In that case,
  // its allocation is permanent and the first_op, last_op members are
  // unused. (The term 'transient' is a misnomer and we should think in
  // terms of 'workspace' instead).
  bool persistent = false;
  // Index of the first op addressing that array. The array must be allocated
  // just before executing this op.
  std::size_t first_op = 0;
  // Index of the last op addressing that array. We want to deallocate the array
  // immediately after executing this op.
  std::size_t last_op = 0;
};

bool StartsAt(const ArrayLifespan& lifespan, std::size_t op_index) {
  return !lifespan.persistent && lifespan.first_op == op_index;
}

bool EndsAt(const ArrayLifespan& lifespan, std::size_t op_index) {
  return !lifespan.persistent && lifespan.last_op == op_index;
}

// Helper function for ComputeArrayLifespans: updates one ArrayLifespan for
// one array for one op.
void UpdateArrayLifespan(
    const std::string& array_name, std::size_t op_index,
    std::unordered_map<std::string, ArrayLifespan>* array_lifespans) {
  if (array_lifespans->count(array_name)) {
    auto& lifespan = array_lifespans->at(array_name);
    if (!lifespan.persistent) {
      lifespan.first_op = std::min(lifespan.first_op, op_index);
      lifespan.last_op = std::max(lifespan.last_op, op_index);
    }
  } else {
    ArrayLifespan lifespan;
    lifespan.first_op = op_index;
    lifespan.last_op = op_index;
    (*array_lifespans)[array_name] = lifespan;
  }
}

// Computes the ArrayLifespan for each array.
void ComputeArrayLifespans(
    const Model& model,
    std::unordered_map<std::string, ArrayLifespan>* array_lifespans) {
  CHECK(array_lifespans->empty());
  for (const auto& rnn_state : model.flags.rnn_states()) {
    ArrayLifespan lifespan;
    lifespan.persistent = true;
    (*array_lifespans)[rnn_state.state_array()] = lifespan;
  }
  for (std::size_t op_index = 0; op_index < model.operators.size();
       op_index++) {
    const auto& op = model.operators[op_index];
    for (const auto& input : op->inputs) {
      UpdateArrayLifespan(input, op_index, array_lifespans);
    }
    for (const auto& output : op->outputs) {
      UpdateArrayLifespan(output, op_index, array_lifespans);
    }
  }
}

inline bool operator==(const Alloc& a, const Alloc& b) {
  CHECK(a.start != b.start || a.end == b.end);
  return a.start == b.start;
}

// Helper to keep track of total allocation size and of currently live
// allocations, and containing the core allocation routine.
class Allocator {
 public:
  Allocator() : total_size_(0) {}

  // Core allocation routine.
  void Allocate(std::size_t size, Alloc* result) {
    if (size == 0) {
      // zero-sized arrays get a dummy alloc of (0, 0) that does not
      // need to be kept in the books (no need to insert that into
      // live_allocs_).
      // Note: zero-sized arrays shouldn't exist, but handling that case
      // here allows such pathological cases to get a cleaner error message
      // later instead of generating spurious allocator failures.
      result->start = 0;
      result->end = 0;
      return;
    }
    // Naive algorithm: pick the first gap between live allocations,
    // that is wide enough for the new array.
    std::size_t pos = 0;
    for (const auto& a : live_allocs_) {
      if (a.start >= pos + size) {
        result->start = pos;
        result->end = pos + size;
        live_allocs_.insert(*result);
        return;
      }
      pos = a.end;
    }
    // No sufficiently wide gap was found before an existing live allocation,
    // so we allocate the new array at the end of the allocation space.
    // We may then have to grow total_size_.
    total_size_ = std::max(total_size_, pos + size);
    result->start = pos;
    result->end = pos + size;
    live_allocs_.insert(*result);
  }

  void Deallocate(const Alloc& a) {
    // Special-case dummy allocs for zero-sized arrays.
    if (a.start == 0 && a.end == 0) {
      // Nothing needs to be done, these aren't kept in the books.
      return;
    }
    auto iter = std::lower_bound(live_allocs_.begin(), live_allocs_.end(), a);
    CHECK(iter != live_allocs_.end());
    CHECK(*iter == a);
    live_allocs_.erase(iter);
  }

  std::size_t total_size() const { return total_size_; }

 private:
  std::size_t total_size_;
  std::set<Alloc> live_allocs_;
};

// Returns the required transient allocation size (in bytes) for a given array,
// or 0 if it's not a transient array.
std::size_t TransientArraySize(const Model& model,
                               const std::string& array_name,
                               std::size_t transient_data_alignment) {
  if (!IsAllocatableTransientArray(model, array_name)) {
    return 0;
  }
  const auto& array = &model.GetArray(array_name);
  CHECK(array->has_shape())
      << "Array '" << array_name << "' doesn't have a shape";
  if (array->data_type == ArrayDataType::kNone) {
    // Catch a typical issue at the moment with RNN states
    for (const auto& rnn_state : model.flags.rnn_states()) {
      if (rnn_state.state_array() == array_name) {
        LOG(FATAL)
            << "A RNN state array, " << array_name << ", still does not "
            << "have a known data type after all graph transformations have "
            << "run.";
      }
    }
    LOG(FATAL) << "An array, " << array_name << ", still does not "
               << "have a known data type after all graph transformations have "
               << "run.";
  }
  const std::size_t elem_size = ElementSize(array->data_type);
  const std::size_t raw_size =
      elem_size * RequiredBufferSizeForShape(array->shape());
  const std::size_t rounded_size =
      RoundUpToNextMultipleOf(raw_size, transient_data_alignment);
  return rounded_size;
}

// Allocates an array: call this for every array just before the first
// op where it is used.
void AllocateTransientArray(const Model& model, const std::string& array_name,
                            Allocator* allocator,
                            std::size_t transient_data_alignment) {
  if (!IsAllocatableTransientArray(model, array_name)) {
    return;
  }
  const std::size_t size =
      TransientArraySize(model, array_name, transient_data_alignment);
  const auto& array = &model.GetArray(array_name);
  CHECK(!array->alloc);
  allocator->Allocate(size, &array->GetOrCreateAlloc());
}

// Deallocates an array: call this for every array just after the last
// op where it is used.
void DeallocateTransientArray(const Model& model, const std::string& array_name,
                              Allocator* allocator) {
  if (!IsAllocatableTransientArray(model, array_name)) {
    return;
  }
  const auto& array = &model.GetArray(array_name);
  CHECK(!!array->alloc);
  allocator->Deallocate(*array->alloc);
}

void PushBackIfNotFound(const std::string& s, std::vector<std::string>* v) {
  if (std::find(v->begin(), v->end(), s) == v->end()) {
    v->push_back(s);
  }
}

}  // namespace

void AllocateTransientArrays(Model* model,
                             std::size_t transient_data_alignment) {
  // Precompute the lifespans for all arrays.
  std::unordered_map<std::string, ArrayLifespan> array_lifespans;
  ComputeArrayLifespans(*model, &array_lifespans);

  // In case of variable batch, our convention will be to compute the
  // allocations for batch==1, then let the inference code multiply all
  // the offsets by the actual runtime batch size. Conveniently,
  // the variable_batch and batch flags are mutually exclusive, and the default
  // value of batch is 1, so we have nothing special to do here. Let us
  // just guard this assumption with a CHECK:
  bool batchless_input_shapes = true;
  for (const auto& input_array : model->flags.input_arrays()) {
    if (!input_array.has_shape() || input_array.shape().dims().empty() ||
        input_array.shape().dims(0) != 1) {
      batchless_input_shapes = false;
      break;
    }
  }
  CHECK(!model->flags.variable_batch() || batchless_input_shapes);

  Allocator allocator;

  // Construct a sorted map of array names, so that other layout engines can
  // match exactly.
  std::map<std::string, const Array*> ordered_arrays_map;
  for (const auto& pair : model->GetArrayMap()) {
    ordered_arrays_map[pair.first] = pair.second.get();
  }

  // Allocate persistent arrays (like RNN states). For them, 'transient'
  // is a misnormer, should read 'workspace'.
  for (const auto& array_pair : ordered_arrays_map) {
    const std::string& array_name = array_pair.first;
    auto it = array_lifespans.find(array_name);
    if (it != array_lifespans.end() && it->second.persistent) {
      AllocateTransientArray(*model, array_name, &allocator,
                             transient_data_alignment);
    }
  }

  for (std::size_t op_index = 0; op_index < model->operators.size();
       op_index++) {
    const auto& op = model->operators[op_index];
    // Allocate those arrays whose lifespan starts exactly here.
    std::vector<std::string> arrays_to_allocate;
    for (const auto& input : op->inputs) {
      if (StartsAt(array_lifespans[input], op_index)) {
        PushBackIfNotFound(input, &arrays_to_allocate);
      }
    }
    for (const auto& output : op->outputs) {
      if (StartsAt(array_lifespans[output], op_index)) {
        PushBackIfNotFound(output, &arrays_to_allocate);
      }
    }
    for (const std::string& array : arrays_to_allocate) {
      AllocateTransientArray(*model, array, &allocator,
                             transient_data_alignment);
    }

    // Deallocate those arrays whose lifespan ends exactly here.
    std::vector<std::string> arrays_to_deallocate;
    for (const auto& input : op->inputs) {
      if (EndsAt(array_lifespans[input], op_index)) {
        PushBackIfNotFound(input, &arrays_to_deallocate);
      }
    }
    for (const auto& output : op->outputs) {
      if (EndsAt(array_lifespans[output], op_index)) {
        PushBackIfNotFound(output, &arrays_to_deallocate);
      }
    }
    for (const std::string& array : arrays_to_deallocate) {
      DeallocateTransientArray(*model, array, &allocator);
    }
  }

  // Just out of curiosity (not used in the actual allocation process)
  // evaluate the optimal total allocated size.
  // First, compute the size of persistent arrays.
  std::size_t optimal_transient_alloc_size = 0;
  std::size_t persistent_alloc_size = 0;
  for (const auto& array_pair : ordered_arrays_map) {
    const std::string& array_name = array_pair.first;
    auto it = array_lifespans.find(array_name);
    if (it != array_lifespans.end() && it->second.persistent) {
      persistent_alloc_size +=
          TransientArraySize(*model, array_name, transient_data_alignment);
    }
  }
  for (const auto& op : model->operators) {
    // for each operator, compute the sum of the sizes of the array that must
    // be live during the execution of this operator, plus the size of
    // persistent arrays that must be live at all times.
    std::vector<std::string> non_persistent_edges;
    for (const auto& input : op->inputs) {
      if (!array_lifespans[input].persistent) {
        PushBackIfNotFound(input, &non_persistent_edges);
      }
    }
    for (const auto& output : op->outputs) {
      if (!array_lifespans[output].persistent) {
        PushBackIfNotFound(output, &non_persistent_edges);
      }
    }
    std::size_t size = persistent_alloc_size;
    for (const std::string& edge : non_persistent_edges) {
      size += TransientArraySize(*model, edge, transient_data_alignment);
    }
    // The optimal total size is the maximum of all operator-specific sizes.
    optimal_transient_alloc_size = std::max(optimal_transient_alloc_size, size);
  }

  model->transient_data_size = allocator.total_size();
  model->transient_data_alignment = transient_data_alignment;
  CHECK_GE(model->transient_data_size, optimal_transient_alloc_size);
  LOG(INFO) << "Total transient array allocated size: "
            << model->transient_data_size << " bytes, "
            << "theoretical optimal value: " << optimal_transient_alloc_size
            << " bytes.";
  CheckInvariants(*model);
}
}  // namespace toco
