/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/experimental/remat/rematerializer.h"

#include <algorithm>
#include <map>
#include <tuple>
#include <vector>

namespace mlir {
namespace TFL {

namespace {

// Helper functions for sorted + deduped int vectors.

std::tuple<std::vector<int>::iterator, bool> Find(const int item,
                                                  std::vector<int>& items) {
  const auto iter = std::lower_bound(items.begin(), items.end(), item);
  return std::make_tuple(iter, iter != items.end() && *iter == item);
}

void Insert(const int item, std::vector<int>& items) {
  const auto [iter, found] = Find(item, items);
  if (!found) items.insert(iter, item);
}

void Erase(const int item, std::vector<int>& items) {
  const auto [iter, found] = Find(item, items);
  if (found) items.erase(iter);
}

}  // namespace

int Rematerializer::AddOperation() {
  operations_.emplace_back();
  return operations_.size() - 1;
}

int Rematerializer::AddTensor(const SizeT size) {
  tensors_.emplace_back();
  tensors_.back().size = size;
  return tensors_.size() - 1;
}

void Rematerializer::DelUse(const int ioperation, const int itensor) {
  auto& tensor = tensors_[itensor];
  auto& operation = operations_[ioperation];

  const auto& size = tensor.size;

  // Was the dependence to be deleted the first/last (or both) use of this
  // tensor?
  const bool was_first_use =
      (!tensor.operations.empty() && ioperation == tensor.first_use());
  const bool was_last_use =
      (!tensor.operations.empty() && ioperation == tensor.last_use());
  Erase(ioperation, tensor.operations);
  Erase(itensor, operation.tensors);
  if (was_first_use) {
    operation.alloc -= size;
    if (!was_last_use) {
      operations_[tensor.first_use()].alloc += size;
    }
  }
  if (was_last_use) {
    operation.dealloc -= size;
    if (!was_first_use) {
      operations_[tensor.last_use()].dealloc += size;
    }
  }
}

void Rematerializer::AddUse(const int ioperation, const int itensor) {
  auto& tensor = tensors_[itensor];
  auto& operation = operations_[ioperation];

  const auto& size = tensor.size;

  const bool will_be_first_use =
      tensor.operations.empty() || ioperation < tensor.first_use();
  const bool will_be_last_use =
      tensor.operations.empty() || ioperation > tensor.last_use();

  if (will_be_first_use) {
    operation.alloc += size;
    if (!will_be_last_use) {
      operations_[tensor.first_use()].alloc -= size;
    }
  }
  if (will_be_last_use) {
    operation.dealloc += size;
    if (!will_be_first_use) {
      operations_[tensor.last_use()].dealloc -= size;
    }
  }
  Insert(ioperation, tensor.operations);
  Insert(itensor, operation.tensors);
}

Rematerializer::MemProfile Rematerializer::GetMemProfile() const {
  std::vector<SizeT> profile(operations_.size());
  MapMem([&](const MemSpec& m) { profile[m.op_index] = m.size; });
  return profile;
}

Rematerializer::MemSpec Rematerializer::GetPeakMemory() const {
  MemSpec peak;
  MapMem([&](const MemSpec& m) { peak = std::max(m, peak, BySize); });
  return peak;
}

}  // namespace TFL
}  // namespace mlir
