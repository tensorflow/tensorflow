/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/linear_optimizer/kernels/resources.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

DataByExample::DataByExample(const string& container, const string& solver_uuid)
    : container_(container), solver_uuid_(solver_uuid) {}

DataByExample::~DataByExample() {}

// static
DataByExample::EphemeralKey DataByExample::MakeKey(const string& example_id) {
  return Fingerprint128(example_id);
}

DataByExample::Data DataByExample::Get(const EphemeralKey& key) {
  mutex_lock l(mu_);
  return data_by_key_[key];
}

void DataByExample::Set(const EphemeralKey& key, const Data& data) {
  mutex_lock l(mu_);
  data_by_key_[key] = data;
}

Status DataByExample::Visit(
    std::function<void(const Data& data)> visitor) const {
  struct State {
    // Snapshoted size of data_by_key_.
    size_t size;

    // Number of elements visited so far.
    size_t num_visited = 0;

    // Current element.
    DataByKey::const_iterator it;
  };

  auto state = [this] {
    mutex_lock l(mu_);
    State result;
    result.size = data_by_key_.size();
    result.it = data_by_key_.cbegin();
    return result;
  }();

  while (state.num_visited < state.size) {
    mutex_lock l(mu_);
    // Since DataByExample is modify-or-append only, a visit will (continue to)
    // be successful if and only if the size of the backing store hasn't
    // changed (since the body of this while-loop is under lock).
    if (data_by_key_.size() != state.size) {
      return errors::Unavailable("The number of elements for ", solver_uuid_,
                                 " has changed which nullifies a visit.");
    }
    for (size_t i = 0; i < kVisitChunkSize && state.num_visited < state.size;
         ++i, ++state.num_visited, ++state.it) {
      visitor(state.it->second);
    }
  }
  return Status::OK();
}

string DataByExample::DebugString() {
  return strings::StrCat("DataByExample(", container_, ", ", solver_uuid_, ")");
}

}  // namespace tensorflow
