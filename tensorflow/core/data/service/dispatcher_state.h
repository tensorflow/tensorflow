/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_STATE_H_
#define TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_STATE_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/data_service.h"
#include "tensorflow/core/data/service/journal.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace data {

// A class encapsulating the journaled state of the dispatcher. All state
// modifications must be done via `ApplyUpdate`. This helps to ensure that
// replaying the journal will allow us to restore the exact same state.
//
// The following usage pattern will keep the journal in sync with the state of
// the dispatcher:
// {
//   mutex_lock l(mu_);
//   Update update = ...  // create an update
//   dispatcher_state.ApplyUpdate(update);
//   journal_writer.write(Update);
//   // Unlock mu_
// }
//
// The division of functionality between DispatcherImpl and DispatcherState is
// as follows:
//   - DispatcherImpl is responsible for handling RPC requests, reading from
//     DispatcherState, and deciding what updates to apply to DispatcherState.
//     DispatcherImpl handles all synchronization.
//   - DispatcherState is responsible for making the state changes requested by
//     DispatcherImpl and for providing DispatcherImpl with read-only access to
//     the state.
//
// DispatcherState is thread-compatible but not thread-safe.
class DispatcherState {
 public:
  DispatcherState();
  DispatcherState(const DispatcherState&) = delete;
  DispatcherState& operator=(const DispatcherState&) = delete;

  // Applies the given update to the dispatcher's state.
  Status Apply(Update update);

  // A dataset registered with the dispatcher.
  struct Dataset {
   public:
    Dataset(int64 dataset_id, int64 fingerprint, const DatasetDef& dataset_def)
        : dataset_id(dataset_id),
          fingerprint(fingerprint),
          dataset_def(dataset_def) {}

    const int64 dataset_id;
    const int64 fingerprint;
    const DatasetDef dataset_def;
  };

  // Gets a dataset by id. Returns NOT_FOUND if there is no such dataset.
  Status DatasetFromId(int64 id, std::shared_ptr<const Dataset>* dataset) const;
  // Gets a dataset by fingerprint. Returns NOT_FOUND if there is no such
  // dataset.
  Status DatasetFromFingerprint(uint64 fingerprint,
                                std::shared_ptr<const Dataset>* dataset) const;

 private:
  // Registers a dataset. The dataset must not already be registered.
  void RegisterDataset(const RegisterDatasetUpdate& register_dataset);

  // Registered datasets, keyed by dataset ids.
  absl::flat_hash_map<int64, std::shared_ptr<Dataset>> datasets_by_id_;
  // Registered datasets, keyed by dataset fingerprints.
  absl::flat_hash_map<uint64, std::shared_ptr<Dataset>>
      datasets_by_fingerprint_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_STATE_H_
