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
// Note that not all state needs to be journaled, and in general we journal
// as little state as possible. For example, worker and task state doesn't need
// to be journaled because we can recover that information from workers when
// they reconnect to a restarted dispatcher.
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
    Dataset(int64 dataset_id, int64 fingerprint, const DatasetDef& dataset_def)
        : dataset_id(dataset_id),
          fingerprint(fingerprint),
          dataset_def(dataset_def) {}

    const int64 dataset_id;
    const int64 fingerprint;
    const DatasetDef dataset_def;
  };

  // A key for identifying a named job. The key contains a user-specified name,
  // as well as an index describing which iteration of the job we are on.
  struct NamedJobKey {
    NamedJobKey(absl::string_view name, int64 index)
        : name(name), index(index) {}

    friend bool operator==(const NamedJobKey& lhs, const NamedJobKey& rhs) {
      return lhs.name == rhs.name && lhs.index == rhs.index;
    }

    template <typename H>
    friend H AbslHashValue(H h, const NamedJobKey& k) {
      return H::combine(std::move(h), k.name, k.index);
    }

    const std::string name;
    const int64 index;
  };

  // A job for processing a dataset.
  struct Job {
    Job(int64 job_id, int64 dataset_id, ProcessingMode processing_mode,
        absl::optional<NamedJobKey> named_job_key)
        : job_id(job_id),
          dataset_id(dataset_id),
          processing_mode(processing_mode),
          named_job_key(named_job_key) {}

    const int64 job_id;
    const int64 dataset_id;
    const ProcessingMode processing_mode;
    const absl::optional<NamedJobKey> named_job_key;
    bool finished = false;
  };

  // Returns the next available dataset id.
  int64 NextAvailableDatasetId() const;
  // Gets a dataset by id. Returns NOT_FOUND if there is no such dataset.
  Status DatasetFromId(int64 id, std::shared_ptr<const Dataset>* dataset) const;
  // Gets a dataset by fingerprint. Returns NOT_FOUND if there is no such
  // dataset.
  Status DatasetFromFingerprint(uint64 fingerprint,
                                std::shared_ptr<const Dataset>* dataset) const;

  // Returns the next available job id.
  int64 NextAvailableJobId() const;
  // Returns a list of all jobs.
  std::vector<std::shared_ptr<const Job>> ListJobs();
  // Gets a job by id. Returns NOT_FOUND if there is no such job.
  Status JobFromId(int64 id, std::shared_ptr<const Job>* job) const;
  // Gets a named job by key. Returns NOT_FOUND if there is no such job.
  Status NamedJobByKey(NamedJobKey key, std::shared_ptr<const Job>* job) const;

 private:
  // Registers a dataset. The dataset must not already be registered.
  void RegisterDataset(const RegisterDatasetUpdate& register_dataset);
  void CreateJob(const CreateJobUpdate& create_job);
  void FinishJob(const FinishJobUpdate& finish_job);

  int64 next_available_dataset_id_ = 0;
  // Registered datasets, keyed by dataset ids.
  absl::flat_hash_map<int64, std::shared_ptr<Dataset>> datasets_by_id_;
  // Registered datasets, keyed by dataset fingerprints.
  absl::flat_hash_map<uint64, std::shared_ptr<Dataset>>
      datasets_by_fingerprint_;

  int64 next_available_job_id_ = 0;
  // Jobs, keyed by job ids.
  absl::flat_hash_map<int64, std::shared_ptr<Job>> jobs_;
  // Named jobs, keyed by their names and indices. Not all jobs have names, so
  // this is a subset of the jobs stored in `jobs_`.
  absl::flat_hash_map<NamedJobKey, std::shared_ptr<Job>> named_jobs_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_STATE_H_
