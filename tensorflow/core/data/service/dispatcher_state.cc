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
#include "tensorflow/core/data/service/dispatcher_state.h"

#include "tensorflow/core/data/service/journal.pb.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace data {

DispatcherState::DispatcherState() {}

Status DispatcherState::Apply(Update update) {
  switch (update.update_type_case()) {
    case Update::kRegisterDataset:
      RegisterDataset(update.register_dataset());
      break;
    // TODO(aaudibert): implement these.
    case Update::kCreateJob:
    case Update::kCreateTask:
    case Update::kFinishJob:
      return errors::Unimplemented("Update type ", update.update_type_case(),
                                   " is not yet supported");
    case Update::UPDATE_TYPE_NOT_SET:
      return errors::Internal("Update type not set.");
  }

  return Status::OK();
}

void DispatcherState::RegisterDataset(
    const RegisterDatasetUpdate& register_dataset) {
  int64 id = register_dataset.dataset_id();
  int64 fingerprint = register_dataset.fingerprint();
  auto dataset = std::make_shared<Dataset>(id, fingerprint,
                                           register_dataset.dataset_def());
  DCHECK(!datasets_by_id_.contains(id));
  datasets_by_id_[id] = dataset;
  DCHECK(!datasets_by_fingerprint_.contains(fingerprint));
  datasets_by_fingerprint_[fingerprint] = dataset;
}

Status DispatcherState::DatasetFromId(
    int64 id, std::shared_ptr<const Dataset>* dataset) const {
  auto it = datasets_by_id_.find(id);
  if (it == datasets_by_id_.end()) {
    return errors::NotFound("Dataset id ", id, " not found");
  }
  *dataset = it->second;
  return Status::OK();
}

Status DispatcherState::DatasetFromFingerprint(
    uint64 fingerprint, std::shared_ptr<const Dataset>* dataset) const {
  auto it = datasets_by_fingerprint_.find(fingerprint);
  if (it == datasets_by_fingerprint_.end()) {
    return errors::NotFound("Dataset fingerprint ", fingerprint, " not found");
  }
  *dataset = it->second;
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
