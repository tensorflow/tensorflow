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

#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/journal.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {

TEST(DispatcherState, RegisterDataset) {
  int64 id = 10;
  uint64 fingerprint = 20;
  DispatcherState state;
  Update update;
  RegisterDatasetUpdate* register_dataset = update.mutable_register_dataset();
  register_dataset->set_dataset_id(id);
  register_dataset->set_fingerprint(fingerprint);
  TF_EXPECT_OK(state.Apply(update));

  {
    std::shared_ptr<const DispatcherState::Dataset> dataset;
    TF_EXPECT_OK(state.DatasetFromFingerprint(fingerprint, &dataset));
    EXPECT_EQ(id, dataset->dataset_id);
  }
  {
    std::shared_ptr<const DispatcherState::Dataset> dataset;
    TF_EXPECT_OK(state.DatasetFromId(id, &dataset));
    EXPECT_EQ(fingerprint, dataset->fingerprint);
  }
}

TEST(DispatcherState, UnknownUpdate) {
  DispatcherState state;
  Update update;
  Status s = state.Apply(update);
  EXPECT_EQ(s.code(), error::INTERNAL);
}

}  // namespace data
}  // namespace tensorflow
