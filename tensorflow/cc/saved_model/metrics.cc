/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/saved_model/metrics.h"

#include <string>

#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/sampler.h"

namespace tensorflow {
namespace metrics {

namespace {

// Counter that tracks total number and `write_version` of SavedModels written.
auto* saved_model_write_counter = monitoring::Counter<1>::New(
    "/tensorflow/core/saved_model/write/count",
    "The number of SavedModels successfully written.", "write_version");

// Counter that tracks total number and `write_version` of SavedModels read.
auto* saved_model_read_counter = monitoring::Counter<1>::New(
    "/tensorflow/core/saved_model/read/count",
    "The number of SavedModels successfully loaded.", "write_version");

// Counter that tracks number of calls for each SavedModel write API. Summing
// across "api_label" is not expected to equal the ".../write/count" cell value
// because programs can invoke more than one API to save a single SM and
// because the API may error out before successfully writing a SM.
auto* saved_model_write_api = monitoring::Counter<1>::New(
    "/tensorflow/core/saved_model/write/api",
    "The API used to write the SavedModel.", "api_label");

// Counter that tracks number of calls for each SavedModel read API. Summing
// across "api_label" is not expected to equal the ".../read/count" cell value
// because programs can invoke more than one API to load a single SM and
// because the API may error out before successfully reading a SM.
auto* saved_model_read_api = monitoring::Counter<1>::New(
    "/tensorflow/core/saved_model/read/api",
    "The API used to load the SavedModel.", "api_label");

// Distribution of checkpoint write durations.
auto* checkpoint_write_durations = monitoring::Sampler<1>::New(
    {
        "/tensorflow/core/checkpoint/write/write_durations",  // Metric name.
        "Distribution of the wall time duration in microseconds of the "
        "checkpoint write operation.",  // Metric description.
        "api_label"                     // Cell label.
    },
    // Scale of 1000, growth factor of 1.5 with upper bound of ~184 minutes.
    monitoring::Buckets::Exponential(1000, 1.5, 41));

// Distribution of checkpoint read durations.
auto* checkpoint_read_durations = monitoring::Sampler<1>::New(
    {
        "/tensorflow/core/checkpoint/read/read_durations",  // Metric name.
        "Distribution of the wall time duration in microseconds of the "
        "checkpoint read operation.",  // Metric description.
        "api_label"                    // Cell label.
    },
    // Scale of 1000, growth factor of 1.5 with upper bound of ~184 minutes.
    monitoring::Buckets::Exponential(1000, 1.5, 41));

// Counter that accumulates total time elapsed between module import time and
// the last successful Checkpoint write prior to job pre-emption or completion.
auto* checkpoint_training_time_saved = monitoring::Counter<1>::New(
    "/tensorflow/core/checkpoint/write/training_time_saved",
    "Total time in microseconds elapsed between two consecutive write "
    "operations in a single job or between Checkpoint construction and the "
    "first write operation.",
    "api_label");

// Counter that records filesize (MB) of written checkpoint. Contains two cells:
// (api_label, filesize). Cardinality should not be an issue as the filesize
// should be equal among all checkpoints written per job.
auto* checkpoint_size = monitoring::Counter<2>::New(
    "/tensorflow/core/checkpoint/write/checkpoint_size",
    "Size of checkpoint (.index and sharded data files), rounded to the "
    "nearest 100 MB.",
    "api_label", "filesize");

}  // namespace

monitoring::CounterCell& SavedModelWrite(absl::string_view write_version) {
  return *saved_model_write_counter->GetCell(std::string(write_version));
}

monitoring::CounterCell& SavedModelRead(absl::string_view write_version) {
  return *saved_model_read_counter->GetCell(std::string(write_version));
}

monitoring::CounterCell& SavedModelWriteApi(absl::string_view api_label) {
  return *saved_model_write_api->GetCell(std::string(api_label));
}

monitoring::CounterCell& SavedModelReadApi(absl::string_view api_label) {
  return *saved_model_read_api->GetCell(std::string(api_label));
}

monitoring::SamplerCell& CheckpointReadDuration(absl::string_view api_label) {
  return *checkpoint_read_durations->GetCell(std::string(api_label));
}

monitoring::SamplerCell& CheckpointWriteDuration(absl::string_view api_label) {
  return *checkpoint_write_durations->GetCell(std::string(api_label));
}

monitoring::CounterCell& TrainingTimeSaved(absl::string_view api_label) {
  return *checkpoint_training_time_saved->GetCell(std::string(api_label));
}

monitoring::CounterCell& CheckpointSize(absl::string_view api_label,
                                        int64_t filesize) {
  return *checkpoint_size->GetCell(std::string(api_label),
                                   std::to_string(filesize));
}

}  // namespace metrics
}  // namespace tensorflow
