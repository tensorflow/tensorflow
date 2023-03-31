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
#include <utility>

#include "json/config.h"
#include "json/json.h"
#include "json/writer.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/protobuf/fingerprint.pb.h"

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

// Gauge that contains the fingerprint (saved_model_checksum) of the newly
// written SavedModel.
auto* saved_model_write_fingerprint = monitoring::Gauge<string, 0>::New(
    "/tensorflow/core/saved_model/write/fingerprint",
    "The fingerprint (saved_model_checksum) of the exported SavedModel.");

// Gauge that contains the path (saved_model_path) of the newly written
// SavedModel.
auto* saved_model_write_path = monitoring::Gauge<string, 0>::New(
    "/tensorflow/core/saved_model/write/path",
    "The path (saved_model_path) of the exported SavedModel.");

// Gauge that contains the path (saved_model_path) and the singleprint
// (concatenation of graph_def_program_hash, signature_def_hash,
// saved_object_graph_hash, and checkpoint_hash) of the newly written
// SavedModel.
auto* saved_model_write_path_and_singleprint =
    monitoring::Gauge<string, 0>::New(
        "/tensorflow/core/saved_model/write/path_and_singleprint",
        "The path (saved_model_path) and singleprint (concatenation of "
        "graph_def_program_hash, signature_def_hash, saved_object_graph_hash, "
        "and checkpoint_hash) of the newly written SavedModel.");

// Gauge that contains the fingerprint (saved_model_checksum) of the loaded
// SavedModel.
auto* saved_model_read_fingerprint = monitoring::Gauge<string, 0>::New(
    "/tensorflow/core/saved_model/read/fingerprint",
    "The fingerprint (saved_model_checksum) of the loaded SavedModel.");

// Gauge that contains the path (saved_model_path) of the loaded SavedModel.
auto* saved_model_read_path = monitoring::Gauge<string, 0>::New(
    "/tensorflow/core/saved_model/read/path",
    "The path (saved_model_path) of the loaded SavedModel.");

// Gauge that contains the path (saved_model_path) and the singleprint
// (concatenation of graph_def_program_hash, signature_def_hash,
// saved_object_graph_hash, and checkpoint_hash) of the loaded SavedModel.
auto* saved_model_read_path_and_singleprint = monitoring::Gauge<string, 0>::New(
    "/tensorflow/core/saved_model/read/path_and_singleprint",
    "The path (saved_model_path) and singleprint (concatenation of "
    "graph_def_program_hash, signature_def_hash, saved_object_graph_hash, "
    "and checkpoint_hash) of the loaded SavedModel.");

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

// Distribution of async checkpoint write durations.
auto* async_checkpoint_write_durations = monitoring::Sampler<1>::New(
    {
        "/tensorflow/core/checkpoint/write/async_write_durations",  // Metric
                                                                    // name.
        "Distribution of the wall time duration in microseconds of the async "
        "checkpoint write operation",  // Metric description.
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

monitoring::CounterCell& SavedModelWriteCount(absl::string_view write_version) {
  return *saved_model_write_counter->GetCell(std::string(write_version));
}

monitoring::CounterCell& SavedModelReadCount(absl::string_view write_version) {
  return *saved_model_read_counter->GetCell(std::string(write_version));
}

monitoring::CounterCell& SavedModelWriteApi(absl::string_view api_label) {
  return *saved_model_write_api->GetCell(std::string(api_label));
}

monitoring::CounterCell& SavedModelReadApi(absl::string_view api_label) {
  return *saved_model_read_api->GetCell(std::string(api_label));
}

monitoring::GaugeCell<string>& SavedModelReadFingerprint() {
  return *saved_model_read_fingerprint->GetCell();
}

monitoring::GaugeCell<string>& SavedModelReadPath() {
  return *saved_model_read_path->GetCell();
}

monitoring::GaugeCell<string>& SavedModelReadPathAndSingleprint() {
  return *saved_model_read_path_and_singleprint->GetCell();
}

monitoring::GaugeCell<string>& SavedModelWriteFingerprint() {
  return *saved_model_write_fingerprint->GetCell();
}

monitoring::GaugeCell<string>& SavedModelWritePath() {
  return *saved_model_write_path->GetCell();
}

monitoring::GaugeCell<string>& SavedModelWritePathAndSingleprint() {
  return *saved_model_write_path_and_singleprint->GetCell();
}

string MakeFingerprintJson(FingerprintDef fingerprint_serialized) {
  Json::Value fingerprint = Json::objectValue;
  fingerprint["saved_model_checksum"] =
      Json::UInt64(fingerprint_serialized.saved_model_checksum());
  fingerprint["graph_def_program_hash"] =
      Json::UInt64(fingerprint_serialized.graph_def_program_hash());
  fingerprint["signature_def_hash"] =
      Json::UInt64(fingerprint_serialized.signature_def_hash());
  fingerprint["saved_object_graph_hash"] =
      Json::UInt64(fingerprint_serialized.saved_object_graph_hash());
  fingerprint["checkpoint_hash"] =
      Json::UInt64(fingerprint_serialized.checkpoint_hash());

  Json::StreamWriterBuilder json_factory;
  return Json::writeString(json_factory, fingerprint);
}

string MakeSavedModelPathAndSingleprint(string path, string singleprint) {
  return absl::StrCat(path, ":", singleprint);
}

std::pair<string, string> ParseSavedModelPathAndSingleprint(
    string path_and_singleprint) {
  size_t delimiter = path_and_singleprint.rfind(':');
  if (delimiter == std::string::npos) {
    return std::pair<string, string>("", "");
  }
  return std::pair<string, string>(path_and_singleprint.substr(0, delimiter),
                                   path_and_singleprint.substr(delimiter + 1));
}

monitoring::SamplerCell& CheckpointReadDuration(absl::string_view api_label) {
  return *checkpoint_read_durations->GetCell(std::string(api_label));
}

monitoring::SamplerCell& CheckpointWriteDuration(absl::string_view api_label) {
  return *checkpoint_write_durations->GetCell(std::string(api_label));
}

monitoring::SamplerCell& AsyncCheckpointWriteDuration(
    absl::string_view api_label) {
  return *async_checkpoint_write_durations->GetCell(std::string(api_label));
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
