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
// APIs for accessing SavedModel and checkpoint metric objects.
//
// In order to collect the data from these metrics, please add the metrics to
// the provided monitoring platform. Unless configured with a user-specified
// monitoring platform, the data is not collected in OSS.

#ifndef TENSORFLOW_CC_SAVED_MODEL_METRICS_H_
#define TENSORFLOW_CC_SAVED_MODEL_METRICS_H_

#include <cstdint>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/protobuf/fingerprint.pb.h"

namespace tensorflow {
namespace metrics {

const char kFingerprintFound[] = "FOUND";
const char kFingerprintNotFound[] = "NOT_FOUND";
const char kFingerprintError[] = "ERROR";

// Returns "/tensorflow/core/saved_model/write/count" cell. This metric
// has 1 field "write_version", which is equal to the
// `tensorflow::libexport::GetWriteVersion` of the protobuf and should be
// incremented when a SavedModel has been successfully written.
monitoring::CounterCell& SavedModelWriteCount(absl::string_view write_version);

// Returns "/tensorflow/core/saved_model/read/count" cell. This metric
// has 1 field "write_version", which is equal to the
// `tensorflow::libexport::GetWriteVersion` of the protobuf, and should be
// incremented when a SavedModel has been successfully read.
monitoring::CounterCell& SavedModelReadCount(absl::string_view write_version);

// Returns "/tensorflow/core/saved_model/write/api" cell. This metric has 1
// field "api_label" which corresponds to a SavedModel write API. The cell for
// `foo` should be incremented when the write API `foo` is called.
monitoring::CounterCell& SavedModelWriteApi(absl::string_view api_label);

// Returns "/tensorflow/core/saved_model/read/api" cell. This metric has 1
// field "api_label" which corresponds to a SavedModel read API. The cell for
// `foo` should be incremented when the read API `foo` is called.
monitoring::CounterCell& SavedModelReadApi(absl::string_view api_label);

// Returns "/tensorflow/core/saved_model/write/fingerprint" cell, which contains
// the saved_model_checksum of the SM's fingerprint when it is exported.
monitoring::GaugeCell<std::string>& SavedModelWriteFingerprint();

// Returns "/tensorflow/core/saved_model/write/path" cell, which contains
// the saved_model_path of the SM when it is exported.
monitoring::GaugeCell<std::string>& SavedModelWritePath();

// Returns "/tensorflow/core/saved_model/write/path_and_fingerprint" cell, which
// contains the path (saved_model_path) and fingerprint (concatenation of
// graph_def_program_hash, signature_def_hash, saved_object_graph_hash,
// and checkpoint_hash) of the SavedModel when it is exported.
monitoring::GaugeCell<std::string>& SavedModelWritePathAndSingleprint();

// Returns "/tensorflow/core/saved_model/read/fingerprint" cell, wich contains
// the saved_model_checksum of the SM's fingerprint when it is imported.
monitoring::GaugeCell<std::string>& SavedModelReadFingerprint();

// Returns "/tensorflow/core/saved_model/read/path" cell, wich contains
// the saved_model_path of the SM when it is imported.
monitoring::GaugeCell<std::string>& SavedModelReadPath();

// Returns "/tensorflow/core/saved_model/read/path_and_fingerprint" cell, which
// contains the path (saved_model_path) and singleprint (concatenation of
// graph_def_program_hash, signature_def_hash, saved_object_graph_hash,
// and checkpoint_hash) of the SavedModel when it is imported.
monitoring::GaugeCell<std::string>& SavedModelReadPathAndSingleprint();

// Returns the fingerprint as a Json string.
std::string MakeFingerprintJson(FingerprintDef fingerprint_def);

// Returns canonical string concatenation of path and singleprint.
absl::StatusOr<std::string> MakeSavedModelPathAndSingleprint(
    std::string path, std::string singleprint);

// Returns path and singleprint as a pair, parsed canonically from the string
// metric.
absl::StatusOr<std::pair<std::string, std::string>>
ParseSavedModelPathAndSingleprint(std::string path_and_singleprint);

// Returns string status indicating whether or not the fingerprint.pb file was
// found when loading the SavedModel.
monitoring::GaugeCell<std::string>& SavedModelFoundFingerprintOnLoad();

// Returns "/tensorflow/core/checkpoint/read/read_durations" cell belonging to
// field `api_label`.
monitoring::SamplerCell& CheckpointReadDuration(absl::string_view api_label);

// Returns "/tensorflow/core/checkpoint/write/write_durations" cell belonging to
// field `api_label`.
monitoring::SamplerCell& CheckpointWriteDuration(absl::string_view api_label);

// Returns "/tensorflow/core/checkpoint/write/async_write_durations" cell
// belonging to field `api_label`.
monitoring::SamplerCell& AsyncCheckpointWriteDuration(
    absl::string_view api_label);

// Returns  "/tensorflow/core/checkpoint/write/training_time_saved" cell
// belonging to field `api_label`.
monitoring::CounterCell& TrainingTimeSaved(absl::string_view api_label);

// Returns  "/tensorflow/core/checkpoint/write/checkpoint_size" cell
// belonging to field (`api_label`, `filesize`).
monitoring::CounterCell& CheckpointSize(absl::string_view api_label,
                                        int64_t filesize);

// Returns "/tensorflow/core/checkpoint/sharding/callback_duration" cell which
// describes how long it took to execute the checkpoint sharding callback in
// microseconds.
monitoring::CounterCell& ShardingCallbackDuration();

// Returns "/tensorflow/core/checkpoint/sharding/num_checkpoint_shards_written"
// cell which describes how many checkpoint shard files were written during
// saving.
monitoring::CounterCell& NumCheckpointShardsWritten();

// Returns "/tensorflow/core/checkpoint/sharding/callback_description" cell
// which describes the callback used to shard the checkpoint during saving.
monitoring::GaugeCell<std::string>& ShardingCallbackDescription();

}  // namespace metrics
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_SAVED_MODEL_METRICS_H_
