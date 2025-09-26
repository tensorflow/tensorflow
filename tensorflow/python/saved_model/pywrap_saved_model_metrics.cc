/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");;
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/python/saved_model/pywrap_saved_model_metrics.h"

#include <cstdint>
#include <exception>
#include <string>
#include <utility>

// Placeholder for lineage logging import.
// Placeholder for lineage logging additional import.
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "pybind11/attr.h"  // from @pybind11
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "tensorflow/cc/saved_model/metrics.h"
#include "tensorflow/core/protobuf/fingerprint.pb.h"

namespace tensorflow {
namespace saved_model {
namespace python {

namespace py = pybind11;

class MetricException : public std::exception {
 public:
  explicit MetricException(const char* m) : message_{m} {}
  const char* what() const noexcept override { return message_.c_str(); }

 private:
  std::string message_ = "";
};

void DefineMetricsModule(py::module main_module) {
  // Deduplicate writes from lineage log. This should reduce the number of
  // outgoing calls to UMB from lineage log. See b/415794129 for details.
  // Placeholder for lineage logging enable dedupe call.
  auto m = main_module.def_submodule("metrics");

  m.doc() = "Python bindings for TensorFlow SavedModel and Checkpoint Metrics.";

  static py::exception<MetricException> fp_ex(m, "MetricException");
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) {
        std::rethrow_exception(p);
      }
    } catch (const MetricException& e) {
      fp_ex(e.what());
    }
  });

  m.def(
      "IncrementWrite",
      [](const char* write_version) {
        metrics::SavedModelWriteCount(write_version).IncrementBy(1);
      },
      py::kw_only(), py::arg("write_version"),
      py::doc("Increment the '/tensorflow/core/saved_model/write/count' "
              "counter."));

  m.def(
      "GetWrite",
      [](const char* write_version) {
        return metrics::SavedModelWriteCount(write_version).value();
      },
      py::kw_only(), py::arg("write_version"),
      py::doc("Get value of '/tensorflow/core/saved_model/write/count' "
              "counter."));

  m.def(
      "IncrementWriteApi",
      [](const char* api_label) {
        metrics::SavedModelWriteApi(api_label).IncrementBy(1);
      },
      py::doc("Increment the '/tensorflow/core/saved_model/write/api' "
              "counter for API with `api_label`"));

  m.def(
      "GetWriteApi",
      [](const char* api_label) {
        return metrics::SavedModelWriteApi(api_label).value();
      },
      py::doc("Get value of '/tensorflow/core/saved_model/write/api' "
              "counter for `api_label` cell."));

  m.def(
      "IncrementRead",
      [](const char* write_version) {
        metrics::SavedModelReadCount(write_version).IncrementBy(1);
      },
      py::kw_only(), py::arg("write_version"),
      py::doc("Increment the '/tensorflow/core/saved_model/read/count' "
              "counter after reading a SavedModel with the specifed "
              "`write_version`."));

  m.def(
      "GetRead",
      [](const char* write_version) {
        return metrics::SavedModelReadCount(write_version).value();
      },
      py::kw_only(), py::arg("write_version"),
      py::doc("Get value of '/tensorflow/core/saved_model/read/count' "
              "counter for SavedModels with the specified `write_version`."));

  m.def(
      "IncrementReadApi",
      [](const char* api_label) {
        metrics::SavedModelReadApi(api_label).IncrementBy(1);
      },
      py::doc("Increment the '/tensorflow/core/saved_model/read/api' "
              "counter for API with `api_label`."));

  m.def(
      "GetReadApi",
      [](const char* api_label) {
        return metrics::SavedModelReadApi(api_label).value();
      },
      py::doc("Get value of '/tensorflow/core/saved_model/read/api' "
              "counter for `api_label` cell."));

  m.def(
      "SetReadFingerprint",
      [](const py::bytes fingerprint) {
        FingerprintDef fingerprint_def;
        fingerprint_def.ParseFromString(std::string(fingerprint));
        metrics::SavedModelReadFingerprint().Set(
            metrics::MakeFingerprintJson(fingerprint_def).c_str());
      },
      py::kw_only(), py::arg("fingerprint"),
      py::doc("Set the '/tensorflow/core/saved_model/read/fingerprint' gauge "
              "with `fingerprint`."));

  m.def(
      "GetReadFingerprint",
      []() { return metrics::SavedModelReadFingerprint().value(); },
      py::doc("Get value of '/tensorflow/core/saved_model/read/fingerprint' "
              "gauge."));

  m.def(
      "SetWriteFingerprint",
      [](const py::bytes fingerprint) {
        FingerprintDef fingerprint_def;
        fingerprint_def.ParseFromString(std::string(fingerprint));
        metrics::SavedModelWriteFingerprint().Set(
            metrics::MakeFingerprintJson(fingerprint_def).c_str());
      },
      py::kw_only(), py::arg("fingerprint"),
      py::doc("Set the '/tensorflow/core/saved_model/write/fingerprint' gauge "
              "with `fingerprint`."));

  m.def(
      "GetWriteFingerprint",
      []() { return metrics::SavedModelWriteFingerprint().value(); },
      py::doc("Get value of '/tensorflow/core/saved_model/write/fingerprint' "
              "gauge."));

  m.def(
      "SetReadPath",
      [](const char* saved_model_path) {
        metrics::SavedModelReadPath().Set(saved_model_path);
      },
      py::kw_only(), py::arg("saved_model_path"),
      py::doc("Set the '/tensorflow/core/saved_model/read/path' gauge "
              "with `saved_model_path`."));

  m.def(
      "GetReadPath", []() { return metrics::SavedModelReadPath().value(); },
      py::doc("Get value of '/tensorflow/core/saved_model/read/path' gauge."));

  m.def(
      "SetWritePath",
      [](const char* saved_model_path) {
        metrics::SavedModelWritePath().Set(saved_model_path);
      },
      py::kw_only(), py::arg("saved_model_path"),
      py::doc("Set the '/tensorflow/core/saved_model/write/path' gauge "
              "with `saved_model_path`."));

  m.def(
      "GetWritePath", []() { return metrics::SavedModelWritePath().value(); },
      py::doc("Get value of '/tensorflow/core/saved_model/write/path' gauge."));

  m.def(
      "SetReadPathAndSingleprint",
      [](const char* path, const char* singleprint) {
        auto path_and_singleprint =
            metrics::MakeSavedModelPathAndSingleprint(path, singleprint);
        if (!path_and_singleprint.ok()) {
          throw MetricException(path_and_singleprint.status().message().data());
        }
        metrics::SavedModelReadPathAndSingleprint().Set(
            path_and_singleprint.value());
        // Placeholder for lineage logging input call.
      },
      py::kw_only(), py::arg("path"), py::arg("singleprint"),
      py::doc(
          "Set the '/tensorflow/core/saved_model/read/path_and_singleprint' "
          "gauge with `path` and `singleprint`."));

  m.def(
      "GetReadPathAndSingleprint",
      []() {
        std::string path_and_singleprint =
            metrics::SavedModelReadPathAndSingleprint().value();
        if (path_and_singleprint.empty())
          return std::pair<std::string, std::string>();
        auto parsed_path_and_singleprint =
            metrics::ParseSavedModelPathAndSingleprint(path_and_singleprint);
        if (!parsed_path_and_singleprint.ok()) {
          throw MetricException(
              parsed_path_and_singleprint.status().message().data());
        }
        return parsed_path_and_singleprint.value();
      },
      py::doc(
          "Get tuple of `path` and `singleprint` values of "
          "'/tensorflow/core/saved_model/read/path_and_singleprint' gauge."));

  m.def(
      "SetWritePathAndSingleprint",
      [](const char* path, const char* singleprint) {
        auto path_and_singleprint =
            metrics::MakeSavedModelPathAndSingleprint(path, singleprint);
        if (!path_and_singleprint.ok()) {
          throw MetricException(path_and_singleprint.status().message().data());
        }
        metrics::SavedModelWritePathAndSingleprint().Set(
            path_and_singleprint.value());
        // Placeholder for lineage logging output call.
      },
      py::kw_only(), py::arg("path"), py::arg("singleprint"),
      py::doc("Set the "
              "'/tensorflow/core/saved_model/write/path_and_singleprint' gauge "
              "with `path` and `singleprint`."));

  m.def(
      "GetWritePathAndSingleprint",
      []() {
        std::string path_and_singleprint =
            metrics::SavedModelWritePathAndSingleprint().value();
        if (path_and_singleprint.empty())
          return std::pair<std::string, std::string>();
        auto parsed_path_and_singleprint =
            metrics::ParseSavedModelPathAndSingleprint(path_and_singleprint);
        if (!parsed_path_and_singleprint.ok()) {
          throw MetricException(
              parsed_path_and_singleprint.status().message().data());
        }
        return parsed_path_and_singleprint.value();
      },
      py::doc(
          "Get tuple of `path` and `singleprint` values of "
          "'/tensorflow/core/saved_model/write/path_and_singleprint' gauge."));

  m.attr("kFingerprintFound") = metrics::kFingerprintFound;
  m.attr("kFingerprintNotFound") = metrics::kFingerprintNotFound;
  m.attr("kFingerprintError") = metrics::kFingerprintError;

  m.def(
      "GetFoundFingerprintOnLoad",
      []() { return metrics::SavedModelFoundFingerprintOnLoad().value(); },
      py::doc(
          "Get value of "
          "'/tensorflow/core/saved_model/found_fingerprint_on_load' gauge."));

  m.def(
      "SetFoundFingerprintOnLoad",
      [](std::string found_status) {
        if (found_status == metrics::kFingerprintFound ||
            found_status == metrics::kFingerprintNotFound ||
            found_status == metrics::kFingerprintError) {
          metrics::SavedModelFoundFingerprintOnLoad().Set(found_status);
        } else {
          metrics::SavedModelFoundFingerprintOnLoad().Set("");
        }
      },
      py::kw_only(), py::arg("found_status"),
      py::doc("Set value of "
              "'/tensorflow/core/saved_model/found_fingerprint_on_load' gauge "
              "with 'found_status' if status is one of { \"FOUND\", "
              "\"NOT_FOUND\", \"ERROR\" }."));

  m.def(
      "AddCheckpointReadDuration",
      [](const char* api_label, double microseconds) {
        metrics::CheckpointReadDuration(api_label).Add(microseconds);
      },
      py::kw_only(), py::arg("api_label"), py::arg("microseconds"),
      py::doc("Add `microseconds` to the cell `api_label`for "
              "'/tensorflow/core/checkpoint/read/read_durations'."));

  m.def(
      "GetCheckpointReadDurations",
      [](const char* api_label) {
        // This function is called sparingly in unit tests, so protobuf
        // (de)-serialization round trip is not an issue.
        return py::bytes(metrics::CheckpointReadDuration(api_label)
                             .value()
                             .SerializeAsString());
      },
      py::kw_only(), py::arg("api_label"),
      py::doc("Get serialized HistogramProto of `api_label` cell for "
              "'/tensorflow/core/checkpoint/read/read_durations'."));

  m.def(
      "AddCheckpointWriteDuration",
      [](const char* api_label, double microseconds) {
        metrics::CheckpointWriteDuration(api_label).Add(microseconds);
      },
      py::kw_only(), py::arg("api_label"), py::arg("microseconds"),
      py::doc("Add `microseconds` to the cell `api_label` for "
              "'/tensorflow/core/checkpoint/write/write_durations'."));

  m.def(
      "GetCheckpointWriteDurations",
      [](const char* api_label) {
        // This function is called sparingly, so protobuf (de)-serialization
        // round trip is not an issue.
        return py::bytes(metrics::CheckpointWriteDuration(api_label)
                             .value()
                             .SerializeAsString());
      },
      py::kw_only(), py::arg("api_label"),
      py::doc("Get serialized HistogramProto of `api_label` cell for "
              "'/tensorflow/core/checkpoint/write/write_durations'."));

  m.def(
      "AddAsyncCheckpointWriteDuration",
      [](const char* api_label, double microseconds) {
        metrics::AsyncCheckpointWriteDuration(api_label).Add(microseconds);
      },
      py::kw_only(), py::arg("api_label"), py::arg("microseconds"),
      py::doc("Add `microseconds` to the cell `api_label` for "
              "'/tensorflow/core/checkpoint/write/async_write_durations'."));

  m.def(
      "GetAsyncCheckpointWriteDurations",
      [](const char* api_label) {
        // This function is called sparingly, so protobuf (de)-serialization
        // round trip is not an issue.
        return py::bytes(metrics::AsyncCheckpointWriteDuration(api_label)
                             .value()
                             .SerializeAsString());
      },
      py::kw_only(), py::arg("api_label"),
      py::doc("Get serialized HistogramProto of `api_label` cell for "
              "'/tensorflow/core/checkpoint/write/async_write_durations'."));

  m.def(
      "AddTrainingTimeSaved",
      [](const char* api_label, double microseconds) {
        metrics::TrainingTimeSaved(api_label).IncrementBy(microseconds);
      },
      py::kw_only(), py::arg("api_label"), py::arg("microseconds"),
      py::doc("Add `microseconds` to the cell `api_label` for "
              "'/tensorflow/core/checkpoint/write/training_time_saved'."));

  m.def(
      "GetTrainingTimeSaved",
      [](const char* api_label) {
        return metrics::TrainingTimeSaved(api_label).value();
      },
      py::kw_only(), py::arg("api_label"),
      py::doc("Get cell `api_label` for "
              "'/tensorflow/core/checkpoint/write/training_time_saved'."));

  m.def(
      "CalculateFileSize",
      [](const char* filename) {
        Env* env = Env::Default();
        uint64 filesize = 0;
        if (!env->GetFileSize(filename, &filesize).ok()) {
          return (int64_t)-1;
        }
        // Convert to MB.
        int64_t filesize_mb = filesize / 1000;
        // Round to the nearest 100 MB.
        // Smaller multiple.
        int64_t a = (filesize_mb / 100) * 100;
        // Larger multiple.
        int64_t b = a + 100;
        // Return closest of two.
        return (filesize_mb - a > b - filesize_mb) ? b : a;
      },
      py::doc("Calculate filesize (MB) for `filename`, rounding to the nearest "
              "100MB. Returns -1 if `filename` is invalid."));

  m.def(
      "RecordCheckpointSize",
      [](const char* api_label, int64_t filesize) {
        metrics::CheckpointSize(api_label, filesize).IncrementBy(1);
      },
      py::kw_only(), py::arg("api_label"), py::arg("filesize"),
      py::doc("Increment the "
              "'/tensorflow/core/checkpoint/write/checkpoint_size' counter for "
              "cell (api_label, filesize) after writing a checkpoint."));

  m.def(
      "GetCheckpointSize",
      [](const char* api_label, uint64 filesize) {
        return metrics::CheckpointSize(api_label, filesize).value();
      },
      py::kw_only(), py::arg("api_label"), py::arg("filesize"),
      py::doc("Get cell (api_label, filesize) for "
              "'/tensorflow/core/checkpoint/write/checkpoint_size'."));

  m.def(
      "GetShardingCallbackDuration",
      []() { return metrics::ShardingCallbackDuration().value(); },
      py::doc("Get value of "
              "'/tensorflow/core/checkpoint/sharding/callback_duration'."));

  m.def(
      "AddShardingCallbackDuration",
      [](int64_t callback_duration) {
        metrics::ShardingCallbackDuration().IncrementBy(callback_duration);
      },
      py::kw_only(), py::arg("callback_duration"),
      py::doc("Set value of "
              "'/tensorflow/core/checkpoint/sharding/callback_duration'."));

  m.def(
      "GetNumCheckpointShardsWritten",
      []() { return metrics::NumCheckpointShardsWritten().value(); },
      py::doc("Get value of "
              "'/tensorflow/core/checkpoint/sharding/"
              "num_checkpoint_shards_written'."));

  m.def(
      "AddNumCheckpointShardsWritten",
      [](int64_t num_shards) {
        metrics::NumCheckpointShardsWritten().IncrementBy(num_shards);
      },
      py::kw_only(), py::arg("num_shards"),
      py::doc("Set value of "
              "'/tensorflow/core/checkpoint/sharding/"
              "num_checkpoint_shards_written'."));

  m.def(
      "GetShardingCallbackDescription",
      []() { return metrics::ShardingCallbackDescription().value(); },
      py::doc("Get value of "
              "'/tensorflow/core/checkpoint/sharding/callback_description'."));

  m.def(
      "SetShardingCallbackDescription",
      [](std::string description) {
        metrics::ShardingCallbackDescription().Set(description);
      },
      py::kw_only(), py::arg("description"),
      py::doc("Set value of "
              "'/tensorflow/core/checkpoint/sharding/callback_description'."));
}

}  // namespace python
}  // namespace saved_model
}  // namespace tensorflow
