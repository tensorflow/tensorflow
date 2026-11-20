/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/profiler/gpu/rocm_tracer_options_utils.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "xla/tsl/profiler/utils/profiler_options_util.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

using tensorflow::ProfileOptions;
using ConfigValue = std::variant<std::string, bool, int64_t>;

// CUDA reads 0 as no limit for the two event limits.
constexpr uint64_t kNoLimit = std::numeric_limits<uint64_t>::max();

// Keys the CUPTI backend implements and the ROCm backend does not, in the
// order cupti_tracer_options_utils.cc parses them, followed by one that no
// backend parses. They draw a warning rather than an error, so that a
// configuration written for CUDA still works here.
//
// TODO(rocm-profiler): this list and the keys applied below restate the CUPTI
// key list by hand. EveryCuptiKeyIsRecognized in the test fails when CUPTI
// gains a key that is missing here. A key table in xla/tsl/profiler/utils read
// by both backends would remove the copy.
constexpr absl::string_view kUnimplementedOnRocm[] = {
    "gpu_enable_nvtx_tracking",
    "gpu_aggregated_tracing",
    "gpu_enable_cupti_activity_graph_trace",
    "gpu_pm_sample_counters",
    "gpu_pm_sample_interval_us",
    "gpu_pm_sample_buffer_size_per_gpu_mb",
    // Documented for JAX but parsed by no backend.
    "gpu_dump_graph_node_mapping",
};

absl::string_view TypeName(const ConfigValue& value) {
  if (std::holds_alternative<int64_t>(value)) {
    return "int64";
  }
  if (std::holds_alternative<bool>(value)) {
    return "bool";
  }
  return "string";
}

// Returns "advanced_configuration key '<key>' <predicate>".
std::string KeyMessage(absl::string_view key, absl::string_view predicate) {
  return absl::StrCat("advanced_configuration key '", key, "' ", predicate);
}

// Reads the gpu_* keys of advanced_configuration, and records every problem
// in `diagnostics`.
class KeyReader {
 public:
  KeyReader(const ProfileOptions& options,
            RocmTracerOptionDiagnostics& diagnostics)
      : options_(options), diagnostics_(diagnostics) {
    for (const auto& [key, unused_value] : options.advanced_configuration()) {
      keys_.insert(key);
    }
  }

  // Returns nullopt if `key` is absent, and also, after recording an error, if
  // it has no value or a value that is not an int64.
  std::optional<int64_t> TakeInt64(absl::string_view key) {
    if (keys_.erase(key) == 0) {
      return std::nullopt;
    }
    std::optional<ConfigValue> value =
        tsl::profiler::GetConfigValue(options_, std::string(key));
    if (!value.has_value()) {
      diagnostics_.errors.push_back(
          KeyMessage(key, "has no value. The key was ignored."));
      return std::nullopt;
    }
    if (!std::holds_alternative<int64_t>(*value)) {
      diagnostics_.errors.push_back(KeyMessage(
          key, absl::StrCat("expects a value of type int64 but got ",
                            TypeName(*value), ". The key was ignored.")));
      return std::nullopt;
    }
    return std::get<int64_t>(*value);
  }

  // Reads one of the gpu_max_* keys. They are documented as int64 but stored
  // in uint64_t fields, where a negative value would wrap to a cap that is
  // never reached. If `zero_effect` is set, a value of 0 draws a warning that
  // completes the sentence "... is 0, which <zero_effect>".
  std::optional<uint64_t> TakeCount(
      absl::string_view key,
      std::optional<absl::string_view> zero_effect = std::nullopt) {
    std::optional<int64_t> value = TakeInt64(key);
    if (!value.has_value()) {
      return std::nullopt;
    }
    if (*value < 0) {
      diagnostics_.errors.push_back(KeyMessage(
          key, absl::StrCat("is ", *value,
                            ", but negative values are not supported. The "
                            "key was ignored.")));
      return std::nullopt;
    }
    if (*value == 0 && zero_effect.has_value()) {
      diagnostics_.warnings.push_back(
          KeyMessage(key, absl::StrCat("is 0, which ", *zero_effect, ".")));
    }
    return static_cast<uint64_t>(*value);
  }

  // Warns about each key in kUnimplementedOnRocm, whatever its value, and
  // reports the other gpu_* keys not read yet as errors. advanced_configuration
  // is shared with other profiler components, for example session_manager's
  // "use_system_hostname", so keys without the gpu_ prefix are ignored.
  void ReportRemainingKeys() {
    for (absl::string_view key : kUnimplementedOnRocm) {
      if (keys_.erase(key) > 0) {
        diagnostics_.warnings.push_back(
            KeyMessage(key, "is not implemented on ROCm and was ignored."));
      }
    }
    std::vector<absl::string_view> unknown;
    for (absl::string_view key : keys_) {
      if (absl::StartsWith(key, "gpu_")) {
        unknown.push_back(key);
      }
    }
    absl::c_sort(unknown);
    for (absl::string_view key : unknown) {
      diagnostics_.errors.push_back(KeyMessage(
          key,
          "is not recognized by the ROCm GPU tracer. The key was ignored."));
    }
  }

 private:
  const ProfileOptions& options_;
  RocmTracerOptionDiagnostics& diagnostics_;
  absl::flat_hash_set<absl::string_view> keys_;
};

}  // namespace

RocmTracerOptionDiagnostics UpdateRocmTracerOptionsFromProfilerOptions(
    const ProfileOptions& profile_options, uint32_t device_count,
    RocmTraceCollectorOptions& collector_options) {
  RocmTracerOptionDiagnostics diagnostics;
  KeyReader reader(profile_options, diagnostics);

  // This budget also covers ROCTX marker events. As on CUDA
  // (CuptiTracer::TooManyCallbackEvents), 0 means no limit. It must not be
  // passed through: the collector drops every event when the limit is 0.
  if (std::optional<uint64_t> value =
          reader.TakeCount("gpu_max_callback_api_events")) {
    collector_options.max_callback_api_events = *value == 0 ? kNoLimit : *value;
  }

  // As above, 0 means no limit on CUDA.
  if (std::optional<uint64_t> value =
          reader.TakeCount("gpu_max_activity_api_events")) {
    collector_options.max_activity_api_events = *value == 0 ? kNoLimit : *value;
    // The collector only counts HIP_API-domain activity events, and every
    // activity event is HIP_OPS, so this limit is never reached yet.
    diagnostics.warnings.push_back(KeyMessage(
        "gpu_max_activity_api_events",
        "is stored but not yet enforced on ROCm; the trace is bounded by "
        "gpu_max_callback_api_events only."));
  }

  if (std::optional<uint64_t> value = reader.TakeCount(
          "gpu_max_annotation_strings",
          "retains no annotations, so kernel events have no op names")) {
    collector_options.max_annotation_strings = *value;
  }

  if (std::optional<int64_t> value =
          reader.TakeInt64("gpu_num_chips_to_profile_per_task")) {
    if (*value < 0) {
      diagnostics.errors.push_back(KeyMessage(
          "gpu_num_chips_to_profile_per_task",
          absl::StrCat("is ", *value,
                       ", but negative values are not supported. Pass 0 to "
                       "profile every GPU. The key was ignored.")));
    } else if (device_count == 0) {
      // 0 asks for every GPU, which is already the case.
      if (*value != 0) {
        diagnostics.warnings.push_back(
            KeyMessage("gpu_num_chips_to_profile_per_task",
                       "has no effect because no GPUs are present."));
      }
    } else if (*value == 0 || *value > device_count) {
      // As on CUDA (device_tracer_cuda.cc), 0 means every GPU. It must not be
      // passed through: the collector drops every event when num_gpus is 0.
      collector_options.num_gpus = device_count;
      if (*value != 0) {
        diagnostics.warnings.push_back(KeyMessage(
            "gpu_num_chips_to_profile_per_task",
            absl::StrCat("is ", *value, ", which exceeds the ", device_count,
                         " GPUs present, so all of them are profiled.")));
      }
    } else {
      collector_options.num_gpus = static_cast<uint32_t>(*value);
      if (*value < device_count) {
        // RocmTraceCollectorImpl::Flush() numbers devices from the lowest
        // device id among the collected events, not from GPU 0.
        diagnostics.warnings.push_back(KeyMessage(
            "gpu_num_chips_to_profile_per_task",
            absl::StrCat(
                "is ", *value, ". All ", device_count,
                " GPUs are still traced, so tracing overhead is unchanged. "
                "After collection, only events from ",
                *value,
                " consecutive GPU ids are kept, starting at the lowest id "
                "that recorded an event; the rest are dropped.")));
      }
    }
  }

  reader.ReportRemainingKeys();
  return diagnostics;
}

void AppendOptionDiagnostics(const RocmTracerOptionDiagnostics& diagnostics,
                             tensorflow::profiler::XSpace* space) {
  for (const std::string& message : diagnostics.errors) {
    LOG(ERROR) << message;
    if (space != nullptr) {
      space->add_errors(message);
    }
  }
  for (const std::string& message : diagnostics.warnings) {
    LOG(WARNING) << message;
    if (space != nullptr) {
      space->add_warnings(message);
    }
  }
}

}  // namespace profiler
}  // namespace xla
