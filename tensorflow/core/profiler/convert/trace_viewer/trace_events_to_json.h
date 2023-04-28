/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_TRACE_VIEWER_TRACE_EVENTS_TO_JSON_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_TRACE_VIEWER_TRACE_EVENTS_TO_JSON_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/time/time.h"
#include "tensorflow/core/profiler/convert/trace_viewer/trace_events_util.h"
#include "tensorflow/core/profiler/convert/trace_viewer/trace_viewer_color.h"
#include "tensorflow/core/profiler/lib/context_types.h"
#include "tensorflow/core/profiler/protobuf/task.pb.h"
#include "tensorflow/core/profiler/protobuf/trace_events.pb.h"
#include "tensorflow/core/profiler/protobuf/trace_events_raw.pb.h"
#include "tensorflow/core/profiler/utils/timespan.h"
#include "tensorflow/tsl/platform/protobuf.h"

namespace tensorflow {
namespace profiler {

// JSON generation options.
struct JsonTraceOptions {
  using Details = std::vector<std::pair<std::string, bool>>;

  // Options and values for filtering based on the "details" menu.
  Details details;

  // If selected_device_ids is set, we add a field "selected_device_ids"
  // in the Trace JSON.
  std::optional<absl::flat_hash_set<uint32_t>> selected_device_ids;

  // Device IDs of devices whose resources should be sorted by name instead of
  // by resource ID.
  absl::flat_hash_set<uint32_t /*device_id*/> sort_resources_by_name;

  // Returns the color for an event.
  TraceEventsColorerInterface* colorer = nullptr;

  bool generate_stack_frames = true;
};

// Counts generated JSON events by type.
class JsonEventCounter {
 public:
  JsonEventCounter() : event_count_(kNumEventTypes, 0) {}
  ~JsonEventCounter() { LOG(INFO) << ToString(); }

  // Types of JSON events (bit.ly/trace-event-format)
  enum EventType {
    kCompleteEvent,
    kCompleteEventWithFlow,
    kCounterEvent,
    kAsyncEvent,
  };

  void Inc(EventType e) { ++event_count_[e]; }

  std::string ToString() const {
    std::string output = "Generated JSON events:";
    for (size_t i = 0; i < event_count_.size(); ++i) {
      absl::StrAppend(&output, " ", kEventTypeName[i], ": ", event_count_[i]);
    }
    return output;
  }

 private:
  static constexpr absl::string_view kEventTypeName[] = {
      "complete",
      "complete+flow",
      "counter",
      "async",
  };

  static constexpr size_t kNumEventTypes = ABSL_ARRAYSIZE(kEventTypeName);

  absl::FixedArray<size_t> event_count_;
};

// Adds a separator between elements of a JSON array or object.
template <typename IOBuffer>
class JsonSeparator {
 public:
  explicit JsonSeparator(IOBuffer* output) : output_(output) {}

  // Does nothing on the first call; adds a comma to the output on subsequent
  // calls.
  void Add() {
    output_->Append(sep_);
    sep_ = ",";
  }

 private:
  IOBuffer* output_;
  absl::string_view sep_;
};

// Converts picoseconds to microseconds.
inline double PicosToMicros(uint64_t ps) { return ps / 1E6; }

// Escapes the contents of "raw" in JSON style.
// Also adds double quotes to the beginning and end of the string.
std::string JsonEscape(absl::string_view raw);

std::string ProtoString(const tsl::protobuf::Message& pb);

template <typename RawDataType, typename IOBuffer>
void WriteTpuData(const RawDataType& data, JsonSeparator<IOBuffer>* separator,
                  IOBuffer* output) {}

// Writes JSON events from a TraceEvent.
template <typename IOBuffer, typename RawDataType>
class JsonEventWriter {
 public:
  JsonEventWriter(const TraceEventsColorerInterface* colorer,
                  const Trace& trace,
                  const std::map<uint64_t, uint64_t>& references,
                  IOBuffer* output)
      : colorer_(colorer),
        trace_(trace),
        references_(references),
        output_(output) {}

  void WriteEvent(const TraceEvent& event) const {
    std::optional<TraceEvent> async_event;
    output_->Append(R"({"pid":)", event.device_id());
    if (event.has_resource_id()) {
      output_->Append(R"(,"tid":)", event.resource_id());
    }
    const std::string& event_name =
        event.has_name_ref() ? trace_.name_table().at(event.name_ref())
                             : event.name();
    output_->Append(R"(,"name":)", JsonEscape(event_name));
    Timespan span = EventSpan(event);
    // "%.17g" is the default double format in proto2::util::JsonFormat.
    absl::Format(output_, R"(,"ts":%.17g)", PicosToMicros(span.begin_ps()));
    JsonEventCounter::EventType event_type = JsonEventCounter::kCounterEvent;
    if (event.has_resource_id()) {
      event_type = event.has_flow_id()
                       ? JsonEventCounter::kCompleteEventWithFlow
                       : JsonEventCounter::kCompleteEvent;
      // A complete event must have a duration, otherwise trace-viewer will
      // extend the event to the end of the trace and append "(Did Not Finish)"
      // to its name. Make the minimum duration 1 picosecond.
      uint64_t duration_ps = std::max(span.duration_ps(), uint64_t{1});
      absl::Format(output_, R"(,"dur":%.17g)", PicosToMicros(duration_ps));

      if (std::optional<uint32_t> color_id = colorer_->GetColor(event)) {
        output_->Append(R"(,"cname":)", TraceViewerColorName(*color_id));
      }

      // FlowV2
      if (event_type == JsonEventCounter::kCompleteEventWithFlow) {
        output_->Append(R"(,"bind_id":)", event.flow_id());
        if (event.has_flow_category()) {
          ContextType type = GetSafeContextType(event.flow_category());
          if (type != ContextType::kGeneric && type != ContextType::kLegacy) {
            const char* category = GetContextTypeString(type);
            output_->Append(R"(,"cat":")", category, R"(")");
          }
        }
        switch (event.flow_entry_type()) {
          case TraceEvent::FLOW_NONE:
            // The caller prevents this case from happenning.
            break;
          case TraceEvent::FLOW_START:
            output_->Append(R"(,"flow_out":true)");
            break;
          case TraceEvent::FLOW_MID:
            output_->Append(R"(,"flow_in":true,"flow_out":true)");
            break;
          case TraceEvent::FLOW_END:
            output_->Append(R"(,"flow_in":true)");
            break;
        }
      }
      output_->Append(R"(,"ph":"X")");
    } else {
      event_type = event.has_flow_id() ? JsonEventCounter::kAsyncEvent
                                       : JsonEventCounter::kCounterEvent;
      if (event_type == JsonEventCounter::kCounterEvent) {
        output_->Append(R"(,"ph":"C")");
      } else {  // async events
        output_->Append(R"(,"id":)", event.flow_id());
        if (event.has_flow_category()) {
          ContextType type = GetSafeContextType(event.flow_category());
          const char* category = GetContextTypeString(type);
          output_->Append(R"(,"cat":")", category, R"(")");
        }
        switch (event.flow_entry_type()) {
          case TraceEvent::FLOW_NONE:
            // The caller prevents this case from happenning.
            break;
          case TraceEvent::FLOW_START:
            output_->Append(R"(,"ph":"b")");
            break;
          case TraceEvent::FLOW_END:
            output_->Append(R"(,"ph":"e")");
            break;
          case TraceEvent::FLOW_MID:
            output_->Append(R"(,"ph":"b")");
            async_event.emplace(event);
            async_event->set_flow_entry_type(TraceEvent::FLOW_END);
            async_event->set_timestamp_ps(event.timestamp_ps() +
                                          event.duration_ps());
            async_event->clear_raw_data();
            break;
        }
      }
    }
    WriteArgs(event);
    if (event.has_serial()) {
      output_->Append(R"(,"z":)", event.serial());
    }

    output_->Append("}");
    counter_.Inc(event_type);
    if (async_event) {
      output_->Append(",");
      WriteEvent(*async_event);
    }
  }

 private:
  void WriteArgs(const TraceEvent& event) const {
    if (!event.has_group_id() && !event.has_raw_data()) {
      return;
    }
    output_->Append(R"(,"args":{)");
    std::optional<uint64_t> stack_frames;
    JsonSeparator<IOBuffer> separator(output_);
    if (event.has_group_id()) {
      separator.Add();
      output_->Append(R"("group_id":)", event.group_id());
    }
    if (event.has_raw_data()) {
      RawDataType data;
      data.ParseFromString(event.raw_data());
      switch (data.raw_data_case()) {
        case RawDataType::RAW_DATA_NOT_SET:
          break;
        case RawDataType::kTpuData:
          WriteTpuData<RawDataType, IOBuffer>(data, &separator, output_);
          break;
        case RawDataType::kDmaActivity:
          separator.Add();
          output_->Append(R"("DMA activity":)",
                          ProtoString(data.dma_activity()));
          break;
        case RawDataType::kArgs:
          for (const auto& arg : data.args().arg()) {
            switch (arg.value_case()) {
              case TraceEventArguments::Argument::kStrValue:
                separator.Add();
                WriteArg(arg.name(), arg.str_value());
                break;
              case TraceEventArguments::Argument::kIntValue:
                separator.Add();
                WriteArg(arg.name(), arg.int_value());
                break;
              case TraceEventArguments::Argument::kUintValue:
                separator.Add();
                WriteArg(arg.name(), arg.uint_value());
                break;
              case TraceEventArguments::Argument::kDoubleValue:
                separator.Add();
                WriteArg(arg.name(), arg.double_value());
                break;
              case TraceEventArguments::Argument::kRefValue: {
                const auto& it = trace_.name_table().find(arg.ref_value());
                if (it != trace_.name_table().end()) {
                  // Each event could only have one stack frame.
                  if (absl::StartsWith(it->second, "@@") && !stack_frames) {
                    stack_frames = arg.ref_value();
                  } else {
                    separator.Add();
                    WriteArg(arg.name(), it->second);
                  }
                }
                break;
              }
              case TraceEventArguments::Argument::VALUE_NOT_SET:
                break;
            }
          }
          break;
      }
    }
    output_->Append("}");

    // Write the optional stack frame.
    if (stack_frames.has_value()) {
      output_->Append(R"(,"sf":)", references_.at(*stack_frames), R"()");
    }
  }
  void WriteArg(absl::string_view name, absl::string_view value) const {
    output_->Append(JsonEscape(name), ":", JsonEscape(value));
  }
  void WriteArg(absl::string_view name, uint64_t value) const {
    // Limit beyond which integers converted to 64-bit IEEE floating point may
    // lose accuracy. JavaScript stores all numbers as doubles, quote the value
    // to preserve accuracy.
    // https://en.wikipedia.org/wiki/Double-precision_floating-point_format
    constexpr uint64_t kIeeeLimit = 1ULL << 53;
    if (value > kIeeeLimit) {
      output_->Append(JsonEscape(name), ":\"", value, "\"");
    } else {
      output_->Append(JsonEscape(name), ":", value);
    }
  }
  void WriteArg(absl::string_view name, int64_t value) const {
    // Limit beyond which integers converted to 64-bit IEEE floating point may
    // lose accuracy. JavaScript stores all numbers as doubles, quote the value
    // to preserve accuracy.
    // https://en.wikipedia.org/wiki/Double-precision_floating-point_format
    constexpr uint64_t kIeeeLimit = 1ULL << 53;
    if (abs(value) > kIeeeLimit) {
      output_->Append(JsonEscape(name), ":\"", value, "\"");
    } else {
      output_->Append(JsonEscape(name), ":", value);
    }
  }
  void WriteArg(absl::string_view name, double value) const {
    if (std::isfinite(value)) {
      output_->Append(JsonEscape(name));
      // "%.17g" is the default double format in proto2::util::JsonFormat.
      absl::Format(output_, ":%.17g", value);
    } else if (std::isinf(value)) {
      output_->Append(JsonEscape(name), R"(:"Infinity")");
    } else if (std::isinf(-value)) {
      output_->Append(JsonEscape(name), R"(:"-Infinity")");
    } else {
      output_->Append(JsonEscape(name), R"(:"NaN")");
    }
  }

  const TraceEventsColorerInterface* colorer_;
  const Trace& trace_;
  const std::map<uint64_t, uint64_t>& references_;
  IOBuffer* output_;
  mutable JsonEventCounter counter_;
};

template <typename IOBuffer>
void WriteTasks(const Trace& trace, IOBuffer* output) {
  const auto& tasks = trace.tasks();
  if (tasks.empty()) return;
  output->Append(R"("tasks":[)");
  JsonSeparator<IOBuffer> task_separator(output);
  std::map<uint32_t, Task> ordered_tasks(tasks.begin(), tasks.end());
  for (const auto& entry : ordered_tasks) {
    const uint32_t host_id = entry.first;
    const auto& task = entry.second;

    task_separator.Add();
    output->Append("{");
    JsonSeparator<IOBuffer> field_separator(output);
    field_separator.Add();
    output->Append(R"("host_id":)", host_id);
    if (task.has_changelist()) {
      field_separator.Add();
      output->Append(R"("changelist":)", task.changelist());
    }
    if (task.has_clean_build()) {
      field_separator.Add();
      output->Append(R"("clean_build":)", task.clean_build());
    }
    if (task.has_build_time()) {
      field_separator.Add();
      output->Append(
          R"("build_time":)",
          JsonEscape(absl::FormatTime(absl::FromUnixNanos(task.build_time()),
                                      absl::UTCTimeZone())));
    }
    if (task.has_build_target()) {
      field_separator.Add();
      output->Append(R"("build_target":)", JsonEscape(task.build_target()));
    }
    if (task.has_command_line()) {
      field_separator.Add();
      output->Append(R"("command_line":)", JsonEscape(task.command_line()));
    }
    if (task.has_start_time()) {
      field_separator.Add();
      output->Append(
          R"("start_time":)",
          JsonEscape(absl::FormatTime(absl::FromUnixNanos(task.start_time()),
                                      absl::UTCTimeZone())));
    }
    if (task.has_gtc_freq_hz()) {
      field_separator.Add();
      output->Append(R"("gtc_freq_hz":)", task.gtc_freq_hz());
    }
    if (task.has_tensor_core_freq_hz()) {
      field_separator.Add();
      output->Append(R"("tensor_core_freq_hz":)", task.tensor_core_freq_hz());
    }
    if (task.has_sparse_core_freq_hz()) {
      field_separator.Add();
      output->Append(R"("sparse_core_freq_hz":)", task.sparse_core_freq_hz());
    }
    output->Append("}");
  }
  output->Append("],");
}

template <typename IOBuffer>
void WriteStackFrames(const Trace& trace,
                      const std::map<uint64_t, uint64_t>& references,
                      IOBuffer* output) {
  const auto& name_table = trace.name_table();
  output->Append(R"("stackFrames":{)");
  JsonSeparator<IOBuffer> separator(output);
  for (const auto& [fp, name] : name_table) {
    if (!absl::StartsWith(name, "@@")) continue;
    separator.Add();
    std::string_view name_view = name;
    absl::ConsumePrefix(&name_view, "@@");
    output->Append(R"(")", references.at(fp), R"(":{"name":)",
                   JsonEscape(name_view), R"(})");
  }
  output->Append("},");
}

template <typename IOBuffer>
void WriteDetails(const JsonTraceOptions::Details& details, IOBuffer* output) {
  if (details.empty()) return;
  output->Append(R"("details":[)");
  JsonSeparator<IOBuffer> separator(output);
  for (const auto& detail : details) {
    separator.Add();
    output->Append(R"({"name":)", JsonEscape(detail.first), R"(,"value":)",
                   detail.second ? "true" : "false", "}");
  }
  output->Append("],");
}

template <typename IOBuffer>
void WriteSelectedDeviceIds(
    const absl::optional<absl::flat_hash_set<uint32_t>>& selected_device_ids,
    IOBuffer* output) {
  if (!selected_device_ids.has_value()) return;

  output->Append(R"("selected_device_ids":[)");
  JsonSeparator<IOBuffer> separator(output);
  for (const auto& device_id : selected_device_ids.value()) {
    separator.Add();
    output->Append(device_id);
  }
  output->Append("],");
}

std::map<uint64_t, uint64_t> BuildStackFrameReferences(const Trace& trace);

template <typename IOBuffer>
void WriteReturnedEventsSize(const int events_size, IOBuffer* output) {
  output->Append(R"("returnedEventsSize":)", events_size, R"(,)");
}

template <typename IOBuffer>
void WriteFilteredByVisibility(bool filtered_by_visibility, IOBuffer* output) {
  absl::string_view filtered_by_visibility_str =
      filtered_by_visibility ? "true" : "false";
  output->Append(R"("filteredByVisibility":)", filtered_by_visibility_str,
                 R"(,)");
}

template <typename IOBuffer>
void WriteTraceFullTimespan(const Trace* trace, IOBuffer* output) {
  auto start_time_ms = trace->min_timestamp_ps() / 1000000000.0;
  auto end_time_ms = trace->max_timestamp_ps() / 1000000000.0;
  output->Append(R"("fullTimespan":[)", start_time_ms, R"(,)", end_time_ms,
                 R"(],)");
}

template <typename IOBuffer, typename TraceEventsContainer,
          typename RawDataType>
void TraceEventsToJson(const JsonTraceOptions& options,
                       const TraceEventsContainer& events, IOBuffer* output) {
  // Set the displayTimeUnit to nanoseconds (default is milliseconds), so the UI
  // uses higher-precision when manipulating event times. Note that the
  // timestamps of trace events are always given in microseconds.
  output->Append(
      R"({"displayTimeUnit":"ns","metadata":{"highres-ticks":true},)");

  WriteDetails(options.details, output);
  WriteSelectedDeviceIds(options.selected_device_ids, output);
  WriteReturnedEventsSize(events.NumEvents(), output);
  WriteFilteredByVisibility(events.FilterByVisibility(), output);
  WriteTraceFullTimespan(&events.trace(), output);

  const Trace& trace = events.trace();

  WriteTasks(trace, output);

  auto references = BuildStackFrameReferences(trace);
  if (options.generate_stack_frames) {
    WriteStackFrames(trace, references, output);
  }

  output->Append(R"("traceEvents":[)");
  JsonSeparator<IOBuffer> separator(output);
  // Write metadata events.
  std::map<uint32_t, Device> ordered_devices(trace.devices().begin(),
                                             trace.devices().end());
  for (const auto& [device_id, device] : ordered_devices) {
    if (device.has_name()) {
      separator.Add();
      output->Append(R"({"args":{"name":)", JsonEscape(device.name()),
                     R"(},"name":"process_name","ph":"M","pid":)", device_id,
                     "}");
    }
    separator.Add();
    output->Append(R"({"args":{"sort_index":)", device_id,
                   R"(},"name":"process_sort_index","ph":"M","pid":)",
                   device_id, "}");
    std::map<uint32_t, Resource> ordered_resources(device.resources().begin(),
                                                   device.resources().end());
    for (const auto& [resource_id, resource] : ordered_resources) {
      if (resource.has_name()) {
        separator.Add();
        output->Append(R"({"args":{"name":)", JsonEscape(resource.name()),
                       R"(},"name":"thread_name","ph":"M","pid":)", device_id,
                       R"(,"tid":)", resource_id, "}");
      }
      if (!options.sort_resources_by_name.count(device_id)) {
        separator.Add();
        output->Append(R"({"args":{"sort_index":)", resource_id,
                       R"(},"name":"thread_sort_index","ph":"M","pid":)",
                       device_id, R"(,"tid":)", resource_id, "}");
      }
    }
  }

  TraceEventsColorerInterface* colorer = options.colorer;
  DefaultTraceEventsColorer default_colorer;
  if (colorer == nullptr) colorer = &default_colorer;
  colorer->SetUp(trace);

  // Write events.
  JsonEventWriter<IOBuffer, RawDataType> writer(colorer, trace, references,
                                                output);
  events.ForAllEvents([&](const TraceEvent& event) {
    separator.Add();
    writer.WriteEvent(event);
  });
  output->Append("]}");
}

class IOBufferAdapter {
 public:
  explicit IOBufferAdapter(std::string* output) : output_(output) {}

  template <typename... AV>
  inline void Append(AV&&... args) {
    absl::StrAppend(output_, std::forward<AV>(args)...);
  }

  // Support IOBufferAdapter as a sink object for absl::Format.
  friend void AbslFormatFlush(IOBufferAdapter* buffer, absl::string_view s) {
    buffer->output_->append(s);
  }

 private:
  std::string* output_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_TRACE_VIEWER_TRACE_EVENTS_TO_JSON_H_
