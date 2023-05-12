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
#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_TRACE_VIEWER_TRACE_EVENT_ARGUMENTS_BUILDER_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_TRACE_VIEWER_TRACE_EVENT_ARGUMENTS_BUILDER_H_

#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/core/profiler/protobuf/trace_events_raw.pb.h"

namespace tensorflow {
namespace profiler {

// Helper class for adding arguments to TraceEventsArguments.
class TraceEventArgumentsBuilder {
 public:
  explicit TraceEventArgumentsBuilder(TraceEventArguments* args)
      : args_(args) {}

  void Append(absl::string_view key, absl::string_view value) {
    auto* arg = args_->add_arg();
    arg->set_name(key.data(), key.size());
    arg->set_str_value(value.data(), value.size());
  }

  void Append(absl::string_view key, int64_t value) {
    auto* arg = args_->add_arg();
    arg->set_name(key.data(), key.size());
    arg->set_int_value(value);
  }

  void Append(absl::string_view key, uint64_t value) {
    auto* arg = args_->add_arg();
    arg->set_name(key.data(), key.size());
    arg->set_uint_value(value);
  }

  void Append(absl::string_view key, double value) {
    auto* arg = args_->add_arg();
    arg->set_name(key.data(), key.size());
    arg->set_double_value(value);
  }

 private:
  TraceEventArguments* args_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_TRACE_VIEWER_TRACE_EVENT_ARGUMENTS_BUILDER_H_
