/* Copyright 2020 The TensorFlow Authors All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/trace_events_to_json.h"

#include "include/json/json.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/protobuf/trace_events.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

string ConvertTextFormattedTraceToJson(const string& trace_str) {
  Trace trace;
  ::tensorflow::protobuf::TextFormat::ParseFromString(trace_str, &trace);
  return TraceEventsToJson(trace);
}

Json::Value ToJsonValue(const string& json_str) {
  Json::Value json;
  Json::Reader reader;
  EXPECT_TRUE(reader.parse(json_str, json));
  return json;
}

TEST(TraceEventsToJson, JsonConversion) {
  string json_output = ConvertTextFormattedTraceToJson(R"(
      devices { key: 2 value {
        name: 'D2'
        device_id: 2
        resources { key: 2 value {
          resource_id: 2
          name: 'R2.2'
        } }
      } }
      devices { key: 1 value {
        name: 'D1'
        device_id: 1
        resources { key: 2 value {
          resource_id: 1
          name: 'R1.2'
        } }
      } }

      trace_events {
        device_id: 1
        resource_id: 2
        name: 'E1.2.1'
        timestamp_ps: 100000
        duration_ps: 10000
        args {
          key: 'long_name'
          value: 'E1.2.1 long'
        }
        args {
          key: 'arg2'
          value: 'arg2 val'
        }
      }
      trace_events {
        device_id: 2
        resource_id: 2
        name: 'E2.2.1 # "comment"'
        timestamp_ps: 105000
      }
  )");
  string expected_json = R"(
  {
    "displayTimeUnit": "ns",
    "metadata": { "highres-ticks": true },
    "traceEvents": [
      {"ph":"M", "pid":1, "name":"process_name", "args":{"name":"D1"}},
      {"ph":"M", "pid":1, "name":"process_sort_index", "args":{"sort_index":1}},
      {"ph":"M", "pid":1, "tid":2, "name":"thread_name",
       "args":{"name":"R1.2"}},
      {"ph":"M", "pid":1, "tid":2, "name":"thread_sort_index",
       "args":{"sort_index":2}},
      {"ph":"M", "pid":2, "name":"process_name", "args":{"name":"D2"}},
      {"ph":"M", "pid":2, "name":"process_sort_index", "args":{"sort_index":2}},
      {"ph":"M", "pid":2, "tid":2, "name":"thread_name",
       "args":{"name":"R2.2"}},
      {"ph":"M", "pid":2, "tid":2, "name":"thread_sort_index",
       "args":{"sort_index":2}},

      {
        "ph" : "X",
        "pid" : 1,
        "tid" : 2,
        "name" : "E1.2.1",
        "ts" : 0.1,
        "dur" : 0.01,
        "args" : {"arg2": "arg2 val", "long_name": "E1.2.1 long"}
      },
      {
        "ph" : "X",
        "pid" : 2,
        "tid" : 2,
        "name" : "E2.2.1 # \"comment\"",
        "ts" : 0.105,
        "dur" : 1e-6
      },
      {}
    ]
  })";
  EXPECT_EQ(ToJsonValue(json_output), ToJsonValue(expected_json));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
