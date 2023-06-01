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

#include "tensorflow/tsl/profiler/convert/trace_events_to_json.h"

#include <string>

#include "json/json.h"
#include "tensorflow/tsl/platform/protobuf.h"
#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/profiler/convert/trace_container.h"
#include "tensorflow/tsl/profiler/protobuf/trace_events.pb.h"

namespace tsl {
namespace profiler {
namespace {

Json::Value ToJsonValue(const std::string& json_str) {
  Json::Value json;
  Json::Reader reader;
  EXPECT_TRUE(reader.parse(json_str, json));
  return json;
}

TEST(TraceEventsToJson, JsonConversion) {
  const std::string metadata_string = R"pb(
    devices {
      key: 2
      value {
        name: 'D2'
        device_id: 2
        resources {
          key: 2
          value { resource_id: 2 name: 'R2.2' }
        }
      }
    }
    devices {
      key: 1
      value {
        name: 'D1'
        device_id: 1
        resources {
          key: 2
          value { resource_id: 1 name: 'R1.2' }
        }
      }
    }
  )pb";

  TraceContainer container;
  EXPECT_TRUE(container.ParseMetadataFromString(metadata_string));

  TraceEvent* event = container.CreateEvent();
  event->set_device_id(1);
  event->set_resource_id(2);
  event->set_name("E1.2.1");
  event->set_timestamp_ps(100000);
  event->set_duration_ps(10000);
  event->mutable_args()->insert({"long_name", "E1.2.1 long"});
  event->mutable_args()->insert({"arg2", "arg2 val"});

  event = container.CreateEvent();
  event->set_device_id(2);
  event->set_resource_id(2);
  event->set_name("E2.2.1 # \"comment\"");
  event->set_timestamp_ps(105000);

  container.CapEvents(2);
  Json::Value json = ToJsonValue(TraceContainerToJson(container));

  Json::Value expected_json = ToJsonValue(R"(
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
  })");

  EXPECT_EQ(json, expected_json);
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
