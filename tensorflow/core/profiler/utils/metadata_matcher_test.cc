/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/utils/metadata_matcher.h"

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tensorflow::profiler::XEventMetadata;
using ::tensorflow::profiler::XPlane;
using ::tensorflow::profiler::XStatMetadata;

TEST(MetadataMatcherTest, GetHostEventTypeTest) {
  for (int event_type = HostEventType::kFirstHostEventType;
       event_type <= HostEventType::kLastHostEventType; ++event_type) {
    XPlane xplane;
    XEventMetadata& metadata = (*xplane.mutable_event_metadata())[0];
    metadata.set_id(0);
    metadata.set_name(std::string(
        GetHostEventTypeStr(static_cast<HostEventType>(event_type))));
    MetadataMatcher metadata_matcher(
        xplane,
        {{GetHostEventTypeStrMap(), HostEventType::kFirstHostEventType}},
        GetStatTypeStrMap());
    XEvent event;
    event.set_metadata_id(0);
    EXPECT_EQ(metadata_matcher.GetEventType(event), event_type);
  }
}

TEST(MetadataMatcherTest, GetStatTypeTest) {
  for (int stat_type = StatType::kFirstStatType;
       stat_type <= StatType::kLastStatType; ++stat_type) {
    XPlane xplane;
    XStatMetadata& metadata = (*xplane.mutable_stat_metadata())[0];
    metadata.set_id(0);
    metadata.set_name(
        std::string(GetStatTypeStr(static_cast<StatType>(stat_type))));
    MetadataMatcher metadata_matcher(
        xplane,
        {{GetHostEventTypeStrMap(), HostEventType::kFirstHostEventType}},
        GetStatTypeStrMap());
    XStat stat;
    stat.set_metadata_id(0);
    EXPECT_EQ(metadata_matcher.GetStatType(stat), stat_type);
  }
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
