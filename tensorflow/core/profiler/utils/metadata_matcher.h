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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_METADATA_MATCHER_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_METADATA_MATCHER_H_

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

// Builds mapping between metadata ids and interesting event and stat types.
// Event and stat types are represented in integer ids. Multiple spans of event
// types can be passed with offset values (i.e., first_event_type) to be
// used to calculate integer ids for event types. Spans and offset values are
// expected to result in a unique integer id for each event type.
class MetadataMatcher {
 public:
  explicit MetadataMatcher(
      const XPlane& xplane,
      const std::vector<std::pair<const absl::Span<const absl::string_view>,
                                  /*first_event_type*/ int>>&
          event_type_metadata_maps,
      const absl::Span<const absl::string_view> stat_type_str_map);

  // Returns EventType if input is one of interesting event types.
  // Otherwise, it returns kUnknownEventType.
  int GetEventType(const XEvent& xevent) const {
    return gtl::FindWithDefault(id_to_event_type_map_, xevent.metadata_id(),
                                /*kUnknownEventType*/ 0);
  }

  // Overload of GetEventType function.
  // Returns EventType if input is one of interesting event types.
  // Otherwise, it returns kUnknownEventType.
  int GetEventType(int64 metadata_id) const {
    return gtl::FindWithDefault(id_to_event_type_map_, metadata_id,
                                /*kUnknownEventType*/ 0);
  }

  // Returns metadata id if xplane has the input event type.
  absl::optional<int64> GetEventMetadataId(int event_type) const {
    if (const int64* id = gtl::FindOrNull(event_type_to_id_map_, event_type)) {
      return *id;
    }
    return absl::nullopt;
  }

  // Returns StatType if input is one of interesting stat types.
  // Otherwise, it returns kUnknownStatType.
  int GetStatType(const XStat& xstat) const {
    return gtl::FindWithDefault(id_to_stat_type_map_, xstat.metadata_id(),
                                /*kUnknownStatType*/ 0);
  }

  // Returns metadata id if xplane has the input stat type.
  absl::optional<int64> GetStatMetadataId(int stat_type) const {
    if (const int64* id = gtl::FindOrNull(stat_type_to_id_map_, stat_type)) {
      return *id;
    }
    return absl::nullopt;
  }

  const XStat* GetStat(const XEvent& event, int stat_type) const;

  absl::optional<std::tuple<const XStat*, const XStat*>> GetStats(
      const XEvent& event, int first_stat_type, int second_stat_type) const;

  absl::optional<std::tuple<const XStat*, const XStat*, const XStat*>> GetStats(
      const XEvent& event, int first_stat_type, int second_stat_type,
      int third_stat_type) const;

  absl::optional<int64> GetIntStatValue(const XEvent& event,
                                        int stat_type) const;

 private:
  // Maps from metada ids to interesting event and stat types.
  // Uninteresting event and stat types are not cached in these maps and
  // considered to be kUnknown*.
  const absl::flat_hash_map<int64, int> id_to_event_type_map_;
  const absl::flat_hash_map<int64, int> id_to_stat_type_map_;
  // Reverse of the above.
  const absl::flat_hash_map<int, int64> event_type_to_id_map_;
  const absl::flat_hash_map<int, int64> stat_type_to_id_map_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_METADATA_MATCHER_H_
