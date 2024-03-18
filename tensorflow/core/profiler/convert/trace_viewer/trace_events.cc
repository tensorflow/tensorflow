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
#include "tensorflow/core/profiler/convert/trace_viewer/trace_events.h"

#include <stddef.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/internal/endian.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/profiler/convert/trace_viewer/trace_events_filter_interface.h"
#include "tensorflow/core/profiler/convert/trace_viewer/trace_events_util.h"
#include "tensorflow/core/profiler/convert/trace_viewer/trace_viewer_visibility.h"
#include "tensorflow/core/profiler/protobuf/trace_events.pb.h"
#include "tensorflow/core/profiler/protobuf/trace_events_raw.pb.h"
#include "tsl/lib/io/table.h"
#include "tsl/lib/io/table_builder.h"
#include "tsl/lib/io/table_options.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/file_system.h"
#include "tsl/platform/status.h"
#include "tsl/profiler/utils/timespan.h"

namespace tensorflow {
namespace profiler {

namespace {

constexpr uint64_t kLayerResolutions[] = {
    1000000000000ull,  // 1 second.
    100000000000ull,  10000000000ull, 1000000000ull, 100000000ull,
    10000000ull,      1000000ull,     100000ull,     10000ull,
    1000ull,          100ull,         10ull,         1ull,
};

constexpr int NumLevels() { return TF_ARRAYSIZE(kLayerResolutions); }
constexpr uint64_t LayerResolutionPs(unsigned level) {
  // This sometimes gets called in a tight loop, so levels are precomputed.
  return level >= NumLevels() ? 0 : kLayerResolutions[level];
}

// Constants used by the LevelDB Table-based efficient trace viewer storage.
static constexpr char kTraceMetadataKey[] = "/trace";
static constexpr absl::string_view kLevelKey("123456789ABCDEFGHIJKLMNOPQ");
static constexpr size_t kLevelDbKeyLength = 10;

// Level Db table don't allow duplicated keys, so we add a tie break at the last
// bytes. the format is zoom[1B] + timestamp[8B] + repetition[1B]
std::string LevelDbTableKey(int zoom_level, uint64_t timestamp,
                            uint64_t repetition) {
  if (repetition >= 256) return std::string();
  std::string output(kLevelDbKeyLength, 0);
  char* ptr = output.data();
  ptr[0] = kLevelKey[zoom_level];
  // The big-endianness preserve the monotonic order of timestamp when convert
  // to lexigraphical order (of Sstable key namespace).
  uint64_t timestamp_bigendian = absl::big_endian::FromHost64(timestamp);
  memcpy(ptr + 1, &timestamp_bigendian, sizeof(uint64_t));
  ptr[9] = repetition;
  return output;
}

uint64_t TimestampFromLevelDbTableKey(absl::string_view level_db_table_key) {
  DCHECK_EQ(level_db_table_key.size(), kLevelDbKeyLength);
  uint64_t value;  // big endian representation of timestamp.
  memcpy(&value, level_db_table_key.data() + 1, sizeof(uint64_t));
  return absl::big_endian::ToHost64(value);
}

bool ReadTraceMetadata(tsl::table::Iterator* iterator,
                       absl::string_view metadata_key, Trace* trace) {
  if (!iterator->Valid()) return false;
  if (iterator->key() != metadata_key) return false;
  auto serialized_trace = iterator->value();
  return trace->ParseFromArray(serialized_trace.data(),
                               serialized_trace.size());
}

// Returns the total number of events.
inline int32_t NumEvents(
    const std::vector<const TraceEventTrack*>& event_tracks) {
  int32_t num_events = 0;
  for (const auto* track : event_tracks) {
    num_events += track->size();
  }
  return num_events;
}

// Mark events with duplicated timestamp with different serial. This is to
// help front end to deduplicate events during streaming mode. The uniqueness
// is guaranteed by the tuple <device_id, timestamp_ps, serial_number>.
// REQUIRES: events is sorted by timestamp_ps
void MaybeAddEventUniqueId(std::vector<TraceEvent*>& events) {
  uint64_t last_ts = UINT64_MAX;
  uint64_t serial = 0;
  for (TraceEvent* event : events) {
    if (event->timestamp_ps() == last_ts) {
      event->set_serial(++serial);
    } else {
      serial = 0;
    }
    last_ts = event->timestamp_ps();
  }
}

}  // namespace

std::vector<TraceEvent*> MergeEventTracks(
    const std::vector<const TraceEventTrack*>& event_tracks) {
  std::vector<TraceEvent*> events;
  events.reserve(NumEvents(event_tracks));
  nway_merge(event_tracks, std::back_inserter(events), TraceEventsComparator());
  return events;
}

std::vector<std::vector<const TraceEvent*>> GetEventsByLevel(
    const Trace& trace, std::vector<TraceEvent*>& events) {
  MaybeAddEventUniqueId(events);

  constexpr int kNumLevels = NumLevels();

  // Track visibility per zoom level.
  tsl::profiler::Timespan trace_span = TraceSpan(trace);
  std::vector<TraceViewerVisibility> visibility_by_level;
  visibility_by_level.reserve(kNumLevels);
  for (int zoom_level = 0; zoom_level < kNumLevels - 1; ++zoom_level) {
    visibility_by_level.emplace_back(trace_span, LayerResolutionPs(zoom_level));
  }

  std::vector<std::vector<const TraceEvent*>> events_by_level(kNumLevels);
  for (const TraceEvent* event : events) {
    int zoom_level = 0;
    // Find the smallest zoom level on which we can distinguish this event.
    for (; zoom_level < kNumLevels - 1; ++zoom_level) {
      if (visibility_by_level[zoom_level].VisibleAtResolution(*event)) {
        break;
      }
    }
    events_by_level[zoom_level].push_back(event);
    // Record the visibility of this event in all higher zoom levels.
    // An event on zoom level N can make events at zoom levels >N invisible.
    for (++zoom_level; zoom_level < kNumLevels - 1; ++zoom_level) {
      visibility_by_level[zoom_level].SetVisibleAtResolution(*event);
    }
  }
  return events_by_level;
}

// Store the contents of this container in an sstable file. The format is as
// follows:
//
// key                     | value
// trace                   | The Trace-proto trace_
// 0<timestamp><serial>    | Event at timestamp visible at a 10ms resolution
// 1<timestamp><serial>    | Event at timestamp visible at a 1ms resolution
// ...
// 7<timestamp><serial>    | Event at timestamp visible at a 1ns resolution
//
// Note that each event only appears exactly once, at the first layer it's
// eligible for.
tsl::Status DoStoreAsLevelDbTable(
    std::unique_ptr<tsl::WritableFile>& file, const Trace& trace,
    const std::vector<std::vector<const TraceEvent*>>& events_by_level) {
  tsl::table::Options options;
  options.block_size = 20 * 1024 * 1024;
  options.compression = tsl::table::kSnappyCompression;
  tsl::table::TableBuilder builder(options, file.get());

  builder.Add(kTraceMetadataKey, trace.SerializeAsString());

  size_t num_of_events_dropped = 0;  // Due to too many timestamp repetitions.
  for (int zoom_level = 0; zoom_level < events_by_level.size(); ++zoom_level) {
    // The key of level db table have to be monotonically increasing, therefore
    // we make the timestamp repetition count as the last byte of key as tie
    // breaker. The hidden assumption was that there are not too many identical
    // timestamp per resolution, (if there are such duplications, we dropped
    // them if it overflow the last byte).
    uint64_t last_timestamp = std::numeric_limits<uint64_t>::max();
    uint64_t last_timestamp_repetition = 0;
    for (const TraceEvent* event : events_by_level[zoom_level]) {
      // NOTE: temporarily mutate the event for the storage efficiency, the
      // timestamp is restored after serialization.
      // NOTE: runtime-proto-const-cast lint error is bogus because we are
      // casting the top level proto.
      TraceEvent* mutable_event = const_cast<TraceEvent*>(event);
      auto timestamp = mutable_event->timestamp_ps();
      mutable_event->clear_timestamp_ps();
      if (timestamp != last_timestamp) {
        last_timestamp = timestamp;
        last_timestamp_repetition = 0;
      } else {
        ++last_timestamp_repetition;
      }
      auto key =
          LevelDbTableKey(zoom_level, timestamp, last_timestamp_repetition);
      if (!key.empty()) {
        builder.Add(key, mutable_event->SerializeAsString());
      } else {
        ++num_of_events_dropped;
      }
      mutable_event->set_timestamp_ps(timestamp);
    }
  }
  absl::string_view filename;
  TF_RETURN_IF_ERROR(file->Name(&filename));
  LOG(INFO) << "Storing " << trace.num_events() - num_of_events_dropped
            << " as LevelDb table fast file: " << filename << " with "
            << num_of_events_dropped << " events dropped.";

  TF_RETURN_IF_ERROR(builder.Finish());
  return file->Close();
}

tsl::Status DoLoadFromLevelDbTable(
    const std::string& filename,
    std::unique_ptr<TraceEventsFilterInterface> filter,
    std::unique_ptr<TraceVisibilityFilter> visibility,
    int64_t filter_by_visibility_threshold, Trace& trace,
    bool& filter_by_visibility,
    const std::function<TraceEvent*(const TraceEvent&)>& copy_event_to_arena,
    const std::function<void(TraceEvent*)>& add_arena_event) {
  uint64_t file_size;
  TF_RETURN_IF_ERROR(tsl::Env::Default()->GetFileSize(filename, &file_size));

  tensorflow::FileSystem* file_system;
  TF_RETURN_IF_ERROR(
      tsl::Env::Default()->GetFileSystemForFile(filename, &file_system));

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(file_system->NewRandomAccessFile(filename, &file));

  tsl::table::Options options;
  options.block_size = 20 * 1024 * 1024;
  tsl::table::Table* table = nullptr;
  TF_RETURN_IF_ERROR(
      tsl::table::Table::Open(options, file.get(), file_size, &table));
  std::unique_ptr<tsl::table::Table> table_deleter(table);
  std::unique_ptr<tsl::table::Iterator> iterator(table->NewIterator());
  if (iterator == nullptr) return tsl::errors::Unknown("Could not open table");

  // Read the metadata.
  iterator->SeekToFirst();
  if (!ReadTraceMetadata(iterator.get(), kTraceMetadataKey, &trace)) {
    return tsl::errors::Unknown("Could not parse Trace proto");
  }

  if (filter) filter->SetUp(trace);

  tsl::profiler::Timespan visible_span;
  uint64_t container_resolution_ps = 0;

  filter_by_visibility = filter_by_visibility_threshold == -1LL ||
                         !trace.has_num_events() ||
                         trace.num_events() >= filter_by_visibility_threshold;
  if (!filter_by_visibility) {
    visibility.reset();  // disable streaming
  }
  if (visibility) {
    visibility->SetUp(trace);
    visible_span = visibility->VisibleSpan();
    container_resolution_ps = visibility->ResolutionPs();
  } else {
    visible_span = TraceSpan(trace);
  }

  // Read events at the different zoom levels.
  std::vector<std::unique_ptr<std::vector<TraceEvent*>>> loaded_events_by_level;
  size_t filtered = 0;
  TraceEvent event;  // Declared outside of the loop to avoid repeated calls to
                     // the constructor and destructor in the loop body. Cleared
                     // by every call to ParseFromCord.
  for (int i = 0;; ++i) {
    loaded_events_by_level.emplace_back(
        std::make_unique<std::vector<TraceEvent*>>());
    auto& loaded_events = *loaded_events_by_level.back();
    uint64_t resolution_ps = LayerResolutionPs(i);
    // Seek to the first element that might be in range. For the initial zoom
    // level, we don't know any bounds as events might be arbitrarily large.
    uint64_t min_timestamp_ps = 0;
    if (i > 0 && visible_span.begin_ps() > LayerResolutionPs(i - 1)) {
      min_timestamp_ps = visible_span.begin_ps() - LayerResolutionPs(i - 1);
    }
    iterator->Seek(LevelDbTableKey(i, i == 0 ? 0 : min_timestamp_ps, 0));
    while (iterator->Valid() && iterator->key().at(0) == kLevelKey[i]) {
      auto serialized_event = iterator->value();
      if (!event.ParseFromArray(serialized_event.data(),
                                serialized_event.size())) {
        return tsl::errors::Unknown("Could not parse TraceEvent proto");
      }
      uint64_t timestamp = TimestampFromLevelDbTableKey(iterator->key());
      event.set_timestamp_ps(timestamp);
      if (event.timestamp_ps() > visible_span.end_ps()) {
        // This (and all following) events are outside of our window.
        break;
      }
      // Filter before copying to the arena as it does not require sorting.
      if (!filter || !filter->Filter(event)) {
        loaded_events.push_back(copy_event_to_arena(event));
      } else {
        ++filtered;
      }
      iterator->Next();
    }
    if (container_resolution_ps >= resolution_ps) {
      // No need to read further, the resolution we just loaded already exceeds
      // the desired resolution.
      break;
    }
  }

  // We have loaded events from different zoom levels. Sort them by timestamp
  // so visibility filtering works as expected.
  std::vector<TraceEvent*> loaded_events;
  nway_merge(loaded_events_by_level, std::back_inserter(loaded_events),
             TraceEventsComparator());
  loaded_events_by_level.clear();

  LOG(INFO) << "Loaded " << loaded_events.size() << " events after filtering "
            << filtered << " events from LevelDb fast file: " << filename;
  size_t visible_events_count = 0;
  for (TraceEvent* event : loaded_events) {
    if (!visibility || !visibility->Filter(*event)) {
      add_arena_event(event);
      ++visible_events_count;
    }
  }
  LOG(INFO) << "Added " << visible_events_count
            << " visible events from LevelDb fast file: " << filename;
  return tsl::OkStatus();
}

}  // namespace profiler
}  // namespace tensorflow
