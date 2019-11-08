/* Copyright 2019 Google LLC. All Rights Reserved.

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

#include "tensorflow/lite/experimental/ruy/trace.h"

#include <algorithm>
#include <cerrno>  // IWYU pragma: keep
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tensorflow/lite/experimental/ruy/check_macros.h"
#include "tensorflow/lite/experimental/ruy/side_pair.h"
#include "tensorflow/lite/experimental/ruy/time.h"

namespace ruy {

#ifdef RUY_TRACE

enum class TraceEvent : std::uint8_t {
  kNone,
  kThreadStart,
  kThreadLoopStart,
  kThreadEnd,
  kBlockReserved,
  kBlockPackedLhs,
  kBlockPackedRhs,
  kBlockFinished
};

struct TraceEntry {
  TimePoint time_point;
  TraceEvent event;
  // ruy-internal thread id i.e. contiguous index into array of threads,
  // with 0 designating the main thread.
  std::uint16_t thread_id = 0;
  // Additional parameters whose meaning depends on the 'event' type.
  std::uint32_t params[1];
};

struct Trace {
  BlockMap block_map;
  // During recording, to avoid having to use locks or atomics, we let
  // each thread append to its own specific vector.
  std::vector<std::vector<TraceEntry>> thread_specific_entries;
  // Global vector of entries into which we coalesce thread_specific_entries
  // after recording is finished, when dumping a trace. See
  // AggregateThreadSpecificEntries.
  std::vector<TraceEntry> entries;
  TimePoint time_start;
  TimePoint time_execute;
  TimePoint time_end;
};

namespace {

// Coalesce Trace::thread_specific_entries into Trace::entries.
void AggregateThreadSpecificEntries(Trace* trace) {
  RUY_CHECK(trace->entries.empty());
  for (auto& thread_specific_entries_vector : trace->thread_specific_entries) {
    for (const TraceEntry& entry : thread_specific_entries_vector) {
      trace->entries.push_back(entry);
    }
    thread_specific_entries_vector.clear();
  }
}

// Sort Trace::entries by ascending time. In case of equal timepoints,
// sort by some semi-arbitrary ordering of event types.
void Sort(Trace* trace) {
  std::sort(std::begin(trace->entries), std::end(trace->entries),
            [](const TraceEntry& a, const TraceEntry& b) -> bool {
              return a.time_point < b.time_point ||
                     (a.time_point == b.time_point &&
                      static_cast<int>(a.event) < static_cast<int>(b.event));
            });
}

// Dump a trace. Assumes that AggregateThreadSpecificEntries and Sort have
// already been called on it.
//
// On some architectures long long ints are not same as std::int64_t, and
// time is printed as %lld, so static_casts are necessary.
void Dump(const Trace& trace) {
  const char* trace_filename = getenv("RUY_TRACE_FILE");
  FILE* trace_file = trace_filename ? fopen(trace_filename, "w") : stderr;
  if (!trace_file) {
    fprintf(stderr, "Failed to open %s for write, errno=%d\n", trace_filename,
            errno);
    RUY_CHECK(false);
  }
  fprintf(trace_file, "thread_count:%d\n", trace.block_map.thread_count);
  fprintf(trace_file, "rows:%d\n", trace.block_map.dims[Side::kLhs]);
  fprintf(trace_file, "cols:%d\n", trace.block_map.dims[Side::kRhs]);
  fprintf(trace_file, "Execute: %lld\n",
          static_cast<long long int>(
              ToInt64Nanoseconds(trace.time_execute - trace.time_start)));
  for (const TraceEntry& entry : trace.entries) {
    long long int time = static_cast<long long int>(
        ToInt64Nanoseconds(entry.time_point - trace.time_start));
    switch (entry.event) {
      case TraceEvent::kThreadStart:
        fprintf(trace_file, "ThreadStart: %lld, %d\n", time, entry.thread_id);
        break;
      case TraceEvent::kThreadLoopStart:
        fprintf(trace_file, "ThreadLoopStart: %lld, %d\n", time,
                entry.thread_id);
        break;
      case TraceEvent::kThreadEnd:
        fprintf(trace_file, "ThreadEnd: %lld, %d\n", time, entry.thread_id);
        break;
      case TraceEvent::kBlockReserved: {
        std::uint32_t block_id = entry.params[0];
        SidePair<int> block;
        GetBlockByIndex(trace.block_map, block_id, &block);
        SidePair<int> start, end;
        GetBlockMatrixCoords(trace.block_map, block, &start, &end);
        fprintf(trace_file,
                "BlockReserved: %lld, %d, %d, %d, %d, %d, %d, %d, %d\n", time,
                entry.thread_id, block_id, block[Side::kLhs], block[Side::kRhs],
                start[Side::kLhs], start[Side::kRhs], end[Side::kLhs],
                end[Side::kRhs]);
        break;
      }
      case TraceEvent::kBlockPackedLhs: {
        std::uint32_t block = entry.params[0];
        int start, end;
        GetBlockMatrixCoords(Side::kLhs, trace.block_map, block, &start, &end);
        fprintf(trace_file, "BlockPackedLhs: %lld, %d, %d, %d, %d\n", time,
                entry.thread_id, block, start, end);
        break;
      }
      case TraceEvent::kBlockPackedRhs: {
        std::uint32_t block = entry.params[0];
        int start, end;
        GetBlockMatrixCoords(Side::kRhs, trace.block_map, block, &start, &end);
        fprintf(trace_file, "BlockPackedRhs: %lld, %d, %d, %d, %d\n", time,
                entry.thread_id, block, start, end);
        break;
      }
      case TraceEvent::kBlockFinished: {
        std::uint32_t block_id = entry.params[0];
        SidePair<int> block;
        GetBlockByIndex(trace.block_map, block_id, &block);
        fprintf(trace_file, "BlockFinished: %lld, %d, %d, %d, %d\n", time,
                entry.thread_id, block_id, block[Side::kLhs],
                block[Side::kRhs]);
        break;
      }
      default:
        RUY_CHECK(false);
    }
  }
  fprintf(trace_file, "End: %lld\n",
          static_cast<long long int>(
              ToInt64Nanoseconds(trace.time_end - trace.time_start)));
  if (trace_filename) {
    fclose(trace_file);
  }
}

}  // anonymous namespace

// Get a Trace object to record to, or null of tracing is not enabled.
Trace* NewTraceOrNull(TracingContext* tracing, int rows, int depth, int cols) {
  if (!tracing->initialized) {
    tracing->initialized = true;
    tracing->enabled = getenv("RUY_TRACE");
    if (!tracing->enabled) {
      return nullptr;
    }
    if (getenv("RUY_TRACE_FILTER_ROWS")) {
      tracing->filter_shape_rows = std::stoi(getenv("RUY_TRACE_FILTER_ROWS"));
    }
    if (getenv("RUY_TRACE_FILTER_DEPTH")) {
      tracing->filter_shape_depth = std::stoi(getenv("RUY_TRACE_FILTER_DEPTH"));
    }
    if (getenv("RUY_TRACE_FILTER_COLS")) {
      tracing->filter_shape_cols = std::stoi(getenv("RUY_TRACE_FILTER_COLS"));
    }
  }
  if (!tracing->enabled) {
    return nullptr;
  }
  if (tracing->filter_shape_rows && rows != tracing->filter_shape_rows) {
    return nullptr;
  }
  if (tracing->filter_shape_depth && depth != tracing->filter_shape_depth) {
    return nullptr;
  }
  if (tracing->filter_shape_cols && cols != tracing->filter_shape_cols) {
    return nullptr;
  }
  // Delete any existing trace.
  delete tracing->trace;
  // Create a new one.
  tracing->trace = new Trace;
  return tracing->trace;
}

// The trace recorded on a context is finalized and dumped by
// this TracingContext destructor.
//
// The idea of dumping on context destructor is that typically one wants to
// run many matrix multiplications, e.g. to hit a steady state in terms of
// performance characteristics, but only trace the last repetition of the
// workload, when that steady state was attained.
TracingContext::~TracingContext() {
  if (trace) {
    AggregateThreadSpecificEntries(trace);
    Sort(trace);
    Dump(*trace);
  }
  delete trace;
}

void TraceRecordStart(Trace* trace) {
  if (trace) {
    trace->time_start = Now();
  }
}

void TraceRecordExecute(const BlockMap& block_map, Trace* trace) {
  if (trace) {
    trace->time_execute = Now();
    trace->block_map = block_map;
    trace->thread_specific_entries.resize(block_map.thread_count);
    for (int thread = 0; thread < block_map.thread_count; thread++) {
      trace->thread_specific_entries[thread].clear();
      // Reserve some large size to avoid frequent heap allocations
      // affecting the recorded timings.
      trace->thread_specific_entries[thread].reserve(16384);
    }
  }
}

void TraceRecordEnd(Trace* trace) {
  if (trace) {
    trace->time_end = Now();
  }
}

void TraceRecordThreadStart(std::uint32_t thread_id, Trace* trace) {
  if (trace) {
    TraceEntry entry;
    entry.event = TraceEvent::kThreadStart;
    entry.time_point = Now();
    entry.thread_id = thread_id;
    trace->thread_specific_entries[thread_id].push_back(entry);
  }
}

void TraceRecordThreadLoopStart(std::uint32_t thread_id, Trace* trace) {
  if (trace) {
    TraceEntry entry;
    entry.event = TraceEvent::kThreadLoopStart;
    entry.time_point = Now();
    entry.thread_id = thread_id;
    trace->thread_specific_entries[thread_id].push_back(entry);
  }
}

void TraceRecordBlockReserved(std::uint32_t thread_id, std::uint32_t block_id,
                              Trace* trace) {
  if (trace) {
    TraceEntry entry;
    entry.event = TraceEvent::kBlockReserved;
    entry.time_point = Now();
    entry.thread_id = thread_id;
    entry.params[0] = block_id;
    trace->thread_specific_entries[thread_id].push_back(entry);
  }
}

void TraceRecordBlockPacked(std::uint32_t thread_id, Side side, int block,
                            Trace* trace) {
  if (trace) {
    TraceEntry entry;
    entry.event = side == Side::kLhs ? TraceEvent::kBlockPackedLhs
                                     : TraceEvent::kBlockPackedRhs;
    entry.time_point = Now();
    entry.thread_id = thread_id;
    entry.params[0] = block;
    trace->thread_specific_entries[thread_id].push_back(entry);
  }
}

void TraceRecordBlockFinished(std::uint32_t thread_id, std::uint32_t block_id,
                              Trace* trace) {
  if (trace) {
    TraceEntry entry;
    entry.event = TraceEvent::kBlockFinished;
    entry.time_point = Now();
    entry.thread_id = thread_id;
    entry.params[0] = block_id;
    trace->thread_specific_entries[thread_id].push_back(entry);
  }
}

void TraceRecordThreadEnd(std::uint32_t thread_id, Trace* trace) {
  if (trace) {
    TraceEntry entry;
    entry.event = TraceEvent::kThreadEnd;
    entry.time_point = Now();
    entry.thread_id = thread_id;
    trace->thread_specific_entries[thread_id].push_back(entry);
  }
}

#endif

}  // namespace ruy
