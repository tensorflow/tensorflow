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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_TRACE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_TRACE_H_

#include <cstdint>

#include "tensorflow/lite/experimental/ruy/block_map.h"
#include "tensorflow/lite/experimental/ruy/side_pair.h"

namespace ruy {

struct Trace;

#ifdef RUY_TRACE

struct TracingContext {
  bool initialized = false;
  bool enabled = false;
  int filter_shape_rows = 0;
  int filter_shape_cols = 0;
  int filter_shape_depth = 0;
  Trace* trace = nullptr;
  ~TracingContext();
};

Trace* NewTraceOrNull(TracingContext* context, int rows, int depth, int cols);
void TraceRecordThreadStart(std::uint32_t thread_id, Trace* trace);
void TraceRecordThreadLoopStart(std::uint32_t thread_id, Trace* trace);
void TraceRecordBlockReserved(std::uint32_t thread_id, std::uint32_t block_id,
                              Trace* trace);
void TraceRecordBlockPacked(std::uint32_t thread_id, Side side, int block,
                            Trace* trace);
void TraceRecordBlockFinished(std::uint32_t thread_id, std::uint32_t block_id,
                              Trace* trace);
void TraceRecordThreadEnd(std::uint32_t thread_id, Trace* trace);
void TraceRecordStart(Trace* trace);
void TraceRecordExecute(const BlockMap& block_map, Trace* trace);
void TraceRecordEnd(Trace* trace);

#else

struct TracingContext {};

inline Trace* NewTraceOrNull(TracingContext*, int, int, int) { return nullptr; }
inline void TraceRecordThreadStart(std::uint32_t, Trace*) {}
inline void TraceRecordThreadLoopStart(std::uint32_t, Trace*) {}
inline void TraceRecordBlockReserved(std::uint32_t, std::uint32_t, Trace*) {}
inline void TraceRecordBlockPacked(std::uint32_t, Side, int, Trace*) {}
inline void TraceRecordBlockFinished(std::uint32_t, std::uint32_t, Trace*) {}
inline void TraceRecordThreadEnd(std::uint32_t, Trace*) {}
inline void TraceRecordStart(Trace*) {}
inline void TraceRecordExecute(const BlockMap&, Trace*) {}
inline void TraceRecordEnd(Trace*) {}

#endif

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_TRACE_H_
