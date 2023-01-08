/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/tensorflow_profiler_logger.h"

#include <stdlib.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <string>

#include "base/addressmap.h"
#include "base/examine_stack.h"
#include "base/low_level_alloc.h"
#include "base/malloc_hook.h"
#include "absl/base/call_once.h"
#include "absl/debugging/stacktrace.h"
#include "absl/debugging/symbolize.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace {

struct Statistics {
  int64_t total_bytes_allocated = 0LL;
  int64_t peak_bytes_in_use = 0LL;
};
static Statistics g_stat_heap;
static Statistics g_stat_dynamic;
static Statistics g_stat_arena;

static char g_current_op_name[256];

static absl::Mutex g_api_mutex(
    absl::kConstInit);  // To make sure public APIs are called synchronously.

struct HeapEvent {
  int64_t timestamp;
  const void* ptr;
  void* stack[3];
  size_t size;
  bool is_alloc;
};

constexpr int kThresholdToCapture = 1024 * 8;
constexpr int MAX_HEAP_EVENTS = 200000;
static HeapEvent g_heap_events[MAX_HEAP_EVENTS];

static std::atomic<int> g_heap_event_idx = 0;
static std::atomic<bool> g_pause_heap_monitor = false;
static absl::once_flag g_install_hooks_once;

// Low-level allocation that bypasses the hooks.
static LowLevelAlloc::Arena* g_map_memory;

static void* RawMalloc(size_t bytes) {
  return LowLevelAlloc::AllocWithArena(bytes, g_map_memory);
}

static void RawFree(void* p) { LowLevelAlloc::Free(p); }

// Address map recorded by OnMemoryAlloc() hook.
static AddressMap<size_t>* g_allocation_map;

// Adds memory trace information for TensorFlow profiler.
// `stat`: Statistics object for the (de)allocation.
// `is_allocating`: Whether memory is being allocated or deallocated.
// `allocation_bytes`: The number of bytes being allocated or deallocated.
// `requested_bytes`: The number of bytes requested for allocation/deallocation.
// `tensor_id`: A unique ID for the tensor being allocated or deallocated.
//              Usually the memory address should be used.
// `name`: The name of the tensor being allocated or deallocated.
// `dims`: The dimension of the tensor in a string form.
std::string AddTraceMeInternal(Statistics* stat, bool is_allocating,
                               const std::string& allocator_name,
                               int64_t tensor_id, const std::string& name,
                               const std::string& dims,
                               int64_t allocation_bytes,
                               int64_t requested_bytes) {
  if (is_allocating) {
    stat->total_bytes_allocated += allocation_bytes;
  } else {
    stat->total_bytes_allocated -= allocation_bytes;
  }
  stat->peak_bytes_in_use =
      std::max(stat->peak_bytes_in_use, stat->total_bytes_allocated);
  int64_t total_bytes_allocated = stat->total_bytes_allocated;
  int64_t peak_bytes_in_use = stat->peak_bytes_in_use;

  std::string res = tensorflow::profiler::TraceMeEncode(
      is_allocating ? "MemoryAllocation" : "MemoryDeallocation",
      // Note that all of these fields are necessary for profiling UI.
      {{"allocator_name", allocator_name},  // name shown on 'Memory ID'
       {"bytes_allocated", total_bytes_allocated},
       {"peak_bytes_in_use", peak_bytes_in_use},
       {"requested_bytes", requested_bytes},
       {"allocation_bytes", allocation_bytes},
       // Note: addr is used as a key to match alloc and dealloc.
       {"addr", tensor_id},
       // Note that we're using tensor.name not op name here.
       {"tf_op", name},
       {"shape", dims}});
  // Note: bytes_reserved, fragmentation, data_type, region_type, id
  // can be potentially useful but not added.
  return res;
}

void AddTraceMe(bool is_allocating, TfLiteTensor* tensor,
                size_t allocation_bytes) {
  if (tensor == nullptr || allocation_bytes == 0) return;
  int64_t tensor_id = reinterpret_cast<int64_t>(tensor->data.raw);
  std::string name;
  if (g_current_op_name[0]) {
    name = g_current_op_name;
  }
  if (tensor->name) {
    name += ":";
    name += tensor->name;
  }
  std::string dims = tensor->dims ? GetShapeDebugString(tensor->dims) : "[]";
  int64_t requested_bytes = is_allocating ? allocation_bytes : 0;
  const std::string allocator_name = "_tflite_native_dynamic";

  tensorflow::profiler::TraceMe::InstantActivity(
      [is_allocating, allocator_name, tensor_id, name, dims, allocation_bytes,
       requested_bytes]() {
        return AddTraceMeInternal(&g_stat_dynamic, is_allocating,
                                  allocator_name, tensor_id, name, dims,
                                  allocation_bytes, requested_bytes);
      },
      /*level=*/tensorflow::profiler::TraceMeLevel::kInfo);
}

void AddArenaTrace(bool is_allocating, int subgraph_index, int arena_id,
                   size_t allocation_bytes) {
  std::string name = "Subgraph" + std::to_string(subgraph_index);
  int64_t tensor_id = arena_id;
  std::string dims = "";
  int64_t requested_bytes = is_allocating ? allocation_bytes : 0;
  const std::string allocator_name = "_tflite_arena";

  tensorflow::profiler::TraceMe::InstantActivity(
      [is_allocating, allocator_name, tensor_id, name, dims, allocation_bytes,
       requested_bytes]() {
        return AddTraceMeInternal(&g_stat_arena, is_allocating, allocator_name,
                                  tensor_id, name, dims, allocation_bytes,
                                  requested_bytes);
      },
      /*level=*/tensorflow::profiler::TraceMeLevel::kInfo);
}

void AddHeapTraceMe(int64_t now, const char* op_name, bool is_allocating,
                    const void* address, size_t allocation_bytes) {
  int64_t tensor_id = reinterpret_cast<int64_t>(address);
  const std::string name = op_name;
  const std::string dims = "[]";
  int64_t requested_bytes = is_allocating ? allocation_bytes : 0;
  const std::string allocator_name = "native_heap";

  std::string res =
      AddTraceMeInternal(&g_stat_heap, is_allocating, allocator_name, tensor_id,
                         name, dims, allocation_bytes, requested_bytes);

  tensorflow::profiler::TraceMeRecorder::Record(
      {res, /*start_time=*/now, /*end_time=*/now});
}

char* GetOpnameFromStacks(void* stack[3]) {
  char symbol1[64];
  if (!absl::Symbolize(stack[0], symbol1, sizeof(symbol1))) {
    snprintf(symbol1, sizeof(symbol1), "%p", stack[0]);
  }
  char symbol2[64];
  if (!absl::Symbolize(stack[1], symbol2, sizeof(symbol2))) {
    snprintf(symbol2, sizeof(symbol2), "%p", stack[1]);
  }
  char symbol3[64];
  if (!absl::Symbolize(stack[2], symbol3, sizeof(symbol3))) {
    snprintf(symbol3, sizeof(symbol3), "%p", stack[2]);
  }
  static char op_name[256];
  snprintf(op_name, sizeof(op_name), "%s/%s/%s", symbol1, symbol2, symbol3);
  return op_name;
}

// Hook for malloc().
void OnMemoryAlloc(const void* ptr, size_t num_bytes) {
  if (g_pause_heap_monitor || num_bytes < kThresholdToCapture ||
      g_heap_event_idx >= MAX_HEAP_EVENTS)
    return;

  HeapEvent* current_event = &g_heap_events[g_heap_event_idx++];
  current_event->timestamp = absl::GetCurrentTimeNanos();
  current_event->ptr = ptr;
  current_event->size = num_bytes;
  g_allocation_map->Insert(ptr, num_bytes);
  current_event->is_alloc = true;
  absl::GetStackTrace(current_event->stack,
                      /* max_depth = */ ABSL_ARRAYSIZE(current_event->stack),
                      /* skip_count = */ 3);
}

// Hook for free().
void OnMemoryDealloc(const void* ptr) {
  if (g_pause_heap_monitor || ptr == nullptr ||
      g_heap_event_idx >= MAX_HEAP_EVENTS)
    return;
  size_t free_size;
  if (!g_allocation_map->FindAndRemove(ptr, &free_size)) return;

  HeapEvent* current_event = &g_heap_events[g_heap_event_idx++];
  current_event->timestamp = absl::GetCurrentTimeNanos();
  current_event->ptr = ptr;
  current_event->size = 0;  // Will figure out later.
  current_event->size = free_size;
  current_event->is_alloc = false;
  absl::GetStackTrace(current_event->stack,
                      /* max_depth = */ ABSL_ARRAYSIZE(current_event->stack),
                      /* skip_count = */ 3);
}

// Set g_pause_heap_monitor to true and returns the old value.
inline bool DisableHeapMonitor() ABSL_EXCLUSIVE_LOCKS_REQUIRED(g_api_mutex) {
  bool old_g_heap_monitor = g_pause_heap_monitor;
  g_pause_heap_monitor = true;
  return old_g_heap_monitor;
}

// Restore g_pause_heap_monitor to the old status.
inline void RestoreHeapMonitor(bool old_g_heap_monitor)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(g_api_mutex) {
  g_pause_heap_monitor = old_g_heap_monitor;
}

}  // namespace

void OnTfLiteOpPrepare(const char* op_name, int subgraph_index,
                       int node_index) {
  snprintf(g_current_op_name, sizeof(g_current_op_name), "%sPrepare_%d",
           op_name, node_index);
  // Updates TF's current annotation object by creating scoped annotation obj.
  tensorflow::profiler::ScopedMemoryDebugAnnotation annotation(
      g_current_op_name);
}

tensorflow::profiler::TraceMe* OnTfLiteSubgraphInvoke(const char* name,
                                                      int index) {
  absl::MutexLock lock(&g_api_mutex);

  absl::call_once(g_install_hooks_once, [] {
    if (g_map_memory == nullptr) g_map_memory = LowLevelAlloc::NewArena(0);
    g_allocation_map = new AddressMap<size_t>(RawMalloc, RawFree);

    MallocHook::AddNewHook(&OnMemoryAlloc);
    MallocHook::AddDeleteHook(&OnMemoryDealloc);
  });

  // Disable heap monitoring to ignore heap activity of this function.
  bool old_g_heap_monitor = DisableHeapMonitor();

  tensorflow::profiler::TraceMe* trace_me =
      new tensorflow::profiler::TraceMe([name, index]() {
        char eventName[256];
        snprintf(eventName, sizeof(eventName), "Subgraph%d", index);
        return tensorflow::profiler::TraceMeEncode(
            eventName, {{"subgraph_name", name}, {"subgraph_index", index}});
      });
  RestoreHeapMonitor(old_g_heap_monitor);
  return trace_me;
}

void PauseHeapMonitoring(bool pause) {
  absl::MutexLock lock(&g_api_mutex);

  g_pause_heap_monitor = pause;
}

void OnTfLiteInterpreterEnd() {
  absl::MutexLock lock(&g_api_mutex);

  MallocHook::RemoveNewHook(&OnMemoryAlloc);
  MallocHook::RemoveDeleteHook(&OnMemoryDealloc);

  printf("Heap monitor captured %d events\n", g_heap_event_idx.load());
  for (int i = 0; i < g_heap_event_idx; i++) {
    HeapEvent* event = &g_heap_events[i];
    AddHeapTraceMe(event->timestamp, GetOpnameFromStacks(event->stack),
                   event->is_alloc, event->ptr, event->size);
  }
}

void OnTfLiteSubgraphInvokeEnd(tensorflow::profiler::TraceMe* trace_me) {
  absl::MutexLock lock(&g_api_mutex);

  // Disable heap monitoring to ignore heap activity of this function.
  bool old_g_heap_monitor = DisableHeapMonitor();
  delete trace_me;
  RestoreHeapMonitor(old_g_heap_monitor);
}

tensorflow::profiler::TraceMe* OnTfLiteOpInvoke(const char* op_name,
                                                int subgraph_index,
                                                int node_index) {
  absl::MutexLock lock(&g_api_mutex);

  // Disable heap monitoring to ignore heap activity of this function.
  bool old_g_heap_monitor = DisableHeapMonitor();

  snprintf(g_current_op_name, sizeof(g_current_op_name), "%s_%d", op_name,
           node_index);
  // Updates TF's current annotation object by creating scoped annotation obj.
  tensorflow::profiler::ScopedMemoryDebugAnnotation annotation(
      g_current_op_name);

  tensorflow::profiler::TraceMe* trace_me = new tensorflow::profiler::TraceMe(
      [op_name, subgraph_index, node_index]() {
        char eventName[256];
        // TF ops should have "<detail>:<op_name>" format.
        snprintf(eventName, sizeof(eventName), "%s:%s", op_name, op_name);
        return tensorflow::profiler::TraceMeEncode(
            eventName, {{"is_eager", 0},
                        {"subgraph_index", subgraph_index},
                        {"node_index", node_index}});
      });
  RestoreHeapMonitor(old_g_heap_monitor);
  return trace_me;
}

void OnTfLiteOpInvokeEnd(tensorflow::profiler::TraceMe* trace_me) {
  absl::MutexLock lock(&g_api_mutex);

  // Disable heap monitoring to ignore heap activity of this function.
  bool old_g_heap_monitor = DisableHeapMonitor();
  delete trace_me;
  RestoreHeapMonitor(old_g_heap_monitor);
}

void OnTfLiteTensorAlloc(TfLiteTensor* tensor, size_t num_bytes) {
  absl::MutexLock lock(&g_api_mutex);

  AddTraceMe(/*is_allocating=*/true, tensor, num_bytes);
}

void OnTfLiteTensorDealloc(TfLiteTensor* tensor) {
  absl::MutexLock lock(&g_api_mutex);

  if (tensor != nullptr) {
    size_t num_bytes = tensor->bytes;
    AddTraceMe(/*is_allocating=*/false, tensor, num_bytes);
  }
}

void OnTfLiteArenaAlloc(int subgraph_index, int arena_id, size_t num_bytes) {
  absl::MutexLock lock(&g_api_mutex);

  if (num_bytes == 0) return;
  AddArenaTrace(/*is_allocating=*/true, subgraph_index, arena_id, num_bytes);
}

void OnTfLiteArenaDealloc(int subgraph_index, int arena_id, size_t num_bytes) {
  absl::MutexLock lock(&g_api_mutex);

  if (num_bytes == 0) return;
  AddArenaTrace(/*is_allocating=*/false, subgraph_index, arena_id, num_bytes);
}

}  // namespace tflite
