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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_MEMORY_PLANNER_GREEDY_MEMORY_PLANNER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_MEMORY_PLANNER_GREEDY_MEMORY_PLANNER_H_

#include "tensorflow/lite/experimental/micro/memory_planner/memory_planner.h"

namespace tflite {

// A memory planner that uses a greedy algorithm to arrange buffers in memory
// to minimize the overall arena size needed.
//
// The algorithm works like this:
//  - The client enters the buffer information through AddBuffer().
//  - When a function like GetOffsetForBuffer() is called, the
//    CalculateOffsetsIfNeeded() method is invoked.
//  - If an up to date plan is not already present, one will be calculated.
//  - The buffers are sorted in descending order of size.
//  - The largest buffer is placed at offset zero.
//  - The rest of the buffers are looped through in descending size order.
//  - The other buffers that need to be in memory at the same time are found.
//  - The first gap between simultaneously active buffers that the current
//    buffer fits into will be used.
//  - If no large-enough gap is found, the current buffer is placed after the
//    last buffer that's simultaneously active.
//  - This continues until all buffers are placed, and the offsets stored.
//
// This is not guaranteed to produce the best placement, since that's an
// NP-Complete problem, but in practice it should produce one that's decent.
class GreedyMemoryPlanner : public MemoryPlanner {
 public:
  // You need to pass in an area of memory to be used for planning. This memory
  // needs to have a lifetime as long as the planner, but isn't owned by this
  // object, so management should be handled by the client. This is so it can be
  // stack or globally allocated if necessary on devices without dynamic memory
  // allocation. How many buffers can be planned for will depend on the size of
  // this scratch memory, so you should enlarge it if you see an error when
  // calling AddBuffer(). The memory can be reused once you're done with the
  // planner, as long as you copy the calculated offsets to another location.
  // Each buffer requires about 36 bytes of scratch.
  GreedyMemoryPlanner(unsigned char* scratch_buffer, int scratch_buffer_size);
  ~GreedyMemoryPlanner() override;

  // Record details of a buffer we want to place.
  TfLiteStatus AddBuffer(ErrorReporter* error_reporter, int size,
                         int first_time_used, int last_time_used) override;

  // Returns the high-water mark of used memory. This is the minimum size of a
  // memory arena you'd need to allocate to hold these buffers.
  int GetMaximumMemorySize() override;

  // How many buffers have been recorded.
  int GetBufferCount() override;

  // Where a given buffer should be placed in the memory arena.
  // This information is stored in the memory arena itself, so once the arena
  // is used for inference, it will be overwritten.
  TfLiteStatus GetOffsetForBuffer(ErrorReporter* error_reporter,
                                  int buffer_index, int* offset) override;

  // Prints an ascii-art diagram of the buffer layout plan.
  void PrintMemoryPlan(ErrorReporter* error_reporter);

  // Debug method to check whether any buffer allocations are overlapping. This
  // is an O(N^2) complexity operation, so only use for testing.
  bool DoAnyBuffersOverlap(ErrorReporter* error_reporter);

  // Used to store a list of buffers ordered by their offset.
  struct ListEntry {
    int offset;
    int requirements_index;
    int next_entry_index;
  };

 private:
  // Whether a buffer is active in a given time range.
  bool DoesEntryOverlapInTime(const ListEntry* entry, const int first_time_used,
                              const int last_time_used) const;

  // Walks the list to return the next buffer that is active in a given time
  // range, or a null pointer if there are none.
  ListEntry* NextSimultaneouslyActiveBuffer(const ListEntry* start,
                                            const int first_time_used,
                                            const int last_time_used);

  // If there isn't an up to date plan, calculate a new one.
  void CalculateOffsetsIfNeeded();

  // How many buffers we can plan for, based on the arena size we're given in
  // the constructor.
  int max_buffer_count_;

  // The number of buffers added so far.
  int buffer_count_;

  // Records the client-provided information about each buffer.
  struct BufferRequirements {
    int size;
    int first_time_used;
    int last_time_used;
  };

  // Working arrays used during the layout algorithm.
  BufferRequirements* requirements_;
  int* buffer_sizes_sorted_by_size_;
  int* buffer_ids_sorted_by_size_;
  ListEntry* buffers_sorted_by_offset_;
  int next_free_entry_;

  // Stores the outcome of the plan, the location of each buffer in the arena.
  int* buffer_offsets_;

  // Whether buffers have been added since the last plan was calculated.
  bool need_to_calculate_offsets_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_MEMORY_PLANNER_GREEDY_MEMORY_PLANNER_H_
