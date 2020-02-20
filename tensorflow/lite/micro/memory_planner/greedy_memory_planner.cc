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

#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"

namespace tflite {

// Simple stable in-place sort function. Not time-efficient for large arrays.
// Would normally be in an anonymous namespace to keep it private, but we want
// to be able to test it externally.
void ReverseSortInPlace(int* values, int* ids, int size) {
  bool any_swapped;
  do {
    any_swapped = false;
    for (int i = 1; i < size; ++i) {
      if (values[i - 1] < values[i]) {
        const int value_temp = values[i - 1];
        values[i - 1] = values[i];
        values[i] = value_temp;
        const int id_temp = ids[i - 1];
        ids[i - 1] = ids[i];
        ids[i] = id_temp;
        any_swapped = true;
      }
    }
  } while (any_swapped);
}

GreedyMemoryPlanner::GreedyMemoryPlanner(unsigned char* scratch_buffer,
                                         int scratch_buffer_size)
    : buffer_count_(0), need_to_calculate_offsets_(true) {
  const int per_buffer_size = sizeof(BufferRequirements) +  // requirements_
                              sizeof(int) +  // buffer_sizes_sorted_by_size_
                              sizeof(int) +  // buffer_ids_sorted_by_size_
                              sizeof(ListEntry) +  // buffers_sorted_by_offset_
                              sizeof(int);         // buffer_offsets_;
  // Allocate the arrays we need within the scratch buffer arena.
  max_buffer_count_ = scratch_buffer_size / per_buffer_size;

  unsigned char* next_free = scratch_buffer;
  requirements_ = reinterpret_cast<BufferRequirements*>(next_free);
  next_free += sizeof(BufferRequirements) * max_buffer_count_;

  buffer_sizes_sorted_by_size_ = reinterpret_cast<int*>(next_free);
  next_free += sizeof(int) * max_buffer_count_;

  buffer_ids_sorted_by_size_ = reinterpret_cast<int*>(next_free);
  next_free += sizeof(int) * max_buffer_count_;

  buffers_sorted_by_offset_ = reinterpret_cast<ListEntry*>(next_free);
  next_free += sizeof(ListEntry) * max_buffer_count_;

  buffer_offsets_ = reinterpret_cast<int*>(next_free);
}

GreedyMemoryPlanner::~GreedyMemoryPlanner() {
  // We don't own the scratch buffer, so don't deallocate anything.
}

TfLiteStatus GreedyMemoryPlanner::AddBuffer(
    tflite::ErrorReporter* error_reporter, int size, int first_time_used,
    int last_time_used) {
  if (buffer_count_ >= max_buffer_count_) {
    TF_LITE_REPORT_ERROR(error_reporter, "Too many buffers (max is %d)",
                         max_buffer_count_);
    return kTfLiteError;
  }
  BufferRequirements* current = &requirements_[buffer_count_];
  current->size = size;
  current->first_time_used = first_time_used;
  current->last_time_used = last_time_used;
  ++buffer_count_;
  need_to_calculate_offsets_ = true;
  return kTfLiteOk;
}

bool GreedyMemoryPlanner::DoesEntryOverlapInTime(
    const GreedyMemoryPlanner::ListEntry* entry, const int first_time_used,
    const int last_time_used) const {
  const BufferRequirements* entry_requirements =
      &requirements_[entry->requirements_index];
  if (entry_requirements->first_time_used > last_time_used) {
    return false;
  }
  if (first_time_used > entry_requirements->last_time_used) {
    return false;
  }
  return true;
}

GreedyMemoryPlanner::ListEntry*
GreedyMemoryPlanner::NextSimultaneouslyActiveBuffer(
    const GreedyMemoryPlanner::ListEntry* start, const int first_time_used,
    const int last_time_used) {
  ListEntry* result = nullptr;
  ListEntry* candidate_next_entry;
  if (start == nullptr) {
    candidate_next_entry = &buffers_sorted_by_offset_[0];
  } else {
    if (start->next_entry_index == -1) {
      return nullptr;
    }
    candidate_next_entry = &buffers_sorted_by_offset_[start->next_entry_index];
  }
  do {
    if (DoesEntryOverlapInTime(candidate_next_entry, first_time_used,
                               last_time_used)) {
      result = candidate_next_entry;
      break;
    }
    if (candidate_next_entry->next_entry_index == -1) {
      break;
    }
    candidate_next_entry =
        &buffers_sorted_by_offset_[candidate_next_entry->next_entry_index];
  } while (true);
  return result;
}

void GreedyMemoryPlanner::CalculateOffsetsIfNeeded() {
  if (!need_to_calculate_offsets_ || (buffer_count_ == 0)) {
    return;
  }
  need_to_calculate_offsets_ = false;

  // Start off by ordering the buffers in descending order of size.
  // This helps find a more compact layout. Intuitively, you can think
  // about putting the large buffers in place first, and then the
  // smaller buffers can fit in the gaps, rather than fragmenting the
  // gaps with small buffers at the beginning.
  for (int i = 0; i < buffer_count_; ++i) {
    buffer_sizes_sorted_by_size_[i] = requirements_[i].size;
    buffer_ids_sorted_by_size_[i] = i;
    buffer_offsets_[i] = -1;
  }
  // This sorting algorithm is naive, and may end up taking a very long time
  // with hundreds of buffers.
  ReverseSortInPlace(buffer_sizes_sorted_by_size_, buffer_ids_sorted_by_size_,
                     buffer_count_);

  // Put the largest buffer at offset zero to start the process.
  ListEntry* first_entry = &buffers_sorted_by_offset_[0];
  first_entry->offset = 0;
  first_entry->requirements_index = buffer_ids_sorted_by_size_[0];
  first_entry->next_entry_index = -1;
  next_free_entry_ = 1;
  buffer_offsets_[buffer_ids_sorted_by_size_[0]] = 0;

  // Work through the rest of the buffers to find a good gap to place each one.
  for (int i = 1; i < buffer_count_; ++i) {
    // The id is the order the buffer was originally added by the client.
    const int buffer_id = buffer_ids_sorted_by_size_[i];
    // Look at what size and time range the buffer needs to be active.
    BufferRequirements* wanted_requirements = &requirements_[buffer_id];
    const int wanted_size = wanted_requirements->size;
    const int wanted_first_time_used = wanted_requirements->first_time_used;
    const int wanted_last_time_used = wanted_requirements->last_time_used;

    // Find the first buffer that's active in our time range. All placed
    // buffers are stored in the order of their starting position in the arena
    // so that it's easy to find the next buffer in memory, and so the gap.
    // The candidate_entry variable holds the buffer that we're considering
    // placing the current buffer after.
    ListEntry* prior_entry = nullptr;
    int candidate_offset = 0;
    // Loop through the offset-ordered list of buffers, looking for gaps.
    while (true) {
      // Find out what the next active buffer is.
      ListEntry* next_entry = NextSimultaneouslyActiveBuffer(
          prior_entry, wanted_first_time_used, wanted_last_time_used);

      if (prior_entry) {
        BufferRequirements* candidate_requirements =
            &requirements_[prior_entry->requirements_index];
        const int prior_entry_offset =
            prior_entry->offset + candidate_requirements->size;
        if (prior_entry_offset > candidate_offset) {
          candidate_offset = prior_entry_offset;
        }
      }
      if (next_entry == nullptr) {
        // We're at the end of the list, so we can always append the buffer
        // here.
        break;
      }
      // Find out how much space there is between us and the next buffer.
      const int gap = next_entry->offset - candidate_offset;
      if (gap >= wanted_size) {
        // This entry has a big enough gap between it and the next, so
        // use it!
        break;
      }
      // The gap wasn't big enough, so move on to another candidate.
      prior_entry = next_entry;
    }
    // At this point, we've either found a gap (possibly at the end of the
    // list) and want to place the buffer there, or there are no other active
    // buffers in this time range and so we can put it at offset zero.
    // Record the buffer's offset in our plan.
    buffer_offsets_[buffer_id] = candidate_offset;
    // Add the newly-placed buffer to our offset-ordered list, so that
    // subsequent passes can fit in their buffers around it.
    ListEntry* new_entry = &buffers_sorted_by_offset_[next_free_entry_];
    new_entry->offset = candidate_offset;
    new_entry->requirements_index = buffer_id;
    const int new_entry_index = next_free_entry_;
    ++next_free_entry_;
    ListEntry* current_entry = first_entry;
    // Make sure that we insert the buffer at the correct place in the ordered
    // list.
    while (true) {
      const int next_entry_index = current_entry->next_entry_index;
      if (next_entry_index == -1) {
        // We're at the end of the list, so just add the new entry here.
        current_entry->next_entry_index = new_entry_index;
        new_entry->next_entry_index = -1;
        break;
      }
      ListEntry* next_entry = &buffers_sorted_by_offset_[next_entry_index];
      if (next_entry->offset > candidate_offset) {
        // We're at the right spot to do an insertion and retain the sorting
        // order, so place the new entry here.
        new_entry->next_entry_index = current_entry->next_entry_index;
        current_entry->next_entry_index = new_entry_index;
        break;
      }
      current_entry = next_entry;
    }
  }
}

size_t GreedyMemoryPlanner::GetMaximumMemorySize() {
  CalculateOffsetsIfNeeded();
  if (buffer_count_ == 0) {
    return 0;
  }
  ListEntry* entry = &buffers_sorted_by_offset_[0];
  size_t max_size = 0;
  while (entry) {
    BufferRequirements* requirements =
        &requirements_[entry->requirements_index];
    // TODO(b/148246793): Update all size and offset variables types from
    //                    int to size_t
    const size_t current_size = entry->offset + requirements->size;
    if (current_size > max_size) {
      max_size = current_size;
    }
    if (entry->next_entry_index == -1) {
      break;
    }
    entry = &buffers_sorted_by_offset_[entry->next_entry_index];
  }
  return max_size;
}

void GreedyMemoryPlanner::PrintMemoryPlan(ErrorReporter* error_reporter) {
  CalculateOffsetsIfNeeded();

  for (int i = 0; i < buffer_count_; ++i) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "Planner buffer ID: %d, calculated offset: %d, size required: %d, "
        "first_time_created: %d, "
        "last_time_used: %d",
        i, buffer_offsets_[i], requirements_[i].size,
        requirements_[i].first_time_used, requirements_[i].last_time_used);
  }

  constexpr int kLineWidth = 80;
  int max_size = kLineWidth;
  int max_time = 0;
  for (int i = 0; i < buffer_count_; ++i) {
    BufferRequirements* requirements = &requirements_[i];
    const int offset = buffer_offsets_[i];
    const int last_time_used = requirements->last_time_used;
    const int size = offset + requirements->size;
    if (size > max_size) {
      max_size = size;
    }
    if (last_time_used > max_time) {
      max_time = last_time_used;
    }
  }

  char line[kLineWidth + 1];
  for (int t = 0; t <= max_time; ++t) {
    for (int c = 0; c < kLineWidth; ++c) {
      line[c] = '.';
    }
    for (int i = 0; i < buffer_count_; ++i) {
      BufferRequirements* requirements = &requirements_[i];
      if ((t < requirements->first_time_used) ||
          (t > requirements->last_time_used)) {
        continue;
      }
      const int offset = buffer_offsets_[i];
      if (offset == -1) {
        continue;
      }
      const int size = requirements->size;
      const int line_start = (offset * kLineWidth) / max_size;
      const int line_end = ((offset + size) * kLineWidth) / max_size;
      for (int n = line_start; n < line_end; ++n) {
        if (line[n] == '.') {
          char display;
          if (i < 10) {
            display = '0' + i;
          } else if (i < 36) {
            display = 'a' + (i - 10);
          } else if (i < 62) {
            display = 'A' + (i - 36);
          } else {
            display = '*';
          }
          line[n] = display;
        } else {
          line[n] = '!';
        }
      }
    }
    line[kLineWidth] = 0;
    TF_LITE_REPORT_ERROR(error_reporter, "%s", line);
  }
}

int GreedyMemoryPlanner::GetBufferCount() { return buffer_count_; }

TfLiteStatus GreedyMemoryPlanner::GetOffsetForBuffer(
    tflite::ErrorReporter* error_reporter, int buffer_index, int* offset) {
  CalculateOffsetsIfNeeded();
  if ((buffer_index < 0) || (buffer_index >= buffer_count_)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "buffer index %d is outside range 0 to %d",
                         buffer_index, buffer_count_);
    return kTfLiteError;
  }
  *offset = buffer_offsets_[buffer_index];
  return kTfLiteOk;
}

bool GreedyMemoryPlanner::DoAnyBuffersOverlap(ErrorReporter* error_reporter) {
  CalculateOffsetsIfNeeded();
  bool were_overlaps_found = false;
  for (int i = 0; i < buffer_count_; ++i) {
    BufferRequirements* a_requirements = &requirements_[i];
    const int a_start_offset = buffer_offsets_[i];
    const int a_first_time_used = a_requirements->first_time_used;
    const int a_last_time_used = a_requirements->last_time_used;
    const int a_end_offset = a_start_offset + a_requirements->size;
    for (int j = 0; j < buffer_count_; ++j) {
      if (i == j) {
        continue;
      }
      BufferRequirements* b_requirements = &requirements_[j];
      const int b_start_offset = buffer_offsets_[j];
      const int b_first_time_used = b_requirements->first_time_used;
      const int b_last_time_used = b_requirements->last_time_used;
      const int b_end_offset = b_start_offset + b_requirements->size;
      if ((a_first_time_used > b_last_time_used) ||
          (b_first_time_used > a_last_time_used)) {
        // Buffers don't overlap in time.
        continue;
      }
      if ((a_start_offset >= b_end_offset) ||
          (b_start_offset >= a_end_offset)) {
        // No overlap in memory.
        continue;
      }
      were_overlaps_found = true;
      TF_LITE_REPORT_ERROR(
          error_reporter, "Overlap: %d (%d=>%d, %d->%d) vs %d (%d=>%d, %d->%d)",
          i, a_first_time_used, a_last_time_used, a_start_offset, a_end_offset,
          j, b_first_time_used, b_last_time_used, b_start_offset, b_end_offset);
    }
  }
  return were_overlaps_found;
}

}  // namespace tflite
