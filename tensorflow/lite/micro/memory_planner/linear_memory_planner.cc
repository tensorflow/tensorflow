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

#include "tensorflow/lite/micro/memory_planner/linear_memory_planner.h"

namespace tflite {

LinearMemoryPlanner::LinearMemoryPlanner()
    : current_buffer_count_(0), next_free_offset_(0) {}
LinearMemoryPlanner::~LinearMemoryPlanner() {}

TfLiteStatus LinearMemoryPlanner::AddBuffer(
    tflite::ErrorReporter* error_reporter, int size, int first_time_used,
    int last_time_used) {
  if (current_buffer_count_ >= kMaxBufferCount) {
    error_reporter->Report("Too many buffers (max is %d)", kMaxBufferCount);
    return kTfLiteError;
  }
  buffer_offsets_[current_buffer_count_] = next_free_offset_;
  next_free_offset_ += size;
  ++current_buffer_count_;
  return kTfLiteOk;
}

size_t LinearMemoryPlanner::GetMaximumMemorySize() { return next_free_offset_; }

int LinearMemoryPlanner::GetBufferCount() { return current_buffer_count_; }

TfLiteStatus LinearMemoryPlanner::GetOffsetForBuffer(
    tflite::ErrorReporter* error_reporter, int buffer_index, int* offset) {
  if ((buffer_index < 0) || (buffer_index >= current_buffer_count_)) {
    error_reporter->Report("buffer index %d is outside range 0 to %d",
                           buffer_index, current_buffer_count_);
    return kTfLiteError;
  }
  *offset = buffer_offsets_[buffer_index];
  return kTfLiteOk;
}

}  // namespace tflite
