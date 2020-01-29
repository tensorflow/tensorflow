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

#ifndef TENSORFLOW_LITE_MICRO_MEMORY_PLANNER_MEMORY_PLANNER_H_
#define TENSORFLOW_LITE_MICRO_MEMORY_PLANNER_MEMORY_PLANNER_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"

namespace tflite {

// Interface class for planning the layout of memory buffers during the
// execution of a graph.
// It's designed to be used by a client that iterates in any order through the
// buffers it wants to lay out, and then calls the getter functions for
// information about the calculated layout. For example:
//
// SomeMemoryPlanner planner;
// planner.AddBuffer(reporter, 100, 0, 1);  // Buffer 0
// planner.AddBuffer(reporter, 50, 2, 3);   // Buffer 1
// planner.AddBuffer(reporter, 50, 2, 3);   // Buffer 2
//
// int offset0;
// TF_EXPECT_OK(planner.GetOffsetForBuffer(reporter, 0, &offset0));
// int offset1;
// TF_EXPECT_OK(planner.GetOffsetForBuffer(reporter, 1, &offset1));
// int offset2;
// TF_EXPECT_OK(planner.GetOffsetForBuffer(reporter, 2, &offset2));
// const int arena_size_needed = planner.GetMaximumMemorySize();
//
// The goal is for applications to be able to experiment with different layout
// strategies without changing their client code, by swapping out classes that
// implement this interface.=
class MemoryPlanner {
 public:
  MemoryPlanner() {}
  virtual ~MemoryPlanner() {}

  // Pass information about a buffer's size and lifetime to the layout
  // algorithm. The order this is called implicitly assigns an index to the
  // result, so the buffer information that's passed into the N-th call of
  // this method will be used as the buffer_index argument to
  // GetOffsetForBuffer().
  virtual TfLiteStatus AddBuffer(tflite::ErrorReporter* error_reporter,
                                 int size, int first_time_used,
                                 int last_time_used) = 0;

  // The largest contguous block of memory that's needed to hold the layout.
  virtual size_t GetMaximumMemorySize() = 0;
  // How many buffers have been added to the planner.
  virtual int GetBufferCount() = 0;
  // Calculated layout offset for the N-th buffer added to the planner.
  virtual TfLiteStatus GetOffsetForBuffer(tflite::ErrorReporter* error_reporter,
                                          int buffer_index, int* offset) = 0;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MEMORY_PLANNER_MEMORY_PLANNER_H_
