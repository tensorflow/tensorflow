/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// Utility functions for performance profiling.

#ifndef TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_TIME_LOG_H_
#define TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_TIME_LOG_H_

#include <stdint.h>

#include "tensorflow/examples/android/jni/object_tracking/logging.h"
#include "tensorflow/examples/android/jni/object_tracking/utils.h"

#ifdef LOG_TIME

// Blend constant for running average.
#define ALPHA 0.98f
#define NUM_LOGS 100

struct LogEntry {
  const char* id;
  int64_t time_stamp;
};

struct AverageEntry {
  const char* id;
  float average_duration;
};

// Storage for keeping track of this frame's values.
extern int num_time_logs;
extern LogEntry time_logs[NUM_LOGS];

// Storage for keeping track of average values (each entry may not be printed
// out each frame).
extern AverageEntry avg_entries[NUM_LOGS];
extern int num_avg_entries;

// Call this at the start of a logging phase.
inline static void ResetTimeLog() {
  num_time_logs = 0;
}


// Log a message to be printed out when printTimeLog is called, along with the
// amount of time in ms that has passed since the last call to this function.
inline static void TimeLog(const char* const str) {
  LOGV("%s", str);
  if (num_time_logs >= NUM_LOGS) {
    LOGE("Out of log entries!");
    return;
  }

  time_logs[num_time_logs].id = str;
  time_logs[num_time_logs].time_stamp = CurrentThreadTimeNanos();
  ++num_time_logs;
}


inline static float Blend(float old_val, float new_val) {
  return ALPHA * old_val + (1.0f - ALPHA) * new_val;
}


inline static float UpdateAverage(const char* str, const float new_val) {
  for (int entry_num = 0; entry_num < num_avg_entries; ++entry_num) {
    AverageEntry* const entry = avg_entries + entry_num;
    if (str == entry->id) {
      entry->average_duration = Blend(entry->average_duration, new_val);
      return entry->average_duration;
    }
  }

  if (num_avg_entries >= NUM_LOGS) {
    LOGE("Too many log entries!");
  }

  // If it wasn't there already, add it.
  avg_entries[num_avg_entries].id = str;
  avg_entries[num_avg_entries].average_duration = new_val;
  ++num_avg_entries;

  return new_val;
}


// Prints out all the timeLog statements in chronological order with the
// interval that passed between subsequent statements.  The total time between
// the first and last statements is printed last.
inline static void PrintTimeLog() {
  LogEntry* last_time = time_logs;

  float average_running_total = 0.0f;

  for (int i = 0; i < num_time_logs; ++i) {
    LogEntry* const this_time = time_logs + i;

    const float curr_time =
        (this_time->time_stamp - last_time->time_stamp) / 1000000.0f;

    const float avg_time = UpdateAverage(this_time->id, curr_time);
    average_running_total += avg_time;

    LOGD("%32s:    %6.3fms    %6.4fms", this_time->id, curr_time, avg_time);
    last_time = this_time;
  }

  const float total_time =
      (last_time->time_stamp - time_logs->time_stamp) / 1000000.0f;

  LOGD("TOTAL TIME:                          %6.3fms    %6.4fms\n",
       total_time, average_running_total);
  LOGD(" ");
}
#else
inline static void ResetTimeLog() {}

inline static void TimeLog(const char* const str) {
  LOGV("%s", str);
}

inline static void PrintTimeLog() {}
#endif

#endif  // TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_TIME_LOG_H_
