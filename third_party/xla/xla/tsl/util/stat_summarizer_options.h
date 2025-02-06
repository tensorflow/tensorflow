/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_UTIL_STAT_SUMMARIZER_OPTIONS_H_
#define XLA_TSL_UTIL_STAT_SUMMARIZER_OPTIONS_H_
namespace tsl {
// Used to control the output of the statistics summarizer;
struct StatSummarizerOptions {
  StatSummarizerOptions()
      : show_run_order(true),
        run_order_limit(0),
        show_time(true),
        time_limit(10),
        show_memory(true),
        memory_limit(10),
        show_type(true),
        show_summary(true),
        format_as_csv(false) {}

  bool show_run_order;
  int run_order_limit;
  bool show_time;
  int time_limit;
  bool show_memory;
  int memory_limit;
  bool show_type;
  bool show_summary;
  bool format_as_csv;
};
}  // namespace tsl

#endif  // XLA_TSL_UTIL_STAT_SUMMARIZER_OPTIONS_H_
