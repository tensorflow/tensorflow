/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_TFPROF_OPTIONS_H_
#define TENSORFLOW_CORE_PROFILER_TFPROF_OPTIONS_H_

#include <set>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tfprof {
static const char* const kOptions[] = {
    "-max_depth",
    "-min_bytes",
    "-min_peak_bytes",
    "-min_residual_bytes",
    "-min_output_bytes",
    "-min_micros",
    "-min_accelerator_micros",
    "-min_cpu_micros",
    "-min_params",
    "-min_float_ops",
    "-min_occurrence",
    "-step",
    "-order_by",
    "-account_type_regexes",
    "-start_name_regexes",
    "-trim_name_regexes",
    "-show_name_regexes",
    "-hide_name_regexes",
    "-account_displayed_op_only",
    "-select",
    "-output",
};

static const char* const kOrderBy[] = {
    "name",         "bytes",     "peak_bytes",         "residual_bytes",
    "output_bytes", "micros",    "accelerator_micros", "cpu_micros",
    "params",       "float_ops", "occurrence",
};

// Append Only.
// TODO(xpan): As we are adding more fields to be selected, we
// need to have a way to tell users what fields are available in which view.
static const char* const kShown[] = {"bytes",          "micros",
                                     "params",         "float_ops",
                                     "tensor_value",   "device",
                                     "op_types",       "occurrence",
                                     "input_shapes",   "accelerator_micros",
                                     "cpu_micros",     "peak_bytes",
                                     "residual_bytes", "output_bytes"};

static const char* const kCmds[] = {
    "scope", "graph", "code", "op", "advise", "set", "help",
};

static const char* const kOutput[] = {"timeline", "stdout", "file", "pprof",
                                      "none"};

static const char* const kTimelineOpts[] = {
    "outfile",
};

static const char* const kTimelineRequiredOpts[] = {"outfile"};

static const char* const kFileOpts[] = {
    "outfile",
};

static const char* const kFileRequiredOpts[] = {
    "outfile",
};

static const char* const kPprofOpts[] = {
    "outfile",
};

static const char* const kPprofRequiredOpts[] = {
    "outfile",
};

struct Options {
 public:
  static tensorflow::Status FromProtoStr(const string& opts_proto_str,
                                         Options* opts);

  virtual ~Options() {}
  Options()
      : Options(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "", {}, {}, {}, {}, {},
                false, {}, "", {}) {}

  Options(int max_depth, int64_t min_bytes, int64_t min_peak_bytes,
          int64_t min_residual_bytes, int64_t min_output_bytes,
          int64_t min_micros, int64_t min_accelerator_micros,
          int64_t min_cpu_micros, int64_t min_params, int64_t min_float_ops,
          int64_t min_occurrence, int64_t step, const string& order_by,
          const std::vector<string>& account_type_regexes,
          const std::vector<string>& start_name_regexes,
          const std::vector<string>& trim_name_regexes,
          const std::vector<string>& show_name_regexes,
          const std::vector<string>& hide_name_regexes,
          bool account_displayed_op_only, const std::vector<string>& select,
          const string& output_type,
          const std::map<string, string>& output_options)
      : max_depth(max_depth),
        min_bytes(min_bytes),
        min_peak_bytes(min_peak_bytes),
        min_residual_bytes(min_residual_bytes),
        min_output_bytes(min_output_bytes),
        min_micros(min_micros),
        min_accelerator_micros(min_accelerator_micros),
        min_cpu_micros(min_cpu_micros),
        min_params(min_params),
        min_float_ops(min_float_ops),
        min_occurrence(min_occurrence),
        step(step),
        order_by(order_by),
        account_type_regexes(account_type_regexes),
        start_name_regexes(start_name_regexes),
        trim_name_regexes(trim_name_regexes),
        show_name_regexes(show_name_regexes),
        hide_name_regexes(hide_name_regexes),
        account_displayed_op_only(account_displayed_op_only),
        select(select.begin(), select.end()),
        output_type(output_type),
        output_options(output_options) {}

  string ToString() const;

  int max_depth;
  int64_t min_bytes;
  int64_t min_peak_bytes;
  int64_t min_residual_bytes;
  int64_t min_output_bytes;
  int64_t min_micros;
  int64_t min_accelerator_micros;
  int64_t min_cpu_micros;
  int64_t min_params;
  int64_t min_float_ops;
  int64_t min_occurrence;
  int64_t step;
  string order_by;

  std::vector<string> account_type_regexes;
  std::vector<string> start_name_regexes;
  std::vector<string> trim_name_regexes;
  std::vector<string> show_name_regexes;
  std::vector<string> hide_name_regexes;
  bool account_displayed_op_only;

  std::set<string> select;

  string output_type;
  std::map<string, string> output_options;
};

// Parse the -output option.
// 'output_opt': User input string with format: output_type:key=value,key=value.
// 'output_type' and 'output_options' are extracted from 'output_opt'.
tensorflow::Status ParseOutput(const string& output_opt, string* output_type,
                               std::map<string, string>* output_options);

}  // namespace tfprof
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_TFPROF_OPTIONS_H_
