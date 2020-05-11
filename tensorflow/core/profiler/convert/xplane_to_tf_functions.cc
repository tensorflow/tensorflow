/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/profiler/convert/xplane_to_tf_functions.h"

#include <stack>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/timespan.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {

namespace {

std::pair<TfFunctionExecutionMode, TfFunctionCompiler> Decode(
    absl::string_view function_name, absl::string_view mode) {
  // mode is one of ["eager", "concrete", "traced-xla", "traced-nonXla",
  // "notTraced-xla", "notTraced-nonXla"]
  if (mode == "eager") return {EAGER_MODE, INVALID_COMPILER};
  if (mode == "concrete") return {CONCRETE_MODE, INVALID_COMPILER};
  if (mode == "traced-xla") return {TRACED_MODE, XLA_COMPILER};
  if (mode == "traced-nonXla") return {TRACED_MODE, OTHER_COMPILER};
  if (mode == "notTraced-xla") return {NOT_TRACED_MODE, XLA_COMPILER};
  if (mode == "notTraced-nonXla") return {NOT_TRACED_MODE, OTHER_COMPILER};
  // Shouldn't reach here.
  LOG(ERROR) << absl::StrCat("tf-function '", function_name,
                             "' has an unexpected execution mode '", mode, "'")
             << std::endl;
  return {INVALID_MODE, INVALID_COMPILER};
  DCHECK(false);
}

double ComputeExpensiveCallPercent(const TfFunction& tf_function) {
  // Computes the expensiveness in terms of time (rather than count).
  uint64 total_call_time_ps = 0;
  uint64 expensive_call_time_ps = 0;
  for (const auto& mode_metrics : tf_function.metrics()) {
    const auto mode = mode_metrics.first;
    const auto& metrics = mode_metrics.second;
    total_call_time_ps += metrics.self_time_ps();
    if (mode == TRACED_MODE || mode == EAGER_MODE) {
      expensive_call_time_ps += metrics.self_time_ps();
    }
  }
  return SafeDivide(100.0 * expensive_call_time_ps, total_call_time_ps);
}

// Each invocation of a tf-function creates an ActivationRecord.
struct ActivationRecord {
  std::string function_name;               // name of the tf-function.
  Timespan timespan;                       // timespan of this invocation.
  TfFunctionExecutionMode execution_mode;  // execution mode.
  TfFunctionCompiler compiler;             // compiler used.
  int64 tracing_count;  // the total tracing count of this function when this
                        // invocation happened.
  uint64 children_duration_ps;  // Sum of the duration of all (immediate)
                                // children tf-functions of this function.
  ActivationRecord()
      : function_name(""),
        execution_mode(INVALID_MODE),
        compiler(INVALID_COMPILER),
        tracing_count(0),
        children_duration_ps(0) {}
  ActivationRecord(absl::string_view name, const Timespan& timespan,
                   TfFunctionExecutionMode exe_mode,
                   TfFunctionCompiler compiler, int64 tracing_cnt)
      : function_name(std::string(name)),
        timespan(timespan),
        execution_mode(exe_mode),
        compiler(compiler),
        tracing_count(tracing_cnt),
        children_duration_ps(0) {}
  std::string DebugString() const {
    return absl::StrCat("{", function_name, ", ",
                        TfFunctionExecutionMode_Name(execution_mode), ", ",
                        TfFunctionCompiler_Name(compiler),
                        ", tracing_count:", tracing_count,
                        ", children_duration:", children_duration_ps,
                        " ps, timespan:", timespan.DebugString(), "}");
  }
};

// Entry or exit point of a tf-function.
struct EntryOrExit {
  bool is_entry;        // true for entry, false for exit.
  int64 index;          // index to the ActivationRecord.
  uint64 timestamp_ps;  // the time when this entry/exit happens.
  EntryOrExit() : is_entry(false), index(-1), timestamp_ps(0) {}
  EntryOrExit(bool is_entry, int64 index, uint64 timestamp_ps)
      : is_entry(is_entry), index(index), timestamp_ps(timestamp_ps) {}
  std::string DebugString() const {
    std::string entry_or_exit = is_entry ? "entry, " : "exit,  ";
    return absl::StrCat("{", entry_or_exit, "idx:", index,
                        ", timestamp:", timestamp_ps, "}");
  }
};

TfFunctionCompiler CombineCompilers(TfFunctionCompiler a,
                                    TfFunctionCompiler b) {
  if (a == INVALID_COMPILER) return b;
  if (b == INVALID_COMPILER) return a;
  if (a == b) return a;
  return MIXED_COMPILER;
}

void CombineTfFunctionMetrics(const TfFunctionMetrics& src,
                              TfFunctionMetrics* dst) {
  dst->set_count(src.count() + dst->count());
  dst->set_self_time_ps(src.self_time_ps() + dst->self_time_ps());
}

void CombineTfFunction(const TfFunction& src, TfFunction* dst) {
  dst->set_total_tracing_count(
      std::max(src.total_tracing_count(), dst->total_tracing_count()));
  dst->set_compiler(CombineCompilers(src.compiler(), dst->compiler()));
  for (const auto& mode_metrics : src.metrics()) {
    int32 execution_mode = mode_metrics.first;
    const TfFunctionMetrics& src_metrics = mode_metrics.second;
    TfFunctionMetrics* dst_metrics =
        gtl::FindOrNull(*dst->mutable_metrics(), execution_mode);
    if (dst_metrics == nullptr) {
      (*dst->mutable_metrics())[execution_mode] = src_metrics;
    } else {
      CombineTfFunctionMetrics(src_metrics, dst_metrics);
    }
  }
  dst->set_expensive_call_percent(ComputeExpensiveCallPercent(*dst));
}

// Execution history of all tf-functions invoked.
class TfFunctionExecutions {
 public:
  explicit TfFunctionExecutions(const XLineVisitor& line) {
    // Creates points_ and activations_ from line.
    line.ForEachEvent([&](const XEventVisitor& event) {
      std::string mode = "";
      int64 tracing_count = 0;
      event.ForEachStat([&mode, &tracing_count](const XStatVisitor& stat) {
        if (stat.Type() == StatType::kTfFunctionCall)
          mode = std::string(stat.StrOrRefValue());
        if (stat.Type() == StatType::kTfFunctionTracingCount)
          tracing_count = stat.IntValue();
      });
      if (mode.empty()) return;

      // event is a tf-function.
      int64 index = activations_.size();
      auto timespan = event.GetTimespan();
      auto mode_compiler = Decode(event.Name(), mode);
      ActivationRecord activation_record =
          ActivationRecord(event.Name(), timespan, mode_compiler.first,
                           mode_compiler.second, tracing_count);
      activations_.push_back(activation_record);
      EntryOrExit entry_point =
          EntryOrExit(/*is_entry=*/true, index, timespan.begin_ps());
      EntryOrExit exit_point =
          EntryOrExit(/*is_entry=*/false, index, timespan.end_ps());
      points_.push_back(entry_point);
      points_.push_back(exit_point);
    });

    // Sorts points_ in ascending order of timestamps.
    auto ascending_in_timestamp = [](const EntryOrExit& a,
                                     const EntryOrExit& b) {
      return a.timestamp_ps < b.timestamp_ps;
    };
    absl::c_sort(points_, ascending_in_timestamp);

    // Calculates the children duration for each activation record.
    CalculateChildrenDurations();
  }

  std::string DebugString() const {
    std::string result = "\nActivations:\n";
    for (auto i = 0; i < activations_.size(); i++) {
      absl::StrAppend(&result, "[", i, "] ", activations_[i].DebugString(),
                      "\n");
    }
    absl::StrAppend(&result, "tf-function Entry/Exit Points:\n");
    for (const auto& pt : points_) {
      absl::StrAppend(&result, pt.DebugString(), "\n");
    }
    return result;
  }

  // Converts this execution history to a TfFunctionDb.
  TfFunctionDb ConvertToTfFunctionDb() {
    TfFunctionDb result;
    for (const auto& record : activations_) {
      TfFunction* fun = &(*result.mutable_tf_functions())[record.function_name];
      fun->set_total_tracing_count(
          std::max(static_cast<int64>(fun->total_tracing_count()),
                   record.tracing_count));
      fun->set_compiler(CombineCompilers(fun->compiler(), record.compiler));
      // The self-time of this function is the difference between the duration
      // of this function and the duration of its children.
      uint64 self_time_ps =
          record.timespan.duration_ps() - record.children_duration_ps;
      // Updates the metrics for this execution mode with this invocation.
      TfFunctionMetrics* metrics =
          &(*fun->mutable_metrics())[record.execution_mode];
      metrics->set_count(metrics->count() + 1);
      metrics->set_self_time_ps(metrics->self_time_ps() + self_time_ps);
    }
    for (auto& name_fun : *result.mutable_tf_functions()) {
      TfFunction& fun = name_fun.second;
      fun.set_expensive_call_percent(ComputeExpensiveCallPercent(fun));
    }
    return result;
  }

  // Calculates the children duration of every tf-function.
  void CalculateChildrenDurations() {
    std::stack<int64> call_stack;
    for (const auto& pt : points_) {
      if (pt.is_entry) {
        // Function entry.
        call_stack.push(pt.index);
      } else {
        // Function exit.
        DCHECK(call_stack.top() == pt.index);  // must be well nested.
        uint64 call_duration = activations_[pt.index].timespan.duration_ps();
        call_stack.pop();
        if (!call_stack.empty()) {
          // call_stack.top() is the parent tf-function; adds call_duration to
          // its children_duration.
          activations_[call_stack.top()].children_duration_ps += call_duration;
        }
      }
    }
  }

 private:
  // ActivationRecords for all tf-function invocations.
  std::vector<ActivationRecord> activations_;
  // Entry and exit points of all invocations.
  std::vector<EntryOrExit> points_;
};

}  // namespace

std::string DebugString(const TfFunctionDb& tf_function_db) {
  std::string str;
  protobuf::TextFormat::PrintToString(tf_function_db, &str);
  return str;
}

void CombineTfFunctionDb(const TfFunctionDb& src, TfFunctionDb* dst) {
  for (const auto& name_function : src.tf_functions()) {
    const auto& name = name_function.first;
    const auto& src_fun = name_function.second;
    TfFunction* dst_fun = gtl::FindOrNull(*dst->mutable_tf_functions(), name);
    if (dst_fun == nullptr) {
      (*dst->mutable_tf_functions())[name] = src_fun;
    } else {
      CombineTfFunction(src_fun, dst_fun);
    }
  }
}

TfFunctionDb ConvertHostThreadsXLineToTfFunctionDb(const XLineVisitor& line) {
  TfFunctionExecutions tf_function_executions = TfFunctionExecutions(line);
  return tf_function_executions.ConvertToTfFunctionDb();
}

}  // namespace profiler
}  // namespace tensorflow
