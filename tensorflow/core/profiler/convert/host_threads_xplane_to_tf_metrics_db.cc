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

#include "tensorflow/core/profiler/convert/host_threads_xplane_to_tf_metrics_db.h"

#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/profiler/convert/op_stack.h"
#include "tensorflow/core/profiler/utils/op_utils.h"
#include "tensorflow/core/profiler/utils/timespan.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

// Type of a TensorFlow Op activity, which is either beginning or ending an Op.
enum TfActivityType { kTfOpBegin, kTfOpEnd };

// Instant activity representing the begin or end of a host-side TF Op.
struct TfActivity {
  // The timestamp in picoseconds when this activity happened.
  uint64 timestamp_ps;
  // The ID of this Op.
  uint32 tf_op_id;
  // Type of this activity.
  TfActivityType activity_type;
  // Full TF op name and type of this activity (backed by XEvent::name).
  TfOp tf_op;
};

// TF Op metrics stored as element in OpStack.
struct TfOpInfo {
  explicit TfOpInfo(uint64 ts) : start_timestamp_ps(ts) {}

  // Start timestamp in picoseconds.
  uint64 start_timestamp_ps;
  // Children duration in picoseconds.
  uint64 children_duration_ps = 0;
};

// Processes a TF-activity on particular core.
void ProcessOneTfActivity(const TfActivity& activity,
                          OpStack<TfOpInfo>* tf_op_stack,
                          TfMetricsDbData* tf_metrics_data) {
  uint32 tf_op_id = activity.tf_op_id;
  switch (activity.activity_type) {
    case kTfOpBegin: {
      tf_op_stack->Push(tf_op_id,
                        absl::make_unique<TfOpInfo>(activity.timestamp_ps));
      break;
    }
    case kTfOpEnd: {
      std::unique_ptr<TfOpInfo> info = tf_op_stack->Pop(tf_op_id);
      if (info == nullptr) {
        // This happens if TraceMes overlap.
        VLOG(1) << "No begin event found for TF activity id=" << tf_op_id
                << " name=" << activity.tf_op.name
                << " type=" << activity.tf_op.type;
        break;
      }
      Timespan tf_op_span =
          PicoSpan(info->start_timestamp_ps, activity.timestamp_ps);
      tf_metrics_data->tf_metrics_db_builder.EnterOp(
          activity.tf_op.name, activity.tf_op.type, tf_op_span.duration_ps(),
          info->children_duration_ps);
      TfOpInfo* parent_info = tf_op_stack->Top();
      if (parent_info != nullptr) {
        parent_info->children_duration_ps += tf_op_span.duration_ps();
      }
      if (IsInfeedEnqueueOp(activity.tf_op.type)) {
        if (tf_metrics_data->last_infeed_enq_duration_ps > 0) {
          DCHECK(tf_metrics_data->last_infeed_enq_start_timestamp_ps <=
                 info->start_timestamp_ps);
          uint64 start_timestamps_ps_diff =
              info->start_timestamp_ps -
              tf_metrics_data->last_infeed_enq_start_timestamp_ps;
          tf_metrics_data->tf_metrics_db_builder.UpdateHostInfeedEnqInfo(
              tf_metrics_data->last_infeed_enq_duration_ps,
              start_timestamps_ps_diff);
        }
        tf_metrics_data->last_infeed_enq_start_timestamp_ps =
            info->start_timestamp_ps;
        tf_metrics_data->last_infeed_enq_duration_ps = tf_op_span.duration_ps();
      }
      break;
    }
  }
}

// Processes all TF-activities on the given core.
void ProcessTfActivities(std::vector<TfActivity>* tf_activities,
                         TfMetricsDbData* tf_metrics_db_data) {
  if (tf_activities->empty()) return;
  absl::c_stable_sort(*tf_activities,
                      [](const TfActivity& a, const TfActivity& b) {
                        return a.timestamp_ps < b.timestamp_ps;
                      });
  OpStack<TfOpInfo> tf_op_stack;
  for (const auto& tf_activity : *tf_activities) {
    ProcessOneTfActivity(tf_activity, &tf_op_stack, tf_metrics_db_data);
  }
  tf_metrics_db_data->tf_metrics_db.set_total_time_ps(
      tf_activities->back().timestamp_ps - tf_activities->front().timestamp_ps);
}

void CollectTfActivities(const XLineVisitor& line,
                         const absl::flat_hash_map<int64, TfOp>& tf_ops,
                         std::vector<TfActivity>* tf_activities) {
  uint32 tf_op_id = 0;
  tf_activities->reserve(line.NumEvents() * 2);
  line.ForEachEvent([&tf_ops, &tf_op_id,
                     &tf_activities](const XEventVisitor& event) {
    const TfOp* tf_op = gtl::FindOrNull(tf_ops, event.Id());
    if (tf_op != nullptr) {
      ++tf_op_id;
      Timespan span(event.TimestampPs(), event.DurationPs());
      tf_activities->push_back({span.begin_ps(), tf_op_id, kTfOpBegin, *tf_op});
      tf_activities->push_back({span.end_ps(), tf_op_id, kTfOpEnd, *tf_op});
    }
  });
}

}  // namespace

TfMetricsDbData ConvertHostThreadsXLineToTfMetricsDbData(
    const XLineVisitor& line, const absl::flat_hash_map<int64, TfOp>& tf_ops) {
  TfMetricsDbData tf_metrics_db_data;
  if (!tf_ops.empty()) {
    std::vector<TfActivity> tf_activities;
    CollectTfActivities(line, tf_ops, &tf_activities);
    ProcessTfActivities(&tf_activities, &tf_metrics_db_data);
  }
  return tf_metrics_db_data;
}

}  // namespace profiler
}  // namespace tensorflow
