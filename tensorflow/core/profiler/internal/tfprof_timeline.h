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

#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_TIMELINE_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_TIMELINE_H_

#include "absl/strings/str_cat.h"
#include "json/json.h"
#include "tensorflow/core/profiler/internal/tfprof_node_show.h"

namespace tensorflow {
namespace tfprof {

typedef std::map<string, string> Event;

// Class for generating timeline json output.
class ChromeTraceFormatter {
 public:
  ChromeTraceFormatter() {}
  // The following methods creates timeline nodes. See chrome tracing format
  // document for details.
  Json::Value CreateEvent(const string& ph, const string& category,
                          const string& name, int64_t pid, int64_t tid,
                          int64_t ts);

  void EmitPID(const string& name, int64_t pid);

  void EmitRegion(int64_t ts, int64_t duration, int64_t pid, int64_t tid,
                  const string& category, const string& name, Json::Value args);

  void EmitFlowStart(const string& name, int64_t ts, int64_t pid, int64_t tid,
                     int64_t flow_id);

  void EmitFlowEnd(const string& name, int64_t ts, int64_t pid, int64_t tid,
                   int64_t flow_id);

  void EmitCounter(const string& category, const string& name, int64_t pid,
                   int64_t ts, const string& device, int64_t bytes,
                   const std::map<int64_t, std::vector<string>>& tensor_mem);

  string Format();

 private:
  // A event is a visualization unit in timeline.
  std::vector<Json::Value> events_;
  std::vector<Json::Value> metadata_;
};

// A process (time series of events) in the timeline.
class Process {
 public:
  Process(const string& device, int64_t pid) : device(device), pid(pid) {}

  // Each lane is a map from start_time to end_time.
  std::vector<std::map<int64_t, int64_t>> lanes;
  // device for the time series.
  string device;
  // unique id for the time series.
  int64_t pid;
};

class TimeNode {
 public:
  TimeNode(Process* process, GraphNode* node, int64_t start_micros,
           int64_t exec_micros)
      : process(process),
        node(node),
        start_micros(start_micros),
        exec_micros(exec_micros),
        tid(-1) {}
  virtual ~TimeNode() {}

  const string& name() { return node->name(); }

  Process* process;
  GraphNode* node;
  int64_t start_micros;
  int64_t exec_micros;
  int64_t tid;
  std::vector<TimeNode*> next_tnodes;
};

// Tracking the memory based on the op input/output, temporary bytes and
// persistent bytes.
// Currently, we calculate a "predicted" memory, but do not use it for display.
// The displayed memory timeline is directly from the TensorFlow allocator,
// which is the groundtruth.
class MemoryTracker {
 public:
  class Device {
   public:
    // map from tensor name to a pair of <alloc time, bytes_in_use>.
    std::map<string, std::map<int64_t, int64_t>> tensor_allocs;
    // ground truth memory stats. time->bytes.
    std::map<int64_t, int64_t> allocations;
    // tracked allocations, might miss some bytes.
    std::map<int64_t, int64_t> tracked_allocations;
  };

  void TrackNode(int64_t step, const GraphNode* node);

  const std::map<string, Device>& devices() const { return devices_; }

 private:
  std::map<string, Device> devices_;
};

class Timeline {
 public:
  Timeline(int64_t step, const string& outfile)
      : step_(step), outfile_(outfile) {}
  ~Timeline() {}

  int64_t step() const { return step_; }
  void SetStep(int64_t step) { step_ = step; }

  void GenerateGraphTimeline(const std::vector<GraphNode*>& gnodes);

  void GenerateScopeTimeline(const ScopeNode* node);

  void GenerateCodeTimeline(const CodeNode* node);

 private:
  void TrackNode(const GraphNode* node) { mem_tracker_.TrackNode(step_, node); }

  void OutputTimeline();

  template <typename Node>
  void EmitTreeNode(const Node* node, int64_t start_time, int64_t duration,
                    int64_t depth, std::set<int64_t>* visited_depth) {
    if (visited_depth->find(depth) == visited_depth->end()) {
      chrome_formatter_.EmitPID(absl::StrCat("Scope:", depth), depth);
      visited_depth->insert(depth);
    }

    Json::Value args(Json::objectValue);
    args["name"] = Json::Value(node->name());
    args["op"] = Json::Value(node->name());
    chrome_formatter_.EmitRegion(start_time, duration, depth, 0, "Op",
                                 node->name(), args);

    int64_t total_micros = 0;
    int64_t c_start_time = start_time;
    for (const Node* child : node->show_children) {
      int64_t total_exec_micros = child->proto().total_exec_micros();
      if (total_exec_micros <= 0) {
        continue;
      }
      EmitTreeNode(child, c_start_time, total_exec_micros, depth + 1,
                   visited_depth);
      c_start_time += total_exec_micros;
      total_micros += total_exec_micros;
    }
    CHECK(total_micros <= duration) << node->name() << " parent:" << duration
                                    << " children:" << total_micros;
  }

  void AllocateTimeNodes(GraphNode* gnode);

  void AllocateLanes();

  int64_t AllocatePID();

  int64_t step_;
  const string outfile_;
  int64_t next_pid_ = 0;
  MemoryTracker mem_tracker_;
  ChromeTraceFormatter chrome_formatter_;
  std::map<string, int64_t> device_pids_;

  std::map<string, std::unique_ptr<Process>> process_;
  std::map<int64_t, std::map<int64_t, std::map<int64_t, TimeNode*>>>
      alloc_nodes_;
  std::map<string, std::map<int64_t, std::unique_ptr<TimeNode>>> tnodes_;
};

}  // namespace tfprof
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_TIMELINE_H_
