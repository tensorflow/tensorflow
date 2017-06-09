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

#ifndef THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_TFPROF_TIMELINE_H_
#define THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_TFPROF_TIMELINE_H_

#include "include/json/json.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/tools/tfprof/internal/tfprof_node_show.h"

namespace tensorflow {
namespace tfprof {

typedef std::map<string, string> Event;

class ChromeTraceFormatter {
 public:
  ChromeTraceFormatter() {}

  Json::Value CreateEvent(const string& ph, const string& category,
                          const string& name, int64 pid, int64 tid, int64 ts);

  void EmitPID(const string& name, int64 pid);

  void EmitRegion(int64 ts, int64 duration, int64 pid, int64 tid,
                  const string& category, const string& name, Json::Value args);

  void EmitFlowStart(const string& name, int64 ts, int64 pid, int64 tid,
                     int64 flow_id);

  void EmitFlowEnd(const string& name, int64 ts, int64 pid, int64 tid,
                   int64 flow_id);

  void EmitCounter(const string& category, const string& name, int64 pid,
                   int64 ts, const string& device, int64 bytes);

  string Format();

 private:
  std::vector<Json::Value> events_;
  std::vector<Json::Value> metadata_;
};

class Process {
 public:
  Process(const string& device, int64 pid) : device(device), pid(pid) {}

  // Each lane is a map from start_time to end_time.
  std::vector<std::map<int64, int64>> lanes;
  string device;
  int64 pid;
};

class TimeNode {
 public:
  TimeNode(Process* process, GraphNode* node, int64 start_micros,
           int64 exec_micros)
      : process(process),
        node(node),
        start_micros(start_micros),
        exec_micros(exec_micros),
        tid(-1) {}
  virtual ~TimeNode() {}

  const string& name() { return node->name(); }

  Process* process;
  GraphNode* node;
  int64 start_micros;
  int64 exec_micros;
  int64 tid;
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
    // The first 3 fields are predicted.
    std::map<string, int64> tensor_size;
    std::map<string, int64> earliest_ref;
    std::map<string, int64> latest_ref;
    // ground truth memory stats. time->bytes.
    std::map<int64, int64> allocator_stats;
  };

  void TrackNode(int64 step, const GraphNode* node);

  void TrackNodeConnection(int64 step, const GraphNode* node,
                           const GraphNode* src);

  const std::map<string, Device>& devices() const { return devices_; }

 private:
  std::map<string, Device> devices_;
};

class Timeline {
 public:
  Timeline(int64 step, const string& outfile)
      : step_(step), outfile_(outfile) {}
  ~Timeline() {}

  int64 step() const { return step_; }
  void SetStep(int64 step) { step_ = step; }

  void GenerateGraphTimeline(const std::vector<GraphNode*>& gnodes);

  void GenerateScopeTimeline(const ScopeNode* node);

  void GenerateCodeTimeline(const CodeNode* node);

  void TrackNode(const GraphNode* node) { mem_tracker_.TrackNode(step_, node); }

  void TrackNodeConnection(GraphNode* node, GraphNode* src) {
    mem_tracker_.TrackNodeConnection(step_, node, src);
  }

 private:
  void OutputTimeline();

  template <typename Node>
  void EmitTreeNode(const Node* node, int64 start_time, int64 duration,
                    int64 depth, std::set<int64>* visited_depth) {
    if (visited_depth->find(depth) == visited_depth->end()) {
      chrome_formatter_.EmitPID(strings::StrCat("Scope:", depth), depth);
      visited_depth->insert(depth);
    }

    Json::Value args(Json::objectValue);
    args["name"] = Json::Value(node->name());
    args["op"] = Json::Value(node->name());
    chrome_formatter_.EmitRegion(start_time, duration, depth, 0, "Op",
                                 node->name(), args);

    int64 total_micros = 0;
    int64 c_start_time = start_time;
    for (const Node* child : node->show_children) {
      int64 total_exec_micros = child->proto().total_exec_micros();
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

  int64 AllocatePID();

  int64 step_;
  const string outfile_;
  int64 next_pid_ = 0;
  int64 allocator_pid_ = -1;
  MemoryTracker mem_tracker_;
  ChromeTraceFormatter chrome_formatter_;
  std::map<string, int64> device_pids_;

  std::map<string, std::unique_ptr<Process>> process_;
  std::map<int64, std::map<int64, std::map<int64, TimeNode*>>> alloc_nodes_;
  std::map<string, std::map<int64, std::unique_ptr<TimeNode>>> tnodes_;
};

}  // namespace tfprof
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_TFPROF_TIMELINE_H_
