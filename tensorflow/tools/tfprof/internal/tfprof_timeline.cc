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

#include "tensorflow/tools/tfprof/internal/tfprof_timeline.h"

#include <utility>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/tools/tfprof/internal/tfprof_utils.h"

namespace tensorflow {
namespace tfprof {

Json::Value ChromeTraceFormatter::CreateEvent(const string& ph,
                                              const string& category,
                                              const string& name, int64 pid,
                                              int64 tid, int64 ts) {
  Json::Value event(Json::objectValue);
  event["ph"] = Json::Value(ph);
  event["cat"] = Json::Value(category);
  event["name"] = Json::Value(name);
  event["pid"] = Json::Value(pid);
  event["tid"] = Json::Value(tid);
  event["ts"] = Json::Value(ts);
  return event;
}

void ChromeTraceFormatter::EmitPID(const string& name, int64 pid) {
  Json::Value event(Json::objectValue);
  event["name"] = Json::Value("process_name");
  event["ph"] = Json::Value("M");
  event["pid"] = Json::Value(pid);
  Json::Value args(Json::objectValue);
  args["name"] = Json::Value(name);
  event["args"] = args;
  metadata_.push_back(event);
}

void ChromeTraceFormatter::EmitRegion(int64 ts, int64 duration, int64 pid,
                                      int64 tid, const string& category,
                                      const string& name, Json::Value args) {
  Json::Value event = CreateEvent("X", category, name, pid, tid, ts);
  event["dur"] = Json::Value(duration);
  event["args"] = std::move(args);
  metadata_.push_back(event);
}

void ChromeTraceFormatter::EmitFlowStart(const string& name, int64 ts,
                                         int64 pid, int64 tid, int64 flow_id) {
  Json::Value event = CreateEvent("s", "DataFlow", name, pid, tid, ts);
  event["id"] = flow_id;
  events_.push_back(event);
}

void ChromeTraceFormatter::EmitFlowEnd(const string& name, int64 ts, int64 pid,
                                       int64 tid, int64 flow_id) {
  Json::Value event = CreateEvent("t", "DataFlow", name, pid, tid, ts);
  event["id"] = flow_id;
  events_.push_back(event);
}

string ChromeTraceFormatter::Format() {
  Json::Value trace;
  trace["traceEvents"] = Json::Value(Json::arrayValue);
  for (const Json::Value& v : metadata_) {
    trace["traceEvents"].append(v);
  }
  for (const Json::Value& v : events_) {
    trace["traceEvents"].append(v);
  }
  return trace.toStyledString();
}

void Timeline::GenerateGraphTimeline(const GraphNode* gnode) {
  fprintf(stdout, "adding graph nodes.\n");
  AddGraphNode(gnode);
  AllocateLanes();
  fprintf(stdout, "generating trace file.\n");
  int64 flow_id = 1;
  for (const auto& process : alloc_nodes_) {
    for (const auto& lane : process.second) {
      for (const auto& node : lane.second) {
        TimeNode* tnode = node.second;

        Json::Value args(Json::objectValue);
        args["name"] = Json::Value(tnode->name);
        args["op"] = Json::Value(tnode->name);
        chrome_formatter_.EmitRegion(node.first, tnode->exec_micros,
                                     process.first, lane.first, "Op",
                                     tnode->name, args);

        for (TimeNode* next_tnode : node.second->next_tnodes) {
          chrome_formatter_.EmitFlowStart(
              tnode->name + "_flow", tnode->start_micros + tnode->exec_micros,
              process.first, lane.first, flow_id);
          chrome_formatter_.EmitFlowEnd(
              tnode->name + "_flow", next_tnode->start_micros,
              next_tnode->process->pid, next_tnode->tid, flow_id);
          flow_id += 1;
        }
      }
    }
  }
  OutputTimeline();
}

void Timeline::GenerateScopeTimeline(const ScopeNode* node) {
  std::set<int64> visited_depth;
  EmitTreeNode(node, 0, node->proto().total_exec_micros(), 0, &visited_depth);
  OutputTimeline();
}

void Timeline::GenerateCodeTimeline(const CodeNode* node) {
  std::set<int64> visited_depth;
  EmitTreeNode(node, 0, node->proto().total_exec_micros(), 0, &visited_depth);
  OutputTimeline();
}

void Timeline::OutputTimeline() {
  Status s =
      WriteStringToFile(Env::Default(), outfile_, chrome_formatter_.Format());
  if (!s.ok()) {
    fprintf(stderr, "Failed to write timeline file: %s\nError: %s\n",
            outfile_.c_str(), s.ToString().c_str());
    return;
  }
  fprintf(stdout, "\n******************************************************\n");
  fprintf(stdout,
          "Timeline file is written to %s.\n"
          "Open a Chrome browser, enter URL chrome://tracing and "
          "load the timeline file.",
          outfile_.c_str());
  fprintf(stdout, "\n******************************************************\n");
  fflush(stdout);
}

std::vector<TimeNode*> Timeline::AddGraphNode(const GraphNode* gnode) {
  std::vector<TimeNode*> tnodes;
  if (!gnode) return tnodes;

  std::vector<TimeNode*> shown_cinputs;
  for (GraphNode* schild : gnode->show_children) {
    std::vector<TimeNode*> inputs = AddGraphNode(schild);
    shown_cinputs.insert(shown_cinputs.end(), inputs.begin(), inputs.end());
  }
  if (!gnode->node->step_stats()) {
    return shown_cinputs;
  }

  const TFGraphNode* node = gnode->node;
  for (const auto& kernel_execs : node->op_kernel_execs()) {
    const string& device = kernel_execs.first;
    const std::vector<std::pair<int64, int64>>& execs = kernel_execs.second;

    if (process_.find(device) == process_.end()) {
      int64 pid = AllocatePID();
      process_[device].reset(new Process(pid));
      chrome_formatter_.EmitPID(device, pid);
    }
    Process* p = process_[device].get();

    for (const auto& exec : execs) {
      int64 start_micros = exec.first;
      int64 exec_micros = exec.second;
      // TODO(xpan): There might be start time duplication here.
      if (tnodes_[device].find(start_micros) == tnodes_[device].end()) {
        // TODO(xpan): Give each kernel call a unique_name.
        tnodes_[device][start_micros].reset(
            new TimeNode(p, node->name(), start_micros, exec_micros));
      }
      TimeNode* tnode_ptr = tnodes_[device][start_micros].get();

      for (int i = 0; i < shown_cinputs.size(); i++) {
        shown_cinputs[i]->next_tnodes.push_back(tnode_ptr);
      }
      tnodes.push_back(tnode_ptr);
    }
  }
  return tnodes;
}

void Timeline::AllocateLanes() {
  for (auto& process : tnodes_) {
    Process* p = process_[process.first].get();
    for (auto& tnode : process.second) {
      int64 start_time = tnode.second->start_micros;
      int64 end_time = tnode.second->exec_micros - 1;

      int64 l = -1;
      for (int i = 0; i < p->lanes.size(); ++i) {
        const auto& lane = p->lanes[i];
        auto cur_it = lane.lower_bound(start_time);
        if (cur_it == lane.end()) {
          --cur_it;
        }
        l = i;
        for (; cur_it != lane.begin(); --cur_it) {
          if (cur_it->second < start_time) {
            break;
          }
          if (cur_it->first <= end_time) {
            l = -1;
            break;
          }
        }
        if (l >= 0) {
          break;
        }
      }
      if (l < 0) {
        l = p->lanes.size();
        std::map<int64, int64> nlane;
        nlane[start_time] = end_time;
        p->lanes.push_back(nlane);
      } else {
        p->lanes[l][start_time] = end_time;
      }
      tnode.second->tid = l;
      alloc_nodes_[p->pid][l][start_time] = tnode.second.get();
    }
  }
}

int64 Timeline::AllocatePID() {
  int64 cur_pid = next_pid_;
  next_pid_ += 1;
  return cur_pid;
}

}  // namespace tfprof
}  // namespace tensorflow
