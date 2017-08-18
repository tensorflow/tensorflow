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

#include "tensorflow/core/profiler/internal/tfprof_timeline.h"

#include <utility>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/profiler/internal/tfprof_utils.h"

namespace tensorflow {
namespace tfprof {
namespace {
string GetTimeDevName(const string& dev) {
  if (dev.find("stream") != dev.npos) {
    return strings::StrCat("Op execution threads: ", dev);
  } else {
    return strings::StrCat("Op scheduling threads: ", dev);
  }
}
string GetMemoryLaneName(const string& dev) {
  return strings::StrCat("mem usage on:", dev);
}
}  // namespace

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

void ChromeTraceFormatter::EmitCounter(const string& category,
                                       const string& name, int64 pid, int64 ts,
                                       const string& device, int64 bytes) {
  Json::Value event = CreateEvent("C", category, name, pid, 0, ts);
  Json::Value args(Json::objectValue);
  args[device] = Json::Value(bytes);
  event["args"] = args;
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
  Json::FastWriter writer;
  string trace_str = writer.write(trace);
  if (trace_str.length() > 200 * 1024 * 1024) {
    fprintf(stderr,
            "Trace file is over 200MB. Chrome might not be able to "
            "display it. Consider to use filters (e.g. -min_micros "
            "> 1000 or -op_type .*gpu:0.* to reduce the size.\n");
  }
  return trace_str;
}

void MemoryTracker::TrackNode(int64 step, const GraphNode* node) {
  if (!node->Trackable(step)) {
    return;
  }
  Device& dev = devices_[node->node->canonical_device()];
  int64 end_micros = node->node->latest_end_micros(step);
  if (node->node->accelerator_persistent_bytes(step) != 0) {
    string tensor_name = strings::StrCat(node->name(), ":", -1);
    dev.earliest_ref[tensor_name] = node->node->all_start_micros(step);
    dev.tensor_size[tensor_name] =
        node->node->accelerator_persistent_bytes(step);
    // TODO(xpan): Need latest_ref?
  }
  if (node->node->accelerator_temp_bytes(step)) {
    string tensor_name = strings::StrCat(node->name(), ":", -2);
    dev.earliest_ref[tensor_name] = node->node->all_start_micros(step);
    dev.latest_ref[tensor_name] = end_micros;
    dev.tensor_size[tensor_name] = node->node->accelerator_temp_bytes(step);
  }
  if (node->node->allocator_bytes_in_use(step) > 0) {
    dev.allocator_stats[end_micros] = node->node->allocator_bytes_in_use(step);
  }
}

void MemoryTracker::TrackNodeConnection(int64 step, const GraphNode* node,
                                        const GraphNode* src) {
  if (!node->Trackable(step) || !src->Trackable(step)) {
    return;
  }
  const auto& output_idx = node->node->src_output_idx().find(src->name());
  if (output_idx == node->node->src_output_idx().end()) {
    return;
  }
  const auto& output = src->node->output_memory(step).find(output_idx->second);
  if (output == src->node->output_memory(step).end()) {
    return;
  }
  int64 output_bytes = output->second.first;
  uint64 output_ptr = output->second.second;

  Device& src_dev = devices_[src->node->canonical_device()];
  string tensor_name = strings::StrCat(output_ptr);
  if (output_ptr == 0) {
    fprintf(stderr, "output no ptr\n");
    tensor_name = strings::StrCat(src->node->name(), ":", output_idx->second);
  }

  src_dev.tensor_size[tensor_name] = output_bytes;
  src_dev.earliest_ref[tensor_name] = src->node->all_start_micros(step);

  int64 src_end_micros = src->node->latest_end_micros(step);

  if (src->node->canonical_device() != node->node->canonical_device()) {
    int64 transfer_micros =
        (src_end_micros + node->node->all_start_micros(step)) / 2;
    src_dev.latest_ref[tensor_name] =
        std::max(src_dev.latest_ref[tensor_name], transfer_micros);

    Device& dest_dev = devices_[node->node->canonical_device()];
    string dest_tensor_name =
        strings::StrCat(tensor_name, node->node->canonical_device());
    dest_dev.tensor_size[dest_tensor_name] = output_bytes;
    dest_dev.earliest_ref[dest_tensor_name] = transfer_micros;
    dest_dev.latest_ref[dest_tensor_name] =
        std::max(dest_dev.latest_ref[dest_tensor_name],
                 node->node->latest_end_micros(step));
  } else {
    src_dev.latest_ref[tensor_name] = std::max(
        src_dev.latest_ref[tensor_name], node->node->latest_end_micros(step));
  }
}

void Timeline::AllocateTimeNodes(GraphNode* gnode) {
  if (gnode->Trackable(step_)) {
    TrackNode(gnode);
    const TFGraphNode* node = gnode->node;
    for (const auto& kernel_execs : node->op_execs(step_)) {
      const string& device = kernel_execs.first;

      if (process_.find(device) == process_.end()) {
        int64 pid = AllocatePID();
        process_[device].reset(new Process(device, pid));
        chrome_formatter_.EmitPID(GetTimeDevName(device), pid);
      }
      Process* p = process_[device].get();

      for (const auto& exec : kernel_execs.second) {
        int64 start_micros = exec.first;
        int64 exec_micros = exec.second;
        // TODO(xpan): There might be start time duplication here.
        if (tnodes_[device].find(start_micros) == tnodes_[device].end()) {
          // TODO(xpan): Give each kernel call a unique_name.
          tnodes_[device][start_micros].reset(
              new TimeNode(p, gnode, start_micros, exec_micros));
        }
      }
    }
  }
  for (GraphNode* n : gnode->show_children) {
    AllocateTimeNodes(n);
  }
}

void Timeline::GenerateGraphTimeline(const std::vector<GraphNode*>& gnodes) {
  for (GraphNode* gnode : gnodes) {
    AllocateTimeNodes(gnode);
  }
  for (auto& process : tnodes_) {
    for (auto& tn : process.second) {
      TimeNode* tnode = tn.second.get();
      for (GraphNode* inp : tnode->node->children) {
        if (!inp->account || !inp->Trackable(step_)) {
          continue;
        }
        TrackNodeConnection(tnode->node, inp);
        for (const auto& kernel_execs : inp->node->op_execs(step_)) {
          if (process.first == kernel_execs.first) {
            // Not interested in flow withthin the same device.
            continue;
          }
          for (const auto& exec : kernel_execs.second) {
            int64 start_micros = exec.first;
            auto cprocess = tnodes_.find(kernel_execs.first);
            if (cprocess == tnodes_.end()) continue;
            auto ctn = cprocess->second.find(start_micros);
            if (ctn == cprocess->second.end()) continue;
            ctn->second->next_tnodes.push_back(tnode);
          }
        }
      }
    }
  }

  AllocateLanes();
  fprintf(stdout, "generating trace file.\n");
  int64 flow_id = 1;
  for (const auto& process : alloc_nodes_) {
    for (const auto& lane : process.second) {
      for (const auto& node : lane.second) {
        TimeNode* tnode = node.second;

        Json::Value args(Json::objectValue);
        args["name"] = Json::Value(tnode->name());
        args["op"] = Json::Value(tnode->name());
        chrome_formatter_.EmitRegion(node.first, tnode->exec_micros,
                                     process.first, lane.first, "Op",
                                     tnode->name(), args);
        // Flow is a directed arrow pointing from src to dst.
        // TODO(xpan): Disable flow to reduce json file size for now. Need
        // to think of a better way to make flow interpretable.
        for (TimeNode* next_tnode : node.second->next_tnodes) {
          chrome_formatter_.EmitFlowStart(
              tnode->name() + "_flow", tnode->start_micros + tnode->exec_micros,
              process.first, lane.first, flow_id);
          chrome_formatter_.EmitFlowEnd(
              tnode->name() + "_flow", next_tnode->start_micros,
              next_tnode->process->pid, next_tnode->tid, flow_id);
          flow_id += 1;
        }
      }
    }
  }
  for (const auto& dev : mem_tracker_.devices()) {
    int64 pid = AllocatePID();
    chrome_formatter_.EmitPID(GetMemoryLaneName(dev.first), pid);
    const MemoryTracker::Device& device = dev.second;

    for (const auto& alloc_stats : device.allocator_stats) {
      chrome_formatter_.EmitCounter("Memory", "Memory Series", pid,
                                    alloc_stats.first, dev.first,
                                    alloc_stats.second);
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

void Timeline::AllocateLanes() {
  for (auto& process : tnodes_) {
    Process* p = process_[process.first].get();
    for (auto& tnode : process.second) {
      int64 start_time = tnode.second->start_micros;
      int64 end_time = tnode.second->start_micros + tnode.second->exec_micros;
      int64 l = -1;
      for (int64 i = 0; i < p->lanes.size(); ++i) {
        const auto& lane = p->lanes[i];
        l = i;
        for (auto cur_it = lane.rbegin(); cur_it != lane.rend(); ++cur_it) {
          if (cur_it->second > start_time) {
            l = -1;
            break;
          }
          if (start_time > cur_it->second) {
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
