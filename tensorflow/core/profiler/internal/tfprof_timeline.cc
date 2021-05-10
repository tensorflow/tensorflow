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

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/profiler/internal/tfprof_utils.h"

namespace tensorflow {
namespace tfprof {
namespace {
int kMaxDisplayedMemNode = 10;

std::string GetTimeDevName(const std::string& dev) {
  if (dev.find("stream") != dev.npos) {
    return absl::StrCat("Op execution threads: ", dev);
  } else {
    return absl::StrCat("Op scheduling threads: ", dev);
  }
}
std::string GetMemoryLaneName(const std::string& dev) {
  return absl::StrCat("mem usage on:", dev);
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
  event["pid"] = Json::Int64(pid);
  event["tid"] = Json::Int64(tid);
  event["ts"] = Json::Int64(ts);
  return event;
}

void ChromeTraceFormatter::EmitPID(const string& name, int64 pid) {
  Json::Value event(Json::objectValue);
  event["name"] = Json::Value("process_name");
  event["ph"] = Json::Value("M");
  event["pid"] = Json::Int64(pid);
  Json::Value args(Json::objectValue);
  args["name"] = Json::Value(name);
  event["args"] = args;
  metadata_.push_back(event);
}

void ChromeTraceFormatter::EmitRegion(int64 ts, int64 duration, int64 pid,
                                      int64 tid, const string& category,
                                      const string& name, Json::Value args) {
  Json::Value event = CreateEvent("X", category, name, pid, tid, ts);
  event["dur"] = Json::Int64(duration);
  event["args"] = std::move(args);
  metadata_.push_back(event);
}

void ChromeTraceFormatter::EmitFlowStart(const string& name, int64 ts,
                                         int64 pid, int64 tid, int64 flow_id) {
  Json::Value event = CreateEvent("s", "DataFlow", name, pid, tid, ts);
  event["id"] = Json::Int64(flow_id);
  events_.push_back(event);
}

void ChromeTraceFormatter::EmitFlowEnd(const string& name, int64 ts, int64 pid,
                                       int64 tid, int64 flow_id) {
  Json::Value event = CreateEvent("t", "DataFlow", name, pid, tid, ts);
  event["id"] = Json::Int64(flow_id);
  events_.push_back(event);
}

void ChromeTraceFormatter::EmitCounter(
    const string& category, const string& name, int64 pid, int64 ts,
    const string& device, int64 bytes,
    const std::map<int64, std::vector<string>>& tensor_mem) {
  Json::Value event = CreateEvent("C", category, "Allocated Bytes", pid, 0, ts);
  Json::Value args(Json::objectValue);
  args["Allocator Bytes in Use"] = Json::Int64(bytes);
  event["args"] = args;
  events_.push_back(event);

  // TODO(xpan): chrome://tracing is not ideal visualization for memory.
  // It would be great to have a customized UI for it.
  Json::Value event2 =
      CreateEvent("C", category, "Top Allocations", pid + 1, 0, ts);
  Json::Value args2(Json::objectValue);
  // Need to reserve the same args for all locations.
  for (int i = 1; i < kMaxDisplayedMemNode; ++i) {
    args2[absl::StrFormat("Top Allocation %02d", i)] = Json::Value("N/A");
  }
  int count = 0;
  for (auto it = tensor_mem.rbegin(); it != tensor_mem.rend(); ++it) {
    for (const string& t : it->second) {
      if (bytes < it->first || count >= kMaxDisplayedMemNode) {
        break;
      }
      args2[absl::StrFormat("Top Allocation %02d", count)] =
          Json::Value(absl::StrCat(it->first / 1000000.0, " MB from ", t));
      ++count;
      bytes -= it->first;
    }
  }
  args2[std::string("Not Displayed")] =
      Json::Value(absl::StrFormat("%.2f MB", bytes / 1000000.0));
  event2["args"] = args2;
  events_.push_back(event2);
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
    absl::FPrintF(stderr,
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

  std::map<int64, int64> allocs;
  for (const auto& alloc : node->node->allocations(step)) {
    allocs[alloc.alloc_micros()] += alloc.alloc_bytes();
    dev.tracked_allocations[alloc.alloc_micros()] += alloc.alloc_bytes();
  }
  dev.tracked_allocations[0] += node->node->accelerator_persistent_bytes();
  allocs[0] += node->node->accelerator_persistent_bytes();

  int64 last = 0;
  std::map<int64, int64>& aggregate_allocs = dev.tensor_allocs[node->name()];
  for (auto it = allocs.begin(); it != allocs.end(); ++it) {
    last += it->second;
    aggregate_allocs[it->first] = last;
  }
  for (const auto& bytes_in_use : node->node->allocator_bytes_in_use(step)) {
    if (bytes_in_use.first <= 0) continue;
    dev.allocations[bytes_in_use.first] = bytes_in_use.second;
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
  // To save memory, we only track cross-device (canonical device) flows.
  for (auto& process : tnodes_) {
    if (!IsCanonicalDevice(process.first)) continue;
    for (auto& tn : process.second) {
      TimeNode* tnode = tn.second.get();
      for (GraphNode* inp : tnode->node->children) {
        if (!inp->account || !inp->Trackable(step_)) {
          continue;
        }
        for (const auto& execs : inp->node->cpu_execs(step_)) {
          if (!IsCanonicalDevice(execs.first)) continue;
          if (process.first == execs.first) {
            // Not interested in flow within the same device.
            continue;
          }
          for (const auto& exec : execs.second) {
            int64 start_micros = exec.first;
            auto cprocess = tnodes_.find(execs.first);
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
  absl::FPrintF(stdout, "generating trace file.\n");
  int64 flow_id = 1;
  for (const auto& process : alloc_nodes_) {
    for (const auto& lane : process.second) {
      for (const auto& node : lane.second) {
        TimeNode* tnode = node.second;

        Json::Value args(Json::objectValue);
        args["name"] = Json::Value(tnode->name());
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
    if (IsPlacedOnCPU(dev.first)) {
      // TODO(xpan): Maybe also support CPU allocator memory tracking.
      continue;
    }
    int64 pid = AllocatePID();
    chrome_formatter_.EmitPID(GetMemoryLaneName(dev.first), pid);
    int64 pid2 = AllocatePID();
    chrome_formatter_.EmitPID(GetMemoryLaneName(dev.first) + " allocations",
                              pid2);

    const MemoryTracker::Device& device = dev.second;

    int64 max_bytes_in_use = 0;
    int64 cur_bytes_in_use = 0;
    int64 last_point = 0;
    for (const auto& alloc : device.allocations) {
      cur_bytes_in_use = alloc.second;
      max_bytes_in_use = std::max(max_bytes_in_use, cur_bytes_in_use);
      // Do not plot too dense to reduce file size.
      int64 ts = alloc.first;
      if (ts - last_point < 100) continue;
      last_point = ts;

      std::map<int64, std::vector<string>> tensor_mem;
      for (const auto& tensor_alloc_it : dev.second.tensor_allocs) {
        const auto& tensor_alloc = tensor_alloc_it.second;
        auto it = tensor_alloc.lower_bound(ts);
        if (it != tensor_alloc.begin()) {
          --it;
        }
        if (it->second > 0) {
          tensor_mem[it->second].push_back(tensor_alloc_it.first);
        }
      }
      chrome_formatter_.EmitCounter("Memory", "Memory Series", pid, ts,
                                    dev.first, cur_bytes_in_use, tensor_mem);
    }
    if (IsPlacedOnAccelerator(dev.first)) {
      absl::FPrintF(stdout, "%s peak memory: %.2f MB\n", dev.first,
                    max_bytes_in_use / 1000000.0);
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
  std::string outfile = absl::StrFormat("%s_%d", outfile_, step());
  Status s =
      WriteStringToFile(Env::Default(), outfile, chrome_formatter_.Format());
  if (!s.ok()) {
    absl::FPrintF(stderr, "Failed to write timeline file: %s\nError: %s\n",
                  outfile, s.ToString());
    return;
  }
  absl::FPrintF(stdout,
                "\n******************************************************\n");
  absl::FPrintF(stdout,
                "Timeline file is written to %s.\n"
                "Open a Chrome browser, enter URL chrome://tracing and "
                "load the timeline file.",
                outfile);
  absl::FPrintF(stdout,
                "\n******************************************************\n");
  fflush(stdout);
}

void Timeline::AllocateLanes() {
  for (auto& process : tnodes_) {
    Process* p = process_[process.first].get();
    for (auto& tnode : process.second) {
      int64 start_time = tnode.second->start_micros;
      int64 end_time = tnode.second->start_micros + tnode.second->exec_micros;
      int64 l = -1;
      for (int64 i = 0, end = p->lanes.size(); i < end; ++i) {
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
