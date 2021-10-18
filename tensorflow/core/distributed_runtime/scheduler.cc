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

#include "tensorflow/core/distributed_runtime/scheduler.h"

#include <queue>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

namespace {

// Initialize the pending count for each node.
void InitializePending(const Graph* graph, std::vector<int>* pending) {
  pending->resize(graph->num_node_ids());
  for (const Node* node : graph->nodes()) {
    const int id = node->id();
    int num_in_edges = 0;
    if (IsMerge(node)) {
      // For forward execution order, Merge nodes are special. We process
      // them only once when one of its inputs is processed.
      for (const Edge* edge : node->in_edges()) {
        if (edge->IsControlEdge()) {
          // Bit 0 is reserved to indicate if there is a data input.
          num_in_edges += 2;
        }
      }
    } else {
      num_in_edges = node->in_edges().size();
    }
    (*pending)[id] = num_in_edges;
  }
}

// Return true if the update makes the destination of the edge ready to run.
bool UpdatePending(const Edge* edge, std::vector<int>* pending_count) {
  const Node* out = edge->dst();
  if (IsMerge(out)) {
    if (edge->IsControlEdge()) {
      (*pending_count)[out->id()] -= 2;
      // Return true if we already got at least one input edge
      //   and a control edge is the enabling one.
      return ((*pending_count)[out->id()] == 1);
    } else {
      int count = (*pending_count)[out->id()];
      (*pending_count)[out->id()] |= 0x1;
      // If the first input edge is the enabling one, the count goes from
      //   0 to 1 in this step. Return true iff count was zero.
      return (count == 0);
    }
  } else {
    return (--(*pending_count)[out->id()] == 0);
  }
}

}  // end namespace

SlackAnalysis::SlackAnalysis(const Graph* g, const CostModel* cost_model)
    : graph_(g), cost_model_(cost_model) {}

Microseconds SlackAnalysis::ComputeAsap(std::vector<Microseconds>* asap_times) {
  asap_times->resize(graph_->num_node_ids());

  std::vector<int> pending_count(graph_->num_node_ids());
  InitializePending(graph_, &pending_count);

  std::deque<const Node*> queue;
  Node* srcNode = graph_->source_node();
  queue.push_back(srcNode);
  (*asap_times)[srcNode->id()] = 0;

  while (!queue.empty()) {
    const Node* curr = queue.front();
    queue.pop_front();
    Microseconds ctime = cost_model_->TimeEstimate(curr);
    for (const Edge* out_edge : curr->out_edges()) {
      // The time needed for 'out' to get its input from 'curr'.
      Microseconds copy_time(0);
      const Node* out = out_edge->dst();
      if (!out_edge->IsControlEdge() &&
          curr->assigned_device_name() != out->assigned_device_name()) {
        // Add an arbitrary 10microsecs for each copy.
        // TODO(yuanbyu): Use below with the real cost model.
        // int index = out_edge->src_output();
        // Bytes nb = cost_model_->SizeEstimate(curr, index);
        // copy_time = CostModel::CopyTimeEstimate(nb);
        copy_time = 10;
      }
      Microseconds new_asap = (*asap_times)[curr->id()] + ctime + copy_time;
      if ((*asap_times)[out->id()] < new_asap) {
        (*asap_times)[out->id()] = new_asap;
      }

      bool is_ready = UpdatePending(out_edge, &pending_count);
      if (is_ready) {
        queue.push_back(out);
      }
    }
  }
  return (*asap_times)[graph_->sink_node()->id()];
}

Microseconds SlackAnalysis::ComputeAlap(std::vector<Microseconds>* alap_times) {
  alap_times->resize(graph_->num_node_ids());

  std::vector<int> pending_count;
  pending_count.resize(graph_->num_node_ids());
  for (const Node* n : graph_->nodes()) {
    // For reverse execution order, Switch nodes are special. We process
    // them only once when one of its outputs is processed.
    if (IsSwitch(n)) {
      int32_t num_control_edges = 0;
      for (const Edge* edge : n->out_edges()) {
        if (edge->IsControlEdge()) {
          num_control_edges++;
        }
      }
      pending_count[n->id()] = num_control_edges + 1;
    } else {
      pending_count[n->id()] = n->out_edges().size();
    }
  }

  std::deque<const Node*> queue;
  Node* sinkNode = graph_->sink_node();
  queue.push_back(sinkNode);
  (*alap_times)[sinkNode->id()] = 0;

  while (!queue.empty()) {
    const Node* curr = queue.front();
    queue.pop_front();
    for (const Edge* in_edge : curr->in_edges()) {
      // The time needed for 'curr' to get its input from 'src'.
      Microseconds copy_time(0);
      const Node* src = in_edge->src();
      if (!in_edge->IsControlEdge() &&
          src->assigned_device_name() != curr->assigned_device_name()) {
        // TODO(yuanbyu): Use the real cost model
        // int index = out_edge->src_output();
        // Bytes nb = cost_model_->SizeEstimate(curr, index);
        // copy_time = CostModel::CopyTimeEstimate(nb);
        copy_time = 10;
      }
      Microseconds ctime = cost_model_->TimeEstimate(src);
      Microseconds new_latest = (*alap_times)[curr->id()] - ctime - copy_time;
      if ((*alap_times)[src->id()] > new_latest) {
        (*alap_times)[src->id()] = new_latest;
      }

      int count = --pending_count[src->id()];
      if (count == 0) {
        queue.push_back(src);
      }
    }
  }
  return (*alap_times)[graph_->source_node()->id()];
}

void SlackAnalysis::ComputeSlack(std::vector<int64_t>* slacks) {
  std::vector<Microseconds> asap_times;
  std::vector<Microseconds> alap_times;
  ComputeAsap(&asap_times);
  ComputeAlap(&alap_times);
  slacks->resize(graph_->num_node_ids());
  Node* srcNode = graph_->source_node();
  Microseconds makespan = alap_times[srcNode->id()];
  for (Node* node : graph_->nodes()) {
    Microseconds latest_stime = alap_times[node->id()] - makespan;
    (*slacks)[node->id()] = (latest_stime - asap_times[node->id()]).value();
  }
}

GreedyScheduler::GreedyScheduler(const DeviceSet* devices,
                                 const CostModel* cost_model, const Graph* g,
                                 std::vector<int64_t>* priority)
    : devices_(devices),
      cost_model_(cost_model),
      graph_(g),
      priority_(priority) {
  for (Device* d : devices_->devices()) {
    Sim* s = new Sim;
    // The number of compute units on a device. Set to 2 for now.
    s->degree_parallelism = 2;
    s->num_running = 0;
    device_states_.insert(std::make_pair(d->name(), s));
  }
}

GreedyScheduler::~GreedyScheduler() {
  for (auto& ds : device_states_) {
    delete ds.second;
  }
}

Microseconds GreedyScheduler::ComputeSchedule(
    std::vector<Microseconds>* start_times) {
  // Initialize pending_count
  std::vector<int> pending_count(graph_->num_node_ids());
  InitializePending(graph_, &pending_count);

  // Initialize event queue
  std::priority_queue<Event> event_queue;
  Event src_event;
  src_event.node = graph_->source_node();
  src_event.time = 0;
  src_event.is_completion = true;
  event_queue.push(src_event);
  Microseconds max_completion = Microseconds(0);

  while (!event_queue.empty()) {
    Event event = event_queue.top();
    event_queue.pop();
    if (event.is_completion) {
      Sim* sim = device_states_[event.node->assigned_device_name()];
      --sim->num_running;

      if (event.time > max_completion) {
        max_completion = event.time;
      }

      for (const Edge* out_edge : event.node->out_edges()) {
        Microseconds copy_time(0);
        const Node* out = out_edge->dst();
        if (!out_edge->IsControlEdge() &&
            event.node->assigned_device_name() != out->assigned_device_name()) {
          // TODO(yuanbyu): Use below with the real cost model.
          // int index = out_edge->src_output();
          // Bytes nb = cost_model_->SizeEstimate(event.node, index);
          // copy_time = CostModel::CopyTimeEstimate(nb);
          copy_time = 10;
        }
        if ((*start_times)[out->id()] < event.time + copy_time) {
          (*start_times)[out->id()] = event.time + copy_time;
        }

        bool is_ready = UpdatePending(out_edge, &pending_count);
        if (is_ready) {
          Event e{out, (*start_times)[out->id()], false};
          event_queue.push(e);
        }
      }
    } else {
      Sim* sim = device_states_[event.node->assigned_device_name()];
      sim->ready_nodes.push_back(event.node);
    }

    for (auto& x : device_states_) {
      Sim* sim = x.second;
      while (sim->num_running < sim->degree_parallelism &&
             !sim->ready_nodes.empty()) {
        Event e;
        e.node = GetNodeWithHighestPriority(sim->ready_nodes);
        e.time = event.time + cost_model_->TimeEstimate(e.node);
        e.is_completion = true;
        event_queue.push(e);
        (*start_times)[e.node->id()] = event.time;
        ++sim->num_running;
      }
    }
  }
  return max_completion;
}

const Node* GreedyScheduler::GetNodeWithHighestPriority(
    const std::vector<const Node*>& nodes) {
  const Node* curr_node = nullptr;
  int64_t curr_priority = kint64max;
  for (const Node* n : nodes) {
    if ((*priority_)[n->id()] < curr_priority) {
      curr_node = n;
      curr_priority = (*priority_)[n->id()];
    }
  }
  return curr_node;
}

PriorityScheduler::PriorityScheduler(const DeviceSet* devices,
                                     const CostModel* cost_model,
                                     const Graph* g)
    : devices_(devices), cost_model_(cost_model), graph_(g) {}

Microseconds PriorityScheduler::ComputeSchedule(
    std::vector<Microseconds>* start_times) {
  std::vector<int64_t> slacks;
  SlackAnalysis slack(graph_, cost_model_);
  slack.ComputeSlack(&slacks);
  GreedyScheduler greedysched(devices_, cost_model_, graph_, &slacks);
  return greedysched.ComputeSchedule(start_times);
}

Microseconds PriorityScheduler::AssignPriorities(
    std::vector<int64_t>* priorities) {
  std::vector<Microseconds> start_times;
  Microseconds makespan = ComputeSchedule(&start_times);

  for (const Node* n : graph_->nodes()) {
    (*priorities)[n->id()] = start_times[n->id()].value();
  }
  return makespan;
}

}  // namespace tensorflow
