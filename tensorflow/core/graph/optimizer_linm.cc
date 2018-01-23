/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/graph/optimizer_linm.h"

#include <unordered_map>
#include <utility>
#include <vector>
#include <deque>
#include <sstream>
#include <string>
#include <iostream>
#include <unordered_set>

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/framework/function.h"

namespace tensorflow {

// Information about a loop argument.
struct Arg {
  // Every loop argument has an Enter node.
  Node* enter = nullptr;

  // Is the loop argument a loop-invariant value? Taken from the `is_constant`
  // attribute on the Enter node.
  bool is_loop_invariant = false;

  // If 'is_loop_invariant' is true, the following are all nullptr. Non-constant
  // arguments must have all of the following nodes:
  Node* merge = nullptr;
  Node* switch_node = nullptr;
  Node* next_iteration = nullptr;
  Node* exit = nullptr;
};

// Information about a loop frame.
struct Frame {
  string name;

  // Pointer to the parent frame. The root frame has a pointer to itself.
  Frame* parent = nullptr;
  int num_children = 0;

  // Arguments to this loop.
  std::vector<Arg> args;

  // The loop condition of the loop. There should be exactly one loop condition
  // in every loop.
  Node* loop_cond = nullptr;

  // Set of nodes that belong to the loop frame.
  std::unordered_set<Node*> nodes;
};

class OptimizerLINM {
 public:
  explicit OptimizerLINM(Graph* g) : g_(g), changed_(false) {}

  Status Optimize();

  bool changed() {return changed_;}

 private:
  Graph* g_;
  bool changed_;
  static int new_enter_id_;
  static int new_const_id_;
  std::vector<ControlFlowInfo> cf_info;
  Status loop_invariant_subgraph_motion(
      const std::unordered_map<Node*, int>& invariant_nodes, Frame* frame);
  NodeDef CreateEnterNode(const string& name, const string& op,
                          const string& frame,
                          const DataType dtype, const int piterations,
                          const std::vector<string>& inputs) {
    NodeDef node;
    node.set_name(name);
    if (!op.empty()) {
      node.set_op(op);
    }
    if (!frame.empty()) {
      AttrValue frame_name;
      frame_name.set_s(frame);
      node.mutable_attr()->insert({"frame_name", frame_name});
      AttrValue data_type;
      data_type.set_type(dtype);
      node.mutable_attr()->insert({"T", data_type});
      AttrValue is_const;
      is_const.set_b(true);
      node.mutable_attr()->insert({"is_constant", is_const});
      AttrValue parallel_iterations;
      parallel_iterations.set_i(piterations);
      node.mutable_attr()->insert({"parallel_iterations", parallel_iterations});
    }
    for (const string& input : inputs) {
      node.add_input(input);
    }
    return node;
  }
};
int OptimizerLINM::new_enter_id_ = 0;
int OptimizerLINM::new_const_id_ = 0;

static void traverse(
    Node* node, std::unordered_map<Node*, int>* invariant_nodes) {
  invariant_nodes->insert(std::make_pair(node, node->out_edges().size()));
  for (auto* e : node->out_edges()) {
    auto* dst = e->dst();
    if (dst->IsEnter() ||
        (invariant_nodes->find(dst) != invariant_nodes->end())) {
      continue;
    }
    bool all_invariant = true;
    for (auto* i : dst->in_edges()) {
      auto* producer = i->src();
      if (i->IsControlEdge()) {
      } else if (producer->IsConstant()) {
        invariant_nodes->insert(
            std::make_pair(producer, producer->out_edges().size()));
      } else if (invariant_nodes->find(producer) == invariant_nodes->end()) {
        all_invariant = false;
      }
    }
    for (auto* i : dst->out_edges()) {
      if (i->IsControlEdge()) {
        all_invariant = false;
        continue;
      }
    }
    if (!all_invariant) {
      continue;
    }
    VLOG(2) << "found loop invariant node " << dst->name();
    for (auto* i : dst->in_edges()) {
      if (!i->IsControlEdge()) {
        auto* producer = i->src();
        invariant_nodes->find(producer)->second--;
      }
    }
    traverse(dst, invariant_nodes);
  }
}

Status OptimizerLINM::loop_invariant_subgraph_motion(
    // TODO(minmin): break this function into 3 functions
    const std::unordered_map<Node*, int>& invariant_nodes, Frame* frame) {
    // loop invariant node/subgraph motion
    for (auto iter = invariant_nodes.begin();
         iter != invariant_nodes.end(); iter++) {
      auto* invariant_node = iter->first;

      if (invariant_node->IsEnter()) {
        VLOG(2) << "handling invariant enter " << iter->first->name();
        const Edge* enter_input = nullptr;
        std::vector<const Edge*> control_edges;
        for (auto* e : invariant_node->in_edges()) {
          if (e->IsControlEdge()) {
            control_edges.push_back(e);
          } else {
            if (enter_input) {
              return errors::InvalidArgument(
                  "Enter Node can't have more than 1 input tensor");
            }
            enter_input = e;
          }
        }
        if (!enter_input) {
          return errors::InvalidArgument(
              "Enter Node can't have 0 input tensor");
        }

        std::vector<const Edge*> out_edges;
        for (auto* e : invariant_node->out_edges()) {
          if (!e->IsControlEdge()) {
            out_edges.push_back(e);
          }
        }
        for (auto* e : out_edges) {
          auto* dst = e->dst();
          if (invariant_nodes.find(dst) != invariant_nodes.end()) {
            int dst_input = e->dst_input();
            g_->RemoveEdge(e);
            g_->AddEdge(enter_input->src(),
                        enter_input->src_output(), dst, dst_input);
            changed_ = true;
            for (auto* control_edge : control_edges) {
              // keep the control edges of Enter Node
              g_->AddControlEdge(control_edge->src(), dst);
              changed_ = true;
            }
          }
        }
      } else if (invariant_node->IsConstant()) {
        VLOG(2) << "handling const " << iter->first->name();
        if (iter->second == 0) {
          // all successor nodes are invariant nodes
          for (auto* e : invariant_node->out_edges()) {
            auto* dst = e->dst();
            if (invariant_nodes.find(dst) == invariant_nodes.end()) {
              return errors::InvalidArgument(
                  "All successors of this Node are expected to be invariant");
            }
          }
          std::vector<const Edge*> in_edges;
          for (auto* i : invariant_node->in_edges()) {
            in_edges.push_back(i);
          }
          for (auto* i : in_edges) {
            if (i->IsControlEdge() &&
                  cf_info[i->src()->id()].frame_name == frame->name) {
              // have to remove control edges from the invariant node
              // when moving this node out of this frame
              g_->RemoveEdge(i);
              changed_ = true;
            } else {
              return errors::InvalidArgument(
                  "A Const Node can't have any in edge other"
                  "than control edge from the same frame");
            }
          }
          g_->AddControlEdge(g_->source_node(), invariant_node);
        } else if (iter->second < invariant_node->out_edges().size()) {
          // some successor nodes are invariant nodes
          NodeDef node;
          std::ostringstream new_const_stream;
          new_const_stream << "linm_new_const_" << new_const_id_++;
          string new_const_name = new_const_stream.str();
          node.set_name(new_const_name);
          node.set_op("Const");
          for (auto it = invariant_node->def().attr().begin();
               it != invariant_node->def().attr().end(); it++) {
            node.mutable_attr()->insert({it->first, it->second});
          }

          Status status;
          Node* new_const = g_->AddNode(node, &status);
          if (status != Status::OK()) {
            return status;
          }
          std::vector<const Edge*> out_edges;
          for (auto* e : invariant_node->out_edges()) {
            out_edges.push_back(e);
          }
          for (auto* e : out_edges) {
            auto* dst = e->dst();
            if (invariant_nodes.find(dst) != invariant_nodes.end()) {
              // connect invariant successor to new_const
              int src_output = e->src_output();
              int dst_input = e->dst_input();
              g_->RemoveEdge(e);
              g_->AddEdge(new_const, src_output, dst, dst_input);
              changed_ = true;
            }
          }
          g_->AddControlEdge(g_->source_node(), new_const);
        }
      } else {
        VLOG(2) << "processing node " << invariant_node->name();
        std::vector<const Edge*> control_edges;
        for (auto* i : invariant_node->in_edges()) {
          if (i->IsControlEdge() &&
              cf_info[i->src()->id()].frame_name == frame->name) {
            control_edges.push_back(i);
          }
        }
        for (auto* i : control_edges) {
          // have to remove control edges from the invariant node
          // when moving this node out of this frame
          g_->RemoveEdge(i);
          changed_ = true;
        }
        if (iter->second == 0) {
          continue;
        }
        std::vector<const Edge*> out_edges;
        for (auto* e : invariant_node->out_edges()) {
          out_edges.push_back(e);
        }
        for (auto* e : out_edges) {
          auto* dst = e->dst();
          if (invariant_nodes.find(dst) == invariant_nodes.end()) {
            // loop variant successor
            int src_output = e->src_output();
            int dst_input = e->dst_input();

            DataType dtype = invariant_node->output_type(src_output);
            int piterations;
            GetNodeAttr(frame->args[0].enter->def(),
                        "parallel_iterations", &piterations);

            std::ostringstream new_enter_stream;
            new_enter_stream << "linm_new_enter_" << new_enter_id_++;
            string new_enter_name = new_enter_stream.str();
            std::ostringstream input_name_stream;
            input_name_stream << invariant_node->def().name()
                              << ":" << src_output;
            string input_name = input_name_stream.str();
            NodeDef node_def = CreateEnterNode(new_enter_name, "Enter",
                                               frame->name, dtype,
                                               piterations, {input_name});
            Status status;
            auto* new_enter = g_->AddNode(node_def, &status);
            if (status != Status::OK()) {
              return status;
            }

            g_->RemoveEdge(e);
            g_->AddEdge(invariant_node, src_output, new_enter, 0);
            g_->AddEdge(new_enter, 0, dst, dst_input);
            changed_ = true;
          }
        }
      }
    }  // end of for
    return Status::OK();
}

Status OptimizerLINM::Optimize() {
  VLOG(2) << "Runing OptimizerLINM Pass";
  // Note: BuildControlFlowInfo() requires that the graph's source node is
  // connected to all source nodes in the graph. Many graphs violate this
  // invariant.
  TF_RETURN_IF_ERROR(BuildControlFlowInfo(g_, &cf_info));

  // Builds Frames, indexed by name.
  std::unordered_map<string, Frame> frames;
  for (Node* node : g_->op_nodes()) {
    const ControlFlowInfo& cf = cf_info[node->id()];

    VLOG(2) << "node: " << node->name() << " frame_name: " << cf.frame_name
            << " frame: " << (cf.frame ? cf.frame->name() : "---")
            << " parent_frame: "
            << (cf.parent_frame ? cf.parent_frame->name() : "---");
    if (cf.frame == nullptr) {
      return errors::InvalidArgument("cf.frame == nullptr for node ",
                                     node->name());
    }
    if (cf.parent_frame == nullptr) {
      return errors::InvalidArgument("cf.parent_frame == nullptr for node ",
                                     node->name());
    }

    Frame& frame = frames[cf.frame_name];
    Frame* parent = &frames[cf_info[cf.parent_frame->id()].frame_name];
    if (frame.parent == nullptr) {
      frame.parent = parent;
      frame.name = cf.frame_name;
      ++parent->num_children;
    } else if (frame.parent != parent) {
      return errors::InvalidArgument("Mismatched parent frames for ",
                                     cf.frame->id(), ": ", parent->name, " vs ",
                                     frame.parent->name);
    }

    if (IsEnter(node)) {
      Arg arg;
      arg.enter = node;
      TF_RETURN_IF_ERROR(GetNodeAttr(arg.enter->attrs(), "is_constant",
                                     &arg.is_loop_invariant));
      frame.args.push_back(arg);
    } else if (IsLoopCond(node)) {
      if (frame.loop_cond) {
        return errors::InvalidArgument(
            "Loop ", cf.frame_name,
            " has more than one LoopCond node: ", node->name(), " and ",
            frame.loop_cond->name());
      }
      frame.loop_cond = node;
    }
    frame.nodes.insert(node);
  }

  // Adds frames with no children (i.e., the innermost frames) to a worklist.
  std::deque<Frame*> worklist;
  for (auto& frame : frames) {
    if (frame.second.num_children == 0) {
      worklist.push_back(&frame.second);
    }
  }

  while (!worklist.empty()) {
    Frame* frame = worklist.front();
    worklist.pop_front();
    if (frame->parent == frame) {
      // Skip the root frame.
      continue;
    }

    VLOG(2) << "begin loop invariant node/subgraph detection";
    std::unordered_map<Node*, int> invariant_nodes;
    // loop invariant node/subgraph detection
    for (auto& arg : frame->args) {
      if (!arg.is_loop_invariant) {
        continue;
      } else {
        traverse(arg.enter, &invariant_nodes);
      }
    }

    VLOG(2) << "begin loop invariant node/subgraph motion";
    TF_RETURN_IF_ERROR(loop_invariant_subgraph_motion(invariant_nodes, frame));
    VLOG(2) << "finished loop invariant node/subgraph motion";

    // If the parent has no remaining children, add it to the worklist.
    --frame->parent->num_children;
    if (frame->parent->num_children == 0) {
      worklist.push_back(frame->parent);
    }
  }  // end of while
  return Status::OK();
}

bool OptimizeLINM(Graph* g) {
  OptimizerLINM opt(g);
  auto ret = opt.Optimize();
  if (ret != Status::OK()) {
    // TODO(minmin): roll back to the graph before this pass instead of exit
    std::cerr << "LINM Failed: " << ret.error_message() << std::endl;
    exit(1);
  }
  return opt.changed();
}

}  // namespace tensorflow
