/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#if defined(INTEL_MKL) && defined(ENABLE_MKL)

#include "tensorflow/core/common_runtime/mkl_tfconversion_pass.h"

#include <memory>
#include <queue>
#include <set>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/mkl_graph_util.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

// This pass inserts Mkl to Tf tensor conversion nodes (represented by C)
// in the graph in between A and B, where A and B match any one
// of the following cases:
//
//  1) A = a node that generates output in the Mkl format and,
//     B = a node that does not accept input in the Mkl format and,
//     A -> B (there is a direct edge between A and B, then
//     We will insert C such that A->C->B.
//
//  2) A = a node that generates output in the Mkl format and,
//     B = NULL (in other words, A is the last node in the graph), then
//     We will insert C such that A->C->B. (C will be the last node.)
//
//  Note that case 1 applies to all outputs of A that are input to B.
//  In other words, the conversions will be required for every output
//  of A that is input to B. For example, let us say the output of A
//  is A1, A2, A3, of which A1 and A2 are in Mkl format, but A3 is not
//  in Mkl format, and all of them are input to B. In such case, we will
//  do the conversion for A1 and A2 only. We do not need to do any conversion
//  for A3.
//
// This pass relies on ops registering themselves about their Mkl compliance.
// An Mkl-compliant op can accept inputs in the Mkl format, and produce outputs
// in the Mkl format. Non-compliant ops accept inputs and outputs in the
// TensorFlow format.
//
// ADDENDUM: For element-wise ops, we may or may not need a conversion to
// take place before we hit the op. For this, we add a new op before each
// element-wise MKL op to deal with the inputs, called _MklInputConversion.
// This pass has been enhanced to add this capability.
//
// The _MklInputConversion op will check the inputs to the elementwise op and
// make sure that either both are in MKL format or both are in TF format,
// depending on their initial state and whether broadcast is needed or not.

class MklToTfConversionPass : public GraphOptimizationPass {
 public:
  MklToTfConversionPass() {}
  Status Run(const GraphOptimizationPassOptions& options);

  // Insert layout conversion node in the graph pointed by g.
  // Function scans the graph for candidate edges where we
  // need to insert conversion nodes.
  //
  // @return true even if single conversion node is inserted;
  // false, otherwise.
  bool RunPass(std::unique_ptr<Graph>* g);

 private:
  // Is the input Op supported by Mkl-specific layout?
  //
  // @input op_name string of the op
  // @input T Datatype to use for checking input op
  // @return true if op is Mkl supported; false, otherwise.
  inline bool IsMklSupportedOp(const string& op_name, DataType T) const {
    return mkl_op_registry::IsMklLayoutDependentOp(op_name, T);
  }

  // Is the input Op supported by Mkl-specific layout AND
  //  is it element-wise?
  //
  // @input op_name string of the op
  // @input T Datatype to use for checking input op
  // @return true if op is Mkl supported; false, otherwise.
  inline bool IsMklElementWiseOp(const string& op_name, DataType T) const {
    return mkl_op_registry::IsMklElementWiseOp(op_name, T);
  }

  // Insert layout conversion node on the edge pointed by 'e' from graph 'g'.
  //
  // Edge will be deleted once a call to this function is successful.
  // Any attempt to use the edge after this call
  // will lead to undefined behaviors.
  //
  // @return Success:OK() if insertion is successful, otherwise returns
  //         appropriate error status code.
  Status InsertConversionNodeOnEdge(std::unique_ptr<Graph>* g, Edge*);

  // For element-wise ops, we need to sanitize the inputs. For this, we add a
  // new node at the input of the replacement element-wise node that checks
  // the inputs and converts one/both of them as required. See the op code
  // comments for details.
  //
  // Insert input conversion node as parent of 'n' from graph 'g'.
  //
  // @return Success:OK() if insertion is successful, otherwise returns
  //         appropriate error status code.
  Status InsertInputConversionNode(std::unique_ptr<Graph>* g, Node*);
};

// We register MklToTf insertion for phase 2 in post-partition grouping
// because we register MklLayoutRewritePass for phase 1 in post-partition
// grouping. We register this pass after partitioning so that we get a
// complete picture of inputs and outputs of the nodes in the graphs.
const OptimizationPassRegistry::Grouping kMklTfConvPassGroup =
    OptimizationPassRegistry::POST_PARTITIONING;
#ifdef ENABLE_MKL
REGISTER_OPTIMIZATION(kMklTfConvPassGroup, 2, MklToTfConversionPass);
#endif  // ENABLE_MKL

Status MklToTfConversionPass::InsertConversionNodeOnEdge(
    std::unique_ptr<Graph>* g, Edge* e) {
  CHECK_NOTNULL(e);

  Node* src = e->src();
  Node* dst = e->dst();

  CHECK_NOTNULL(src);
  CHECK_NOTNULL(dst);

  Node* conversion_node = nullptr;
  DataType src_datatype = src->output_type(e->src_output());
  DataType dst_datatype = dst->input_type(e->dst_input());
  string data_format;

  // We compare source and destination datatypes only when both are found.
  if (src_datatype != dst_datatype) {
    string err_msg = "T attribute of " + src->name() + ":" +
                     std::to_string(e->src_output()) + " and " + dst->name() +
                     ":" + std::to_string(e->dst_input()) +
                     " do not"
                     " match. Will not insert MklToTf node in such case.";
    return Status(error::Code::INVALID_ARGUMENT, err_msg.c_str());
  }

  TF_CHECK_OK(
      NodeBuilder((*g)->NewName("Mkl2Tf"), "_MklToTf")
          .Input(src, e->src_output())
          .Input(src, DataIndexToMetaDataIndex(
                          e->src_output(),
                          src->num_outputs()))  // Get an Mkl tensor slot
                                                // from the Tf tensor slot.
          .Device(src->def().device())  // We want to get conversion node
                                        // on same device as source node.
          .Attr("T", src_datatype)
          .Finalize(&**g, &conversion_node));

  CHECK_NOTNULL(conversion_node);
  // TODO(Intel-tf) MklToTf accepts only NHWC or NCHW, but doesn't seem to be
  // using data_format. This code might be redundant.
  if (GetNodeAttr(src->def(), "data_format", &data_format) == Status::OK() &&
      (data_format == ToString(FORMAT_NHWC) ||
       data_format == ToString(FORMAT_NCHW))) {
    conversion_node->AddAttr("data_format", data_format);
  }

  // Get assigned device from source node and apply it to conversion node.
  // We want conversion node to be on the same device as the source node.
  conversion_node->set_assigned_device_name(src->assigned_device_name());

  // Set the Mkl op label for this op.
  conversion_node->AddAttr("_kernel",
                           mkl_op_registry::kMklLayoutDependentOpLabel);

  // Now that we have added edge from src->conversion_node, let's add edge from
  // output of conversion_node to the dest node. Since conversion_node
  // has only 1 output, the src_output of conversion_node is 0.
  CHECK_NOTNULL((*g)->AddEdge(conversion_node, 0, dst, e->dst_input()));

  VLOG(1) << "MklToTfConversionPass: Inserting Conversion node on: "
          << src->type_string() << " and " << dst->type_string()
          << " successful.";

  // Remove src->dst edge now.
  (*g)->RemoveEdge(e);
  return Status::OK();
}

Status MklToTfConversionPass::InsertInputConversionNode(
    std::unique_ptr<Graph>* g, Node* n) {
  CHECK_NOTNULL(n);

  // Get the input nodes and edges
  std::vector<const Edge*> edges;
  TF_CHECK_OK(n->input_edges(&edges));
  if (edges.size() != 4) {
    return Status(error::Code::INVALID_ARGUMENT,
                  "MKL Binary Element-wise op should have exactly 2 data"
                  " inputs and 2 metadata inputs");
  }

  // Sanity check: ensure that both inputs are of the expected type, and the
  // same type as input type
  CHECK_EQ(BaseType(edges[0]->src()->output_type(edges[0]->src_output())),
           BaseType(edges[1]->src()->output_type(edges[1]->src_output())));
  CHECK_EQ(BaseType(edges[0]->src()->output_type(edges[0]->src_output())),
           BaseType(n->input_type(0)));

  // Check ordering of edges
  for (uint32 i = 0; i < 4; i++) {
    CHECK_EQ((edges[i]->dst_input() == i), true);
  }

  // Build the conversion node and specify src as input.
  Node* conversion_node = nullptr;

  TF_CHECK_OK(
      NodeBuilder((*g)->NewName("MklInputConversion"), "_MklInputConversion")
          .Input(edges[0]->src(), edges[0]->src_output())
          .Input(edges[1]->src(), edges[1]->src_output())
          .Input(edges[2]->src(), edges[2]->src_output())
          .Input(edges[3]->src(), edges[3]->src_output())
          .Device(n->def().device())
          .Attr("T", n->input_type(0))
          .Finalize(&**g, &conversion_node));

  CHECK_NOTNULL(conversion_node);

  // Change the destination of any control edges to the InputConversion node
  if (edges.size() != n->in_edges().size()) {
    std::vector<const Edge*> edges_to_remove;
    for (const Edge* e : n->in_edges()) {
      if (e->IsControlEdge()) {
        CHECK_NOTNULL((*g)->AddControlEdge(e->src(), conversion_node));
        edges_to_remove.push_back(e);
      }
    }
    for (const Edge* e : edges_to_remove) {
      (*g)->RemoveEdge(e);
    }
  }

  // TODO(Intel-tf) MklInputConversion accepts only NHWC or NCHW, but doesn't
  // seem to be using data_format. This code might be redundant.
  string data_format;
  if (GetNodeAttr(edges[0]->src()->def(), "data_format", &data_format) ==
          Status::OK() &&
      (data_format == ToString(FORMAT_NHWC) ||
       data_format == ToString(FORMAT_NCHW))) {
    conversion_node->AddAttr("data_format", data_format);
  }

  // Get assigned device from destination node and apply it to conversion node.
  // We want conversion node to be on the same device as the destination node.
  conversion_node->set_assigned_device_name(n->assigned_device_name());

  // Set the Mkl op label for this op.
  conversion_node->AddAttr("_kernel",
                           mkl_op_registry::kMklLayoutDependentOpLabel);

  // Now that we have added edges from src->conversion_node, let's add edge from
  // output of conversion_node to the element-wise node.
  CHECK_NOTNULL((*g)->AddEdge(conversion_node, 0, n, edges[0]->dst_input()));
  CHECK_NOTNULL((*g)->AddEdge(conversion_node, 1, n, edges[1]->dst_input()));
  CHECK_NOTNULL((*g)->AddEdge(conversion_node, 2, n, edges[2]->dst_input()));
  CHECK_NOTNULL((*g)->AddEdge(conversion_node, 3, n, edges[3]->dst_input()));

  VLOG(1) << "MklToTfConversionPass - InputConversion: Inserting input "
          << "conversion node on: " << n->type_string() << " successful.";

  // Remove src->dst edge now.
  (*g)->RemoveEdge(edges[0]);
  (*g)->RemoveEdge(edges[1]);
  (*g)->RemoveEdge(edges[2]);
  (*g)->RemoveEdge(edges[3]);

  return Status::OK();
}

bool MklToTfConversionPass::RunPass(std::unique_ptr<Graph>* g) {
  bool result = false;

  CHECK_NOTNULL(g);

  DumpGraph("Before MklToTfConversionPass", &**g);

  // Since we are looking for an Mkl-supported op node immediately
  // followed by a non-Mkl op node, we will just iterate over edge
  // set of the graph.
  // edge set whose source and destination are candidates for
  // inserting conversion node
  std::vector<Edge*> candidate_edges;

  for (const Edge* e : (*g)->edges()) {
    Node* src = e->src();
    Node* dst = e->dst();

    // We skip control edges.
    if (e->IsControlEdge()) {
      continue;
    }

    // We skip adding MklToTf on an edge between X->MklToTf or
    // MklToTf->X, where X is any node.
    if (src->type_string().compare("_MklToTf") == 0 ||
        dst->type_string().compare("_MklToTf") == 0) {
      continue;
    }

    VLOG(1) << "MklToTfConversionPass: InsertConversionNodes: "
            << src->type_string() << " and " << dst->type_string();

    // Let's get source and destination data type.
    // We cannot check datatype on destination node because destination node
    // may not be Mkl node.
    DataType src_datatype;
    DataType dst_datatype;
    bool src_is_mkl_op =
        (GetNodeAttr(src->def(), "T", &src_datatype) == Status::OK() &&
         IsMklSupportedOp(src->type_string(), src_datatype));
    bool dst_is_mkl_op =
        (GetNodeAttr(dst->def(), "T", &dst_datatype) == Status::OK() &&
         IsMklSupportedOp(dst->type_string(), dst_datatype));

    // Check if src with is Mkl-compliant, while dst is not Mkl-compliant.
    if (src_is_mkl_op && !dst_is_mkl_op) {
      VLOG(1) << "MklToTfConversionPass: Scheduled nodes " << src->name()
              << " and " << dst->name() << " for inserting conversion nodes";
      candidate_edges.push_back(const_cast<Edge*>(e));
    }
  }

  // Process all candidate edges and insert conversion nodes on them.
  for (Edge* e : candidate_edges) {
    // Even if we insert conversion node on a single edge, we
    // need to return true.
    string src_name = e->src()->name();
    string dst_name = e->dst()->name();
    if (InsertConversionNodeOnEdge(g, e) == Status::OK()) {
      VLOG(1) << "MklToTfConversionPass: Inserted conversion "
              << "node on edge between " << src_name << " and " << dst_name;
      result = true;
    }
  }

  DumpGraph("After MklToTfConversionPass", &**g);

  //---------------------------------------------------------------------------
  // Check all nodes and add an input-conversion-node if the node is an mkl
  // element-wise node.
  VLOG(1) << "Before running MklToTfConversionPass - InputConversion";

  std::vector<Node*> candidate_nodes;
  std::vector<Node*> order;
  GetReversePostOrder(**g, &order);  // This will give us topological sort.

  for (Node* n : order) {
    // If node is not an op or it does not have a datatype, then skip.
    DataType datatype;
    if (!n->IsOp() || (GetNodeAttr(n->def(), "T", &datatype) != Status::OK())) {
      continue;
    }
    if (IsMklElementWiseOp(n->type_string(), datatype)) {
      // If the input node is an input-conversion op, skip
      Node* input_node = nullptr;
      TF_CHECK_OK(n->input_node(0, &input_node));
      DataType input_datatype;
      if ((GetNodeAttr(n->def(), "T", &input_datatype) == Status::OK()) &&
          (input_node->type_string().compare("_MklInputConversion") == 0)) {
        continue;
      }

      VLOG(1) << "MklToTfConversionPass: InputConversion: Scheduled node "
              << n->name() << " for inserting input conversion node";
      candidate_nodes.push_back(const_cast<Node*>(n));
    }
  }

  // Process all candidate edges and insert conversion nodes on them.
  for (Node* n : candidate_nodes) {
    // Even if we insert conversion node on a single node, we
    // need to return true.
    if (InsertInputConversionNode(g, n) == Status::OK()) {
      VLOG(1) << "MklToTfConversionPass: Inserted conversion "
              << "on node " << n->name();
      result = true;
    }
  }
  DumpGraph("After MklToTfConversionPass - InputConversion", &**g);

  // We need to return true even if we insert one conversion node
  // anywhere in the graph.
  return result;
}

//////////////////////////////////////////////////////////////////////////////
//              Run function for the pass
//////////////////////////////////////////////////////////////////////////////

bool InsertMklToTfConversionNodes(std::unique_ptr<Graph>* g) {
  return MklToTfConversionPass().RunPass(g);
}

Status MklToTfConversionPass::Run(const GraphOptimizationPassOptions& options) {
  if (options.graph == nullptr && options.partition_graphs == nullptr) {
    return Status::OK();
  }
  if (!IsMKLEnabled()) {
    VLOG(2) << "TF-MKL: MKL is not enabled";
    return Status::OK();
  }
  if (NativeFormatEnabled()) {
    VLOG(2)
        << "Running in native format mode, MklToTfConversionPass won't run.";
    return Status::OK();
  }

  auto process_graph = [&](std::unique_ptr<Graph>* g) {
    // Get the ownership of graph
    std::unique_ptr<Graph>* ng = std::move(g);
    RunPass(ng);
    // Return the ownership of graph back
    g->reset(ng->release());
  };

  if (kMklTfConvPassGroup != OptimizationPassRegistry::POST_PARTITIONING) {
    // For any pre-partitioning phase, graph is stored in options.graph.
    process_graph(options.graph);
  } else {
    // For post partitioning phase, graphs are stored in
    // options.partition_graphs.
    for (auto& pg : *options.partition_graphs) {
      process_graph(&pg.second);
    }
  }

  return Status::OK();
}

}  // namespace tensorflow

#endif  // defined(INTEL_MKL) && defined(ENABLE_MKL)
