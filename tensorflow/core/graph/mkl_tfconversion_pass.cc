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

#ifdef INTEL_MKL

#include <memory>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"

#include "tensorflow/core/graph/mkl_tfconversion_pass.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {

// This pass inserts Mkl to Tf tensor conversion nodes (represented by C)
// in the graph in between A and B, where A and B match any one
// of the following
// cases:
//  1) A = layer/Op that generates output in Mkl format and,
//     B = layer/Op that does not accept input in Mkl format and,
//     A -> B (there is a direct edge between A and B, then
//     We will insert C such that A->C->B.
//
//  2) A = layer/Op that generates output in Mkl format and,
//     B = NULL (in other words, A is the last layer in the graph), then
//     We will insert C such that A->C->B. (C will be the last layer.)
//
//  Note that case 1 applies to all outputs of A that are input to B.
//  In other words, the conversions will be required for every output
//  of A that is input to B. For example, let us say the output of A
//  is A1, A2, A3, of which A1 and A2 are in Mkl format, but A3 is not
//  in Mkl format, and all of them are input to B. In such case, we will
//  do the conversion for A1 and A2 only. We do not need to do any conversion
//  for A3.
//
// This pass relies on layers registering themselves about their Mkl compliant.
// Mkl compliant layer can accept inputs in Mkl format, and produce output in
// Mkl format. Non-compliant layer accepts inputs and outputs in
// TensorFlow format.
//
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
    return mkl_layer_registry::IsMklLayer(op_name, T);
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
};

// We register MklToTf insertion for phase 1 in post-partition grouping.
// We register this pass after partitioning so that we get a complete
// picture of inputs and outputs of the nodes in the graphs.
const OptimizationPassRegistry::Grouping kMklTfConvPassGroup =
    OptimizationPassRegistry::POST_PARTITIONING;
REGISTER_OPTIMIZATION(kMklTfConvPassGroup, 1, MklToTfConversionPass);

Status MklToTfConversionPass::InsertConversionNodeOnEdge(
    std::unique_ptr<Graph>* g, Edge* e) {
  CHECK_NOTNULL(e);

  Node* src = e->src();
  Node* dst = e->dst();

  CHECK_NOTNULL(src);
  CHECK_NOTNULL(dst);

  Node* conversion_node = nullptr;
  DataType src_datatype = DT_INVALID;
  DataType dst_datatype = DT_INVALID;
  string data_format;

  TF_CHECK_OK(GetNodeAttr(src->def(), "T", &src_datatype));
  TF_CHECK_OK(GetNodeAttr(dst->def(), "T", &dst_datatype));
  if (src_datatype != dst_datatype) {
    string err_msg = "T attribute of " + src->name() + " and " + dst->name() +
                     " do not match. Will not insert" +
                     " MklToTf node in such case.";
    return Status(error::Code::INVALID_ARGUMENT, err_msg.c_str());
  }

  // Lets build the conversion node and specify src as input.
  TF_CHECK_OK(
      NodeBuilder((*g)->NewName("Mkl2Tf"), "MklToTf")
          .Input(src, e->src_output())
          .Input(src, e->src_output() + 1)  // Mkl tensor immediately
                                            // follows Tf tensor.
          .Device(src->def().device())      // We want to get conversion node
                                            // on same device as source node.
          .Attr("T", src_datatype)
          .Finalize(&**g, &conversion_node));

  CHECK_NOTNULL(conversion_node);
  if (GetNodeAttr(src->def(), "data_format", &data_format) == Status::OK()) {
    conversion_node->AddAttr("data_format", data_format);
  }

  // Get assigned device from source node and apply it to conversion node.
  // We want conversion node to be on the same device as the source node.
  conversion_node->set_assigned_device_name(src->assigned_device_name());

  // Set the Mkl layer label for this op.
  conversion_node->AddAttr("_kernel", mkl_layer_registry::kMklLayerLabel);

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

bool MklToTfConversionPass::RunPass(std::unique_ptr<Graph>* g) {
  bool result = false;

  CHECK_NOTNULL(g);

  DumpGraph("Before MklToTfConversionPass", &**g);

  // Since we are looking for mkl-supported op node immediately
  // followed by non-mkl op node, we will just iterate over edge
  // set of the graph.
  // vector to maintain candiadate edges whose source and destination
  // are candidate for inserting conversion node
  std::vector<Edge*> candidate_edges;

  for (const Edge* e : (*g)->edges()) {
    Node* src = e->src();
    Node* dst = e->dst();

    // We skip control edges.
    if (e->IsControlEdge()) {
      continue;
    }

    // We skip adding MklToTf on an edge between X->MklToTf or
    // MklToTf->X, where X is any layer.
    if (src->type_string().compare("MklToTf") == 0 ||
        dst->type_string().compare("MklToTf") == 0) {
      continue;
    }

    VLOG(1) << "MklToTfConversionPass: InsertConversionNodes: "
            << src->type_string() << " and " << dst->type_string();

    // Let's get source and destination data type.
    DataType src_datatype = DT_INVALID;
    if (GetNodeAttr(src->def(), "T", &src_datatype) != Status::OK()) {
      continue;
    }
    // We cannot check datatype on destination node because destination node
    // may not be Mkl node.
    DataType dst_datatype = DT_INVALID;
    GetNodeAttr(dst->def(), "T", &dst_datatype);

    // Check if src with is Mkl-compliant, while dst is not Mkl-compliant.

    if (IsMklSupportedOp(src->type_string(), src_datatype) &&
        !IsMklSupportedOp(dst->type_string(), dst_datatype)) {
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

#endif
