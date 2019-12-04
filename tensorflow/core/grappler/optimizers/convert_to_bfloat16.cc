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

#ifdef INTEL_MKL
#ifdef ENABLE_INTEL_MKL_BFLOAT16

#include "tensorflow/core/grappler/optimizers/convert_to_bfloat16.h"

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {

Status BFloat16Converter::InsertCastNode(Graph* g, const Edge* e,
                                         DataType src_dtype,
                                         DataType dst_dtype) {
  string name = strings::StrCat(e->src()->name(), "_", e->src_output(), "_",
                                e->dst_input(), "_cast");
  Node* cast_node = nullptr;
  TF_RETURN_IF_ERROR(
      NodeBuilder(g->NewName(name), "Cast")
          .Input(e->src(), e->src_output())
          .Device(e->src()->def().device())  // We want to get cast node
                                             // on same device as source node.
          .Attr("SrcT", src_dtype)
          .Attr("DstT", dst_dtype)
          .Attr("Truncate", false)
          .Finalize(g, &cast_node));
  cast_node->set_assigned_device_name(e->src()->assigned_device_name());
  const int kCastNodeOutputSlot = 0;
  g->AddEdge(cast_node, kCastNodeOutputSlot, e->dst(), e->dst_input());
  g->RemoveEdge(e);
  return Status::OK();
}

Status BFloat16Converter::ConvertToBFloat16(Graph* g) {
  std::vector<Node*> order;
  GetReversePostOrder(*g, &order);  // This will give us topological sort.
  for (Node* n : order) {
    if (!n->IsOp() || !CanOpRunOnCPUDevice(n)) {
      continue;
    }

    // Change op type to BFLOAT16 if an op is in FP32 and it can support
    // BFLOAT16.
    DataType T = DT_INVALID;
    if (!ShouldSkipOp(n) &&
        (GetNodeAttr(n->def(), "T", &T) == Status::OK()) & (T == DT_FLOAT) &&
        CanOpSupportBFloat16(n)) {
      // Rewrite this op if we want to always rewrite it or if all the inputs of
      // this op are in BFloat16 type.
      string action = "";
      if (RewriteIfAllInputsInBFloat16Op(n)) {
        if (AreAllInputsInBFloat16(n)) {
          action = "changing type to BFLOAT16";
          ChangeDataType(n, DT_BFLOAT16);
        } else {
          action = "skipping BFLOAT16 type conversion";
        }
      } else if (AlwaysRewriteOp(n)) {
        action = "changing type to BFLOAT16";
        ChangeDataType(n, DT_BFLOAT16);
      }
      VLOG(1) << name() << ": " << action << ": op=" << n->name()
              << ", op_type=" << n->type_string();
    }

    // Traverse all input edges and insert cast nodes on an edge if its
    // source type is different than destination type.
    for (const Edge* e : n->in_edges()) {
      // No need to do anything for coontrol edges since current node will not
      // be deleted anyway.
      if (e->IsControlEdge()) continue;
      DataType src_dtype = e->src()->output_type(e->src_output());
      DataType dst_dtype = e->dst()->input_type(e->dst_input());
      // Only insert Cast nodes between float32 and bfloat16 because we dont
      // anticipate bfloat16 usage in non-FP32 models.
      if (src_dtype != dst_dtype &&
          (src_dtype == DT_FLOAT || src_dtype == DT_BFLOAT16) &&
          (dst_dtype == DT_FLOAT || dst_dtype == DT_BFLOAT16)) {
        string cast_type = "";
        if (src_dtype == DT_FLOAT)
          cast_type = "FP32toBF16";
        else
          cast_type = "BF16toFP32";
        VLOG(1) << name() << ": inserting " << cast_type << " Cast on edge "
                << e->src()->name() << ":" << e->src_output() << " and "
                << e->dst()->name() << ":" << e->dst_input();
        TF_RETURN_IF_ERROR(InsertCastNode(g, e, src_dtype, dst_dtype));
      }
    }
  }
  return Status::OK();
}

Status BFloat16Converter::Optimize(Cluster* cluster, const GrapplerItem& item,
                                   GraphDef* output_graphdef) {
  // TODO(nhasabni): what to do with other args?
  Graph output_graph(OpRegistry::Global());
  // TODO(nhasabni): Transformation of GraphDef (using MutableGraphView) is much
  // more efficient than of Graph. See if we can use that approach.
  Status status = ConvertGraphDefToGraph(GraphConstructorOptions(), item.graph,
                                         &output_graph);
  if (status != Status::OK()) {
    LOG(WARNING) << name() << " graph optimizer FAILED: " << status.ToString();
    return errors::Aborted("Nothing to do.");
  }

  status = ConvertToBFloat16(&output_graph);
  if (status != Status::OK()) {
    LOG(WARNING) << name() << " graph optimizer FAILED: " << status.ToString();
    return errors::Aborted("Nothing to do.");
  }

  output_graph.ToGraphDef(output_graphdef);
  return status;
}

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // ENABLE_INTEL_MKL_BFLOAT16
#endif  // INTEL_MKL
