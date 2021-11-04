/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/ir/importexport/tests/roundtrip/roundtrip.h"

#include "absl/strings/str_cat.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/ir/importexport/export.h"
#include "tensorflow/core/ir/importexport/import.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/protobuf.h"

using mlir::MLIRContext;

namespace tensorflow {

// Applies various normalization to a NodeDef to make it possible to perform
// textual comparison (for example splat constant are detected, NaN are removed,
// control input are alphabetically sorted, etc).
void NormalizeNode(NodeDef* node) {
  for (auto& named_attr : (*node->mutable_attr())) {
    AttrValue& attr_val = named_attr.second;
    if (attr_val.has_tensor()) {
      auto* tensor = attr_val.mutable_tensor();
      switch (tensor->dtype()) {
        // There is no compression or canonicalization for DT_STRING, let's
        // just strip it entirely for now so it is ignored from the comparison.
        case DT_STRING: {
          const TensorShape shape(tensor->tensor_shape());
          if (!tensor->tensor_content().empty()) {
            tensor->mutable_tensor_content()->clear();
          } else {
            tensor->mutable_string_val()->Clear();
          }
          break;
        }
        case DT_FLOAT:
          tensor::CompressTensorProtoInPlace(1, 1.0, tensor);
          for (float& val : *tensor->mutable_float_val())
            if (std::isnan(val)) val = -42.;
          break;
        case DT_DOUBLE:
          tensor::CompressTensorProtoInPlace(1, 1.0, tensor);
          for (double& val : *tensor->mutable_double_val())
            if (std::isnan(val)) val = -42.;
          break;
        case DT_COMPLEX64:
          tensor::CompressTensorProtoInPlace(1, 1.0, tensor);
          for (float& val : *tensor->mutable_scomplex_val())
            if (std::isnan(val)) val = -42.;
          break;
        case DT_COMPLEX128:
          tensor::CompressTensorProtoInPlace(1, 1.0, tensor);
          for (double& val : *tensor->mutable_dcomplex_val())
            if (std::isnan(val)) val = -42.;
          break;
        case DT_VARIANT: {
          Tensor t;
          if (t.FromProto(*tensor)) t.AsProtoField(tensor);
          break;
        }
        default:
          tensor::CompressTensorProtoInPlace(1, 1.0, tensor);
      }
    }
  }
  // Sort control inputs alphabetically.
  for (auto it = node->mutable_input()->begin(),
            end = node->mutable_input()->end();
       it != end; ++it) {
    if (it->empty() || it->front() != '^') continue;
    std::sort(it, end);
  }

  const OpDef* op_def = nullptr;
  (void)tensorflow::OpRegistry::Global()->LookUpOpDef(node->op(), &op_def);
  if (op_def) StripDefaultsFromNodeDef(*op_def, node);
  node->clear_experimental_type();
}

void NormalizeTensorData(GraphDef& graphdef) {
  FunctionDefLibrary* library = graphdef.mutable_library();
  llvm::sort(*library->mutable_function(),
             [](FunctionDef& lhs, FunctionDef& rhs) {
               return lhs.signature().name() < rhs.signature().name();
             });

  for (int i = 0; i < graphdef.node_size(); ++i)
    NormalizeNode(graphdef.mutable_node(i));
  llvm::sort(*graphdef.mutable_node(),
             [](const NodeDef& lhs, const NodeDef& rhs) {
               return lhs.name() < rhs.name();
             });
  for (int func_id = 0; func_id < library->function_size(); ++func_id) {
    FunctionDef* func = library->mutable_function(func_id);
    llvm::sort(*func->mutable_node_def(), [](NodeDef& lhs, NodeDef& rhs) {
      return lhs.name() < rhs.name();
    });
    for (int node_id = 0; node_id < func->node_def_size(); ++node_id) {
      NodeDef* node = func->mutable_node_def(node_id);
      NormalizeNode(node);
    }
    for (const auto& it : *func->mutable_ret()) {
      func->mutable_ret()->at(it.first) = it.second;
      // Eliminate empty arg_attr entries.
      llvm::SmallVector<int> to_erase;
      for (auto& arg_attr : *func->mutable_arg_attr()) {
        if (arg_attr.second.attr().empty()) {
          to_erase.push_back(arg_attr.first);
        }
      }
      for (int idx : to_erase) func->mutable_arg_attr()->erase(idx);
    }
  }
}

Status TestRoundTrip(GraphDef& graphdef) {
  MLIRContext context;
  GraphDebugInfo debug_info;
  auto errorOrModule =
      mlir::tfg::ImportGraphDefToMlir(&context, debug_info, graphdef);
  if (!errorOrModule.ok()) {
    LOG(ERROR) << errorOrModule.status();
    llvm::errs()
        << "\n\n=========\n=========\n=========\n=========\n=========\n"
        << graphdef.DebugString()
        << "=========\n=========\n=========\n=========\n";
    return errorOrModule.status();
  }
  GraphDef new_graph;
  auto module = errorOrModule.ValueOrDie().get();
  Status status = tensorflow::ExportMlirToGraphdef(module, &new_graph);
  if (!status.ok()) {
    LOG(ERROR) << "Error exporting MLIR module to GraphDef: " << status;
    return status;
  }
  GraphDef original_graph;
  {
    GraphConstructorOptions options;
    options.allow_internal_ops = true;
    options.add_default_attributes = true;
    Graph graph(OpRegistry::Global());
    GraphDef preprocessed_graphdef(graphdef);
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
        options, std::move(preprocessed_graphdef), &graph));
    graph.ToGraphDef(&original_graph);
  }
  NormalizeTensorData(new_graph);
  NormalizeTensorData(original_graph);
  if (!tensorflow::protobuf::util::MessageDifferencer::Equivalent(
          original_graph, new_graph)) {
    LOG(ERROR) << "GraphDef didn't Roundtrip:";
    llvm::errs()
        << "\n=========\n\n"
        << module
        << "\n\n=========\n=========\n=========\n=========\n=========\n"
        << graphdef.DebugString()
        << "=========\n=========\n=========\n=========\n";
    return errors::InvalidArgument("GraphDef didn't roundtrip");
  }
  return {};
}

}  // namespace tensorflow
