/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/flex/delegate_data.h"

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/delegates/flex/util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace flex {

namespace {

// Builds a `FunctionDef` proto that contains two nodes:
// The first node is a constant node which has the value of the resource key,
// the second node is a `TfLiteSubgraphExecute` node which will take the
// resource key, and the subgraph's inputs as arguments. The function's return
// value is the return value of `TfLiteSubgraphExecute`.
void BuildFunctionDefProto(const std::string& function_name,
                           const Subgraph& subgraph,
                           tensorflow::FunctionDef& fdef) {
  // Map inputs/outputs to types.
  std::vector<std::string> inputs, outputs;
  inputs.reserve(subgraph.inputs().size());
  outputs.reserve(subgraph.outputs().size());
  for (int i = 0; i < subgraph.inputs().size(); ++i) {
    inputs.push_back(absl::StrCat(
        "args_", i, ": ",
        TfLiteTypeToTfTypeName(subgraph.tensor(subgraph.inputs()[i])->type)));
  }
  for (int i = 0; i < subgraph.outputs().size(); ++i) {
    outputs.push_back(absl::StrCat(
        "res_", i, ": ",
        TfLiteTypeToTfTypeName(subgraph.tensor(subgraph.outputs()[i])->type)));
  }
  std::vector<tensorflow::FunctionDefHelper::Node> nodes;
  // The first node is a constant node containing the string value for the
  // resource name.
  nodes.push_back(tensorflow::FunctionDefHelper::Const<tensorflow::tstring>(
      "SubgraphResourceKey", function_name));
  // Builds the `TfLiteSubgraphExecute` node.
  tensorflow::FunctionDefHelper::Node execute_node;
  execute_node.ret.push_back("InvokeTfLite");
  execute_node.op = "TfLiteSubgraphExecute";
  execute_node.arg.push_back("SubgraphResourceKey:output:0");
  for (int i = 0; i < subgraph.inputs().size(); ++i) {
    execute_node.arg.push_back(absl::StrCat("args_", i));
  }
  nodes.push_back(execute_node);

  std::vector<std::pair<std::string, std::string>> ret_def;
  ret_def.reserve(subgraph.outputs().size());
  for (int i = 0; i < subgraph.outputs().size(); ++i) {
    ret_def.emplace_back(absl::StrCat("res_", i),
                         absl::StrCat("InvokeTfLite:output:", i));
  }
  fdef = tensorflow::FunctionDefHelper::Create(function_name, inputs, outputs,
                                               /*attr_def=*/{}, nodes, ret_def);
  // Insert input/output type attrs.
  tensorflow::AttrValue tin_attrs, tout_attrs;
  for (int i = 0; i < subgraph.inputs().size(); ++i) {
    TF_DataType dtype = tflite::flex::GetTensorFlowDataType(
        subgraph.tensor(subgraph.inputs()[i])->type);
    tin_attrs.mutable_list()->add_type(tensorflow::DataType(dtype));
  }
  for (int i = 0; i < subgraph.outputs().size(); ++i) {
    TF_DataType dtype = tflite::flex::GetTensorFlowDataType(
        subgraph.tensor(subgraph.outputs()[i])->type);
    tout_attrs.mutable_list()->add_type(tensorflow::DataType(dtype));
  }
  fdef.mutable_node_def(1)->mutable_attr()->insert({"Tin", tin_attrs});
  fdef.mutable_node_def(1)->mutable_attr()->insert({"Tout", tout_attrs});
}

// Returns a list of subgraph names which have associated function attributes.
absl::Status GetSubgraphNamesForFunctionExecution(
    const std::vector<std::unique_ptr<Subgraph>>& subgraphs,
    std::set<std::string>* result) {
  tensorflow::NodeDef node_def;
  for (const auto& subgraph : subgraphs) {
    for (const auto& node_and_reg : subgraph->nodes_and_registration()) {
      if (node_and_reg.second.builtin_code != tflite::BuiltinOperator_CUSTOM) {
        // If this isn't a custom op, skip.
        continue;
      }
      const std::string custom_name = node_and_reg.second.custom_name;
      if (custom_name.substr(0, strlen(tflite::kFlexCustomCodePrefix)) !=
          tflite::kFlexCustomCodePrefix) {
        // Skip if this is not a flex op.
        continue;
      }
      // The flexbuffer contains a vector where the first elements is the
      // op name and the second is a serialized NodeDef.
      const flexbuffers::Vector& v =
          flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(
                                   node_and_reg.first.custom_initial_data),
                               node_and_reg.first.custom_initial_data_size)
              .AsVector();
      // TODO(b/181352924): Use proto arena if we see performance regression.
      if (!node_def.ParseFromString(v[1].AsString().str())) {
        return absl::Status(absl::StatusCode::kInternal,
                            "could not parse NodeDef");
      }
      // Loop through all the attributes in this node to check if it has
      // function attribute.
      for (const auto& attr : node_def.attr()) {
        if (attr.second.has_func()) {
          result->insert(attr.second.func().name());
        }
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status RegisterFunctionDefForSubgraphs(
    Subgraph& main_subgraph,
    const std::function<absl::Status(
        const std::vector<std::unique_ptr<Subgraph>>&, std::set<std::string>*)>&
        select_subgraphs_to_register,
    tensorflow::ResourceMgr* resource_mgr,
    tensorflow::EagerContext* eager_context, TfLiteDelegate* flex_delegate) {
  std::vector<std::unique_ptr<Subgraph>>* subgraphs =
      main_subgraph.GetSubgraphs();
  if (!subgraphs) {
    // If there are no subgraphs associated with the main subgraph, we will
    // return ok status because no FunctionDef needs to be registered.
    return absl::OkStatus();
  }
  std::set<std::string> function_subgraphs;
  TF_RETURN_IF_ERROR(
      select_subgraphs_to_register(*subgraphs, &function_subgraphs));
  for (int i = 0; i < subgraphs->size(); ++i) {
    if (subgraphs->at(i)->GetName() == "main") {
      continue;
    }
    const std::string subgraph_name = subgraphs->at(i)->GetName();
    if (!function_subgraphs.count(subgraph_name)) {
      continue;
    }
    // This is to ensure that we only register FunctionDefs for subgraphs that
    // are used by TF ops to invoke functions.
    auto* subgraph_resource =
        new TFLiteSubgraphResource(*(subgraphs->at(i)), flex_delegate);
    TF_RETURN_IF_ERROR(resource_mgr->Create<TFLiteSubgraphResource>(
        "flex", subgraph_name, subgraph_resource));
    tensorflow::FunctionDef fdef;
    BuildFunctionDefProto(subgraph_name, *(subgraphs->at(i)), fdef);
    TF_RETURN_IF_ERROR(eager_context->AddFunctionDef(fdef));
  }
  return absl::OkStatus();
}

DelegateData::DelegateData() {}

DelegateData::~DelegateData() {
  if (eager_context_) {
    // Notify the eager context to clean up the resource being held before
    // destructing the `DelegateData`.
    eager_context_->HostCPU()->ClearResourceMgr();
    eager_context_->Unref();
  }
}

absl::Status DelegateData::Prepare(
    const tensorflow::SessionOptions& session_options, Subgraph* main_subgraph,
    TfLiteDelegate* flex_delegate) {
  if (eager_context_) {
    return absl::Status();
  }
  if (flex_delegate == nullptr && main_subgraph != nullptr) {
    return absl::Status(
        absl::StatusCode::kFailedPrecondition,
        "flex_delegate must be non-null when main_subgraph is provided.");
  }

  std::vector<std::unique_ptr<tensorflow::Device>> devices;

  TF_RETURN_IF_ERROR(tensorflow::DeviceFactory::AddDevices(
      session_options, "/job:localhost/replica:0/task:0", &devices));

  auto device_mgr =
      std::make_unique<tensorflow::StaticDeviceMgr>(std::move(devices));
  // Note that Rendezvous is ref-counted so it will be automatically deleted.
  auto rendezvous = tsl::core::RefCountPtr<tensorflow::IntraProcessRendezvous>(
      new tensorflow::IntraProcessRendezvous(device_mgr.get()));
  eager_context_ = new tensorflow::EagerContext(
      session_options,
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
      /*async=*/false, device_mgr.release(), /*device_mgr_owned*/ true,
      std::move(rendezvous), nullptr);

  if (main_subgraph) {
    TF_RETURN_IF_ERROR(RegisterFunctionDefForSubgraphs(
        *main_subgraph, GetSubgraphNamesForFunctionExecution,
        eager_context_->HostCPU()->resource_manager(), eager_context_,
        flex_delegate));
  }
  return absl::Status();
}

}  // namespace flex
}  // namespace tflite
