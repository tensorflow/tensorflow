/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/list_flex_ops.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "json/json.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace flex {

std::string OpListToJSONString(const OpKernelSet& flex_ops) {
  Json::Value result(Json::arrayValue);
  for (const OpKernel& op : flex_ops) {
    Json::Value op_kernel(Json::arrayValue);
    op_kernel.append(Json::Value(op.op_name));
    op_kernel.append(Json::Value(op.kernel_name));
    result.append(op_kernel);
  }
  return Json::FastWriter().write(result);
}

// Find the class name of the op kernel described in the node_def from the pool
// of registered ops. If no kernel class is found, return an empty string.
string FindTensorflowKernelClass(tensorflow::NodeDef* node_def) {
  if (!node_def || node_def->op().empty()) {
    LOG(FATAL) << "Invalid NodeDef";
  }

  const tensorflow::OpRegistrationData* op_reg_data;
  auto status =
      tensorflow::OpRegistry::Global()->LookUp(node_def->op(), &op_reg_data);
  if (!status.ok()) {
    LOG(FATAL) << "Op " << node_def->op() << " not found: " << status;
  }
  AddDefaultsToNodeDef(op_reg_data->op_def, node_def);

  tensorflow::DeviceNameUtils::ParsedName parsed_name;
  if (!tensorflow::DeviceNameUtils::ParseFullName(node_def->device(),
                                                  &parsed_name)) {
    LOG(FATAL) << "Failed to parse device from node_def: "
               << node_def->ShortDebugString();
  }
  string class_name;
  if (!tensorflow::FindKernelDef(
           tensorflow::DeviceType(parsed_name.type.c_str()), *node_def,
           nullptr /* kernel_def */, &class_name)
           .ok()) {
    LOG(FATAL) << "Failed to find kernel class for op: " << node_def->op();
  }
  return class_name;
}

void AddFlexOpsFromModel(const tflite::Model* model, OpKernelSet* flex_ops) {
  // Read flex ops.
  auto* subgraphs = model->subgraphs();
  if (!subgraphs) return;
  for (int subgraph_index = 0; subgraph_index < subgraphs->size();
       ++subgraph_index) {
    const tflite::SubGraph* subgraph = subgraphs->Get(subgraph_index);
    auto* operators = subgraph->operators();
    auto* opcodes = model->operator_codes();
    if (!operators || !opcodes) continue;
    for (int i = 0; i < operators->size(); ++i) {
      const tflite::Operator* op = operators->Get(i);
      const tflite::OperatorCode* opcode = opcodes->Get(op->opcode_index());
      if (tflite::GetBuiltinCode(opcode) != tflite::BuiltinOperator_CUSTOM ||
          !tflite::IsFlexOp(opcode->custom_code()->c_str())) {
        continue;
      }

      // Remove the "Flex" prefix from op name.
      std::string flex_op_name(opcode->custom_code()->c_str());
      std::string tf_op_name =
          flex_op_name.substr(strlen(tflite::kFlexCustomCodePrefix));

      // Read NodeDef and find the op kernel class.
      if (op->custom_options_format() !=
          tflite::CustomOptionsFormat_FLEXBUFFERS) {
        LOG(FATAL) << "Invalid CustomOptionsFormat";
      }
      const flatbuffers::Vector<uint8_t>* custom_opt_bytes =
          op->custom_options();
      if (custom_opt_bytes && custom_opt_bytes->size()) {
        // NOLINTNEXTLINE: It is common to use references with flatbuffer.
        const flexbuffers::Vector& v =
            flexbuffers::GetRoot(custom_opt_bytes->data(),
                                 custom_opt_bytes->size())
                .AsVector();
        std::string nodedef_str = v[1].AsString().str();
        tensorflow::NodeDef nodedef;
        if (nodedef_str.empty() || !nodedef.ParseFromString(nodedef_str)) {
          LOG(FATAL) << "Failed to parse data into a valid NodeDef";
        }
        // Flex delegate only supports running flex ops with CPU.
        *nodedef.mutable_device() = "/CPU:0";
        std::string kernel_class = FindTensorflowKernelClass(&nodedef);
        flex_ops->insert({tf_op_name, kernel_class});
      }
    }
  }
}
}  // namespace flex
}  // namespace tflite
