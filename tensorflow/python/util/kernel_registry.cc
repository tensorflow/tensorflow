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
#include "tensorflow/python/util/kernel_registry.h"

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace swig {

string TryFindKernelClass(const string& serialized_node_def) {
  tensorflow::NodeDef node_def;
  if (!node_def.ParseFromString(serialized_node_def)) {
    LOG(WARNING) << "Error parsing node_def";
    return "";
  }

  const tensorflow::OpRegistrationData* op_reg_data;
  auto status =
      tensorflow::OpRegistry::Global()->LookUp(node_def.op(), &op_reg_data);
  if (!status.ok()) {
    LOG(WARNING) << "Op " << node_def.op() << " not found: " << status;
    return "";
  }
  AddDefaultsToNodeDef(op_reg_data->op_def, &node_def);

  tensorflow::DeviceNameUtils::ParsedName parsed_name;
  if (!tensorflow::DeviceNameUtils::ParseFullName(node_def.device(),
                                                  &parsed_name)) {
    LOG(WARNING) << "Failed to parse device from node_def: "
                 << node_def.ShortDebugString();
    return "";
  }
  string class_name = "";
  status = tensorflow::FindKernelDef(
      tensorflow::DeviceType(parsed_name.type.c_str()), node_def,
      nullptr /* kernel_def */, &class_name);
  if (!status.ok()) {
    LOG(WARNING) << "Op [" << node_def.op() << "]: " << status;
  }
  return class_name;
}

}  // namespace swig
}  // namespace tensorflow
