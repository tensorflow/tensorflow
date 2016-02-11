/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/framework/memory_types.h"

#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

// Fills memory_types for either input or output, setting everything
// to DEVICE_MEMORY except those args in host_memory_args.  Removes
// elements of host_memory_args that were used.
void MemoryTypesHelper(const NameRangeMap& name_map,
                       std::vector<string>* host_memory_args,
                       MemoryTypeVector* memory_types) {
  // Set total to the largest endpoint of anything in the name_map.
  int total = 0;
  for (const auto& item : name_map) {
    total = std::max(total, item.second.second);
  }

  // Now that we know the size, fill with the default 'DEVICE_MEMORY'.
  memory_types->clear();
  memory_types->resize(total, DEVICE_MEMORY);

  // Update args that have been marked as in "HOST_MEMORY".
  size_t keep = 0;
  for (size_t i = 0; i < host_memory_args->size(); ++i) {
    auto iter = name_map.find((*host_memory_args)[i]);
    if (iter != name_map.end()) {
      for (int j = iter->second.first; j < iter->second.second; ++j) {
        (*memory_types)[j] = HOST_MEMORY;
      }
    } else {
      // (*host_memory_args)[i] not found, save it for the next pass.
      if (i > keep) (*host_memory_args)[keep] = (*host_memory_args)[i];
      ++keep;
    }
  }
  host_memory_args->resize(keep);
}

Status MemoryTypesForNode(DeviceType device_type, const NodeDef& ndef,
                          const OpDef& op_def,
                          const NameRangeMap& input_name_map,
                          const NameRangeMap& output_name_map,
                          MemoryTypeVector* input_memory_types,
                          MemoryTypeVector* output_memory_types) {
  Status status;
  const KernelDef* kdef = nullptr;
  TF_RETURN_IF_ERROR(FindKernelDef(device_type, ndef, &kdef));

  if (kdef != nullptr) {
    const auto& from_proto = kdef->host_memory_arg();
    std::vector<string> host_memory_args(from_proto.begin(), from_proto.end());
    MemoryTypesHelper(input_name_map, &host_memory_args, input_memory_types);
    MemoryTypesHelper(output_name_map, &host_memory_args, output_memory_types);
    if (!host_memory_args.empty()) {
      return errors::InvalidArgument(
          "HostMemory args '", str_util::Join(host_memory_args, "', '"),
          "' not found in OpDef: ", SummarizeOpDef(op_def));
    }
  }
  return status;
}

}  // namespace

Status MemoryTypesForNode(const OpRegistryInterface* op_registry,
                          DeviceType device_type, const NodeDef& ndef,
                          MemoryTypeVector* input_memory_types,
                          MemoryTypeVector* output_memory_types) {
  // Look up the Op registered for this op name.
  Status status;
  const OpDef* op_def = op_registry->LookUp(ndef.op(), &status);
  if (op_def == nullptr) return status;

  NameRangeMap inputs;
  NameRangeMap outputs;
  status = NameRangesForNode(ndef, *op_def, &inputs, &outputs);
  if (!status.ok()) return status;

  return MemoryTypesForNode(device_type, ndef, *op_def, inputs, outputs,
                            input_memory_types, output_memory_types);
}

}  // namespace tensorflow
