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

#include "tensorflow/core/framework/memory_types.h"

#include <utility>

#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {
// Returns the largest endpoint of anything in the name_map.
int GetTotal(const NameRangeMap& name_map) {
  int total = 0;
  for (const auto& item : name_map) {
    total = std::max(total, item.second.second);
  }
  return total;
}

// Fills memory_types for either input or output, setting everything
// to DEVICE_MEMORY except those args in host_memory_args.  Removes
// elements of host_memory_args that were used.
void MemoryTypesHelper(const NameRangeMap& name_map,
                       std::vector<string>* host_memory_args,
                       MemoryTypeVector* memory_types) {
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

bool IsFunctionCallOp(const string& op_type) {
  return op_type == "SymbolicGradient" || op_type == "PartitionedCall" ||
         op_type == "StatefulPartitionedCall" || op_type == "While" ||
         op_type == "StatelessWhile";
}

}  // namespace

MemoryType MTypeFromDType(const DataType dtype) {
  return (dtype == DT_INT32 || DataTypeAlwaysOnHost(dtype)) ? HOST_MEMORY
                                                            : DEVICE_MEMORY;
}

MemoryType MTypeFromDTypeIntsOnDevice(const DataType dtype) {
  return DataTypeAlwaysOnHost(dtype) ? HOST_MEMORY : DEVICE_MEMORY;
}

Status MemoryTypesForNode(const OpRegistryInterface* op_registry,
                          const DeviceType& device_type, const NodeDef& ndef,
                          MemoryTypeVector* inp_mtypes,
                          MemoryTypeVector* out_mtypes) {
  // Look up the Op registered for this op name.
  const OpDef* op_def;
  TF_RETURN_IF_ERROR(op_registry->LookUpOpDef(ndef.op(), &op_def));

  // Look up the Kernel registered for this node def.
  const KernelDef* kdef = nullptr;
  Status status =
      FindKernelDef(device_type, ndef, &kdef, nullptr /* kernel_class_name */);

  DataTypeVector inp_dtypes;
  DataTypeVector out_dtypes;
  TF_RETURN_IF_ERROR(
      InOutTypesForNode(ndef, *op_def, &inp_dtypes, &out_dtypes));

  inp_mtypes->clear();
  out_mtypes->clear();

  bool has_xla_compile = [&] {
    const auto& it = ndef.attr().find(kXlaMustCompileAttr);
    return it != ndef.attr().end() && it->second.b();
  }();

  bool has_kernel_def = status.ok() && !IsFunctionCallOp(ndef.op());
  auto host_memory_required = [&](const DataType& dt) {
    bool int32_on_device =
        has_kernel_def || device_type.type_string() == "TPU" || has_xla_compile;
    return DataTypeAlwaysOnHost(dt) || (dt == DT_INT32 && !int32_on_device);
  };

  if (has_kernel_def) {
    // Gets the input/output names and their corresponding endpoint ranges.
    NameRangeMap inp_names;
    NameRangeMap out_names;
    TF_RETURN_IF_ERROR(
        NameRangesForNode(ndef, *op_def, &inp_names, &out_names));

    // Now that we know the size, fill with the default 'DEVICE_MEMORY'.
    inp_mtypes->resize(GetTotal(inp_names), DEVICE_MEMORY);
    out_mtypes->resize(GetTotal(out_names), DEVICE_MEMORY);

    // Fills in host memory types based on the kernel def.
    const auto& from_proto = kdef->host_memory_arg();
    std::vector<string> host_memory_args(from_proto.begin(), from_proto.end());
    MemoryTypesHelper(inp_names, &host_memory_args, inp_mtypes);
    MemoryTypesHelper(out_names, &host_memory_args, out_mtypes);
    if (!host_memory_args.empty()) {
      return errors::InvalidArgument(
          "HostMemory args '", absl::StrJoin(host_memory_args, "', '"),
          "' not found in OpDef: ", SummarizeOpDef(*op_def));
    }
  } else {
    // Set all the datatype to DEVICE_MEMORY by default, later on change it to
    // HOST_MEMORY where it is required by the datatype.
    inp_mtypes->resize(inp_dtypes.size(), DEVICE_MEMORY);
    out_mtypes->resize(out_dtypes.size(), DEVICE_MEMORY);
  }
  CHECK_LE(inp_mtypes->size(), inp_dtypes.size());
  CHECK_LE(out_mtypes->size(), out_dtypes.size());

  // Mark e.g. all resource and string types as host memory.
  for (int i = 0; i < inp_mtypes->size(); ++i) {
    if (host_memory_required(inp_dtypes[i])) {
      (*inp_mtypes)[i] = HOST_MEMORY;
    }
  }
  for (int i = 0; i < out_mtypes->size(); ++i) {
    if (host_memory_required(out_dtypes[i])) {
      (*out_mtypes)[i] = HOST_MEMORY;
    }
  }

  std::vector<int32> hostmem_attr;
  if (TryGetNodeAttr(ndef, "_input_hostmem", &hostmem_attr)) {
    for (int32 i : hostmem_attr) {
      if (0 <= i && i < inp_mtypes->size()) {
        (*inp_mtypes)[i] = HOST_MEMORY;
      }
    }
  }
  hostmem_attr.clear();
  if (TryGetNodeAttr(ndef, "_output_hostmem", &hostmem_attr)) {
    for (int32 i : hostmem_attr) {
      if (0 <= i && i < out_mtypes->size()) {
        (*out_mtypes)[i] = HOST_MEMORY;
      }
    }
  }

  return Status::OK();
}

}  // namespace tensorflow
