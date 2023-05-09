/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/arg_ret_placement.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/full_type_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow::full_type {

MemoryType MemoryTypeFromFullTypeId(FullTypeId id) {
  if (id == TFT_SHAPE_TENSOR) {
    return HOST_MEMORY;
  }
  return DEVICE_MEMORY;
}

bool LogMemoryTypeMismatch(bool use_host_memory, const FullTypeDef& ft) {
  FullTypeId id = ft.type_id();
  if (id == TFT_PRODUCT) {
    LOG(ERROR) << "Unexpected full type information for tensor, which should "
                  "not start with TFT_PRODUCT\n"
               << ft.DebugString();
    return false;
  }
  MemoryType mt_from_ft = MemoryTypeFromFullTypeId(id);
  if (use_host_memory != (mt_from_ft == HOST_MEMORY)) {
    VLOG(1) << "use_host_memory=" << use_host_memory
            << "but full type information is\n"
            << ft.DebugString();
    return false;
  }
  return true;
}

Status CheckMemoryType(bool use_host_memory, const FullTypeDef& ft) {
  FullTypeId id = ft.type_id();
  MemoryType mt_from_ft = MemoryTypeFromFullTypeId(id);
  if (id == TFT_PRODUCT) {
    return errors::Internal(
        "Unexpected full type information for tensor, which should not start "
        "with TFT_PRODUCT\n",
        ft.DebugString());
  }
  if (use_host_memory != (mt_from_ft == HOST_MEMORY)) {
    return errors::Internal("use_host_memory=", use_host_memory,
                            " but full type information is\n",
                            ft.DebugString());
  }
  return OkStatus();
}

// Note that ints_on_device is only true for single device functions
// (i.e. for cases where Placer is not run).
static Status SetMemoryTypeForNode(
    const Node* node, const DataType dtype, bool is_arg, bool weak_flag,
    bool ints_on_device, MemoryTypeVector* memory_types,
    std::vector<AllocatorAttributes>* alloc_attrs) {
  const Node* n;
  int output_idx;
  if (is_arg) {
    DCHECK(node->op_def().name() == "_Arg" ||
           node->op_def().name() == "_DeviceArg");
    output_idx = 0;
    n = node;
  } else {
    // "_Retval" nodes are sinks, they do not have an output (to any other
    // node in the subgraph for the function that they are in) so they do
    // not have any useful full type information. Instead get the full type
    // of the input to the _Rval op.
    DCHECK(node->op_def().name() == "_Retval" ||
           node->op_def().name() == "_DeviceRetval");
    const Edge* edge;
    TF_RETURN_IF_ERROR(node->input_edge(0, &edge));
    n = edge->src();
    output_idx = edge->src_output();
  }
  MemoryType mt_from_dtype = ints_on_device ? MTypeFromDTypeIntsOnDevice(dtype)
                                            : MTypeFromDType(dtype);
  if (dtype == DT_INT32) {
    if (n->def().has_experimental_type()) {
      bool valid_full_type_information = false;
      auto ft = n->def().experimental_type();
      if (ft.type_id() == TFT_PRODUCT) {
        FullTypeId id = GetArgDefaultUnset(ft, output_idx).type_id();
        MemoryType mt_from_ft = MemoryTypeFromFullTypeId(id);
        if ((id == TFT_TENSOR) || (id == TFT_SHAPE_TENSOR)) {
          valid_full_type_information = mt_from_dtype == mt_from_ft;
        } else if (id == TFT_UNSET) {
          valid_full_type_information = mt_from_dtype != HOST_MEMORY;
        }
      }
      if (!valid_full_type_information) {
        if (weak_flag) {
          VLOG(1) << "node=" << n->name() << " (op=" << n->def().op()
                  << ") has an int32 output with unexpected full type "
                  << "information with ints_on_device=" << ints_on_device
                  << "\n"
                  << n->def().DebugString();
        } else {
          return errors::Internal(
              "node=", n->name(), " (op=", n->def().op(),
              ") has an int32 output with unexpected full type information ",
              "with ints_on_device=", ints_on_device, "\n",
              n->def().DebugString());
        }
      }
    } else if (mt_from_dtype == HOST_MEMORY) {
      if (weak_flag) {
        VLOG(1) << "node=" << n->name() << " (op=" << n->def().op()
                << ") has a HOST_MEMORY int32 output but does not have "
                << "(TFT_SHAPE_TENSOR) full type information.";
      } else {
        return errors::Internal(
            "node=", n->name(), " (op=", n->def().op(),
            ")  has a HOST_MEMORY int32 output but does not have "
            "(TFT_SHAPE_TENSOR) full type information.");
      }
    }
  }
  if (memory_types != nullptr) {
    memory_types->push_back(mt_from_dtype);
  }
  if (alloc_attrs != nullptr) {
    AllocatorAttributes aa;
    aa.set_on_host(mt_from_dtype == HOST_MEMORY);
    alloc_attrs->push_back(aa);
  }
  return OkStatus();
}

// This helper function takes a list of nodes.
static Status SetMemoryTypeHelper(
    const gtl::InlinedVector<Node*, 4>& nodes, const DataTypeVector& dtypes,
    bool is_arg, bool weak_flag, MemoryTypeVector* memory_types,
    std::vector<AllocatorAttributes>* alloc_attrs) {
  DCHECK_EQ(nodes.size(), dtypes.size());
  if (alloc_attrs != nullptr) {
    alloc_attrs->reserve(nodes.size());
  }
  for (int i = 0; i < nodes.size(); ++i) {
    TF_RETURN_IF_ERROR(SetMemoryTypeForNode(nodes[i], dtypes[i], is_arg,
                                            weak_flag, /*ints_on_device=*/false,
                                            memory_types, alloc_attrs));
  }
  return OkStatus();
}

// This helper function takes a list of pairs that contain an arg node.
// Note that ints_on_device is only true for single device functions
// (i.e. for cases where Placer is not run). The DataType specified by the "T"
// attr of input nodes is used.
static Status SetMemoryTypeHelper(
    const std::vector<std::pair<Node*, FunctionArgIndex>> arg_nodes,
    bool weak_flag, bool ints_on_device,
    std::vector<AllocatorAttributes>* alloc_attrs) {
  DCHECK(alloc_attrs != nullptr);
  alloc_attrs->reserve(arg_nodes.size());
  for (const auto& arg : arg_nodes) {
    const AttrValue* attr_value = arg.first->attrs().Find("T");
    if (attr_value == nullptr) {
      return errors::Internal("Arg node missing T attribute");
    }
    DataType dtype = attr_value->type();
    TF_RETURN_IF_ERROR(SetMemoryTypeForNode(
        arg.first, dtype, /*is_arg=*/true, weak_flag, ints_on_device,
        /*memory_types=*/nullptr, alloc_attrs));
  }
  return OkStatus();
}

// This helper function takes a list of pairs that contain a ret node.
// Note that ints_on_device is only true for single device functions
// (i.e. for cases where Placer is not run). The DataType specified by the "T"
// attr of input nodes is used.
static Status SetMemoryTypeHelper(
    const std::vector<std::pair<Node*, int>> ret_nodes, bool weak_flag,
    bool ints_on_device, std::vector<AllocatorAttributes>* alloc_attrs) {
  DCHECK(alloc_attrs != nullptr);
  alloc_attrs->reserve(ret_nodes.size());
  for (const auto& ret : ret_nodes) {
    const AttrValue* attr_value = ret.first->attrs().Find("T");
    if (attr_value == nullptr) {
      return errors::Internal("Ret node missing T attribute");
    }
    DataType dtype = attr_value->type();
    TF_RETURN_IF_ERROR(SetMemoryTypeForNode(
        ret.first, dtype, /*is_arg=*/false, weak_flag, ints_on_device,
        /*memory_types=*/nullptr, alloc_attrs));
  }
  return OkStatus();
}

Status SetMemoryTypeForArgs(const gtl::InlinedVector<Node*, 4>& nodes,
                            const DataTypeVector& dtypes,
                            MemoryTypeVector& memory_types) {
  return SetMemoryTypeHelper(nodes, dtypes, /*is_arg=*/true,
                             /*weak_flag=*/false, &memory_types, nullptr);
}

// TODO(b/258849883) Delete the `Weak...` versions of these functions once
// everything is working with the version without `Weak`.

Status WeakSetMemoryTypeForArgs(const gtl::InlinedVector<Node*, 4>& nodes,
                                const DataTypeVector& dtypes,
                                MemoryTypeVector& memory_types) {
  return SetMemoryTypeHelper(nodes, dtypes, /*is_arg=*/true,
                             /*weak_flag=*/true, &memory_types, nullptr);
}

Status SetMemoryTypeForRets(const gtl::InlinedVector<Node*, 4>& nodes,
                            const DataTypeVector& dtypes,
                            MemoryTypeVector& memory_types) {
  return SetMemoryTypeHelper(nodes, dtypes, /*is_arg=*/false,
                             /*weak_flag=*/false, &memory_types, nullptr);
}

Status WeakSetMemoryTypeForRets(const gtl::InlinedVector<Node*, 4>& nodes,
                                const DataTypeVector& dtypes,
                                MemoryTypeVector& memory_types) {
  return SetMemoryTypeHelper(nodes, dtypes, /*is_arg=*/false,
                             /*weak_flag=*/true, &memory_types, nullptr);
}

Status SetAllocAttrsForArgs(const gtl::InlinedVector<Node*, 4>& nodes,
                            const DataTypeVector& dtypes,
                            std::vector<AllocatorAttributes>& alloc_attrs) {
  return SetMemoryTypeHelper(nodes, dtypes, /*is_arg=*/true,
                             /*weak_flag=*/false, nullptr, &alloc_attrs);
}

Status WeakSetAllocAttrsForArgs(const gtl::InlinedVector<Node*, 4>& nodes,
                                const DataTypeVector& dtypes,
                                std::vector<AllocatorAttributes>& alloc_attrs) {
  return SetMemoryTypeHelper(nodes, dtypes, /*is_arg=*/true,
                             /*weak_flag=*/true, nullptr, &alloc_attrs);
}

Status SetAllocAttrsForRets(const gtl::InlinedVector<Node*, 4>& nodes,
                            const DataTypeVector& dtypes,
                            std::vector<AllocatorAttributes>& alloc_attrs) {
  return SetMemoryTypeHelper(nodes, dtypes, /*is_arg=*/false,
                             /*weak_flag=*/false, nullptr, &alloc_attrs);
}

Status WeakSetAllocAttrsForRets(const gtl::InlinedVector<Node*, 4>& nodes,
                                const DataTypeVector& dtypes,
                                std::vector<AllocatorAttributes>& alloc_attrs) {
  return SetMemoryTypeHelper(nodes, dtypes, /*is_arg=*/false,
                             /*weak_flag=*/true, nullptr, &alloc_attrs);
}

Status SingleDeviceSetAllocAttrsForArgs(
    std::vector<std::pair<Node*, FunctionArgIndex>> arg_nodes,
    bool ints_on_device, std::vector<AllocatorAttributes>& alloc_attrs) {
  return SetMemoryTypeHelper(arg_nodes, /*weak_flag=*/false, ints_on_device,
                             &alloc_attrs);
}

Status WeakSingleDeviceSetAllocAttrsForArgs(
    std::vector<std::pair<Node*, FunctionArgIndex>> arg_nodes,
    bool ints_on_device, std::vector<AllocatorAttributes>& alloc_attrs) {
  return SetMemoryTypeHelper(arg_nodes, /*weak_flag=*/true, ints_on_device,
                             &alloc_attrs);
}

Status SingleDeviceSetAllocAttrsForRets(
    const std::vector<std::pair<Node*, int>> ret_nodes, bool ints_on_device,
    std::vector<AllocatorAttributes>& alloc_attrs) {
  return SetMemoryTypeHelper(ret_nodes, /*weak_flag=*/false, ints_on_device,
                             &alloc_attrs);
}

Status WeakSingleDeviceSetAllocAttrsForRets(
    const std::vector<std::pair<Node*, int>> ret_nodes, bool ints_on_device,
    std::vector<AllocatorAttributes>& alloc_attrs) {
  return SetMemoryTypeHelper(ret_nodes, /*weak_flag=*/true, ints_on_device,
                             &alloc_attrs);
}

}  // namespace tensorflow::full_type
