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

// Note that ints_on_device is only true for single device functions
// (i.e. for cases where Placer is not run).
static Status SetMemoryTypeHelper(
    const gtl::InlinedVector<Node*, 4>& nodes, const DataTypeVector& dtypes,
    bool is_arg, bool weak_flag, bool ints_on_device,
    MemoryTypeVector* memory_types,
    std::vector<AllocatorAttributes>* alloc_attrs) {
  DCHECK_EQ(nodes.size(), dtypes.size());
  if (alloc_attrs != nullptr) {
    alloc_attrs->reserve(nodes.size());
  }
  for (int i = 0; i < nodes.size(); ++i) {
    const auto& node = nodes[i];

    Node* n;
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
    const auto& t = dtypes[i];
    MemoryType mt_from_dtype =
        ints_on_device ? MTypeFromDTypeIntsOnDevice(t) : MTypeFromDType(t);
    if (t == DT_INT32) {
      if (n->def().has_experimental_type()) {
        bool valid_full_type_information = false;
        auto ft = n->def().experimental_type();
        if (ft.type_id() == TFT_PRODUCT) {
          FullTypeId id = GetArgDefaultUnset(ft, output_idx).type_id();
          if (id == TFT_SHAPE_TENSOR) {
            valid_full_type_information = mt_from_dtype == HOST_MEMORY;
          } else if (id == TFT_TENSOR) {
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
  }
  return OkStatus();
}

Status SetMemoryTypeForArgs(const gtl::InlinedVector<Node*, 4>& nodes,
                            const DataTypeVector& dtypes,
                            MemoryTypeVector& memory_types) {
  return SetMemoryTypeHelper(nodes, dtypes, /*is_arg=*/true,
                             /*weak_flag=*/false, /*ints_on_device=*/false,
                             &memory_types, nullptr);
}

// TODO(b/258849883) Delete the `Weak...` versions of these functions once
// everything is working with the version without `Weak`.

Status WeakSetMemoryTypeForArgs(const gtl::InlinedVector<Node*, 4>& nodes,
                                const DataTypeVector& dtypes,
                                MemoryTypeVector& memory_types) {
  return SetMemoryTypeHelper(nodes, dtypes, /*is_arg=*/true,
                             /*weak_flag=*/true, /*ints_on_device=*/false,
                             &memory_types, nullptr);
}

Status SetMemoryTypeForRets(const gtl::InlinedVector<Node*, 4>& nodes,
                            const DataTypeVector& dtypes,
                            MemoryTypeVector& memory_types) {
  return SetMemoryTypeHelper(nodes, dtypes, /*is_arg=*/false,
                             /*weak_flag=*/false, /*ints_on_device=*/false,
                             &memory_types, nullptr);
}

Status WeakSetMemoryTypeForRets(const gtl::InlinedVector<Node*, 4>& nodes,
                                const DataTypeVector& dtypes,
                                MemoryTypeVector& memory_types) {
  return SetMemoryTypeHelper(nodes, dtypes, /*is_arg=*/false,
                             /*weak_flag=*/true, /*ints_on_device=*/false,
                             &memory_types, nullptr);
}

Status SetAllocAttrsForArgs(const gtl::InlinedVector<Node*, 4>& nodes,
                            const DataTypeVector& dtypes,
                            std::vector<AllocatorAttributes>& alloc_attrs) {
  return SetMemoryTypeHelper(nodes, dtypes, /*is_arg=*/true,
                             /*weak_flag=*/false, /*ints_on_device=*/false,
                             nullptr, &alloc_attrs);
}

Status WeakSetAllocAttrsForArgs(const gtl::InlinedVector<Node*, 4>& nodes,
                                const DataTypeVector& dtypes,
                                std::vector<AllocatorAttributes>& alloc_attrs) {
  return SetMemoryTypeHelper(nodes, dtypes, /*is_arg=*/true,
                             /*weak_flag=*/true, /*ints_on_device=*/false,
                             nullptr, &alloc_attrs);
}

Status SetAllocAttrsForRets(const gtl::InlinedVector<Node*, 4>& nodes,
                            const DataTypeVector& dtypes,
                            std::vector<AllocatorAttributes>& alloc_attrs) {
  return SetMemoryTypeHelper(nodes, dtypes, /*is_arg=*/false,
                             /*weak_flag=*/false, /*ints_on_device=*/false,
                             nullptr, &alloc_attrs);
}

Status WeakSetAllocAttrsForRets(const gtl::InlinedVector<Node*, 4>& nodes,
                                const DataTypeVector& dtypes,
                                std::vector<AllocatorAttributes>& alloc_attrs) {
  return SetMemoryTypeHelper(nodes, dtypes, /*is_arg=*/false,
                             /*weak_flag=*/true, /*ints_on_device=*/false,
                             nullptr, &alloc_attrs);
}

Status SingleDeviceSetAllocAttrsForArgs(
    const gtl::InlinedVector<Node*, 4>& nodes, const DataTypeVector& dtypes,
    bool ints_on_device, std::vector<AllocatorAttributes>& alloc_attrs) {
  return SetMemoryTypeHelper(nodes, dtypes, /*is_arg=*/true,
                             /*weak_flag=*/false, ints_on_device, nullptr,
                             &alloc_attrs);
}

Status WeakSingleDeviceSetAllocAttrsForArgs(
    const gtl::InlinedVector<Node*, 4>& nodes, const DataTypeVector& dtypes,
    bool ints_on_device, std::vector<AllocatorAttributes>& alloc_attrs) {
  return SetMemoryTypeHelper(nodes, dtypes, /*is_arg=*/true,
                             /*weak_flag=*/true, ints_on_device, nullptr,
                             &alloc_attrs);
}

Status SingleDeviceSetAllocAttrsForRets(
    const gtl::InlinedVector<Node*, 4>& nodes, const DataTypeVector& dtypes,
    bool ints_on_device, std::vector<AllocatorAttributes>& alloc_attrs) {
  return SetMemoryTypeHelper(nodes, dtypes, /*is_arg=*/false,
                             /*weak_flag=*/false, ints_on_device, nullptr,
                             &alloc_attrs);
}

Status WeakSingleDeviceSetAllocAttrsForRets(
    const gtl::InlinedVector<Node*, 4>& nodes, const DataTypeVector& dtypes,
    bool ints_on_device, std::vector<AllocatorAttributes>& alloc_attrs) {
  return SetMemoryTypeHelper(nodes, dtypes, /*is_arg=*/false,
                             /*weak_flag=*/true, ints_on_device, nullptr,
                             &alloc_attrs);
}

}  // namespace tensorflow::full_type
