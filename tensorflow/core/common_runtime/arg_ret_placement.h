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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_ARG_RET_PLACEMENT_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_ARG_RET_PLACEMENT_H_

#include <utility>
#include <vector>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow::full_type {

// Set the contents of memory_types for args (inputs to functions, "_Arg" ops)
// based on dtype. Raises an error if an int32 arg does not have
// expected full_type information. If an error raised about bad full
// time information causes a breakage, changing `SetMemoryTypeForArgs` to
// `WeakSetMemoryTypeForArgs` is a possible work around.
Status SetMemoryTypeForArgs(const gtl::InlinedVector<Node*, 4>& nodes,
                            const DataTypeVector& dtypes,
                            MemoryTypeVector& memory_types);

// TODO(b/258849883) Delete the `Weak...` versions of these functions once
// everything is working with the version without `Weak`.

// Set the contents of memory_types for args (inputs to functions, "_Arg" ops)
// based on dtype. Logging of warnings if an int32 arg does not have
// expected full_type information can be enabled.
Status WeakSetMemoryTypeForArgs(const gtl::InlinedVector<Node*, 4>& nodes,
                                const DataTypeVector& dtypes,
                                MemoryTypeVector& memory_types);

// Set the contents of memory_types for rets (outputs from functions, "_Retval"
// ops) based on dtype. Raises an error if an int32 ret does not have
// expected full_type information (i.e. if the source of the input to the ret
// does not have expected full type information). If an error raised about bad
// full time information causes a breakage, changing `SetMemoryTypeForRets` to
// `WeakSetMemoryTypeForRets` is a possible work around.
Status SetMemoryTypeForRets(const gtl::InlinedVector<Node*, 4>& nodes,
                            const DataTypeVector& dtypes,
                            MemoryTypeVector& memory_types);

// Set the contents of memory_types for rets (outputs from functions, "_Retval"
// ops) based on dtype. Logging of warnings if an int32 ret does not have
// expected full_type information (i.e. if the source of the input to the ret
// does not have expected full type information) can be enabled.
Status WeakSetMemoryTypeForRets(const gtl::InlinedVector<Node*, 4>& nodes,
                                const DataTypeVector& dtypes,
                                MemoryTypeVector& memory_types);

// Set the contents of alloc_attrs for args (inputs to functions, "_Arg" ops)
// based on dtype. Raises an error if an int32 arg does not have
// expected full_type information. If an error raised about bad full
// time information causes a breakage, changing `SetAllocAttrsForArgs` to
// `WeakSetAllocAttrsForArgs` is a possible work around.
Status SetAllocAttrsForArgs(const gtl::InlinedVector<Node*, 4>& nodes,
                            const DataTypeVector& dtypes,
                            std::vector<AllocatorAttributes>& alloc_attrs);

// Set the contents of alloc_attrs for args (inputs to functions, "_Arg" ops)
// based on dtype. Logging of warnings if an int32 arg does not have
// expected full_type information can be enabled.
Status WeakSetAllocAttrsForArgs(const gtl::InlinedVector<Node*, 4>& nodes,
                                const DataTypeVector& dtypes,
                                std::vector<AllocatorAttributes>& alloc_attrs);

// Set the contents of alloc_attrs for rets (outputs from functions, "_Retval"
// ops) based on dtype. Raises an error if an int32 ret does not have
// expected full_type information (i.e. if the source of the input to the ret
// does not have expected full type information). If an error raised about bad
// full time information causes a breakage, changing `SetAllocAttrsForRets` to
// `WeakSetAllocAttrsForRets` is a possible work around.
Status SetAllocAttrsForRets(const gtl::InlinedVector<Node*, 4>& nodes,
                            const DataTypeVector& dtypes,
                            std::vector<AllocatorAttributes>& alloc_attrs);

// Set the contents of alloc_attrs for rets (outputs from functions, "_Retval"
// ops) based on dtype. Logging of warnings if an int32 ret does not have
// expected full_type information (i.e. if the source of the input to the ret
// does not have expected full type information) can be enabled.
Status WeakSetAllocAttrsForRets(const gtl::InlinedVector<Node*, 4>& nodes,
                                const DataTypeVector& dtypes,
                                std::vector<AllocatorAttributes>& alloc_attrs);

// Set the contents of alloc_attrs for args (inputs to functions, "_Arg" ops)
// for a single device funtion based on dtype. Raises an error if an int32 arg
// does not have expected full_type information. If an error raised about bad
// full time information causes a breakage, changing
// `SingleDeviceSetAllocAttrsForArgs` to `WeakSingleDeviceSetAllocAttrsForArgs`
// is a possible work around. The DataType specified by the "T" attr of input
// nodes is used.
Status SingleDeviceSetAllocAttrsForArgs(
    std::vector<std::pair<Node*, FunctionArgIndex>> arg_nodes,
    bool ints_on_device, std::vector<AllocatorAttributes>& alloc_attrs);

// Set the contents of alloc_attrs for args (inputs to functions, "_Arg" ops)
// for a single device based on dtype. Logging of warnings if an int32 arg does
// not have expected full_type information can be enabled. The DataType
// specified by the "T" attr of input nodes is used.
Status WeakSingleDeviceSetAllocAttrsForArgs(
    std::vector<std::pair<Node*, FunctionArgIndex>> arg_nodes,
    bool ints_on_device, std::vector<AllocatorAttributes>& alloc_attrs);

// Set the contents of alloc_attrs for rets (outputs from functions, "_Retval"
// ops) for a single device based on dtype. Raises an error if an int32 ret does
// not have expected full_type information (i.e. if the source of the input to
// the ret does not have expected full type information). If an error raised
// about bad full time information causes a breakage, changing
// `SingleDeviceSetAllocAttrsForRets` to `WeakSingleDeviceSetAllocAttrsForRets`
// is a possible work around. The DataType specified by the "T" attr of input
// nodes is used.
Status SingleDeviceSetAllocAttrsForRets(
    std::vector<std::pair<Node*, int>> ret_nodes, bool ints_on_device,
    std::vector<AllocatorAttributes>& alloc_attrs);

// Set the contents of alloc_attrs for rets (outputs from functions, "_Retval"
// ops) for a single device based on dtype. Logging of warnings if an int32 ret
// does not have expected full_type information (i.e. if the source of the input
// to the ret does not have expected full type information) can be enabled. The
// DataType specified by the "T" attr of input nodes is used.
Status WeakSingleDeviceSetAllocAttrsForRets(
    std::vector<std::pair<Node*, int>> ret_nodes, bool ints_on_device,
    std::vector<AllocatorAttributes>& alloc_attrs);

// Given a FullTypeId, return the corresponding MemoryTypes (i.e. return
// HOST_MEMORY for TFT_SHAPE_TENSOR, DEVICE_MEMORY othersize).
MemoryType MemoryTypeFromFullTypeId(FullTypeId id);

// Check that use_host_memory is true iff FT has type_id TFT_SHAPE_TENSOR
// and logging of a warning if not can be enabled. Returns true if check passes.
// Note the FT is expected to be the full type information for a tensor, not for
// the whole ouput of an op, i.e. it should not have an outer TFT_PRODUCT.
bool LogMemoryTypeMismatch(bool use_host_memory, const FullTypeDef& ft);

// Check that use_host_memory is true iff FT has type_id TFT_SHAPE_TENSOR
// and raise an error if not. Note the FT is expected to be the full type
// information for a tensor, not for the whole ouput of an op, i.e. it should
// not have an outer TFT_PRODUCT.
Status CheckMemoryType(bool use_host_memory, const FullTypeDef& ft);

}  // namespace tensorflow::full_type

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_ARG_RET_PLACEMENT_H_
