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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_MEMORY_TYPES_H_
#define THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_MEMORY_TYPES_H_

#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Returns an error iff *g running on a single device of 'device_type'
// has memory type mismatch for any edge's source and destination.
Status ValidateMemoryTypes(const DeviceType& device_type, const Graph* g);

// Updates '*g' so that every edge's source and destination has
// compatible memory types by inserting proper HostSend/Recv and
// Send/HostRecv nodes.  'device_type' specifies the type of device on
// which '*g' is going to run on and that device has the name
// 'device_name'.
//
// Returns OK if '*g' is updated properly (ValidateMemoryTypes(g) must
// be OK). Otherwise, returns an error and '*g' may be in an
// invalidate state and the caller should discard it.
Status EnsureMemoryTypes(const DeviceType& device_type,
                         const string& device_name, Graph* g);

// Get the memory type for 'index'th output of node 'n' in graph 'g', when
// running on 'device_type'.
Status MemoryTypeForOutput(const DeviceType& device_type, const Graph* g,
                           const Node* n, int index, MemoryType* memory_type);

}  // end namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_MEMORY_TYPES_H_
