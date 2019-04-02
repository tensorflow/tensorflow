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

// Contains utilities for clustering compilable graph nodes via XLA.

#ifndef TENSORFLOW_COMPILER_JIT_XLA_CLUSTER_UTIL_H_
#define TENSORFLOW_COMPILER_JIT_XLA_CLUSTER_UTIL_H_

#include "absl/types/optional.h"
#include "tensorflow/compiler/jit/graphcycles/graphcycles.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

// The attribute that marks nodes to be grouped into functions by the
// encapsulate subgraphs pass.
extern const char* const kXlaClusterAttr;

// The attribute that marks nodes in a cluster to be placed outside the xla
// compilation by the encapsulate subgraphs pass.
extern const char* const kXlaOutsideCompilationAttr;

// The attribute that marks certain inputs to a Node as required to be a
// constant at compile time.  If this attribute is present then the
// CompileTimeConstantInput information in the corresponding XlaOpKernel is
// ignored.
//
// The value for this attribute, if present, has to be a list of strings naming
// the inputs to the node that must be constant.
extern const char* const kXlaCompileTimeConstantInputsAttr;

using OrderedNodeSet = std::set<Node*, NodeComparatorID>;

// Returns the DeviceType corresponding to 'device'.
Status DeviceToDeviceType(const string& device, DeviceType* device_type);

// Returns true if `node` has a ref tensor input that it forwards to its output.
bool HasForwardedRefInput(const Node& node);

// Creates a graph representation to enable cycle detection when clustering.
// This representation handles loops in graph by disconnecting each loop from
// the enclosing graph.
//
// Returns true for success and false for valid graphs that we can't handle yet
// (b/127521408).
xla::StatusOr<bool> CreateCycleDetectionGraph(const Graph* graph,
                                              GraphCycles* cycles);

// Returns the XLA cluster in which `node` is placed if it is in an XLA cluster,
// otherwise returns nullopt.
absl::optional<absl::string_view> GetXlaClusterForNode(const Node& node);

// Removes `node_def` its XLA cluster (by clearing its _XlaCluster attribute).
void RemoveFromXlaCluster(NodeDef* node_def);

// Removes `node` its XLA cluster (by clearing its _XlaCluster attribute).
void RemoveFromXlaCluster(Node* node);

// Returns true if `node` has a DT_RESOURCE typed input or output.
bool HasResourceInputOrOutput(const Node& node);

// Adds edges to `cycles` to prevent clustering resource operations that cannot
// be legally clustered.
Status AdjustCycleDetectionGraphForResourceOps(
    const Graph* graph, const FunctionLibraryDefinition* flib_def,
    const std::function<Status(const Node&, bool*)>& resource_ops_to_ignore,
    GraphCycles* cycles);

// Picks the device for which XLA should compile a cluster that contains
// operations placed in devices in `device_names`.  For instance a cluster that
// contains operations solely placed on the CPU will be compiled into a CPU
// executable by XLA, whereas a cluster that contains operations placed on the
// CPU and also operations placed on the GPU will be compiled into a GPU
// executable.
//
// Returns a non-OK Status if no unambiguous choice of device exists.
//
// We choose the device using the following rules:
//
//  - It is an error for `device_names` to contain more than one device of the
//    same type.
//  - GPU is preferred over CPU.
//  - If `allow_mixing_unknown_and_cpu` is true then unknown devices are
//    preferred over CPU.
//  - XLA devices count as "unrecognized devices".
//
// This set of rules above implicitly assume that XLA:GPU can compile all
// operations in the cluster that XLA:CPU can compile, and if
// `allow_mixing_unknown_and_cpu` then the unrecognized device can also compile
// all operations in the cluster that XLA:CPU can compile.
//
// We provide the `allow_mixing_unknown_and_cpu` knob so that we can do both of
// the following things:
//
// - Let MarkForCompilationPass not inject CPU-placed operations into clusters
//   that will run on unknown devices (because the unknown XLA backend may not
//   support every operation supported by CPU).
// - Let BuildXlaOpsPass successfully infer a compilation device for a cluster
//   that contains nodes placed on both the CPU and on unknown devices.  In this
//   case it is the responsibility of the optimization pass that injected the
//   CPU nodes into the cluster to ensure that these nodes can be compiled by
//   the unknown XLA backend.
Status PickDeviceForXla(absl::Span<const string> device_names,
                        bool allow_mixing_unknown_and_cpu,
                        string* out_device_picked);

// This is like `PickDeviceForXla` except that it returns false (instead of a
// non-OK Status) in `out_can_pick_device` if no unambiguous choice of device
// exists.
Status CanPickDeviceForXla(absl::Span<const string> device_names,
                           bool allow_mixing_unknown_and_cpu,
                           bool* out_can_pick_device);

// Determines the global jit level based on GraphOptimizationPassOptions,
// --tf_xla_auto_jit and whether the graph is a single GPU graph.
OptimizerOptions::GlobalJitLevel GetGlobalJitLevelForGraph(
    const GraphOptimizationPassOptions& options);

// Returns true if `g` is a single-GPU graph.  A single-GPU graph uses exactly
// one GPU (and any number of CPUs).
bool IsSingleGpuGraph(const Graph& g);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_CLUSTER_UTIL_H_
