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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_ATTRIBUTE_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_ATTRIBUTE_UTILS_H_

#include <string>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/tf2xla/tf2xla_defs.h"

namespace mlir {
namespace TF {

// TODO(b/229028654) Use definitions from tf2xla_defs.h directly. We currently
// don't do this to avoid explicit casts (implicit conversion from
// `absl::string_view` to `llvm::StringRef` is not supported until C++17).

// Whether soft placement is allowed. If true, the marked node is eligible for
// outside compilation.
inline constexpr llvm::StringRef kAllowSoftPlacementAttr =
    "allow_soft_placement";

// Marks a node for XLA compilation. The attribute value indicates the
// compilation device type.
inline constexpr llvm::StringRef kCompileDeviceTypeAttr =
    "_xla_compile_device_type";
// The attribute value speicifes the preferred outlined function name in
// ClusterOutliningPass.
inline constexpr llvm::StringRef kClusterOutlinedFunctionNameAttr =
    "_cluster_outlined_function_name";
// Marks a node for replication. The attribute value indicates the replication
// metadata op.
inline constexpr llvm::StringRef kReplicationInfoAttr = "_replication_info";
// Marks a node for XLA-TPU compilation. The attribute value indicates the
// associated compilation cluster and replication metadata op.
inline constexpr llvm::StringRef kTpuReplicateAttr = "_tpu_replicate";
// Device types.
inline constexpr llvm::StringRef kTpuDevice = "TPU";
// _xla_outside_compilation
inline constexpr llvm::StringRef kXlaOutsideCompilationAttr =
    "_xla_outside_compilation";
// device attr
inline constexpr llvm::StringRef kDeviceAttr = "device";
// Function attribute to signal that a function should be skipped from TPU
// island outlining. The attribute is set in
// `TpuV1BridgeExecutorIslandCoarsening` and removed in the subsequent
// `TPUBridgeExecutorIslandOutlining` pass.
inline constexpr llvm::StringRef kSkipIslandOutlining =
    "_skip_island_outlining";
// Function attribute to signal which argument contains bounded dynamic
// dimension.
inline constexpr llvm::StringRef kDynamicArgIndexAttr = "_dynamic_arg_index";

// This string attribute encodes parallel execution groups and their associated
// branches. It has the following format:
// `_parallel_execution_ids= group1:branch1,group2:branch2,...`
// For example, if we have IR as follows:
//
// tf_executor.island wraps "tf.OpA"
// tf_executor.island {
//  "tf_device.replicate" {n = 2} {
//    "tf.OpB"
//    "tf_device.parallel_execute"() ({
//      "tf.OpC"
//    }, {
//      "tf.OpD"
//    })
//  }
//
// The above IR will be flattened after `ReplicateToIslandPass` and
// `ParallelExecuteToIslandsPass` as follows:
//
// tf_executor.island wraps "tf.OpA"
// tf_executor.island {_parallel_execution_ids=r0:0} wraps "tf.OpB"
// tf_executor.island {_parallel_execution_ids=r0:0,p0:0} wraps "tf.OpC"
// tf_executor.island {_parallel_execution_ids=r0:0,p0:1} wraps "tf.OpD"
// tf_executor.island {_parallel_execution_ids=r0:1} wraps "tf.OpB"
// tf_executor.island {_parallel_execution_ids=r0:1,p0:0} wraps "tf.OpC"
// tf_executor.island {_parallel_execution_ids=r0:1,p0:1} wraps "tf.OpD"
//
// "tf.OpA" will not have `_parallel_execution_ids` attr,
//          means it does not belong to any parallel execution groups.
// First instance of "tf.OpB" after flattening will have
//          `_parallel_execution_ids = "r0:0"`,
//          which represents the first branch of replicate group 0.
// Second instance of "tf.OpB" after flattening will have
//          `_parallel_execution_ids = "r0:1"`
//          which represents the second branch of replicate group 0.
// First instance of "tf.OpC" after flattening will have
//          `_parallel_execution_ids = "r0:0,p0:0"`
//          which represents the first branch of replicate group 0 and
//          the first branch of parallel group 0.
// Second instance of "tf.OpC" after flattening will have
//          `_parallel_execution_ids = "r0:1,p0:0"`
//          which represents the second branch of replicate group 0 and
//          the first branch of parallel group 0.
// First instance of "tf.OpD" after flattening will have
//          `_parallel_execution_ids = "r0:0,p0:1"`
//          which represents the first branch of replicate group 0 and
//          the second branch of parallel group 0.
// Second instance of "tf.OpD" after flattening will have
//          `_parallel_execution_ids = "r0:1,p0:1"`
//          which represents the second branch of replicate group 0 and
//          the second branch of parallel group 0.
inline constexpr llvm::StringRef kParallelExecAnnotation =
    "_parallel_execution_ids";

// Logging

// Name of component for error logging. This name is fixed and required to
// enable logging.
inline const char kBridgeComponent[] = "TFXLABridge";
inline const char kMlirPh1BridgeCounterReplicated[] = "replicated";
inline const char kMlirPh1BridgeCounterNonReplicated[] = "nonreplicated";
inline const char kMlirPh1BridgeCounterV1[] = "v1";
inline const char kMlirPh1BridgeCounterV2[] = "v2";
inline const char kMlirPh1BridgeCounterTpu[] = "tpu";
inline const char kMlirPh1BridgeCounterNonTpu[] = "cpu/gpu";
inline const char kXlaOutsideCompilation[] = "_xla_outside_compilation";

// Copies attributes that satisfy the given predicate from `from` to `to`.
template <typename Predicate>
void CopyAttributes(Operation *from, Operation *to, Predicate P) {
  for (const NamedAttribute &attr : from->getAttrs())
    if (P(attr)) to->setAttr(attr.getName(), attr.getValue());
}

// Copies attributes whose name begins with an _ from `from` to `to`.
inline void CopyUnderscoredAttributes(Operation *from, Operation *to) {
  CopyAttributes(from, to, [](const NamedAttribute &attr) {
    return attr.getName().strref().front() == '_';
  });
}

// Copies outside compilation attribute from `from` to `to`.
inline void CopyXlaOutsideCompilationAttributes(Operation *from,
                                                Operation *to) {
  CopyAttributes(from, to, [](const NamedAttribute &attr) {
    return attr.getName().strref() == kXlaOutsideCompilationAttr;
  });
}

// Copies attributes that are either `device` or whose name begins with an _
// from `from` to `to`.
// TODO(b/158769932): This should be a general feature instead post some policy
// discussion.
inline void CopyDeviceAndUnderscoredAttributes(Operation *from, Operation *to) {
  auto device = mlir::StringAttr::get(from->getContext(), "device");
  CopyAttributes(from, to, [&device](const NamedAttribute &attr) {
    return attr.getName().strref().front() == '_' || attr.getName() == device;
  });
}

// Forward declare these passthrough ops.
// TODO(jpienaar): Remove these and use trait instead.
class IdentityOp;
class IdentityNOp;

// Returns if a value corresponds to a constant, returns the matched constant
// as an attribute.
template <typename AttrT>
bool GetValueAsConstant(Value val, AttrT &attr) {
  while (auto result = mlir::dyn_cast<OpResult>(val)) {
    Operation *op = result.getOwner();
    if (!isa<IdentityOp>(op) && !isa<IdentityNOp>(op)) break;
    val = op->getOperand(result.getResultNumber());
  }
  return matchPattern(val, m_Constant(&attr));
}

// Checks if both compilation and replication attributes are present in the
// operation, and if their values are valid.
LogicalResult HasValidCompilationAndReplicationAttributes(Operation &op);

// Checks if the device attribute is valid.
LogicalResult IsValidDeviceTypeOrEmpty(StringAttr attr);

using ParallelExecutionIdPairs =
    llvm::SmallVector<std::pair<std::string, std::string>, 8>;
// Parses the parallel execution attribute for `op` and fills `id_pairs` with
// the corresponding (group ID,branch ID) pairs.
// Returns `failure` if the attribute is malformed.
LogicalResult ParseParallelExecutionIds(Operation *op,
                                        ParallelExecutionIdPairs &id_pairs);

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_ATTRIBUTE_UTILS_H_
