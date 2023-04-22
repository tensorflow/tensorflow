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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OPS_A_M_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OPS_A_M_H_

#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/ControlFlowInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"  // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "mlir/Interfaces/LoopLikeInterface.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_interfaces.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_traits.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_verifiers.h"

namespace mlir {
namespace TF {

class YieldOp;

}  // namespace TF
}  // namespace mlir

// TODO(b/131258166): TensorFlow's mutex.h defines a `mutex_lock` macro, whose
// purpose is to catch bug on `tensorflow::mutex_lock`. We don't use
// `tensorflow::mutex_lock` here but we have ops (`tf.MutexLock` and
// `tf.ConsumeMutexLock`) with getter methods named as `mutex_lock()`. Need to
// undefine here to avoid expanding the getter symbol as macro when including
// both mutex.h and this header file.
#undef mutex_lock

#define GET_OP_FWD_DEFINES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_all_ops.h.inc"
#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h.inc"

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OPS_A_M_H_
