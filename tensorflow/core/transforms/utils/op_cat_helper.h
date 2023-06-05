/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TRANSFORMS_UTILS_OP_CAT_HELPER_H_
#define TENSORFLOW_CORE_TRANSFORMS_UTILS_OP_CAT_HELPER_H_

#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/tf_op_wrapper.h"

namespace mlir {
namespace tfg {
// A Helper class to identify if an op belongs to certain op category.
class OpCatHelper {
 public:
  OpCatHelper() = default;
  explicit OpCatHelper(TFGraphDialect *dialect) : dialect_(dialect) {}

  bool IsAggregate(TFOp op);
  bool IsCommutative(TFOp op);

  // Returns true if it's a splat tensor type and has the splat value 1.
  bool IsOnes(TFOp op);
  // Returns true if it's a splat tensor type and has the splat value 0.
  bool IsZeros(TFOp op);

  // Returns true if the op is known to use persistent memory to store its
  // value.
  bool IsPersistent(TFOp op);

  // Returns true if the op belongs to the NC_DATASET class (see graph/graph.h).
  bool IsDataset(TFOp op);

  TFGraphDialect *getDialect() const { return dialect_; }

 protected:
  TFGraphDialect *dialect_;
};
}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_TRANSFORMS_UTILS_OP_CAT_HELPER_H_
