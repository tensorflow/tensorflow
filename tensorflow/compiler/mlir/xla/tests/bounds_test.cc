/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/xla/bounds.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/IntegerSet.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {
namespace {

mlir::RankedTensorType MakeBoundedType(mlir::MLIRContext& context) {
  mlir::Attribute encoding;
  mlir::RankedTensorType t = mlir::RankedTensorType::get(
      /*shape=*/{100, 100, 100, 100},
      /*elementType=*/mlir::IntegerType::get(&context, 32));

  encoding = t.getEncoding();

  encoding =
      addOrModifyUpperBound(&context, encoding, /*dimension=*/0, /*limit=*/33);
  encoding =
      addOrModifyUpperBound(&context, encoding, /*dimension=*/1, /*limit=*/42);

  // set dimension 2 twice:
  encoding =
      addOrModifyUpperBound(&context, encoding, /*dimension=*/2, /*limit=*/77);
  encoding =
      addOrModifyUpperBound(&context, encoding, /*dimension=*/2, /*limit=*/50);

  // rewrite the type
  return mlir::RankedTensorType::get(t.getShape(), t.getElementType(),
                                     encoding);
}

TEST(BoundsTest, Storage) {
  mlir::MLIRContext context;
  auto t = MakeBoundedType(context);
  CHECK_EQ(getUpperBoundFromAttr(t.getEncoding(), 0), 33);
  CHECK_EQ(getUpperBoundFromAttr(t.getEncoding(), 1), 42);
  CHECK_EQ(getUpperBoundFromAttr(t.getEncoding(), 2), 50);
  CHECK_EQ(getUpperBoundFromAttr(t.getEncoding(), 3), -1);

  // getUpperBoundsForTensor() returns the dimension size for unbounded
  // dimensions.
  llvm::SmallVector<int64_t, 4> expected({33, 42, 50, 100});
  auto result = getUpperBoundsForTensor(t.getEncoding(), t);
  CHECK(result == expected);
}

TEST(BoundsTest, AttributeInterface) {
  mlir::MLIRContext context;
  mlir::IntegerSetAttr::attachInterface<mlir::BoundsAttrInterface>(context);

  auto t = MakeBoundedType(context);
  CHECK(t.getEncoding().isa<mlir::BoundsAttrInterface>());
  mlir::BoundsAttrInterface bounds =
      t.getEncoding().cast<mlir::BoundsAttrInterface>();
  CHECK_EQ(bounds.getBound(0), 33);
  CHECK_EQ(bounds.getBound(1), 42);
  CHECK_EQ(bounds.getBound(2), 50);  // latest value
  CHECK_EQ(bounds.getBound(3), -1);  // no bound

  // Interface equivalent of getUpperBoundsForTensor()
  llvm::SmallVector<int64_t, 4> expected({33, 42, 50, 100});
  auto result = bounds.getBoundsForTensor(t);
  CHECK(result == expected);
}

TEST(BoundsTest, TypeInterface) {
  mlir::MLIRContext context;
  mlir::RankedTensorType::attachInterface<mlir::BoundedRankedTensorType>(
      context);

  auto tensor_type = MakeBoundedType(context);
  CHECK(tensor_type.isa<mlir::BoundedRankedTensorType>());
  auto t = tensor_type.cast<mlir::BoundedRankedTensorType>();
  CHECK_EQ(t.getBound(0), 33);
  CHECK_EQ(t.getBound(1), 42);
  CHECK_EQ(t.getBound(2), 50);  // latest value
  CHECK_EQ(t.getBound(3), -1);  // no bound
  llvm::SmallVector<int64_t, 4> expected({33, 42, 50, 100});
  auto result = t.getBounds();
  CHECK(result == expected);
}

}  // namespace
}  // namespace tensorflow
