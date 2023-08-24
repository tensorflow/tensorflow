
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

#include "tensorflow/dtensor/mlir/spmd_expander.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status_matchers.h"

namespace tensorflow {
namespace dtensor {
namespace {

using ::testing::IsNull;
using ::testing::NotNull;

class DummyExpander : public SPMDExpanderBase {
  StatusOr<mlir::Operation*> ExpandOp(mlir::Operation* op) override {
    return errors::Unimplemented("");
  }

  StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutForward(
      mlir::Operation* op,
      const llvm::DenseMap<int, Layout>& input_layouts) override {
    return errors::Unimplemented("");
  }
  StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutBackward(
      mlir::Operation* op,
      const llvm::DenseMap<int, Layout>& output_layouts) override {
    return errors::Unimplemented("");
  }
};

class SPMDExpanderRegistryTest : public ::testing::Test {
 public:
  SPMDExpanderRegistryTest() {
    registry_.RegisterPropagateFn(mlir::TF::AddOp::getOperationName().str(),
                                  std::make_unique<DummyExpander>());
  }

 protected:
  SPMDExpanderRegistry registry_;
};

TEST_F(SPMDExpanderRegistryTest, LookupFromOpName) {
  EXPECT_THAT(registry_.GetPropagateFnForFullOpName("tf.Add"), NotNull());
  EXPECT_THAT(registry_.GetPropagateFnForFullOpName("Unknown"), IsNull());
}

}  // namespace
}  // namespace dtensor
}  // namespace tensorflow
