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

#include "tensorflow/compiler/mlir/tf2xla/api/v1/legalize_tf.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/compiler/mlir/tf2xla/api/v1/device_type.pb.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"

namespace tensorflow {
namespace tf2xla {
namespace v1 {

TEST(TestLegalizeMlirToXlaHlo, LegalizeMlirToXlaHlo) {
  mlir::ModuleOp module;
  std::vector<TensorShape> arg_shapes;
  tsl::StatusOr<tensorflow::XlaCompilationResult> compilation_result =
      LegalizeMlirToXlaHlo(module, arg_shapes, DeviceType::XLA_TPU_JIT,
                           /*use_tuple_args=*/true);

  EXPECT_FALSE(compilation_result.ok());
}

}  // namespace v1
}  // namespace tf2xla
}  // namespace tensorflow
