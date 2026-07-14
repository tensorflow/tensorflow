// Copyright 2025 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/dtensor/cc/small_constant_optimization.h"

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"

namespace tensorflow {
namespace dtensor {
namespace {

TEST(SmallConstantOptimization, NullTensorReturnsNullopt) {
  Layout layout = Layout::Empty();
  TF_Status* tf_status = TF_NewStatus();

  std::optional<NodeDef> result = ExtractSmallTensorValue(
      /*context=*/nullptr,
      /*tensor=*/nullptr, layout, tf_status);

  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(TF_GetCode(tf_status), TF_INVALID_ARGUMENT);

  TF_DeleteStatus(tf_status);
}

}  // namespace
}  // namespace dtensor
}  // namespace tensorflow
