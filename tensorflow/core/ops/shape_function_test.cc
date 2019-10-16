/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/platform/test.h"

// Test to ensure that all core ops have shape functions defined. This is done
// by looking at all ops registered in the test binary.

namespace tensorflow {

TEST(ShapeFunctionTest, RegisteredOpsHaveShapeFns) {
  OpRegistry* op_registry = OpRegistry::Global();
  std::vector<OpRegistrationData> op_data;
  op_registry->GetOpRegistrationData(&op_data);
  for (const OpRegistrationData& op_reg_data : op_data) {
    EXPECT_TRUE(op_reg_data.shape_inference_fn != nullptr)
        << op_reg_data.op_def.name();
  }
}

}  // namespace tensorflow
