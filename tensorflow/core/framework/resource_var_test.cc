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

#include "tensorflow/core/framework/resource_var.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace core {

TEST(ResourceVarTest, Uninitialize) {
  RefCountPtr<Var> var{new Var(DT_INT32)};
  EXPECT_FALSE(var->is_initialized);
  EXPECT_TRUE(var->tensor()->data() == nullptr);

  *(var->tensor()) = Tensor(DT_INT32, TensorShape({1}));
  var->is_initialized = true;
  EXPECT_TRUE(var->tensor()->data() != nullptr);

  var->Uninitialize();
  EXPECT_FALSE(var->is_initialized);
  EXPECT_TRUE(var->tensor()->data() == nullptr);
}
}  // namespace core
}  // namespace tensorflow
