/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"

namespace xla {
namespace {
TEST_F(ClientLibraryTestBase, DeepGraph) {
  // TODO(b/62624812): To trigger the stack overflow this test is
  // intended to track, we need to set kDepth to 20000.
  // Unfortunately, setting it that high causes the test to time out.
  const int kDepth = 200;
  XlaBuilder b(TestName());
  XlaOp x;
  XlaOp y;
  auto x_data = CreateR0Parameter<int32>(3, 0, "x", &b, &x);
  auto y_data = CreateR0Parameter<int32>(1, 1, "y", &b, &y);
  XlaOp z = x;
  for (int i = 0; i < kDepth; ++i) {
    z = Add(z, y);
  }
  ComputeAndCompareR0<int32>(&b, /*expected=*/kDepth + 3,
                             {x_data.get(), y_data.get()});
}
}  // namespace
}  // namespace xla
