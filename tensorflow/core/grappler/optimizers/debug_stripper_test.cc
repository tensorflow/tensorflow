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

#include "tensorflow/core/grappler/optimizers/debug_stripper.h"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class DebugStripperTest : public GrapplerTest {};

// TODO(haoliang): Add tests for different removal operations.
TEST_F(DebugStripperTest, OutputEqualToInput) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto c = ops::Const(s.WithOpName("c"), 0, {});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  DebugStripper optimizer;
  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
