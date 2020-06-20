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

#include "tensorflow/python/framework/python_op_gen.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(PythonOpGen, Basic) {
  OpList ops;
  OpRegistry::Global()->Export(false, &ops);

  ApiDefMap api_def_map(ops);

  string code = GetPythonOps(ops, api_def_map, {}, "");

  EXPECT_TRUE(absl::StrContains(code, "def case"));

  // TODO(mdan): Add tests to verify type annotations are correctly added.
}

// TODO(mdan): Include more tests with synhtetic ops and api defs.

}  // namespace
}  // namespace tensorflow
