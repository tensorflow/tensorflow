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

#include "tensorflow/contrib/xlagen/xlagen.h"

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/platform/test.h"


namespace tensorflow {

TEST(XlaGen, TestBasic) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::Placeholder(scope.WithOpName("A"), DT_INT32,
                            ops::Placeholder::Shape({ 2, 2 }));
  auto b = ops::Placeholder(scope.WithOpName("B"), DT_INT32,
                            ops::Placeholder::Shape({ 2 }));
  auto c = ops::Add(scope.WithOpName("C"), a, b);

  GraphDef graphdef;
  TF_CHECK_OK(scope.ToGraphDef(&graphdef));

  auto s = xlagen::GraphDefToXlaSessionModule({"C:0"}, graphdef);
  EXPECT_TRUE(s.ok()) << s.status();
  std::unique_ptr<xla::SessionModule> module = std::move(s.ValueOrDie());
  LOG(INFO) << module->DebugString();
}

}  // end namespace tensorflow
