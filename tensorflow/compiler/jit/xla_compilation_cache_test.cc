/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/xla_compilation_cache.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(XlaCompilationCacheTest, SignatureEquality) {
  NameAttrList fn;
  fn.set_name("afunction");
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kConstant;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({4, 0});
  args[0].constant_value = Tensor(DT_INT32, {4, 0});
  TF_ASSERT_OK_AND_ASSIGN(XlaCompilationCache::Signature s1,
                          XlaCompilationCache::BuildSignature(fn, args));

  args[0].type = DT_FLOAT;
  args[0].constant_value = Tensor(DT_FLOAT, {4, 0});
  TF_ASSERT_OK_AND_ASSIGN(XlaCompilationCache::Signature s2,
                          XlaCompilationCache::BuildSignature(fn, args));

  args[0].shape = TensorShape({0, 4});
  args[0].constant_value = Tensor(DT_FLOAT, {0, 4});
  TF_ASSERT_OK_AND_ASSIGN(XlaCompilationCache::Signature s3,
                          XlaCompilationCache::BuildSignature(fn, args));

  std::vector<XlaCompilationCache::Signature> signatures = {s1, s2, s3};
  for (int i = 0; i < signatures.size(); ++i) {
    for (int j = 0; j < signatures.size(); ++j) {
      EXPECT_EQ(i == j, signatures[i] == signatures[j])
          << signatures[i].HumanString() << " " << signatures[j].HumanString();
    }
  }
}

}  // namespace
}  // namespace tensorflow
