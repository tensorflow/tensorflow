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

#include "tensorflow/c/eager/runtime.h"

#include <memory>
#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

TEST(AttrTypeMap, Lookup) {
  const AttrTypeMap* m = nullptr;
  Status s = AttrTypeMapForOp("ThisOpCannotPossiblyExist", &m);
  EXPECT_FALSE(s.ok());
  s = AttrTypeMapForOp("MatMul", &m);
  ASSERT_TRUE(s.ok()) << s;

  TF_AttrType t;
  unsigned char is_list = 1;
  s = AttrTypeByName(*m, "ThisAttribyteCannotPossiblyExist", &t, &is_list);
  EXPECT_FALSE(s.ok());
  EXPECT_NE(is_list, 0);
  s = AttrTypeByName(*m, "transpose_a", &t, &is_list);
  ASSERT_TRUE(s.ok()) << s;
  EXPECT_EQ(TF_ATTR_BOOL, t);
  EXPECT_EQ(is_list, 0);

  s = AttrTypeMapForOp("Squeeze", &m);
  ASSERT_TRUE(s.ok()) << s;
  s = AttrTypeByName(*m, "squeeze_dims", &t, &is_list);
  ASSERT_TRUE(s.ok()) << s;
  EXPECT_EQ(TF_ATTR_INT, t);
  EXPECT_NE(is_list, 0);
}

}  // namespace
}  // namespace tensorflow
