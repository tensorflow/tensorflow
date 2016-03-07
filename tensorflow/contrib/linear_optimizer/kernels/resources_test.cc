/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/contrib/linear_optimizer/kernels/resources.h"

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(DataByExample, MakeKeyIsCollisionResistent) {
  const DataByExample::Key key = DataByExample::MakeKey("TheExampleId");
  EXPECT_NE(key.first, key.second);
  EXPECT_NE(key.first & 0xFFFFFFFF, key.second);
}

TEST(DataByExample, ElementAccessAndMutationWorks) {
  const string container = "TheContainer";
  const string solver = "TheSolver";
  ResourceMgr rm;
  ASSERT_TRUE(
      rm.Create(container, solver, new DataByExample(container, solver)).ok());

  DataByExample* duals_by_example;
  ASSERT_TRUE(rm.Lookup(container, solver, &duals_by_example).ok());

  const DataByExample::Key key1 = DataByExample::MakeKey("TheExampleId1");
  EXPECT_EQ(0, (*duals_by_example)[key1].dual);

  (*duals_by_example)[key1].dual = 1;
  EXPECT_EQ(1, (*duals_by_example)[key1].dual);

  const DataByExample::Key key2 = DataByExample::MakeKey("TheExampleId2");
  EXPECT_NE((*duals_by_example)[key1].dual, (*duals_by_example)[key2].dual);

  // TODO(katsiapis): Use core::ScopedUnref once it's moved out of internal.
  duals_by_example->Unref();
}

}  // namespace
}  // namespace tensorflow
