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

#include "tensorflow/compiler/xla/differential_set.h"

#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

TEST(DifferentialSetTest, TellsWhetherSetContainsSomethingHeld) {
  DifferentialSet<int> set;
  set.Add(1);
  set.Add(2);
  EXPECT_FALSE(set.Contains(3));
  EXPECT_TRUE(set.Contains(1));
  EXPECT_TRUE(set.Contains(2));
  EXPECT_FALSE(set.Contains(0));
}

TEST(DifferentialSetTest, TellsWhetherSetContainsSomethingParentHolds) {
  DifferentialSet<int> parent;
  parent.Add(1);
  DifferentialSet<int> child(&parent);
  child.Add(2);

  // Test properties of the child.
  EXPECT_FALSE(child.Contains(3));
  EXPECT_TRUE(child.Contains(1));
  EXPECT_TRUE(child.Contains(2));
  EXPECT_FALSE(child.Contains(0));

  // Test properties of the parent.
  EXPECT_TRUE(parent.Contains(1));
  EXPECT_FALSE(parent.Contains(2));
}

}  // namespace
}  // namespace xla
