/* Copyright 2025 The OpenXLA Authors.

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

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

TEST(HloInstruction, SetFrontendAttribute) {
  HloConstantInstruction instr(ShapeUtil::MakeShape(U32, {3, 2}));
  instr.set_frontend_attribute("key1", "value1");
  EXPECT_EQ(instr.get_frontend_attribute("key1").value(), "value1");
  instr.set_frontend_attribute("key1", "value2");
  EXPECT_EQ(instr.get_frontend_attribute("key1").value(), "value2");
}

TEST(HloInstruction, AddFrontendAttribute) {
  HloConstantInstruction instr(ShapeUtil::MakeShape(U32, {3, 2}));
  EXPECT_TRUE(instr.add_frontend_attribute("key1", "value1"));
  EXPECT_EQ(instr.get_frontend_attribute("key1").value(), "value1");
  EXPECT_FALSE(instr.add_frontend_attribute("key1", "value2"));
  EXPECT_EQ(instr.get_frontend_attribute("key1").value(), "value1");
}

TEST(HloInstruction, SetFrontendAttributes) {
  HloConstantInstruction instr(ShapeUtil::MakeShape(U32, {3, 2}));
  instr.add_frontend_attribute("key1", "value1");
  FrontendAttributes attributes;
  attributes.mutable_map()->insert({"key1", "value2"});
  attributes.mutable_map()->insert({"key2", "value2"});
  instr.set_frontend_attributes(attributes);
  EXPECT_EQ(instr.get_frontend_attribute("key1").value(), "value2")
      << "key1 should be overwritten";
  EXPECT_EQ(instr.get_frontend_attribute("key2").value(), "value2");
}

TEST(HloInstruction, AddFrontendAttributes) {
  HloConstantInstruction instr(ShapeUtil::MakeShape(U32, {3, 2}));
  instr.add_frontend_attribute("key1", "value1");
  FrontendAttributes attributes;
  attributes.mutable_map()->insert({"key1", "value2"});
  attributes.mutable_map()->insert({"key2", "value2"});
  instr.add_frontend_attributes(attributes);
  EXPECT_EQ(instr.get_frontend_attribute("key1").value(), "value1")
      << "key1 should not be overwritten";
  EXPECT_EQ(instr.get_frontend_attribute("key2").value(), "value2");
}

}  // namespace
}  // namespace xla
