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

#include "xla/hlo/ir/hlo_original_value.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/hash/hash_testing.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_original_value_util.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tuple_tree.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Optional;
using Node = TupleTree<std::optional<OriginalArray>>::Node;

TEST(OriginalArrayTest, ToString) {
  OriginalArray array1{"inst1", {}};
  EXPECT_EQ(array1.ToString(), "\"inst1\"");

  OriginalArray array2{"inst2", {1, 2}};
  EXPECT_EQ(array2.ToString(), "\"inst2\" {1,2}");
}

TEST(OriginalArrayTest, ProtoSerde) {
  OriginalArray array1{"inst1", {}};
  OriginalArrayProto proto1 = array1.ToProto();
  EXPECT_EQ(proto1.instruction_name(), "inst1");
  EXPECT_TRUE(proto1.shape_index().empty());
  OriginalArray array1_from_proto = OriginalArray::FromProto(proto1);
  EXPECT_EQ(array1_from_proto, array1);

  OriginalArray array2{"inst2", {1, 2}};
  OriginalArrayProto proto2 = array2.ToProto();
  EXPECT_EQ(proto2.instruction_name(), "inst2");
  EXPECT_THAT(proto2.shape_index(), ElementsAre(1, 2));
  OriginalArray array2_from_proto = OriginalArray::FromProto(proto2);
  EXPECT_EQ(array2_from_proto, array2);
}

TEST(OriginalArrayTest, EqualityAndHashing) {
  OriginalArray array1{"inst1", {}};
  OriginalArray array2{"inst1", {}};
  OriginalArray array3{"inst2", {}};
  OriginalArray array4{"inst1", {1}};
  OriginalArray array5{"inst1", {1}};

  EXPECT_EQ(array1, array2);
  EXPECT_NE(array1, array3);
  EXPECT_NE(array1, array4);
  EXPECT_EQ(array4, array5);

  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      array1,
      array2,
      array3,
      array4,
      array5,
  }));
}

TEST(OriginalValueTest, ToStringScalar) {
  OriginalValue value(Node::Leaf(OriginalArray{"inst1", {}}));
  EXPECT_EQ(value.ToString(), "{\"inst1\"}");
}

TEST(OriginalValueTest, ToStringTuple) {
  OriginalValue value(
      Node::Tuple({Node::Leaf(OriginalArray{"inst1", {1}}),
                   Node::Leaf(OriginalArray{"inst2", {2}}),
                   Node::Tuple({Node::Leaf(OriginalArray{"inst3", {3}}),
                                Node::Leaf(std::nullopt)})}));
  EXPECT_EQ(value.ToString(),
            "({\"inst1\" {1}}, {\"inst2\" {2}}, ({\"inst3\" {3}}, {}))");
}

TEST(OriginalValueTest, ToStringSynthetic) {
  OriginalValue value = OriginalValue::SyntheticCall();
  EXPECT_EQ(value.ToString(), "[synthetic_call]");
}

TEST(OriginalValueTest, ProtoSerde) {
  OriginalValue value(Node::Tuple({Node::Leaf(OriginalArray{"inst1", {1}}),
                                   Node::Leaf(OriginalArray{"inst2", {2}})}));

  OriginalValueProto proto = value.ToProto();
  std::shared_ptr<OriginalValue> value_from_proto =
      OriginalValue::FromProto(proto);
  EXPECT_EQ(*value_from_proto, value);

  // Test with nullopt
  OriginalValue value_with_null(Node::Tuple(
      {Node::Leaf(OriginalArray{"inst1", {1}}), Node::Leaf(std::nullopt)}));
  OriginalValueProto proto_with_null = value_with_null.ToProto();
  std::shared_ptr<OriginalValue> value_with_null_from_proto =
      OriginalValue::FromProto(proto_with_null);
  EXPECT_EQ(value_with_null_from_proto->ToString(), value_with_null.ToString());
  EXPECT_EQ(*value_with_null_from_proto, value_with_null);

  // Test with synthetic call.
  OriginalValue value_synthetic = OriginalValue::SyntheticCall();
  OriginalValueProto proto_synthetic = value_synthetic.ToProto();
  std::shared_ptr<OriginalValue> value_synthetic_from_proto =
      OriginalValue::FromProto(proto_synthetic);
  EXPECT_TRUE(value_synthetic_from_proto->is_synthetic_call());
  EXPECT_EQ(*value_synthetic_from_proto, value_synthetic);
}

TEST(OriginalValueTest, ElementAccess) {
  OriginalValue value(Node::Tuple({Node::Leaf(OriginalArray{"inst1", {1}}),
                                   Node::Leaf(OriginalArray{"inst2", {2}})}));

  EXPECT_THAT(value.original_array({0}),
              Optional(Eq(OriginalArray{"inst1", {1}})));
  EXPECT_THAT(value.original_array({1}),
              Optional(Eq(OriginalArray{"inst2", {2}})));

  *value.mutable_original_array({1}) = OriginalArray{"inst3", {3}};
  EXPECT_THAT(value.original_array({1}),
              Optional(Eq(OriginalArray{"inst3", {3}})));
}

TEST(OriginalValueTest, Elements) {
  OriginalValue value(
      Node::Tuple({Node::Leaf(OriginalArray{"inst1", {1}}),
                   Node::Tuple({Node::Leaf(OriginalArray{"inst2", {2}})}),
                   Node::Leaf(OriginalArray{"inst3", {3}})}));

  std::vector<std::pair<ShapeIndex, std::optional<OriginalArray>>> elements;
  for (const auto& element : value.mutable_original_arrays()) {
    elements.push_back(element);
  }

  EXPECT_EQ(elements.size(), 3);
  EXPECT_EQ(elements[0].first, ShapeIndex({0}));
  EXPECT_THAT(elements[0].second, Optional(Eq(OriginalArray{"inst1", {1}})));
  EXPECT_EQ(elements[1].first, ShapeIndex({1, 0}));
  EXPECT_THAT(elements[1].second, Optional(Eq(OriginalArray{"inst2", {2}})));
  EXPECT_EQ(elements[2].first, ShapeIndex({2}));
  EXPECT_THAT(elements[2].second, Optional(Eq(OriginalArray{"inst3", {3}})));
}

TEST(OriginalValueTest, EqualityAndHashing) {
  OriginalValue value1(Node::Leaf(OriginalArray{"inst1", {}}));
  OriginalValue value2(Node::Leaf(OriginalArray{"inst1", {}}));
  OriginalValue value3(Node::Leaf(OriginalArray{"inst2", {}}));
  OriginalValue value4(Node::Tuple({Node::Leaf(OriginalArray{"inst1", {1}}),
                                    Node::Leaf(OriginalArray{"inst2", {2}})}));
  OriginalValue value5(Node::Tuple({Node::Leaf(OriginalArray{"inst1", {1}}),
                                    Node::Leaf(OriginalArray{"inst2", {2}})}));
  OriginalValue value_with_root_value(Node::Tuple(
      OriginalArray{"root", {}}, {Node::Leaf(OriginalArray{"inst1", {1}}),
                                  Node::Leaf(OriginalArray{"inst2", {2}})}));
  OriginalValue synthetic1 = OriginalValue::SyntheticCall();
  OriginalValue synthetic2 = OriginalValue::SyntheticCall();

  EXPECT_EQ(value1, value2);
  EXPECT_NE(value1, value3);
  EXPECT_NE(value1, value4);
  EXPECT_EQ(value4, value5);
  EXPECT_EQ(value4, value_with_root_value);
  EXPECT_EQ(synthetic1, synthetic2);
  EXPECT_NE(value1, synthetic1);

  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      value1,
      value2,
      value3,
      value4,
      value5,
      value_with_root_value,
      synthetic1,
      synthetic2,
  }));
}

using OriginalValueHloTest = HloHardwareIndependentTestBase;

TEST_F(OriginalValueHloTest, CreateFromInstruction) {
  const char* hlo_string = R"(
HloModule test

ENTRY main {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT tuple = (f32[], f32[]) tuple(p0, p1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* p0 = module->entry_computation()->parameter_instruction(0);
  HloInstruction* p1 = module->entry_computation()->parameter_instruction(1);
  HloInstruction* tuple = module->entry_computation()->root_instruction();

  p0->set_original_value(OriginalValue::CreateFromInstruction(p0, "prefix_"));
  p1->set_original_value(OriginalValue::CreateFromInstruction(p1, "prefix_"));
  tuple->set_original_value(OriginalValue::CreateFromInstruction(tuple));

  EXPECT_EQ(p0->original_value()->ToString(), "{\"prefix_p0\"}");
  EXPECT_EQ(p1->original_value()->ToString(), "{\"prefix_p1\"}");
  EXPECT_EQ(tuple->original_value()->ToString(),
            "({\"prefix_p0\"}, {\"prefix_p1\"})");
}

TEST_F(OriginalValueHloTest, CreateFromInstructionGTE) {
  const char* hlo_string = R"(
HloModule test

ENTRY main {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  tuple = (f32[], f32[]) tuple(p0, p1)
  ROOT gte = f32[] get-tuple-element(tuple), index=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* p0 = module->entry_computation()->parameter_instruction(0);
  HloInstruction* p1 = module->entry_computation()->parameter_instruction(1);
  HloInstruction* tuple = FindInstruction(module.get(), "tuple");
  HloInstruction* gte = module->entry_computation()->root_instruction();

  p0->set_original_value(OriginalValue::CreateFromInstruction(p0));
  p1->set_original_value(OriginalValue::CreateFromInstruction(p1));
  tuple->set_original_value(OriginalValue::CreateFromInstruction(tuple));
  gte->set_original_value(OriginalValue::CreateFromInstruction(gte));

  EXPECT_EQ(gte->original_value()->ToString(), "{\"p1\"}");
}

TEST_F(OriginalValueHloTest, CreateFromInstructionGteSynthetic) {
  const char* hlo_string = R"(
HloModule test

ENTRY main {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  tuple = (f32[], f32[]) tuple(p0, p1)
  ROOT gte = f32[] get-tuple-element(tuple), index=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* tuple = FindInstruction(module.get(), "tuple");
  HloInstruction* gte = module->entry_computation()->root_instruction();

  tuple->set_original_value(
      std::make_shared<OriginalValue>(OriginalValue::SyntheticCall()));
  gte->set_original_value(OriginalValue::CreateFromInstruction(gte));

  EXPECT_EQ(gte->original_value(), nullptr);
}

TEST_F(OriginalValueHloTest, CreateFromInstructionTuple) {
  const char* hlo_string = R"(
HloModule test

ENTRY main {
 ROOT p0 = (f32[], f32[]) parameter(0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* p0 = module->entry_computation()->parameter_instruction(0);
  p0->set_original_value(OriginalValue::CreateFromInstruction(p0));

  EXPECT_EQ(p0->original_value()->ToString(), "({\"p0\" {0}}, {\"p0\" {1}})");
}

TEST_F(OriginalValueHloTest, CreateFromInstructionTupleWithSyntheticElement) {
  const char* hlo_string = R"(
HloModule test

ENTRY main {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT tuple = (f32[], f32[]) tuple(p0, p1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* p0 = module->entry_computation()->parameter_instruction(0);
  HloInstruction* p1 = module->entry_computation()->parameter_instruction(1);
  HloInstruction* tuple = module->entry_computation()->root_instruction();

  p0->set_original_value(OriginalValue::CreateFromInstruction(p0));
  p1->set_original_value(
      std::make_shared<OriginalValue>(OriginalValue::SyntheticCall()));
  tuple->set_original_value(OriginalValue::CreateFromInstruction(tuple));

  ASSERT_NE(tuple->original_value(), nullptr);
  EXPECT_EQ(tuple->original_value()->ToString(), "({\"p0\"}, {})");
}

TEST_F(OriginalValueHloTest, CopyOriginalValue) {
  const char* hlo_string = R"(
HloModule test

ENTRY main {
  ROOT p0 = f32[] parameter(0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* p0 = module->entry_computation()->parameter_instruction(0);
  p0->set_original_value(OriginalValue::CreateFromInstruction(p0));

  std::unique_ptr<HloInstruction> clone = p0->Clone();

  CopyOriginalValue(p0, clone.get(), /*clone=*/false, /*issue_warning=*/false);
  EXPECT_EQ(p0->original_value(), clone->original_value());

  CopyOriginalValue(p0, clone.get(), /*clone=*/true, /*issue_warning=*/false);
  EXPECT_NE(p0->original_value(), clone->original_value());
  EXPECT_EQ(*p0->original_value(), *clone->original_value());
}

TEST_F(OriginalValueHloTest, CopyOriginalValueSynthetic) {
  const char* hlo_string = R"(
HloModule test

ENTRY main {
  ROOT p0 = f32[] parameter(0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* p0 = module->entry_computation()->parameter_instruction(0);
  p0->set_original_value(
      std::make_shared<OriginalValue>(OriginalValue::SyntheticCall()));

  std::unique_ptr<HloInstruction> clone = p0->Clone();

  CopyOriginalValue(p0, clone.get(), /*clone=*/false, /*issue_warning=*/false);
  EXPECT_EQ(p0->original_value(), clone->original_value());

  CopyOriginalValue(p0, clone.get(), /*clone=*/true, /*issue_warning=*/false);
  EXPECT_EQ(p0->original_value(), clone->original_value());
}

TEST_F(OriginalValueHloTest, DeduplicateOriginalValues) {
  const char* hlo_string = R"(
HloModule test

ENTRY main {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  n0 = f32[] negate(p0)
  n1 = f32[] negate(p1)
  ROOT add = f32[] add(n0, n1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* p1 = FindInstruction(module.get(), "p1");
  HloInstruction* n0 = FindInstruction(module.get(), "n0");
  HloInstruction* n1 = FindInstruction(module.get(), "n1");

  auto value1 =
      std::make_shared<OriginalValue>(Node::Leaf(OriginalArray{"instA", {}}));
  auto value2 =
      std::make_shared<OriginalValue>(Node::Leaf(OriginalArray{"instB", {}}));
  auto value1_dup =
      std::make_shared<OriginalValue>(Node::Leaf(OriginalArray{"instA", {}}));

  p0->set_original_value(value1);
  p1->set_original_value(value2);
  n0->set_original_value(value1_dup);
  n1->set_original_value(value2);  // Intentional same shared_ptr

  EXPECT_NE(p0->original_value(),
            n0->original_value());  // Different shared_ptr
  EXPECT_EQ(*p0->original_value(), *n0->original_value());  // Same value
  EXPECT_EQ(p1->original_value(), n1->original_value());    // Same shared_ptr

  DeduplicateOriginalValues(module.get());

  EXPECT_EQ(p0->original_value(),
            n0->original_value());  // Should be same shared_ptr now
  EXPECT_EQ(p1->original_value(), n1->original_value());
  EXPECT_NE(p0->original_value(), p1->original_value());
}

TEST_F(OriginalValueHloTest, DeduplicateOriginalValuesWithSynthetic) {
  const char* hlo_string = R"(
HloModule test

ENTRY main {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* p1 = FindInstruction(module.get(), "p1");

  auto value1 = std::make_shared<OriginalValue>(OriginalValue::SyntheticCall());
  auto value2 = std::make_shared<OriginalValue>(OriginalValue::SyntheticCall());

  p0->set_original_value(value1);
  p1->set_original_value(value2);

  EXPECT_NE(p0->original_value(), p1->original_value());
  EXPECT_EQ(*p0->original_value(), *p1->original_value());

  DeduplicateOriginalValues(module.get());

  EXPECT_EQ(p0->original_value(), p1->original_value());
}

TEST_F(OriginalValueHloTest, InferGetTupleElementOriginalValue) {
  const char* hlo_string = R"(
HloModule test

ENTRY main {
  p0 = f32[] parameter(0), origin={{"p0"}}
  p1 = f32[] parameter(1)
  tuple = (f32[], f32[]) tuple(p0, p1)
  ROOT gte = f32[] get-tuple-element(tuple), index=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  const HloInstruction* gte = module->entry_computation()->root_instruction();

  EXPECT_NE(gte->original_value(), nullptr);
  EXPECT_EQ(gte->original_value()->ToString(), R"({"p0"})");
}

}  // namespace
}  // namespace xla
