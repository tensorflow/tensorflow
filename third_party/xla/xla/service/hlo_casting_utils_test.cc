/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/hlo/ir/hlo_casting_utils.h"

#include "xla/hlo/ir/hlo_instruction.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

class DummyInstruction : public HloInstruction {
 public:
  DummyInstruction()
      : HloInstruction(HloOpcode::kConstant,
                       ShapeUtil::MakeValidatedShape(F32, {}).value()) {}

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kConstant;
  }
};

class AnotherDummyInstruction : public HloInstruction {
 public:
  AnotherDummyInstruction()
      : HloInstruction(HloOpcode::kParameter,
                       ShapeUtil::MakeValidatedShape(F32, {}).value()) {}

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kParameter;
  }
};

TEST(HloCastingUtilsTest, CastSucceeds) {
  DummyInstruction instruction;
  DummyInstruction* casted =
      Cast<DummyInstruction>(static_cast<HloInstruction*>(&instruction));
  ASSERT_EQ(casted, &instruction);
}

TEST(HloCastingUtilsTest, CastDiesForWrongType) {
  AnotherDummyInstruction instruction;
  ASSERT_DEATH(
      Cast<DummyInstruction>(static_cast<HloInstruction*>(&instruction)), "");
}

TEST(HloCastingUtilsTest, CastDiesForNullptr) {
  HloInstruction* null = nullptr;
  ASSERT_DEATH(Cast<DummyInstruction>(null), "");
}

TEST(HloCastingUtilsTest, DynCastSucceeds) {
  DummyInstruction instruction;
  DummyInstruction* casted =
      DynCast<DummyInstruction>(static_cast<HloInstruction*>(&instruction));
  ASSERT_EQ(casted, &instruction);
}

TEST(HloCastingUtilsTest, DynCastReturnsNullptrForWrongType) {
  AnotherDummyInstruction instruction;
  DummyInstruction* casted =
      DynCast<DummyInstruction>(static_cast<HloInstruction*>(&instruction));
  ASSERT_EQ(casted, nullptr);
}

TEST(HloCastingUtilsTest, DynCastDiesForNullptr) {
  HloInstruction* null = nullptr;
  ASSERT_DEATH(DynCast<DummyInstruction>(null), "");
}

}  // namespace
}  // namespace xla
