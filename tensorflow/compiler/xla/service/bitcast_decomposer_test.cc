/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/bitcast_decomposer.h"

#include <string>
#include <tuple>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/random/random.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

namespace m = ::xla::match;

std::vector<Shape> AllPermutationsOfShape(const Shape& s) {
  std::vector<int64_t> dims_perm(s.dimensions_size());
  absl::c_iota(dims_perm, 0);
  std::vector<int64_t> layout_perm(s.dimensions_size());
  absl::c_iota(layout_perm, 0);

  std::vector<Shape> ret;
  do {
    do {
      Shape new_shape = ShapeUtil::MakeShapeWithLayout(
          s.element_type(),  //
          ComposePermutations(s.dimensions(), dims_perm),
          ComposePermutations(s.layout().minor_to_major(), layout_perm));
      ret.push_back(new_shape);
    } while (absl::c_next_permutation(layout_perm));
  } while (absl::c_next_permutation(dims_perm));
  return ret;
}

std::vector<Shape> AllPermutationsOfShapes(
    std::vector<std::vector<int64_t>> dims_list) {
  std::vector<Shape> ret;
  for (const auto& dims : dims_list) {
    std::vector<Shape> perms = AllPermutationsOfShape(
        ShapeUtil::MakeShapeWithDescendingLayout(F32, dims));
    ret.insert(ret.end(), perms.begin(), perms.end());
  }
  return ret;
}

class BitcastDecomposerParameterizedTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          std::tuple<Shape /*src*/, Shape /*dst*/>> {
 public:
  BitcastDecomposerParameterizedTest()
      : HloTestBase(/*verifier_layout_sensitive=*/false,
                    /*allow_mixed_precision_in_hlo_verifier=*/false) {}

 protected:
  absl::BitGen rand_;
};

INSTANTIATE_TEST_SUITE_P(
    Handcrafted, BitcastDecomposerParameterizedTest,
    ::testing::Values(std::make_tuple(
        ShapeUtil::MakeShapeWithLayout(F32, {1, 2, 4, 2, 2}, {2, 4, 3, 1, 0}),
        ShapeUtil::MakeShapeWithLayout(F32, {1, 2, 2, 2, 4},
                                       {4, 3, 2, 1, 0}))));

// Skip most tests in sanitizer/debug builds, otherwise this times out.
#if !defined(ADDRESS_SANITIZER) && !defined(MEMORY_SANITIZER) && \
    !defined(THREAD_SANITIZER) && defined(NDEBUG)
INSTANTIATE_TEST_SUITE_P(
    Combinatorial, BitcastDecomposerParameterizedTest,
    ::testing::Combine(
        // src shapes
        ::testing::Values(
            ShapeUtil::MakeShapeWithDescendingLayout(F32, {4, 10, 100}),
            ShapeUtil::MakeShapeWithDescendingLayout(F32, {4, 1, 10, 100}),
            ShapeUtil::MakeShapeWithLayout(F32, {4, 10, 100}, {0, 2, 1})),
        // dst shapes
        ::testing::ValuesIn(AllPermutationsOfShapes({
            // Original shape without degenerate dims.
            {4, 10, 100},
            // Original shape with degenerate dims.
            {1, 4, 10, 100},
            // Redistributing elements between dims while maintaining the same
            // rank.
            {2, 20, 100},
            {40, 10, 10},
            {1, 40, 10, 10},
            // Merging dims without redistributing elements between dims.
            {40, 100},
            {400, 10},
            {4000},
            {1, 4000},
            // Merging dims and redistributing elements between dims.
            {20, 200},
            // Splitting dims without redistributing elements between dims.
            {2, 2, 10, 100},
            {4, 2, 5, 100},
            // Splitting dims and redistributing between the dims.
            {2, 5, 5, 80},
        }))));
#endif

TEST_P(BitcastDecomposerParameterizedTest, DoIt) {
  auto [src, dst] = GetParam();

  const char* const kModuleTemplate = R"(
  HloModule module

  fused_comp {
    lhs  = $0 parameter(0)
    ROOT root = $1 bitcast(lhs)
  }

  ENTRY main {
    ROOT fusion = $1 fusion($0 parameter(0)), kind=kLoop, calls=fused_comp
  })";
  std::string module_string =
      absl::Substitute(kModuleTemplate, ShapeUtil::HumanStringWithLayout(src),
                       ShapeUtil::HumanStringWithLayout(dst));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));

  // Actually compiling and running the module is expensive; we can't afford to
  // do it for all 9000 (as of writing) tests. Pick a random 1% of them to
  // execute.
  bool execute_module = absl::Bernoulli(this->rand_, 0.01);
  Literal param, expected_val;
  if (execute_module) {
    param = MakeFakeLiteral(src).ValueOrDie();
    expected_val = ExecuteNoHloPasses(module->Clone(), {&param});
  }

  BitcastDecomposer pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());
  if (!changed) {
    // The pass shouldn't change the bitcast if and only if it's already a
    // reshape-is-bitcast.
    EXPECT_TRUE(ShapeUtil::ReshapeIsBitcast(src, dst));
    return;
  }

  // The result must be of the form transpose(bitcast(transpose(param))), except
  // that any of these operations can be skipped.
  const HloInstruction* root = module->entry_computation()
                                   ->root_instruction()
                                   ->fused_instructions_computation()
                                   ->root_instruction();
  const HloInstruction* bitcast = nullptr;
  const HloInstruction* transpose1 = nullptr;
  const HloInstruction* transpose2 = nullptr;
  ASSERT_THAT(
      root, ::testing::AnyOf(
                GmockMatch(m::Bitcast(&bitcast, m::Parameter(0))),
                GmockMatch(m::Transpose(&transpose1, m::Parameter(0))),
                GmockMatch(m::Transpose(&transpose1,
                                        m::Bitcast(&bitcast, m::Parameter(0)))),
                GmockMatch(m::Bitcast(
                    &bitcast, m::Transpose(&transpose1, m::Parameter(0)))),
                GmockMatch(m::Transpose(
                    &transpose2,
                    m::Bitcast(&bitcast,
                               m::Transpose(&transpose1, m::Parameter(0)))))));
  if (bitcast != nullptr) {
    EXPECT_TRUE(ShapeUtil::ReshapeIsBitcast(bitcast->operand(0)->shape(),
                                            bitcast->shape()));
  }
  if (transpose1 != nullptr) {
    EXPECT_TRUE(ShapeUtil::TransposeIsBitcast(transpose1->operand(0)->shape(),
                                              transpose1->shape(),
                                              transpose1->dimensions()));
  }
  if (transpose2 != nullptr) {
    EXPECT_TRUE(ShapeUtil::TransposeIsBitcast(transpose2->operand(0)->shape(),
                                              transpose2->shape(),
                                              transpose2->dimensions()));
  }

  if (execute_module) {
    auto actual_val = ExecuteNoHloPasses(module->Clone(), {&param});
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_val, actual_val));
  }
}

}  // anonymous namespace
}  // namespace xla
