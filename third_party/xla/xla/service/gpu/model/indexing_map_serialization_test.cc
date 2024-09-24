/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/model/indexing_map_serialization.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/service/gpu/model/indexing_test_utils.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class IndexingMapSerializationTest : public HloTestBase {
 public:
  mlir::MLIRContext mlir_context_;
  void ParseAndCheck(absl::string_view indexing_map_str) {
    auto indexing_map = ParseIndexingMap(indexing_map_str, &mlir_context_);
    ASSERT_TRUE(indexing_map.has_value());
    EXPECT_THAT(indexing_map->ToString(),
                MatchIndexingString(indexing_map_str));
  }
};

TEST_F(IndexingMapSerializationTest, EmptyMap) { ParseAndCheck("() -> ()"); }

TEST_F(IndexingMapSerializationTest, DimsOnly) {
  ParseAndCheck(R"(
    (d0, d1) -> (d0 mod 2 + d1),
    domain:
    d0 in [0, 3],
    d1 in [0, 4],
    is_simplified: true
  )");
}

TEST_F(IndexingMapSerializationTest, SymbolsOnly) {
  ParseAndCheck(R"(
    ()[s0, s1] -> (s0 floordiv s1),
    domain:
    s0 in [0, 3],
    s1 in [0, 4],
    is_simplified: true
  )");
}

TEST_F(IndexingMapSerializationTest, DimsAndSymbolsNoConstraints) {
  ParseAndCheck(R"(
    (d0, d1)[s0, s1, s2] -> (s2, d0 + d1, s1, s0),
    domain:
    d0 in [0, 3],
    d1 in [0, 4],
    s0 in [0, 1],
    s1 in [0, 1],
    s2 in [0, 3],
    is_simplified: false
  )");
}

TEST_F(IndexingMapSerializationTest, DimsAndSymbolsAndConstraints) {
  ParseAndCheck(R"(
    (d0, d1)[s0, s1, s2] -> (s2, d0 + d1, s1, s0),
    domain:
    d0 in [0, 3],
    d1 in [0, 4],
    s0 in [0, 1],
    s1 in [0, 1],
    s2 in [0, 3],
    d0 mod 4 in [0, 0],
    d1 + s0 in [0, 45],
    is_simplified: false
  )");
}

// This test will be updated when the printing uses types of variables.
TEST_F(IndexingMapSerializationTest, CustomNames) {
  auto indexing_map_str = R"(
    (th_x, bl_x)[vector_elem, reduced_dim, contracted_dim]
      -> (contracted_dim, th_x + bl_x, reduced_dim, vector_elem),
    domain:
    th_x in [0, 3],
    bl_x in [0, 4],
    vector_elem in [0, 1],
    reduced_dim in [0, 1],
    contracted_dim in [0, 3],
    th_x mod 4 in [0, 0],
    bl_x + vector_elem in [0, 45],
    is_simplified: false
  )";
  auto indexing_map_golden = R"(
    (d0, d1)[s0, s1, s2] -> (s2, d0 + d1, s1, s0),
    domain:
    d0 in [0, 3],
    d1 in [0, 4],
    s0 in [0, 1],
    s1 in [0, 1],
    s2 in [0, 3],
    d0 mod 4 in [0, 0],
    d1 + s0 in [0, 45],
    is_simplified: false
  )";
  auto indexing_map = ParseIndexingMap(indexing_map_str, &mlir_context_);
  ASSERT_TRUE(indexing_map.has_value());
  EXPECT_THAT(indexing_map->ToString(),
              MatchIndexingString(indexing_map_golden));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
