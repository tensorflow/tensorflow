/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/model/tile_analysis.h"

#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Property;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

using TileAnalysisTest = HloTestBase;

TEST_F(TileAnalysisTest, ComposeTilesSucceeds) {
  Tile producer{/*offsets=*/{16, 256}, /*sizes=*/{16, 32}};
  Tile consumer{/*offsets=*/{2, 8}, /*sizes=*/{1, 7}};
  EXPECT_THAT(ComposeTiles(producer, consumer),
              IsOkAndHolds(AllOf(Property(&Tile::offsets, ElementsAre(18, 264)),
                                 Property(&Tile::sizes, ElementsAre(1, 7)))));
}

TEST_F(TileAnalysisTest, ComposeTilesDimensionalityMismatch) {
  Tile producer{/*offsets=*/{16}, /*sizes=*/{16}};
  Tile consumer{/*offsets=*/{2, 8}, /*sizes=*/{1, 7}};
  EXPECT_THAT(ComposeTiles(producer, consumer),
              StatusIs(tsl::error::INTERNAL, HasSubstr("Tile rank mismatch")));
}

TEST_F(TileAnalysisTest, ComposeTilesConsumerTileLargerThanProducer) {
  Tile producer{/*offsets=*/{16, 256}, /*sizes=*/{16, 32}};
  Tile consumer{/*offsets=*/{2, 8}, /*sizes=*/{15, 7}};
  EXPECT_THAT(ComposeTiles(producer, consumer),
              StatusIs(tsl::error::INTERNAL,
                       HasSubstr("Composition leads to an OOB tile")));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
