/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/gpu/matmul_indexing_utils.h"

#include "absl/strings/string_view.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/shape.h"
#include "xla/test.h"
#include "tsl/platform/status_matchers.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;
using ::tsl::testing::IsOkAndHolds;

TEST(GetNonContractingDimsTest, Valid) {
  Shape shape = ParseShape("f32[1,2,3,4,5,6]").value();
  EXPECT_THAT(GetNonContractingDims(shape, /*batch_dims=*/{4},
                                    /*contracting_dims=*/{1, 5}),
              IsOkAndHolds(ElementsAre(0, 2, 3)));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
