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

#include "xla/service/spmd/shardy/utils.h"

#include <string>

#include <gtest/gtest.h>
#include "llvm/ADT/BitVector.h"

namespace xla {
namespace sdy {

TEST(UtilsTest, DuplicateShardingsAtIndices) {
  std::string inputShardings =
      "#sdy.sharding_per_value<["
      "<@mesh, [{\"x\"}, {}]>, <@mesh, [{}, {\"y\"}]>, "
      "<@mesh, [{}, {}]>, <@mesh, [{\"x\"}, {\"y\"}]>]>";
  llvm::BitVector indicesToDuplicate(4);
  EXPECT_EQ(duplicateShardingsAtIndices(inputShardings, indicesToDuplicate),
            inputShardings);

  indicesToDuplicate.set(0);
  EXPECT_EQ(duplicateShardingsAtIndices(inputShardings, indicesToDuplicate),
            "#sdy.sharding_per_value<["
            "<@mesh, [{\"x\"}, {}]>, <@mesh, [{\"x\"}, {}]>, "
            "<@mesh, [{}, {\"y\"}]>, <@mesh, [{}, {}]>, "
            "<@mesh, [{\"x\"}, {\"y\"}]>]>");

  indicesToDuplicate.reset();
  indicesToDuplicate.set(1);
  indicesToDuplicate.set(3);
  EXPECT_EQ(duplicateShardingsAtIndices(inputShardings, indicesToDuplicate),
            "#sdy.sharding_per_value<["
            "<@mesh, [{\"x\"}, {}]>, <@mesh, [{}, {\"y\"}]>, "
            "<@mesh, [{}, {\"y\"}]>, <@mesh, [{}, {}]>, "
            "<@mesh, [{\"x\"}, {\"y\"}]>, <@mesh, [{\"x\"}, {\"y\"}]>]>");

  indicesToDuplicate.reset();
  indicesToDuplicate.set(1);
  indicesToDuplicate.set(2);
  EXPECT_EQ(duplicateShardingsAtIndices(inputShardings, indicesToDuplicate),
            "#sdy.sharding_per_value<["
            "<@mesh, [{\"x\"}, {}]>, <@mesh, [{}, {\"y\"}]>, "
            "<@mesh, [{}, {\"y\"}]>, <@mesh, [{}, {}]>, "
            "<@mesh, [{}, {}]>, <@mesh, [{\"x\"}, {\"y\"}]>]>");
}

}  // namespace sdy
}  // namespace xla
