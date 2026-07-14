// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/python/ifrt_proxy/integration_tests/scoped_pjrt_cpu_via_proxy.h"

int main(int argc, char** argv) {
  const std::string disabled[] = {
      // PjRtCpuBuffer::ToLiteral() currently does not respect the layout of the
      // destination literal.
      "ArrayImplTest.MakeArrayFromHostBufferAndCopyToHostBufferWithByteStrides",

      // Arrays created using `MakeArraysFromHostBufferShards()` do not indicate
      // correct custom layouts even if the given layout is a concrete default
      // layout. PjRt-IFRT uses `ClientMakeArraysFromHostBufferShards()`
      // internally, which lowers `MakeArraysFromHostBufferShards()` call into
      // legacy API calls that do not yet support custom layouts, and thus the
      // output arrays only can have default layouts.
      "ArrayImplTest.MakeArraysFromHostBufferShardsWithLayout",

      // `ShardingParamSharding` does not support serialization yet.
      // TODO(b/282757875): Enable the test once IFRT implements
      // `ShardingParamShardingSerDes`.
      "ArrayImplTest.AssembleAndDisassembleArray",
  };

  const std::string filter = absl::StrCat("-", absl::StrJoin(disabled, ":"));
#ifdef GTEST_FLAG_SET
  GTEST_FLAG_SET(filter, filter.c_str());
#else
  testing::GTEST_FLAG(filter) = filter.c_str();
#endif

  testing::InitGoogleTest(&argc, argv);
  xla::ifrt::proxy::test_util::ScopedPjRtCpuViaProxy scoped_pjrt_cpu_via_proxy;
  return RUN_ALL_TESTS();
}
