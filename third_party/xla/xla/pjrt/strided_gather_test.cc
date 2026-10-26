/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/pjrt/strided_gather.h"

#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xla/pjrt/transpose.h"

namespace xla {
namespace {

using ::testing::ElementsAreArray;

struct StridedGatherTestParam {
  StridedGather::Spec spec;
  StridedGather::Spec expected_spec;
  int64_t expected_num_chunks;
  int64_t expected_chunk_size;
  bool expected_is_identity;
  std::vector<StridedGather::Chunk> expected_chunks;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const StridedGatherTestParam& param) {
    absl::Format(&sink,
                 "{spec: %v, expected_spec: %v, expected_num_chunks: %d, "
                 "expected_chunk_size: %d, expected_is_identity: %v, "
                 "expected_chunks: [%s]}",
                 param.spec, param.expected_spec, param.expected_num_chunks,
                 param.expected_chunk_size, param.expected_is_identity,
                 absl::StrJoin(param.expected_chunks, ", "));
  }
};

class StridedGatherTest
    : public testing::TestWithParam<StridedGatherTestParam> {};

TEST_P(StridedGatherTest, StridedGather) {
  const StridedGatherTestParam& param = GetParam();
  StridedGather gather(param.spec);

  EXPECT_EQ(gather.spec(), param.expected_spec);
  EXPECT_EQ(gather.num_chunks(), param.expected_num_chunks);
  EXPECT_EQ(gather.chunk_size(), param.expected_chunk_size);
  EXPECT_EQ(gather.IsIdentity(), param.expected_is_identity);
  EXPECT_THAT(gather.chunks(), ElementsAreArray(param.expected_chunks));

  std::vector<uint8_t> src(param.spec.src_byte_size);
  std::iota(src.begin(), src.end(), 0);

  std::vector<uint8_t> chunk_result;
  for (const StridedGather::Chunk& chunk : gather.chunks()) {
    chunk_result.insert(chunk_result.end(), src.data() + chunk.offset,
                        src.data() + chunk.offset + chunk.size);
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<TransposePlan> plan,
                       gather.CreateTransposePlan(TransposePlan::Create));
  std::vector<uint8_t> transpose_result(gather.num_chunks() *
                                        gather.chunk_size());
  plan->Execute(src.data() + gather.spec().src_byte_offset,
                transpose_result.data());
  EXPECT_EQ(transpose_result, chunk_result);
}

INSTANTIATE_TEST_SUITE_P(
    StridedGather, StridedGatherTest,
    testing::Values(
        StridedGatherTestParam{
            /*spec=*/
            {
                /*element_size=*/4,
                /*counts=*/{4, 8},
                /*src_byte_offset=*/0,
                /*src_byte_size=*/128,
                /*src_byte_strides=*/{32, 4},
            },
            /*expected_spec=*/
            {
                /*element_size=*/128,
                /*counts=*/{},
                /*src_byte_offset=*/0,
                /*src_byte_size=*/128,
                /*src_byte_strides=*/{},
            },
            /*expected_num_chunks=*/1,
            /*expected_chunk_size=*/128,
            /*expected_is_identity=*/true,
            /*expected_chunks=*/
            {
                StridedGather::Chunk{/*offset=*/0, /*size=*/128},
            },
        },
        StridedGatherTestParam{
            /*spec=*/
            {
                /*element_size=*/4,
                /*counts=*/{1, 2, 2, 3},
                /*src_byte_offset=*/8,
                /*src_byte_size=*/200,
                /*src_byte_strides=*/{200, 40, 12, 4},
            },
            /*expected_spec=*/
            {
                /*element_size=*/24,
                /*counts=*/{2},
                /*src_byte_offset=*/8,
                /*src_byte_size=*/200,
                /*src_byte_strides=*/{40},
            },
            /*expected_num_chunks=*/2,
            /*expected_chunk_size=*/24,
            /*expected_is_identity=*/false,
            /*expected_chunks=*/
            {
                StridedGather::Chunk{/*offset=*/8, /*size=*/24},
                StridedGather::Chunk{/*offset=*/48, /*size=*/24},
            },
        },
        StridedGatherTestParam{
            /*spec=*/
            {
                /*element_size=*/4,
                /*counts=*/{2, 2, 3},
                /*src_byte_offset=*/0,
                /*src_byte_size=*/144,
                /*src_byte_strides=*/{8, 72, 24},
            },
            /*expected_spec=*/
            {
                /*element_size=*/4,
                /*counts=*/{2, 6},
                /*src_byte_offset=*/0,
                /*src_byte_size=*/144,
                /*src_byte_strides=*/{8, 24},
            },
            /*expected_num_chunks=*/12,
            /*expected_chunk_size=*/4,
            /*expected_is_identity=*/false,
            /*expected_chunks=*/
            {
                StridedGather::Chunk{/*offset=*/0, /*size=*/4},
                StridedGather::Chunk{/*offset=*/24, /*size=*/4},
                StridedGather::Chunk{/*offset=*/48, /*size=*/4},
                StridedGather::Chunk{/*offset=*/72, /*size=*/4},
                StridedGather::Chunk{/*offset=*/96, /*size=*/4},
                StridedGather::Chunk{/*offset=*/120, /*size=*/4},
                StridedGather::Chunk{/*offset=*/8, /*size=*/4},
                StridedGather::Chunk{/*offset=*/32, /*size=*/4},
                StridedGather::Chunk{/*offset=*/56, /*size=*/4},
                StridedGather::Chunk{/*offset=*/80, /*size=*/4},
                StridedGather::Chunk{/*offset=*/104, /*size=*/4},
                StridedGather::Chunk{/*offset=*/128, /*size=*/4},
            },
        },
        StridedGatherTestParam{
            /*spec=*/
            {
                /*element_size=*/4,
                /*counts=*/{0, 4},
                /*src_byte_offset=*/0,
                /*src_byte_size=*/0,
                /*src_byte_strides=*/{16, 4},
            },
            /*expected_spec=*/
            {
                /*element_size=*/0,
                /*counts=*/{0},
                /*src_byte_offset=*/0,
                /*src_byte_size=*/0,
                /*src_byte_strides=*/{0},
            },
            /*expected_num_chunks=*/0,
            /*expected_chunk_size=*/0,
            /*expected_is_identity=*/true,
            /*expected_chunks=*/{},
        },
        StridedGatherTestParam{
            /*spec=*/
            {
                /*element_size=*/4,
                /*counts=*/{0, 4},
                /*src_byte_offset=*/16,
                /*src_byte_size=*/64,
                /*src_byte_strides=*/{16, 4},
            },
            /*expected_spec=*/
            {
                /*element_size=*/0,
                /*counts=*/{0},
                /*src_byte_offset=*/0,
                /*src_byte_size=*/-1,
                /*src_byte_strides=*/{0},
            },
            /*expected_num_chunks=*/0,
            /*expected_chunk_size=*/0,
            /*expected_is_identity=*/false,
            /*expected_chunks=*/{},
        },
        StridedGatherTestParam{
            /*spec=*/
            {
                /*element_size=*/0,
                /*counts=*/{4},
                /*src_byte_offset=*/16,
                /*src_byte_size=*/64,
                /*src_byte_strides=*/{0},
            },
            /*expected_spec=*/
            {
                /*element_size=*/0,
                /*counts=*/{0},
                /*src_byte_offset=*/0,
                /*src_byte_size=*/-1,
                /*src_byte_strides=*/{0},
            },
            /*expected_num_chunks=*/0,
            /*expected_chunk_size=*/0,
            /*expected_is_identity=*/false,
            /*expected_chunks=*/{},
        },
        StridedGatherTestParam{
            /*spec=*/
            {
                /*element_size=*/4,
                /*counts=*/{1, 1},
                /*src_byte_offset=*/0,
                /*src_byte_size=*/4,
                /*src_byte_strides=*/{4, 4},
            },
            /*expected_spec=*/
            {
                /*element_size=*/4,
                /*counts=*/{},
                /*src_byte_offset=*/0,
                /*src_byte_size=*/4,
                /*src_byte_strides=*/{},
            },
            /*expected_num_chunks=*/1,
            /*expected_chunk_size=*/4,
            /*expected_is_identity=*/true,
            /*expected_chunks=*/
            {
                StridedGather::Chunk{/*offset=*/0, /*size=*/4},
            },
        }));

}  // namespace
}  // namespace xla
