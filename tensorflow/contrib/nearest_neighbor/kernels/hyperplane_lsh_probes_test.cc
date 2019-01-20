/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/nearest_neighbor/kernels/hyperplane_lsh_probes.h"

#include <vector>

#include "tensorflow/core/kernels/ops_testutil.h"

namespace {

using tensorflow::uint32;

typedef tensorflow::nearest_neighbor::HyperplaneMultiprobe<float, uint32>
    Multiprobe;

void CheckSequenceSingleTable(Multiprobe* multiprobe,
                              const std::vector<uint32>& expected_probes) {
  uint32 cur_probe;
  int_fast32_t cur_table;
  for (int ii = 0; ii < expected_probes.size(); ++ii) {
    ASSERT_TRUE(multiprobe->GetNextProbe(&cur_probe, &cur_table));
    EXPECT_EQ(expected_probes[ii], cur_probe);
    EXPECT_EQ(0, cur_table);
  }
}

void CheckSequenceMultipleTables(
    Multiprobe* multiprobe,
    const std::vector<std::pair<uint32, int_fast32_t>>& expected_result) {
  uint32 cur_probe;
  int_fast32_t cur_table;
  for (int ii = 0; ii < expected_result.size(); ++ii) {
    ASSERT_TRUE(multiprobe->GetNextProbe(&cur_probe, &cur_table));
    EXPECT_EQ(expected_result[ii].first, cur_probe);
    EXPECT_EQ(expected_result[ii].second, cur_table);
  }
}

// Just the first two probes for two tables and two hyperplanes pro table.
TEST(HyperplaneMultiprobeTest, SimpleTest1) {
  Multiprobe multiprobe(2, 2);
  Multiprobe::Vector hash_vector(4);
  hash_vector << -1.0, 1.0, 1.0, -1.0;
  std::vector<std::pair<uint32, int_fast32_t>> expected_result = {{1, 0},
                                                                  {2, 1}};
  multiprobe.SetupProbing(hash_vector, expected_result.size());
  CheckSequenceMultipleTables(&multiprobe, expected_result);
}

// Checking that the beginning of a probing sequence for a single table is
// correct.
TEST(HyperplaneMultiprobeTest, SimpleTest2) {
  Multiprobe multiprobe(4, 1);
  Multiprobe::Vector hash_vector(4);
  hash_vector << -2.0, -0.9, -0.8, -0.7;
  std::vector<uint32> expected_result = {0, 1, 2, 4, 3};
  multiprobe.SetupProbing(hash_vector, expected_result.size());
  CheckSequenceSingleTable(&multiprobe, expected_result);
}

// Checking that the probing sequence for a single table is exhaustive.
TEST(HyperplaneMultiprobeTest, SimpleTest3) {
  Multiprobe multiprobe(3, 1);
  Multiprobe::Vector hash_vector(3);
  hash_vector << -1.0, -10.0, -0.1;
  std::vector<uint32> expected_result = {0, 1, 4, 5, 2, 3, 6, 7};
  multiprobe.SetupProbing(hash_vector, expected_result.size());
  CheckSequenceSingleTable(&multiprobe, expected_result);
}

// Checking that the probing sequence is generated correctly across tables.
TEST(HyperplaneMultiprobeTest, SimpleTest4) {
  Multiprobe multiprobe(2, 2);
  Multiprobe::Vector hash_vector(4);
  hash_vector << -0.2, 0.9, 0.1, -1.0;
  std::vector<std::pair<uint32, int_fast32_t>> expected_result = {
      {1, 0}, {2, 1}, {0, 1}, {3, 0}, {0, 0}, {2, 0}, {3, 1}, {1, 1}};
  multiprobe.SetupProbing(hash_vector, expected_result.size());
  CheckSequenceMultipleTables(&multiprobe, expected_result);
}

// Slightly larger test that checks whether we have an exhaustive probing
// sequence (but this test does not check the order).
TEST(HyperplaneMultiprobeTest, ExhaustiveTest1) {
  int dim = 8;
  int num_tables = 10;
  Multiprobe multiprobe(dim, num_tables);
  Multiprobe::Vector hash_vector(dim * num_tables);

  std::mt19937 random_generator(487344882);
  std::normal_distribution<float> distribution(0.0, 1.0);
  for (int ii = 0; ii < dim * num_tables; ++ii) {
    hash_vector[ii] = distribution(random_generator);
  }

  std::vector<std::vector<bool>> checked_cell(num_tables);
  for (int ii = 0; ii < num_tables; ++ii) {
    checked_cell[ii].resize(1 << dim);
    std::fill(checked_cell[ii].begin(), checked_cell[ii].end(), false);
  }

  int num_probes = (1 << dim) * num_tables;
  multiprobe.SetupProbing(hash_vector, num_probes);
  uint32 cur_probe;
  int_fast32_t cur_table;
  for (int ii = 0; ii < num_probes; ++ii) {
    ASSERT_TRUE(multiprobe.GetNextProbe(&cur_probe, &cur_table));
    ASSERT_LE(0, cur_probe);
    ASSERT_LT(cur_probe, 1 << dim);
    ASSERT_LE(0, cur_table);
    ASSERT_LT(cur_table, num_tables);
    EXPECT_FALSE(checked_cell[cur_table][cur_probe]);
    checked_cell[cur_table][cur_probe] = true;
  }
}

}  // namespace
