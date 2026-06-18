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
#include "tensorflow/lite/experimental/ml_adjacent/data/owning_vector_ref.h"

#include <algorithm>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/ml_adjacent/lib.h"

namespace ml_adj {
namespace data {
namespace {

using ::testing::ElementsAreArray;
using ::testing::IsEmpty;

TEST(OwningVectorRefTest, ConstructFloat32) {
  OwningVectorRef t(etype_t::f32);

  EXPECT_EQ(t.Type(), etype_t::f32);
  EXPECT_EQ(t.NumElements(), 0);
  EXPECT_EQ(t.Bytes(), 0);
  EXPECT_THAT(t.Dims(), IsEmpty());
}

TEST(OwningVectorRefTest, ResizeFromEmptyFloat32) {
  OwningVectorRef t(etype_t::f32);
  t.Resize({2, 2});

  EXPECT_THAT(t.Dims(), ElementsAreArray<dim_t>({2, 2}));
  EXPECT_EQ(t.NumElements(), 4);
  ASSERT_EQ(t.Bytes(), 4 * sizeof(float));

  // Check buffer is correct.
  float* write_f_start = reinterpret_cast<float*>(t.Data());
  float* write_f_end = write_f_start + t.NumElements();
  std::fill(write_f_start, write_f_end, 0.5f);

  const float* read_f_start = reinterpret_cast<const float*>(t.Data());
  for (int i = 0; i < t.NumElements(); ++i) {
    EXPECT_EQ(read_f_start[i], 0.5f);
  }
}

TEST(OwningVectorRefTest, ResizeDownFloat32) {
  OwningVectorRef t(etype_t::f32);
  t.Resize({2, 2});

  float* write_f_start = reinterpret_cast<float*>(t.Data());
  float* write_f_end = write_f_start + t.NumElements();
  std::fill(write_f_start, write_f_end, 0.5f);

  t.Resize({3});
  ASSERT_THAT(t.Dims(), ElementsAreArray<dim_t>({3}));
  EXPECT_EQ(t.NumElements(), 3);
  ASSERT_EQ(t.Bytes(), 3 * sizeof(float));

  const float* read_f_start = reinterpret_cast<const float*>(t.Data());
  for (int i = 0; i < t.NumElements(); ++i) {
    EXPECT_EQ(read_f_start[i], 0.5f);
  }
}

TEST(OwningVectorRefTest, IgnoresDimsForNumElementsAfterFirstNonPositive) {
  OwningVectorRef t(etype_t::f32);
  t.Resize({3, 0, 0, 2});

  EXPECT_EQ(t.Type(), etype_t::f32);
  EXPECT_EQ(t.NumElements(), 3);
  EXPECT_EQ(t.Bytes(), 3 * sizeof(float));
  EXPECT_THAT(t.Dims(), ElementsAreArray<dim_t>({3, 0, 0, 2}));
}

}  // namespace
}  // namespace data
}  // namespace ml_adj
