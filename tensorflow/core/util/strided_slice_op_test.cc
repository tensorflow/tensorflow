/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/strided_slice_op.h"

#include <algorithm>
#include <tuple>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status_matchers.h"

namespace tensorflow {
namespace {

using ::testing::PrintToString;

using Vec = typename StridedSliceAssignBCast::Vec;
struct BroadcastPair {
  Vec from;
  Vec to;

  friend std::ostream& operator<<(std::ostream& os, const BroadcastPair& pair) {
    return os << strings::StrCat("BroadcastPair{", PrintToString(pair.from),
                                 "->", PrintToString(pair.to), "}");
  }
};

struct BroadcastRemap {
  int64_t dims;
  Vec map;

  friend std::ostream& operator<<(std::ostream& os,
                                  const BroadcastRemap& remap) {
    return os << strings::StrCat("BroadcastRemap{", remap.dims, ", ",
                                 PrintToString(remap.map), "}");
  }
};

int64_t NumberOfElements(const Vec& shape) {
  int64_t number_of_elements = 1;
  for (int64_t elem : shape) {
    number_of_elements *= elem;
  }
  return number_of_elements;
}

MATCHER_P2(Broadcasts, input_shape, output_shape,
           strings::StrCat("broadcasts ", PrintToString(input_shape), " to ",
                           PrintToString(output_shape))) {
  const size_t size = input_shape.size();
  for (size_t i = 0; i < size; ++i) {
    if (!((arg[i] == 1 && input_shape[i] == output_shape[i]) ||
          (arg[i] == output_shape[i] && input_shape[i] == 1))) {
      return false;
    }
  }
  return true;
}

MATCHER_P(HasSuffix, suffix, "") {
  const size_t offset = arg.size() - suffix.size();
  for (size_t i = 0; i < suffix.size(); ++i) {
    if (suffix[i] != arg[i + offset]) {
      return false;
    }
  }
  return true;
}

MATCHER_P(HasSameNumberOfElementsAs, other, "") {
  return NumberOfElements(arg) == NumberOfElements(other);
}

TEST(StridedSliceAssignBCastTest, BroadcastingToSameRankWorks) {
  const BroadcastPair test_pairs[] = {
      {/*from=*/Vec{1}, /*to=*/Vec{5}},
      {/*from=*/Vec{1, 1}, /*to=*/Vec{4, 5}},
      {/*from=*/Vec{1, 5}, /*to=*/Vec{4, 5}},
      {/*from=*/Vec{4, 1}, /*to=*/Vec{4, 5}},
      {/*from=*/Vec{1, 1, 1}, /*to=*/Vec{2, 4, 5}},
      {/*from=*/Vec{1, 1, 5}, /*to=*/Vec{2, 4, 5}},
      {/*from=*/Vec{1, 4, 5}, /*to=*/Vec{2, 4, 5}},
      {/*from=*/Vec{2, 1, 5}, /*to=*/Vec{2, 4, 5}},
      {/*from=*/Vec{2, 4, 1}, /*to=*/Vec{2, 4, 5}},
  };
  for (const BroadcastPair& test_pair : test_pairs) {
    StridedSliceAssignBCast bcast(test_pair.from, test_pair.to);
    EXPECT_TRUE(bcast.IsValid()) << test_pair;
    EXPECT_TRUE(bcast.IsBroadcastingRequired());
    EXPECT_EQ(bcast.result_shape(), test_pair.to);
    EXPECT_EQ(bcast.reshape(), test_pair.from);
    EXPECT_THAT(bcast.bcast(), Broadcasts(test_pair.from, test_pair.to));
  }
}

TEST(StridedSliceAssignBCastTest, BroadcastingToLargerRankWorks) {
  const BroadcastPair test_pairs[] = {
      {/*from=*/Vec{}, /*to=*/Vec{2, 4, 5}},
      {/*from=*/Vec{1}, /*to=*/Vec{2, 4, 5}},
      {/*from=*/Vec{5}, /*to=*/Vec{2, 4, 5}},
      {/*from=*/Vec{1, 1}, /*to=*/Vec{2, 4, 5}},
      {/*from=*/Vec{1, 5}, /*to=*/Vec{2, 4, 5}},
      {/*from=*/Vec{4, 1}, /*to=*/Vec{2, 4, 5}},
      {/*from=*/Vec{4, 5}, /*to=*/Vec{2, 4, 5}},
  };
  for (const BroadcastPair& test_pair : test_pairs) {
    StridedSliceAssignBCast bcast(test_pair.from, test_pair.to);
    EXPECT_TRUE(bcast.IsValid()) << test_pair;
    EXPECT_TRUE(bcast.IsBroadcastingRequired());
    EXPECT_EQ(bcast.result_shape(), test_pair.to);
    EXPECT_THAT(bcast.reshape(), HasSuffix(test_pair.from));
    EXPECT_THAT(bcast.reshape(), HasSameNumberOfElementsAs(test_pair.from));
    EXPECT_THAT(bcast.bcast(), Broadcasts(bcast.reshape(), test_pair.to));
  }
}

TEST(StridedSliceAssignBCastTest, BroadcastingToSmallerRankWorks) {
  const BroadcastPair test_pairs[] = {
      {/*from=*/Vec{1, 1}, /*to=*/Vec{5}},
      {/*from=*/Vec{1, 1, 5}, /*to=*/Vec{4, 5}},
      {/*from=*/Vec{1, 4, 1}, /*to=*/Vec{4, 5}},
      {/*from=*/Vec{1, 1, 1, 5}, /*to=*/Vec{4, 5}},
      {/*from=*/Vec{1, 1, 4, 1}, /*to=*/Vec{4, 5}},
  };
  for (const BroadcastPair& test_pair : test_pairs) {
    StridedSliceAssignBCast bcast(test_pair.from, test_pair.to);
    EXPECT_TRUE(bcast.IsValid()) << test_pair;
    EXPECT_TRUE(bcast.IsBroadcastingRequired());
    EXPECT_EQ(bcast.result_shape(), test_pair.to);
    EXPECT_THAT(test_pair.from, HasSuffix(bcast.reshape()));
    EXPECT_THAT(bcast.reshape(), HasSameNumberOfElementsAs(test_pair.from));
    EXPECT_THAT(bcast.bcast(), Broadcasts(bcast.reshape(), test_pair.to));
  }
}

TEST(StridedSliceAssignBCastTest, ReshapeOnlyWorks) {
  // Same shape or one is prefixed by ones.
  const BroadcastPair test_pairs[] = {
      {/*from=*/Vec{}, /*to=*/Vec{1, 1}},
      {/*from=*/Vec{5}, /*to=*/Vec{5}},
      {/*from=*/Vec{5}, /*to=*/Vec{1, 5}},
      {/*from=*/Vec{1, 1}, /*to=*/Vec{}},
      {/*from=*/Vec{1, 5}, /*to=*/Vec{5}},
      {/*from=*/Vec{2, 4, 5}, /*to=*/Vec{2, 4, 5}},
      {/*from=*/Vec{2, 4, 5}, /*to=*/Vec{1, 1, 1, 2, 4, 5}},
      {/*from=*/Vec{1, 1, 1, 2, 4, 5}, /*to=*/Vec{2, 4, 5}},
  };
  for (const BroadcastPair& test_pair : test_pairs) {
    StridedSliceAssignBCast bcast(test_pair.from, test_pair.to);
    EXPECT_TRUE(bcast.IsValid()) << test_pair;
    EXPECT_FALSE(bcast.IsBroadcastingRequired());
    EXPECT_EQ(bcast.result_shape(), test_pair.to);
    EXPECT_THAT(bcast.reshape(), HasSameNumberOfElementsAs(test_pair.from));
    EXPECT_THAT(bcast.bcast(), Broadcasts(bcast.reshape(), test_pair.to));
  }
}

TEST(StridedSliceAssignBCastTest, InvalidBroadcastFails) {
  const BroadcastPair test_pairs[] = {
      {/*from=*/Vec{5}, /*to=*/Vec{1}},
      {/*from=*/Vec{3}, /*to=*/Vec{4, 5}},
      {/*from=*/Vec{4}, /*to=*/Vec{4, 5}},
      {/*from=*/Vec{5}, /*to=*/Vec{}},

      {/*from=*/Vec{3, 5}, /*to=*/Vec{4, 5}},
      {/*from=*/Vec{4, 3}, /*to=*/Vec{4, 5}},
      {/*from=*/Vec{5, 5}, /*to=*/Vec{1, 5}},
      {/*from=*/Vec{2, 4}, /*to=*/Vec{2, 4, 5}},
      {/*from=*/Vec{4, 3}, /*to=*/Vec{2, 4, 5}},
      {/*from=*/Vec{3, 5}, /*to=*/Vec{2, 4, 5}},
      {/*from=*/Vec{3, 5}, /*to=*/Vec{5}},
      {/*from=*/Vec{3, 5}, /*to=*/Vec{}},

      {/*from=*/Vec{3, 4, 5}, /*to=*/Vec{2, 4, 5}},
      {/*from=*/Vec{2, 4, 5}, /*to=*/Vec{1, 4, 5}},
      {/*from=*/Vec{2, 3, 5}, /*to=*/Vec{2, 4, 5}},
      {/*from=*/Vec{2, 4, 5}, /*to=*/Vec{2, 4, 5, 2}},
      {/*from=*/Vec{2, 4, 5}, /*to=*/Vec{2, 4, 5, 1}},
      {/*from=*/Vec{2, 4, 5}, /*to=*/Vec{2, 4, 1, 5}},
      {/*from=*/Vec{2, 4, 5}, /*to=*/Vec{4, 5}},
      {/*from=*/Vec{2, 4, 5}, /*to=*/Vec{2, 4}},
      {/*from=*/Vec{1, 4, 5}, /*to=*/Vec{4, 1}},
      {/*from=*/Vec{1, 4, 5}, /*to=*/Vec{5}},
      {/*from=*/Vec{1, 4, 5}, /*to=*/Vec{}},
  };
  for (const BroadcastPair& test_pair : test_pairs) {
    StridedSliceAssignBCast bcast(test_pair.from, test_pair.to);
    EXPECT_FALSE(bcast.IsValid()) << test_pair;
  }
}

TEST(StridedSliceAssignBCastTest, RemapDimensionsToItselfWorks) {
  const std::pair<BroadcastPair, BroadcastRemap> test_inputs[] = {
      {BroadcastPair{/*from=*/Vec{}, /*to=*/Vec{}},
       BroadcastRemap{/*dims=*/0, /*map=*/Vec{}}},
      {BroadcastPair{/*from=*/Vec{4, 5}, /*to=*/Vec{4, 5}},
       BroadcastRemap{/*dims=*/2, /*map=*/Vec{0, 1}}},
      {BroadcastPair{/*from=*/Vec{2, 4, 5}, /*to=*/Vec{2, 4, 5}},
       BroadcastRemap{/*dims=*/3, /*map=*/Vec{0, 1, 2}}},
  };
  for (const auto& test_input : test_inputs) {
    const BroadcastPair& test_pair = test_input.first;
    const BroadcastRemap& test_remap = test_input.second;
    StridedSliceAssignBCast bcast(test_pair.from, test_pair.to);
    ASSERT_TRUE(bcast.IsValid());
    EXPECT_TRUE(bcast.RemapDimensions(test_remap.dims, test_remap.map))
        << PrintToString(test_input);
    EXPECT_EQ(bcast.result_shape(), test_pair.to);
    EXPECT_THAT(bcast.bcast(),
                Broadcasts(bcast.reshape(), bcast.result_shape()));
  }
}

TEST(StridedSliceAssignBCastTest, RemapDimensionsRemovingAxesWorks) {
  // Tuples of {broadcast inputs, remapping info, expected result shape}.
  // In all practical cases, only dimensions of size 1 will be removed.
  const std::tuple<BroadcastPair, BroadcastRemap, Vec> test_inputs[] = {
      {BroadcastPair{/*from=*/Vec{2, 1, 4, 1, 5}, /*to=*/Vec{2, 1, 4, 1, 5}},
       BroadcastRemap{/*dims=*/3, /*map=*/Vec{0, -1, 1, -1, 2}}, Vec{2, 4, 5}},
      {BroadcastPair{/*from=*/Vec{1, 4, 1}, /*to=*/Vec{1, 4, 1}},
       BroadcastRemap{/*dims=*/1, /*map=*/Vec{-1, 0, -1}}, Vec{4}},
      {BroadcastPair{/*from=*/Vec{1, 1, 1}, /*to=*/Vec{1, 1, 1}},
       BroadcastRemap{/*dims=*/0, /*map=*/Vec{-1, -1, -1}}, Vec{}},
  };
  for (const auto& test_input : test_inputs) {
    const BroadcastPair& test_pair = std::get<0>(test_input);
    const BroadcastRemap& test_remap = std::get<1>(test_input);
    const Vec& expected_result_shape = std::get<2>(test_input);
    StridedSliceAssignBCast bcast(test_pair.from, test_pair.to);
    ASSERT_TRUE(bcast.IsValid());
    EXPECT_TRUE(bcast.RemapDimensions(test_remap.dims, test_remap.map))
        << PrintToString(test_input);
    EXPECT_EQ(bcast.result_shape(), expected_result_shape);
    EXPECT_THAT(bcast.bcast(),
                Broadcasts(bcast.reshape(), bcast.result_shape()));
  }
}

TEST(StridedSliceAssignBCastTest, RemapDimensionsAddingAxesWorks) {
  // Tuples of {broadcast inputs, remapping info, expected result shape}.
  const std::tuple<BroadcastPair, BroadcastRemap, Vec> test_inputs[] = {
      {BroadcastPair{/*from=*/Vec{2, 4, 5}, /*to=*/Vec{2, 4, 5}},
       BroadcastRemap{/*dims=*/5, /*map=*/Vec{0, 2, 4}}, Vec{2, 1, 4, 1, 5}},
      {BroadcastPair{/*from=*/Vec{4, 5}, /*to=*/Vec{4, 5}},
       BroadcastRemap{/*dims=*/4, /*map=*/Vec{1, 2}}, Vec{1, 4, 5, 1}},
      {BroadcastPair{/*from=*/Vec{}, /*to=*/Vec{}},
       BroadcastRemap{/*dims=*/3, /*map=*/Vec{}}, Vec{1, 1, 1}},
  };
  for (const auto& test_input : test_inputs) {
    const BroadcastPair& test_pair = std::get<0>(test_input);
    const BroadcastRemap& test_remap = std::get<1>(test_input);
    const Vec& expected_result_shape = std::get<2>(test_input);
    StridedSliceAssignBCast bcast(test_pair.from, test_pair.to);
    ASSERT_TRUE(bcast.IsValid());
    EXPECT_TRUE(bcast.RemapDimensions(test_remap.dims, test_remap.map))
        << PrintToString(test_input);
    EXPECT_EQ(bcast.result_shape(), expected_result_shape);
    EXPECT_THAT(bcast.bcast(),
                Broadcasts(bcast.reshape(), bcast.result_shape()));
  }
}

TEST(StridedSliceAssignBCastTest, RemapDimensionsAddingAndRemovingAxesWorks) {
  // Tuples of {broadcast inputs, remapping info, expected result shape}.
  const std::tuple<BroadcastPair, BroadcastRemap, Vec> test_inputs[] = {
      // Adds and removes dimensions.
      {BroadcastPair{/*from=*/Vec{1}, /*to=*/Vec{1}},
       BroadcastRemap{/*dims=*/1, /*map=*/Vec{-1}}, Vec{1}},
      {BroadcastPair{/*from=*/Vec{1}, /*to=*/Vec{1}},
       BroadcastRemap{/*dims=*/3, /*map=*/Vec{-1}}, Vec{1, 1, 1}},
      {BroadcastPair{/*from=*/Vec{1, 5}, /*to=*/Vec{1, 5}},
       BroadcastRemap{/*dims=*/3, /*map=*/Vec{-1, 1}}, Vec{1, 5, 1}},
      {BroadcastPair{/*from=*/Vec{1, 5}, /*to=*/Vec{2, 1, 4, 1, 5}},
       BroadcastRemap{/*dims=*/4, /*map=*/Vec{0, -1, 1, -1, 3}},
       Vec{2, 4, 1, 5}},
  };
  for (const auto& test_input : test_inputs) {
    const BroadcastPair& test_pair = std::get<0>(test_input);
    const BroadcastRemap& test_remap = std::get<1>(test_input);
    const Vec& expected_result_shape = std::get<2>(test_input);
    StridedSliceAssignBCast bcast(test_pair.from, test_pair.to);
    ASSERT_TRUE(bcast.IsValid());
    EXPECT_TRUE(bcast.RemapDimensions(test_remap.dims, test_remap.map))
        << PrintToString(test_input);
    EXPECT_EQ(bcast.result_shape(), expected_result_shape);
    EXPECT_THAT(bcast.bcast(),
                Broadcasts(bcast.reshape(), bcast.result_shape()));
  }
}

TEST(StridedSliceAssignBCastTest, RemapDimensionsInvalidSizeFails) {
  const std::pair<BroadcastPair, BroadcastRemap> test_inputs[] = {
      // Map size must equal target `to` size.
      {BroadcastPair{/*from=*/Vec{}, /*to=*/Vec{}},
       BroadcastRemap{/*dims=*/0, /*map=*/Vec{-1}}},
      {BroadcastPair{/*from=*/Vec{2, 4, 5}, /*to=*/Vec{2, 4, 5}},
       BroadcastRemap{/*dims=*/3, /*map=*/Vec{0, 1, -1, 2}}},
      {BroadcastPair{/*from=*/Vec{2, 4, 5}, /*to=*/Vec{2, 4, 5}},
       BroadcastRemap{/*dims=*/3, /*map=*/Vec{0, 2}}},
  };
  for (const auto& test_input : test_inputs) {
    const BroadcastPair& test_pair = test_input.first;
    const BroadcastRemap& test_remap = test_input.second;
    StridedSliceAssignBCast bcast(test_pair.from, test_pair.to);
    ASSERT_TRUE(bcast.IsValid());
    EXPECT_FALSE(bcast.RemapDimensions(test_remap.dims, test_remap.map))
        << PrintToString(test_input);
  }
}

TEST(StridedSliceAssignBCastTest, RemapDimensionsOutOfBoundsFails) {
  const std::pair<BroadcastPair, BroadcastRemap> test_inputs[] = {
      // Dimensions must be < dims.
      {BroadcastPair{/*from=*/Vec{2, 4, 5}, /*to=*/Vec{2, 4, 5}},
       BroadcastRemap{/*dims=*/3, /*map=*/Vec{0, 1, 3}}},
      {BroadcastPair{/*from=*/Vec{2, 4, 5}, /*to=*/Vec{2, 4, 5}},
       BroadcastRemap{/*dims=*/2, /*map=*/Vec{0, 1, 2}}},
  };
  for (const auto& test_input : test_inputs) {
    const BroadcastPair& test_pair = test_input.first;
    const BroadcastRemap& test_remap = test_input.second;
    StridedSliceAssignBCast bcast(test_pair.from, test_pair.to);
    ASSERT_TRUE(bcast.IsValid());
    EXPECT_FALSE(bcast.RemapDimensions(test_remap.dims, test_remap.map))
        << PrintToString(test_input);
  }
}

using IntVector = gtl::InlinedVector<int64_t, 4>;

TensorShape AsTensorShape(gtl::ArraySlice<int64_t> dim_sizes) {
  TensorShape out;
  TF_CHECK_OK(TensorShape::BuildTensorShape(dim_sizes, &out));
  return out;
}

TEST(ValidateStridedSliceOpTest, BasicStride) {
  Tensor begin_tensor = test::AsTensor<int32_t>({1, 1});
  Tensor end_tensor = test::AsTensor<int32_t>({7, 7});
  Tensor strides_tensor = test::AsTensor<int32_t>({2, 2});
  TensorShape input_shape = AsTensorShape({10, 10});
  int32_t begin_mask_spec = 0x2;
  int32_t end_mask_spec = 0x1;
  int32_t ellipsis_mask = 0x0;
  int32_t new_axis_mask = 0x0;
  int32_t shrink_axis_mask = 0x0;

  // Function outputs.
  TensorShape processing_shape, final_shape;
  bool is_identity, is_simple_slice, slice_dim0;
  IntVector begin, end, strides;
  StridedSliceShapeSpec shape_spec;

  TF_EXPECT_OK(ValidateStridedSliceOp(
      &begin_tensor, &end_tensor, strides_tensor, input_shape, begin_mask_spec,
      end_mask_spec, ellipsis_mask, new_axis_mask, shrink_axis_mask,
      &processing_shape, &final_shape, &is_identity, &is_simple_slice,
      &slice_dim0, &begin, &end, &strides, &shape_spec));

  EXPECT_EQ(processing_shape, AsTensorShape({5, 4}));
  EXPECT_EQ(final_shape, AsTensorShape({5, 4}));
  EXPECT_FALSE(is_identity);
  EXPECT_FALSE(is_simple_slice);
  EXPECT_FALSE(slice_dim0);
  EXPECT_EQ(begin, (IntVector{1, 0}));
  EXPECT_EQ(end, (IntVector{10, 7}));
  EXPECT_EQ(strides, (IntVector{2, 2}));
}

TEST(ValidateStridedSliceOpTest, NegativeBeginEnd) {
  Tensor begin_tensor = test::AsTensor<int32_t>({-9, /*Will use bounds*/ -20});
  Tensor end_tensor = test::AsTensor<int32_t>({-3, -3});
  Tensor strides_tensor = test::AsTensor<int32_t>({2, 2});
  TensorShape input_shape = AsTensorShape({10, 10});
  int32_t begin_mask_spec = 0x0;
  int32_t end_mask_spec = 0x0;
  int32_t ellipsis_mask = 0x0;
  int32_t new_axis_mask = 0x0;
  int32_t shrink_axis_mask = 0x0;

  // Function outputs.
  TensorShape processing_shape, final_shape;
  bool is_identity, is_simple_slice, slice_dim0;
  IntVector begin, end, strides;
  StridedSliceShapeSpec shape_spec;

  TF_EXPECT_OK(ValidateStridedSliceOp(
      &begin_tensor, &end_tensor, strides_tensor, input_shape, begin_mask_spec,
      end_mask_spec, ellipsis_mask, new_axis_mask, shrink_axis_mask,
      &processing_shape, &final_shape, &is_identity, &is_simple_slice,
      &slice_dim0, &begin, &end, &strides, &shape_spec));

  EXPECT_EQ(processing_shape, AsTensorShape({3, 4}));
  EXPECT_EQ(final_shape, AsTensorShape({3, 4}));
  EXPECT_EQ(begin, (IntVector{1, 0}));
  EXPECT_EQ(end, (IntVector{7, 7}));
}

TEST(ValidateStridedSliceOpTest, EmptyOutputDim) {
  Tensor begin_tensor = test::AsTensor<int32_t>({1, 1});
  Tensor end_tensor = test::AsTensor<int32_t>({7, 1});
  Tensor strides_tensor = test::AsTensor<int32_t>({2, 1});
  TensorShape input_shape = AsTensorShape({10, 10});
  int32_t begin_mask_spec = 0x0;
  int32_t end_mask_spec = 0x0;
  int32_t ellipsis_mask = 0x0;
  int32_t new_axis_mask = 0x0;
  int32_t shrink_axis_mask = 0x0;

  // Function outputs.
  TensorShape processing_shape, final_shape;
  bool is_identity, is_simple_slice, slice_dim0;
  IntVector begin, end, strides;
  StridedSliceShapeSpec shape_spec;

  TF_EXPECT_OK(ValidateStridedSliceOp(
      &begin_tensor, &end_tensor, strides_tensor, input_shape, begin_mask_spec,
      end_mask_spec, ellipsis_mask, new_axis_mask, shrink_axis_mask,
      &processing_shape, &final_shape, &is_identity, &is_simple_slice,
      &slice_dim0, &begin, &end, &strides, &shape_spec));

  EXPECT_EQ(processing_shape, AsTensorShape({3, 0}));
  EXPECT_EQ(final_shape, AsTensorShape({3, 0}));
}

TEST(ValidateStridedSliceOpTest, ZeroStrideFails) {
  Tensor begin_tensor = test::AsTensor<int32_t>({1, 1});
  Tensor end_tensor = test::AsTensor<int32_t>({7, 7});
  Tensor strides_tensor = test::AsTensor<int32_t>({0, 2});
  TensorShape input_shape = AsTensorShape({10, 10});
  int32_t begin_mask_spec = 0x2;
  int32_t end_mask_spec = 0x1;
  int32_t ellipsis_mask = 0x0;
  int32_t new_axis_mask = 0x0;
  int32_t shrink_axis_mask = 0x0;

  // Function outputs.
  TensorShape processing_shape, final_shape;
  bool is_identity, is_simple_slice, slice_dim0;
  IntVector begin, end, strides;
  StridedSliceShapeSpec shape_spec;

  EXPECT_THAT(
      ValidateStridedSliceOp(
          &begin_tensor, &end_tensor, strides_tensor, input_shape,
          begin_mask_spec, end_mask_spec, ellipsis_mask, new_axis_mask,
          shrink_axis_mask, &processing_shape, &final_shape, &is_identity,
          &is_simple_slice, &slice_dim0, &begin, &end, &strides, &shape_spec),
      tsl::testing::StatusIs(
          tsl::error::Code::INVALID_ARGUMENT,
          ::testing::ContainsRegex("strides.* must be non-zero")));
}

TEST(ValidateStridedSliceOpTest, ShrinkAxis) {
  // Shrink middle axis.
  Tensor begin_tensor = test::AsTensor<int16_t>({0, 1, 0});
  Tensor end_tensor = test::AsTensor<int16_t>({3, 1, 5});
  Tensor strides_tensor = test::AsTensor<int16_t>({1, 1, 1});
  TensorShape input_shape = AsTensorShape({3, 4, 5});
  int32_t begin_mask_spec = 0x2;
  int32_t end_mask_spec = 0x2;
  int32_t ellipsis_mask = 0x0;
  int32_t new_axis_mask = 0x0;
  int32_t shrink_axis_mask = 0x2;

  // Function outputs.
  TensorShape processing_shape, final_shape;
  bool is_identity, is_simple_slice, slice_dim0;
  IntVector begin, end, strides;
  StridedSliceShapeSpec shape_spec;

  TF_EXPECT_OK(ValidateStridedSliceOp(
      &begin_tensor, &end_tensor, strides_tensor, input_shape, begin_mask_spec,
      end_mask_spec, ellipsis_mask, new_axis_mask, shrink_axis_mask,
      &processing_shape, &final_shape, &is_identity, &is_simple_slice,
      &slice_dim0, &begin, &end, &strides, &shape_spec));

  EXPECT_EQ(final_shape, AsTensorShape({3, 5}));
}

TEST(ValidateStridedSliceOpTest, ShrinkSliceOutOfBoundsFails) {
  // Shrink middle axis.
  Tensor begin_tensor = test::AsTensor<int16_t>({0, 7, 0});
  Tensor end_tensor = test::AsTensor<int16_t>({3, 7, 5});
  Tensor strides_tensor = test::AsTensor<int16_t>({1, 1, 1});
  TensorShape input_shape = AsTensorShape({3, 4, 5});
  int32_t begin_mask_spec = 0x2;
  int32_t end_mask_spec = 0x2;
  int32_t ellipsis_mask = 0x0;
  int32_t new_axis_mask = 0x0;
  int32_t shrink_axis_mask = 0x2;

  // Function outputs.
  TensorShape processing_shape, final_shape;
  bool is_identity, is_simple_slice, slice_dim0;
  IntVector begin, end, strides;
  StridedSliceShapeSpec shape_spec;

  EXPECT_THAT(
      ValidateStridedSliceOp(
          &begin_tensor, &end_tensor, strides_tensor, input_shape,
          begin_mask_spec, end_mask_spec, ellipsis_mask, new_axis_mask,
          shrink_axis_mask, &processing_shape, &final_shape, &is_identity,
          &is_simple_slice, &slice_dim0, &begin, &end, &strides, &shape_spec),
      tsl::testing::StatusIs(
          tsl::error::Code::INVALID_ARGUMENT,
          ::testing::ContainsRegex("slice index .* out of bounds")));
}

TEST(ValidateStridedSliceOpTest, ShrinkAxisNegativeStrideFails) {
  // Shrink middle axis.
  Tensor begin_tensor = test::AsTensor<int16_t>({0, 1, 0});
  Tensor end_tensor = test::AsTensor<int16_t>({3, 2, 5});
  Tensor strides_tensor = test::AsTensor<int16_t>({1, -1, 1});
  TensorShape input_shape = AsTensorShape({3, 4, 5});
  int32_t begin_mask_spec = 0x2;
  int32_t end_mask_spec = 0x2;
  int32_t ellipsis_mask = 0x0;
  int32_t new_axis_mask = 0x0;
  int32_t shrink_axis_mask = 0x2;

  // Function outputs.
  TensorShape processing_shape, final_shape;
  bool is_identity, is_simple_slice, slice_dim0;
  IntVector begin, end, strides;
  StridedSliceShapeSpec shape_spec;

  EXPECT_THAT(
      ValidateStridedSliceOp(
          &begin_tensor, &end_tensor, strides_tensor, input_shape,
          begin_mask_spec, end_mask_spec, ellipsis_mask, new_axis_mask,
          shrink_axis_mask, &processing_shape, &final_shape, &is_identity,
          &is_simple_slice, &slice_dim0, &begin, &end, &strides, &shape_spec),
      tsl::testing::StatusIs(
          tsl::error::Code::INVALID_ARGUMENT,
          ::testing::ContainsRegex("only stride 1 allowed")));
}

TEST(ValidateStridedSliceOpTest, NewAxis) {
  Tensor begin_tensor = test::AsTensor<int64_t>({0, 0});
  Tensor end_tensor = test::AsTensor<int64_t>({10, 10});
  Tensor strides_tensor = test::AsTensor<int64_t>({1, 1});
  TensorShape input_shape = AsTensorShape({10, 10});
  int32_t begin_mask_spec = 0x0;
  int32_t end_mask_spec = 0x0;
  int32_t ellipsis_mask = 0x0;
  int32_t new_axis_mask = 0x2;
  int32_t shrink_axis_mask = 0x0;

  // Function outputs.
  TensorShape processing_shape, final_shape;
  bool is_identity, is_simple_slice, slice_dim0;
  IntVector begin, end, strides;
  StridedSliceShapeSpec shape_spec;

  TF_EXPECT_OK(ValidateStridedSliceOp(
      &begin_tensor, &end_tensor, strides_tensor, input_shape, begin_mask_spec,
      end_mask_spec, ellipsis_mask, new_axis_mask, shrink_axis_mask,
      &processing_shape, &final_shape, &is_identity, &is_simple_slice,
      &slice_dim0, &begin, &end, &strides, &shape_spec));

  EXPECT_EQ(processing_shape, AsTensorShape({10, 10}));
  EXPECT_EQ(final_shape, AsTensorShape({10, 1, 10}));
}

TEST(ValidateStridedSliceOpTest, Ellipsis) {
  Tensor begin_tensor = test::AsTensor<int32_t>({0, 0});
  Tensor end_tensor = test::AsTensor<int32_t>({10, 10});
  Tensor strides_tensor = test::AsTensor<int32_t>({1, 1});
  TensorShape input_shape = AsTensorShape({10, 10});
  int32_t begin_mask_spec = 0x0;
  int32_t end_mask_spec = 0x0;
  int32_t ellipsis_mask = 0x1;
  int32_t new_axis_mask = 0x2;
  int32_t shrink_axis_mask = 0x0;

  // Function outputs.
  TensorShape processing_shape, final_shape;
  bool is_identity, is_simple_slice, slice_dim0;
  IntVector begin, end, strides;
  StridedSliceShapeSpec shape_spec;

  TF_EXPECT_OK(ValidateStridedSliceOp(
      &begin_tensor, &end_tensor, strides_tensor, input_shape, begin_mask_spec,
      end_mask_spec, ellipsis_mask, new_axis_mask, shrink_axis_mask,
      &processing_shape, &final_shape, &is_identity, &is_simple_slice,
      &slice_dim0, &begin, &end, &strides, &shape_spec));

  EXPECT_EQ(processing_shape, AsTensorShape({10, 10}));
  EXPECT_EQ(final_shape, AsTensorShape({10, 10, 1}));
}

TEST(ValidateStridedSliceOpTest, MultipleEllipsisFails) {
  Tensor begin_tensor = test::AsTensor<int32_t>({0, 0});
  Tensor end_tensor = test::AsTensor<int32_t>({10, 10});
  Tensor strides_tensor = test::AsTensor<int32_t>({1, 1});
  TensorShape input_shape = AsTensorShape({10, 10});
  int32_t begin_mask_spec = 0x0;
  int32_t end_mask_spec = 0x0;
  int32_t ellipsis_mask = 0x3;
  int32_t new_axis_mask = 0x0;
  int32_t shrink_axis_mask = 0x0;

  // Function outputs.
  TensorShape processing_shape, final_shape;
  bool is_identity, is_simple_slice, slice_dim0;
  IntVector begin, end, strides;
  StridedSliceShapeSpec shape_spec;

  EXPECT_THAT(
      ValidateStridedSliceOp(
          &begin_tensor, &end_tensor, strides_tensor, input_shape,
          begin_mask_spec, end_mask_spec, ellipsis_mask, new_axis_mask,
          shrink_axis_mask, &processing_shape, &final_shape, &is_identity,
          &is_simple_slice, &slice_dim0, &begin, &end, &strides, &shape_spec),
      tsl::testing::StatusIs(tsl::error::Code::INVALID_ARGUMENT,
                             "Multiple ellipses in slice spec not allowed"));
}

TEST(ValidateStridedSliceOpTest, WrongBeginTensorFails) {
  Tensor begin_tensor = test::AsTensor<int32_t>({0});  // Incorrect shape.
  Tensor end_tensor = test::AsTensor<int32_t>({10, 10});
  Tensor strides_tensor = test::AsTensor<int32_t>({1, 1});
  TensorShape input_shape = AsTensorShape({10, 10});
  int32_t begin_mask_spec = 0x0;
  int32_t end_mask_spec = 0x0;
  int32_t ellipsis_mask = 0x1;
  int32_t new_axis_mask = 0x2;
  int32_t shrink_axis_mask = 0x0;

  // Function outputs.
  TensorShape processing_shape, final_shape;
  bool is_identity, is_simple_slice, slice_dim0;
  IntVector begin, end, strides;
  StridedSliceShapeSpec shape_spec;

  EXPECT_THAT(
      ValidateStridedSliceOp(
          &begin_tensor, &end_tensor, strides_tensor, input_shape,
          begin_mask_spec, end_mask_spec, ellipsis_mask, new_axis_mask,
          shrink_axis_mask, &processing_shape, &final_shape, &is_identity,
          &is_simple_slice, &slice_dim0, &begin, &end, &strides, &shape_spec),
      tsl::testing::StatusIs(
          tsl::error::Code::INVALID_ARGUMENT,
          ::testing::ContainsRegex("Expected .* equal size tensors")));
}

TEST(ValidateStridedSliceOpTest, WrongStridesTensorWithNullBeginFails) {
  Tensor end_tensor = test::AsTensor<int32_t>({10, 10});
  Tensor strides_tensor = test::AsTensor<int32_t>({1});  // Invalid shape.
  TensorShape input_shape = AsTensorShape({10, 10});
  int32_t begin_mask_spec = 0x0;
  int32_t end_mask_spec = 0x0;
  int32_t ellipsis_mask = 0x1;
  int32_t new_axis_mask = 0x2;
  int32_t shrink_axis_mask = 0x0;

  // Function outputs.
  TensorShape processing_shape, final_shape;
  bool is_identity, is_simple_slice, slice_dim0;
  IntVector begin, end, strides;
  StridedSliceShapeSpec shape_spec;

  EXPECT_THAT(
      ValidateStridedSliceOp(
          /*begin_tensor=*/nullptr, &end_tensor, strides_tensor, input_shape,
          begin_mask_spec, end_mask_spec, ellipsis_mask, new_axis_mask,
          shrink_axis_mask, &processing_shape, &final_shape, &is_identity,
          &is_simple_slice, &slice_dim0, &begin, &end, &strides, &shape_spec),
      tsl::testing::StatusIs(
          tsl::error::Code::INVALID_ARGUMENT,
          ::testing::ContainsRegex("Expected .* equal size tensors")));
}

TEST(ValidateStridedSliceOpTest, NullBeginEndWithShrinkAxis) {
  Tensor strides_tensor = test::AsTensor<int32_t>({2, -2, 1});
  TensorShape input_shape = AsTensorShape({10, 10, 1});
  int32_t begin_mask_spec = 0x3;
  int32_t end_mask_spec = 0x3;
  int32_t ellipsis_mask = 0x0;
  int32_t new_axis_mask = 0x0;
  int32_t shrink_axis_mask = 0x4;

  // Function outputs.
  TensorShape processing_shape, final_shape;
  bool is_identity, is_simple_slice, slice_dim0;
  IntVector begin, end, strides;
  StridedSliceShapeSpec shape_spec;

  TF_EXPECT_OK(ValidateStridedSliceOp(
      /*begin_tensor=*/nullptr, /*end_tensor=*/nullptr, strides_tensor,
      input_shape, begin_mask_spec, end_mask_spec, ellipsis_mask, new_axis_mask,
      shrink_axis_mask, &processing_shape, &final_shape, &is_identity,
      &is_simple_slice, &slice_dim0, &begin, &end, &strides, &shape_spec));

  EXPECT_EQ(processing_shape, AsTensorShape({5, 5, 1}));
  EXPECT_EQ(final_shape, AsTensorShape({5, 5}));
  EXPECT_EQ(strides, (IntVector{2, -2, 1}));
}

TEST(ValidateStridedSliceOpTest, UnknownInputRankFails) {
  Tensor end_tensor = test::AsTensor<int32_t>({10, 10});
  Tensor strides_tensor = test::AsTensor<int32_t>({1, 1});
  PartialTensorShape input_shape;  // Unknown rank.
  int32_t begin_mask_spec = 0x0;
  int32_t end_mask_spec = 0x0;
  int32_t ellipsis_mask = 0x1;
  int32_t new_axis_mask = 0x2;
  int32_t shrink_axis_mask = 0x0;

  // Function outputs.
  TensorShape processing_shape, final_shape;
  bool is_identity, is_simple_slice, slice_dim0;
  IntVector begin, end, strides;
  StridedSliceShapeSpec shape_spec;

  EXPECT_THAT(
      ValidateStridedSliceOp(
          /*begin_tensor=*/nullptr, &end_tensor, strides_tensor, input_shape,
          begin_mask_spec, end_mask_spec, ellipsis_mask, new_axis_mask,
          shrink_axis_mask, &processing_shape, &final_shape, &is_identity,
          &is_simple_slice, &slice_dim0, &begin, &end, &strides, &shape_spec),
      tsl::testing::StatusIs(tsl::error::Code::INVALID_ARGUMENT,
                             ::testing::ContainsRegex("unknown rank")));
}

TEST(ValidateStridedSliceOpTest, PartialInputShape) {
  Tensor end_tensor = test::AsTensor<int32_t>({10, 10});
  Tensor strides_tensor = test::AsTensor<int32_t>({1, 1});
  PartialTensorShape input_shape;
  TF_CHECK_OK(
      PartialTensorShape::BuildPartialTensorShape({10, -1}, &input_shape));
  int32_t begin_mask_spec = 0x0;
  int32_t end_mask_spec = 0x0;
  int32_t ellipsis_mask = 0x0;
  int32_t new_axis_mask = 0x0;
  int32_t shrink_axis_mask = 0x0;

  // Function outputs.
  PartialTensorShape processing_shape, final_shape;
  bool is_identity, is_simple_slice, slice_dim0;
  IntVector begin, end, strides;
  StridedSliceShapeSpec shape_spec;

  TF_EXPECT_OK(ValidateStridedSliceOp(
      /*begin_tensor=*/nullptr, &end_tensor, strides_tensor, input_shape,
      begin_mask_spec, end_mask_spec, ellipsis_mask, new_axis_mask,
      shrink_axis_mask, &processing_shape, &final_shape, &is_identity,
      &is_simple_slice, &slice_dim0, &begin, &end, &strides, &shape_spec));
}

}  // namespace
}  // namespace tensorflow
