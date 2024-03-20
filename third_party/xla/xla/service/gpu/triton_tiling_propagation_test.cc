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

#include "xla/service/gpu/triton_tiling_propagation.h"

#include <vector>

#include <gtest/gtest.h>
#include "xla/tests/hlo_test_base.h"

namespace xla::gpu {
namespace {

using TritonTilingPropagationTest = HloTestBase;
using triton_fusion::DimensionOrder;

DimensionOrder FromFragments(DimensionOrder::Fragments fragments) {
  DimensionOrder dim_order;
  DimensionOrder::Fragments& tensor_fragments_order =
      dim_order.TensorFragmentsOrder();
  DimensionOrder::FragmentOrders& dim_fragments_orders =
      dim_order.DimFragmentsOrders();
  for (const DimensionOrder::Fragment& fragment : fragments) {
    tensor_fragments_order.push_back(fragment);
    dim_fragments_orders[fragment.dst_dim_number()].push_back(
        tensor_fragments_order.size());
  }
  return dim_order;
}

TEST_F(
    TritonTilingPropagationTest,
    DimensionOrdersRemainPhysicallyEquivalentAfterInsertingTrivialDimensions) {
  DimensionOrder::Fragment fragment_1(/*dst_dim_number=*/0, /*count=*/97);
  DimensionOrder::Fragment fragment_2(/*dst_dim_number=*/0, /*count=*/1);
  DimensionOrder dimension_order_1 = FromFragments({fragment_1, fragment_2});

  DimensionOrder::Fragment fragment_3(/*dst_dim_number=*/0, /*count=*/97);
  DimensionOrder::Fragment fragment_4(/*dst_dim_number=*/1, /*count=*/1);
  DimensionOrder dimension_order_2 = FromFragments({fragment_3, fragment_4});

  // They should be equivalent because fragment_2 and fragment_4 both have count
  // 1, so they don't affect the physical representation.
  EXPECT_TRUE(dimension_order_1.IsPhysicallyEquivalent(dimension_order_2));
}

TEST_F(
    TritonTilingPropagationTest,
    IterationSpecsRemainPhysicallyEquivalentAfterInsertingTrivialDimensions) {
  TensorIterationSpec::IterationSpecFragment fragment_1 = {
      /*stride=*/1, /*count=*/97, /*slice_start=*/0, /*sliced_count=*/97,
      /*subfragments=*/{97}};
  TensorIterationSpec spec_1;
  spec_1[0].push_back(fragment_1);

  TensorIterationSpec::IterationSpecFragment fragment_2 = {
      /*stride=*/1, /*count=*/97, /*slice_start=*/0, /*sliced_count=*/97,
      /*subfragments=*/{97}};
  TensorIterationSpec::IterationSpecFragment fragment_3 = {
      /*stride=*/97, /*count=*/1, /*slice_start=*/0, /*sliced_count=*/1,
      /*subfragments=*/{1}};
  TensorIterationSpec spec_2;
  spec_2[0].push_back(fragment_2);
  spec_2[1].push_back(fragment_3);

  // spec_2's extra dimension is degenerate, so it should have the same physical
  // representation as spec_1.
  EXPECT_TRUE(spec_1.IsPhysicallyEquivalent(spec_2));
}

TEST_F(TritonTilingPropagationTest,
       DimensionsShouldNotBeRemovedByToTensorIterationSpec) {
  DimensionOrder::Fragment fragment_0(/*dst_dim_number=*/0, /*count=*/97);
  DimensionOrder::Fragment fragment_1(/*dst_dim_number=*/1, /*count=*/1);
  DimensionOrder dimension_order = FromFragments({fragment_0, fragment_1});
  TensorIterationSpec spec = dimension_order.ToTensorIterationSpec();
  const TensorIterationSpec::DimIterationSpec* dim_spec_0 = spec.Find(0);
  EXPECT_NE(dim_spec_0, nullptr);
  EXPECT_EQ(dim_spec_0->size(), 1);
  EXPECT_EQ(dim_spec_0->at(0).count, 97);

  const TensorIterationSpec::DimIterationSpec* dim_spec_1 = spec.Find(1);
  EXPECT_NE(dim_spec_1, nullptr);
  EXPECT_EQ(dim_spec_1->size(), 1);
  EXPECT_EQ(dim_spec_1->at(0).count, 1);
}

}  // namespace
}  // namespace xla::gpu
