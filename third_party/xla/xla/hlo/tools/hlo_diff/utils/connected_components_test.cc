// Copyright 2025 The OpenXLA Authors
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

#include "xla/hlo/tools/hlo_diff/utils/connected_components.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace hlo_diff {

namespace {
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

// Helper function to create a simple HloComputation
std::unique_ptr<HloComputation> MakeComputation(absl::string_view name) {
  auto builder = HloComputation::Builder(name);
  HloInstruction* x = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "x"));
  builder.AddInstruction(HloInstruction::CreateTuple({x}));
  return builder.Build();
}

class ConnectedComponentsFinderTest : public ::testing::Test {};

TEST_F(ConnectedComponentsFinderTest, EmptyGraph) {
  ConnectedComponentsFinder cc_finder;
  EXPECT_THAT(cc_finder.FindConnectedComponents(), IsEmpty());
}

TEST_F(ConnectedComponentsFinderTest, SingleNodeNoEdges) {
  ConnectedComponentsFinder cc_finder;
  auto c1 = MakeComputation("c1");
  cc_finder.AddEdge(c1.get(), c1.get());  // Adding a self-loop, node exists
  auto components = cc_finder.FindConnectedComponents();
  ASSERT_THAT(components.size(), 1);
  EXPECT_THAT(components[0], UnorderedElementsAre(c1.get()));
}

TEST_F(ConnectedComponentsFinderTest, TwoSeparateNodes) {
  ConnectedComponentsFinder cc_finder;
  auto c1 = MakeComputation("c1");
  auto c2 = MakeComputation("c2");
  // Don't add an edge between c1 and c2
  cc_finder.AddEdge(c1.get(), c1.get());
  cc_finder.AddEdge(c2.get(), c2.get());
  auto components = cc_finder.FindConnectedComponents();
  ASSERT_THAT(components.size(), 2);
  EXPECT_THAT(components, UnorderedElementsAre(UnorderedElementsAre(c1.get()),
                                               UnorderedElementsAre(c2.get())));
}

TEST_F(ConnectedComponentsFinderTest, TwoConnectedNodes) {
  ConnectedComponentsFinder cc_finder;
  auto c1 = MakeComputation("c1");
  auto c2 = MakeComputation("c2");
  cc_finder.AddEdge(c1.get(), c2.get());
  auto components = cc_finder.FindConnectedComponents();
  ASSERT_THAT(components.size(), 1);
  EXPECT_THAT(components[0], UnorderedElementsAre(c1.get(), c2.get()));
}

TEST_F(ConnectedComponentsFinderTest, ThreeNodesLinearConnection) {
  ConnectedComponentsFinder cc_finder;
  auto c1 = MakeComputation("c1");
  auto c2 = MakeComputation("c2");
  auto c3 = MakeComputation("c3");
  cc_finder.AddEdge(c1.get(), c2.get());
  cc_finder.AddEdge(c2.get(), c3.get());
  auto components = cc_finder.FindConnectedComponents();
  ASSERT_THAT(components.size(), 1);
  EXPECT_THAT(components[0],
              UnorderedElementsAre(c1.get(), c2.get(), c3.get()));
}

TEST_F(ConnectedComponentsFinderTest, ThreeNodesTriangleConnection) {
  ConnectedComponentsFinder cc_finder;
  auto c1 = MakeComputation("c1");
  auto c2 = MakeComputation("c2");
  auto c3 = MakeComputation("c3");
  cc_finder.AddEdge(c1.get(), c2.get());
  cc_finder.AddEdge(c2.get(), c3.get());
  cc_finder.AddEdge(c3.get(), c1.get());
  auto components = cc_finder.FindConnectedComponents();
  ASSERT_THAT(components.size(), 1);
  EXPECT_THAT(components[0],
              UnorderedElementsAre(c1.get(), c2.get(), c3.get()));
}

TEST_F(ConnectedComponentsFinderTest, MixedConnectedAndSeparate) {
  ConnectedComponentsFinder cc_finder;
  auto c1 = MakeComputation("c1");
  auto c2 = MakeComputation("c2");
  auto c3 = MakeComputation("c3");
  auto c4 = MakeComputation("c4");
  auto c5 = MakeComputation("c5");
  cc_finder.AddEdge(c1.get(), c2.get());
  cc_finder.AddEdge(c2.get(), c3.get());
  cc_finder.AddEdge(c4.get(), c4.get());  // c4 is separate
  auto components = cc_finder.FindConnectedComponents();
  ASSERT_THAT(components.size(), 2);
  EXPECT_THAT(
      components,
      UnorderedElementsAre(UnorderedElementsAre(c1.get(), c2.get(), c3.get()),
                           UnorderedElementsAre(c4.get())));
}

TEST_F(ConnectedComponentsFinderTest, LargerComponentOk) {
  ConnectedComponentsFinder cc_finder;
  auto c1 = MakeComputation("c1");
  auto c2 = MakeComputation("c2");
  auto c3 = MakeComputation("c3");
  auto c4 = MakeComputation("c4");
  auto c5 = MakeComputation("c5");
  auto c6 = MakeComputation("c6");
  cc_finder.AddEdge(c1.get(), c2.get());
  cc_finder.AddEdge(c2.get(), c3.get());
  cc_finder.AddEdge(c3.get(), c4.get());
  cc_finder.AddEdge(c4.get(), c1.get());
  cc_finder.AddEdge(c3.get(), c5.get());
  cc_finder.AddEdge(c6.get(), c6.get());

  auto components = cc_finder.FindConnectedComponents();

  ASSERT_THAT(components.size(), 2);
  EXPECT_THAT(components, UnorderedElementsAre(
                              UnorderedElementsAre(c1.get(), c2.get(), c3.get(),
                                                   c4.get(), c5.get()),
                              UnorderedElementsAre(c6.get())));
}

}  // namespace
}  // namespace hlo_diff
}  // namespace xla
