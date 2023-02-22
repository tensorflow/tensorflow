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

#include "tensorflow/compiler/xla/service/hlo_phi_graph.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace {
class PhiGraphTest : public ::testing::Test {
 protected:
  HloValue NewHloValue(bool is_phi) {
    static int64_t id = 0;
    return HloValue(id++, dummy_inst_.get(), {}, is_phi);
  }

  void SetUp() override {
    dummy_inst_ = HloInstruction::CreateConstant(LiteralUtil::CreateR0(0.0f));
  }

  // Dummy instruction used to fill unrelated argument when creating a
  // HloValue.
  std::unique_ptr<HloInstruction> dummy_inst_;
};

TEST_F(PhiGraphTest, SelfReferencingPhi) {
  // Def A = non-phi
  // Def B = phi(B, A)
  //
  // Optimize B into A.
  PhiGraph phi_graph;
  HloValue A = NewHloValue(false);
  HloValue B = NewHloValue(true);
  phi_graph.RegisterPhi(B, {&A, &B});
  phi_graph.Optimize();
  EXPECT_EQ(A.id(), phi_graph.FindOptimizedValue(B.id()));
}

TEST_F(PhiGraphTest, PhiWithSameInputs) {
  // Def A = non-phi
  // Def B = phi(A, A)
  //
  // Optimize B into A.
  PhiGraph phi_graph;
  HloValue A = NewHloValue(false);
  HloValue B = NewHloValue(true);
  phi_graph.RegisterPhi(B, {&A, &A});
  phi_graph.Optimize();
  EXPECT_EQ(A.id(), phi_graph.FindOptimizedValue(B.id()));
}

TEST_F(PhiGraphTest, CircularPhi) {
  // def A = phi(B, C)
  // def B = phi(C, D)
  // def C = phi(A, B)
  // def D = non-phi
  // Replace A, B, and C with D:
  PhiGraph phi_graph;
  HloValue A = NewHloValue(true);
  HloValue B = NewHloValue(true);
  HloValue C = NewHloValue(true);
  HloValue D = NewHloValue(false);
  phi_graph.RegisterPhi(A, {&B, &C});
  phi_graph.RegisterPhi(B, {&D, &C});
  phi_graph.RegisterPhi(C, {&A, &B});
  phi_graph.Optimize();
  EXPECT_EQ(D.id(), phi_graph.FindOptimizedValue(A.id()));
  EXPECT_EQ(D.id(), phi_graph.FindOptimizedValue(B.id()));
  EXPECT_EQ(D.id(), phi_graph.FindOptimizedValue(C.id()));
}

TEST_F(PhiGraphTest, NestedPhiReduction) {
  // def A = phi(B, C)
  // def B = phi(C, E)
  // def C = phi(A, B)
  // def D = non-phi
  // def E = Phi(D, D)
  // 1. Replace E with D
  // 2. Replace A B and C with E/D
  PhiGraph phi_graph;
  HloValue A = NewHloValue(true);
  HloValue B = NewHloValue(true);
  HloValue C = NewHloValue(true);
  HloValue D = NewHloValue(false);
  HloValue E = NewHloValue(true);
  phi_graph.RegisterPhi(A, {&B, &C});
  phi_graph.RegisterPhi(B, {&E, &C});
  phi_graph.RegisterPhi(C, {&A, &B});
  phi_graph.RegisterPhi(E, {&D, &D});
  phi_graph.Optimize();
  EXPECT_EQ(D.id(), phi_graph.FindOptimizedValue(A.id()));
  EXPECT_EQ(D.id(), phi_graph.FindOptimizedValue(B.id()));
  EXPECT_EQ(D.id(), phi_graph.FindOptimizedValue(C.id()));
  EXPECT_EQ(D.id(), phi_graph.FindOptimizedValue(E.id()));
}

}  // namespace
}  // namespace xla
