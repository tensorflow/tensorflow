/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/utils/colocation.h"

#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {

class ColocationTest : public ::testing::Test {};

bool VerifyNodeHasColocation(const NodeDef& ndef, const string& coloc) {
  if (ndef.attr().empty()) {
    return false;
  }
  if (ndef.attr().find("_class") == ndef.attr().end()) {
    return false;
  }
  return ndef.attr().at("_class").list().s(0) == coloc;
}

TEST(ColocationTest, ReassignColocation_SingleNode) {
  // Node A colocates with B, but node B is not in the graph.
  //   A
  //   |
  //   |
  //  [B]

  NodeDef ndef;
  const absl::Status status =
      NodeDefBuilder("A", "Const").Attr("_class", {"loc:@B"}).Finalize(&ndef);
  TF_EXPECT_OK(status);
  GraphDef gdef = test::function::GDef({ndef});

  EXPECT_EQ(1, gdef.node_size());
  EXPECT_EQ(1, gdef.node(0).attr_size());

  ReassignColocation(&gdef);

  // Validates that node A's colocation info is cleared.
  EXPECT_EQ(1, gdef.node_size());
  EXPECT_EQ(0, gdef.node(0).attr_size());
}

TEST(ColocationTest, ReassignColocation_MultiNode_SingleGroup) {
  // Node A, B, C colocate with X. D colocates with C. E colocates with D.
  // Node X is not in the graph.
  //  A   B   C---D---E
  //  |   |   |
  //  |   |   |
  //  +--[X]--+
  // After re-assign of colocation, A, B, C, D should colocate with E.
  // A   B   C   D
  // |   |   |   |
  // |   |   |   |
  // +---+-E-+---+

  NodeDef ndef_a, ndef_b, ndef_c, ndef_d, ndef_e;
  absl::Status status =
      NodeDefBuilder("A", "Const").Attr("_class", {"loc:@X"}).Finalize(&ndef_a);
  TF_EXPECT_OK(status);
  status =
      NodeDefBuilder("B", "Const").Attr("_class", {"loc:@X"}).Finalize(&ndef_b);
  TF_EXPECT_OK(status);
  status =
      NodeDefBuilder("C", "Const").Attr("_class", {"loc:@X"}).Finalize(&ndef_c);
  TF_EXPECT_OK(status);
  status =
      NodeDefBuilder("D", "Const").Attr("_class", {"loc:@C"}).Finalize(&ndef_d);
  TF_EXPECT_OK(status);
  status =
      NodeDefBuilder("E", "Const").Attr("_class", {"loc:@D"}).Finalize(&ndef_e);
  TF_EXPECT_OK(status);
  GraphDef gdef =
      test::function::GDef({ndef_a, ndef_b, ndef_c, ndef_d, ndef_e});

  EXPECT_EQ(5, gdef.node_size());
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(0), "loc:@X"));  // A
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(1), "loc:@X"));  // B
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(2), "loc:@X"));  // C
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(3), "loc:@C"));  // D
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(4), "loc:@D"));  // E

  ReassignColocation(&gdef);

  EXPECT_EQ(5, gdef.node_size());
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(0), "loc:@E"));  // A
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(1), "loc:@E"));  // B
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(2), "loc:@E"));  // C
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(3), "loc:@E"));  // D
  EXPECT_EQ(0, gdef.node(4).attr_size());                        // E
}

TEST(ColocationTest, ReassignColocation_MultiNode_MultiGroup) {
  // Before re-assign:
  // Node A, B, C colocate with X. D colocates with C. E colocates with D.
  // Node U, V colocates with W. Node X, W are not in the graph:
  //  A   B   C---D---E
  //  |   |   |
  //  |   |   |
  //  +--[X]--+
  //
  //  U       V
  //  |       |
  //  |       |
  //  +--[W]--+
  //
  // After re-assign:
  // A, B, C, D should colocate with E. U should colocate with V.
  // A   B   C   D
  // |   |   |   |
  // |   |   |   |
  // +---+-E-+---+
  //
  // U
  // |
  // |
  // V

  NodeDef ndef_a, ndef_b, ndef_c, ndef_d, ndef_e, ndef_u, ndef_v;
  absl::Status status =
      NodeDefBuilder("A", "Const").Attr("_class", {"loc:@X"}).Finalize(&ndef_a);
  TF_EXPECT_OK(status);
  status =
      NodeDefBuilder("B", "Const").Attr("_class", {"loc:@X"}).Finalize(&ndef_b);
  TF_EXPECT_OK(status);
  status =
      NodeDefBuilder("C", "Const").Attr("_class", {"loc:@X"}).Finalize(&ndef_c);
  TF_EXPECT_OK(status);
  status =
      NodeDefBuilder("D", "Const").Attr("_class", {"loc:@C"}).Finalize(&ndef_d);
  TF_EXPECT_OK(status);
  status =
      NodeDefBuilder("E", "Const").Attr("_class", {"loc:@D"}).Finalize(&ndef_e);
  TF_EXPECT_OK(status);
  status =
      NodeDefBuilder("U", "Const").Attr("_class", {"loc:@W"}).Finalize(&ndef_u);
  TF_EXPECT_OK(status);
  status =
      NodeDefBuilder("V", "Const").Attr("_class", {"loc:@W"}).Finalize(&ndef_v);
  TF_EXPECT_OK(status);
  GraphDef gdef = test::function::GDef(
      {ndef_a, ndef_b, ndef_c, ndef_d, ndef_e, ndef_u, ndef_v});

  EXPECT_EQ(7, gdef.node_size());
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(0), "loc:@X"));  // A
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(1), "loc:@X"));  // B
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(2), "loc:@X"));  // C
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(3), "loc:@C"));  // D
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(4), "loc:@D"));  // E
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(5), "loc:@W"));  // U
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(6), "loc:@W"));  // V

  ReassignColocation(&gdef);

  EXPECT_EQ(7, gdef.node_size());
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(0), "loc:@E"));  // A
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(1), "loc:@E"));  // B
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(2), "loc:@E"));  // C
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(3), "loc:@E"));  // D
  EXPECT_EQ(0, gdef.node(4).attr_size());                        // E
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(5), "loc:@V"));  // U
  EXPECT_EQ(0, gdef.node(6).attr_size());                        // V
}

}  // namespace grappler
}  // namespace tensorflow
