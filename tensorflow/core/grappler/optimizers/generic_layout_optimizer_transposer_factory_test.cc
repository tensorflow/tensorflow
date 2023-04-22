/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer_factory.h"

#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {

namespace {

void CheckSameTransposerForOps(absl::Span<const string> ops,
                               TransposerFactory* factory,
                               absl::flat_hash_set<Transposer*>* transposers) {
  absl::flat_hash_set<Transposer*> created_transposers;
  for (int i = 0; i < ops.size(); i++) {
    NodeDef node;
    node.set_op(ops[i]);
    std::shared_ptr<Transposer> transposer1 = factory->GetTransposer(node);
    ASSERT_NE(transposer1, nullptr);
    if (i == 0) {  // Newly added/created transposer for op.
      EXPECT_TRUE(transposers->insert(transposer1.get()).second);
    } else {
      EXPECT_FALSE(transposers->insert(transposer1.get()).second);
    }
    // Get transposer for same op and compare if same transposer is returned.
    std::shared_ptr<Transposer> transposer2 = factory->GetTransposer(node);
    ASSERT_NE(transposer2, nullptr);
    EXPECT_EQ(transposer1.get(), transposer2.get());
    created_transposers.insert(transposer1.get());
  }
  if (!ops.empty()) {
    // Only one transposer should be added/created for ops.
    EXPECT_EQ(created_transposers.size(), 1);
  }
}

TEST(TransposerFactoryTest, SanityCheck) {
  TransposerFactory factory;
  absl::flat_hash_set<Transposer*> transposers;

  CheckSameTransposerForOps(
      {"Conv2D", "FusedBatchNorm", "DepthwiseConv2dNative"}, &factory,
      &transposers);

  CheckSameTransposerForOps({"AvgPoolGrad"}, &factory, &transposers);

  CheckSameTransposerForOps({"BiasAddGrad"}, &factory, &transposers);

  CheckSameTransposerForOps({"_FusedBatchNormEx"}, &factory, &transposers);

  CheckSameTransposerForOps({"FusedBatchNormGrad", "FusedBatchNormGradV2"},
                            &factory, &transposers);

  CheckSameTransposerForOps(
      {"Conv2DBackpropFilter", "DepthwiseConv2dNativeBackpropFilter"}, &factory,
      &transposers);

  CheckSameTransposerForOps(
      {"Conv2DBackpropInput", "DepthwiseConv2dNativeBackpropInput"}, &factory,
      &transposers);

  CheckSameTransposerForOps({"MaxPoolGrad", "MaxPoolGradGrad"}, &factory,
                            &transposers);

  CheckSameTransposerForOps({"MaxPoolGradV2", "MaxPoolGradGradV2"}, &factory,
                            &transposers);

  CheckSameTransposerForOps({"AddN"}, &factory, &transposers);

  CheckSameTransposerForOps({"IdentityN"}, &factory, &transposers);

  CheckSameTransposerForOps({"Merge", "RefMerge"}, &factory, &transposers);

  CheckSameTransposerForOps({"Select"}, &factory, &transposers);

  CheckSameTransposerForOps({"Switch", "RefSwitch"}, &factory, &transposers);

  CheckSameTransposerForOps({"Betainc"}, &factory, &transposers);

  CheckSameTransposerForOps({"TanhGrad"}, &factory, &transposers);

  CheckSameTransposerForOps({"Squeeze"}, &factory, &transposers);

  CheckSameTransposerForOps({"MaxPoolV2"}, &factory, &transposers);

  CheckSameTransposerForOps({"RealDiv", "Atan2", "Complex"}, &factory,
                            &transposers);

  CheckSameTransposerForOps({"Concat", "ConcatV2"}, &factory, &transposers);

  CheckSameTransposerForOps({"Pad", "PadV2", "MirrorPad", "MirrorPadGrad"},
                            &factory, &transposers);

  CheckSameTransposerForOps({"ReverseV2"}, &factory, &transposers);

  CheckSameTransposerForOps({"Tile"}, &factory, &transposers);

  CheckSameTransposerForOps({"Shape"}, &factory, &transposers);

  CheckSameTransposerForOps({"ShapeN"}, &factory, &transposers);

  CheckSameTransposerForOps({"Fill"}, &factory, &transposers);

  CheckSameTransposerForOps({"Slice"}, &factory, &transposers);

  CheckSameTransposerForOps({"Split"}, &factory, &transposers);

  CheckSameTransposerForOps({"SplitV"}, &factory, &transposers);

  CheckSameTransposerForOps({"StridedSlice"}, &factory, &transposers);

  CheckSameTransposerForOps({"Sum", "Mean", "Prod", "Max", "Min", "All", "Any"},
                            &factory, &transposers);

  NodeDef node_unknown;
  node_unknown.set_op("UnknownOp");
  std::shared_ptr<Transposer> transposer_unknown =
      factory.GetTransposer(node_unknown);
  EXPECT_TRUE(transposer_unknown == nullptr);
}

TEST(TransposerFactoryTest, ShouldUseAllOpTransposer) {
  TransposerFactory factory;
  std::vector<OpDef> op_defs;
  OpRegistry::Global()->GetRegisteredOps(&op_defs);
  NodeDef node;
  AttrValue value;
  value.set_type(DataType::DT_DOUBLE);
  node.mutable_attr()->insert({"T", value});
  for (const OpDef& op_def : op_defs) {
    node.set_op(op_def.name());
    std::shared_ptr<Transposer> transposer = factory.GetTransposer(node);
    if (transposer != nullptr) {
      EXPECT_TRUE(IsLayoutSensitiveOp(node) || IsLayoutAgnosticOp(node))
          << "Transposer for op \"" << node.op()
          << "\" is created but not used. Add it to IsLayourSensitiveOp or "
             "IslayoutAgnosticOp.";
    }
  }
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
