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

#include "tensorflow/core/grappler/optimizers/shape_optimizer.h"

#include <cstdint>

#include "absl/status/status.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class ShapeOptimizerTest : public GrapplerTest {};

TEST_F(ShapeOptimizerTest, OptimizeShapeProduct) {
  // Set the device to CPU zero, because the shape optimizer will only optimize
  // Prod to Size when a concrete Size kernel is available.
  tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");
  Output a = ops::Const(s.WithOpName("a"), 3.14f, {32, 16});
  Output c = ops::Shape(s.WithOpName("c"), a);
  Output d = ops::Const(s.WithOpName("d"), 0, {1});
  ops::ReduceProd::Attrs attrs;
  Output e = ops::ReduceProd(s.WithOpName("e"), c, d, attrs.KeepDims(false));
  Output f = ops::ReduceProd(s.WithOpName("f"), c, d, attrs.KeepDims(true));

  GrapplerItem item;
  item.fetch = {"e", "f"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  GraphDef output;
  ShapeOptimizer optimizer;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "e") {
      found++;
      EXPECT_EQ("Size", node.op());
      EXPECT_EQ("a", node.input(0));
    } else if (node.name() == "f") {
      found++;
      EXPECT_EQ("Prod", node.op());
      EXPECT_EQ("c", node.input(0));
    }
  }
  EXPECT_EQ(2, found);

  auto tensors_actual = EvaluateNodes(output, item.fetch);
  EXPECT_NEAR(tensors_expected[0].scalar<int>()(),
              tensors_actual[0].scalar<int>()(), 0);
  EXPECT_NEAR(tensors_expected[1].scalar<int>()(),
              tensors_actual[1].scalar<int>()(), 0);
}

TEST_F(ShapeOptimizerTest, DoesNotOptimizeShapeAsProdReductionInput) {
  tensorflow::Scope s =
      tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");
  Output data = ops::Const(s.WithOpName("data"), 2, {2, 2});
  Output x = ops::Const(s.WithOpName("x"), 3.14f, {1});
  Output shape = ops::Shape(s.WithOpName("shape"), x);
  Output product =
      ops::ReduceProd(s.WithOpName("product"), data, shape);

  GrapplerItem item;
  item.fetch = {"product"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  GraphDef output;
  ShapeOptimizer optimizer;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "product") {
      ++found;
      EXPECT_EQ(node.op(), "Prod");
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "data");
      EXPECT_EQ(node.input(1), "shape");
    }
  }
  EXPECT_EQ(found, 1);

  auto tensors_actual = EvaluateNodes(output, item.fetch);
  test::ExpectTensorEqual<int>(tensors_expected[0], tensors_actual[0]);
}

TEST_F(ShapeOptimizerTest, OptimizeShapeProductMissingKernel) {
  {
    // Skip this test if no GPU is available.
    std::vector<std::unique_ptr<Device>> devices;
    SessionOptions session_options;
    session_options.config.mutable_gpu_options()
        ->set_per_process_gpu_memory_fraction(0.1);
    session_options.env = Env::Default();
    TF_CHECK_OK(DeviceFactory::GetFactory(DEVICE_GPU)
                    ->AddDevices(session_options, "", &devices));
    bool found_gpu = false;
    for (const auto& d : devices) {
      if (d->device_type() == DEVICE_GPU) {
        found_gpu = true;
        break;
      }
    }
    if (!found_gpu) {
      LOG(INFO) << "Skipping test that requires GPU.";
      return;
    }
  }

  tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");
  Output a = ops::Const(s.WithOpName("a"), std::string("Hello"), {32, 16});
  Output c = ops::Shape(s.WithOpName("c"), a);
  Output d = ops::Const(s.WithOpName("d"), 0, {1});
  ops::ReduceProd::Attrs attrs;
  Output e = ops::ReduceProd(s.WithDevice("/gpu:0").WithOpName("e"), c, d,
                             attrs.KeepDims(false));

  GrapplerItem item;
  item.fetch = {"e"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  GraphDef output;
  ShapeOptimizer optimizer;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "e") {
      found++;
      EXPECT_EQ("Size", node.op());
      EXPECT_EQ("a", node.input(0));
      EXPECT_EQ("/cpu:0", node.device());
    }
  }
  EXPECT_EQ(1, found);

  auto tensors_actual = EvaluateNodes(output, item.fetch);
  EXPECT_NEAR(tensors_expected[0].scalar<int>()(),
              tensors_actual[0].scalar<int>()(), 0);
}

TEST_F(ShapeOptimizerTest, OptimizeShapeRatio) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a = ops::Const(s.WithOpName("a"), 3.14f, {32, 32});
  Output b = ops::Const(s.WithOpName("b"), 3.14f, {32, 16});
  Output c = ops::Size(s.WithOpName("c"), a);
  Output d = ops::Size(s.WithOpName("d"), b);
  Output e = ops::Div(s.WithOpName("e"), c, d);

  GrapplerItem item;
  item.fetch = {"e"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  GraphDef output;
  ShapeOptimizer optimizer;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "e") {
      found++;
      EXPECT_EQ("Const", node.op());
    }
  }
  EXPECT_EQ(1, found);

  auto tensors_actual = EvaluateNodes(output, item.fetch);
  EXPECT_NEAR(tensors_expected[0].scalar<int>()(),
              tensors_actual[0].scalar<int>()(), 0);
}

TEST_F(ShapeOptimizerTest, OptimizeStaticSizeRatioAtAndAboveInt32Limit) {
  GrapplerItem item;
  auto add_const = [&item](const char* name, int64_t dim) {
    NodeDef* x = item.graph.add_node();
    x->set_name(name);
    x->set_op("Const");
    (*x->mutable_attr())["dtype"].set_type(DT_FLOAT);
    TensorProto* tx = (*x->mutable_attr())["value"].mutable_tensor();
    tx->set_dtype(DT_FLOAT);
    tx->add_float_val(1.0f);
    tx->mutable_tensor_shape()->add_dim()->set_size(dim);
  };
  auto add_size = [&item](const char* name, const char* input,
                          DataType out_type) {
    NodeDef* size = item.graph.add_node();
    size->set_name(name);
    size->set_op("Size");
    size->add_input(input);
    (*size->mutable_attr())["T"].set_type(DT_FLOAT);
    (*size->mutable_attr())["out_type"].set_type(out_type);
  };
  auto add_div = [&item](const char* name, const char* numerator,
                         const char* denominator, DataType type) {
    NodeDef* div = item.graph.add_node();
    div->set_name(name);
    div->set_op("Div");
    div->add_input(numerator);
    div->add_input(denominator);
    (*div->mutable_attr())["T"].set_type(type);
  };

  add_const("x_at_limit", INT32_MAX);
  add_const("x_above_limit", int64_t{1} << 31);
  add_const("y", 1);
  add_size("size_x_at_limit_32", "x_at_limit", DT_INT32);
  add_size("size_x_above_limit_32", "x_above_limit", DT_INT32);
  add_size("size_y_32", "y", DT_INT32);
  add_size("size_x_above_limit_64", "x_above_limit", DT_INT64);
  add_size("size_y_64", "y", DT_INT64);
  add_div("ratio_at_limit_32", "size_x_at_limit_32", "size_y_32",
          DT_INT32);
  add_div("ratio_above_limit_32", "size_x_above_limit_32", "size_y_32",
          DT_INT32);
  add_div("ratio_above_limit_64", "size_x_above_limit_64", "size_y_64",
          DT_INT64);

  ShapeOptimizer optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "ratio_at_limit_32") {
      ++found;
      EXPECT_EQ(node.op(), "Const");
      ASSERT_EQ(node.attr().at("value").tensor().int_val_size(), 1);
      EXPECT_EQ(node.attr().at("value").tensor().int_val(0), INT32_MAX);
    } else if (node.name() == "ratio_above_limit_32") {
      ++found;
      EXPECT_EQ(node.op(), "Div");
    } else if (node.name() == "ratio_above_limit_64") {
      ++found;
      EXPECT_EQ(node.op(), "Const");
      ASSERT_EQ(node.attr().at("value").tensor().int64_val_size(), 1);
      EXPECT_EQ(node.attr().at("value").tensor().int64_val(0),
                int64_t{1} << 31);
    }
  }
  EXPECT_EQ(found, 3);
}

TEST_F(ShapeOptimizerTest, HandlesMalformedProdInputs) {
  GrapplerItem item;
  NodeDef* x = item.graph.add_node();
  x->set_name("x");
  x->set_op("Placeholder");
  (*x->mutable_attr())["dtype"].set_type(DT_FLOAT);

  NodeDef* shape_x = item.graph.add_node();
  shape_x->set_name("shape_x");
  shape_x->set_op("Shape");
  shape_x->add_input("x");
  (*shape_x->mutable_attr())["T"].set_type(DT_FLOAT);
  (*shape_x->mutable_attr())["out_type"].set_type(DT_INT32);

  NodeDef* malformed_prod = item.graph.add_node();
  malformed_prod->set_name("malformed_prod");
  malformed_prod->set_op("Prod");
  malformed_prod->add_input("shape_x");
  malformed_prod->add_input("^x");
  (*malformed_prod->mutable_attr())["T"].set_type(DT_INT32);

  ShapeOptimizer optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "malformed_prod") {
      ++found;
      EXPECT_EQ(node.op(), "Prod");
    }
  }
  EXPECT_EQ(found, 1);
}

TEST_F(ShapeOptimizerTest, HandlesControlInputInDivDataSlot) {
  // x has a static shape, so the same Div with two data inputs folds into a
  // Const. With a control input in data slot 0 it must stay unchanged.
  GrapplerItem item;
  NodeDef* x = item.graph.add_node();
  x->set_name("x");
  x->set_op("Const");
  (*x->mutable_attr())["dtype"].set_type(DT_FLOAT);
  TensorProto* tx = (*x->mutable_attr())["value"].mutable_tensor();
  tx->set_dtype(DT_FLOAT);
  tx->add_float_val(1.0f);
  tx->mutable_tensor_shape()->add_dim()->set_size(4);

  NodeDef* size_x = item.graph.add_node();
  size_x->set_name("size_x");
  size_x->set_op("Size");
  size_x->add_input("x");
  (*size_x->mutable_attr())["T"].set_type(DT_FLOAT);
  (*size_x->mutable_attr())["out_type"].set_type(DT_INT32);

  NodeDef* size_y = item.graph.add_node();
  size_y->set_name("size_y");
  size_y->set_op("Size");
  size_y->add_input("x");
  (*size_y->mutable_attr())["T"].set_type(DT_FLOAT);
  (*size_y->mutable_attr())["out_type"].set_type(DT_INT32);

  NodeDef* div = item.graph.add_node();
  div->set_name("control_div");
  div->set_op("Div");
  div->add_input("^size_x");
  div->add_input("size_y");
  (*div->mutable_attr())["T"].set_type(DT_INT32);

  ShapeOptimizer optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "control_div") {
      ++found;
      EXPECT_EQ(node.op(), "Div");
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "^size_x");
      EXPECT_EQ(node.input(1), "size_y");
    }
  }
  EXPECT_EQ(found, 1);

  GrapplerItem valid_item;
  valid_item.graph = item.graph;
  for (NodeDef& node : *valid_item.graph.mutable_node()) {
    if (node.name() == "control_div") {
      node.set_input(0, "size_x");
    }
  }
  GraphDef valid_output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, valid_item, &valid_output));

  found = 0;
  for (const NodeDef& node : valid_output.node()) {
    if (node.name() == "control_div") {
      ++found;
      EXPECT_EQ(node.op(), "Const");
      ASSERT_EQ(node.attr().at("value").tensor().int_val_size(), 1);
      EXPECT_EQ(node.attr().at("value").tensor().int_val(0), 1);
    }
  }
  EXPECT_EQ(found, 1);
}

TEST_F(ShapeOptimizerTest, HandlesMissingIntegerTypeAttributes) {
  GrapplerItem item;
  NodeDef* x = item.graph.add_node();
  x->set_name("x");
  x->set_op("Placeholder");
  (*x->mutable_attr())["dtype"].set_type(DT_FLOAT);

  NodeDef* shape_x = item.graph.add_node();
  shape_x->set_name("shape_x");
  shape_x->set_op("Shape");
  shape_x->add_input("x");
  (*shape_x->mutable_attr())["T"].set_type(DT_FLOAT);
  (*shape_x->mutable_attr())["out_type"].set_type(DT_INT32);

  NodeDef* untyped_prod = item.graph.add_node();
  untyped_prod->set_name("untyped_prod");
  untyped_prod->set_op("Prod");
  untyped_prod->add_input("shape_x");
  untyped_prod->add_input("shape_x");

  NodeDef* size_x = item.graph.add_node();
  size_x->set_name("size_x");
  size_x->set_op("Size");
  size_x->add_input("x");
  (*size_x->mutable_attr())["T"].set_type(DT_FLOAT);
  (*size_x->mutable_attr())["out_type"].set_type(DT_INT32);

  NodeDef* untyped_div = item.graph.add_node();
  untyped_div->set_name("untyped_div");
  untyped_div->set_op("Div");
  untyped_div->add_input("size_x");
  untyped_div->add_input("size_x");

  ShapeOptimizer optimizer;
  GraphDef output;
  absl::Status status = optimizer.Optimize(nullptr, item, &output);
  EXPECT_EQ(status.code(), absl::StatusCode::kAborted);
  EXPECT_EQ(output.node_size(), 0);

  int checked = 0;
  for (const NodeDef& node : item.graph.node()) {
    if (node.name() == "untyped_prod") {
      ++checked;
      EXPECT_EQ(node.op(), "Prod");
      EXPECT_EQ(node.attr().count("T"), 0);
    } else if (node.name() == "untyped_div") {
      ++checked;
      EXPECT_EQ(node.op(), "Div");
      EXPECT_EQ(node.attr().count("T"), 0);
    }
  }
  EXPECT_EQ(checked, 2);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
