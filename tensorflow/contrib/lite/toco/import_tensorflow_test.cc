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
#include "tensorflow/contrib/lite/toco/import_tensorflow.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"

namespace toco {

using port::Status;
using tensorflow::AttrValue;
using tensorflow::DT_BOOL;
using tensorflow::DT_FLOAT;
using tensorflow::DT_INT32;
using tensorflow::DT_INT64;
using tensorflow::DT_QUINT8;
using tensorflow::DT_STRING;
using tensorflow::NodeDef;

namespace internal {
Status ImportTensorFlowNode(const NodeDef&, const TensorFlowImportFlags&,
                            Model*);
}  // namespace internal

namespace {

class ShapeImportTest : public ::testing::TestWithParam<tensorflow::DataType> {
 protected:
  ShapeImportTest() {}

  void BuildConstNode(std::initializer_list<int64_t> shape,
                      tensorflow::DataType dtype, int64_t num_elements,
                      NodeDef* node) {
    node->set_op("Const");
    node->set_name("Node1");

    // An attribute describing the type of this const node.
    AttrValue dtype_attr;
    SetAttrValue(dtype, &dtype_attr);
    (*node->mutable_attr())["dtype"] = dtype_attr;

    // An attribute describing the content of this const node.
    tensorflow::TensorProto t;
    t.set_dtype(dtype);
    auto* s = t.mutable_tensor_shape();
    for (auto d : shape) {
      s->add_dim()->set_size(d);
    }

    // TODO(ahentz): also need to test via tensor_content()
    switch (dtype) {
      case DT_FLOAT:
        for (int64_t i = 0; i < num_elements; ++i) {
          t.add_float_val(i / 10000.0);
        }
        break;
      case DT_INT32:
        for (int64_t i = 0; i < num_elements; ++i) {
          t.add_int_val(i % std::numeric_limits<int>::max());
        }
        break;
      case DT_QUINT8:
        for (int64_t i = 0; i < num_elements; ++i) {
          t.add_int_val(i % std::numeric_limits<uint8_t>::max());
        }
        break;
      case DT_INT64:
        for (int64_t i = 0; i < num_elements; ++i) {
          t.add_int64_val(i);
        }
        break;
      case DT_STRING:
        break;
      case DT_BOOL:
        for (int64_t i = 0; i < num_elements; ++i) {
          t.add_bool_val(i % 2);
        }
        break;
      default:
        break;
    }

    AttrValue value_attr;
    SetAttrValue(t, &value_attr);
    (*node->mutable_attr())["value"] = value_attr;
  }

  Status ImportNode(const NodeDef& node) {
    Model model;
    return internal::ImportTensorFlowNode(node, TensorFlowImportFlags(),
                                          &model);
  }
};

std::vector<tensorflow::DataType> TestTypes() {
  return {DT_FLOAT, DT_INT32, DT_INT64, DT_BOOL, DT_QUINT8};
}

TEST_P(ShapeImportTest, ShapeElementIsNegative) {
  NodeDef node;
  BuildConstNode({1, -2, 10}, GetParam(), 0, &node);
  auto status = ImportNode(node);
  EXPECT_EQ(status.error_message(),
            "Tensor shape should not include negative values (while processing "
            "node 'Node1')");
}
INSTANTIATE_TEST_CASE_P(ShapeElementIsNegative, ShapeImportTest,
                        ::testing::ValuesIn(TestTypes()));

TEST_P(ShapeImportTest, ShapeElementTooLarge) {
  NodeDef node;
  BuildConstNode({3000000000}, GetParam(), 0, &node);
  auto status = ImportNode(node);
  EXPECT_EQ(status.error_message(),
            "Shape element overflows (while processing node 'Node1')");
}
INSTANTIATE_TEST_CASE_P(ShapeElementTooLarge, ShapeImportTest,
                        ::testing::ValuesIn(TestTypes()));

TEST_P(ShapeImportTest, ShapeTooLarge) {
  NodeDef node;
  BuildConstNode({1000000, 2000000, 2000000, 2000000}, GetParam(), 0, &node);
  auto status = ImportNode(node);
  EXPECT_EQ(status.error_message(),
            "Tensor shape is too large (while processing node 'Node1')");
}
INSTANTIATE_TEST_CASE_P(ShapeTooLarge, ShapeImportTest,
                        ::testing::ValuesIn(TestTypes()));

TEST_P(ShapeImportTest, ValidShapeButZeroElements) {
  NodeDef node;
  BuildConstNode({1, 2, 2, 2}, GetParam(), 0, &node);
  auto status = ImportNode(node);
  EXPECT_THAT(
      status.error_message(),
      ::testing::MatchesRegex(
          "Neither input_content .0. nor .*_val .0. have the right "
          "dimensions .8. for this .* tensor .while processing node 'Node1'."));
}
INSTANTIATE_TEST_CASE_P(ValidShapeButZeroElements, ShapeImportTest,
                        ::testing::ValuesIn(TestTypes()));

}  // namespace
}  // namespace toco
