/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Declare here, so we don't need a public header.
absl::Status FreezeRequantizationRanges(const GraphDef& input_graph_def,
                                        const TransformFuncContext& context,
                                        GraphDef* output_graph_def);
struct MinMaxRecord {
  string name;
  float min;
  float max;
};
absl::Status ExtractMinMaxRecords(const string& log_file_name,
                                  std::vector<MinMaxRecord>* records);

class FreezeRequantizationRangesTest : public ::testing::Test {
 protected:
  void TestFreezeRequantizationRanges() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor quantized_tensor(DT_QUINT8, TensorShape({1, 6}));
    test::FillValues<quint8>(&quantized_tensor, {0, 0, 0, 0, 0, 0});
    Output quantized_op = Const(root.WithOpName("quantized_op"),
                                Input::Initializer(quantized_tensor));

    Tensor quantized_min_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&quantized_min_tensor, {2.0f});
    Output quantized_min_op = Const(root.WithOpName("quantized_min_op"),
                                    Input::Initializer(quantized_min_tensor));

    Tensor quantized_max_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&quantized_max_tensor, {2.0f});
    Output quantized_max_op = Const(root.WithOpName("quantized_max_op"),
                                    Input::Initializer(quantized_min_tensor));

    Tensor offset_tensor(DT_QUINT8, TensorShape({6}));
    test::FillValues<quint8>(&offset_tensor, {1, 2, 3, 4, 5, 6});
    Output offset_op =
        Const(root.WithOpName("offset_op"), Input::Initializer(offset_tensor));

    Tensor offset_min_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&offset_min_tensor, {0.0f});
    Output offset_min_op = Const(root.WithOpName("offset_min_op"),
                                 Input::Initializer(offset_min_tensor));

    Tensor offset_max_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&offset_max_tensor, {255.0f});
    Output offset_max_op = Const(root.WithOpName("offset_max_op"),
                                 Input::Initializer(offset_max_tensor));

    QuantizedBiasAdd quantized_bias_add_op(
        root.WithOpName("bias_add_op"), quantized_op, offset_op,
        quantized_min_op, quantized_max_op, offset_min_op, offset_max_op,
        DT_QINT32);

    RequantizationRange requantization_range_op(
        root.WithOpName("requantization_range_op"),
        quantized_bias_add_op.output, quantized_bias_add_op.min_out,
        quantized_bias_add_op.max_out);

    Requantize requantize_op(
        root.WithOpName("requantize_op"), quantized_bias_add_op.output,
        quantized_bias_add_op.min_out, quantized_bias_add_op.max_out,
        requantization_range_op.output_min, requantization_range_op.output_max,
        DT_QUINT8);

    Output dequantize_op =
        Dequantize(root.WithOpName("dequantize_op"), requantize_op.output,
                   requantize_op.output_min, requantize_op.output_max);

    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));

    const string min_max_log_file_name =
        io::JoinPath(testing::TmpDir(), "min_max_log_file.txt");
    {
      std::unique_ptr<WritableFile> file;
      TF_ASSERT_OK(
          Env::Default()->NewWritableFile(min_max_log_file_name, &file));
      TF_ASSERT_OK(file->Append("Something irrelevant\n"));
      TF_ASSERT_OK(
          file->Append("[SomePrefix] "
                       ";requantization_range_op__print__;__requant_min_max:"
                       "[-2.4313571][10.584145]\n"));
      TF_ASSERT_OK(file->Append("Something else irrelevant\n"));
    }

    TransformFuncContext context;
    context.input_names = {};
    context.output_names = {"dequantize_op"};
    context.params = {{"min_max_log_file", {min_max_log_file_name}}};

    GraphDef frozen_graph_def;
    TF_EXPECT_OK(
        FreezeRequantizationRanges(graph_def, context, &frozen_graph_def));

    std::map<string, const NodeDef*> node_map;
    MapNamesToNodes(frozen_graph_def, &node_map);
    EXPECT_EQ(0, node_map.count("requantization_range_op"));
    EXPECT_EQ(1, node_map.count("requantize_op"));
    const string& min_input =
        NodeNameFromInput(node_map.at("requantize_op")->input(3));
    ASSERT_EQ(1, node_map.count(min_input));
    EXPECT_EQ("Const", node_map.at(min_input)->op());
    const string& max_input =
        NodeNameFromInput(node_map.at("requantize_op")->input(4));
    ASSERT_EQ(1, node_map.count(max_input));
    EXPECT_EQ("Const", node_map.at(max_input)->op());

    std::unique_ptr<Session> original_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(original_session->Create(graph_def));
    std::vector<Tensor> original_outputs;
    TF_ASSERT_OK(
        original_session->Run({}, {"dequantize_op"}, {}, &original_outputs));

    std::unique_ptr<Session> frozen_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(frozen_session->Create(frozen_graph_def));
    std::vector<Tensor> frozen_outputs;
    TF_ASSERT_OK(
        frozen_session->Run({}, {"dequantize_op"}, {}, &frozen_outputs));

    ASSERT_EQ(original_outputs.size(), frozen_outputs.size());
    ASSERT_EQ(1, frozen_outputs.size());
    test::ExpectTensorNear<float>(original_outputs[0], frozen_outputs[0], 0.5);
  }

  void TestExtractMinMaxRecords() {
    const string min_max_log_file_name =
        io::JoinPath(testing::TmpDir(), "min_max_log_file2.txt");
    {
      std::unique_ptr<WritableFile> file;
      TF_ASSERT_OK(
          Env::Default()->NewWritableFile(min_max_log_file_name, &file));
      TF_ASSERT_OK(file->Append("Something irrelevant\n"));
      TF_ASSERT_OK(
          file->Append("[SomePrefix] "
                       ";requantization_range_op__print__;__requant_min_max:"
                       "[-2.4313571][10.584145]\n"));
      TF_ASSERT_OK(file->Append("Something else irrelevant\n"));
      TF_ASSERT_OK(file->Append(
          "[SomeOtherPrefix] "
          ";other_requantization_range_op__print__;__requant_min_max:"
          "[-1.0][2.0]\n"));
      TF_ASSERT_OK(file->Append("Something else irrelevant\n"));
      TF_ASSERT_OK(
          file->Append("[SomePrefix] "
                       ";requantization_range_op__print__;__requant_min_max:"
                       "[-1.bad][2.0]\n"));
    }
    std::vector<MinMaxRecord> records;
    TF_ASSERT_OK(ExtractMinMaxRecords(min_max_log_file_name, &records));
    ASSERT_EQ(2, records.size());
    EXPECT_EQ("requantization_range_op", records[0].name);
    EXPECT_NEAR(-2.4313571f, records[0].min, 1e-5f);
    EXPECT_NEAR(10.584145f, records[0].max, 1e-5f);
    EXPECT_EQ("other_requantization_range_op", records[1].name);
    EXPECT_NEAR(-1.0f, records[1].min, 1e-5f);
    EXPECT_NEAR(2.0f, records[1].max, 1e-5f);
  }
};

TEST_F(FreezeRequantizationRangesTest, TestFreezeRequantizationRanges) {
  TestFreezeRequantizationRanges();
}

TEST_F(FreezeRequantizationRangesTest, TestExtractMinMaxRecords) {
  TestExtractMinMaxRecords();
}

}  // namespace graph_transforms
}  // namespace tensorflow
