/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/parse_example/parse_example.h"

#include <cstdint>
#include <initializer_list>
#include <string>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/core/example/feature_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace ops {
namespace custom {

namespace tf = ::tensorflow;

const char* kNodeDefTxt = R"pb(
  name: "ParseExample/ParseExample"
  op: "ParseExample"
  input: "serialized"
  input: "ParseExample/ParseExample/names"
  input: "ParseExample/ParseExample/dense_keys_0"
  input: "ParseExample/Const"
  attr {
    key: "Ndense"
    value { i: 1 }
  }
  attr {
    key: "Nsparse"
    value { i: 0 }
  }
  attr {
    key: "Tdense"
    value { list { type: DT_FLOAT } }
  }
  attr {
    key: "dense_shapes"
    value { list { shape { dim { size: 2 } } } }
  }
  attr {
    key: "sparse_types"
    value { list { type: DT_FLOAT } }
  }
)pb";

const char* kNodeDefTxt2 = R"pb(
  name: "ParseExample/ParseExample"
  op: "ParseExample"
  input: "serialized"
  input: "ParseExample/ParseExample/names"
  input: "ParseExample/ParseExample/sparse_keys_0"
  attr {
    key: "Ndense"
    value { i: 0 }
  }
  attr {
    key: "Nsparse"
    value { i: 1 }
  }
  attr {
    key: "Tdense"
    value {}
  }
  attr {
    key: "dense_shapes"
    value {}
  }
  attr {
    key: "sparse_types"
    value { list { type: DT_FLOAT } }
  }
)pb";

const char* kNodeDefTxt3 = R"pb(
  name: "ParseExample/ParseExample"
  op: "ParseExample"
  input: "serialized"
  input: "ParseExample/ParseExample/names"
  input: "ParseExample/ParseExample/sparse_keys_0"
  attr {
    key: "Ndense"
    value { i: 1 }
  }
  attr {
    key: "Nsparse"
    value { i: 0 }
  }
  attr {
    key: "Tdense"
    value { list { type: DT_STRING } }
  }
  attr {
    key: "dense_shapes"
    value { list { shape { dim { size: 1 } } } }
  }
  attr {
    key: "sparse_types"
    value { list { type: DT_FLOAT } }
  }
)pb";

const char* kNodeDefTxt4 = R"pb(
  name: "ParseExample/ParseExample"
  op: "ParseExample"
  input: "serialized"
  input: "ParseExample/ParseExample/names"
  input: "ParseExample/ParseExample/sparse_keys_0"
  attr {
    key: "Ndense"
    value { i: 0 }
  }
  attr {
    key: "Nsparse"
    value { i: 1 }
  }
  attr {
    key: "Tdense"
    value {}
  }
  attr {
    key: "dense_shapes"
    value {}
  }
  attr {
    key: "sparse_types"
    value { list { type: DT_STRING } }
  }
)pb";

const char* kNodeDefTxt5 = R"pb(
  name: "ParseExample/ParseExample"
  op: "ParseExample"
  input: "serialized"
  input: "ParseExample/ParseExample/names"
  input: "ParseExample/ParseExample/dense_keys_0"
  input: "ParseExample/Const"
  attr {
    key: "Ndense"
    value { i: 1 }
  }
  attr {
    key: "Nsparse"
    value { i: 0 }
  }
  attr {
    key: "Tdense"
    value { list { type: DT_FLOAT } }
  }
  attr {
    key: "dense_shapes"
    value {}
  }
  attr {
    key: "sparse_types"
    value { list { type: DT_FLOAT } }
  }
)pb";

template <typename DefaultType>
class ParseExampleOpModel : public SingleOpModel {
 public:
  ParseExampleOpModel(std::vector<std::string> serialized_examples,
                      std::vector<std::string> sparse_keys,
                      std::vector<std::string> dense_keys,
                      std::initializer_list<DefaultType> dense_defaults,
                      std::vector<TensorType> dense_types,
                      std::vector<TensorType> sparse_types,
                      const char* text_def, int dense_size = 2) {
    // Example
    const int input_size = serialized_examples.size();
    auto input_tensor_data = TensorData(TensorType_STRING, {input_size});
    string_indices_.push_back(AddInput(input_tensor_data));
    // Names
    string_indices_.push_back(
        AddConstInput<std::string>(TensorData(TensorType_STRING, {0}), {""}));
    std::for_each(sparse_keys.begin(), sparse_keys.end(), [&](auto&&) {
      string_indices_.push_back(AddInput(TensorData(TensorType_STRING, {1})));
    });
    std::for_each(dense_keys.begin(), dense_keys.end(), [&](auto&&) {
      string_indices_.push_back(AddInput(TensorData(TensorType_STRING, {1})));
    });
    if (dense_size > 0) {
      dense_defaults_ = AddConstInput<DefaultType>(
          TensorData(dense_types[0], {dense_size}), dense_defaults);
    }
    if (!sparse_keys.empty()) {
      for (int i = 0; i < sparse_keys.size(); i++) {
        sparse_indices_outputs_.push_back(AddOutput(TensorType_INT64));
      }
      for (int i = 0; i < sparse_keys.size(); i++) {
        sparse_values_outputs_.push_back(AddOutput(sparse_types[i]));
      }
      for (int i = 0; i < sparse_keys.size(); i++) {
        sparse_shapes_outputs_.push_back(AddOutput({TensorType_INT64, {2}}));
      }
    }
    for (int i = 0; i < dense_keys.size(); i++) {
      dense_outputs_.push_back(AddOutput({dense_types[i], {dense_size}}));
    }

    tf::NodeDef nodedef;
    tf::protobuf::TextFormat::Parser parser;
    tf::protobuf::io::ArrayInputStream input_stream(text_def, strlen(text_def));
    if (!parser.Parse(&input_stream, &nodedef)) {
      abort();
    }
    std::string serialized_nodedef;
    nodedef.SerializeToString(&serialized_nodedef);
    flexbuffers::Builder fbb;
    fbb.Vector([&]() {
      fbb.String(nodedef.op());
      fbb.String(serialized_nodedef);
    });
    fbb.Finish();
    const auto buffer = fbb.GetBuffer();
    SetCustomOp("ParseExample", buffer, Register_PARSE_EXAMPLE);
    BuildInterpreter({{input_size}});
    int idx = 0;
    PopulateStringTensor(string_indices_[idx++], serialized_examples);
    PopulateStringTensor(string_indices_[idx++], {""});
    for (const auto& key : sparse_keys) {
      PopulateStringTensor(string_indices_[idx++], {key});
    }
    for (const auto& key : dense_keys) {
      PopulateStringTensor(string_indices_[idx++], {key});
    }
  }

  void ResizeInputTensor(std::vector<std::vector<int>> input_shapes) {
    for (size_t i = 0; i < input_shapes.size(); ++i) {
      const int input_idx = interpreter_->inputs()[i];
      if (input_idx == kTfLiteOptionalTensor) continue;
      const auto& shape = input_shapes[i];
      if (shape.empty()) continue;
      CHECK(interpreter_->ResizeInputTensor(input_idx, shape) == kTfLiteOk);
    }
  }

  template <typename T>
  std::vector<T> GetSparseIndicesOutput(int i) {
    return ExtractVector<T>(sparse_indices_outputs_[i]);
  }

  template <typename T>
  std::vector<T> GetSparseValuesOutput(int i) {
    return ExtractVector<T>(sparse_values_outputs_[i]);
  }

  template <typename T>
  std::vector<T> GetSparseShapesOutput(int i) {
    return ExtractVector<T>(sparse_shapes_outputs_[i]);
  }

  template <typename T>
  std::vector<T> GetDenseOutput(int i) {
    return ExtractVector<T>(dense_outputs_[i]);
  }

  std::vector<std::string> GetStringOutput(int i) {
    auto* t = interpreter_->tensor(i);
    int count = GetStringCount(t);
    std::vector<std::string> v;
    for (int i = 0; i < count; ++i) {
      auto ref = GetString(t, i);
      v.emplace_back(ref.str, ref.len);
    }
    return v;
  }

  int DenseDefaults() { return dense_defaults_; }

  int SparseValuesOutputs(int i) { return sparse_values_outputs_[i]; }

  int DenseOutputs(int i) { return dense_outputs_[i]; }

  std::vector<int> dense_outputs_;
  std::vector<int> sparse_indices_outputs_;
  std::vector<int> sparse_shapes_outputs_;
  std::vector<int> sparse_values_outputs_;
  std::vector<int> string_indices_;
  int dense_defaults_ = -1;
};

TEST(ParseExampleOpsTest, SimpleTest) {
  tf::Example example;
  tf::AppendFeatureValues<float>({1.5f, 1.5f}, "time", &example);
  tf::AppendFeatureValues<float>({1.0f, 1.0f}, "num", &example);
  ParseExampleOpModel<float> m({example.SerializeAsString()}, {}, {"time"},
                               {0.f, 0.f}, {TensorType_FLOAT32}, {},
                               kNodeDefTxt);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDenseOutput<float>(0),
              ElementsAreArray(ArrayFloatNear({1.5f, 1.5f})));
}

TEST(ParseExampleOpsTest, SparseTest) {
  tf::Example example;
  tf::AppendFeatureValues<float>({1.5f}, "time", &example);
  ParseExampleOpModel<float> m({example.SerializeAsString()}, {"time"}, {}, {},
                               {}, {TensorType_FLOAT32}, kNodeDefTxt2, 0);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetSparseIndicesOutput<int64_t>(0),
              ElementsAreArray(ArrayFloatNear({0, 0})));
  EXPECT_THAT(m.GetSparseValuesOutput<float>(0),
              ElementsAreArray(ArrayFloatNear({1.5f})));
  EXPECT_THAT(m.GetSparseShapesOutput<int64_t>(0),
              ElementsAreArray(ArrayFloatNear({1, 1})));
}

TEST(ParseExampleOpsTest, SimpleBytesTest) {
  tf::Example example;
  const std::string test_data = "simpletest";
  tf::AppendFeatureValues<tensorflow::tstring>({test_data}, "time", &example);
  tf::AppendFeatureValues<float>({1.0f, 1.0f}, "num", &example);
  std::string default_value = "missing";
  ParseExampleOpModel<std::string> m({example.SerializeAsString()}, {},
                                     {"time"}, {default_value},
                                     {TensorType_STRING}, {}, kNodeDefTxt3, 1);
  m.PopulateStringTensor(m.DenseDefaults(), {default_value});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  std::vector<string> c = m.GetStringOutput(m.DenseOutputs(0));
  EXPECT_EQ(1, c.size());
  EXPECT_EQ(test_data, c[0]);
}

TEST(ParseExampleOpsTest, SparseBytesTest) {
  tf::Example example;
  const std::string test_data = "simpletest";
  tf::AppendFeatureValues<tensorflow::tstring>({test_data, test_data}, "time",
                                               &example);
  tf::AppendFeatureValues<float>({1.0f, 1.0f}, "num", &example);
  ParseExampleOpModel<std::string> m({example.SerializeAsString()}, {"time"},
                                     {}, {}, {}, {TensorType_STRING},
                                     kNodeDefTxt4, 0);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetSparseIndicesOutput<int64_t>(0),
              testing::ElementsAreArray({0, 0, 0, 1}));
  auto values = m.GetStringOutput(m.SparseValuesOutputs(0));
  EXPECT_EQ(2, values.size());
  EXPECT_EQ(test_data, values[0]);
  EXPECT_EQ(test_data, values[1]);
  EXPECT_THAT(m.GetSparseShapesOutput<int64_t>(0),
              testing::ElementsAreArray({1, 2}));
}

TEST(ParseExampleOpsTest, ResizeTest) {
  const int num_tests = 3;
  std::vector<tf::Example> examples(num_tests);
  std::vector<std::vector<float>> expected(num_tests);
  std::vector<std::vector<std::string>> inputs(num_tests);
  std::vector<int> sizes;
  for (int i = 0; i < num_tests; ++i) {
    float val = i;
    std::initializer_list<float> floats = {val + val / 10.f, -val - val / 10.f};
    tf::AppendFeatureValues<float>({val, val}, "num", &examples[i]);
    tf::AppendFeatureValues<float>(floats, "time", &examples[i]);
    sizes.push_back((num_tests - i) * 2);
    for (int j = 0; j < sizes.back(); ++j) {
      inputs[i].push_back(examples[i].SerializeAsString());
      expected[i].insert(expected[i].end(), floats.begin(), floats.end());
    }
  }

  ParseExampleOpModel<float> m(inputs[0], {}, {"time"}, {0.f, 0.f},
                               {TensorType_FLOAT32}, {}, kNodeDefTxt);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDenseOutput<float>(0),
              ElementsAreArray(ArrayFloatNear(expected[0])));

  for (int i = 1; i < num_tests; ++i) {
    m.ResizeInputTensor({{sizes[i]}});
    m.AllocateAndDelegate(false);
    m.PopulateStringTensor(0, inputs[i]);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetDenseOutput<float>(0),
                ElementsAreArray(ArrayFloatNear(expected[i])));
  }
}

TEST(ParseExampleOpsTest, ResizeMissingInfoTest) {
  const int num_tests = 3;
  std::vector<tf::Example> examples(num_tests);
  std::vector<std::vector<float>> expected(num_tests);
  std::vector<std::vector<std::string>> inputs(num_tests);
  std::vector<int> sizes;
  for (int i = 0; i < num_tests; ++i) {
    float val = i;
    std::initializer_list<float> floats = {val + val / 10.f, -val - val / 10.f};
    tf::AppendFeatureValues<float>({val, val}, "num", &examples[i]);
    tf::AppendFeatureValues<float>(floats, "time", &examples[i]);
    sizes.push_back((num_tests - i) * 2);
    for (int j = 0; j < sizes.back(); ++j) {
      inputs[i].push_back(examples[i].SerializeAsString());
      expected[i].insert(expected[i].end(), floats.begin(), floats.end());
    }
  }

  ParseExampleOpModel<float> m(inputs[0], {}, {"time"}, {0.f, 0.f},
                               {TensorType_FLOAT32}, {}, kNodeDefTxt5);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDenseOutput<float>(0),
              ElementsAreArray(ArrayFloatNear(expected[0])));

  for (int i = 1; i < num_tests; ++i) {
    m.ResizeInputTensor({{sizes[i]}});
    m.AllocateAndDelegate(false);
    m.PopulateStringTensor(0, inputs[i]);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetDenseOutput<float>(0),
                ElementsAreArray(ArrayFloatNear(expected[i])));
  }
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
