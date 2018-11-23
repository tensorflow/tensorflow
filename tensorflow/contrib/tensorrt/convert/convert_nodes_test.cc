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

#include "tensorflow/contrib/tensorrt/convert/convert_nodes.h"

#include <memory>
#include <unordered_map>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/contrib/tensorrt/log/trt_logger.h"
#include "tensorflow/contrib/tensorrt/plugin/trt_plugin_factory.h"
#include "tensorflow/core/framework/node_def.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"  // NOLINT
#include "tensorflow/core/public/session.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "cuda/include/cuda.h"
#include "cuda/include/cuda_runtime_api.h"
#include "tensorrt/include/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

using ::testing::ElementsAre;

// TODO(laigd): put this into some test utils file.
void ExpectStatus(Status status, error::Code code = error::OK,
                  const char* substr = nullptr) {
  EXPECT_EQ(code, status.code())
      << status << " vs expected error code \"" << error::Code_Name(code)
      << "\" and message \"" << substr << "\"";
  if (substr) {
    EXPECT_THAT(status.error_message(), ::testing::HasSubstr(substr)) << status;
  }
}

nvinfer1::Dims GetTestDims(const std::vector<int>& d) {
  nvinfer1::Dims dims;
  dims.nbDims = d.size();
  for (int i = 0; i < d.size(); ++i) {
    dims.d[i] = d[i];
  }
  return dims;
}

NodeDef MakeNodeDef(const string& name, const string& op,
                    const std::vector<string>& inputs) {
  NodeDef node_def;
  node_def.set_name(name);
  node_def.set_op(op);
  for (const string& input : inputs) {
    node_def.add_input(input);
  }
  return node_def;
}

template <typename T>
NodeDef MakeConstNodeDef(const string& name, const std::vector<T>& vals,
                         const TensorShape& shape) {
  Scope s = Scope::NewRootScope();
  Tensor t = ::tensorflow::test::AsTensor<T>(vals, shape);
  auto const_op = ops::Const(s.WithOpName(name), t);
  return const_op.node()->def();
}

template <typename T>
NodeDef MakeConstNodeDef(const string& name, const std::vector<T>& vals) {
  TensorShape shape;
  const std::vector<int32> shape_dims = {static_cast<int32>(vals.size())};
  TF_EXPECT_OK(TensorShapeUtils::MakeShape(shape_dims, &shape));
  return MakeConstNodeDef(name, vals, shape);
}

bool TrtDimsEquals(const nvinfer1::Dims& lhs, const nvinfer1::Dims& rhs) {
  if (lhs.nbDims != rhs.nbDims) return false;
  for (int i = 0; i < lhs.nbDims; ++i) {
    if (lhs.d[i] != rhs.d[i]) return false;
    // We don't check the types in the tests.
  }
  return true;
}

bool TrtDimsEqualsArray(const std::vector<int>& lhs,
                        const nvinfer1::Dims& rhs) {
  return TrtDimsEquals(GetTestDims(lhs), rhs);
}

bool TrtShapedWeightsEquals(const TRT_ShapedWeights& lhs,
                            const TRT_ShapedWeights& rhs) {
  return TrtDimsEquals(lhs.shape_, rhs.shape_) && lhs.type_ == rhs.type_ &&
         lhs.GetValues() == rhs.GetValues();
}

template <typename T>
void ValidateWeights(const TRT_ShapedWeights& weights,
                     const std::vector<int>& expected_dims,
                     const std::vector<T>& expected_value) {
  EXPECT_TRUE(TrtDimsEqualsArray(expected_dims, weights.shape_))
      << weights.DebugString();
  ASSERT_EQ(expected_value.size(), weights.count()) << weights.DebugString();
  const T* actual_values = static_cast<const T*>(weights.GetValues());
  for (int i = 0; i < expected_value.size(); ++i) {
    EXPECT_EQ(expected_value[i], actual_values[i]);
  }
}

// Fake ITensor implementation for testing purposes.
class FakeITensor : public nvinfer1::ITensor {
 public:
  FakeITensor() {}

  FakeITensor(const nvinfer1::Dims& dims) : dims_(dims) {}

  FakeITensor(const std::vector<int>& dims) : dims_(GetTestDims(dims)) {}

  void setName(const char* name) override { name_ = name; }

  const char* getName() const override { return name_.c_str(); }

  void setDimensions(nvinfer1::Dims dimensions) override { dims_ = dimensions; }

  nvinfer1::Dims getDimensions() const override { return dims_; }

  void setType(nvinfer1::DataType type) override { type_ = type; }

  nvinfer1::DataType getType() const override { return type_; }

  bool isNetworkInput() const override { return false; }

  bool isNetworkOutput() const override { return false; }

  void setBroadcastAcrossBatch(bool broadcastAcrossBatch) override {}

  bool getBroadcastAcrossBatch() const override { return false; }

  nvinfer1::TensorLocation getLocation() const override { return location_; }

  void setLocation(nvinfer1::TensorLocation location) override {
    location_ = location;
  }

#if NV_TENSORRT_MAJOR >= 5
  bool setDynamicRange(float min, float max) override {}
#endif

 private:
  string name_;
  nvinfer1::Dims dims_;
  nvinfer1::DataType type_;
  nvinfer1::TensorLocation location_;
};

TEST(TRT_ShapedWeights_Test, Basic) {
  // Test constructor with no arguments.
  {
    TRT_ShapedWeights weights;
    TRT_ShapedWeights copy(weights);
    for (auto ptr : {&weights, &copy}) {
      nvinfer1::Weights trt_weights = ptr->GetTrtWeights();
      EXPECT_EQ(nvinfer1::DataType::kFLOAT, trt_weights.type);
      EXPECT_EQ(nullptr, trt_weights.values);
      EXPECT_EQ(0, trt_weights.count);

      EXPECT_EQ(nullptr, ptr->GetValues());
      EXPECT_EQ(0, ptr->count());
      EXPECT_EQ(0, ptr->size_bytes());
    }
  }
  // Test constructor with DataType argument.
  {
    TRT_ShapedWeights weights(DT_FLOAT);
    TRT_ShapedWeights copy(weights);
    for (auto ptr : {&weights, &copy}) {
      nvinfer1::Weights trt_weights = ptr->GetTrtWeights();
      EXPECT_EQ(nvinfer1::DataType::kFLOAT, trt_weights.type);
      EXPECT_EQ(nullptr, trt_weights.values);
      EXPECT_EQ(0, trt_weights.count);

      EXPECT_EQ(nullptr, ptr->GetValues());
      EXPECT_EQ(0, ptr->count());
      EXPECT_EQ(0, ptr->size_bytes());
    }
  }
  // Test constructor with DataType and nvinfer1::Dims arguments.
  {
    TrtWeightStore store;
    TRT_ShapedWeights weights =
        store.GetTempWeights(DT_FLOAT, GetTestDims({2, 5}));
    TRT_ShapedWeights copy(weights);
    for (auto ptr : {&weights, &copy}) {
      nvinfer1::Weights trt_weights = ptr->GetTrtWeights();
      EXPECT_EQ(nvinfer1::DataType::kFLOAT, trt_weights.type);
      EXPECT_NE(nullptr, trt_weights.values);
      EXPECT_EQ(10, trt_weights.count);

      EXPECT_EQ(trt_weights.values, ptr->GetValues());
      EXPECT_EQ(10, ptr->count());
      EXPECT_EQ(40, ptr->size_bytes());
    }
    // Test that it doesn't copy the underlying buffer.
    EXPECT_EQ(weights.GetValues(), copy.GetValues());
  }
}

TEST(TRT_TensorOrWeights_Test, Basic) {
  // Test constructor with no arguments.
  {
    TRT_TensorOrWeights tw;
    TRT_TensorOrWeights copy(tw);
    TRT_TensorOrWeights assigned;
    assigned = tw;
    for (auto ptr : {&tw, &copy, &assigned}) {
      EXPECT_EQ(false, ptr->is_tensor());
      EXPECT_EQ(false, ptr->is_weights());
      EXPECT_EQ(-1, ptr->batch_size());
    }
  }

  // Test constructor with ITensor and batch size argument.
  {
    nvinfer1::Dims dims;
    dims.nbDims = 1;
    dims.d[0] = 1;
    FakeITensor itensor(dims);
    TRT_TensorOrWeights tw(&itensor);
    TRT_TensorOrWeights tw1(&itensor, /*batch_size=*/1);

    for (auto original_ptr : {&tw, &tw1}) {
      TRT_TensorOrWeights copy(*original_ptr);
      TRT_TensorOrWeights assigned;
      assigned = *original_ptr;

      for (auto ptr : {original_ptr, &copy, &assigned}) {
        EXPECT_EQ(true, ptr->is_tensor());
        EXPECT_EQ(false, ptr->is_weights());
        if (original_ptr == &tw) {
          EXPECT_EQ(-1, ptr->batch_size());
        } else {
          EXPECT_EQ(1, ptr->batch_size());
        }
        EXPECT_EQ(&itensor, ptr->tensor());
        EXPECT_TRUE(TrtDimsEqualsArray({1}, ptr->GetTrtDims()))
            << "- expected: " << DebugString(dims)
            << "\n        vs\n-   actual: " << DebugString(ptr->GetTrtDims());
      }
    }
  }
  // Test constructor which creates and owns an ITensor.
  {
    nvinfer1::Dims dims;
    dims.nbDims = 1;
    dims.d[0] = 1;
    TRT_TensorOrWeights tw(nvinfer1::DataType::kFLOAT, dims, /*batch_size=*/1);
    TRT_TensorOrWeights copy(tw);
    TRT_TensorOrWeights assigned;
    assigned = tw;

    for (auto ptr : {&tw, &copy, &assigned}) {
      EXPECT_EQ(true, ptr->is_tensor());
      EXPECT_EQ(false, ptr->is_weights());
      EXPECT_EQ(1, ptr->batch_size());
      EXPECT_NE(nullptr, ptr->tensor());
      EXPECT_TRUE(TrtDimsEqualsArray({1}, ptr->GetTrtDims()))
          << "- expected: " << DebugString(dims)
          << "\n        vs\n-   actual: " << DebugString(ptr->GetTrtDims());
    }
  }
  // Test constructor with TRT_ShapedWeights argument.
  {
    TRT_ShapedWeights weights;
    TRT_TensorOrWeights tw(weights);
    TRT_TensorOrWeights copy(tw);
    TRT_TensorOrWeights assigned;
    assigned = tw;
    for (auto ptr : {&tw, &copy, &assigned}) {
      EXPECT_EQ(false, ptr->is_tensor());
      EXPECT_EQ(true, ptr->is_weights());
      EXPECT_TRUE(TrtShapedWeightsEquals(weights, ptr->weights()));

      nvinfer1::Dims dims;
      dims.nbDims = 0;
      EXPECT_TRUE(TrtDimsEqualsArray({}, ptr->GetTrtDims()))
          << "- expected: " << DebugString(dims)
          << "\n        vs\n-   actual: " << DebugString(ptr->GetTrtDims());
    }
  }
}

class ValidatorTest : public ::testing::Test {
 public:
  void AddOpValidator(const string& op_name, OpConverter op_validator) {
    validator_.op_validators_[op_name] = op_validator;
  }

  Status ConvertToTensorOrWeights(
      const NodeDef& node_def, int output_port,
      const grappler::GraphProperties& graph_properties,
      TRT_TensorOrWeights* tensor_or_weights) {
    return validator_.ConvertToTensorOrWeights(
        node_def, output_port, graph_properties, tensor_or_weights);
  }

 protected:
  TrtNodeValidator validator_;
};

TEST_F(ValidatorTest, ConvertToTensorOrWeights) {
  // Convert Const.
  {
    NodeDef node_def = MakeConstNodeDef<float>("my_const", {1.0f, 2.0f});
    TRT_TensorOrWeights output;
    grappler::GrapplerItem item;
    grappler::GraphProperties graph_properties(item);
    ExpectStatus(ConvertToTensorOrWeights(node_def, /*output_port=*/0,
                                          graph_properties, &output));
    ValidateWeights<float>(output.weights(), {2}, {1.0, 2.0});
  }

  // Helper method to run ConvertToTensorOrWeights() with predefined parameters.
  auto convert_to_tensor_or_weights = [this](const std::vector<int64>& dims,
                                             TRT_TensorOrWeights* output) {
    Scope s = Scope::NewRootScope();
    const auto attrs = ops::Placeholder::Shape(PartialTensorShape{dims});
    auto feed = ops::Placeholder(s.WithOpName("feed"), DT_FLOAT, attrs);
    auto add = ops::Add(s.WithOpName("add"), feed, feed);

    grappler::GrapplerItem item;
    TF_EXPECT_OK(s.ToGraphDef(&item.graph));
    grappler::GraphProperties graph_properties(item);
    TF_EXPECT_OK(graph_properties.InferStatically(true));
    const NodeDef& node_def = add.operation.node()->def();
    return this->ConvertToTensorOrWeights(node_def, /*output_port=*/0,
                                          graph_properties, output);
  };
  // Convert non-Const with #dims > nvinfer1::Dims::MAX_DIMS+1.
  {
    TRT_TensorOrWeights output;
    ExpectStatus(
        convert_to_tensor_or_weights(
            std::vector<int64>(nvinfer1::Dims::MAX_DIMS + 2, 1), &output),
        error::OUT_OF_RANGE, "Input tensor rank is greater than 9");
  }
  // Convert non-Const with #dims < 2.
  {
    TRT_TensorOrWeights output;
    ExpectStatus(
        convert_to_tensor_or_weights({1}, &output), error::INVALID_ARGUMENT,
        "Input tensor with rank<2 is not supported since the first dimension "
        "is treated as batch dimension by TRT");
  }
  // Convert non-Const. We test the case where the non-batch dimemsion is
  // unknown as well, to make sure the validator allows that.
  for (const int32 non_batch_dim : {-1, 2}) {
    const int32 batch_size = 12;
    TRT_TensorOrWeights output;
    ExpectStatus(
        convert_to_tensor_or_weights({batch_size, non_batch_dim}, &output));
    EXPECT_EQ(true, output.is_tensor());
    EXPECT_EQ(batch_size, output.batch_size());
    EXPECT_NE(nullptr, output.tensor());
    EXPECT_TRUE(TrtDimsEqualsArray({non_batch_dim}, output.GetTrtDims()))
        << "- expected: {" << non_batch_dim << "} \n        vs\n"
        << "-   actual: " << DebugString(output.GetTrtDims());
  }
}

TEST_F(ValidatorTest, ValidateNode) {
  grappler::GrapplerItem item;
  grappler::GraphProperties graph_properties(item);

  bool start_conversion = false;
  bool should_fail = false;
  auto op_converter = [&start_conversion,
                       &should_fail](OpConverterParams* params) -> Status {
    if (should_fail) return errors::InvalidArgument("");
    if (!params->validation_only) start_conversion = true;
    return Status::OK();
  };
  NodeDef node_def = MakeNodeDef("my_op", "MyOp", {});

  // Validator not registered, validation should pass.
  TF_EXPECT_OK(validator_.ValidateNode(node_def, {}, graph_properties));

  // Register validator.
  AddOpValidator("MyOp", op_converter);
  TF_EXPECT_OK(validator_.ValidateNode(node_def, {}, graph_properties));
  EXPECT_EQ(false, start_conversion);

  // Let the converter return error.
  should_fail = true;
  ExpectStatus(validator_.ValidateNode(node_def, {}, graph_properties),
               error::INVALID_ARGUMENT);
}

class ConverterTest : public ::testing::Test {
 public:
  ConverterTest() {
    builder_.reset(nvinfer1::createInferBuilder(logger_));
    network_.reset(builder_->createNetwork());
    converter_.reset(new Converter(network_.get(), /*fp16=*/false));
    weight_store_ = &converter_->weight_store_;
  }

  void AddOpConverter(const string& op_name, OpConverter op_converter) {
    converter_->op_registry_[op_name] = op_converter;
  }

  // Below we expose private methods of Converter for testing.

  Status MaybeUpdateBatchSize(int batch_size) {
    return converter_->MaybeUpdateBatchSize(batch_size);
  }

  Status AddTensorOrWeights(const string& name, TRT_TensorOrWeights input) {
    return converter_->AddTensorOrWeights(name, input);
  }

  Status GetTensorOrWeights(const string& name, TRT_TensorOrWeights* output) {
    return converter_->GetTensorOrWeights(name, output);
  }

  Status GetInputs(const NodeDef& node_def,
                   std::vector<TRT_TensorOrWeights>* inputs) const {
    return converter_->GetInputs(node_def, inputs);
  }

  int batch_size() const { return converter_->batch_size_; }

 private:
  Logger logger_;
  // These members are ordered in a way such that the destruction order is:
  // converter_ -> network_ -> builder_
  TrtUniquePtrType<nvinfer1::IBuilder> builder_;
  TrtUniquePtrType<nvinfer1::INetworkDefinition> network_;

 protected:
  std::unique_ptr<Converter> converter_;
  TrtWeightStore* weight_store_;
};

TEST_F(ConverterTest, ConvertNode) {
  FakeITensor output_tensors[2];
  auto op_converter = [&output_tensors](OpConverterParams* params) -> Status {
    nvinfer1::Dims dims = params->inputs[0].tensor()->getDimensions();
    for (int i = 0; i < 2; ++i) {
      dims.d[0] += 1;
      output_tensors[i].setDimensions(dims);
      params->outputs->push_back(TRT_TensorOrWeights(&output_tensors[i]));
    }
    return Status::OK();
  };
  NodeDef node_def = MakeNodeDef("my_op", "MyOp", {"my_input"});
  TF_EXPECT_OK(converter_->AddInputTensor(
      "my_input", nvinfer1::DataType::kFLOAT, GetTestDims({123}), 1));

  // Converter not registered.
  ExpectStatus(converter_->ConvertNode(node_def), error::UNIMPLEMENTED,
               "No converter registered for op: MyOp");

  // Register the converter and retry.
  AddOpConverter("MyOp", op_converter);
  TF_EXPECT_OK(converter_->ConvertNode(node_def));

  TRT_TensorOrWeights actual_output_1;
  TF_EXPECT_OK(GetTensorOrWeights("my_op", &actual_output_1));
  EXPECT_EQ(&output_tensors[0], actual_output_1.tensor());
  EXPECT_EQ(124, actual_output_1.tensor()->getDimensions().d[0]);

  TRT_TensorOrWeights actual_output_2;
  TF_EXPECT_OK(GetTensorOrWeights("my_op:1", &actual_output_2));
  EXPECT_EQ(&output_tensors[1], actual_output_2.tensor());
  EXPECT_EQ(125, actual_output_2.tensor()->getDimensions().d[0]);
}

TEST_F(ConverterTest, AddAndGetInputs) {
  NodeDef node_def;
  node_def.add_input("^control_input");
  node_def.add_input("input");
  node_def.add_input("input:0");
  node_def.add_input("input:1");
  node_def.add_input("weird_input:2:3:4:0");

  TF_EXPECT_OK(converter_->AddInputTensor("input", nvinfer1::DataType::kFLOAT,
                                          GetTestDims({1}), 1));
  TF_EXPECT_OK(converter_->AddInputTensor("input:1", nvinfer1::DataType::kINT32,
                                          GetTestDims({2, 3}), 1));
  TF_EXPECT_OK(converter_->AddInputTensor(
      "weird_input:2:3:4", nvinfer1::DataType::kHALF, GetTestDims({5, 3}), 1));

  std::vector<TRT_TensorOrWeights> inputs;
  TF_EXPECT_OK(GetInputs(node_def, &inputs));

  EXPECT_EQ(4, inputs.size());
  EXPECT_EQ(inputs[0].tensor(), inputs[1].tensor());

  EXPECT_EQ(nvinfer1::DataType::kFLOAT, inputs[0].tensor()->getType());
  EXPECT_EQ(nvinfer1::DataType::kINT32, inputs[2].tensor()->getType());
  EXPECT_EQ(nvinfer1::DataType::kHALF, inputs[3].tensor()->getType());
  EXPECT_TRUE(TrtDimsEqualsArray({1}, inputs[0].tensor()->getDimensions()));
  EXPECT_TRUE(TrtDimsEqualsArray({2, 3}, inputs[2].tensor()->getDimensions()));
  EXPECT_TRUE(TrtDimsEqualsArray({5, 3}, inputs[3].tensor()->getDimensions()));
}

TEST_F(ConverterTest, RenameAndMarkOutputTensors) {
  // Test that the tensor are actually named and marked as output after
  // Converter::RenameAndMarkOutputTensors() is called.

  // Register a custom converter which shuffles the input. We use it to build a
  // TRT network whose output will be later marked.
  std::vector<nvinfer1::ITensor*> output_tensors;
  auto op_converter = [&output_tensors](OpConverterParams* params) -> Status {
    nvinfer1::Permutation perm;
    perm.order[0] = 1;
    perm.order[1] = 0;
    for (int i = 0; i < 2; ++i) {
      nvinfer1::ITensor* input_tensor =
          const_cast<nvinfer1::ITensor*>(params->inputs[0].tensor());
      nvinfer1::IShuffleLayer* layer =
          params->converter->network()->addShuffle(*input_tensor);
      layer->setFirstTranspose(perm);
      nvinfer1::ITensor* output_tensor = layer->getOutput(0);
      params->outputs->emplace_back(output_tensor);
      output_tensors.push_back(output_tensor);
    }
    TRT_ShapedWeights output_weights(DT_FLOAT);
    params->outputs->emplace_back(output_weights);
    return Status::OK();
  };
  AddOpConverter("MyOp", op_converter);

  // Run the conversion.
  NodeDef node_def = MakeNodeDef("my_op", "MyOp", {"my_input"});
  TF_EXPECT_OK(converter_->AddInputTensor(
      "my_input", nvinfer1::DataType::kFLOAT, GetTestDims({1, 2}), 1));
  TF_EXPECT_OK(converter_->ConvertNode(node_def));

  // Mark a weight as output, should fail.
  ExpectStatus(
      converter_->RenameAndMarkOutputTensors({{"my_op:2", "my_output"}}),
      error::INVALID_ARGUMENT, "Output my_op:2 is weights not tensor");

  // Mark tensors as output, should pass.
  TF_EXPECT_OK(converter_->RenameAndMarkOutputTensors(
      {{"my_op", "my_output"}, {"my_op:1", "my_output_1"}}));
  EXPECT_EQ(2, output_tensors.size());
  for (auto output_tensor : output_tensors) {
    EXPECT_TRUE(TrtDimsEqualsArray({2, 1}, output_tensor->getDimensions()));
  }
  EXPECT_EQ("my_output", string(output_tensors[0]->getName()));
  EXPECT_EQ("my_output_1", string(output_tensors[1]->getName()));
}

TEST_F(ConverterTest, TransposeTensor) {
  nvinfer1::ITensor* input_tensor = converter_->network()->addInput(
      "", nvinfer1::DataType::kFLOAT, GetTestDims({2, 3, 5}));
  const nvinfer1::ITensor* output_tensor = nullptr;

  // Rank doesn't match.
  ExpectStatus(
      converter_->TransposeTensor(input_tensor, {0, 1}, &output_tensor),
      error::INVALID_ARGUMENT,
      "Rank of perm for transpose does not match with that of the input");

  // Transpose at batch dimension.
  ExpectStatus(
      converter_->TransposeTensor(input_tensor, {1, 0, 2, 3}, &output_tensor),
      error::UNIMPLEMENTED, "Transpose at batch dimension is not supported.");

  // OK.
  TF_EXPECT_OK(
      converter_->TransposeTensor(input_tensor, {0, 3, 1, 2}, &output_tensor));
  EXPECT_TRUE(TrtDimsEqualsArray({5, 2, 3}, output_tensor->getDimensions()))
      << DebugString(*output_tensor);
}

TEST_F(ConverterTest, PrepareTensorForShape_Tensor) {
  nvinfer1::ITensor* input_tensor = converter_->network()->addInput(
      "", nvinfer1::DataType::kFLOAT, GetTestDims({2, 3, 5}));
  TRT_TensorOrWeights tw(input_tensor);
  const nvinfer1::ITensor* output_tensor = nullptr;

  // Shape size doesn't match.
  ExpectStatus(converter_->PrepareTensorForShape(tw, GetTestDims({2, 3, 6}),
                                                 &output_tensor),
               error::INVALID_ARGUMENT, "Reshape shapes are not compatible.");

  // TODO(aaroey): we should check the case where uninferred dimensions are not
  // an exact divisor of input dim ensions, e.g. for dims {-1, 7}.

  // Infer shape, ok.
  TF_EXPECT_OK(converter_->PrepareTensorForShape(tw, GetTestDims({-1, 2}),
                                                 &output_tensor));
  EXPECT_TRUE(TrtDimsEqualsArray({15, 2}, output_tensor->getDimensions()))
      << DebugString(*output_tensor);

  // Regular shape.
  TF_EXPECT_OK(converter_->PrepareTensorForShape(tw, GetTestDims({10, 3}),
                                                 &output_tensor));
  EXPECT_TRUE(TrtDimsEqualsArray({10, 3}, output_tensor->getDimensions()))
      << DebugString(*output_tensor);
}

TEST_F(ConverterTest, PrepareTensorForShape_Weights) {
  TRT_ShapedWeights weights =
      weight_store_->GetTempWeights(DT_FLOAT, GetTestDims({2, 3, 5}));
  TRT_TensorOrWeights tw(weights);
  const nvinfer1::ITensor* output_tensor = nullptr;
  TF_EXPECT_OK(converter_->PrepareTensorForShape(tw, GetTestDims({10, 3}),
                                                 &output_tensor));
  EXPECT_TRUE(TrtDimsEqualsArray({10, 3}, output_tensor->getDimensions()))
      << DebugString(*output_tensor);
}

TEST_F(ConverterTest, MaybeUpdateBatchSize) {
  EXPECT_EQ(-1, batch_size());

  TF_EXPECT_OK(MaybeUpdateBatchSize(-1));
  EXPECT_EQ(-1, batch_size());

  TF_EXPECT_OK(MaybeUpdateBatchSize(123));
  EXPECT_EQ(123, batch_size());

  TF_EXPECT_OK(MaybeUpdateBatchSize(123));
  EXPECT_EQ(123, batch_size());

  TF_EXPECT_OK(MaybeUpdateBatchSize(-1));
  EXPECT_EQ(123, batch_size());

  ExpectStatus(MaybeUpdateBatchSize(124), error::INVALID_ARGUMENT,
               "Provided batch size does not match converter batch size");
}

TEST_F(ConverterTest, AddAndGetTensorOrWeights) {
  // Add a tensor.
  FakeITensor fake_tensor;
  TRT_TensorOrWeights tensor(&fake_tensor);
  EXPECT_EQ(-1, tensor.batch_size());
  TF_EXPECT_OK(MaybeUpdateBatchSize(123));
  TF_EXPECT_OK(AddTensorOrWeights("my_tensor", tensor));

  // Get the added tensor.
  TRT_TensorOrWeights added_tensor;
  TF_EXPECT_OK(GetTensorOrWeights("my_tensor", &added_tensor));
  EXPECT_EQ(123, added_tensor.batch_size());

  // Add the same tensor again.
  ExpectStatus(AddTensorOrWeights("my_tensor", tensor), error::ALREADY_EXISTS,
               "tensor/weights my_tensor already exist");
}

// Class to test various op converters, using both a TrtNodeValidator and
// Converter.
class OpConverterTest : public ::testing::Test {
 public:
  OpConverterTest() : scope_(Scope::NewRootScope()) {
    QCHECK_EQ(0, cudaStreamCreate(&stream_));
    Reset();
  }

  ~OpConverterTest() override { QCHECK_EQ(0, cudaStreamDestroy(stream_)); }

  Status GetTensorOrWeights(const string& name, TRT_TensorOrWeights* output) {
    return converter_->GetTensorOrWeights(name, output);
  }

  void Reset() {
    validator_.reset(nullptr);
    converter_.reset(nullptr);

    // Reset the INetworkDefinition.
    engine_.reset(nullptr);
    network_.reset(nullptr);
    builder_.reset(nvinfer1::createInferBuilder(logger_));
    network_.reset(builder_->createNetwork());
    builder_->setMaxBatchSize(1);

    // Reset the validator and converter.
    validator_.reset(new TrtNodeValidator);
    converter_.reset(new Converter(network_.get(), /*fp16=*/false));

    // Reset other related artifacts.
    scope_ = Scope::NewRootScope();
    validator_inputs_.clear();
  }

  // TODO(laigd): test fp16 and int8 support.
  void BuildAndRun(const char* input_name, const std::vector<float>& input_data,
                   const char* output_name, std::vector<float>* output_data) {
    // Mark the output tensor as TRT engine output.
    TF_EXPECT_OK(converter_->RenameAndMarkOutputTensors(
        {{string(output_name), string(output_name)}}));

    // Build the TRT engine.
    ASSERT_EQ(nullptr, engine_.get());
    engine_.reset(builder_->buildCudaEngine(*converter_->network()));
    CHECK_NOTNULL(engine_.get());

    // Execute the TRT engine.
    const int input_size = input_data.size() * sizeof(float);
    const int output_size = output_data->size() * sizeof(float);
    const int input_index = engine_->getBindingIndex(input_name);
    const int output_index = engine_->getBindingIndex(output_name);

    ASSERT_EQ(engine_->getNbBindings(), 2);
    void* buffers[2];
    ASSERT_EQ(0, cudaMalloc(&buffers[input_index], input_size));
    ASSERT_EQ(0, cudaMalloc(&buffers[output_index], output_size));
    ASSERT_EQ(0, cudaMemcpyAsync(buffers[input_index], input_data.data(),
                                 input_size, cudaMemcpyHostToDevice, stream_));
    TrtUniquePtrType<nvinfer1::IExecutionContext> execution_context(
        engine_->createExecutionContext());
    execution_context->enqueue(/*batchSize=*/1, buffers, stream_, nullptr);
    ASSERT_EQ(0, cudaMemcpyAsync(output_data->data(), buffers[output_index],
                                 output_size, cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_);
    ASSERT_EQ(0, cudaFree(buffers[input_index]));
    ASSERT_EQ(0, cudaFree(buffers[output_index]));
  }

  bool HasStaticShape(const nvinfer1::Dims& dims) const {
    if (dims.nbDims < 0) return false;
    for (int i = 0; i < dims.nbDims; ++i) {
      if (dims.d[i] < 0) return false;
    }
    return true;
  }

  // Add ITensor for both validation and conversion.
  void AddTestTensor(
      const char* name, const std::vector<int32>& dims, int batch_size = 1,
      nvinfer1::DataType trt_dtype = nvinfer1::DataType::kFLOAT) {
    DataType tf_dtype = DT_FLOAT;
    switch (trt_dtype) {
      case nvinfer1::DataType::kFLOAT:
        tf_dtype = DT_FLOAT;
        break;
      case nvinfer1::DataType::kINT32:
        tf_dtype = DT_INT32;
        break;
      default:
        ASSERT_TRUE(false) << "Unexpected data type "
                           << static_cast<int>(trt_dtype);
    }
    ops::Placeholder::Attrs attrs;
    TF_EXPECT_OK(TensorShapeUtils::MakeShape(dims, &attrs.shape_));
    attrs.shape_.InsertDim(0, batch_size);
    auto input = ops::Placeholder(scope_.WithOpName(name), tf_dtype, attrs);
    validator_inputs_[name] = input.operation.node()->def();

    // Add a real ITensor for conversion conditionally.
    const nvinfer1::Dims trt_dims = GetTestDims(dims);
    if (HasStaticShape(trt_dims)) {
      TF_EXPECT_OK(
          converter_->AddInputTensor(name, trt_dtype, trt_dims, batch_size));
      ASSERT_EQ(batch_size, converter_->batch_size_);
    }
  }

  // Add weights for both validation and conversion.
  template <typename T>
  void AddTestWeights(const char* name, const std::vector<int>& dims,
                      const std::vector<T>& values) {
    const DataType dtype = DataTypeToEnum<T>::v();
    const nvinfer1::Dims trt_dims = GetTestDims(dims);
    const int64_t num_elements = TrtDimsNumElements(trt_dims);
    QCHECK_EQ(num_elements, values.size())
        << num_elements << " vs " << values.size();
    TRT_ShapedWeights weights(dtype);
    if (num_elements) {
      weights = converter_->weight_store_.GetTempWeights(dtype, trt_dims);
      QCHECK_EQ(weights.size_bytes(), sizeof(T) * values.size())
          << weights.size_bytes() << " vs " << sizeof(T) * values.size();
      memcpy(const_cast<void*>(weights.GetValues()), values.data(),
             weights.size_bytes());
    }
    // Add weights for validation.
    TensorShape shape;
    TF_EXPECT_OK(TensorShapeUtils::MakeShape(dims, &shape));
    validator_inputs_[name] = MakeConstNodeDef<T>(name, values, shape);
    // Add weights for conversion.
    TF_EXPECT_OK(
        converter_->AddTensorOrWeights(name, TRT_TensorOrWeights{weights}));
  }

  // Test validation in validation-only mode.
  void RunValidation(const NodeDef& node_def,
                     error::Code expected_code = error::OK,
                     const char* expected_msg_substr = nullptr) {
    std::vector<std::pair<const NodeDef*, int>> input_node_and_ports;
    for (const string& input : node_def.input()) {
      input_node_and_ports.emplace_back(&validator_inputs_[input], 0);
    }
    grappler::GrapplerItem item;
    TF_EXPECT_OK(scope_.ToGraphDef(&item.graph));
    grappler::GraphProperties graph_properties(item);
    TF_EXPECT_OK(graph_properties.InferStatically(true));

    ExpectStatus(validator_->ValidateNode(node_def, input_node_and_ports,
                                          graph_properties),
                 expected_code, expected_msg_substr);
  }

  void RunConversion(const NodeDef& node_def,
                     error::Code expected_code = error::OK,
                     const char* expected_msg_substr = nullptr) {
    ExpectStatus(converter_->ConvertNode(node_def), expected_code,
                 expected_msg_substr);
  }

  // Helper method to run both validation and conversion, when the expected
  // output are same.
  void RunValidationAndConversion(const NodeDef& node_def,
                                  error::Code expected_code = error::OK,
                                  const char* expected_msg_substr = nullptr,
                                  bool should_run_conversion = true) {
    RunValidation(node_def, expected_code, expected_msg_substr);
    if (should_run_conversion) {
      RunConversion(node_def, expected_code, expected_msg_substr);
    }
  }

  std::unique_ptr<Converter> converter_;
  std::unique_ptr<TrtNodeValidator> validator_;

 private:
  Logger logger_;
  TrtUniquePtrType<nvinfer1::IBuilder> builder_;
  TrtUniquePtrType<nvinfer1::INetworkDefinition> network_;
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine_;
  cudaStream_t stream_;
  Scope scope_;
  std::unordered_map<string, NodeDef> validator_inputs_;
};

template <DataType dtype, typename InputCType, typename OutputCType>
void TestConvertConst(OpConverterTest* test) {
  NodeDef node_def;
  node_def.set_name("my_const");
  node_def.set_op("Const");

  auto reset_and_test = [&node_def, test](
                            const Tensor& tensor, const bool as_tensor_content,
                            const std::vector<int>& expected_dims,
                            const std::vector<OutputCType>& expected_value) {
    test->Reset();

    auto& attr = *node_def.mutable_attr();
    if (as_tensor_content) {
      tensor.AsProtoTensorContent(attr["value"].mutable_tensor());
    } else {
      tensor.AsProtoField(attr["value"].mutable_tensor());
    }
    test->RunValidationAndConversion(node_def);
    TRT_TensorOrWeights output;
    TF_EXPECT_OK(test->GetTensorOrWeights("my_const", &output));
    ValidateWeights(output.weights(), expected_dims, expected_value);
  };

  auto& attr = *node_def.mutable_attr();
  attr["dtype"].set_type(dtype);
  {
    // By default empty tensor will pick DT_FLOAT as data type and we fix it
    // here.
    attr["value"].mutable_tensor()->set_dtype(dtype);
    Tensor t;  // Empty tensor.
    reset_and_test(t, false, {}, {});
  }
  {
    Tensor t = ::tensorflow::test::AsScalar<InputCType>(12);
    reset_and_test(t, false, {1}, {12});
    reset_and_test(t, true, {1}, {12});
  }
  {
    Tensor t = ::tensorflow::test::AsTensor<InputCType>({1, 2});
    reset_and_test(t, false, {2}, {1, 2});
    reset_and_test(t, true, {2}, {1, 2});
  }
  {
    Tensor t = ::tensorflow::test::AsTensor<InputCType>({1, 2, 3, 4, 5, 6},
                                                        TensorShape({2, 3}));
    reset_and_test(t, false, {2, 3}, {1, 2, 3, 4, 5, 6});
    reset_and_test(t, true, {2, 3}, {1, 2, 3, 4, 5, 6});
  }
}

TEST_F(OpConverterTest, ConvertConst) {
  {
    Reset();
    NodeDef node_def = MakeNodeDef("my_const", "Const", {"input"});
    AddTestTensor("input", {1});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Constant node is expected to have empty input list: my_const");
  }
  {
    Reset();
    NodeDef node_def = MakeConstNodeDef<double>("my_const", {});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "Unsupported data type double");
  }

  TestConvertConst<DT_FLOAT, float, float>(this);
  TestConvertConst<DT_INT8, int8, int32>(this);
  TestConvertConst<DT_INT32, int32, int32>(this);
}

TEST_F(OpConverterTest, ConvertTranspose) {
  {
    // Input list is empty, should fail.
    NodeDef node_def = MakeNodeDef("my_transpose", "Transpose", {});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Input expects tensor and weights, at my_transpose");
  }

  // Get the NodeDef for Transpose.
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT);
  auto weights = ops::Placeholder(s.WithOpName("weights"), DT_INT32);
  auto transpose = ops::Transpose(s.WithOpName("my_transpose"), input, weights);
  const NodeDef& node_def = transpose.operation.node()->def();

  {
    // Permutation is a tensor, should fail.
    Reset();
    AddTestTensor("input", {1, 2, 3});
    AddTestTensor("weights", {3});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Input expects tensor and weights, at my_transpose");
  }
  {
    // Transpose at batch dimension, should fail.
    Reset();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<int32>("weights", {4}, {1, 0, 2, 3});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "Transpose at batch dimension is not supported");
  }
  {
    // Permutation rank doesn't match, should fail.
    Reset();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<int32>("weights", {3}, {0, 1, 2});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Rank of perm for transpose does not match with that of the input.");
  }
  {
    // Ok.
    Reset();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<int32>("weights", {4}, {0, 3, 1, 2});
    RunConversion(node_def);
    TRT_TensorOrWeights output;
    TF_EXPECT_OK(GetTensorOrWeights("my_transpose", &output));
    EXPECT_TRUE(output.is_tensor());
    EXPECT_TRUE(TrtDimsEqualsArray({3, 1, 2}, output.tensor()->getDimensions()))
        << output.DebugString();

    std::vector<float> output_data(6);
    BuildAndRun("input", {1, 2, 3, 4, 5, 6}, "my_transpose", &output_data);
    EXPECT_THAT(output_data, ElementsAre(1, 4, 2, 5, 3, 6));
  }
}

TEST_F(OpConverterTest, ConvertReshape) {
  {
    // Input list is empty, should fail.
    NodeDef node_def = MakeNodeDef("my_reshape", "Reshape", {});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Input expects weights for shape, at my_reshape");
  }

  // Get the NodeDef for Reshape.
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT);
  auto weights = ops::Placeholder(s.WithOpName("weights"), DT_INT32);
  auto reshape = ops::Reshape(s.WithOpName("my_reshape"), input, weights);
  const NodeDef& node_def = reshape.operation.node()->def();

  {
    // Shape is a tensor, should fail.
    Reset();
    AddTestTensor("input", {1, 2, 3});
    AddTestTensor("weights", {3});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Input expects weights for shape, at my_reshape");
  }
  {
    // Reshape to scalar, should fail.
    Reset();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<int32>("weights", {0}, {});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "Reshape to shape=[] is not supported, at my_reshape");
  }

  struct TestParams {
    TestParams(int input_batch_size, const std::vector<int>& input_tensor_dims,
               const std::vector<int>& input_shape)
        : batch_size(input_batch_size),
          tensor_dims(input_tensor_dims),
          shape(input_shape) {}
    int batch_size;
    std::vector<int> tensor_dims;
    std::vector<int> shape;
  };

  // Reshape at batch dimension, should fail.
  const int kReshapeBatchDimsCases = 5;
  TestParams params[kReshapeBatchDimsCases] = {
      TestParams{1, {1, 2, 3}, {3, 1, 1, 2}},
      TestParams{1, {1, 2, -1}, {-1, 1, 1, 2}},
      TestParams{1, {1, 2, 3}, {-1, 1, 1, 2}},
      TestParams{-1, {1, 2, 3}, {1, 1, 1, 2}},
      TestParams{-1, {-1, 2, 3}, {1, 1, 1, 6}},  // TODO(laigd): it should pass.
  };
  for (int i = 0; i < kReshapeBatchDimsCases; ++i) {
    Reset();
    const std::vector<int>& dims = params[i].tensor_dims;
    AddTestTensor("input", dims, params[i].batch_size);
    AddTestWeights<int32>("weights", {4}, params[i].shape);
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "Reshape on batch dimension is not supported, at my_reshape",
        /*should_run_conversion=*/(dims[0] > 0 && dims[1] > 0 && dims[2] > 0));
  }

  // Reshape on non batch dimensions, ok.
  const int kReshapeOKCases = 3;
  TestParams ok_params[kReshapeOKCases] = {
      TestParams{-1, {1, 2, 3}, {-1, 1, 3, 2}},
      TestParams{1, {1, 2, 3}, {-1, 1, 3, 2}},
      TestParams{1, {1, 2, 3}, {1, 1, 3, 2}},
  };
  for (int i = 0; i < kReshapeOKCases; ++i) {
    Reset();
    AddTestTensor("input", ok_params[i].tensor_dims, ok_params[i].batch_size);
    AddTestWeights<int32>("weights", {4}, ok_params[i].shape);
    RunConversion(node_def);
    TRT_TensorOrWeights output;
    TF_EXPECT_OK(GetTensorOrWeights("my_reshape", &output));
    EXPECT_TRUE(output.is_tensor());
    EXPECT_TRUE(TrtDimsEqualsArray({1, 3, 2}, output.tensor()->getDimensions()))
        << output.DebugString();

    std::vector<float> output_data(6);
    BuildAndRun("input", {1, 2, 3, 4, 5, 6}, "my_reshape", &output_data);
    EXPECT_THAT(output_data, ElementsAre(1, 2, 3, 4, 5, 6));
  }
}

TEST_F(OpConverterTest, ConvertMatMul) {
  {
    // Input list is empty, should fail.
    NodeDef node_def = MakeNodeDef("my_matmul", "MatMul", {});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Input expects tensor and weights, at my_matmul");
  }

  // Get the NodeDef for MatMul.
  auto get_matmul_nodedef = [](DataType dtype, bool transpose_a,
                               bool transpose_b) -> NodeDef {
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), dtype);
    auto weights = ops::Placeholder(s.WithOpName("weights"), dtype);
    const auto matmul_attrs =
        ops::MatMul::TransposeA(transpose_a).TransposeB(transpose_b);
    auto matmul =
        ops::MatMul(s.WithOpName("my_matmul"), input, weights, matmul_attrs);
    return matmul.operation.node()->def();
  };

  {
    // Unsupported data type.
    Reset();
    NodeDef node_def = get_matmul_nodedef(DT_INT32, false, false);
    AddTestTensor("input", {2}, /*batch_size=*/1, nvinfer1::DataType::kINT32);
    AddTestWeights<int32>("weights", {2, 1}, {3, 5});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "Data type is not supported, for node my_matmul got int32");
  }
  // transpose_a is set.
  for (bool transpose_b : {false, true}) {
    Reset();
    NodeDef node_def =
        get_matmul_nodedef(DT_FLOAT, /*transpose_a=*/true, transpose_b);
    AddTestTensor("input", {2}, /*batch_size=*/1);
    AddTestWeights<float>("weights", {2, 2}, {0, 1, 2, 3});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "transpose_a is not supported for TensorRT FullyConnected");
  }
  // OK.
  for (bool transpose_b : {false, true}) {
    Reset();
    NodeDef node_def =
        get_matmul_nodedef(DT_FLOAT, /*transpose_a=*/false, transpose_b);
    AddTestTensor("input", {2}, /*batch_size=*/1);
    AddTestWeights<float>("weights", {2, 2}, {0, 1, 2, 3});
    RunConversion(node_def);
    TRT_TensorOrWeights output;
    TF_EXPECT_OK(GetTensorOrWeights("my_matmul", &output));
    EXPECT_TRUE(output.is_tensor());
    EXPECT_TRUE(TrtDimsEqualsArray({2}, output.tensor()->getDimensions()))
        << output.DebugString();

    std::vector<float> output_data(2);
    BuildAndRun("input", {0, 1}, "my_matmul", &output_data);
    if (transpose_b) {
      EXPECT_THAT(output_data, ElementsAre(1, 3));
    } else {
      EXPECT_THAT(output_data, ElementsAre(2, 3));
    }
  }
}

template <DataType dtype>
void TestConvertBiasAdd(OpConverterTest* test) {
  // Get the NodeDef for BiasAdd.
  auto get_biasadd_nodedef = [](const string& data_format) -> NodeDef {
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), dtype);
    auto weights = ops::Placeholder(s.WithOpName("weights"), dtype);
    const auto biasadd_attrs = ops::BiasAdd::DataFormat(data_format);
    auto biasadd =
        ops::BiasAdd(s.WithOpName("my_biasadd"), input, weights, biasadd_attrs);
    return biasadd.operation.node()->def();
  };

  typedef typename EnumToDataType<dtype>::Type CType;
  for (const string& data_format : {"NHWC", "NCHW"}) {
    for (const int trt_input_rank : {1, 2, 3, 4}) {
      test->Reset();
      NodeDef node_def = get_biasadd_nodedef(data_format);

      // Add input, dims_array will be like {2, 1, ..., 1, 3}
      std::vector<int32> dims_array(trt_input_rank, 1);
      if (trt_input_rank == 1) {
        dims_array[0] = (data_format == "NHWC" ? 3 : 2);
      } else {
        dims_array[0] = 2;
        dims_array[trt_input_rank - 1] = 3;
      }
      test->AddTestTensor("input", dims_array, /*batch_size=*/1);

      // Add bias weights.
      const int channel_size = (data_format == "NHWC" ? 3 : 2);
      std::vector<CType> bias(channel_size);
      std::iota(bias.begin(), bias.end(), 1);  // bias will be {1, 2, 3, ...}
      test->AddTestWeights<CType>("weights", {channel_size}, bias);

      // Run the conversion.
      test->RunValidationAndConversion(node_def);
      TRT_TensorOrWeights output;
      TF_EXPECT_OK(test->GetTensorOrWeights("my_biasadd", &output));
      EXPECT_TRUE(output.is_tensor());
      EXPECT_TRUE(
          TrtDimsEqualsArray(dims_array, output.tensor()->getDimensions()))
          << output.DebugString();

      // Build and run the engine.
      const int num_input = TrtDimsNumElements(GetTestDims(dims_array));
      ASSERT_EQ(trt_input_rank > 1 ? 6 : (data_format == "NHWC" ? 3 : 2),
                num_input);
      std::vector<CType> output_data(num_input);
      test->BuildAndRun("input", std::vector<CType>(num_input, CType(0)),
                        "my_biasadd", &output_data);
      if (trt_input_rank == 1) {
        if (data_format == "NHWC") {
          EXPECT_THAT(output_data, ElementsAre(1, 2, 3));
        } else {
          EXPECT_THAT(output_data, ElementsAre(1, 2));
        }
      } else {
        if (data_format == "NHWC") {
          EXPECT_THAT(output_data, ElementsAre(1, 2, 3, 1, 2, 3));
        } else {
          EXPECT_THAT(output_data, ElementsAre(1, 1, 1, 2, 2, 2));
        }
      }
    }
  }
}

TEST_F(OpConverterTest, ConvertBiasAdd) {
  {
    // Input list is empty, should fail.
    NodeDef node_def = MakeNodeDef("my_biasadd", "BiasAdd", {});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Input expects tensor and weights, at my_biasadd");
  }

  // OK.
  TestConvertBiasAdd<DT_FLOAT>(this);
  // TODO(laigd): uncomment this after cl/220663893 is submitted.
  // TestConvertBiasAdd<DT_INT32>(this);
  // TestConvertBiasAdd<DT_HALF>(this);
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
