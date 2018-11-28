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
#include "tensorflow/core/lib/strings/strcat.h"
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

using ::tensorflow::strings::StrCat;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

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

nvinfer1::DataType TfDataTypeToTrt(DataType tf_dtype) {
  switch (tf_dtype) {
    case DT_FLOAT:
      return nvinfer1::DataType::kFLOAT;
    case DT_HALF:
      return nvinfer1::DataType::kHALF;
    case DT_INT32:
      return nvinfer1::DataType::kINT32;
    default:
      QCHECK(false) << "Unexpected data type " << DataTypeString(tf_dtype);
  }
}

DataType TrtDataTypeToTf(nvinfer1::DataType trt_dtype) {
  switch (trt_dtype) {
    case nvinfer1::DataType::kFLOAT:
      return DT_FLOAT;
    case nvinfer1::DataType::kHALF:
      return DT_HALF;
    case nvinfer1::DataType::kINT32:
      return DT_INT32;
    default:
      QCHECK(false) << "Unexpected data type " << static_cast<int>(trt_dtype);
  }
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

// TODO(laigd): define a parameterized matcher that can compare against the
// vector.
void ExpectTrtDimsEqualsArray(const std::vector<int>& lhs,
                              const nvinfer1::Dims& rhs) {
  EXPECT_TRUE(TrtDimsEqualsArray(lhs, rhs))
      << "expected: " << DebugString(GetTestDims(lhs)) << "\n"
      << "  actual: " << DebugString(rhs);
}

template <typename T>
void ExpectArrayNear(const std::vector<T>& lhs, const std::vector<T>& rhs) {
  ASSERT_EQ(lhs.size(), rhs.size());
  for (int i = 0; i < lhs.size(); i++) {
    EXPECT_FLOAT_EQ(lhs[i], rhs[i]);
  }
}

// Eigen::half cannot implicitly convert to float which is required for
// EXPECT_FLOAT_EQ.
template <>
void ExpectArrayNear(const std::vector<Eigen::half>& lhs,
                     const std::vector<Eigen::half>& rhs) {
  ASSERT_EQ(lhs.size(), rhs.size());
  for (int i = 0; i < lhs.size(); i++) {
    EXPECT_FLOAT_EQ(Eigen::half_impl::half_to_float(lhs[i]),
                    Eigen::half_impl::half_to_float(rhs[i]));
  }
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
  ExpectTrtDimsEqualsArray(expected_dims, weights.shape_);
  ASSERT_EQ(expected_value.size(), weights.count()) << weights.DebugString();
  const T* actual_values = static_cast<const T*>(weights.GetValues());
  for (int i = 0; i < expected_value.size(); ++i) {
    EXPECT_EQ(expected_value[i], actual_values[i]);
  }
}

// Fake ITensor implementation for testing purposes.
class FakeITensor : public nvinfer1::ITensor {
 public:
  FakeITensor() : dynamic_range_(0.0f) {}

  FakeITensor(const nvinfer1::Dims& dims) : dims_(dims), dynamic_range_(0.0f) {}

  FakeITensor(const std::vector<int>& dims)
      : dims_(GetTestDims(dims)), dynamic_range_(0.0f) {}

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
  bool setDynamicRange(float min, float max) override {
    dynamic_range_ = std::max(std::abs(min), std::abs(max));
    return true;
  }

  float getDynamicRange() const override { return dynamic_range_; }
#endif

 private:
  string name_;
  nvinfer1::Dims dims_;
  nvinfer1::DataType type_;
  nvinfer1::TensorLocation location_;
  float dynamic_range_;
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
        ExpectTrtDimsEqualsArray({1}, ptr->GetTrtDims());
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
      ExpectTrtDimsEqualsArray({1}, ptr->GetTrtDims());
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
      ExpectTrtDimsEqualsArray({}, ptr->GetTrtDims());
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
    ExpectTrtDimsEqualsArray({non_batch_dim}, output.GetTrtDims());
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
    converter_.reset(new Converter(network_.get(),
                                   /*precision_mode=*/FP32MODE,
                                   /*use_calibration=*/false));
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

  Status GetWeightRange(const TRT_ShapedWeights& weights, float* out_min,
                        float* out_max) const {
    return converter_->GetWeightRange(weights, out_min, out_max);
  }

  void PropagateQuantizationRanges() {
    converter_->PropagateQuantizationRanges();
  }

  int batch_size() const { return converter_->batch_size_; }

  std::unordered_map<nvinfer1::ITensor*, float>& quantization_ranges() {
    return converter_->quantization_ranges_;
  }

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
  ExpectTrtDimsEqualsArray({1}, inputs[0].tensor()->getDimensions());
  ExpectTrtDimsEqualsArray({2, 3}, inputs[2].tensor()->getDimensions());
  ExpectTrtDimsEqualsArray({5, 3}, inputs[3].tensor()->getDimensions());
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
    ExpectTrtDimsEqualsArray({2, 1}, output_tensor->getDimensions());
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
  ExpectTrtDimsEqualsArray({5, 2, 3}, output_tensor->getDimensions());
}

TEST_F(ConverterTest, PrepareTensorForShape_Tensor) {
  nvinfer1::ITensor* input_tensor = converter_->network()->addInput(
      "", nvinfer1::DataType::kFLOAT, GetTestDims({2, 3, 5}));
  TRT_TensorOrWeights tw(input_tensor);
  const nvinfer1::ITensor* output_tensor = nullptr;

  // Shape size doesn't match.
  ExpectStatus(converter_->PrepareTensorForShape(tw, GetTestDims({2, 3, 6}),
                                                 &output_tensor),
               error::INVALID_ARGUMENT, "Reshape shapes are not compatible");

  // TODO(aaroey): we should check the case where uninferred dimensions are not
  // an exact divisor of input dim ensions, e.g. for dims {-1, 7}.

  // Infer shape, ok.
  TF_EXPECT_OK(converter_->PrepareTensorForShape(tw, GetTestDims({-1, 2}),
                                                 &output_tensor));
  ExpectTrtDimsEqualsArray({15, 2}, output_tensor->getDimensions());

  // Regular shape.
  TF_EXPECT_OK(converter_->PrepareTensorForShape(tw, GetTestDims({10, 3}),
                                                 &output_tensor));
  ExpectTrtDimsEqualsArray({10, 3}, output_tensor->getDimensions());
}

TEST_F(ConverterTest, PrepareTensorForShape_Weights) {
  TRT_ShapedWeights weights =
      weight_store_->GetTempWeights(DT_FLOAT, GetTestDims({2, 3, 5}));
  TRT_TensorOrWeights tw(weights);
  const nvinfer1::ITensor* output_tensor = nullptr;
  TF_EXPECT_OK(converter_->PrepareTensorForShape(tw, GetTestDims({10, 3}),
                                                 &output_tensor));
  ExpectTrtDimsEqualsArray({10, 3}, output_tensor->getDimensions());
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

template <typename T>
void TestGetWeightRange(ConverterTest* test, TrtWeightStore* weight_store) {
  TRT_ShapedWeights weights =
      weight_store->GetTempWeights(DataTypeToEnum<T>::v(), GetTestDims({2, 3}));
  const std::vector<T> values = {T(3), T(1), T(2), T(6), T(5), T(4)};
  memcpy(const_cast<void*>(weights.GetValues()), values.data(),
         weights.size_bytes());

  float out_min = 0.0f;
  float out_max = 0.0f;
  TF_EXPECT_OK(test->GetWeightRange(weights, &out_min, &out_max));
  EXPECT_EQ(1.0f, out_min);
  EXPECT_EQ(6.0f, out_max);
}

TEST_F(ConverterTest, GetWeightRange) {
  TestGetWeightRange<float>(this, weight_store_);
  TestGetWeightRange<Eigen::half>(this, weight_store_);
  TestGetWeightRange<int32>(this, weight_store_);
}

TEST_F(ConverterTest, ProvideQuantizationRange) {
  FakeITensor fake_tensor;
  // Assymetric range
  converter_->ProvideQuantizationRange(&fake_tensor, 0.0f, 6.0f);
  EXPECT_EQ(6.0f, quantization_ranges()[&fake_tensor]);
  converter_->ProvideQuantizationRange(&fake_tensor, 1.0f, 6.0f);
  EXPECT_EQ(6.0f, quantization_ranges()[&fake_tensor]);
  converter_->ProvideQuantizationRange(&fake_tensor, -8.0f, 6.0f);
  EXPECT_EQ(8.0f, quantization_ranges()[&fake_tensor]);
  converter_->ProvideQuantizationRange(&fake_tensor, -8.123f, -6.123f);
  EXPECT_EQ(8.123f, quantization_ranges()[&fake_tensor]);
  // Symmetric range
  converter_->ProvideQuantizationRange(&fake_tensor, -6.123f, 6.123f);
  EXPECT_EQ(6.123f, quantization_ranges()[&fake_tensor]);
}

TEST_F(ConverterTest, MaybeApplyQuantizationRanges) {
  // input -> infer1 -> infer2 -> infer3
  FakeITensor input, infer_1, infer_2, infer_3;
  FakeITensor not_infer;
  Converter int8_converter(/*trt_network=*/nullptr, INT8MODE,
                           /*use_calibration=*/true);
  int8_converter.ProvideQuantizationRange(&input, -5.0f, 5.0f);
  int8_converter.ProvideQuantizationRange(&not_infer, -100.0f, 100.0f);
  int8_converter.MarkQuantizationRangesAsInferrable(&input, &infer_1);
  int8_converter.MarkQuantizationRangesAsInferrable(&infer_1, &infer_2);
  int8_converter.MarkQuantizationRangesAsInferrable(&infer_2, &infer_3);

  // Input range should be inferred along the chain and applied to tensors.
  int8_converter.MaybeApplyQuantizationRanges();
#if NV_TENSORRT_MAJOR >= 5
  EXPECT_EQ(input.getDynamicRange(), 5.0f);
  EXPECT_EQ(infer_1.getDynamicRange(), 5.0f);
  EXPECT_EQ(infer_2.getDynamicRange(), 5.0f);
  EXPECT_EQ(infer_3.getDynamicRange(), 5.0f);
  EXPECT_EQ(not_infer.getDynamicRange(), 100.0f);
#endif
}

TEST_F(ConverterTest, PropagateQuantizationRanges) {
  // infer0 <-> infer1 <-> infer2 <-> infer3
  //              |
  //            infer4 <-> infer5
  FakeITensor infer[6];
  FakeITensor not_infer;
  converter_->ProvideQuantizationRange(&infer[4], -5.0f, 5.0f);
  converter_->MarkQuantizationRangesAsInferrable(&infer[0], &infer[1]);
  converter_->MarkQuantizationRangesAsInferrable(&infer[1], &infer[2]);
  converter_->MarkQuantizationRangesAsInferrable(&infer[3], &infer[2]);
  converter_->MarkQuantizationRangesAsInferrable(&infer[4], &infer[1]);
  converter_->MarkQuantizationRangesAsInferrable(&infer[4], &infer[5]);

  // Input range should be inferred along the chain.
  PropagateQuantizationRanges();
  auto ranges = quantization_ranges();
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(5.0f, ranges[&infer[i]]);
  }
  EXPECT_EQ(ranges.count(&not_infer), 0);
}

TEST_F(ConverterTest, GetTrtBroadcastShape) {
  const bool kIsTensor = true;
  const bool kIsNotTensor = false;
  auto symmetric_test = [this](const std::vector<int>& operand_1_shape,
                               const std::vector<int>& operand_2_shape,
                               const bool operand_1_is_tensor,
                               const bool operand_2_is_tensor,
                               const std::vector<int>& expected_operand_1_shape,
                               const std::vector<int>& expected_operand_2_shape,
                               error::Code expected_code = error::OK,
                               const char* expected_error_msg_substr = nullptr,
                               const int operand_1_batch_size = -1,
                               const int operand_2_batch_size = -1) {
    auto create_tensor_or_weights = [](const std::vector<int>& shape,
                                       bool is_tensor, int batch_size = -1) {
      if (is_tensor) {
        return TRT_TensorOrWeights{nvinfer1::DataType::kFLOAT,
                                   GetTestDims(shape), batch_size};
      }
      TRT_ShapedWeights weights;
      weights.shape_ = GetTestDims(shape);
      return TRT_TensorOrWeights(weights);
    };

    nvinfer1::Dims operand_1_new_dims, operand_2_new_dims;
    TRT_TensorOrWeights operand_1 = create_tensor_or_weights(
        operand_1_shape, operand_1_is_tensor, operand_1_batch_size);
    TRT_TensorOrWeights operand_2 = create_tensor_or_weights(
        operand_2_shape, operand_2_is_tensor, operand_2_batch_size);

    // operand_1 broadcast operand_2
    ExpectStatus(
        this->converter_->GetTrtBroadcastShape(
            operand_1, operand_2, &operand_1_new_dims, &operand_2_new_dims),
        expected_code, expected_error_msg_substr);
    if (expected_code == error::OK) {
      ExpectTrtDimsEqualsArray(expected_operand_1_shape, operand_1_new_dims);
      ExpectTrtDimsEqualsArray(expected_operand_2_shape, operand_2_new_dims);
    }
    // operand_2 broadcast operand_1
    ExpectStatus(
        this->converter_->GetTrtBroadcastShape(
            operand_2, operand_1, &operand_2_new_dims, &operand_1_new_dims),
        expected_code, expected_error_msg_substr);
    if (expected_code == error::OK) {
      ExpectTrtDimsEqualsArray(expected_operand_1_shape, operand_1_new_dims);
      ExpectTrtDimsEqualsArray(expected_operand_2_shape, operand_2_new_dims);
    }
  };

  // Both inputs are weights.
  symmetric_test(
      {1}, {1}, kIsNotTensor, kIsNotTensor, {}, {}, error::INVALID_ARGUMENT,
      "Broadcasting requires at least one of the operands be tensors");

  // One tensor and one weights.
  symmetric_test({1, 1, 1}, {2}, kIsTensor, kIsNotTensor, {1, 1, 1}, {1, 1, 2});
  symmetric_test({1, 1, 2}, {2}, kIsTensor, kIsNotTensor, {1, 1, 2}, {1, 1, 2});
  symmetric_test({1, 3, 2}, {1}, kIsTensor, kIsNotTensor, {1, 3, 2}, {1, 1, 1});
  symmetric_test({1, 1, 1}, {2, 3}, kIsTensor, kIsNotTensor, {1, 1, 1},
                 {1, 2, 3});
  symmetric_test({1, 1, 1}, {2, 3, 4}, kIsTensor, kIsNotTensor, {1, 1, 1},
                 {2, 3, 4});
  symmetric_test({1, 1, 1}, {1, 2, 3, 4}, kIsTensor, kIsNotTensor, {1, 1, 1},
                 {2, 3, 4});
  symmetric_test({1, 3, 4}, {1, 2, 1, 4}, kIsTensor, kIsNotTensor, {1, 3, 4},
                 {2, 1, 4});
  symmetric_test({1, 1, 1}, {2, 1, 1, 1}, kIsTensor, kIsNotTensor, {}, {},
                 error::INVALID_ARGUMENT, "Infeasible broadcast scheme");
  symmetric_test({1, 1, 1}, {2, 1, 1, 1}, kIsTensor, kIsNotTensor, {}, {},
                 error::INVALID_ARGUMENT, "Infeasible broadcast scheme",
                 /*operand_1_batch_size=*/2);
  symmetric_test({1, 1, 1}, {1, 1, 1, 1, 1}, kIsTensor, kIsNotTensor, {}, {},
                 error::INVALID_ARGUMENT,
                 "Broadcasting beyond batch dimension is not supported "
                 "(tensor #dims 4 vs broadcast #dims 5)");

  // Both inputs are tensors.
  symmetric_test({1, 1, 1}, {1, 1}, kIsTensor, kIsTensor, {}, {},
                 error::INVALID_ARGUMENT,
                 "Broadcasting beyond batch dimension is not supported "
                 "(tensor #dims 3 vs broadcast #dims 4)");
  symmetric_test({1, 3, 4}, {2, 1, 4}, kIsTensor, kIsTensor, {1, 3, 4},
                 {2, 1, 4});
  symmetric_test({1, 1, 1}, {1, 1, 1, 1}, kIsTensor, kIsTensor, {}, {},
                 error::INVALID_ARGUMENT,
                 "Broadcasting beyond batch dimension is not supported "
                 "(tensor #dims 4 vs broadcast #dims 5)");
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
    converter_.reset(new Converter(network_.get(),
                                   /*precision_mode=*/FP32MODE,
                                   /*use_calibration=*/false));

    // Reset other related artifacts.
    scope_ = Scope::NewRootScope();
    validator_inputs_.clear();
  }

  // TODO(laigd): test fp16 and int8 support.
  template <typename T>
  void BuildAndRun(
      const std::vector<std::pair<const char*, const std::vector<T>>>&
          input_data,
      const char* output_name, std::vector<T>* output_data) {
    // Mark the output tensor as TRT engine output.
    TF_EXPECT_OK(converter_->RenameAndMarkOutputTensors(
        {{string(output_name), string(output_name)}}));

    // Build the TRT engine.
    ASSERT_EQ(nullptr, engine_.get());
    engine_.reset(builder_->buildCudaEngine(*converter_->network()));
    CHECK_NOTNULL(engine_.get());

    // Execute the TRT engine.
    ASSERT_LE(input_data.size() + 1, 3);
    void* buffers[3];
    for (const auto name_and_data : input_data) {
      const int input_size = name_and_data.second.size() * sizeof(T);
      const int input_index = engine_->getBindingIndex(name_and_data.first);
      ASSERT_EQ(0, cudaMalloc(&buffers[input_index], input_size));
      ASSERT_EQ(
          0, cudaMemcpyAsync(buffers[input_index], name_and_data.second.data(),
                             input_size, cudaMemcpyHostToDevice, stream_));
    }

    const int output_size = output_data->size() * sizeof(T);
    const int output_index = engine_->getBindingIndex(output_name);
    ASSERT_EQ(0, cudaMalloc(&buffers[output_index], output_size));

    ASSERT_EQ(engine_->getNbBindings(), input_data.size() + 1);

    TrtUniquePtrType<nvinfer1::IExecutionContext> execution_context(
        engine_->createExecutionContext());
    execution_context->enqueue(/*batchSize=*/1, buffers, stream_, nullptr);
    ASSERT_EQ(0, cudaMemcpyAsync(output_data->data(), buffers[output_index],
                                 output_size, cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_);

    for (int i = 0; i < input_data.size() + 1; ++i) {
      ASSERT_EQ(0, cudaFree(buffers[i]));
    }
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
    DataType tf_dtype = TrtDataTypeToTf(trt_dtype);
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

  // Expose quantization_ranges_ for tests
  std::unordered_map<nvinfer1::ITensor*, float>& quantization_ranges() {
    return converter_->quantization_ranges_;
  }

  std::unique_ptr<Converter> converter_;
  std::unique_ptr<TrtNodeValidator> validator_;

 private:
  Logger logger_;
  TrtUniquePtrType<nvinfer1::IBuilder> builder_;
  TrtUniquePtrType<nvinfer1::INetworkDefinition> network_;
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine_;
  cudaStream_t stream_;
  // Used to create placeholders with shape and data type information. The
  // created placeholders will be used as inputs to the node to be verified,
  // thus we need the shape and data type information to get a non-empty
  // GraphProperties.
  // TODO(laigd): consider use this Scope to create the NodeDef to verify.
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
    RunValidationAndConversion(node_def);
    TRT_TensorOrWeights output;
    TF_EXPECT_OK(GetTensorOrWeights("my_transpose", &output));
    EXPECT_TRUE(output.is_tensor());
    ExpectTrtDimsEqualsArray({3, 1, 2}, output.tensor()->getDimensions());

    std::vector<float> output_data(6);
    BuildAndRun<float>({{"input", {1, 2, 3, 4, 5, 6}}}, "my_transpose",
                       &output_data);
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
    RunValidationAndConversion(node_def);
    TRT_TensorOrWeights output;
    TF_EXPECT_OK(GetTensorOrWeights("my_reshape", &output));
    EXPECT_TRUE(output.is_tensor());
    ExpectTrtDimsEqualsArray({1, 3, 2}, output.tensor()->getDimensions());

    std::vector<float> output_data(6);
    BuildAndRun<float>({{"input", {1, 2, 3, 4, 5, 6}}}, "my_reshape",
                       &output_data);
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
    RunValidationAndConversion(node_def);
    TRT_TensorOrWeights output;
    TF_EXPECT_OK(GetTensorOrWeights("my_matmul", &output));
    EXPECT_TRUE(output.is_tensor());
    ExpectTrtDimsEqualsArray({2}, output.tensor()->getDimensions());

    std::vector<float> output_data(2);
    BuildAndRun<float>({{"input", {0, 1}}}, "my_matmul", &output_data);
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
      test->AddTestTensor("input", dims_array, /*batch_size=*/1,
                          TfDataTypeToTrt(dtype));

      // Add bias weights.
      const int channel_size = (data_format == "NHWC" ? 3 : 2);
      std::vector<CType> bias(channel_size);
      for (int i = 0; i < channel_size; ++i) {
        bias[i] = CType(i + 1);  // bias will be {1, 2, 3, ...}
      }
      test->AddTestWeights<CType>("weights", {channel_size}, bias);

      // Run the conversion.
      test->RunValidationAndConversion(node_def);
      TRT_TensorOrWeights output;
      TF_EXPECT_OK(test->GetTensorOrWeights("my_biasadd", &output));
      EXPECT_TRUE(output.is_tensor());
      ExpectTrtDimsEqualsArray(dims_array, output.tensor()->getDimensions());

      // Build and run the engine.
      const int num_input = TrtDimsNumElements(GetTestDims(dims_array));
      ASSERT_EQ(trt_input_rank > 1 ? 6 : (data_format == "NHWC" ? 3 : 2),
                num_input);
      std::vector<CType> output_data(num_input);
      test->BuildAndRun<CType>(
          {{"input", std::vector<CType>(num_input, CType(0))}}, "my_biasadd",
          &output_data);
      if (trt_input_rank == 1) {
        if (data_format == "NHWC") {
          EXPECT_THAT(output_data, ElementsAre(CType(1), CType(2), CType(3)));
        } else {
          EXPECT_THAT(output_data, ElementsAre(CType(1), CType(2)));
        }
      } else {
        if (data_format == "NHWC") {
          EXPECT_THAT(output_data, ElementsAre(CType(1), CType(2), CType(3),
                                               CType(1), CType(2), CType(3)));
        } else {
          EXPECT_THAT(output_data, ElementsAre(CType(1), CType(1), CType(1),
                                               CType(2), CType(2), CType(2)));
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

  // OK. Note that kINT32 is not supported by IScaleLayer, so we don't test
  // DT_INT32 type here.
  TestConvertBiasAdd<DT_FLOAT>(this);
  TestConvertBiasAdd<DT_HALF>(this);
}

template <typename OpType>
NodeDef GetBinaryOpNodeDef(const string& input_name_l,
                           const string& input_name_r, DataType dtype) {
  Scope s = Scope::NewRootScope();
  auto input_l = ops::Placeholder(s.WithOpName(input_name_l), dtype);
  auto input_r = ops::Placeholder(s.WithOpName(input_name_r), dtype);
  auto op = OpType(s.WithOpName("my_binary"), input_l, input_r);
  return op.operation.node()->def();
}

void CheckAddedLayers(OpConverterTest* test, bool expect_scale_layer) {
  bool element_wise_layer_found = false;
  bool scale_layer_found = false;
  for (int i = 0; i < test->converter_->network()->getNbLayers(); i++) {
    nvinfer1::ILayer* layer = test->converter_->network()->getLayer(i);
    if (dynamic_cast<nvinfer1::IScaleLayer*>(layer)) {
      scale_layer_found = true;
    } else if (dynamic_cast<nvinfer1::IElementWiseLayer*>(layer)) {
      element_wise_layer_found = true;
    }
  }
  EXPECT_EQ(expect_scale_layer, scale_layer_found);
  EXPECT_NE(expect_scale_layer, element_wise_layer_found);
}

template <typename OpType, DataType dtype>
void TestBinaryTensorOpWeightNoBroadcast(OpConverterTest* test) {
  typedef typename EnumToDataType<dtype>::Type CType;
  for (auto swap_inputs : {false, true}) {
    test->Reset();
    NodeDef node_def;
    if (swap_inputs) {
      node_def = GetBinaryOpNodeDef<OpType>("weights", "input", dtype);
    } else {
      node_def = GetBinaryOpNodeDef<OpType>("input", "weights", dtype);
    }

    const std::vector<CType> operand1{CType(3), CType(7.5)};
    const std::vector<CType> operand2{CType(2), CType(3)};

    // It requires the dims to be at least of rank 3 to apply an IScaleLayer.
    test->AddTestTensor("input", /*dims=*/{1, 1, 2}, /*batch_size=*/1,
                        TfDataTypeToTrt(dtype));
    test->AddTestWeights<CType>("weights", /*dims=*/{1, 1, 2},
                                /*values=*/swap_inputs ? operand1 : operand2);
    test->RunValidationAndConversion(node_def);

    // Make sure it does use BinaryTensorOpWeight, not BinaryTensorOpTensor.
    CheckAddedLayers(test, /*expect_scale_layer=*/true);

    // Check the dims of the output ITensor.
    TRT_TensorOrWeights output;
    TF_EXPECT_OK(test->GetTensorOrWeights("my_binary", &output));
    EXPECT_TRUE(output.is_tensor());
    ExpectTrtDimsEqualsArray({1, 1, 2}, output.tensor()->getDimensions());

    std::vector<CType> output_data(2);
    test->BuildAndRun<CType>(
        {{"input",
          /*input_data=*/swap_inputs ? operand2 : operand1}},
        "my_binary", &output_data);
    if (node_def.op() == "Add") {
      EXPECT_THAT(output_data, ElementsAre(CType(5), CType(10.5)));
    } else if (node_def.op() == "Sub") {
      EXPECT_THAT(output_data, ElementsAre(CType(1), CType(4.5)));
    } else if (node_def.op() == "Mul") {
      EXPECT_THAT(output_data, ElementsAre(CType(6), CType(22.5)));
    } else if (node_def.op() == "Div") {
      EXPECT_THAT(output_data, ElementsAre(CType(1.5), CType(2.5)));
    } else if (node_def.op() == "RealDiv") {
      EXPECT_THAT(output_data, ElementsAre(CType(1.5), CType(2.5)));
    } else {
      ASSERT_TRUE(false);
    }
  }
}

template <DataType dtype>
void TestBinaryTensorOpWeightWithChannelWiseBroadcast(OpConverterTest* test) {
  typedef typename EnumToDataType<dtype>::Type CType;
  const NodeDef node_def =
      GetBinaryOpNodeDef<ops::Add>("input", "weights", dtype);
  const std::vector<CType> input{CType(1), CType(2), CType(3), CType(4)};
  const std::vector<CType> weights{CType(10), CType(20)};
  // There are two types of valid dim pairs which requires channel-wise
  // broadcasting:
  // - input dims (X Y Z) vs weights dims (X 1 1)
  // - input dims (X Y Z) vs weights dims (Z)
  // Here X=Z=2 and Y=1.
  for (auto weights_dims : std::vector<std::vector<int>>{{2, 1, 1}, {2}}) {
    test->Reset();
    test->AddTestTensor("input", /*dims=*/{2, 1, 2}, /*batch_size=*/1,
                        TfDataTypeToTrt(dtype));
    test->AddTestWeights<CType>("weights", weights_dims, weights);
    test->RunValidationAndConversion(node_def);

    // Make sure it does use BinaryTensorOpWeight, not BinaryTensorOpTensor.
    CheckAddedLayers(test, /*expect_scale_layer=*/true);

    // Check the dims of the output ITensor.
    TRT_TensorOrWeights output;
    TF_EXPECT_OK(test->GetTensorOrWeights("my_binary", &output));
    EXPECT_TRUE(output.is_tensor());
    ExpectTrtDimsEqualsArray({2, 1, 2}, output.tensor()->getDimensions());

    std::vector<CType> output_data(4);
    test->BuildAndRun<CType>({{"input", input}}, "my_binary", &output_data);
    if (weights_dims.size() == 1) {
      EXPECT_THAT(output_data,
                  ElementsAre(CType(11), CType(22), CType(13), CType(24)));
    } else {
      EXPECT_THAT(output_data,
                  ElementsAre(CType(11), CType(12), CType(23), CType(24)));
    }
  }
}

template <DataType dtype>
void TestBinaryTensorOpWeightWithUniformlyBroadcast(OpConverterTest* test) {
  typedef typename EnumToDataType<dtype>::Type CType;
  const NodeDef node_def =
      GetBinaryOpNodeDef<ops::Add>("input", "weights", dtype);
  const std::vector<CType> input{CType(1), CType(2), CType(3), CType(4)};
  const std::vector<CType> weights{CType(10)};
  test->Reset();
  test->AddTestTensor("input", /*dims=*/{2, 1, 2}, /*batch_size=*/1,
                      TfDataTypeToTrt(dtype));
  test->AddTestWeights<CType>("weights", {1, 1, 1, 1}, weights);
  test->RunValidationAndConversion(node_def);

  // Make sure it does use BinaryTensorOpWeight, not BinaryTensorOpTensor.
  CheckAddedLayers(test, /*expect_scale_layer=*/true);

  // Check the dims of the output ITensor.
  TRT_TensorOrWeights output;
  TF_EXPECT_OK(test->GetTensorOrWeights("my_binary", &output));
  EXPECT_TRUE(output.is_tensor());
  ExpectTrtDimsEqualsArray({2, 1, 2}, output.tensor()->getDimensions());

  std::vector<CType> output_data(4);
  test->BuildAndRun<CType>({{"input", input}}, "my_binary", &output_data);
  EXPECT_THAT(output_data,
              ElementsAre(CType(11), CType(12), CType(13), CType(14)));
}

template <typename OpType>
void TestBinaryTensorOpWeightFallback(OpConverterTest* test,
                                      const std::vector<int32>& input_dims,
                                      const std::vector<int>& weights_dims,
                                      error::Code code = error::OK,
                                      const char* error_msg_substr = nullptr,
                                      const int input_batch_size = 1) {
  const DataType dtype = DT_FLOAT;
  typedef typename EnumToDataType<dtype>::Type CType;
  const size_t num_inputs = TrtDimsNumElements(GetTestDims(input_dims));
  const size_t num_weights = TrtDimsNumElements(GetTestDims(weights_dims));

  test->Reset();
  const NodeDef node_def =
      GetBinaryOpNodeDef<OpType>("input", "weights", dtype);
  test->AddTestTensor("input", /*dims=*/input_dims, input_batch_size,
                      TfDataTypeToTrt(dtype));
  test->AddTestWeights<CType>(
      "weights", /*dims=*/weights_dims,
      /*values=*/std::vector<CType>(num_weights, CType(1)));
  test->RunValidationAndConversion(node_def, code, error_msg_substr);
  if (code != error::OK) return;

  // Make sure it does use BinaryTensorOpTensor, not BinaryTensorOpWeight.
  CheckAddedLayers(test, /*expect_scale_layer=*/false);

  TRT_TensorOrWeights output;
  TF_EXPECT_OK(test->GetTensorOrWeights("my_binary", &output));
  EXPECT_TRUE(output.is_tensor());

  // Check the dims of the output ITensor.
  std::vector<int> expected_output_dims = input_dims;
  for (int i = expected_output_dims.size() - 1, j = weights_dims.size() - 1;
       i >= 0 && j >= 0; --i, --j) {
    if (expected_output_dims[i] == 1) {
      expected_output_dims[i] = weights_dims[j];
    }
  }
  ExpectTrtDimsEqualsArray(expected_output_dims,
                           output.tensor()->getDimensions());

  // Check the result of running the engine.
  const int expected_num_outputs =
      TrtDimsNumElements(GetTestDims(expected_output_dims));
  std::vector<CType> output_data(expected_num_outputs);
  test->BuildAndRun<CType>(
      {{"input",
        /*input_data=*/std::vector<CType>(num_inputs, CType(2))}},
      "my_binary", &output_data);
  if (node_def.op() == "Add") {
    EXPECT_THAT(output_data, ElementsAreArray(std::vector<CType>(
                                 expected_num_outputs, CType(3))));
  } else if (node_def.op() == "Minimum") {
    EXPECT_THAT(output_data, ElementsAreArray(std::vector<CType>(
                                 expected_num_outputs, CType(1))));
  } else {
    ASSERT_TRUE(false);
  }
}

template <typename OpType, DataType dtype>
void TestBinaryTensorOpTensor(OpConverterTest* test) {
  typedef typename EnumToDataType<dtype>::Type CType;
  test->Reset();
  const NodeDef node_def =
      GetBinaryOpNodeDef<OpType>("input1", "input2", dtype);
  test->AddTestTensor("input1", /*dims=*/{1, 2}, /*batch_size=*/1,
                      TfDataTypeToTrt(dtype));
  test->AddTestTensor("input2", /*dims=*/{2, 1}, /*batch_size=*/1,
                      TfDataTypeToTrt(dtype));
  test->RunValidationAndConversion(node_def);

  // Make sure it does use BinaryTensorOpTensor, not BinaryTensorOpWeight.
  CheckAddedLayers(test, /*expect_scale_layer=*/false);

  // Check output dims.
  TRT_TensorOrWeights output;
  TF_EXPECT_OK(test->GetTensorOrWeights("my_binary", &output));
  EXPECT_TRUE(output.is_tensor());
  ExpectTrtDimsEqualsArray({2, 2}, output.tensor()->getDimensions());

  std::vector<CType> output_data(4);
  // After broadcasting first input becomes {3, 6, 3, 6} and second input
  // becomes {2, 3, 2, 3}.
  test->BuildAndRun<CType>(
      {{"input1", {CType(3), CType(6)}}, {"input2", {CType(2), CType(3)}}},
      "my_binary", &output_data);
  if (node_def.op() == "Add") {
    EXPECT_THAT(output_data,
                ElementsAre(CType(5), CType(8), CType(6), CType(9)));
  } else if (node_def.op() == "Sub") {
    EXPECT_THAT(output_data,
                ElementsAre(CType(1), CType(4), CType(0), CType(3)));
  } else if (node_def.op() == "Mul") {
    EXPECT_THAT(output_data,
                ElementsAre(CType(6), CType(12), CType(9), CType(18)));
  } else if (node_def.op() == "Div") {
    EXPECT_THAT(output_data,
                ElementsAre(CType(1.5), CType(3), CType(1), CType(2)));
  } else if (node_def.op() == "RealDiv") {
    EXPECT_THAT(output_data,
                ElementsAre(CType(1.5), CType(3), CType(1), CType(2)));
  } else if (node_def.op() == "Minimum") {
    EXPECT_THAT(output_data,
                ElementsAre(CType(2), CType(2), CType(3), CType(3)));
  } else if (node_def.op() == "Maximum") {
    EXPECT_THAT(output_data,
                ElementsAre(CType(3), CType(6), CType(3), CType(6)));
  } else {
    ASSERT_TRUE(false);
  }
}

TEST_F(OpConverterTest, ConvertBinary) {
  // Input size doesn't match, should fail.
  for (size_t num_inputs = 0; num_inputs < 2; ++num_inputs) {
    Reset();
    NodeDef node_def = MakeNodeDef("my_add", "Add", {num_inputs, "input"});
    AddTestTensor("input", {1}, /*batch_size=*/1, nvinfer1::DataType::kFLOAT);
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "Binary ops require two inputs, at my_add");
  }
  {
    // Both inputs are weights.
    Reset();
    NodeDef node_def = MakeNodeDef("my_add", "Add", {"weights1", "weights2"});
    AddTestWeights<float>("weights1", {1}, {1});
    AddTestWeights<float>("weights2", {1}, {1});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "Constant folding is falled back to TensorFlow, binary op received "
        "both input as constant at: my_add");
  }

  // Test BinaryTensorOpWeight() without broadcasting.
  TestBinaryTensorOpWeightNoBroadcast<ops::Add, DT_FLOAT>(this);
  TestBinaryTensorOpWeightNoBroadcast<ops::Sub, DT_FLOAT>(this);
  TestBinaryTensorOpWeightNoBroadcast<ops::Mul, DT_FLOAT>(this);
  TestBinaryTensorOpWeightNoBroadcast<ops::Div, DT_FLOAT>(this);
  TestBinaryTensorOpWeightNoBroadcast<ops::RealDiv, DT_FLOAT>(this);
#if 0
  // TODO(b/119560144): it doesn't support FP16 constants and the following test
  // will fail.
  TestBinaryTensorOpWeightNoBroadcast<ops::Add, DT_HALF>(this);
  TestBinaryTensorOpWeightNoBroadcast<ops::Sub, DT_HALF>(this);
  TestBinaryTensorOpWeightNoBroadcast<ops::Mul, DT_HALF>(this);
  TestBinaryTensorOpWeightNoBroadcast<ops::Div, DT_HALF>(this);
  TestBinaryTensorOpWeightNoBroadcast<ops::RealDiv, DT_HALF>(this);
#endif

  // Test BinaryTensorOpWeight() with channel-wise broadcasting.
  TestBinaryTensorOpWeightWithChannelWiseBroadcast<DT_FLOAT>(this);

  // Test BinaryTensorOpWeight() with uniformly broadcasting.
  TestBinaryTensorOpWeightWithUniformlyBroadcast<DT_FLOAT>(this);

  // Test BinaryTensorOpWeight() falling back to BinaryTensorOpTensor().
  // Unsupported op.
  TestBinaryTensorOpWeightFallback<ops::Minimum>(this, {1, 1, 1}, {1});
  // Rank of input tensor dimension <3.
  TestBinaryTensorOpWeightFallback<ops::Add>(this, {1, 1}, {1});
  // Broadcast on batch dimension, should fail.
  TestBinaryTensorOpWeightFallback<ops::Add>(
      this, {1, 1, 1}, {2, 1, 1, 1}, error::INVALID_ARGUMENT,
      "Unsupported binary op broadcast scheme for op my_binary",
      /*input_batch_size=*/2);
  // Incompatible dims with per-channel mode.
  TestBinaryTensorOpWeightFallback<ops::Add>(this, {1, 1, 1}, {1, 2, 1});
  // Incompatible dims.
  TestBinaryTensorOpWeightFallback<ops::Add>(this, {1, 2, 1}, {2});

  // Test BinaryTensorOpTensor() with broadcasting.
  TestBinaryTensorOpTensor<ops::Add, DT_FLOAT>(this);
  TestBinaryTensorOpTensor<ops::Sub, DT_FLOAT>(this);
  TestBinaryTensorOpTensor<ops::Mul, DT_FLOAT>(this);
  TestBinaryTensorOpTensor<ops::Div, DT_FLOAT>(this);
  TestBinaryTensorOpTensor<ops::RealDiv, DT_FLOAT>(this);
  TestBinaryTensorOpTensor<ops::Minimum, DT_FLOAT>(this);
  TestBinaryTensorOpTensor<ops::Maximum, DT_FLOAT>(this);

  TestBinaryTensorOpTensor<ops::Add, DT_HALF>(this);
  TestBinaryTensorOpTensor<ops::Sub, DT_HALF>(this);
  TestBinaryTensorOpTensor<ops::Mul, DT_HALF>(this);
  TestBinaryTensorOpTensor<ops::Div, DT_HALF>(this);
  TestBinaryTensorOpTensor<ops::RealDiv, DT_HALF>(this);
  TestBinaryTensorOpTensor<ops::Minimum, DT_HALF>(this);
  TestBinaryTensorOpTensor<ops::Maximum, DT_HALF>(this);
}

TEST_F(OpConverterTest, ConvertQuantize) {
  for (const string& op :
       {"FakeQuantWithMinMaxArgs", "FakeQuantWithMinMaxVars",
        "QuantizeAndDequantizeV2", "QuantizeAndDequantizeV3"}) {
    // Input list is empty, should fail.
    NodeDef node_def = MakeNodeDef("my_quantize", op, {});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        StrCat("Invalid number of inputs for ", op, ", at my_quantize")
            .c_str());
  }
  {
    // FakeQuantWithMinMaxArgs attributes are empty, should fail.
    NodeDef node_def =
        MakeNodeDef("my_quantize", "FakeQuantWithMinMaxArgs", {"input"});
    AddTestTensor("input", {1, 2, 3});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Min or max attribute not found for FakeQuantWithMinMaxArgs "
        "at my_quantize");
  }
  {
    // FakeQuantWithMinMaxArgs ranges set via attributes, ok.
    Reset();
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT);
    auto quantize_attrs = ops::FakeQuantWithMinMaxArgs::Min(-6.0f).Max(6.0f);
    auto quantize = ops::FakeQuantWithMinMaxArgs(s.WithOpName("my_quantize"),
                                                 input, quantize_attrs);
    const NodeDef& node_def = quantize.operation.node()->def();
    AddTestTensor("input", {1, 2, 3});
    RunValidationAndConversion(node_def);
    TRT_TensorOrWeights output;
    TF_EXPECT_OK(GetTensorOrWeights("my_quantize", &output));
    EXPECT_TRUE(output.is_tensor());
    auto ranges = quantization_ranges();
    EXPECT_EQ(1, ranges.count(output.tensor()));
    EXPECT_EQ(6.0f, ranges[output.tensor()]);
  }
  {
    // FakeQuantWithMinMaxVars ranges set via inputs, ok.
    Reset();
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT);
    auto weights_min = ops::Placeholder(s.WithOpName("weights_min"), DT_FLOAT);
    auto weights_max = ops::Placeholder(s.WithOpName("weights_max"), DT_FLOAT);
    auto quantize = ops::FakeQuantWithMinMaxVars(
        s.WithOpName("my_quantize"), input, weights_min, weights_max);
    const NodeDef& node_def = quantize.operation.node()->def();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<float>("weights_min", {1}, {-6.0f});
    AddTestWeights<float>("weights_max", {1}, {6.0f});
    RunValidationAndConversion(node_def);
    TRT_TensorOrWeights output;
    TF_EXPECT_OK(GetTensorOrWeights("my_quantize", &output));
    EXPECT_TRUE(output.is_tensor());
    auto ranges = quantization_ranges();
    EXPECT_EQ(1, ranges.count(output.tensor()));
    EXPECT_EQ(6.0f, ranges[output.tensor()]);
  }
  {
    // QuantizeAndDequantizeV2 ranges set via inputs, ok.
    Reset();
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT);
    auto weights_min = ops::Placeholder(s.WithOpName("weights_min"), DT_FLOAT);
    auto weights_max = ops::Placeholder(s.WithOpName("weights_max"), DT_FLOAT);
    auto quantize = ops::QuantizeAndDequantizeV2(
        s.WithOpName("my_quantize"), input, weights_min, weights_max);
    const NodeDef& node_def = quantize.operation.node()->def();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<float>("weights_min", {1}, {-6.0f});
    AddTestWeights<float>("weights_max", {1}, {6.0f});
    RunValidationAndConversion(node_def);
    TRT_TensorOrWeights output;
    TF_EXPECT_OK(GetTensorOrWeights("my_quantize", &output));
    EXPECT_TRUE(output.is_tensor());
    auto ranges = quantization_ranges();
    EXPECT_EQ(1, ranges.count(output.tensor()));
    EXPECT_EQ(6.0f, ranges[output.tensor()]);
  }
  {
    // QuantizeAndDequantizeV2 Range inputs are tensors, should fail.
    Reset();
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT);
    auto weights_min = ops::Placeholder(s.WithOpName("weights_min"), DT_FLOAT);
    auto weights_max = ops::Placeholder(s.WithOpName("weights_max"), DT_FLOAT);
    auto quantize = ops::QuantizeAndDequantizeV2(
        s.WithOpName("my_quantize"), input, weights_min, weights_max);
    const NodeDef& node_def = quantize.operation.node()->def();
    AddTestTensor("input", {1, 2, 3});
    AddTestTensor("weights_min", {1});
    AddTestTensor("weights_max", {1});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Min and max inputs for QuantizeAndDequantizeV2 must be weights not "
        "tensors, at my_quantize");
  }
  {
    // QuantizeAndDequantizeV3 ranges set via inputs, ok.
    Reset();
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT);
    auto weights_min = ops::Placeholder(s.WithOpName("weights_min"), DT_FLOAT);
    auto weights_max = ops::Placeholder(s.WithOpName("weights_max"), DT_FLOAT);
    auto num_bits = ops::Placeholder(s.WithOpName("num_bits"), DT_INT32);
    auto quantize = ops::QuantizeAndDequantizeV3(
        s.WithOpName("my_quantize"), input, weights_min, weights_max, num_bits);
    const NodeDef& node_def = quantize.operation.node()->def();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<float>("weights_min", {1}, {-6.0f});
    AddTestWeights<float>("weights_max", {1}, {6.0f});
    AddTestWeights<int>("num_bits", {1}, {8});
    RunValidationAndConversion(node_def);
    TRT_TensorOrWeights output;
    TF_EXPECT_OK(GetTensorOrWeights("my_quantize", &output));
    EXPECT_TRUE(output.is_tensor());
    auto ranges = quantization_ranges();
    EXPECT_EQ(1, ranges.count(output.tensor()));
    EXPECT_EQ(6.0f, ranges[output.tensor()]);
  }
}

TEST_F(OpConverterTest, ConvertRelu6) {
  {
    // Input list is empty, should fail.
    NodeDef node_def = MakeNodeDef("my_relu6", "Relu6", {});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Invalid number of inputs for Relu6, at my_relu6");
  }

  // Get the NodeDef for Relu6.
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT);
  auto relu6 = ops::Relu6(s.WithOpName("my_relu6"), input);
  const NodeDef node_def = relu6.operation.node()->def();
  {
    // Input is weights, should fail.
    Reset();
    AddTestWeights<float>("input", {1}, {1.0f});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "Relu6 is only implemented for tensors, not weights, at my_relu6");
  }
  {
    // Clip tensor values and set quantization ranges, ok.
    Reset();
    AddTestTensor("input", {1, 2, 3});
    RunValidationAndConversion(node_def);
    TRT_TensorOrWeights output;
    TF_EXPECT_OK(GetTensorOrWeights("my_relu6", &output));
    EXPECT_TRUE(output.is_tensor());
    auto ranges = quantization_ranges();
    EXPECT_EQ(ranges[output.tensor()], 6.0f);

    std::vector<float> output_data(6);
    BuildAndRun<float>({{"input", {-100, -1, 0, 3, 5, 9}}}, "my_relu6",
                       &output_data);
    EXPECT_THAT(output_data, ElementsAre(0, 0, 0, 3, 5, 6));
  }
}

template <DataType dtype>
void TestConvertSquare(OpConverterTest* test) {
  test->Reset();
  typedef typename EnumToDataType<dtype>::Type CType;

  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), dtype);
  auto square = ops::Square(s.WithOpName("my_square"), input);
  NodeDef node_def = square.operation.node()->def();

  test->AddTestTensor("input", {1, 20});
  test->RunValidationAndConversion(node_def);
  TRT_TensorOrWeights output;
  TF_EXPECT_OK(test->GetTensorOrWeights("my_square", &output));
  EXPECT_TRUE(output.is_tensor());
  ExpectTrtDimsEqualsArray({1, 20}, output.tensor()->getDimensions());

  const int num_inputs = 20;
  std::vector<CType> input_data(num_inputs);
  std::vector<CType> expected_output_data(num_inputs);
  for (int i = 0; i < 20; i++) {
    const CType value = CType(i - 9);
    input_data[i] = value;
    expected_output_data[i] = value * value;
  }
  std::vector<CType> output_data(num_inputs);
  test->BuildAndRun<CType>({{"input", input_data}}, "my_square", &output_data);
  ExpectArrayNear(expected_output_data, output_data);
}

TEST_F(OpConverterTest, ConvertSquare) {
  {
    // Input list is empty, should fail.
    NodeDef node_def = MakeNodeDef("my_square", "Square", {});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "Square expects one input, at my_square");
  }
  {
    // Input is weights, should fail.
    Reset();
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT);
    auto square = ops::Square(s.WithOpName("my_square"), input);
    NodeDef node_def = square.operation.node()->def();
    AddTestWeights<float>("input", {1, 2, 3}, {1, 2, 3, 4, -5, 6});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "Square is only implemented for tensors, at my_square");
  }

  // OK. Note that kINT32 is not supported by IElementWiseLayer, so we don't
  // test DT_INT32 type here.
  TestConvertSquare<DT_FLOAT>(this);
  // TODO(tmorris): Looks like there may be a bug with this layer for FP16
  // inputs. Disabling for now.
  // TestConvertSquare<DT_HALF>(this);
}

TEST_F(OpConverterTest, ConvertActivation) {
  {
    // Input list is empty, should fail.
    NodeDef node_def = MakeNodeDef("my_act", "Relu", {});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "Relu expects one input, at my_act");
  }
  {
    // Input is weights, should fail.
    Reset();
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT);
    auto relu = ops::Relu(s.WithOpName("my_act"), input);
    const NodeDef& node_def = relu.operation.node()->def();
    AddTestWeights<int32>("input", {1, 2, 3}, {-3, -2, -1, 0, 1, 2});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "Relu is only implemented for tensors, at my_act");
  }

  // Get nodedef for activation layer.
  auto get_act_nodedef = [](string op_name) -> NodeDef {
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT);
    if (op_name == "Relu") {
      auto act = ops::Relu(s.WithOpName("my_act"), input);
      return act.operation.node()->def();
    } else if (op_name == "Sigmoid") {
      auto act = ops::Sigmoid(s.WithOpName("my_act"), input);
      return act.operation.node()->def();
    } else if (op_name == "Tanh") {
      auto act = ops::Tanh(s.WithOpName("my_act"), input);
      return act.operation.node()->def();
    }
    EXPECT_TRUE(false);
    return NodeDef();
  };
  // Get expected output for activation layer.
  auto get_act_output = [](string op_name, float input) -> float {
    if (op_name == "Relu") {
      return (input > 0.0f) ? input : 0.0f;
    } else if (op_name == "Sigmoid") {
      return 1.0f / (1.0f + std::exp(-input));
    } else if (op_name == "Tanh") {
      return std::tanh(input);
    }
    EXPECT_TRUE(false);
    return 0;
  };

  // Ok.
  for (string op_name : {"Relu", "Sigmoid", "Tanh"}) {
    Reset();
    NodeDef node_def = get_act_nodedef(op_name);
    AddTestTensor("input", {1, 2, 3});
    RunValidationAndConversion(node_def);
    TRT_TensorOrWeights output;
    TF_EXPECT_OK(GetTensorOrWeights("my_act", &output));
    EXPECT_TRUE(output.is_tensor());
    ExpectTrtDimsEqualsArray({1, 2, 3}, output.tensor()->getDimensions());

    const std::vector<float> input_data = {-100, -2, -1, 0, 1, 100};
    std::vector<float> output_data(6);
    BuildAndRun<float>({{"input", input_data}}, "my_act", &output_data);
    for (int i = 0; i < input_data.size(); i++) {
      const float expected_output = get_act_output(op_name, input_data[i]);
      EXPECT_FLOAT_EQ(output_data[i], expected_output);
    }
  }
}

TEST_F(OpConverterTest, ConvertExpandDims) {
  {
    // Input list is empty, should fail.
    NodeDef node_def = MakeNodeDef("my_expanddims", "ExpandDims", {});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Two inputs expected for ExpandDims, at my_expanddims");
  }

  // Get the NodeDef for ExpandDims.
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT);
  auto weights = ops::Placeholder(s.WithOpName("weights"), DT_INT32);
  auto expanddims = ops::ExpandDims(s.WithOpName("my_expanddims"), input, weights);
  const NodeDef& node_def = expanddims.operation.node()->def();

  {
    // Axis is a tensor, should fail.
    Reset();
    AddTestTensor("input", {1, 2, 3});
    AddTestTensor("weights", {3});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "ExpandDims expects weights for axis, at my_expanddims");
  }
  {
    // Add dim at batch dimension, should fail.
    Reset();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<int32>("weights", {1}, {0});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
        "Modifying batch dimension is not supported for ExpandDims, at my_expanddims");
  }
  {
    // Add dim at batch dimension via negative axis, should fail.
    Reset();
    AddTestTensor("input", {1, 2, 3});
    // Input is rank 4 (batch dim included)
    AddTestWeights<int32>("weights", {1}, {-5});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
        "Modifying batch dimension is not supported for ExpandDims, at my_expanddims");
  }
  {
    // Axis > rank(input), should fail.
    Reset();
    AddTestTensor("input", {1, 2, 3});
    // Input is rank 4 (batch dim included)
    AddTestWeights<int32>("weights", {1}, {5});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
        "Axis for ExpandDims is invalid, must be in the range "
        "[-rank(input) - 1, rank(input)], at my_expanddims");
  }
  {
    // Axis < -rank(input)-1, should fail.
    Reset();
    AddTestTensor("input", {1, 2, 3});
    // Input is rank 4 (batch dim included)
    AddTestWeights<int32>("weights", {1}, {-6});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
        "Axis for ExpandDims is invalid, must be in the range "
        "[-rank(input) - 1, rank(input)], at my_expanddims");
  }

  struct TestParams {
    TestParams(const std::vector<int>& input_dims,
               int axis,
               const std::vector<int>& expected_output_dims)
        : input_dims(input_dims),
          axis(axis),
          expected_output_dims(expected_output_dims) {}
    std::vector<int> input_dims;
    int axis;
    std::vector<int> expected_output_dims;
  };

  // Ok.
  const int kExpandDimsOKCases = 8;
  TestParams ok_params[kExpandDimsOKCases] = {
      TestParams{{2, 3}, 1, {1, 2, 3}},
      TestParams{{2, 3}, -3, {1, 2, 3}},
      TestParams{{2, 3}, 3, {2, 3, 1}},
      TestParams{{2, 3}, -1, {2, 3, 1}},
      TestParams{{2, 3}, 2, {2, 1, 3}},
      TestParams{{2, 3}, -2, {2, 1, 3}},
      TestParams{{6}, 1, {1, 6}},
      TestParams{{6}, -1, {6, 1}},
  };
  for (int i = 0; i < kExpandDimsOKCases; ++i) {
    Reset();
    AddTestTensor("input", ok_params[i].input_dims);
    AddTestWeights<int32>("weights", {1}, {ok_params[i].axis});
    RunValidationAndConversion(node_def);
    TRT_TensorOrWeights output;
    TF_EXPECT_OK(GetTensorOrWeights("my_expanddims", &output));
    EXPECT_TRUE(output.is_tensor());
    ExpectTrtDimsEqualsArray(ok_params[i].expected_output_dims,
                             output.tensor()->getDimensions());

    std::vector<float> output_data(6);
    BuildAndRun<float>({{"input", {1, 2, 3, 4, 5, 6}}}, "my_expanddims",
                       &output_data);
    EXPECT_THAT(output_data, ElementsAre(1, 2, 3, 4, 5, 6));
  }
}

TEST_F(OpConverterTest, ConvertSqueeze) {
  {
    // Input list is empty, should fail.
    NodeDef node_def = MakeNodeDef("my_squeeze", "Squeeze", {});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "One input expected for Squeeze, at my_squeeze");
  }
  {
    // No attrs, should fail.
    Reset();
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT);
    auto squeeze = ops::Squeeze(s.WithOpName("my_squeeze"), input);
    const NodeDef& node_def = squeeze.operation.node()->def();
    AddTestTensor("input", {1, 2, 3});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "Squeeze is only implemented for explicit dims, at my_squeeze");
  }

  // Get the NodeDef for Squeeze.
  auto get_squeeze_nodedef = [](std::vector<int> axis) -> NodeDef {
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT);
    ops::Squeeze::Attrs squeeze_attrs;
    squeeze_attrs.axis_ = gtl::ArraySlice<int>(axis);
    auto squeeze = ops::Squeeze(s.WithOpName("my_squeeze"), input,
                                squeeze_attrs);
    return squeeze.operation.node()->def();
  };

  {
    // Squeeze batch dim, should fail.
    Reset();
    NodeDef node_def = get_squeeze_nodedef({0});
    AddTestTensor("input", {1, 2, 3});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "Cannot squeeze batch dimension, at my_squeeze");
  }
  {
    // Squeeze batch dim via negative axis, should fail.
    Reset();
    NodeDef node_def = get_squeeze_nodedef({-4});
    AddTestTensor("input", {1, 2, 3});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "Cannot squeeze batch dimension, at my_squeeze");
  }
  {
    // Squeeze >= rank(input), should fail.
    Reset();
    NodeDef node_def = get_squeeze_nodedef({4});
    AddTestTensor("input", {1, 2, 3});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Axis for Squeeze is invalid, must be in the range "
        "[-rank(input), rank(input)), at my_squeeze");
  }
  {
    // Squeeze < -rank(input), should fail.
    Reset();
    NodeDef node_def = get_squeeze_nodedef({-5});
    AddTestTensor("input", {1, 2, 3});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Axis for Squeeze is invalid, must be in the range "
        "[-rank(input), rank(input)), at my_squeeze");
  }

  struct TestParams {
    TestParams(const std::vector<int>& input_dims,
               const std::vector<int>& axis,
               const std::vector<int>& expected_output_dims)
        : input_dims(input_dims),
          axis(axis),
          expected_output_dims(expected_output_dims) {}
    std::vector<int> input_dims;
    std::vector<int> axis;
    std::vector<int> expected_output_dims;
  };

  // Ok.
  const int kSqueezeOKCases = 10;
  TestParams ok_params[kSqueezeOKCases] = {
      TestParams{{1, 2, 3}, {1}, {2, 3}},
      TestParams{{1, 2, 3}, {-3}, {2, 3}},
      TestParams{{2, 3, 1}, {3}, {2, 3}},
      TestParams{{2, 3, 1}, {-1}, {2, 3}},
      TestParams{{1, 2, 1, 3, 1}, {1, 3, 5}, {2, 3}},
      TestParams{{1, 2, 1, 3, 1}, {3, 1, 5}, {2, 3}},
      TestParams{{1, 2, 1, 3, 1}, {-1, -3, -5}, {2, 3}},
      TestParams{{1, 2, 1, 3, 1}, {1, -3, 5}, {2, 3}},
      TestParams{{1, 6}, {1}, {6}},
      TestParams{{6, 1}, {2}, {6}},
  };
  for (int i = 0; i < kSqueezeOKCases; ++i) {
    Reset();
    NodeDef node_def = get_squeeze_nodedef(ok_params[i].axis);
    AddTestTensor("input", ok_params[i].input_dims);
    RunValidationAndConversion(node_def);
    TRT_TensorOrWeights output;
    TF_EXPECT_OK(GetTensorOrWeights("my_squeeze", &output));
    EXPECT_TRUE(output.is_tensor());
    ExpectTrtDimsEqualsArray(ok_params[i].expected_output_dims,
                             output.tensor()->getDimensions());

    std::vector<float> output_data(6);
    BuildAndRun<float>({{"input", {1, 2, 3, 4, 5, 6}}}, "my_squeeze",
                       &output_data);
    EXPECT_THAT(output_data, ElementsAre(1, 2, 3, 4, 5, 6));
  }
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
