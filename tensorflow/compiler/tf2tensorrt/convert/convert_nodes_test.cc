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

#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_engine_utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "tensorflow/core/common_runtime/gpu/gpu_managed_allocator.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"  // NOLINT
#include "tensorflow/core/public/session.h"
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

using absl::StrCat;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Matcher;

// TensorRT modes for testing. We define the following three modes:
// 1. Implicit batch mode: The tensors have static (known) input shape and the
//    the batch dimension (first dim) is removed from the TRT tensor shape. In
//    a loose notation: trt_shape = tf_shape[1:]. This is the standard mode of
//    a TensorRT network definition  before TensorRT 6.
// 2. Explicit batch mode: static (known) input shape, but the batch dimension
//    is part of the trt tensor shape. (trt_shape = tf_shape)
// 3. Dynamic shape mode allows unknown input shapes, and requires explicit
//    batch size definition (trt_shape = tf_shape).
//
// Note that the Converter only distinguishes between two modes:
// - use_implicit_batch == true, this corresponds to kImplicitBatch,
// - use_implicit_batch == false which includes both kExplicitBatch and
//   kDynamicShape.
//
// For the converter, the distinction between explicit batch or dynamic shape
// mode follows from the input tensors of the network: dynamic shape input
// implies dynamic shape mode, while static shape input tensors imply explicit
// batch mode. We want to test all these modes, therefore we define the
// TrtTestMode with the following three options.
enum class TrtTestMode {
  kImplicitBatch = 0,
  kExplicitBatch = 1,
  kDynamicShape = 2
};

#if IS_TRT_VERSION_GE(6, 0, 0, 0)
constexpr std::array<TrtTestMode, 3> ValidTrtModes = {
    TrtTestMode::kImplicitBatch, TrtTestMode::kExplicitBatch,
    TrtTestMode::kDynamicShape};
#else
constexpr std::array<TrtTestMode, 1> ValidTrtModes = {
    TrtTestMode::kImplicitBatch};
#endif

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

// Prints the vector to the output stream.
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
  if (!v.empty()) {
    os << '[';
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(os, ", "));
    os << "\b\b]";
  }
  return os;
}

nvinfer1::DataType TfDataTypeToTrt(DataType tf_type) {
  nvinfer1::DataType trt_type;
  Status status = TfTypeToTrtType(tf_type, &trt_type);
  EXPECT_EQ(status, Status::OK());
  return trt_type;
}

DataType TrtDataTypeToTf(nvinfer1::DataType trt_type) {
  DataType tf_type;
  Status status = TrtTypeToTfType(trt_type, &tf_type);
  EXPECT_EQ(status, Status::OK());
  return tf_type;
}

NodeDef MakeNodeDef(const string& name, const string& op,
                    const std::vector<string>& inputs,
                    const std::map<string, AttrValue> attrs = {}) {
  NodeDef node_def;
  node_def.set_name(name);
  node_def.set_op(op);
  for (const string& input : inputs) {
    node_def.add_input(input);
  }
  for (const auto& attr : attrs) {
    (*node_def.mutable_attr())[attr.first] = attr.second;
  }
  return node_def;
}

template <typename T>
NodeDef MakeConstNodeDef(const string& name, const std::vector<T>& vals,
                         const TensorShape& shape) {
  Scope s = Scope::NewRootScope();
  Tensor t = test::AsTensor<T>(vals, shape);
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

Matcher<std::vector<float>> ArrayFloatNear(const std::vector<float>& values,
                                           float max_abs_error = 1e-5,
                                           bool nan_sensitive = false) {
  std::vector<Matcher<float>> matchers;
  matchers.reserve(values.size());
  for (const float& v : values) {
    if (nan_sensitive) {
      matchers.emplace_back(::testing::NanSensitiveFloatNear(v, max_abs_error));
    } else if (max_abs_error == 0) {
      matchers.emplace_back(::testing::FloatEq(v));
    } else {
      EXPECT_GE(max_abs_error, 0);
      matchers.emplace_back(::testing::FloatNear(v, max_abs_error));
    }
  }
  return ElementsAreArray(matchers);
}

template <typename T>
void ExpectArrayNear(const std::vector<T>& lhs, absl::Span<const T> rhs) {
  ASSERT_EQ(lhs.size(), rhs.size());
  for (int i = 0; i < lhs.size(); i++) {
    EXPECT_FLOAT_EQ(lhs[i], rhs[i]);
  }
}

// Eigen::half cannot implicitly convert to float which is required for
// EXPECT_FLOAT_EQ.
template <>
void ExpectArrayNear(const std::vector<Eigen::half>& lhs,
                     absl::Span<const Eigen::half> rhs) {
  ASSERT_EQ(lhs.size(), rhs.size());
  for (int i = 0; i < lhs.size(); i++) {
    EXPECT_FLOAT_EQ(Eigen::half_impl::half_to_float(lhs[i]),
                    Eigen::half_impl::half_to_float(rhs[i]));
  }
}

template <typename T>
void ExpectArrayAlmostEqual(const std::vector<T>& lhs, absl::Span<const T> rhs,
                            T tolerance) {
  ASSERT_EQ(lhs.size(), rhs.size());
  for (int i = 0; i < lhs.size(); i++) {
    EXPECT_NEAR(lhs[i], rhs[i], tolerance);
  }
}

// Eigen::half cannot implicitly convert to float which is required for
// EXPECT_NEAR.
template <>
void ExpectArrayAlmostEqual(const std::vector<Eigen::half>& lhs,
                            absl::Span<const Eigen::half> rhs,
                            Eigen::half tolerance) {
  ASSERT_EQ(lhs.size(), rhs.size());
  for (int i = 0; i < lhs.size(); i++) {
    EXPECT_NEAR(Eigen::half_impl::half_to_float(lhs[i]),
                Eigen::half_impl::half_to_float(rhs[i]),
                Eigen::half_impl::half_to_float(tolerance));
  }
}

bool TrtShapedWeightsEquals(const TRT_ShapedWeights& lhs,
                            const TRT_ShapedWeights& rhs) {
  return TrtDimsEquals(lhs.shape_, rhs.shape_) &&
         lhs.TrtDType() == rhs.TrtDType() && lhs.GetValues() == rhs.GetValues();
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

template <typename CType>
std::vector<CType> InitTestVector(int size, CType start_value = CType(0)) {
  std::vector<CType> res;
  res.reserve(size);
  for (int i = 0; i < size; ++i) {
    res.push_back(start_value + CType(i));
  }
  return res;
}

template <typename InCType, typename OutCType>
struct StaticCaster {
  OutCType operator()(InCType in) const { return static_cast<OutCType>(in); }
};

template <typename InCType, typename OutCType>
std::vector<OutCType> CastTestVector(
    const gtl::ArraySlice<InCType>& vals) {  // non-absl ok
  std::vector<OutCType> res(vals.size());
  std::transform(vals.begin(), vals.end(), res.begin(),
                 StaticCaster<InCType, OutCType>());
  return res;
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

#if IS_TRT_VERSION_GE(5, 0, 0, 0)
  bool setDynamicRange(float min, float max) override {
    dynamic_range_ = std::max(std::abs(min), std::abs(max));
    return true;
  }

  float getDynamicRange() const override { return dynamic_range_; }
#endif

#if IS_TRT_VERSION_GE(5, 1, 0, 0)
  bool dynamicRangeIsSet() const override { return true; }

  void resetDynamicRange() override {}

  float getDynamicRangeMin() const override { return 0.f; }

  float getDynamicRangeMax() const override { return 0.f; }
#endif

#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  void setAllowedFormats(nvinfer1::TensorFormats formats) override {}

  nvinfer1::TensorFormats getAllowedFormats() const override { return 1; }

  bool isShapeTensor() const override { return false; }
  bool isExecutionTensor() const override { return true; }

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
    TRT_ShapedWeights weights(nvinfer1::DataType::kFLOAT);
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
        store.GetTempWeights(nvinfer1::DataType::kFLOAT, GetTestDims({2, 5}));
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
        ASSERT_TRUE(ptr->is_tensor());
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
      ASSERT_TRUE(ptr->is_tensor());
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
      ExpectTrtDimsEqualsArray({}, ptr->GetTrtDims());
    }
  }
}

class ValidatorTest : public ::testing::Test {
 public:
  std::unordered_map<string, OpConverter>& op_validators(
      TrtNodeValidator* validator) {
    return validator->op_validators_;
  }

  Status ConvertToTensorOrWeights(const Scope& scope, const Node* node,
                                  int output_port,
                                  TRT_TensorOrWeights* tensor_or_weights) {
    grappler::GrapplerItem item;
    TF_EXPECT_OK(scope.ToGraphDef(&item.graph));
    grappler::GraphProperties graph_properties(item);
    TF_EXPECT_OK(graph_properties.InferStatically(true));

    TrtNodeValidator validator(graph_properties, TrtPrecisionMode::FP32,
                               /*use_calibration=*/false,
                               /*use_implicit_batch=*/true);
    return validator.ConvertToTensorOrWeights(node->def(), output_port,
                                              tensor_or_weights);
  }

  const std::set<string>* GetQuantizeOps(TrtNodeValidator* validator) {
    return validator->quantize_ops;
  }
};

TEST_F(ValidatorTest, QuantizeOpsAreRegistered) {
  grappler::GrapplerItem item;
  grappler::GraphProperties graph_properties(item);
  TrtNodeValidator validator(graph_properties, TrtPrecisionMode::FP32,
                             /*use_calibration=*/false,
                             /*use_implicit_batch=*/true);
  for (const string& quantize_op : *GetQuantizeOps(&validator)) {
    QCHECK(op_validators(&validator).count(quantize_op));
  }
}

TEST_F(ValidatorTest, ConvertToTensorOrWeights) {
  // Convert Const.
  {
    Scope s = Scope::NewRootScope();
    auto node =
        ops::Const(s.WithOpName("my_const"), {1.0f, 2.0f}, TensorShape({2}));
    TRT_TensorOrWeights output;
    ExpectStatus(ConvertToTensorOrWeights(s, node.op().node(),
                                          /*output_port=*/0, &output));
    ValidateWeights<float>(output.weights(), {2}, {1.0, 2.0});
  }

  // Helper method to run ConvertToTensorOrWeights() with predefined parameters.
  auto convert_to_tensor_or_weights = [this](const std::vector<int64>& dims,
                                             TRT_TensorOrWeights* output) {
    Scope s = Scope::NewRootScope();
    const auto attrs = ops::Placeholder::Shape(PartialTensorShape{dims});
    auto feed = ops::Placeholder(s.WithOpName("feed"), DT_FLOAT, attrs);
    auto add = ops::Add(s.WithOpName("add"), feed, feed);
    return this->ConvertToTensorOrWeights(s, add.operation.node(),
                                          /*output_port=*/0, output);
  };
  // Convert non-Const with #dims > nvinfer1::Dims::MAX_DIMS+1.
  {
    TRT_TensorOrWeights output;
    ExpectStatus(
        convert_to_tensor_or_weights(
            std::vector<int64>(nvinfer1::Dims::MAX_DIMS + 2, 1), &output),
        error::OUT_OF_RANGE, "Input tensor rank is greater than 9");
  }
  // Convert non-Const with #dims < 1.
  {
    TRT_TensorOrWeights output;
    ExpectStatus(
        convert_to_tensor_or_weights({}, &output), error::INVALID_ARGUMENT,
        "Scalar input tensor is not supported since the first dimension "
        "is treated as batch dimension by TRT");
  }
  // Convert non-Const. We test the case where the non-batch dimension is
  // unknown as well, to make sure the validator allows that.
  for (const int32 non_batch_dim : {-1, 2}) {
    const int32 batch_size = 12;
    TRT_TensorOrWeights output;
    ExpectStatus(
        convert_to_tensor_or_weights({batch_size, non_batch_dim}, &output));
    ASSERT_TRUE(output.is_tensor());
    EXPECT_EQ(batch_size, output.batch_size());
    EXPECT_NE(nullptr, output.tensor());
    ExpectTrtDimsEqualsArray({non_batch_dim}, output.GetTrtDims());
  }
}

TEST_F(ValidatorTest, IsTensorRTCandidate_Basics) {
  Scope s = Scope::NewRootScope();
  auto input =
      ops::Const(s.WithOpName("const"), {1.0f, 2.0f}, TensorShape({2}));
  auto add = ops::Add(s.WithOpName("add"), input, input);
  const Node* add_node = add.operation.node();

  grappler::GrapplerItem item;
  TF_EXPECT_OK(s.ToGraphDef(&item.graph));
  grappler::GraphProperties graph_properties(item);
  TF_EXPECT_OK(graph_properties.InferStatically(true));
  TrtNodeValidator validator(graph_properties, TrtPrecisionMode::FP32,
                             /*use_calibration=*/false,
                             /*use_implicit_batch=*/true);

  bool start_conversion = false;
  bool should_fail = false;
  auto op_converter = [&start_conversion,
                       &should_fail](OpConverterParams* params) -> Status {
    if (should_fail) return errors::InvalidArgument("");
    if (!params->validation_only) start_conversion = true;
    return Status::OK();
  };

  // Validator not registered.
  ASSERT_EQ(1, op_validators(&validator).erase("Add"));
  ExpectStatus(validator.IsTensorRTCandidate(add_node), error::UNIMPLEMENTED,
               "Op type Add is not supported.");

  // Register validator.
  op_validators(&validator)["Add"] = op_converter;
  TF_EXPECT_OK(validator.IsTensorRTCandidate(add_node));
  EXPECT_EQ(false, start_conversion);

  // Let the converter return error.
  should_fail = true;
  ExpectStatus(validator.IsTensorRTCandidate(add_node),
               error::INVALID_ARGUMENT);
}

TEST(TrtNodeValidator, IsTensorRTCandidate) {
  // Create a graph containing both TRT-compatible and TRT-incompatible nodes
  // and use it to test TrtNodeValidator::IsTensorRTCandidate().
  const std::vector<int32> input_shape_array{2, 2};
  TensorShape input_shape;
  TF_EXPECT_OK(TensorShapeUtils::MakeShape(input_shape_array, &input_shape));

  Scope s = Scope::NewRootScope();
  ops::Placeholder::Attrs feed_attrs;
  TF_EXPECT_OK(
      TensorShapeUtils::MakeShape(input_shape_array, &feed_attrs.shape_));

  // Compatible input.
  auto feed = ops::Placeholder(s.WithOpName("feed"), DT_FLOAT, feed_attrs);
  auto const_1 = ops::Const(s.WithOpName("const_1"), 1.0f, input_shape);

  // Compatible MatMul.
  auto matmul = ops::MatMul(s.WithOpName("matmul"), feed, const_1);

  // Incompatible MatMul.
  ops::MatMul::Attrs matmul_attrs;
  matmul_attrs.transpose_a_ = true;
  auto incompatible_matmul = ops::MatMul(s.WithOpName("incompatible_matmul"),
                                         feed, const_1, matmul_attrs);

  // Unsupported op.
  auto unsupported_op = ops::Erf(s.WithOpName("sin"), feed);

  // Incompatible input.
  auto incompatible_feed = ops::Placeholder(s.WithOpName("feed"), DT_DOUBLE);
  auto const_2 = ops::Const(s.WithOpName("const_2"), 1.0, input_shape);
  // Compatible op with incompatible input.
  auto matmul_with_incompatible_input =
      ops::MatMul(s.WithOpName("matmul_with_incompatible_input"),
                  incompatible_feed, const_2);

  // Quantize ops.
  auto quantize_attrs = ops::FakeQuantWithMinMaxArgs::Min(-6.0f).Max(6.0f);
  auto quantize = ops::FakeQuantWithMinMaxArgs(s.WithOpName("quantize"), feed,
                                               quantize_attrs);

  // Get GrapplerItem and GraphProperties.
  grappler::GrapplerItem item;
  TF_EXPECT_OK(s.ToGraphDef(&item.graph));
  Tensor feed_tensor(DT_FLOAT, input_shape);
  item.feed.push_back(std::make_pair("feed", feed_tensor));
  grappler::GraphProperties graph_properties(item);
  TF_EXPECT_OK(graph_properties.InferStatically(true));

  for (const TrtPrecisionMode precision_mode :
       {TrtPrecisionMode::FP32, TrtPrecisionMode::INT8}) {
    TrtNodeValidator validator(graph_properties, precision_mode,
                               /*use_calibration=*/false,
                               /*use_implicit_batch=*/true);
    TF_EXPECT_OK(validator.IsTensorRTCandidate(matmul.operation.node()));
    ExpectStatus(
        validator.IsTensorRTCandidate(incompatible_matmul.operation.node()),
        error::INVALID_ARGUMENT,
        "Cannot transpose first input if it is a tensor with fewer than 2 "
        "non-batch dimensions.");
    ExpectStatus(validator.IsTensorRTCandidate(unsupported_op.operation.node()),
                 error::UNIMPLEMENTED, "Op type Erf is not supported");
    ExpectStatus(validator.IsTensorRTCandidate(
                     matmul_with_incompatible_input.operation.node()),
                 error::INTERNAL,
                 "Failed to convert input feed_1 to a TRT_TensorOrWeights");
    if (precision_mode == TrtPrecisionMode::INT8) {
      TF_EXPECT_OK(validator.IsTensorRTCandidate(quantize.operation.node()));
    } else {
      ExpectStatus(validator.IsTensorRTCandidate(quantize.operation.node()),
                   error::UNIMPLEMENTED,
                   "Op type FakeQuantWithMinMaxArgs is not supported");
    }
  }
}

class ConverterTest : public ::testing::Test {
 public:
  ConverterTest() { Reset(); }

  void Reset() {
    converter_ =
        std::move(Converter::Create(TrtPrecisionMode::FP32,
                                    /*use_calibration=*/false, &logger_,
                                    /*use_implicit_batch=*/true)
                      .ValueOrDie());
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
      nvinfer1::ITensor* input_tensor = params->inputs[0].tensor();
      nvinfer1::IShuffleLayer* layer =
          params->converter->network()->addShuffle(*input_tensor);
      layer->setFirstTranspose(perm);
      nvinfer1::ITensor* output_tensor = layer->getOutput(0);
      params->outputs->emplace_back(output_tensor);
      output_tensors.push_back(output_tensor);
    }
    TRT_ShapedWeights output_weights(nvinfer1::DataType::kFLOAT);
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
  nvinfer1::ITensor* output_tensor = nullptr;

  // Rank doesn't match.
  ExpectStatus(
      converter_->TransposeTensor(input_tensor, {0, 1}, "Bad perm",
                                  &output_tensor),
      error::INVALID_ARGUMENT,
      "Rank of perm for transpose does not match with that of the input");

  // Transpose at batch dimension.
  ExpectStatus(converter_->TransposeTensor(input_tensor, {1, 0, 2, 3},
                                           "Batch perm", &output_tensor),
               error::UNIMPLEMENTED,
               "Transpose at batch dimension is not supported.");

  // OK.
  TF_EXPECT_OK(converter_->TransposeTensor(input_tensor, {0, 3, 1, 2}, "OK",
                                           &output_tensor));
  ExpectTrtDimsEqualsArray({5, 2, 3}, output_tensor->getDimensions());
}

void TestPrepareTensorForShape(
    const std::vector<int>& input_dims, const std::vector<int>& reshape_dims,
    const std::vector<int>& expected_tensor_dims, bool input_is_tensor,
    Converter* converter, TrtWeightStore* weight_store,
    error::Code expected_code = error::OK,
    const char* expected_error_msg_substr = nullptr) {
  TRT_TensorOrWeights input;
  if (input_is_tensor) {
    input = TRT_TensorOrWeights(converter->network()->addInput(
        "", nvinfer1::DataType::kFLOAT, GetTestDims(input_dims)));
  } else {
    input = TRT_TensorOrWeights(weight_store->GetTempWeights(
        nvinfer1::DataType::kFLOAT, GetTestDims(input_dims)));
  }
  nvinfer1::ITensor* output_tensor = nullptr;

  for (bool validation_only : {false, true}) {
    const Status status = converter->PrepareTensorForShape(
        input, GetTestDims(reshape_dims), validation_only, &output_tensor);
    if (expected_code == error::OK) {
      TF_EXPECT_OK(status);
      if (validation_only) {
        EXPECT_EQ(nullptr, output_tensor);
      } else {
        ExpectTrtDimsEqualsArray(expected_tensor_dims,
                                 output_tensor->getDimensions());
      }
    } else {
      ExpectStatus(status, expected_code, expected_error_msg_substr);
    }
  }
}

TEST_F(ConverterTest, PrepareTensorForShape) {
  for (bool input_is_tensor : {true, false}) {
    // Shape size doesn't match.
    Reset();
    TestPrepareTensorForShape({2, 3, 5}, {2, 3, 6}, {}, input_is_tensor,
                              converter_.get(), weight_store_,
                              error::INVALID_ARGUMENT, "Incompatible shapes");

    // Regular shape.
    Reset();
    TestPrepareTensorForShape({2, 3, 5}, {10, 3}, {10, 3}, input_is_tensor,
                              converter_.get(), weight_store_);

    // Reshape to zero rank.
    Reset();
    TestPrepareTensorForShape({1, 1}, {}, {}, input_is_tensor, converter_.get(),
                              weight_store_);
  }

  // Tensor input with zero rank.
  Reset();
  TestPrepareTensorForShape({}, {1, 1}, {1, 1}, /*input_is_tensor=*/true,
                            converter_.get(), weight_store_);

  // TODO(aaroey): we should check the case where uninferred dimensions are
  // not an exact divisor of input dim ensions, e.g. for dims {-1, 7}.

  // Infer tensor shape, ok.
  Reset();
  TestPrepareTensorForShape({2, 3, 5}, {-1, 2}, {15, 2},
                            /*input_is_tensor=*/true, converter_.get(),
                            weight_store_);

  // Infer weight shape, should fail.
  Reset();
  TestPrepareTensorForShape({2, 3, 5}, {-1, 2}, {15, 2},
                            /*input_is_tensor=*/false, converter_.get(),
                            weight_store_, error::INVALID_ARGUMENT,
                            "Shape is not fully defined");
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
  TRT_ShapedWeights weights = weight_store->GetTempWeights(
      TfDataTypeToTrt(DataTypeToEnum<T>::v()), GetTestDims({2, 3}));
  const std::vector<T> values = {T(3), T(1), T(2), T(6), T(5), T(4)};
  memcpy(weights.GetValues(), values.data(), weights.size_bytes());

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
  // Asymmetric range
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
  Logger logger;
  auto int8_converter = Converter::Create(TrtPrecisionMode::INT8,
                                          /*use_calibration=*/true, &logger,
                                          /*use_implicit_batch=*/true)
                            .ValueOrDie();
  int8_converter->ProvideQuantizationRange(&input, -5.0f, 5.0f);
  int8_converter->ProvideQuantizationRange(&not_infer, -100.0f, 100.0f);
  int8_converter->MarkQuantizationRangesAsInferrable(&input, &infer_1);
  int8_converter->MarkQuantizationRangesAsInferrable(&infer_1, &infer_2);
  int8_converter->MarkQuantizationRangesAsInferrable(&infer_2, &infer_3);

  // Input range should be inferred along the chain and applied to tensors.
  int8_converter->MaybeApplyQuantizationRanges();
#if IS_TRT_VERSION_GE(5, 0, 0, 0)
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
        GetTrtBroadcastShape(operand_1, operand_2, /*check_feasibility=*/true,
                             /*use_implicit_batch=*/true, &operand_1_new_dims,
                             &operand_2_new_dims),
        expected_code, expected_error_msg_substr);
    if (expected_code == error::OK) {
      ExpectTrtDimsEqualsArray(expected_operand_1_shape, operand_1_new_dims);
      ExpectTrtDimsEqualsArray(expected_operand_2_shape, operand_2_new_dims);
    }
    // operand_2 broadcast operand_1
    ExpectStatus(
        GetTrtBroadcastShape(operand_2, operand_1, /*check_feasibility=*/true,
                             /*use_implicit_batch=*/true, &operand_2_new_dims,
                             &operand_1_new_dims),
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
  symmetric_test({3}, {1, 1, 3}, kIsTensor, kIsNotTensor, {}, {},
                 error::INVALID_ARGUMENT,
                 "Broadcasting beyond batch dimension is not supported "
                 "(tensor #dims 2 vs broadcast #dims 3)",
                 /*operand_1_batch_size=*/2);

  // Both inputs are tensors.
  symmetric_test({1, 1, 1}, {1, 1}, kIsTensor, kIsTensor, {}, {},
                 error::INVALID_ARGUMENT,
                 "Broadcasting beyond batch dimension is not supported "
                 "(tensor #dims 3 vs broadcast #dims 4)");
  symmetric_test({1, 3}, {3}, kIsTensor, kIsTensor, {}, {},
                 error::INVALID_ARGUMENT,
                 "Broadcasting beyond batch dimension is not supported "
                 "(tensor #dims 2 vs broadcast #dims 3)");
  symmetric_test({1, 3, 4}, {2, 1, 4}, kIsTensor, kIsTensor, {1, 3, 4},
                 {2, 1, 4});
  symmetric_test({1, 1, 1}, {1, 1, 1, 1}, kIsTensor, kIsTensor, {}, {},
                 error::INVALID_ARGUMENT,
                 "Broadcasting beyond batch dimension is not supported "
                 "(tensor #dims 4 vs broadcast #dims 5)");
  symmetric_test({2, 3}, {7, 5}, kIsTensor, kIsTensor, {}, {},
                 error::INVALID_ARGUMENT, "Infeasible broadcast scheme");
}

TEST_F(ConverterTest, CreateConstantLayer) {
  for (auto dtype : {nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT32}) {
    TRT_ShapedWeights weights =
        weight_store_->GetTempWeights(dtype, GetTestDims({2, 3, 5}));
    nvinfer1::ITensor* tensor =
        converter_->CreateConstantLayer(weights, GetTestDims({3, 10}));
    ASSERT_NE(nullptr, tensor);
    EXPECT_EQ(dtype, tensor->getType())
        << "Expected " << DebugString(dtype) << " vs. actual "
        << DebugString(tensor->getType());
    ExpectTrtDimsEqualsArray({3, 10}, tensor->getDimensions());
  }
}

class ConvertGraphDefToEngineTest : public ::testing::Test {
 public:
  Status RunConvertGraphDefToEngine(Scope* s) {
    GraphDef gdef;
    TF_EXPECT_OK(s->ToGraphDef(&gdef));
    std::vector<PartialTensorShape> input_shapes;
    int batch_size = -1;
    for (const NodeDef& node : gdef.node()) {
      absl::string_view node_name(node.name());
      if (absl::ConsumePrefix(&node_name, IONamePrefixes::kInputPHName)) {
        int port = -1;
        EXPECT_TRUE(absl::SimpleAtoi(node_name, &port)) << node.name();
        if (input_shapes.size() < port + 1) input_shapes.resize(port + 1);
        input_shapes[port] =
            PartialTensorShape(node.attr().at("shape").shape());
        if (batch_size == -1) {
          batch_size = input_shapes[port].dim_size(0);
        } else {
          EXPECT_EQ(batch_size, input_shapes[port].dim_size(0));
        }
      }
    }
    // TODO(laigd): execute the engine and get outputs.
    return ConvertGraphDefToEngine(
        gdef, TrtPrecisionMode::FP32, /*max_batch_size=*/1,
        /*max_workspace_size_bytes=*/64 << 20, input_shapes, &logger_,
        /*allocator=*/nullptr, /*calibrator=*/nullptr, &engine_,
        /*use_calibration=*/false, /*use_implicit_batch=*/true,
        /*convert_successfully=*/nullptr, /*profiles=*/nullptr);
  }

 protected:
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine_;

 private:
  Logger logger_;
};

TEST_F(ConvertGraphDefToEngineTest, IdentityGraph) {
  Scope s = Scope::NewRootScope();
  auto input =
      ops::Placeholder(s.WithOpName(StrCat(IONamePrefixes::kInputPHName, 0)),
                       DT_FLOAT, ops::Placeholder::Shape({1, 1}));
  auto output = ops::Identity(s.WithOpName("identity1"), input);
  output = ops::Identity(s.WithOpName("identity2"), output);
  output = ops::Identity(s.WithOpName(StrCat(IONamePrefixes::kOutputPHName, 0)),
                         output);
  // If the converter marks the input tensor as output tensor, the conversion
  // below will fail with:
  // > TensorRTOutputPH_0 cannot be both input and output
  // > Network must have at least one output
  TF_EXPECT_OK(RunConvertGraphDefToEngine(&s));
}

// Returns a vector of shapes from a vector of input tensors. This can be used
// to create optimization profiles.
Status GetShapeFromDataVec(DataVec input_data,
                           std::vector<TensorShape>* shape_vec) {
  shape_vec->reserve(input_data.size());
  std::transform(input_data.begin(), input_data.end(),
                 std::back_inserter(*shape_vec),
                 [](InputOutputData x) { return x.tensor.shape(); });
  return Status::OK();
}

template <typename T>
inline absl::Span<const T> GetSpanForData(const InputOutputData& data) {
  const auto& tensor_map = data.tensor.flat<T>();
  return absl::Span<const T>(tensor_map.data(), tensor_map.size());
}

std::vector<float> GetDataAsFloat(InputOutputData& data) {
  if (data.tensor.dtype() == DT_FLOAT) {
    auto span = GetSpanForData<float>(data);
    return std::vector<float>(span.begin(), span.end());
  }
  if (data.tensor.dtype() == DT_HALF) {
    return CastTestVector<Eigen::half, float>(
        GetSpanForData<Eigen::half>(data));
  }
  if (data.tensor.dtype() == DT_INT32) {
    return CastTestVector<int32, float>(GetSpanForData<int32>(data));
  }
  LOG(FATAL) << "DataType not supported for testing "
             << DataTypeString(data.tensor.dtype());
}
// Class to test various op converters, using both a TrtNodeValidator and
// Converter.
class OpConverterTest : public ::testing::Test {
 public:
  OpConverterTest()
      : scope_(Scope::NewRootScope()), allocator_(new GpuManagedAllocator()) {
    QCHECK_EQ(0, cudaStreamCreate(&stream_));
    Reset();
  }

  ~OpConverterTest() override { QCHECK_EQ(0, cudaStreamDestroy(stream_)); }

  Status GetTensorOrWeights(const string& name, TRT_TensorOrWeights* output) {
    return converter_->GetTensorOrWeights(name, output);
  }

  void Reset(TrtPrecisionMode precision_mode_to_test = TrtPrecisionMode::FP32,
             TrtTestMode trt_mode = TrtTestMode::kImplicitBatch) {
    // Destroy existing TRT objects in a proper order.
    converter_.reset(nullptr);
    engine_.reset(nullptr);

    // Re-create them in proper order.
    converter_ =
        std::move(Converter::Create(precision_mode_to_test,
                                    /*use_calibration=*/false, &logger_,
                                    /*use_implicit_batch=*/trt_mode ==
                                        TrtTestMode::kImplicitBatch)
                      .ValueOrDie());

    // Reset other related artifacts.
    scope_ = Scope::NewRootScope();
  }

  // Constructs a flat tensor with 'vals' in Unified Memory.
  template <typename T>
  Tensor AsTensor(gtl::ArraySlice<T> vals) {  // non-absl ok
    Tensor ret(allocator_.get(), DataTypeToEnum<T>::value,
               {static_cast<int64>(vals.size())});
    std::copy_n(vals.data(), vals.size(), ret.flat<T>().data());
    return ret;
  }

  // Constructs a tensor of "shape" with values "vals" in Unified Memory.
  template <typename T>
  Tensor AsTensor(gtl::ArraySlice<T> vals,  // non-absl ok
                  const TensorShape& shape) {
    Tensor ret(allocator_.get(), DataTypeToEnum<T>::value,
               {static_cast<int64>(vals.size())});
    CHECK(ret.CopyFrom(AsTensor(vals), shape));
    return ret;
  }

  // Constructs a tensor with given values (vals). The tensor type is defined by
  // the tf_dtype argument, its shape is given by input_dims. The tensor is
  // constructed using the allocator of OpConverterTest in Unified Memory.
  template <typename T>
  Tensor AsTensor(std::vector<T> vals, const std::vector<int> input_dims,
                  DataType tf_dtype) {
    Tensor ret(allocator_.get(), tf_dtype, {static_cast<int64>(vals.size())});
    if (tf_dtype == DT_FLOAT) {
      auto conv_vals = CastTestVector<T, float>(vals);
      std::copy_n(conv_vals.data(), conv_vals.size(), ret.flat<float>().data());
    } else if (tf_dtype == DT_HALF) {
      auto conv_vals = CastTestVector<T, Eigen::half>(vals);
      std::copy_n(conv_vals.data(), conv_vals.size(),
                  ret.flat<Eigen::half>().data());
    } else if (tf_dtype == DT_INT32) {
      auto conv_vals = CastTestVector<T, int32>(vals);
      std::copy_n(conv_vals.data(), conv_vals.size(), ret.flat<int32>().data());
    } else {
      LOG(FATAL) << "Cannot create tensor with type "
                 << DataTypeString(tf_dtype);
    }
    TensorShape shape;
    TF_EXPECT_OK(TensorShapeUtils::MakeShape(input_dims, &shape));
    CHECK(ret.CopyFrom(ret, shape));
    return ret;
  }

  // Constructs a flat tensor in Unified Memory.
  template <typename T>
  Tensor ConstructTensor(int data_size, const T& value = T()) {
    std::vector<T> values(data_size, value);
    return AsTensor<T>(values);
  }

  // Constructs a flat tensor in Unified Memory.
  template <typename T>
  Tensor ConstructTensor(int data_size, const T& value, DataType tf_dtype) {
    std::vector<T> values(data_size, value);
    return AsTensor<T>(values, {data_size}, tf_dtype);
  }

  void CheckDataTypeMatches(const DataVec& datas) {
    for (const auto& data : datas) {
      const int input_index = engine_->getBindingIndex(data.name.c_str());
      ASSERT_NE(-1, input_index);
      const nvinfer1::DataType trt_dtype =
          engine_->getBindingDataType(input_index);
      const DataType tf_dtype = TrtDataTypeToTf(trt_dtype);
      ASSERT_EQ(data.tensor.dtype(), tf_dtype)
          << DataTypeString(data.tensor.dtype()) << " vs. "
          << DataTypeString(tf_dtype);
    }
  }

  Status BuildAndRun(const DataVec& input_data, DataVec* output_data,
                     const int batch_size = 1) {
    // Mark the output tensor as TRT engine output.
    std::vector<Converter::EngineOutputInfo> output_info;
    for (const auto& data : *output_data) {
      output_info.push_back(
          {data.name, data.name, TfDataTypeToTrt(data.tensor.dtype())});
    }
    TF_RETURN_IF_ERROR(converter_->RenameAndMarkOutputTensors(output_info));

    // Build the TRT engine.
    if (engine_.get() != nullptr) {
      return errors::Internal("Engine already exists");
    }
    TrtShapeOptimizationProfile profiles;
    if (!converter_->use_implicit_batch()) {
      // Create a single optimization profile for explicit batch mode
      std::vector<TensorShape> input_shapes;
      TF_RETURN_IF_ERROR(GetShapeFromDataVec(input_data, &input_shapes));
      profiles.AddShape(input_shapes);
      profiles.InitProfiles();
    }
    TF_RETURN_IF_ERROR(
        converter_->BuildCudaEngine(&engine_,
                                    /*max_batch_size=*/batch_size,
                                    /*max_workspace_size_bytes=*/1 << 26,
                                    /*allocator=*/nullptr,
                                    /*calibrator=*/nullptr,
                                    /*profiles=*/&profiles));
    CHECK_NOTNULL(engine_.get());
    CheckDataTypeMatches(input_data);
    CheckDataTypeMatches(*output_data);

    const int num_bindings = input_data.size() + output_data->size();
    std::vector<void*> buffers(num_bindings);

    if (engine_->getNbBindings() != num_bindings) {
      return errors::Internal("Number of bindings do not match");
    }
    // Since we have only 1 optimization profile (which is enabled by default)
    // it is fine to create execution context directly, instead of calling
    // profiles.CreateExecutionContexts()
    TrtUniquePtrType<nvinfer1::IExecutionContext> execution_context(
        engine_->createExecutionContext());

    // Prepare input bindings.
    TF_RETURN_IF_ERROR(SetTrtEngineInputs(
        engine_.get(), execution_context.get(), 0, buffers,
        converter_->use_implicit_batch(), batch_size, nullptr, &input_data));
    // Prepare output bindings.
    TF_RETURN_IF_ERROR(SetTrtEngineOutputs(
        engine_.get(), execution_context.get(), 0, buffers,
        converter_->use_implicit_batch(), batch_size, nullptr, output_data));
    // Execute the TRT engine.
    TF_RETURN_IF_ERROR(TrtEnqueue(execution_context.get(), buffers, stream_,
                                  converter_->use_implicit_batch(),
                                  batch_size));
    cudaStreamSynchronize(stream_);
    return Status::OK();
  }

  bool HasStaticShape(const nvinfer1::Dims& dims) const {
    if (dims.nbDims < 0) return false;
    for (int i = 0; i < dims.nbDims; ++i) {
      if (dims.d[i] < 0) return false;
    }
    return true;
  }

  bool HasStaticShape(std::vector<int> dims) const {
    return !absl::c_any_of(dims, [](int i) { return i < 0; });
  }

  // Adds ITensor for both validation and conversion, assuming explicit batch
  // dimension is included in dims (ie for an NCHW tensor dims = {N, C, H, W}).
  void AddTestTensorWithTFDims(
      const string& name, const std::vector<int32>& dims,
      nvinfer1::DataType trt_dtype = nvinfer1::DataType::kFLOAT) {
    DataType tf_dtype = TrtDataTypeToTf(trt_dtype);
    ops::Placeholder::Attrs attrs;
    TF_EXPECT_OK(TensorShapeUtils::MakeShape(dims, &attrs.shape_));

    auto input = ops::Placeholder(scope_.WithOpName(name), tf_dtype, attrs);
    node_inputs_[name] = input.output;

    // Add a real ITensor for conversion conditionally.
    const nvinfer1::Dims trt_dims =
        TensorShapeToTrtDims(attrs.shape_, converter_->use_implicit_batch());
    if (!converter_->use_implicit_batch() || HasStaticShape(trt_dims)) {
      int batch_size = dims[0];
      TF_EXPECT_OK(
          converter_->AddInputTensor(name, trt_dtype, trt_dims, batch_size));
    }
  }

  // Adds ITensor for both validation and conversion. The difference compared to
  // AddTestTensorWithTFDims is in the meaning of the dims parameter. To define
  // a tensor with NCHW shape, here we set dims = {C,H,W} and batch_size = N.
  // TODO(tfeher) remove this function once all test are updated to use the
  // other version of AddTestTensor (defined by
  // ParameterizedOpConverterTestBase).
  void AddTestTensor(
      const string& name, const std::vector<int32>& dims, int batch_size = 1,
      nvinfer1::DataType trt_dtype = nvinfer1::DataType::kFLOAT) {
    std::vector<int32> dims_with_batch(dims.size() + 1);
    dims_with_batch[0] = batch_size;
    std::copy(dims.begin(), dims.end(), dims_with_batch.begin() + 1);
    AddTestTensorWithTFDims(name, dims_with_batch, trt_dtype);
    if (HasStaticShape(dims)) {
      ASSERT_EQ(batch_size, converter_->batch_size_);
    }
  }

  // Add weights for both validation and conversion.
  template <typename T>
  void AddTestWeights(const string& name, const std::vector<int>& dims,
                      const std::vector<T>& values) {
    // Add weights for validation.
    TensorShape shape;
    TF_EXPECT_OK(TensorShapeUtils::MakeShape(dims, &shape));
    Tensor t = AsTensor<T>(values, shape);
    node_inputs_[name] = ops::Const(scope_.WithOpName(name), t);

    // Add weights for conversion.
    const nvinfer1::DataType dtype = TfDataTypeToTrt(DataTypeToEnum<T>::v());
    const nvinfer1::Dims trt_dims = GetTestDims(dims);
    const int64_t num_elements = TrtWeightDimsNumElements(trt_dims);
    QCHECK_EQ(num_elements, values.size())
        << num_elements << " vs " << values.size();
    TRT_ShapedWeights weights(dtype);
    if (num_elements) {
      weights = converter_->weight_store_.GetTempWeights(dtype, trt_dims);
      QCHECK_EQ(weights.size_bytes(), sizeof(T) * values.size())
          << weights.size_bytes() << " vs " << sizeof(T) * values.size();
      memcpy(weights.GetValues(), values.data(), weights.size_bytes());
    }
    TF_EXPECT_OK(
        converter_->AddTensorOrWeights(name, TRT_TensorOrWeights{weights}));
  }

  template <typename T>
  void AddTestWeights(const string& name, const std::vector<int>& dims,
                      const std::vector<T>& values, DataType tf_dtype) {
    if (tf_dtype == DT_FLOAT) {
      AddTestWeights(name, dims, CastTestVector<T, float>(values));
    } else if (tf_dtype == DT_HALF) {
      AddTestWeights(name, dims, CastTestVector<T, Eigen::half>(values));
    } else if (tf_dtype == DT_INT32) {
      AddTestWeights(name, dims, CastTestVector<T, int32>(values));
    } else {
      FAIL() << "Cannot create test weights with type "
             << DataTypeString(tf_dtype);
    }
  }

  // Test validation in validation-only mode.
  void RunValidation(const Node* node, error::Code expected_code = error::OK,
                     const char* expected_msg_substr = nullptr) {
    grappler::GrapplerItem item;
    TF_EXPECT_OK(scope_.ToGraphDef(&item.graph));
    grappler::GraphProperties graph_properties(item);
    TF_EXPECT_OK(graph_properties.InferStatically(true));

    TrtNodeValidator validator(graph_properties, converter_->precision_mode(),
                               /*use_calibration=*/false,
                               converter_->use_implicit_batch());
    ExpectStatus(validator.IsTensorRTCandidate(node), expected_code,
                 expected_msg_substr);
  }

  void RunConversion(const Node* node, error::Code expected_code = error::OK,
                     const char* expected_msg_substr = nullptr) {
    ExpectStatus(converter_->ConvertNode(node->def()), expected_code,
                 expected_msg_substr);
  }

  // Helper method to run both validation and conversion, when the expected
  // output are same.
  void RunValidationAndConversion(const NodeDef& node_def,
                                  error::Code expected_code = error::OK,
                                  const char* expected_msg_substr = nullptr,
                                  bool should_run_conversion = true) {
    // Add the node to the graph.
    // TODO(laigd): we should accept a function that adds the node using
    // `scope_`, so individual test case can reuse the scope object and we don't
    // need to add the edges here by ourselves.
    Graph* graph = scope_.graph();
    Status status;
    Node* node = graph->AddNode(std::move(node_def), &status);
    TF_EXPECT_OK(status);
    for (int i = 0; i < node_def.input().size(); ++i) {
      const string& input_name = node_def.input(i);
      const auto& itr = node_inputs_.find(input_name);
      QCHECK(itr != node_inputs_.end());
      const Output& input = itr->second;
      graph->AddEdge(input.node(), input.index(), node, i);
    }

    RunValidation(node, expected_code, expected_msg_substr);
    if (should_run_conversion) {
      RunConversion(node, expected_code, expected_msg_substr);
    }
  }

  // Helper method to run both validation and conversion, and check the output
  // shape.
  void RunValidationAndConversion(const NodeDef& node_def, const Status& status,
                                  const char* output_name,
                                  const std::vector<int>& exp_out_dims) {
    RunValidationAndConversion(node_def, status.code(),
                               status.error_message().c_str(), true);
    if (status.ok()) {
      TRT_TensorOrWeights output;
      TF_EXPECT_OK(GetTensorOrWeights(output_name, &output));
      ASSERT_TRUE(output.is_tensor());
      if (converter_->use_implicit_batch() && !exp_out_dims.empty()) {
        // We only check output shape implicit batch mode. In dynamic shape
        // mode we need to wait for the concrate input shapes to be defined
        // (by setBindingDimensions before enqueue) before we can check
        // whether the output dims are equal.
        //
        // TODO(tamas) enable this check in explicit_batch_mode

        // Removing batch dim
        auto out_dims =
            std::vector<int>(exp_out_dims.begin() + 1, exp_out_dims.end());
        ExpectTrtDimsEqualsArray(out_dims, output.tensor()->getDimensions());
      }
    }
  }

  // Expose quantization_ranges_ for tests
  std::unordered_map<nvinfer1::ITensor*, float>& quantization_ranges() {
    return converter_->quantization_ranges_;
  }

  void PropagateQuantizationRanges() {
    converter_->PropagateQuantizationRanges();
  }
  std::unique_ptr<Converter> converter_;

 private:
  Logger logger_;
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine_;
  cudaStream_t stream_;
  // Used to create placeholders with shape and data type information. The
  // created placeholders will be used as inputs to the node to be verified,
  // thus we need the shape and data type information to get a non-empty
  // GraphProperties.
  Scope scope_;
  std::unordered_map<string, Output> node_inputs_;
  std::unique_ptr<Allocator> allocator_;
};

// General test parameters to be used with ops that take a single input tensor.
struct TestParamBase {
  // Concrete input dimensions for the test (including the batch dim)
  std::vector<int> input_dims;

  // Dimensions to define an input with PartialTensorShape. This can be used to
  // define networks with dynamic input shape. It can be left empty, in that
  // case AddTestTensor sets partial shapes that are appropriate to TrtTestMode.
  std::vector<int> partial_input_dims;

  // Concrete (static) output dimensions, including batch size as first dim
  std::vector<int> expected_output_dims;

  // Parameter vector, has converter specific meaning.
  std::vector<int> param;

  // Expected status of conversion (with concrete error message)
  Status status;

  // Expected status of BuildAndRun
  Status runtime_status;
};

std::ostream& operator<<(std::ostream& os, const TestParamBase& p) {
  os << "input_dims" << p.input_dims;
  if (!p.partial_input_dims.empty()) {
    os << ", partial_input_dims" << p.partial_input_dims;
  }
  if (!p.expected_output_dims.empty()) {
    os << ", exp_out_dims" << p.expected_output_dims;
  }
  if (!p.param.empty()) {
    os << ", param" << p.param;
  }
  os << ", " << p.status;
  return os;
}

// Parameterized version of OpConverterTest. We have the following parameters:
// 1. TrtTestMode: implicit batch, explicit batch, dynamic shape modes
// 2. DataType of the input TF tensors: DT_FLOAT, DT_HALF, DT_INT32
// 3. TrtPrecisionMode argument for the Converter: FP32, FP16, INT8
// We will introduce subclasses that will be instantiated using different
// combinations of the DataType and TrtPrecisionMode parameters.
class ParameterizedOpConverterTestBase
    : public OpConverterTest,
      public ::testing::WithParamInterface<
          std::tuple<TrtTestMode, DataType, TrtPrecisionMode>> {
 public:
  ParameterizedOpConverterTestBase()
      : trt_mode(std::get<0>(GetParam())),
        tf_dtype(std::get<1>(GetParam())),
        converter_precision(std::get<2>(GetParam())) {}

  void Reset() {
    OpConverterTest::Reset(converter_precision, trt_mode);
    input_data_.clear();
  }

  // Adds an input ITensor for TRT network. Also creates the corresponding TF
  // tensor, and stores it in the list of inputs (input_data_).
  //
  // The TF tensor is always created with concrete static input shape given by
  // dims. The ITensor can have static or dynamic shape based on the trt_mode
  // attribute. The ITensor shape is set automatically according to the trt_mode
  // parameter, unless the user overrides it with an explicit
  // partial_input_shape_dims argument.
  //
  // Parameters:
  // - name of the input node
  // - dims actual dimensions of the tensor that we will use during the test
  //   (including explicit batch dim)
  // - values initial values for the TF tensor
  // - dtype data type of the tensor
  // - partial_input_shape dimensions which can incude unknown shapes. This can
  //   be empty, in that case the partial_input_shape will be set automatically
  //   depending on the trt_mode argument. (This argument also includes explicit
  //   batch dim).
  //
  template <typename T>
  void AddTestTensor(const string& name, const std::vector<int32>& dims,
                     DataType tf_dtype, const std::vector<T>& values,
                     const std::vector<int32>& partial_input_shape_dims = {}) {
    std::vector<int32> partial_shape;
    if (!partial_input_shape_dims.empty()) {
      partial_shape = partial_input_shape_dims;
    } else {
      if (trt_mode == TrtTestMode::kDynamicShape) {
        // In dynamic shape mode we make all dims unknown.
        partial_shape = std::vector<int32>(dims.size(), -1);
      } else {
        // Use static (known) input shapes.
        partial_shape = dims;
      }
    }
    AddTestTensorWithTFDims(name, partial_shape, TfDataTypeToTrt(tf_dtype));
    if (!values.empty()) {
      VLOG(2) << "Adding test tensor: " << name << " "
              << DataTypeString(tf_dtype);
      InputOutputData data{name, AsTensor(values, dims, tf_dtype)};
      VLOG(2) << "Added tensor: " << data.name
              << DataTypeString(data.tensor.dtype());
      input_data_.push_back(data);
    }
  }

  // Adds test tensor (same as above) but with the default tf_dtype defined by
  // the test params.
  void AddTestTensor(const string& name, const std::vector<int32>& dims,
                     const std::vector<float>& values = {},
                     const std::vector<int32>& partial_input_shape_dims = {}) {
    AddTestTensor<float>(name, dims, tf_dtype, values,
                         partial_input_shape_dims);
  }

  // Builds and runs the converted network. Checks output tensor shape. Tests
  // output values using a matcher. The network can have multiple input and
  // output tensors. The inputs are defined by the input_data_ member variable.
  void BuildAndRun(const string& name,
                   const std::vector<std::vector<int>>& expected_output_dims,
                   const Status& expected_runtime_status,
                   const std::vector<Matcher<std::vector<float>>>& matcher) {
    TensorShape shape;
    const int n_output = expected_output_dims.size();
    ASSERT_EQ(n_output, matcher.size());
    DataVec output_data;
    for (int i = 0; i < n_output; i++) {
      TF_EXPECT_OK(
          TensorShapeUtils::MakeShape(expected_output_dims[i], &shape));
      string out_name = (n_output == 1) ? name : StrCat(name, ":", i);
      InputOutputData data{out_name,
                           ConstructTensor(shape.num_elements(), 0, tf_dtype)};
      output_data.push_back(data);
    }
    ASSERT_FALSE(input_data_.empty());
    const int batch_size = input_data_[0].tensor.shape().dim_size(0);
    Status stat =
        OpConverterTest::BuildAndRun(input_data_, &output_data, batch_size);
    ASSERT_EQ(expected_runtime_status, stat);
    if (expected_runtime_status.ok() && stat.ok()) {
      for (int i = 0; i < n_output; i++) {
        // Check the shape of the actual output tensors
        TF_EXPECT_OK(
            TensorShapeUtils::MakeShape(expected_output_dims[i], &shape));
        EXPECT_TRUE(output_data[i].tensor.shape() == shape)
            << "Expected shape: " << shape.DebugString() << ", actual shape"
            << output_data[i].tensor.shape().DebugString();
        EXPECT_THAT(GetDataAsFloat(output_data[i]), matcher[i]);
      }
    }
  }

  // Runs validation and conversion. If conversion is successfull then builds
  // the TRT network, executes it and checks the output.
  void TestOpConverter(const string& name, const NodeDef node_def,
                       const std::vector<int>& expected_output_dims,
                       const Status& expected_conversion_status,
                       const Status& expected_runtime_status,
                       const Matcher<std::vector<float>>& matcher) {
    RunValidationAndConversion(node_def, expected_conversion_status,
                               name.c_str(), expected_output_dims);
    if (expected_conversion_status.ok()) {
      BuildAndRun(name, std::vector<std::vector<int>>({expected_output_dims}),
                  expected_runtime_status,
                  std::vector<Matcher<std::vector<float>>>({matcher}));
    }
  }

 protected:
  const TrtTestMode trt_mode;
  const DataType tf_dtype;
  const TrtPrecisionMode converter_precision;
  DataVec input_data_;
};

// Op converter test in FP32 mode. While for debugging purposes it might make
// sense to run over all possible combinations, normally a subset of them
// would be sufficient:
// - All valid options to TrtTestMode (implicit, explicit, dynamic shape)
// - DataType: is the TF data type of the input tensors. This usually only
//   influences the data type added by Converter::AddInputTensor. We test the
//   valid combinations of input data types in AddAndGetInputs, therefore
//   for most of the OpConverterTest its is sufficient to test for DT_FLOAT.
// - TrtPrecisionMode: valid options are FP32, FP16 and INT8. This influences
//   how TRT handles the precision inside the TRT network, but should not matter
//   for the TF -> TRT conversion. Therefore it should be sufficient to test
//   for FP32.
class OpConverterTest1 : public ParameterizedOpConverterTestBase {};

// Instantiate parameter combinations to OpConverterTest1
INSTANTIATE_TEST_CASE_P(
    OpConvTestInstantiation, OpConverterTest1,
    ::testing::Combine(::testing::ValuesIn(ValidTrtModes),
                       ::testing::Values(DT_FLOAT),
                       ::testing::Values(TrtPrecisionMode::FP32)));

template <typename T>
void CopyTensorElements(const Tensor& tensor, protobuf::RepeatedField<T>* out) {
  out->Clear();
  if (tensor.NumElements() == 0) return;

  // TensorProto does not need to have all the elements present and can truncate
  // trailing elements with the same value for compressed representation. Such
  // elements are derived based on the tensor shape.
  const auto flat = tensor.flat<T>();
  int64 last_index = 0;
  for (int64 i = 0; i < tensor.NumElements(); ++i) {
    if (flat(i) != flat(last_index)) {
      last_index = i;
    }
  }

  int num_out_elements = last_index + 1;
  out->Reserve(num_out_elements);
  out->AddNAlreadyReserved(num_out_elements);
  const T* src = flat.data();
  T* dst = out->mutable_data();
  std::copy(src, src + num_out_elements, dst);
}

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

    TensorProto* tensor_attr =
        (*node_def.mutable_attr())["value"].mutable_tensor();
    tensor_attr->Clear();

    if (as_tensor_content) {
      tensor.AsProtoTensorContent(tensor_attr);
    } else {
      tensor.shape().AsProto(tensor_attr->mutable_tensor_shape());
      tensor_attr->set_dtype(tensor.dtype());

      if (tensor.dtype() == DT_FLOAT) {
        CopyTensorElements<float>(tensor, tensor_attr->mutable_float_val());
      } else if (tensor.dtype() == DT_INT32) {
        CopyTensorElements<int32>(tensor, tensor_attr->mutable_int_val());
      } else {
        tensor.AsProtoField(tensor_attr);
      }
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
    Tensor t(dtype);  // Empty tensor.
    reset_and_test(t, false, {}, {});
  }
  {
    Tensor t = test::AsScalar<InputCType>(12);
    reset_and_test(t, false, {1}, {12});
    reset_and_test(t, true, {1}, {12});
  }
  {
    Tensor t = test->AsTensor<InputCType>({1, 2});
    reset_and_test(t, false, {2}, {1, 2});
    reset_and_test(t, true, {2}, {1, 2});
  }
  {
    Tensor t =
        test->AsTensor<InputCType>({1, 2, 3, 4, 5, 6}, TensorShape({2, 3}));
    reset_and_test(t, false, {2, 3}, {1, 2, 3, 4, 5, 6});
    reset_and_test(t, true, {2, 3}, {1, 2, 3, 4, 5, 6});
  }
  {
    // Set all tensor elements to the same value. Such tensors are encoded
    // using a single element list in tensor proto.
    Tensor t =
        test->AsTensor<InputCType>({1, 1, 1, 1, 1, 1}, TensorShape({2, 3}));
    reset_and_test(t, false, {2, 3}, {1, 1, 1, 1, 1, 1});
    reset_and_test(t, true, {2, 3}, {1, 1, 1, 1, 1, 1});
  }
  {
    // Set trailing tensor elements to the same value. Such tensors are
    // encoded by truncating all equal elements except the first one.
    Tensor t =
        test->AsTensor<InputCType>({2, 2, 1, 1, 1, 1}, TensorShape({2, 3}));
    reset_and_test(t, false, {2, 3}, {2, 2, 1, 1, 1, 1});
    reset_and_test(t, true, {2, 3}, {2, 2, 1, 1, 1, 1});
  }
}

TEST_F(OpConverterTest, ConvertConst) {
  {
    Reset();
    NodeDef node_def = MakeConstNodeDef<double>("my_const", {});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "Unsupported data type double");
  }
  {
    Reset();
    Tensor tensor = AsTensor<int64>({1, std::numeric_limits<int64>::max(), 1, 1,
                                     1, std::numeric_limits<int64>::lowest()},
                                    TensorShape({2, 3}));
    NodeDef node_def;
    node_def.set_name("my_const");
    node_def.set_op("Const");
    (*node_def.mutable_attr())["dtype"].set_type(DT_INT64);
    TensorProto* tensor_attr =
        (*node_def.mutable_attr())["value"].mutable_tensor();
    tensor_attr->Clear();
    tensor.AsProtoTensorContent(tensor_attr);
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "outside the range of int32");
  }

  TestConvertConst<DT_FLOAT, float, float>(this);
  TestConvertConst<DT_INT8, int8, int32>(this);
  TestConvertConst<DT_UINT8, uint8, int32>(this);
  TestConvertConst<DT_INT16, int16, int32>(this);
  TestConvertConst<DT_UINT16, uint16, int32>(this);
  TestConvertConst<DT_INT32, int32, int32>(this);
  TestConvertConst<DT_UINT32, uint32, int32>(this);
  TestConvertConst<DT_INT64, int64, int32>(this);
  TestConvertConst<DT_UINT64, uint64, int32>(this);
}

TEST_P(OpConverterTest1, ConvertTranspose) {
  // Get the NodeDef for Transpose.
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), tf_dtype);
  auto weights = ops::Placeholder(s.WithOpName("weights"), DT_INT32);
  auto transpose = ops::Transpose(s.WithOpName("my_transpose"), input, weights);
  const NodeDef& node_def = transpose.operation.node()->def();

  std::vector<TestParamBase> test_params = {
      // For the first test we leave param empty. This signals to use a
      // input as weight which will be invalid
      TestParamBase{{3, 1, 2, 1},
                    {},
                    {},
                    {},
                    Status(error::UNIMPLEMENTED,
                           "The input \"perm\" for Transpose must be a "
                           "constant, at my_transpose")},
      TestParamBase{{1, 1, 2, 3},
                    {},
                    {},
                    {0, 1, 2},
                    Status(error::INVALID_ARGUMENT,
                           "Rank of perm for transpose does not match with "
                           "that of the input.")},
      // Transpose batch dim
      TestParamBase{
          {1, 1, 2, 3},
          {},
          {3, 2, 1, 1},
          {3, 2, 1, 0},
          (trt_mode == TrtTestMode::kImplicitBatch)
              ? Status(error::UNIMPLEMENTED,
                       "Transpose at batch dimension is not supported")
              : Status::OK()},
      TestParamBase{{1, 1, 2, 3}, {}, {1, 3, 1, 2}, {0, 3, 1, 2}},
  };
  if (trt_mode == TrtTestMode::kDynamicShape) {
    // Dynamic shape tests where some shapes are known
    test_params.push_back(TestParamBase{
        {1, 1, 2, 3}, {-1, 1, 2, -1}, {1, 3, 1, 2}, {0, 3, 1, 2}});
  }
  std::vector<float> expected_values{1, 4, 2, 5, 3, 6};
  for (auto p : test_params) {
    SCOPED_TRACE(p);
    Reset();
    AddTestTensor("input", p.input_dims, {1, 2, 3, 4, 5, 6},
                  p.partial_input_dims);
    if (p.param.empty()) {
      AddTestTensor("weights", {3});
    } else {
      AddTestWeights<int32>("weights", {static_cast<int>(p.param.size())},
                            p.param);
    }
    TestOpConverter("my_transpose", node_def, p.expected_output_dims, p.status,
                    p.runtime_status, ElementsAreArray(expected_values));
  }
}

TEST_F(OpConverterTest, ConvertReshape) {
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
        node_def, error::UNIMPLEMENTED,
        "The input \"shape\" for Reshape must be a constant, at my_reshape");
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
  {
    // Reshape tensor with zero rank to empty tensor, should fail.
    Reset();
    AddTestTensor("input", {});
    AddTestWeights<int32>("weights", {1, 0, 1}, {});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "Reshape to shape=[] is not supported, at my_reshape");
  }

  struct TestParams {
    int batch_size;
    std::vector<int> tensor_dims;
    std::vector<int> shape;
  };

  // Reshape at batch dimension, should fail.
  std::vector<TestParams> params = {
      TestParams{1, {1, 2, 3}, {3, 1, 1, 2}},
      TestParams{1, {1, 2, -1}, {-1, 1, 1, 2}},
      TestParams{1, {1, 2, 3}, {-1, 1, 1, 2}},
      TestParams{-1, {1, 2, 3}, {1, 1, 1, 2}},
      TestParams{-1, {-1, 2, 3}, {1, 1, 1, 6}},  // TODO(laigd): it should pass.
  };
  for (int i = 0; i < params.size(); ++i) {
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
  std::vector<TestParams> ok_params = {
      TestParams{-1, {1, 2, 3}, {-1, 1, 3, 2}},
      TestParams{1, {1, 2, 3}, {-1, 1, 3, 2}},
      TestParams{1, {1, 2, 3}, {1, 1, 3, 2}},
      TestParams{2, {1, 2, 3}, {2, 1, 3, 2}},
      TestParams{1, {1, 1}, {1}},
      TestParams{1, {}, {1, 1}},
      TestParams{2, {1, 1}, {2}},
      TestParams{2, {}, {2, 1}},
  };
  for (int i = 0; i < ok_params.size(); ++i) {
    const int batch_size = std::max(1, ok_params[i].batch_size);
    const auto& shape = ok_params[i].shape;
    Reset();
    AddTestTensor("input", ok_params[i].tensor_dims, batch_size);
    AddTestWeights<int32>("weights", {static_cast<int>(shape.size())}, shape);
    RunValidationAndConversion(node_def);

    TRT_TensorOrWeights output;
    TF_EXPECT_OK(GetTensorOrWeights("my_reshape", &output));
    ASSERT_TRUE(output.is_tensor());
    const std::vector<int> expected_output_dims(shape.begin() + 1, shape.end());
    const nvinfer1::Dims actual_output_dims = output.tensor()->getDimensions();
    ExpectTrtDimsEqualsArray(expected_output_dims, actual_output_dims);

    std::vector<float> input_vec(TrtTensorDimsNumElements(actual_output_dims) *
                                 batch_size);
    std::iota(input_vec.begin(), input_vec.end(), 1);
    const DataVec input_data{{"input", AsTensor<float>(input_vec)}};
    DataVec output_data{
        {"my_reshape", ConstructTensor<float>(input_vec.size())}};
    TF_EXPECT_OK(BuildAndRun(input_data, &output_data, batch_size));
    EXPECT_THAT(GetSpanForData<float>(output_data[0]),
                ElementsAreArray(input_vec));
  }
}

// Helper function for testing MatMul and BatchMatMul
// get_matmul corresponds to the function used to generate the node. It should
// accept (DataType, transpose_a, transpose_b) as parameters.
void TestMatMulHelper(
    OpConverterTest* test,
    const std::function<NodeDef(DataType, bool, bool)>& get_matmul,
    const std::string& op_name) {
  // HACK: This needs to be done in a better way.
  const bool is_batch_matmul = op_name == "BatchMatMul";
  {
    // Unsupported data type.
    test->Reset();
    NodeDef node_def = get_matmul(DT_INT32, false, false);
    test->AddTestTensor("input", {2}, /*batch_size=*/1,
                        nvinfer1::DataType::kINT32);
    test->AddTestWeights<int32>("weights", {2, 1}, {3, 5});
    test->RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        StrCat("Data type int32 is not supported for ", op_name,
               ", must be one of [float, half], at my_matmul")
            .c_str());
  }
  // OK.
  for (bool transpose_a : {false, true}) {
    for (bool transpose_b : {false, true}) {
      test->Reset();
      NodeDef node_def = get_matmul(DT_FLOAT, transpose_a, transpose_b);
      test->AddTestTensor("input", {2}, /*batch_size=*/1);
      test->AddTestWeights<float>("weights", {2, 2}, {0, 1, 2, 3});
      if (is_batch_matmul) {
        test->RunValidationAndConversion(
            node_def, error::UNIMPLEMENTED,
            "TensorRT does not support batched constants.");
        continue;
      } else if (transpose_a) {
        test->RunValidationAndConversion(
            node_def, error::INVALID_ARGUMENT,
            "Cannot transpose first input if it is a tensor with fewer than 2 "
            "non-batch dimensions");
        continue;
      }
      test->RunValidationAndConversion(node_def);
      TRT_TensorOrWeights output;
      TF_EXPECT_OK(test->GetTensorOrWeights("my_matmul", &output));
      ASSERT_TRUE(output.is_tensor());
      ExpectTrtDimsEqualsArray({2}, output.tensor()->getDimensions());

      const DataVec input_data{{"input", test->AsTensor<float>({0, 1})}};
      DataVec output_data{{"my_matmul", test->ConstructTensor<float>(2)}};
      TF_EXPECT_OK(test->BuildAndRun(input_data, &output_data));
      if (transpose_b) {
        EXPECT_THAT(GetSpanForData<float>(output_data[0]), ElementsAre(1, 3));
      } else {
        EXPECT_THAT(GetSpanForData<float>(output_data[0]), ElementsAre(2, 3));
      }
    }
  }
  // OK, 3D inputs
  for (bool transpose_b : {false, true}) {
    test->Reset();
    NodeDef node_def = get_matmul(DT_FLOAT, /*transpose_a=*/false, transpose_b);
    test->AddTestTensor("input", {2}, /*batch_size=*/1);
    test->AddTestWeights<float>("weights", {2, 2}, {0, 1, 2, 3});
    if (is_batch_matmul) {
      test->RunValidationAndConversion(
          node_def, error::UNIMPLEMENTED,
          "TensorRT does not support batched constants.");
      continue;
    }
    test->RunValidationAndConversion(node_def);
    TRT_TensorOrWeights output;
    TF_EXPECT_OK(test->GetTensorOrWeights("my_matmul", &output));
    ASSERT_TRUE(output.is_tensor());
    ExpectTrtDimsEqualsArray({2}, output.tensor()->getDimensions());
    const DataVec input_data{{"input", test->AsTensor<float>({0, 1})}};
    DataVec output_data{{"my_matmul", test->ConstructTensor<float>(2)}};
    TF_EXPECT_OK(test->BuildAndRun(input_data, &output_data));
    if (transpose_b) {
      EXPECT_THAT(GetSpanForData<float>(output_data[0]), ElementsAre(1, 3));
    } else {
      EXPECT_THAT(GetSpanForData<float>(output_data[0]), ElementsAre(2, 3));
    }
  }
}

template <typename LayerType>
void CheckAddedLayers(OpConverterTest* test, bool expect_found) {
  bool layer_found = false;
  for (int i = 0; i < test->converter_->network()->getNbLayers(); i++) {
    nvinfer1::ILayer* layer = test->converter_->network()->getLayer(i);
    if (dynamic_cast<LayerType*>(layer)) {
      layer_found = true;
    }
  }
  EXPECT_EQ(expect_found, layer_found);
}

TEST_F(OpConverterTest, ConvertMatMul) {
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

  // Additional test cases specific to MatMul
  {
    // Can only transpose A if it is 2D in TRT
    Reset();
    NodeDef node_def = get_matmul_nodedef(DT_FLOAT, true, false);
    AddTestTensor("input", {2}, /*batch_size=*/1);
    AddTestWeights<float>("weights", {2, 2}, {0, 1, 2, 3});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Cannot transpose first input if it is a tensor with fewer than 2 "
        "non-batch dimensions.");
  }
  {
    // B must always have 2 non-batch dimensions
    Reset();
    NodeDef node_def = get_matmul_nodedef(DT_FLOAT, false, false);
    AddTestTensor("input", {2}, /*batch_size=*/1);
    AddTestTensor("weights", {2}, /*batch_size=*/1);
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Second input must either be a constant, or contain at least 2 "
        "non-batch dimensions.");
  }
  {
    // We can never transpose weights that are not 2D.
    Reset();
    NodeDef node_def = get_matmul_nodedef(DT_FLOAT, true, false);
    AddTestWeights<float>("input", {1, 1, 2}, {0, 1});
    AddTestTensor("weights", {2, 2}, /*batch_size=*/1);
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Cannot currently transpose constant input if it is not 2 dimensional");
  }
  {
    // Make sure that INT8 mode uses IFullyConnectedLayer when possible.
    Reset(TrtPrecisionMode::INT8);
    NodeDef node_def = get_matmul_nodedef(DT_FLOAT, false, false);
    AddTestTensor("input", {2, 1, 1});
    AddTestWeights<float>("weights", {2, 2}, {0, 1, 2, 3});
    RunValidationAndConversion(node_def);
    CheckAddedLayers<nvinfer1::IMatrixMultiplyLayer>(this, false);
    CheckAddedLayers<nvinfer1::IFullyConnectedLayer>(this, true);
  }
  {
    // Make sure that INT8 mode doesn't try to use IFullyConnectedLayer when not
    // compatible. In this case we can't use FC because weights is a tensor.
    Reset(TrtPrecisionMode::INT8);
    NodeDef node_def = get_matmul_nodedef(DT_FLOAT, false, false);
    AddTestTensor("input", {2, 1, 1});
    AddTestTensor("weights", {2, 2});
    RunValidationAndConversion(node_def);
    CheckAddedLayers<nvinfer1::IMatrixMultiplyLayer>(this, true);
    CheckAddedLayers<nvinfer1::IFullyConnectedLayer>(this, false);
  }
  TestMatMulHelper(this, get_matmul_nodedef, "MatMul");
}

TEST_F(OpConverterTest, ConvertBatchMatMul) {
  // Get the NodeDef for BatchMatMul.
  auto get_batch_matmul_nodedef = [](DataType dtype, bool transpose_a,
                                     bool transpose_b) -> NodeDef {
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), dtype);
    auto weights = ops::Placeholder(s.WithOpName("weights"), dtype);
    const auto matmul_attrs =
        ops::BatchMatMul::AdjX(transpose_a).AdjY(transpose_b);
    auto matmul = ops::BatchMatMul(s.WithOpName("my_matmul"), input, weights,
                                   matmul_attrs);
    return matmul.operation.node()->def();
  };

  {
    // Can't broadcast two tensor inputs of different rank.
    Reset();
    NodeDef node_def = get_batch_matmul_nodedef(DT_FLOAT, false, false);
    AddTestTensor("input", {1, 2, 2}, /*batch_size=*/2);
    AddTestTensor("weights", {2}, /*batch_size=*/2);
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "Inputs must have the same rank if they are both tensors.");
  }
  {
    // Make sure that INT8 mode doesn't try to use IFullyConnectedLayer when not
    // compatible. In this case we can't use FC because transpose_a is true.
    Reset(TrtPrecisionMode::INT8);
    NodeDef node_def = get_batch_matmul_nodedef(DT_FLOAT, true, false);
    AddTestTensor("input", {1, 2, 2});
    AddTestWeights<float>("weights", {2, 2}, {0, 1, 2, 3});
    RunValidationAndConversion(node_def);
    CheckAddedLayers<nvinfer1::IMatrixMultiplyLayer>(this, true);
    CheckAddedLayers<nvinfer1::IFullyConnectedLayer>(this, false);
  }

  for (bool transpose_a : {false, true}) {
    for (bool transpose_b : {false, true}) {
      Reset();
      NodeDef node_def =
          get_batch_matmul_nodedef(DT_FLOAT, transpose_a, transpose_b);
      AddTestTensor("input", {2, 2}, /*batch_size=*/1);
      AddTestWeights<float>("weights", {1, 2, 2}, {1, 2, 3, 4});

      RunValidationAndConversion(node_def);
      TRT_TensorOrWeights output;
      TF_EXPECT_OK(GetTensorOrWeights("my_matmul", &output));
      ASSERT_TRUE(output.is_tensor());
      ExpectTrtDimsEqualsArray({2, 2}, output.tensor()->getDimensions());
      const DataVec input_data{{"input", AsTensor<float>({0, 1, 2, 3})}};
      DataVec output_data{{"my_matmul", ConstructTensor<float>(4)}};
      TF_EXPECT_OK(BuildAndRun(input_data, &output_data));
      if (!transpose_a && !transpose_b) {
        EXPECT_THAT(GetSpanForData<float>(output_data[0]),
                    ElementsAre(3, 4, 11, 16));
      } else if (transpose_a && transpose_b) {
        EXPECT_THAT(GetSpanForData<float>(output_data[0]),
                    ElementsAre(4, 8, 7, 15));
      } else if (transpose_a) {
        EXPECT_THAT(GetSpanForData<float>(output_data[0]),
                    ElementsAre(6, 8, 10, 14));
      } else if (transpose_b) {
        EXPECT_THAT(GetSpanForData<float>(output_data[0]),
                    ElementsAre(2, 4, 8, 18));
      }
    }
  }

  TestMatMulHelper(this, get_batch_matmul_nodedef, "BatchMatMul");
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
      ASSERT_TRUE(output.is_tensor());
      ExpectTrtDimsEqualsArray(dims_array, output.tensor()->getDimensions());

      // Build and run the engine.
      const int num_input = TrtTensorDimsNumElements(GetTestDims(dims_array));
      ASSERT_EQ(trt_input_rank > 1 ? 6 : (data_format == "NHWC" ? 3 : 2),
                num_input);

      const DataVec input_data{
          {"input", test->ConstructTensor<CType>(num_input, CType(0))}};
      DataVec output_data{
          {"my_biasadd", test->ConstructTensor<CType>(num_input)}};
      TF_EXPECT_OK(test->BuildAndRun(input_data, &output_data));
      if (trt_input_rank == 1) {
        if (data_format == "NHWC") {
          EXPECT_THAT(GetSpanForData<CType>(output_data[0]),
                      ElementsAre(CType(1), CType(2), CType(3)));
        } else {
          EXPECT_THAT(GetSpanForData<CType>(output_data[0]),
                      ElementsAre(CType(1), CType(2)));
        }
      } else {
        if (data_format == "NHWC") {
          EXPECT_THAT(GetSpanForData<CType>(output_data[0]),
                      ElementsAre(CType(1), CType(2), CType(3), CType(1),
                                  CType(2), CType(3)));
        } else {
          EXPECT_THAT(GetSpanForData<CType>(output_data[0]),
                      ElementsAre(CType(1), CType(1), CType(1), CType(2),
                                  CType(2), CType(2)));
        }
      }
    }
  }
}

TEST_F(OpConverterTest, ConvertBiasAdd) {
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

template <typename OpType, DataType dtype>
void TestBinaryOp(OpConverterTest* test, bool operand_1_is_tensor,
                  bool operand_2_is_tensor) {
  typedef typename EnumToDataType<dtype>::Type CType;
  test->Reset();
  const NodeDef node_def =
      GetBinaryOpNodeDef<OpType>("input1", "input2", dtype);
  if (operand_1_is_tensor) {
    test->AddTestTensor("input1", /*dims=*/{1, 2}, /*batch_size=*/2,
                        TfDataTypeToTrt(dtype));
  } else {
    test->AddTestWeights("input1", /*dims=*/{1, 2},
                         /*values=*/std::vector<CType>{CType(3), CType(6)});
  }
  if (operand_2_is_tensor) {
    test->AddTestTensor("input2", /*dims=*/{2, 1}, /*batch_size=*/2,
                        TfDataTypeToTrt(dtype));
  } else {
    test->AddTestWeights("input2", /*dims=*/{2, 1},
                         /*values=*/std::vector<CType>{CType(2), CType(3)});
  }
  test->RunValidationAndConversion(node_def);

  DataVec input_data;
  if (operand_1_is_tensor) {
    input_data.push_back(
        {"input1",
         test->AsTensor<CType>({CType(3), CType(6), CType(3), CType(6)})});
  }
  if (operand_2_is_tensor) {
    input_data.push_back(
        {"input2",
         test->AsTensor<CType>({CType(2), CType(3), CType(2), CType(3)})});
  }
  DataVec output_data{{"my_binary", test->ConstructTensor<CType>(8)}};
  // Check output dims.
  TRT_TensorOrWeights output;
  TF_EXPECT_OK(test->GetTensorOrWeights("my_binary", &output));
  ASSERT_TRUE(output.is_tensor());
  ExpectTrtDimsEqualsArray({2, 2}, output.tensor()->getDimensions());
  // After broadcasting first input becomes {3, 6, 3, 6} and second input
  // becomes {2, 3, 2, 3}.
  TF_EXPECT_OK(test->BuildAndRun(input_data, &output_data, /*batch_size=*/2));
  if (node_def.op() == "Add") {
    EXPECT_THAT(
        GetSpanForData<CType>(output_data[0]),
        ElementsAreArray(CastTestVector<int, CType>({5, 8, 6, 9, 5, 8, 6, 9})));
  } else if (node_def.op() == "Sub") {
    EXPECT_THAT(
        GetSpanForData<CType>(output_data[0]),
        ElementsAreArray(CastTestVector<int, CType>({1, 4, 0, 3, 1, 4, 0, 3})));
  } else if (node_def.op() == "Mul") {
    EXPECT_THAT(GetSpanForData<CType>(output_data[0]),
                ElementsAreArray(
                    CastTestVector<int, CType>({6, 12, 9, 18, 6, 12, 9, 18})));
  } else if (node_def.op() == "Div") {
    EXPECT_THAT(GetSpanForData<CType>(output_data[0]),
                ElementsAreArray(CastTestVector<float, CType>(
                    {1.5, 3, 1, 2, 1.5, 3, 1, 2})));
  } else if (node_def.op() == "RealDiv") {
    EXPECT_THAT(GetSpanForData<CType>(output_data[0]),
                ElementsAreArray(CastTestVector<float, CType>(
                    {1.5, 3, 1, 2, 1.5, 3, 1, 2})));
  } else if (node_def.op() == "FloorDiv") {
    EXPECT_THAT(GetSpanForData<CType>(output_data[0]),
                ElementsAreArray(
                    CastTestVector<float, CType>({1, 3, 1, 2, 1, 3, 1, 2})));
  } else if (node_def.op() == "Minimum") {
    EXPECT_THAT(
        GetSpanForData<CType>(output_data[0]),
        ElementsAreArray(CastTestVector<int, CType>({2, 2, 3, 3, 2, 2, 3, 3})));
  } else if (node_def.op() == "Maximum") {
    EXPECT_THAT(
        GetSpanForData<CType>(output_data[0]),
        ElementsAreArray(CastTestVector<int, CType>({3, 6, 3, 6, 3, 6, 3, 6})));
  } else if (node_def.op() == "Pow") {
    ExpectArrayNear(
        CastTestVector<int, CType>({9, 36, 27, 216, 9, 36, 27, 216}),
        GetSpanForData<CType>(output_data[0]));
  } else {
    ASSERT_TRUE(false);
  }
}

TEST_F(OpConverterTest, ConvertBinary) {
  AttrValue dtype;
  dtype.set_type(DT_FLOAT);
  {
    // Both inputs are weights.
    Reset();
    NodeDef node_def =
        MakeNodeDef("my_add", "Add", {"weights1", "weights2"}, {{"T", dtype}});
    AddTestWeights<float>("weights1", {1}, {1});
    AddTestWeights<float>("weights2", {1}, {1});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "Constant folding is falled back to TensorFlow, binary op received "
        "both input as constant at: my_add");
  }

  // Test combinations of tensor vs weight inputs (except when both inputs are
  // weights).
  for (const bool operand_1_is_tensor : {true, false}) {
    for (const bool operand_2_is_tensor : {true, false}) {
      if (!operand_1_is_tensor && !operand_2_is_tensor) continue;
      // FP32 tests
      TestBinaryOp<ops::Add, DT_FLOAT>(this, operand_1_is_tensor,
                                       operand_2_is_tensor);
      TestBinaryOp<ops::Sub, DT_FLOAT>(this, operand_1_is_tensor,
                                       operand_2_is_tensor);
      TestBinaryOp<ops::Mul, DT_FLOAT>(this, operand_1_is_tensor,
                                       operand_2_is_tensor);
      TestBinaryOp<ops::Div, DT_FLOAT>(this, operand_1_is_tensor,
                                       operand_2_is_tensor);
      TestBinaryOp<ops::RealDiv, DT_FLOAT>(this, operand_1_is_tensor,
                                           operand_2_is_tensor);
      TestBinaryOp<ops::Minimum, DT_FLOAT>(this, operand_1_is_tensor,
                                           operand_2_is_tensor);
      TestBinaryOp<ops::Maximum, DT_FLOAT>(this, operand_1_is_tensor,
                                           operand_2_is_tensor);
      TestBinaryOp<ops::Pow, DT_FLOAT>(this, operand_1_is_tensor,
                                       operand_2_is_tensor);
      // FP16 tests
      // TODO(tmorris): Use templates to avoid duplication.
      TestBinaryOp<ops::Add, DT_HALF>(this, operand_1_is_tensor,
                                      operand_2_is_tensor);
      TestBinaryOp<ops::Sub, DT_HALF>(this, operand_1_is_tensor,
                                      operand_2_is_tensor);
      TestBinaryOp<ops::Mul, DT_HALF>(this, operand_1_is_tensor,
                                      operand_2_is_tensor);
      TestBinaryOp<ops::Div, DT_HALF>(this, operand_1_is_tensor,
                                      operand_2_is_tensor);
      TestBinaryOp<ops::RealDiv, DT_HALF>(this, operand_1_is_tensor,
                                          operand_2_is_tensor);
      TestBinaryOp<ops::Minimum, DT_HALF>(this, operand_1_is_tensor,
                                          operand_2_is_tensor);
      TestBinaryOp<ops::Maximum, DT_HALF>(this, operand_1_is_tensor,
                                          operand_2_is_tensor);
      TestBinaryOp<ops::Pow, DT_HALF>(this, operand_1_is_tensor,
                                      operand_2_is_tensor);
    }
  }
}

NodeDef GetAddNNodeDef(const std::vector<string>& input_names, DataType dtype) {
  Scope s = Scope::NewRootScope();
  OutputList inputs;
  for (const string& name : input_names) {
    inputs.push_back(ops::Placeholder(s.WithOpName(name), dtype));
  }
  auto op = ops::AddN(s.WithOpName("my_addn"), inputs);
  return op.operation.node()->def();
}

template <DataType dtype>
void TestAddN(OpConverterTest* test) {
  typedef typename EnumToDataType<dtype>::Type CType;
  {
    // All inputs are tensors.
    test->Reset();
    DataVec input_data;
    for (const auto name : {"inp1", "inp2", "inp3"}) {
      test->AddTestTensor(name, /*dims=*/{1, 2}, /*batch_size=*/2,
                          TfDataTypeToTrt(dtype));
      input_data.push_back({name, test->AsTensor<CType>({CType(1), CType(2),
                                                         CType(3), CType(4)})});
    }
    const NodeDef node_def = GetAddNNodeDef({"inp1", "inp2", "inp3"}, dtype);
    test->RunValidationAndConversion(node_def);

    TRT_TensorOrWeights output;
    TF_EXPECT_OK(test->GetTensorOrWeights("my_addn", &output));
    ASSERT_TRUE(output.is_tensor());
    ExpectTrtDimsEqualsArray({1, 2}, output.tensor()->getDimensions());

    DataVec output_data{{"my_addn", test->ConstructTensor<CType>(4)}};
    TF_EXPECT_OK(test->BuildAndRun(input_data, &output_data, /*batch_size=*/2));
    EXPECT_THAT(GetSpanForData<CType>(output_data[0]),
                ElementsAreArray(CastTestVector<int, CType>({3, 6, 9, 12})));
  }
  {
    // Input contains tensors and weights.
    test->Reset();
    DataVec input_data;
    for (const auto name : {"inp1", "inp2"}) {
      test->AddTestTensor(name, /*dims=*/{1, 2}, /*batch_size=*/1,
                          TfDataTypeToTrt(dtype));
      input_data.push_back({name, test->AsTensor<CType>({CType(1), CType(2)})});
    }
    test->AddTestWeights("inp3", /*dims=*/{1, 1, 2},
                         /*values=*/std::vector<CType>{CType(3), CType(4)});
    const NodeDef node_def = GetAddNNodeDef({"inp1", "inp2", "inp3"}, dtype);
    test->RunValidationAndConversion(node_def);

    TRT_TensorOrWeights output;
    TF_EXPECT_OK(test->GetTensorOrWeights("my_addn", &output));
    ASSERT_TRUE(output.is_tensor());
    ExpectTrtDimsEqualsArray({1, 2}, output.tensor()->getDimensions());

    DataVec output_data{{"my_addn", test->ConstructTensor<CType>(2)}};
    TF_EXPECT_OK(test->BuildAndRun(input_data, &output_data));
    EXPECT_THAT(GetSpanForData<CType>(output_data[0]),
                ElementsAreArray(CastTestVector<int, CType>({5, 8})));
  }
}

TEST_F(OpConverterTest, ConvertAddN) {
  {
    // Weights with batch dim that is not 1.
    Reset();
    const NodeDef node_def = GetAddNNodeDef({"tensor", "weights"}, DT_FLOAT);
    AddTestTensor("tensor", /*dims=*/{1, 2}, /*batch_size=*/2);
    AddTestWeights<float>("weights", {2, 1, 2}, {0, 1, 2, 3});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Weights input to AddN is required to have batch dimension 1.");
  }
  TestAddN<DT_FLOAT>(this);
  TestAddN<DT_HALF>(this);
}

TEST_F(OpConverterTest, ConvertQuantize) {
  {
    // FakeQuantWithMinMaxArgs attributes are empty, should fail.
    Reset(TrtPrecisionMode::INT8);
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
    Reset(TrtPrecisionMode::INT8);
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
    ASSERT_TRUE(output.is_tensor());
    auto ranges = quantization_ranges();
    EXPECT_EQ(1, ranges.count(output.tensor()));
    EXPECT_EQ(6.0f, ranges[output.tensor()]);
  }
  {
    // FakeQuantWithMinMaxVars ranges set via inputs, ok.
    Reset(TrtPrecisionMode::INT8);
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
    ASSERT_TRUE(output.is_tensor());
    auto ranges = quantization_ranges();
    EXPECT_EQ(1, ranges.count(output.tensor()));
    EXPECT_EQ(6.0f, ranges[output.tensor()]);
  }
  {
    // QuantizeAndDequantizeV2 ranges set via inputs, ok.
    Reset(TrtPrecisionMode::INT8);
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
    ASSERT_TRUE(output.is_tensor());
    auto ranges = quantization_ranges();
    EXPECT_EQ(1, ranges.count(output.tensor()));
    EXPECT_EQ(6.0f, ranges[output.tensor()]);
  }
  {
    // QuantizeAndDequantizeV2 Range inputs are tensors, should fail.
    Reset(TrtPrecisionMode::INT8);
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
        node_def, error::UNIMPLEMENTED,
        "The input \"input_min\" for QuantizeAndDequantizeV2 must be a constant"
        ", at my_quantize");
  }
  {
    // QuantizeAndDequantizeV3 ranges set via inputs, ok.
    Reset(TrtPrecisionMode::INT8);
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
    ASSERT_TRUE(output.is_tensor());
    auto ranges = quantization_ranges();
    EXPECT_EQ(1, ranges.count(output.tensor()));
    EXPECT_EQ(6.0f, ranges[output.tensor()]);
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

  test->AddTestTensor("input", {1, 20}, /*batch_size=*/1,
                      TfDataTypeToTrt(dtype));
  test->RunValidationAndConversion(node_def);
  TRT_TensorOrWeights output;
  TF_EXPECT_OK(test->GetTensorOrWeights("my_square", &output));
  ASSERT_TRUE(output.is_tensor());
  ExpectTrtDimsEqualsArray({1, 20}, output.tensor()->getDimensions());

  const int num_inputs = 20;
  std::vector<CType> inputs(num_inputs);
  std::vector<CType> expected_outputs(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    const CType value = CType(i - 9);
    inputs[i] = value;
    expected_outputs[i] = value * value;
  }
  const DataVec input_data{{"input", test->AsTensor<CType>(inputs)}};
  // Engine outputs are converted to FP16 automatically if we set FP16 mode in
  // the builder.
  DataVec output_data{{"my_square", test->ConstructTensor<CType>(num_inputs)}};
  TF_EXPECT_OK(test->BuildAndRun(input_data, &output_data));
  ExpectArrayNear(expected_outputs, GetSpanForData<CType>(output_data[0]));
}

TEST_F(OpConverterTest, ConvertSquare) {
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
        "The input \"x\" for Square must be a tensor, at my_square");
  }

  // OK. Note that kINT32 is not supported by IElementWiseLayer, so we don't
  // test DT_INT32 type here.
  TestConvertSquare<DT_FLOAT>(this);
  TestConvertSquare<DT_HALF>(this);
}

#if IS_TRT_VERSION_GE(5, 1, 0, 0)
TEST_F(OpConverterTest, ConvertCombinedNMS) {
  // Get the NodeDef for CombinedNMS.
  auto get_nms_nodedef = []() -> NodeDef {
    Scope s = Scope::NewRootScope();
    auto boxes_tensor = ops::Placeholder(s.WithOpName("boxes"), DT_FLOAT);
    auto scores_tensor = ops::Placeholder(s.WithOpName("scores"), DT_FLOAT);
    auto max_output_size_per_class =
        ops::Placeholder(s.WithOpName("max_output_size_per_class"), DT_INT32);
    auto max_total_size =
        ops::Placeholder(s.WithOpName("max_total_size"), DT_INT32);
    auto iou_threshold =
        ops::Placeholder(s.WithOpName("iou_threshold"), DT_FLOAT);
    auto score_threshold =
        ops::Placeholder(s.WithOpName("score_threshold"), DT_FLOAT);
    auto nms_attrs = ops::CombinedNonMaxSuppression::Attrs().PadPerClass(false);

    auto nms_op = ops::CombinedNonMaxSuppression(
        s.WithOpName("my_nms"), boxes_tensor, scores_tensor,
        max_output_size_per_class, max_total_size, iou_threshold,
        score_threshold, nms_attrs);
    return nms_op.operation.node()->def();
  };

  struct TestParams {
    const std::vector<int32> boxes_tensor_dims;
    const std::vector<int32> scores_tensor_dims;
    const int32 max_output_size_per_class;
    const int32 max_total_size;
    const float iou_threshold;
    const float score_threshold;
    const std::vector<int32> expected_nmsed_boxes_dims;
    const std::vector<int32> expected_nmsed_scores_dims;
    const std::vector<int32> expected_nmsed_classes_dims;
  };

  // Ok.
  std::vector<TestParams> ok_params = {
      // TODO(aaroey): there is a bug in TRT's CombinedNonMaxSuppression
      // implementation that, the extra output classes that are outside of the
      // range specified by valid_detections[i] are not zeros but -1s.
      TestParams{{1, 1, 4}, {1, 3}, 3, 2, .5f, 0, {2, 4}, {2}, {2}}};

  for (int i = 0; i < ok_params.size(); ++i) {
    Reset();

    AddTestTensor("boxes", ok_params[i].boxes_tensor_dims);
    AddTestTensor("scores", ok_params[i].scores_tensor_dims);
    AddTestWeights<int32>("max_output_size_per_class", {1},
                          {ok_params[i].max_output_size_per_class});
    AddTestWeights<int32>("max_total_size", {1}, {ok_params[i].max_total_size});
    AddTestWeights<float>("iou_threshold", {1}, {ok_params[i].iou_threshold});
    AddTestWeights<float>("score_threshold", {1},
                          {ok_params[i].score_threshold});

    RunValidationAndConversion(get_nms_nodedef());

    TRT_TensorOrWeights nmsed_boxes;
    TRT_TensorOrWeights nmsed_scores;
    TRT_TensorOrWeights nmsed_classes;
    TRT_TensorOrWeights valid_detections;

    TF_EXPECT_OK(GetTensorOrWeights("my_nms", &nmsed_boxes));
    TF_EXPECT_OK(GetTensorOrWeights("my_nms:1", &nmsed_scores));
    TF_EXPECT_OK(GetTensorOrWeights("my_nms:2", &nmsed_classes));
    TF_EXPECT_OK(GetTensorOrWeights("my_nms:3", &valid_detections));

    ASSERT_TRUE(nmsed_boxes.is_tensor());
    ASSERT_TRUE(nmsed_scores.is_tensor());
    ASSERT_TRUE(nmsed_classes.is_tensor());
    ASSERT_TRUE(valid_detections.is_tensor());

    ExpectTrtDimsEqualsArray(ok_params[i].expected_nmsed_boxes_dims,
                             nmsed_boxes.tensor()->getDimensions());
    ExpectTrtDimsEqualsArray(ok_params[i].expected_nmsed_scores_dims,
                             nmsed_scores.tensor()->getDimensions());
    ExpectTrtDimsEqualsArray(ok_params[i].expected_nmsed_classes_dims,
                             nmsed_classes.tensor()->getDimensions());
    ExpectTrtDimsEqualsArray({}, valid_detections.tensor()->getDimensions());

    DataVec output_data{
        {"my_nms", ConstructTensor<float>(8)},
        {"my_nms:1", ConstructTensor<float>(2)},
        {"my_nms:2", ConstructTensor<float>(2)},
        {"my_nms:3", ConstructTensor<int32>(1)},
    };
    const DataVec input_data{{"boxes", AsTensor<float>({0, 0, 0.3, 0.4})},
                             {"scores", AsTensor<float>({0.4, 0.7, 0.3})}};
    TF_EXPECT_OK(BuildAndRun(input_data, &output_data));
    EXPECT_THAT(GetSpanForData<float>(output_data[0]),
                ElementsAre(0, 0, 0.3, 0.4, 0, 0, 0.3, 0.4));
    EXPECT_THAT(GetSpanForData<float>(output_data[1]), ElementsAre(0.7, 0.4));
    EXPECT_THAT(GetSpanForData<float>(output_data[2]), ElementsAre(1, 0));
    EXPECT_THAT(GetSpanForData<int32>(output_data[3]), ElementsAre(2));
  }
}
#endif  // IS_TRT_VERSION_GE(5, 1, 0, 0)

template <typename T>
NodeDef CreateUnaryOp(DataType tf_dtype) {
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), tf_dtype);
  return T(s.WithOpName("my_unary"), input).operation.node()->def();
}

constexpr float kLeakyReluAlpha = 0.2f;
template <>
NodeDef CreateUnaryOp<ops::internal::LeakyRelu>(DataType tf_dtype) {
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), tf_dtype);
  return ops::internal::LeakyRelu(
             s.WithOpName("my_unary"), input,
             ops::internal::LeakyRelu::Alpha(kLeakyReluAlpha))
      .operation.node()
      ->def();
}

TEST_P(OpConverterTest1, ConvertActivation) {
  {
    // Input is weights, should fail.
    Reset();
    const NodeDef& node_def = CreateUnaryOp<ops::Relu>(tf_dtype);
    AddTestWeights<int32>("input", {1, 2, 3}, {-3, -2, -1, 0, 1, 2});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"input\" for Relu must be a tensor, at my_unary");
  }

  constexpr float kSeluAlpha = 1.7580993408473768599402175208123f;
  constexpr float kSeluScale = 1.0507009873554804934193349852946f;
  using OpFunc = std::function<NodeDef(DataType)>;
  using ValFunc = float (*)(float);
  std::map<std::string, std::pair<OpFunc, ValFunc>> op_map;

#define ADD_OP(name, op, compute) \
  op_map[name] = std::make_pair(CreateUnaryOp<op>, compute)
  ADD_OP("LeakyRelu", ops::internal::LeakyRelu,
         [](float x) { return (x > 0.0f) ? x : x * kLeakyReluAlpha; });
  ADD_OP("Relu", ops::Relu, [](float x) { return (x > 0.0f) ? x : 0.0f; });
  ADD_OP("Relu6", ops::Relu6,
         [](float x) { return std::min(std::max(x, 0.0f), 6.0f); });
  ADD_OP("Sigmoid", ops::Sigmoid,
         [](float x) { return 1.0f / (1.0f + std::exp(-x)); });
  ADD_OP("Tanh", ops::Tanh, static_cast<ValFunc>(std::tanh));
  ADD_OP("Elu", ops::Elu,
         [](float x) { return (x > 0.0f) ? x : std::exp(x) - 1; });
  ADD_OP("Selu", ops::Selu, [](float x) {
    return (x > 0.0f) ? kSeluScale * x
                      : kSeluScale * kSeluAlpha * (std::exp(x) - 1);
  });
  ADD_OP("Softsign", ops::Softsign,
         [](float x) { return x / (std::abs(x) + 1); });
  ADD_OP("Softplus", ops::Softplus,
         [](float x) { return std::log(std::exp(x) + 1); });
#undef ADD_OP

  // Get list of ops to test.
  std::vector<string> ops_to_test;
  // Add all ops supported by ConvertActivation.
  auto* map = ActivationTypeMap();
  ops_to_test.reserve(map->size());
  for (auto& pair : *map) {
    ops_to_test.push_back(pair.first);
  }
  // Add other activation ops to test.
  ops_to_test.push_back("Relu6");
  ops_to_test.push_back("LeakyRelu");
  auto p = TestParamBase{
      {1, 1, 2, 3},  // input dims
      {},            // input partial dims
      {1, 1, 2, 3},  // expected output dims
  };
  // Ok.
  for (const string& op_name : ops_to_test) {
    if (!op_map.count(op_name)) {
      FAIL() << "Activation op test map does not contain op " << op_name;
    }
    Reset();
    NodeDef node_def = op_map[op_name].first(tf_dtype);
    const std::vector<float> input = {-100, -2, -1, 0, 1, 88};
    AddTestTensor("input", p.input_dims, input);

    // std::exp in Softplus will overflow for input > 88
    std::vector<float> output_values;
    std::transform(input.begin(), input.end(),
                   std::back_inserter(output_values), op_map[op_name].second);
    TestOpConverter("my_unary", node_def, p.expected_output_dims, Status::OK(),
                    Status::OK(), ArrayFloatNear(output_values, 0, false));

    TRT_TensorOrWeights output;
    TF_EXPECT_OK(GetTensorOrWeights("my_unary", &output));

    // Certain activations should set quantization range automatically.
    auto ranges = quantization_ranges();
    if (op_name == "Relu6") {
      EXPECT_EQ(ranges[output.tensor()], 6.0f);
    } else if (op_name == "Sigmoid" || op_name == "Tanh" ||
               op_name == "Softsign") {
      EXPECT_EQ(ranges[output.tensor()], 1.0f);
    }
  }
}

TEST_F(OpConverterTest, ConvertExpandDims) {
  // Get the NodeDef for ExpandDims.
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT);
  auto weights = ops::Placeholder(s.WithOpName("weights"), DT_INT32);
  auto expanddims =
      ops::ExpandDims(s.WithOpName("my_expanddims"), input, weights);
  const NodeDef& node_def = expanddims.operation.node()->def();
  {
    // Input is weights, should fail.
    Reset();
    AddTestWeights<int32>("input", {1, 2, 3}, {1, 2, 3, 4, 5, 6});
    AddTestWeights<int32>("weights", {1}, {1});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "The input \"input\" for ExpandDims must be a "
                               "tensor, at my_expanddims");
  }
  {
    // Axis is a tensor, should fail.
    Reset();
    AddTestTensor("input", {1, 2, 3});
    AddTestTensor("weights", {3});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "The input \"axis\" for ExpandDims must be a "
                               "constant, at my_expanddims");
  }
  {
    // Add dim at batch dimension, should fail.
    Reset();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<int32>("weights", {1}, {0});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "TensorRT does not allow manipulation of the batch dimension, at "
        "my_expanddims");
  }
  {
    // Add dim at batch dimension via negative axis, should fail.
    Reset();
    AddTestTensor("input", {1, 2, 3});
    // Input is rank 4 (batch dim included)
    AddTestWeights<int32>("weights", {1}, {-5});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "TensorRT does not allow manipulation of the batch dimension, at "
        "my_expanddims");
  }
  {
    // Axis > rank(input), should fail.
    Reset();
    AddTestTensor("input", {1, 2, 3});
    // Input is rank 4 (batch dim included)
    AddTestWeights<int32>("weights", {1}, {5});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Axis value of 5 is out of bounds, must be in range [-5, 5), at "
        "my_expanddims");
  }
  {
    // Axis < -rank(input)-1, should fail.
    Reset();
    AddTestTensor("input", {1, 2, 3});
    // Input is rank 4 (batch dim included)
    AddTestWeights<int32>("weights", {1}, {-6});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Axis value of -6 is out of bounds, must be in range [-5, 5), at "
        "my_expanddims");
  }

  struct TestParams {
    std::vector<int> input_dims;
    int axis;
    std::vector<int> expected_output_dims;
  };

  // Ok.
  std::vector<TestParams> ok_params = {
      TestParams{{2, 3}, 1, {1, 2, 3}}, TestParams{{2, 3}, -3, {1, 2, 3}},
      TestParams{{2, 3}, 3, {2, 3, 1}}, TestParams{{2, 3}, -1, {2, 3, 1}},
      TestParams{{2, 3}, 2, {2, 1, 3}}, TestParams{{2, 3}, -2, {2, 1, 3}},
      TestParams{{6}, 1, {1, 6}},       TestParams{{6}, -1, {6, 1}},
  };
  for (int i = 0; i < ok_params.size(); ++i) {
    Reset();
    AddTestTensor("input", ok_params[i].input_dims);
    AddTestWeights<int32>("weights", {1}, {ok_params[i].axis});
    RunValidationAndConversion(node_def);
    TRT_TensorOrWeights output;
    TF_EXPECT_OK(GetTensorOrWeights("my_expanddims", &output));
    ASSERT_TRUE(output.is_tensor());
    ExpectTrtDimsEqualsArray(ok_params[i].expected_output_dims,
                             output.tensor()->getDimensions());

    const DataVec input_data{{"input", AsTensor<float>({1, 2, 3, 4, 5, 6})}};
    DataVec output_data{{"my_expanddims", ConstructTensor<float>(6)}};
    TF_EXPECT_OK(BuildAndRun(input_data, &output_data));
    EXPECT_THAT(GetSpanForData<float>(output_data[0]),
                ElementsAre(1, 2, 3, 4, 5, 6));
  }
}

TEST_P(OpConverterTest1, ConvertSqueeze) {
  const bool use_implicit_batch = (trt_mode == TrtTestMode::kImplicitBatch);
  // Get the NodeDef for Squeeze.
  auto get_squeeze_nodedef = [](std::vector<int> axes,
                                DataType tf_dtype) -> NodeDef {
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), tf_dtype);
    if (!axes.empty()) {
      ops::Squeeze::Attrs squeeze_attrs;
      squeeze_attrs.axis_ = gtl::ArraySlice<int>(axes);  // non-absl ok
      auto squeeze =
          ops::Squeeze(s.WithOpName("my_squeeze"), input, squeeze_attrs);
      return squeeze.operation.node()->def();
    } else {
      auto squeeze = ops::Squeeze(s.WithOpName("my_squeeze"), input);
      return squeeze.operation.node()->def();
    }
  };
  std::vector<TestParamBase> test_params = {
      TestParamBase{
          {1, 2, 1, 3},  // input dims
          {},            // input partial dims
          {2, 3},        // expected output dims
          {},            // axis
          trt_mode == TrtTestMode::kExplicitBatch
              ? Status::OK()
              : Status{error::UNIMPLEMENTED,
                       "Squeeze is not implemented for empty squeeze_dims, at "
                       "my_squeeze"}},
      TestParamBase{{1, 2, 1, 3},
                    {},
                    {2, 1, 3},
                    {0},
                    use_implicit_batch
                        ? Status{error::UNIMPLEMENTED,
                                 "TensorRT does not allow manipulation of the "
                                 "batch dimension, at my_squeeze"}
                        : Status::OK()},
      TestParamBase{{1, 2, 1, 3},
                    {},
                    {2, 1, 3},
                    {-4},
                    use_implicit_batch
                        ? Status{error::UNIMPLEMENTED,
                                 "TensorRT does not allow manipulation of the "
                                 "batch dimension, at my_squeeze"}
                        : Status::OK()},
      TestParamBase{
          {1, 1, 2, 3},
          {},
          {},
          {4},
          Status{error::INVALID_ARGUMENT,
                 "Axis value of 4 is out of bounds, must be in range [-4, 4), "
                 "at my_squeeze"}},
      TestParamBase{
          {1, 1, 2, 3},
          {},
          {},
          {-5},
          Status{error::INVALID_ARGUMENT,
                 "Axis value of -5 is out of bounds, must be in range [-4, 4), "
                 "at my_squeeze"}},
      TestParamBase{{1, 1, 2, 3}, {}, {1, 2, 3}, {1}},
      TestParamBase{{1, 1, 2, 3}, {}, {1, 2, 3}, {-3}},
      TestParamBase{{1, 2, 3, 1}, {}, {1, 2, 3}, {3}},
      TestParamBase{{1, 2, 3, 1}, {}, {1, 2, 3}, {-1}},
      TestParamBase{{1, 1, 2, 1, 3, 1}, {}, {1, 2, 3}, {1, 3, 5}},
      TestParamBase{{1, 1, 2, 1, 3, 1}, {}, {1, 2, 3}, {3, 1, 5}},
      TestParamBase{{1, 1, 2, 1, 3, 1}, {}, {1, 2, 3}, {-1, -3, -5}},
      TestParamBase{{1, 1, 2, 1, 3, 1}, {}, {1, 2, 3}, {1, -3, 5}},
      TestParamBase{{1, 1, 6}, {}, {1, 6}, {1}},
      TestParamBase{{1, 6, 1}, {}, {1, 6}, {2}},
  };
  auto squeeze_non_singleton = TestParamBase{
      {1, 1, 2, 3},
      {},
      {},
      {2},
      Status{error::INVALID_ARGUMENT,
             "Dimension 2 with size 2 cannot be squeezed because it must be "
             "size 1, at my_squeeze"}};

  if (trt_mode == TrtTestMode::kDynamicShape) {
    // In this test we try to squeeze axis=2 which has size > 1. In dynamic
    // shape mode the converter sees only -1, so it cannot catch this error.
    squeeze_non_singleton.status = Status::OK();  // conversion status
    squeeze_non_singleton.runtime_status =
        errors::InvalidArgument("Negative number of dimensions -1");
    // Dynamic shape tests with partially known input shape
    test_params.push_back(TestParamBase{{2, 1, 3}, {2, -1, 3}, {2, 3}, {1}});
    test_params.push_back(TestParamBase{{2, 1, 3}, {2, 1, -1}, {2, 3}, {1}});
  }
  test_params.push_back(squeeze_non_singleton);

  for (TestParamBase p : test_params) {
    SCOPED_TRACE(p);
    Reset();
    NodeDef node_def = get_squeeze_nodedef(p.param, tf_dtype);
    AddTestTensor("input", p.input_dims, {1, 2, 3, 4, 5, 6},
                  p.partial_input_dims);
    TestOpConverter("my_squeeze", node_def, p.expected_output_dims, p.status,
                    p.runtime_status, ElementsAreArray({1, 2, 3, 4, 5, 6}));
  }
}

TEST_F(OpConverterTest, ConvertStridedSlice) {
  // Get nodedef for StridedSlice layer.
  auto get_strided_slice_nodedef =
      [](int64 begin_mask = 0, int64 end_mask = 0, int64 ellipsis_mask = 0,
         int64 new_axis_mask = 0, int64 shrink_axis_mask = 0) -> NodeDef {
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT);
    auto begin = ops::Placeholder(s.WithOpName("begin"), DT_INT32);
    auto end = ops::Placeholder(s.WithOpName("end"), DT_INT32);
    auto strides = ops::Placeholder(s.WithOpName("strides"), DT_INT32);
    ops::StridedSlice::Attrs attrs = ops::StridedSlice::Attrs()
                                         .BeginMask(begin_mask)
                                         .EndMask(end_mask)
                                         .EllipsisMask(ellipsis_mask)
                                         .NewAxisMask(new_axis_mask)
                                         .ShrinkAxisMask(shrink_axis_mask);
    auto strided_slice = ops::StridedSlice(s.WithOpName("my_strided_slice"),
                                           input, begin, end, strides, attrs);
    return strided_slice.operation.node()->def();
  };

  {
    // Input is weights, should fail.
    Reset();
    NodeDef node_def = get_strided_slice_nodedef();
    AddTestWeights<int32>("input", {1, 2, 3}, {1, 2, 3, 4, 5, 6});
    AddTestWeights<int32>("begin", {4}, {0, 0, 0, 0});
    AddTestWeights<int32>("end", {4}, {1, 1, 2, 3});
    AddTestWeights<int32>("strides", {4}, {1, 1, 1, 1});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "The input \"input\" for StridedSlice must be a "
                               "tensor, at my_strided_slice");
  }
  {
    // Begin, end, strides are tensors, should fail.
    Reset();
    NodeDef node_def = get_strided_slice_nodedef();
    AddTestTensor("input", {1, 2, 3});
    AddTestTensor("begin", {4});
    AddTestTensor("end", {4});
    AddTestTensor("strides", {4});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"begin\" for StridedSlice must be a constant, at "
        "my_strided_slice");
  }
  {
    // Modify batch dim, should fail.
    Reset();
    NodeDef node_def = get_strided_slice_nodedef();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<int32>("begin", {4}, {0, 0, 0, 0});
    AddTestWeights<int32>("end", {4}, {0, 1, 2, 3});
    AddTestWeights<int32>("strides", {4}, {1, 1, 1, 1});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "TensorRT does not allow modifications to the batch dimension, at "
        "my_strided_slice");
  }
  {
    // Dynamic batch size without end_mask, should fail.
    Reset();
    NodeDef node_def = get_strided_slice_nodedef();
    AddTestTensor("input", {1, 2, 3}, /*batch_size=*/-1);
    AddTestWeights<int32>("begin", {4}, {0, 0, 0, 0});
    AddTestWeights<int32>("end", {4}, {1, 1, 2, 3});
    AddTestWeights<int32>("strides", {4}, {1, 1, 1, 1});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "TensorRT does not allow modifications to the batch dimension, at "
        "my_strided_slice");
  }
  {
    // Dynamic batch size but using end_mask, ok.
    Reset();
    NodeDef node_def = get_strided_slice_nodedef(/*begin_mask=*/0,
                                                 /*end_mask=*/1);
    AddTestTensor("input", {1, 2, 3}, /*batch_size=*/-1);
    AddTestWeights<int32>("begin", {4}, {0, 0, 0, 0});
    AddTestWeights<int32>("end", {4}, {0, 1, 2, 2});
    AddTestWeights<int32>("strides", {4}, {1, 1, 1, 1});
    RunValidationAndConversion(node_def);
  }
// TRT 5.1+ supports strides (disabled until 5.1.3.1 due to bugs)
#if IS_TRT_VERSION_GE(5, 1, 3, 1)
  {
    // Negative strides, should fail.
    Reset();
    NodeDef node_def = get_strided_slice_nodedef();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<int32>("begin", {4}, {0, 0, 0, 0});
    AddTestWeights<int32>("end", {4}, {1, 1, 2, 3});
    AddTestWeights<int32>("strides", {4}, {1, 1, 1, -1});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "Negative or zero stride values are not "
                               "supported for StridedSlice, at "
                               "my_strided_slice");
  }
#else
  {
    // Stride is not 1, should fail.
    Reset();
    NodeDef node_def = get_strided_slice_nodedef();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<int32>("begin", {4}, {0, 0, 0, 0});
    AddTestWeights<int32>("end", {4}, {1, 1, 2, 3});
    AddTestWeights<int32>("strides", {4}, {1, 2, 1, 3});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "Strides other than 1 are not supported with "
                               "this version of TRT, at my_strided_slice");
  }
#endif
  {
    // Size of sliced dim is negative, should fail.
    Reset();
    NodeDef node_def = get_strided_slice_nodedef();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<int32>("begin", {4}, {0, 0, 2, 0});
    AddTestWeights<int32>("end", {4}, {1, 1, 0, 3});
    AddTestWeights<int32>("strides", {4}, {1, 1, 1, 1});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "\"size\" cannot be negative or zero for "
                               "StridedSlice, at my_strided_slice");
  }

  struct TestParams {
    std::vector<int> input_dims;
    std::vector<int> begin;
    std::vector<int> end;
    std::vector<int> strides;
    int begin_mask;
    int end_mask;
    int ellipsis_mask;
    int new_axis_mask;
    int shrink_axis_mask;
    std::vector<int> expected_output_dims;
    std::vector<float> expected_output;
  };

  auto get_mask = [](const std::vector<int>& mask) {
    int result = 0;
    for (int i = 0; i < mask.size(); i++) {
      if (mask[i]) result += (1 << i);
    }
    return result;
  };

  // Same input is used for all tests.
  const std::vector<float> ok_input = {1, 2, 3, 4, 5, 6};

  // Ok.
  std::vector<TestParams> ok_params = {
    // 2D Crop.
    TestParams{
        /*input_dims=*/{1, 2, 3},
        /*begin=*/{0, 0, 0, 0},
        /*end=*/{0, 0, 1, 2},
        /*strides=*/{1, 1, 1, 1},
        /*begin_mask=*/get_mask({0, 0, 0, 0}),
        /*end_mask=*/get_mask({1, 1, 0, 0}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{1, 1, 2},
        /*expected_output=*/{1, 2},
    },
    TestParams{
        /*input_dims=*/{1, 2, 3},
        /*begin=*/{0, 0, 1, 1},
        /*end=*/{0, 0, 0, 0},
        /*strides=*/{1, 1, 1, 1},
        /*begin_mask=*/get_mask({0, 0, 0, 0}),
        /*end_mask=*/get_mask({1, 1, 1, 1}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{1, 1, 2},
        /*expected_output=*/{5, 6},
    },
    TestParams{
        /*input_dims=*/{1, 2, 3},
        /*begin=*/{0, 0, 1, 1},
        /*end=*/{0, 1, 2, 3},
        /*strides=*/{1, 1, 1, 1},
        /*begin_mask=*/get_mask({0, 0, 0, 0}),
        /*end_mask=*/get_mask({1, 1, 0, 0}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{1, 1, 2},
        /*expected_output=*/{5, 6},
    },
    // 2D Crop, with transpose.
    TestParams{
        /*input_dims=*/{2, 3, 1},
        /*begin=*/{0, 0, 0, 0},
        /*end=*/{0, 1, 2, 1},
        /*strides=*/{1, 1, 1, 1},
        /*begin_mask=*/get_mask({0, 0, 0, 0}),
        /*end_mask=*/get_mask({1, 0, 0, 0}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{1, 2, 1},
        /*expected_output=*/{1, 2},
    },
    TestParams{
        /*input_dims=*/{2, 3, 1},
        /*begin=*/{0, 1, 1, 0},
        /*end=*/{0, 2, 3, 1},
        /*strides=*/{1, 1, 1, 1},
        /*begin_mask=*/get_mask({0, 0, 0, 0}),
        /*end_mask=*/get_mask({1, 0, 0, 0}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{1, 2, 1},
        /*expected_output=*/{5, 6},
    },
    TestParams{
        /*input_dims=*/{2, 1, 3},
        /*begin=*/{0, 0, 0, 0},
        /*end=*/{0, 1, 1, 2},
        /*strides=*/{1, 1, 1, 1},
        /*begin_mask=*/get_mask({0, 0, 0, 0}),
        /*end_mask=*/get_mask({1, 0, 0, 0}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{1, 1, 2},
        /*expected_output=*/{1, 2},
    },
    TestParams{
        /*input_dims=*/{2, 1, 3},
        /*begin=*/{0, 1, 0, 1},
        /*end=*/{0, 2, 1, 3},
        /*strides=*/{1, 1, 1, 1},
        /*begin_mask=*/get_mask({0, 0, 0, 0}),
        /*end_mask=*/get_mask({1, 0, 0, 0}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{1, 1, 2},
        /*expected_output=*/{5, 6},
    },
    // 2D Crop, with reshape.
    TestParams{
        /*input_dims=*/{2, 3},
        /*begin=*/{0, 0, 0},
        /*end=*/{0, 1, 2},
        /*strides=*/{1, 1, 1},
        /*begin_mask=*/get_mask({0, 0, 0}),
        /*end_mask=*/get_mask({1, 0, 0}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{1, 2},
        /*expected_output=*/{1, 2},
    },
    TestParams{
        /*input_dims=*/{2, 3},
        /*begin=*/{0, 1, 1},
        /*end=*/{0, 0, 0},
        /*strides=*/{1, 1, 1},
        /*begin_mask=*/get_mask({0, 0, 0}),
        /*end_mask=*/get_mask({1, 1, 1}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{1, 2},
        /*expected_output=*/{5, 6},
    },
    // 1D Crop.
    TestParams{
        /*input_dims=*/{1, 2, 3},
        /*begin=*/{0, 0, 0, 0},
        /*end=*/{0, 0, 0, 2},
        /*strides=*/{1, 1, 1, 1},
        /*begin_mask=*/get_mask({0, 0, 0, 0}),
        /*end_mask=*/get_mask({1, 1, 1, 0}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{1, 2, 2},
        /*expected_output=*/{1, 2, 4, 5},
    },
    TestParams{
        /*input_dims=*/{1, 2, 3},
        /*begin=*/{0, 0, 1, 0},
        /*end=*/{0, 0, 0, 0},
        /*strides=*/{1, 1, 1, 1},
        /*begin_mask=*/get_mask({0, 0, 0, 0}),
        /*end_mask=*/get_mask({1, 1, 1, 1}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{1, 1, 3},
        /*expected_output=*/{4, 5, 6},
    },
    // 1D Crop, with transpose.
    TestParams{
        /*input_dims=*/{2, 3, 1},
        /*begin=*/{0, 0, 0, 0},
        /*end=*/{0, 1, 0, 0},
        /*strides=*/{1, 1, 1, 1},
        /*begin_mask=*/get_mask({0, 0, 0, 0}),
        /*end_mask=*/get_mask({1, 0, 1, 1}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{1, 3, 1},
        /*expected_output=*/{1, 2, 3},
    },
    TestParams{
        /*input_dims=*/{2, 3, 1},
        /*begin=*/{0, 1, 0, 0},
        /*end=*/{0, 0, 0, 0},
        /*strides=*/{1, 1, 1, 1},
        /*begin_mask=*/get_mask({0, 0, 0, 0}),
        /*end_mask=*/get_mask({1, 1, 1, 1}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{1, 3, 1},
        /*expected_output=*/{4, 5, 6},
    },
    // 1D Crop, with reshape.
    TestParams{
        /*input_dims=*/{6},
        /*begin=*/{0, 0},
        /*end=*/{0, 3},
        /*strides=*/{1, 1},
        /*begin_mask=*/get_mask({0, 0}),
        /*end_mask=*/get_mask({1, 0}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{3},
        /*expected_output=*/{1, 2, 3},
    },
    TestParams{
        /*input_dims=*/{1, 6},
        /*begin=*/{0, 0, 2},
        /*end=*/{0, 0, 5},
        /*strides=*/{1, 1, 1},
        /*begin_mask=*/get_mask({0, 0, 0}),
        /*end_mask=*/get_mask({1, 1, 0}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{1, 3},
        /*expected_output=*/{3, 4, 5},
    },
    TestParams{
        /*input_dims=*/{6, 1},
        /*begin=*/{0, 2, 0},
        /*end=*/{0, 5, 0},
        /*strides=*/{1, 1, 1},
        /*begin_mask=*/get_mask({0, 0, 0}),
        /*end_mask=*/get_mask({1, 0, 1}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{3, 1},
        /*expected_output=*/{3, 4, 5},
    },
    // Negative axis.
    TestParams{
        /*input_dims=*/{6, 1},
        /*begin=*/{0, -6, 0},
        /*end=*/{0, -3, 0},
        /*strides=*/{1, 1, 1},
        /*begin_mask=*/get_mask({0, 0, 0}),
        /*end_mask=*/get_mask({1, 0, 1}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{3, 1},
        /*expected_output=*/{1, 2, 3},
    },
    TestParams{
        /*input_dims=*/{6, 1},
        /*begin=*/{0, 0, 0},
        /*end=*/{0, -1, 0},
        /*strides=*/{1, 1, 1},
        /*begin_mask=*/get_mask({0, 0, 0}),
        /*end_mask=*/get_mask({1, 0, 1}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{5, 1},
        /*expected_output=*/{1, 2, 3, 4, 5},
    },
    // Clamp out of bounds begin and end.
    TestParams{
        /*input_dims=*/{1, 2, 3},
        /*begin=*/{0, 0, -9999, -9},
        /*end=*/{0, 1, 1000, 4},
        /*strides=*/{1, 1, 1, 1},
        /*begin_mask=*/get_mask({0, 0, 0, 0}),
        /*end_mask=*/get_mask({1, 0, 0, 0}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{1, 2, 3},
        /*expected_output=*/{1, 2, 3, 4, 5, 6},
    },
#if IS_TRT_VERSION_GE(5, 1, 3, 1)
    // Strides
    TestParams{
        /*input_dims=*/{6},
        /*begin=*/{0, 0},
        /*end=*/{0, 5},
        /*strides=*/{1, 2},
        /*begin_mask=*/get_mask({0, 0}),
        /*end_mask=*/get_mask({1, 0}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{3},
        /*expected_output=*/{1, 3, 5},
    },
    TestParams{
        /*input_dims=*/{6},
        /*begin=*/{0, 0},
        /*end=*/{0, 6},
        /*strides=*/{1, 2},
        /*begin_mask=*/get_mask({0, 0}),
        /*end_mask=*/get_mask({1, 0}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{3},
        /*expected_output=*/{1, 3, 5},
    },
    TestParams{
        /*input_dims=*/{6},
        /*begin=*/{0, 1},
        /*end=*/{0, 6},
        /*strides=*/{1, 2},
        /*begin_mask=*/get_mask({0, 0}),
        /*end_mask=*/get_mask({1, 0}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{3},
        /*expected_output=*/{2, 4, 6},
    },
    TestParams{
        /*input_dims=*/{6},
        /*begin=*/{0, 2},
        /*end=*/{0, 6},
        /*strides=*/{1, 3},
        /*begin_mask=*/get_mask({0, 0}),
        /*end_mask=*/get_mask({1, 0}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{2},
        /*expected_output=*/{3, 6},
    },
#endif
    // ellipsis_mask
    TestParams{
        /*input_dims=*/{1, 2, 3},
        /*begin=*/{0, 1},
        /*end=*/{0, 2},
        /*strides=*/{1, 1},
        /*begin_mask=*/get_mask({0, 0, 0, 0}),
        /*end_mask=*/get_mask({0, 0, 0, 0}),
        /*ellipsis_mask=*/get_mask({1, 0, 0, 0}),
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{1, 2, 1},
        /*expected_output=*/{2, 5},
    },
    TestParams{
        /*input_dims=*/{1, 2, 3},
        /*begin=*/{0, 0, 1},
        /*end=*/{0, 0, 2},
        /*strides=*/{1, 1, 1},
        /*begin_mask=*/get_mask({1, 0, 0, 0}),
        /*end_mask=*/get_mask({1, 0, 0, 0}),
        /*ellipsis_mask=*/get_mask({0, 1, 0, 0}),
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{1, 2, 1},
        /*expected_output=*/{2, 5},
    },
    TestParams{
        /*input_dims=*/{1, 2, 3},
        /*begin=*/{0, 0, 0, 1},
        /*end=*/{0, 1, 2, 2},
        /*strides=*/{1, 1, 1, 1},
        /*begin_mask=*/get_mask({0, 0, 0, 0}),
        /*end_mask=*/get_mask({0, 0, 0, 0}),
        /*ellipsis_mask=*/get_mask({1, 0, 0, 0}),
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{1, 2, 1},
        /*expected_output=*/{2, 5},
    },
    TestParams{
        /*input_dims=*/{1, 2, 3},
        /*begin=*/{0, 0, 0, 1},
        /*end=*/{1, 1, 2, 2},
        /*strides=*/{1, 1, 1, 1},
        /*begin_mask=*/get_mask({0, 0, 0, 0}),
        /*end_mask=*/get_mask({0, 0, 0, 0}),
        /*ellipsis_mask=*/get_mask({0, 1, 0, 0}),
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{1, 2, 1},
        /*expected_output=*/{2, 5},
    },
#if IS_TRT_VERSION_GE(5, 1, 3, 1)
    TestParams{
        /*input_dims=*/{1, 2, 3},
        /*begin=*/{0, 0, 0, 0, 1},
        /*end=*/{0, 1, 1, 2, 2},
        /*strides=*/{1, 1, 1, 1, 1},
        /*begin_mask=*/get_mask({0, 0, 0, 0}),
        /*end_mask=*/get_mask({0, 0, 0, 0}),
        /*ellipsis_mask=*/get_mask({1, 0, 0, 0}),
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/0,
        /*expected_output_dims=*/{1, 2, 1},
        /*expected_output=*/{2, 5},
    },
    // shrink_axis_mask
    TestParams{
        /*input_dims=*/{1, 2, 3},
        /*begin=*/{0, 0, 0, 1},
        /*end=*/{0, 0, 0, 2},
        /*strides=*/{1, 1, 1, 1},
        /*begin_mask=*/get_mask({1, 1, 1, 0}),
        /*end_mask=*/get_mask({1, 1, 1, 0}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/get_mask({0, 0, 0, 1}),
        /*expected_output_dims=*/{1, 2},
        /*expected_output=*/{2, 5},
    },
    TestParams{
        /*input_dims=*/{1, 2, 3},
        /*begin=*/{0, 0, 0, 1},
        /*end=*/{0, 1, 2, 2},
        /*strides=*/{1, 1, 1, 1},
        /*begin_mask=*/get_mask({1, 0, 0, 0}),
        /*end_mask=*/get_mask({1, 0, 0, 0}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/get_mask({0, 1, 0, 1}),
        /*expected_output_dims=*/{2},
        /*expected_output=*/{2, 5},
    },
    TestParams{
        /*input_dims=*/{6},
        /*begin=*/{0, 0},
        /*end=*/{0, 1},
        /*strides=*/{1, 1},
        /*begin_mask=*/get_mask({1, 0}),
        /*end_mask=*/get_mask({1, 0}),
        /*ellipsis_mask=*/0,
        /*new_axis_mask=*/0,
        /*shrink_axis_mask=*/get_mask({0, 1}),
        /*expected_output_dims=*/{},
        /*expected_output=*/{1},
    },
#endif  // IS_TRT_VERSION_GE(5, 1, 3, 1)
  };

  for (int i = 0; i < ok_params.size(); i++) {
    Reset();
    NodeDef node_def = get_strided_slice_nodedef(
        ok_params[i].begin_mask, ok_params[i].end_mask,
        ok_params[i].ellipsis_mask, ok_params[i].new_axis_mask,
        ok_params[i].shrink_axis_mask);
    AddTestTensor("input", ok_params[i].input_dims);
    AddTestWeights<int32>("begin",
                          {static_cast<int>(ok_params[i].begin.size())},
                          ok_params[i].begin);
    AddTestWeights<int32>("end", {static_cast<int>(ok_params[i].end.size())},
                          ok_params[i].end);
    AddTestWeights<int32>("strides",
                          {static_cast<int>(ok_params[i].strides.size())},
                          ok_params[i].strides);
    RunValidationAndConversion(node_def);

    TRT_TensorOrWeights output;
    TF_EXPECT_OK(GetTensorOrWeights("my_strided_slice", &output));
    ASSERT_TRUE(output.is_tensor());
    ExpectTrtDimsEqualsArray(ok_params[i].expected_output_dims,
                             output.tensor()->getDimensions());

    const DataVec input_data{{"input", AsTensor<float>(ok_input)}};
    DataVec output_data{
        {"my_strided_slice",
         ConstructTensor<float>(ok_params[i].expected_output.size())}};
    TF_EXPECT_OK(BuildAndRun(input_data, &output_data));
    EXPECT_THAT(GetSpanForData<float>(output_data[0]),
                ElementsAreArray(ok_params[i].expected_output));
  }
}

TEST_F(OpConverterTest, ConvertSlice) {
  // Get nodedef for Slice layer.
  auto get_slice_nodedef = []() -> NodeDef {
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT);
    auto begin = ops::Placeholder(s.WithOpName("begin"), DT_INT32);
    auto size = ops::Placeholder(s.WithOpName("size"), DT_INT32);
    auto slice = ops::Slice(s.WithOpName("my_slice"), input, begin, size);
    return slice.operation.node()->def();
  };

  {
    // Begin is below bounds, should fail.
    Reset();
    NodeDef node_def = get_slice_nodedef();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<int32>("begin", {4}, {0, 0, -1, 0});
    AddTestWeights<int32>("size", {4}, {1, 1, 2, 3});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "\"begin\" for dimension 2 in Slice is out of range, at my_slice");
  }
  {
    // Begin is above bounds, should fail.
    Reset();
    NodeDef node_def = get_slice_nodedef();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<int32>("begin", {4}, {0, 0, 3, 0});
    AddTestWeights<int32>("size", {4}, {1, 1, 2, 3});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "\"begin\" for dimension 2 in Slice is out of range, at my_slice");
  }
  {
    // Size is below bounds, should fail.
    Reset();
    NodeDef node_def = get_slice_nodedef();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<int32>("begin", {4}, {0, 0, 0, 0});
    AddTestWeights<int32>("size", {4}, {1, 1, 2, -2});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "\"begin\" + \"size\" for dimension 3 in Slice is out of range, at "
        "my_slice");
  }
  {
    // Size is above bounds, should fail.
    Reset();
    NodeDef node_def = get_slice_nodedef();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<int32>("begin", {4}, {0, 0, 0, 0});
    AddTestWeights<int32>("size", {4}, {1, 1, 3, 3});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "\"begin\" + \"size\" for dimension 2 in Slice is out of range, at "
        "my_slice");
  }
  {
    // Modify batch dim, should fail.
    Reset();
    NodeDef node_def = get_slice_nodedef();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<int32>("begin", {4}, {0, 0, 0, 0});
    AddTestWeights<int32>("size", {4}, {0, 1, 2, 3});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "TensorRT does not allow modifications to the batch dimension, at "
        "my_slice");
  }
  {
    // Dynamic batch size with size[0] not -1, should fail.
    Reset();
    NodeDef node_def = get_slice_nodedef();
    AddTestTensor("input", {1, 2, 3}, /*batch_size=*/-1);
    AddTestWeights<int32>("begin", {4}, {0, 0, 0, 0});
    AddTestWeights<int32>("size", {4}, {1, 1, 2, 3});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "TensorRT does not allow modifications to the batch dimension, at "
        "my_slice");
  }
  {
    // Dynamic batch size but using size[0] of -1, ok.
    Reset();
    NodeDef node_def = get_slice_nodedef();
    AddTestTensor("input", {1, 2, 3}, /*batch_size=*/-1);
    AddTestWeights<int32>("begin", {4}, {0, 0, 0, 0});
    AddTestWeights<int32>("size", {4}, {-1, 1, 2, 2});
    RunValidationAndConversion(node_def);
  }

  struct TestParams {
    std::vector<int> input_dims;
    std::vector<int> begin;
    std::vector<int> size;
    std::vector<int> expected_output_dims;
    std::vector<int> expected_output;
  };

  // Ok.
  std::vector<TestParams> ok_params = {
      TestParams{{1, 2, 3},
                 {0, 0, 0, 0},
                 {-1, -1, -1, -1},
                 {1, 2, 3},
                 {1, 2, 3, 4, 5, 6}},
      TestParams{
          {1, 2, 3}, {0, 0, 0, 0}, {1, 1, 2, 3}, {1, 2, 3}, {1, 2, 3, 4, 5, 6}},
      TestParams{
          {1, 2, 3}, {0, 0, 0, 0}, {1, -1, 2, 2}, {1, 2, 2}, {1, 2, 4, 5}},
      TestParams{{6}, {0, 1}, {1, 5}, {5}, {2, 3, 4, 5, 6}},
      TestParams{{6}, {0, 1}, {-1, 3}, {3}, {2, 3, 4}},
  };

  for (int i = 0; i < ok_params.size(); i++) {
    Reset();
    NodeDef node_def = get_slice_nodedef();
    AddTestTensor("input", ok_params[i].input_dims);
    AddTestWeights<int32>("begin",
                          {static_cast<int>(ok_params[i].begin.size())},
                          ok_params[i].begin);
    AddTestWeights<int32>("size", {static_cast<int>(ok_params[i].size.size())},
                          ok_params[i].size);
    RunValidationAndConversion(node_def);

    TRT_TensorOrWeights output;
    TF_EXPECT_OK(GetTensorOrWeights("my_slice", &output));
    ASSERT_TRUE(output.is_tensor());
    ExpectTrtDimsEqualsArray(ok_params[i].expected_output_dims,
                             output.tensor()->getDimensions());

    const DataVec input_data{{"input", AsTensor<float>({1, 2, 3, 4, 5, 6})}};
    DataVec output_data{{"my_slice", ConstructTensor<float>(
                                         ok_params[i].expected_output.size())}};
    TF_EXPECT_OK(BuildAndRun(input_data, &output_data));
    EXPECT_THAT(GetSpanForData<float>(output_data[0]),
                ElementsAreArray(ok_params[i].expected_output));
  }
}

TEST_P(OpConverterTest1, ConvertConv2D) {
  // Get nodedef for Conv2D layer.
  DataType tf_type = tf_dtype;
  auto get_conv2d_nodedef =
      [tf_type](std::vector<int> strides = {1, 1, 1, 1},
                string padding = "SAME", string data_format = "NCHW",
                std::vector<int> dilations = {1, 1, 1, 1}) -> NodeDef {
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), tf_type);
    auto filter = ops::Placeholder(s.WithOpName("weights"), tf_type);
    ops::Conv2D::Attrs attrs =
        ops::Conv2D::Attrs().DataFormat(data_format).Dilations(dilations);
    auto conv2d = ops::Conv2D(s.WithOpName("my_conv2d"), input, filter, strides,
                              padding, attrs);
    return conv2d.operation.node()->def();
  };

  {
    // Input is weights, should fail.
    Reset();
    NodeDef node_def = get_conv2d_nodedef();
    AddTestWeights<float>("input", {1, 2, 3}, {1, 2, 3, 4, 5, 6});
    AddTestWeights<float>("weights", {3, 3, 1, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"input\" for Conv2D must be a tensor, at my_conv2d");
  }
  {
    // Filter is tensor, should fail.
    Reset();
    NodeDef node_def = get_conv2d_nodedef();
    AddTestTensor("input", {3, 1, 2, 1});
    AddTestTensor("weights", {3, 3, 1, 1});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"filter\" for Conv2D must be a constant, at my_conv2d");
  }
  {
    // Filter is not 4D, should fail.
    Reset();
    NodeDef node_def = get_conv2d_nodedef();
    AddTestTensor("input", {1, 1, 2, 3});
    AddTestWeights<float>("weights", {3, 3, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Conv2D expects kernel of dimension 4, at my_conv2d");
  }
  {
    // Dilations is not 4D, should fail.
    Reset();
    NodeDef node_def =
        get_conv2d_nodedef({1, 1, 1, 1}, "SAME", "NCHW", {1, 1, 1});
    AddTestTensor("input", {1, 1, 2, 3});
    AddTestWeights<float>("weights", {3, 3, 1, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Convolution dilations field must specify 4 dimensions, at my_conv2d");
  }
  {
    // Dilation value is not 1 for channel, should fail.
    Reset();
    NodeDef node_def =
        get_conv2d_nodedef({1, 1, 1, 1}, "SAME", "NCHW", {1, 2, 1, 1});
    AddTestTensor("input", {1, 1, 2, 3});
    AddTestWeights<float>("weights", {3, 3, 1, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "Dilation rate must be 1 for batch and channel "
                               "dimensions, at my_conv2d");
  }
  {
    // Dilation value is not 1 for channel (NHWC), should fail.
    Reset();
    NodeDef node_def =
        get_conv2d_nodedef({1, 1, 1, 1}, "SAME", "NHWC", {1, 1, 1, 2});
    AddTestTensor("input", {1, 2, 3, 1});
    AddTestWeights<float>("weights", {3, 3, 1, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "Dilation rate must be 1 for batch and channel "
                               "dimensions, at my_conv2d");
  }
  {
    // Strides is not 4D, should fail.
    Reset();
    NodeDef node_def =
        get_conv2d_nodedef({1, 1, 1}, "SAME", "NCHW", {1, 1, 1, 1});
    AddTestTensor("input", {1, 1, 2, 3});
    AddTestWeights<float>("weights", {3, 3, 1, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Convolution strides field must specify 4 dimensions, at my_conv2d");
  }
  {
    // Stride value is not 1 for channel, should fail.
    Reset();
    NodeDef node_def =
        get_conv2d_nodedef({1, 2, 1, 1}, "SAME", "NCHW", {1, 1, 1, 1});
    AddTestTensor("input", {1, 1, 2, 3});
    AddTestWeights<float>("weights", {3, 3, 1, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "Stride must be 1 for batch and channel dimensions, at my_conv2d");
  }
  if (trt_mode == TrtTestMode::kDynamicShape) {
    Reset();
    NodeDef node_def = get_conv2d_nodedef();
    // Channel dim unknown, should fail.
    AddTestTensorWithTFDims("input", {-1, -1, -1, -1},
                            TfDataTypeToTrt(tf_type));
    AddTestWeights<float>("weights", {1, 2, 1, 1}, {-1, 1});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Channel dimension must be static, at my_conv2d");
  }

  struct TestParams {
    std::vector<int> input_dims;
    std::vector<float> input;
    std::vector<int> filter_dims;
    std::vector<float> filter;
    std::vector<int> strides;
    string padding;
    string data_format;
    std::vector<int> dilations;
    std::vector<int> expected_output_dims;
    std::vector<float> expected_output;
  };

  // Ok.
  std::vector<TestParams> ok_params = {
      // Basic
      TestParams{/*input_dims=*/{1, 1, 2, 3},
                 /*input=*/{0, 1, 2, 3, 3, 4},
                 /*filter_dims=*/{1, 2, 1, 1},
                 /*filter=*/{-1, 1},
                 /*strides=*/{1, 1, 1, 1},
                 /*padding=*/"VALID",
                 /*data_format=*/"NCHW",
                 /*dilations=*/{1, 1, 1, 1},
                 /*expected_output_dims=*/{1, 1, 2, 2},
                 /*expected_output=*/{1, 1, 0, 1}},
      // SAME padding (Asymmetric)
      TestParams{/*input_dims=*/{1, 1, 2, 3},
                 /*input=*/{0, 1, 2, 3, 3, 4},
                 /*filter_dims=*/{1, 2, 1, 1},
                 /*filter=*/{-1, 1},
                 /*strides=*/{1, 1, 1, 1},
                 /*padding=*/"SAME",
                 /*data_format=*/"NCHW",
                 /*dilations=*/{1, 1, 1, 1},
                 /*expected_output_dims=*/{1, 1, 2, 3},
                 /*expected_output=*/{1, 1, -2, 0, 1, -4}},
      // SAME padding (Symmetric)
      TestParams{/*input_dims=*/{1, 1, 2, 3},
                 /*input=*/{0, 1, 2, 3, 3, 4},
                 /*filter_dims=*/{1, 3, 1, 1},
                 /*filter=*/{-1, 0, 1},
                 /*strides=*/{1, 1, 1, 1},
                 /*padding=*/"SAME",
                 /*data_format=*/"NCHW",
                 /*dilations=*/{1, 1, 1, 1},
                 /*expected_output_dims=*/{1, 1, 2, 3},
                 /*expected_output=*/{1, 2, -1, 3, 1, -3}},
      // NHWC
      TestParams{/*input_dims=*/{1, 2, 3, 1},
                 /*input=*/{0, 1, 2, 3, 3, 4},
                 /*filter_dims=*/{1, 2, 1, 1},
                 /*filter=*/{-1, 1},
                 /*strides=*/{1, 1, 1, 1},
                 /*padding=*/"VALID",
                 /*data_format=*/"NHWC",
                 /*dilations=*/{1, 1, 1, 1},
                 /*expected_output_dims=*/{1, 2, 2, 1},
                 /*expected_output=*/{1, 1, 0, 1}},
      // Dilated
      TestParams{/*input_dims=*/{1, 1, 2, 3},
                 /*input=*/{0, 1, 2, 3, 3, 4},
                 /*filter_dims=*/{1, 2, 1, 1},
                 /*filter=*/{-1, 1},
                 /*strides=*/{1, 1, 1, 1},
                 /*padding=*/"VALID",
                 /*data_format=*/"NCHW",
                 /*dilations=*/{1, 1, 1, 2},
                 /*expected_output_dims=*/{1, 1, 2, 1},
                 /*expected_output=*/{2, 1}},
      // Strided
      TestParams{/*input_dims=*/{1, 1, 2, 4},
                 /*input=*/{0, 1, 2, 2, 3, 4, 4, 7},
                 /*filter_dims=*/{1, 2, 1, 1},
                 /*filter=*/{-1, 1},
                 /*strides=*/{1, 1, 1, 2},
                 /*padding=*/"VALID",
                 /*data_format=*/"NCHW",
                 /*dilations=*/{1, 1, 1, 1},
                 /*expected_output_dims=*/{1, 1, 2, 2},
                 /*expected_output=*/{1, 0, 1, 3}},
  };

  for (int i = 0; i < ok_params.size(); i++) {
    Reset();
    NodeDef node_def =
        get_conv2d_nodedef(ok_params[i].strides, ok_params[i].padding,
                           ok_params[i].data_format, ok_params[i].dilations);
    std::vector<int> partial_input_shape;
    if (trt_mode == TrtTestMode::kDynamicShape) {
      // The channel dim cannot have unknown size, fix that.
      partial_input_shape.resize(ok_params[i].input_dims.size(), -1);
      int channel_id = (ok_params[i].data_format == "NCHW") ? 1 : 3;
      partial_input_shape[channel_id] = ok_params[i].input_dims[channel_id];
    }

    AddTestTensor("input", ok_params[i].input_dims, tf_dtype,
                  ok_params[i].input, partial_input_shape);
    AddTestWeights<float>("weights", ok_params[i].filter_dims,
                          ok_params[i].filter);

    TestOpConverter("my_conv2d", node_def, ok_params[i].expected_output_dims,
                    Status::OK(), Status::OK(),
                    ElementsAreArray(ok_params[i].expected_output));
  }
}

TEST_F(OpConverterTest, ConvertConv2DBackpropInput) {
  // Get nodedef for Conv2D layer.
  auto get_conv2d_backprop_input_nodedef =
      [](std::vector<int> strides = {1, 1, 1, 1}, string padding = "SAME",
         string data_format = "NCHW",
         std::vector<int> dilations = {1, 1, 1, 1}) -> NodeDef {
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT);
    auto filter = ops::Placeholder(s.WithOpName("weights"), DT_FLOAT);
    auto input_sizes = ops::Placeholder(s.WithOpName("input_sizes"), DT_INT32);
    ops::Conv2DBackpropInput::Attrs attrs = ops::Conv2DBackpropInput::Attrs()
                                                .DataFormat(data_format)
                                                .Dilations(dilations);
    auto conv2d = ops::Conv2DBackpropInput(
        s.WithOpName("my_conv2d_backprop_input"), input_sizes, filter, input,
        strides, padding, attrs);
    return conv2d.operation.node()->def();
  };

  {
    // Dilation + Conv2DBackpropInput, should fail.
    Reset();
    NodeDef node_def = get_conv2d_backprop_input_nodedef({1, 1, 1, 1}, "SAME",
                                                         "NHWC", {1, 1, 2, 1});
    AddTestTensor("input", {2, 3, 1});
    AddTestWeights<float>("weights", {3, 3, 1, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    AddTestWeights<int>("input_sizes", {4}, {1, 2, 3, 1});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "Dilation with Conv2DBackpropInput "
                               "(conv2d_transpose) is not supported, "
                               "at my_conv2d_backprop_input");
  }

  struct TestParams {
    std::vector<int> input_dims;
    std::vector<float> input;
    std::vector<int> filter_dims;
    std::vector<float> filter;
    std::vector<int> strides;
    string padding;
    string data_format;
    std::vector<int> dilations;
    std::vector<int> expected_output_dims;
    std::vector<float> expected_output;
  };

  // Ok.
  std::vector<TestParams> ok_params = {
      // Transpose Strided
      TestParams{/*input_dims=*/{1, 2, 2},
                 /*input=*/{0, 1, 2, 3},
                 /*filter_dims=*/{1, 2, 1, 1},
                 /*filter=*/{-1, 1},
                 /*strides=*/{1, 1, 1, 2},
                 /*padding=*/"SAME",
                 /*data_format=*/"NCHW",
                 /*dilations=*/{1, 1, 1, 1},
                 /*expected_output_dims=*/{1, 2, 4},
                 /*expected_output=*/{0, 0, -1, 1, -2, 2, -3, 3}},
      // Transpose Strided NHWC
      TestParams{/*input_dims=*/{2, 2, 1},
                 /*input=*/{0, 1, 2, 3},
                 /*filter_dims=*/{1, 2, 1, 1},
                 /*filter=*/{-1, 1},
                 /*strides=*/{1, 1, 2, 1},
                 /*padding=*/"SAME",
                 /*data_format=*/"NHWC",
                 /*dilations=*/{1, 1, 1, 1},
                 /*expected_output_dims=*/{2, 4, 1},
                 /*expected_output=*/{0, 0, -1, 1, -2, 2, -3, 3}},
      // Transpose Strided NHWC with VALID padding
      TestParams{/*input_dims=*/{3, 1, 1},
                 /*input=*/{0, 1, 2},
                 /*filter_dims=*/{2, 1, 1, 1},
                 /*filter=*/{-1, 1},
                 /*strides=*/{1, 2, 1, 1},
                 /*padding=*/"VALID",
                 /*data_format=*/"NHWC",
                 /*dilations=*/{1, 1, 1, 1},
                 /*expected_output_dims=*/{7, 1, 1},
                 /*expected_output=*/{0, 0, -1, 1, -2, 2, 0}},
  };

  for (int i = 0; i < ok_params.size(); i++) {
    for (int input_sizes_length : {2, 4}) {
      Reset();
      NodeDef node_def = get_conv2d_backprop_input_nodedef(
          ok_params[i].strides, ok_params[i].padding, ok_params[i].data_format,
          ok_params[i].dilations);
      AddTestTensor("input", ok_params[i].input_dims);
      AddTestWeights<float>("weights", ok_params[i].filter_dims,
                            ok_params[i].filter);

      std::vector<int> tf_input_sizes = ok_params[i].expected_output_dims;
      if (input_sizes_length == 4) {
        tf_input_sizes.insert(tf_input_sizes.begin(),
                              1);  // Add batch dimension.
        QCHECK_EQ(4, tf_input_sizes.size());
        AddTestWeights<int>("input_sizes", {4}, tf_input_sizes);
      } else {
        // Remove the channel dimension.
        if (ok_params[i].data_format == "NHWC") {
          tf_input_sizes.pop_back();
        } else {
          tf_input_sizes.erase(tf_input_sizes.begin());
        }
        QCHECK_EQ(2, tf_input_sizes.size());
        AddTestWeights<int>("input_sizes", {2}, tf_input_sizes);
      }

      RunValidationAndConversion(node_def);
      TRT_TensorOrWeights output;
      TF_EXPECT_OK(GetTensorOrWeights("my_conv2d_backprop_input", &output));
      ASSERT_TRUE(output.is_tensor());
      ExpectTrtDimsEqualsArray(ok_params[i].expected_output_dims,
                               output.tensor()->getDimensions());

      const DataVec input_data{{"input", AsTensor<float>(ok_params[i].input)}};
      DataVec output_data{
          {"my_conv2d_backprop_input",
           ConstructTensor<float>(ok_params[i].expected_output.size())}};
      TF_EXPECT_OK(BuildAndRun(input_data, &output_data));
      EXPECT_THAT(GetSpanForData<float>(output_data[0]),
                  ElementsAreArray(ok_params[i].expected_output));
    }
  }
}

#if IS_TRT_VERSION_GE(6, 0, 0, 0)
TEST_F(OpConverterTest, ConvertConv3D) {
  // Get nodedef for Conv3D layer.
  auto get_conv3d_nodedef =
      [](std::vector<int> strides = {1, 1, 1, 1, 1}, string padding = "SAME",
         string data_format = "NCDHW",
         std::vector<int> dilations = {1, 1, 1, 1, 1},
         bool is_conv3d_backprop_input = false) -> NodeDef {
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT);
    auto filter = ops::Placeholder(s.WithOpName("weights"), DT_FLOAT);

    if (is_conv3d_backprop_input) {
      auto input_sizes =
          ops::Placeholder(s.WithOpName("input_sizes"), DT_INT32);
      ops::Conv3DBackpropInputV2::Attrs attrs =
          ops::Conv3DBackpropInputV2::Attrs()
              .DataFormat(data_format)
              .Dilations(dilations);
      auto conv3d =
          ops::Conv3DBackpropInputV2(s.WithOpName("my_conv3d"), input_sizes,
                                     filter, input, strides, padding, attrs);
      return conv3d.operation.node()->def();
    } else {
      ops::Conv3D::Attrs attrs =
          ops::Conv3D::Attrs().DataFormat(data_format).Dilations(dilations);
      auto conv3d = ops::Conv3D(s.WithOpName("my_conv3d"), input, filter,
                                strides, padding, attrs);
      return conv3d.operation.node()->def();
    }
  };

  {
    // Input is weights, should fail.
    Reset();
    NodeDef node_def = get_conv3d_nodedef();

    AddTestWeights<float>("input", {1, 2, 3}, {1, 2, 3, 4, 5, 6});
    AddTestWeights<float>("weights", {3, 3, 1, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"input\" for Conv3D must be a tensor, at my_conv3d");
  }
  {
    // Filter is tensor, should fail.
    Reset();
    NodeDef node_def = get_conv3d_nodedef();
    AddTestTensor("input", {1, 2, 3});
    AddTestTensor("weights", {3, 3, 1, 1, 3, 3, 1, 1});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"filter\" for Conv3D must be a constant, at my_conv3d");
  }
  {
    // Filter is not 5D, should fail.
    Reset();
    NodeDef node_def = get_conv3d_nodedef();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<float>("weights", {3, 3, 1, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Conv3D expects kernel of dimension 5, at my_conv3d");
  }
  {
    // Dilations is not 5D, should fail.
    Reset();
    NodeDef node_def =
        get_conv3d_nodedef({1, 1, 1, 1, 1}, "SAME", "NCDHW", {1, 1, 1, 1});
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<float>(
        "weights", {3, 3, 1, 1, 1},
        {1, 2, 3, 4, 5, 6, 7, 8, 9});  // Dimensions, then values
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Convolution dilations field must specify 5 dimensions, at my_conv3d");
  }
  {
    // Dilation value is not 1 for channel, should fail.
    Reset();
    NodeDef node_def =
        get_conv3d_nodedef({1, 1, 1, 1, 1}, "SAME", "NCDHW", {1, 2, 1, 1, 1});
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<float>("weights", {3, 3, 1, 1, 1},
                          {1, 2, 3, 4, 5, 6, 7, 8, 9});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "Dilation rate must be 1 for batch and channel "
                               "dimensions, at my_conv3d");
  }
  {
    // Dilation value is not 1 for channel (NDHWC), should fail.
    Reset();
    NodeDef node_def =
        get_conv3d_nodedef({1, 1, 1, 1, 1}, "SAME", "NDHWC", {1, 1, 1, 1, 2});
    AddTestTensor("input", {2, 3, 1});
    AddTestWeights<float>("weights", {3, 3, 1, 1, 1},
                          {1, 2, 3, 4, 5, 6, 7, 8, 9});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "Dilation rate must be 1 for batch and channel "
                               "dimensions, at my_conv3d");
  }
  {
    // Dilation + Conv3DBackpropInputV2, should fail.
    Reset();
    NodeDef node_def = get_conv3d_nodedef({1, 1, 1, 1, 1}, "SAME", "NDHWC",
                                          {1, 1, 2, 1, 1}, true);
    AddTestTensor("input", {2, 3, 1});
    AddTestWeights<float>("weights", {3, 3, 1, 1, 1},
                          {1, 2, 3, 4, 5, 6, 7, 8, 9});
    AddTestWeights<int>("input_sizes", {4}, {1, 2, 3, 1});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "Dilation with Conv3DBackpropInputV2 "
                               "(conv3d_transpose) is not supported, "
                               "at my_conv3d");
  }
  {
    // Asymmetric+ Conv3DBackpropInputV2, should fail.
    Reset();
    NodeDef node_def = get_conv3d_nodedef({1, 1, 1, 1, 1}, "SAME", "NDHWC",
                                          {1, 1, 1, 1, 1}, true);
    AddTestTensor("input", {1, 2, 2, 2});
    AddTestWeights<float>("weights", {1, 1, 2, 1, 1}, {1, 1});
    AddTestWeights<int>("input_sizes", {8}, {1, 2, 3, 4, 5, 6, 7, 8});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "Asymmetric padding with Conv3DBackpropInputV2 "
                               "(conv3d_transpose) is not supported, at "
                               "my_conv3d");
  }
  {
    // Strides is not 5D, should fail.
    Reset();
    NodeDef node_def = get_conv3d_nodedef({1, 1, 1, 1, 1, 1}, "SAME", "NCDHW",
                                          {1, 1, 1, 1, 1});
    AddTestTensor("input", {1, 2, 2, 2});
    AddTestWeights<float>("weights", {1, 1, 2, 1, 1}, {1, 1});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Convolution strides field must specify 5 dimensions, at my_conv3d");
  }
  {
    // Stride value is not 1 for channel, should fail.
    Reset();
    NodeDef node_def =
        get_conv3d_nodedef({1, 2, 1, 1, 1}, "SAME", "NCDHW", {1, 1, 1, 1, 1});
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<float>("weights", {3, 3, 1, 1, 1},
                          {1, 2, 3, 4, 5, 6, 7, 8, 9});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "Stride must be 1 for batch and channel dimensions, at my_conv3d");
  }
  struct TestParams {
    std::vector<int> input_dims;
    std::vector<float> input;
    std::vector<int> filter_dims;
    std::vector<float> filter;
    std::vector<int> strides;
    string padding;
    string data_format;
    std::vector<int> dilations;
    bool is_conv3d_backprop_input;
    std::vector<int> expected_output_dims;
    std::vector<float> expected_output;
  };

  // Start here
  std::vector<TestParams> ok_params = {
      // Basic - just 1x1 conv - input = output
      TestParams{
          /*input_dims=*/{1, 3, 3, 3},  // CDHW
          /*input=*/{1, 2,  15,  3, 6,  -3, 22, 1, 88, 56, 36, 1,  1, 105,
                     1, 16, -28, 1, 42, 9,  3,  1, 7,  1,  11, 61, 5},
          /*filter_dims=*/{1, 1, 1, 1, 1},  // DRSCK
          /*filter=*/{1},
          /*strides=*/{1, 1, 1, 1, 1},
          /*padding=*/"VALID",
          /*data_format=*/"NCDHW",
          /*dilations=*/{1, 1, 1, 1, 1},
          /*is_conv3d_backprop_input=*/false,
          /*expected_output_dims=*/{1, 3, 3, 3},
          /*expected_output=*/{1,  2,  15, 3, 6,   -3, 22, 1,   88,
                               56, 36, 1,  1, 105, 1,  16, -28, 1,
                               42, 9,  3,  1, 7,   1,  11, 61,  5}},
      // Basic - 2x1 filter
      TestParams{/*input_dims=*/{1, 3, 3, 3},  // CDHW
                 /*input=*/{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6},
                 /*filter_dims=*/{2, 1, 1, 1, 1},  // DRSCK
                 /*filter=*/{1, 1},
                 /*strides=*/{1, 1, 1, 1, 1},
                 /*padding=*/"VALID",
                 /*data_format=*/"NCDHW",
                 /*dilations=*/{1, 1, 1, 1, 1},
                 /*is_conv3d_backprop_input=*/false,
                 /*expected_output_dims=*/{1, 2, 3, 3},
                 /*expected_output=*/
                 {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 7}},
      // SAME padding (Asymmetric)
      TestParams{
          /*input_dims=*/{1, 2, 3, 2},  // CDHW
          /*input=*/{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
          /*filter_dims=*/{2, 1, 1, 1, 1},  // DRSCK
          /*filter=*/{-1, 1},
          /*strides=*/{1, 1, 1, 1, 1},
          /*padding=*/"SAME",
          /*data_format=*/"NCDHW",
          /*dilations=*/{1, 1, 1, 1, 1},
          /*is_conv3d_backprop_input=*/false,
          /*expected_output_dims=*/{1, 2, 3, 2},
          /*expected_output=*/
          {6, 6, 6, 6, 6, 6, -6, -7, -8, -9, -10,
           -11}  // Diff in first 2 depths is const 6
      },
      // SAME padding (Symmetric)
      TestParams{
          /*input_dims=*/{1, 2, 3, 2},  // CDHW
          /*input=*/{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
          /*filter_dims=*/{3, 1, 1, 1, 1},  // DRSCK
          /*filter=*/{-1, 0, 1},
          /*strides=*/{1, 1, 1, 1, 1},
          /*padding=*/"SAME",
          /*data_format=*/"NCDHW",
          /*dilations=*/{1, 1, 1, 1, 1},
          /*is_conv3d_backprop_input=*/false,
          /*expected_output_dims=*/{1, 2, 3, 2},
          /*expected_output=*/
          {6, 7, 8, 9, 10, 11, 0, -1, -2, -3, -4,
           -5}  // Swaps front two depths, negates
      },

      // NDHWC (multi-channel)
      TestParams{
          /*input_dims=*/{2, 3, 2, 2},  // DHWC
          /*input=*/{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
          /*filter_dims=*/{2, 1, 1, 2, 1},  // DRSCK
          /*filter=*/{-1, 1, 1, -1},
          /*strides=*/{1, 1, 1, 1, 1},
          /*padding=*/"VALID",
          /*data_format=*/"NDHWC",
          /*dilations=*/{1, 1, 1, 1, 1},
          /*is_conv3d_backprop_input=*/false,
          /*expected_output_dims=*/{1, 3, 2, 1},
          /*expected_output=*/{0, 0, 0, 0, 0, 0}  // Each filter opposes the
                                                  // other
      },

      // Dilated
      TestParams{
          /*input_dims=*/{1, 3, 3, 3},  // CDHW
          /*input=*/{1,   1,   1,   1,   1, 1, 1, 1, 1, -10, -10, -10, -10, -10,
                     -10, -10, -10, -10, 7, 7, 7, 7, 7, 7,   7,   7,   7},
          /*filter_dims=*/{2, 1, 1, 1, 1},  // DRSCK
          /*filter=*/{1, 1},
          /*strides=*/{1, 1, 1, 1, 1},
          /*padding=*/"VALID",
          /*data_format=*/"NCDHW",
          /*dilations=*/{1, 1, 2, 1, 1},
          /*is_conv3d_backprop_input=*/false,
          /*expected_output_dims=*/{1, 1, 3, 3},
          /*expected_output=*/{8, 8, 8, 8, 8, 8, 8, 8, 8}  // Only front depth
                                                           // is valid, skips
                                                           // neg values
      },
      // Strided
      TestParams{
          /*input_dims=*/{1, 3, 3, 3},
          /*input=*/{1, 0, 2, 0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 5, 0, 6, 0, 0, 0, 7, 0, 8},
          /*filter_dims=*/{1, 1, 1, 1, 1},
          /*filter=*/{1},
          /*strides=*/{1, 1, 2, 2, 2},
          /*padding=*/"VALID",
          /*data_format=*/"NCDHW",
          /*dilations=*/{1, 1, 1, 1, 1},
          /*is_conv3d_backprop_input=*/false,
          /*expected_output_dims=*/{1, 2, 2, 2},
          /*expected_output=*/{1, 2, 3, 4, 5, 6, 7, 8}  // Should only pick up
                                                        // the corners
      },
      // Transpose Strided
      TestParams{/*input_dims=*/{1, 2, 2, 2},  // CDHW
                 /*input=*/{1, 2, 3, 4, 5, 6, 7, 8},
                 /*filter_dims=*/{1, 1, 1, 1, 1},
                 /*filter=*/{1},
                 /*strides=*/{1, 1, 2, 2, 2},
                 /*padding=*/"VALID",
                 /*data_format=*/"NCDHW",
                 /*dilations=*/{1, 1, 1, 1, 1},
                 /*is_conv3d_backprop_input=*/true,
                 /*expected_output_dims=*/{1, 3, 3, 3},
                 /*expected_output=*/
                 {1, 0, 2, 0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 5, 0, 6, 0, 0, 0, 7, 0, 8}},  // Cube
                                                            // expands and
                                                            // fills
                                                            // center with
                                                            // zeroes

  };

  for (int i = 0; i < ok_params.size(); i++) {
    Reset();
    NodeDef node_def = get_conv3d_nodedef(
        ok_params[i].strides, ok_params[i].padding, ok_params[i].data_format,
        ok_params[i].dilations, ok_params[i].is_conv3d_backprop_input);
    AddTestTensor("input", ok_params[i].input_dims);
    AddTestWeights<float>("weights", ok_params[i].filter_dims,
                          ok_params[i].filter);
    if (ok_params[i].is_conv3d_backprop_input) {
      AddTestWeights<float>(
          "input_sizes",
          {static_cast<int>(ok_params[i].expected_output.size())},
          ok_params[i].expected_output);
    }
    RunValidationAndConversion(node_def);
    TRT_TensorOrWeights output;
    TF_EXPECT_OK(GetTensorOrWeights("my_conv3d", &output));
    ASSERT_TRUE(output.is_tensor());
    ExpectTrtDimsEqualsArray(ok_params[i].expected_output_dims,
                             output.tensor()->getDimensions());

    const DataVec input_data{{"input", AsTensor<float>(ok_params[i].input)}};
    DataVec output_data{
        {"my_conv3d",
         ConstructTensor<float>(ok_params[i].expected_output.size())}};
    TF_EXPECT_OK(BuildAndRun(input_data, &output_data));
    EXPECT_THAT(GetSpanForData<float>(output_data[0]),
                ElementsAreArray(ok_params[i].expected_output));
  }
}

TEST_F(OpConverterTest, ConvertPool3D) {
  // Get nodedef for MaxPool3D and AvgPool3D layers.
  auto get_pool3d_nodedef = [](std::vector<int> ksize = {1, 1, 1, 1, 1},
                               std::vector<int> strides = {1, 1, 1, 1, 1},
                               string padding = "SAME",
                               string data_format = "NCDHW",
                               const bool is_max_pooling = true) -> NodeDef {
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT);

    if (is_max_pooling) {
      ops::MaxPool3D::Attrs attrs =
          ops::MaxPool3D::Attrs().DataFormat(data_format);
      auto pool3d = ops::MaxPool3D(s.WithOpName("my_maxpool3d"), input, ksize,
                                   strides, padding, attrs);
      return pool3d.operation.node()->def();
    } else {
      ops::AvgPool3D::Attrs attrs =
          ops::AvgPool3D::Attrs().DataFormat(data_format);
      auto pool3d = ops::AvgPool3D(s.WithOpName("my_avgpool3d"), input, ksize,
                                   strides, padding, attrs);
      return pool3d.operation.node()->def();
    }
  };

  {
    // Input is weights, should fail.
    Reset();
    NodeDef node_def = get_pool3d_nodedef();

    AddTestWeights<float>("input", {1, 2, 3}, {1, 2, 3, 4, 5, 6});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"input\" for MaxPool3D must be a tensor, at my_maxpool3d");
  }

  struct TestParams {
    std::vector<int> input_dims;
    std::vector<float> input;
    std::vector<int> ksize;
    std::vector<int> strides;
    string padding;
    string data_format;
    bool is_max_pooling;
    std::vector<int> expected_output_dims;
    std::vector<float> expected_output;
  };

  // Start here
  const std::vector<float> common_array{-4, 2,  15, 3, 6,   -3, 22, 1,   88,
                                        56, 36, 1,  1, 105, 1,  16, -28, 1,
                                        42, 9,  3,  1, 7,   1,  11, 61,  5};
  std::vector<TestParams> ok_params = {
      // Basic - just 1x1 max pooling - input = output
      TestParams{/*input_dims=*/{1, 3, 3, 3},
                 /*input=*/common_array,
                 /*ksize=*/{1, 1, 1, 1, 1},
                 /*strides=*/{1, 1, 1, 1, 1},
                 /*padding=*/"VALID",
                 /*data_format=*/"NCDHW",
                 /*is_max_pooling=*/true,
                 /*expected_output_dims=*/{1, 3, 3, 3},
                 /*expected_output=*/common_array},
      // Basic - just 1x1 avg pooling - input = output
      TestParams{/*input_dims=*/{1, 3, 3, 3},
                 /*input=*/common_array,
                 /*ksize=*/{1, 1, 1, 1, 1},
                 /*strides=*/{1, 1, 1, 1, 1},
                 /*padding=*/"VALID",
                 /*data_format=*/"NCDHW",
                 /*is_max_pooling=*/false,
                 /*expected_output_dims=*/{1, 3, 3, 3},
                 /*expected_output=*/common_array},
      // Basic - just 1x1 max pooling - input = output, SAME padding
      TestParams{/*input_dims=*/{1, 3, 3, 3},
                 /*input=*/common_array,
                 /*ksize=*/{1, 1, 1, 1, 1},
                 /*strides=*/{1, 1, 1, 1, 1},
                 /*padding=*/"SAME",
                 /*data_format=*/"NCDHW",
                 /*is_max_pooling=*/true,
                 /*expected_output_dims=*/{1, 3, 3, 3},
                 /*expected_output=*/common_array},
      // Basic - just 1x1 avg pooling - input = output, SAME padding
      TestParams{/*input_dims=*/{1, 3, 3, 3},
                 /*input=*/common_array,
                 /*ksize=*/{1, 1, 1, 1, 1},
                 /*strides=*/{1, 1, 1, 1, 1},
                 /*padding=*/"VALID",
                 /*data_format=*/"NCDHW",
                 /*is_max_pooling=*/false,
                 /*expected_output_dims=*/{1, 3, 3, 3},
                 /*expected_output=*/common_array},
      // 3x3 max pooling
      TestParams{/*input_dims=*/{1, 3, 3, 3},
                 /*input=*/common_array,
                 /*ksize=*/{1, 1, 3, 3, 3},
                 /*strides=*/{1, 1, 1, 1, 1},
                 /*padding=*/"VALID",
                 /*data_format=*/"NCDHW",
                 /*is_max_pooling=*/true,
                 /*expected_output_dims=*/{1, 1, 1, 1},
                 /*expected_output=*/{105}},
      // 3x3 avg pooling
      TestParams{/*input_dims=*/{1, 3, 3, 3},
                 /*input=*/common_array,
                 /*ksize=*/{1, 1, 3, 3, 3},
                 /*strides=*/{1, 1, 1, 1, 1},
                 /*padding=*/"VALID",
                 /*data_format=*/"NCDHW",
                 /*is_max_pooling=*/false,
                 /*expected_output_dims=*/{1, 1, 1, 1},
                 /*expected_output=*/{17}},
      // 3x3 max pooling, NDHWC
      TestParams{/*input_dims=*/{3, 3, 3, 1},
                 /*input=*/common_array,
                 /*ksize=*/{1, 3, 3, 3, 1},
                 /*strides=*/{1, 1, 1, 1, 1},
                 /*padding=*/"VALID",
                 /*data_format=*/"NDHWC",
                 /*is_max_pooling=*/true,
                 /*expected_output_dims=*/{1, 1, 1, 1},
                 /*expected_output=*/{105}},
      // 3x3 avg pooling, NDHWC
      TestParams{/*input_dims=*/{3, 3, 3, 1},
                 /*input=*/common_array,
                 /*ksize=*/{1, 3, 3, 3, 1},
                 /*strides=*/{1, 1, 1, 1, 1},
                 /*padding=*/"VALID",
                 /*data_format=*/"NDHWC",
                 /*is_max_pooling=*/false,
                 /*expected_output_dims=*/{1, 1, 1, 1},
                 /*expected_output=*/{17}},
      // Strided max
      TestParams{
          /*input_dims=*/{1, 3, 3, 3},
          /*input=*/{1, 0, 2, 0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 5, 0, 6, 0, 0, 0, 7, 0, 8},
          /*ksize=*/{1, 1, 1, 1, 1},
          /*strides=*/{1, 1, 2, 2, 2},
          /*padding=*/"VALID",
          /*data_format=*/"NCDHW",
          /*is_max_pooling=*/true,
          /*expected_output_dims=*/{1, 2, 2, 2},
          /*expected_output=*/{1, 2, 3, 4, 5, 6, 7, 8}  // Should only pick up
                                                        // the corners
      },
      // Strided avg
      TestParams{
          /*input_dims=*/{1, 3, 3, 3},
          /*input=*/{1, 0, 2, 0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 5, 0, 6, 0, 0, 0, 7, 0, 8},
          /*ksize=*/{1, 1, 1, 1, 1},
          /*strides=*/{1, 1, 2, 2, 2},
          /*padding=*/"VALID",
          /*data_format=*/"NCDHW",
          /*is_max_pooling=*/false,
          /*expected_output_dims=*/{1, 2, 2, 2},
          /*expected_output=*/{1, 2, 3, 4, 5, 6, 7, 8}  // Should only pick up
                                                        // the corners
      }};

  for (int i = 0; i < ok_params.size(); i++) {
    Reset();
    NodeDef node_def = get_pool3d_nodedef(
        ok_params[i].ksize, ok_params[i].strides, ok_params[i].padding,
        ok_params[i].data_format, ok_params[i].is_max_pooling);
    AddTestTensor("input", ok_params[i].input_dims);
    RunValidationAndConversion(node_def);
    TRT_TensorOrWeights output;
    string expected_node_name =
        ok_params[i].is_max_pooling ? "my_maxpool3d" : "my_avgpool3d";
    TF_EXPECT_OK(GetTensorOrWeights(expected_node_name, &output));
    ASSERT_TRUE(output.is_tensor());
    ExpectTrtDimsEqualsArray(ok_params[i].expected_output_dims,
                             output.tensor()->getDimensions());

    const DataVec input_data{{"input", AsTensor<float>(ok_params[i].input)}};
    DataVec output_data{
        {expected_node_name,
         ConstructTensor<float>(ok_params[i].expected_output.size())}};
    TF_EXPECT_OK(BuildAndRun(input_data, &output_data));
    EXPECT_THAT(GetSpanForData<float>(output_data[0]),
                ElementsAreArray(ok_params[i].expected_output));
  }
}
#endif  // IS_TRT_VERSION_GE(6, 0, 0, 0)

TEST_F(OpConverterTest, ConvertTopK) {
  // TODO(tmorris): This test isn't setting the input dtype properly. TopK with
  // int32 is unsupported by TRT.
  for (const auto dtype : {DT_FLOAT}) {
    // Get the NodeDef for TopKV2.
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), dtype);
    auto weights = ops::Placeholder(s.WithOpName("weights"), DT_INT32);
    auto topk = ops::TopK(s.WithOpName("my_topk"), input, weights);
    const NodeDef& node_def = topk.operation.node()->def();
    {
      // K is a tensor, should fail.
      Reset();
      AddTestTensor("input", {1, 2, 3}, /*batch_size=*/1,
                    /*trt_dtype=*/TfDataTypeToTrt(dtype));
      AddTestTensor("weights", {2});
      RunValidationAndConversion(
          node_def, error::UNIMPLEMENTED,
          "The input \"k\" for TopKV2 must be a constant, at my_topk");
    }
    {
      // Ok.
      Reset();
      AddTestTensor("input", {1, 2, 5});
      AddTestWeights<int32>("weights", {1}, {2});
      RunValidationAndConversion(node_def);
      TRT_TensorOrWeights outputs[2];
      TF_EXPECT_OK(GetTensorOrWeights("my_topk", &outputs[0]));
      TF_EXPECT_OK(GetTensorOrWeights("my_topk:1", &outputs[1]));
      for (auto& output : outputs) {
        ASSERT_TRUE(output.is_tensor());
        ExpectTrtDimsEqualsArray({1, 2, 2}, output.tensor()->getDimensions());
      }

      const DataVec input_data{
          {"input", AsTensor<float>({-9, 3, 5, 1, 6, -5, 7, 1, 0, -1})}};
      DataVec output_data{{"my_topk", ConstructTensor<float>(4)},
                          {"my_topk:1", ConstructTensor<int32>(4)}};
      TF_EXPECT_OK(BuildAndRun(input_data, &output_data));
      EXPECT_THAT(GetSpanForData<float>(output_data[0]),
                  ElementsAre(6, 5, 7, 1));
      EXPECT_THAT(GetSpanForData<int32>(output_data[1]),
                  ElementsAre(4, 2, 1, 2));
    }
  }
}

template <DataType dtype>
void TestConvertGather(OpConverterTest* test) {
  typedef typename EnumToDataType<dtype>::Type CType;

  // Get the NodeDef for GatherV2.
  Scope s = Scope::NewRootScope();
  auto params = ops::Placeholder(s.WithOpName("params"), dtype);
  auto indices = ops::Placeholder(s.WithOpName("indices"), DT_INT32);
  auto axis = ops::Placeholder(s.WithOpName("axis"), DT_INT32);
  auto gather = ops::GatherV2(s.WithOpName("my_gather"), params, indices, axis);
  const NodeDef& node_def = gather.operation.node()->def();

  struct TestParams {
    // TF shape of the input 'params' (including batch dimension).
    std::vector<int> params_shape;
    // TF shape of the input 'indices' (including batch dimension).
    std::vector<int> indices_shape;
    std::vector<int> indices;
    int axis;
    // Expected TF shape of the output (including batch dimension).
    std::vector<int> expected_output_shape;
    std::vector<int> expected_output;
    bool params_is_tensor;
  };

  // Input is the same {1, 2, 3, 4, 5, 6} for all cases.
  const std::vector<CType> params_input = {CType(1), CType(2), CType(3),
                                           CType(4), CType(5), CType(6)};
  std::vector<TestParams> ok_params = {
      // Vector indices, and output rank is rank(params).
      TestParams{
          /*params_shape=*/{1, 1, 2, 3},
          /*indices_shape=*/{1},
          /*indices=*/{0},
          /*axis=*/3,
          /*expected_output_shape=*/{1, 1, 2, 1},
          /*expected_output=*/{1, 4},
          /*params_is_tensor=*/true,
      },
      TestParams{
          /*params_shape=*/{1, 1, 2, 3},
          /*indices_shape=*/{1},
          /*indices=*/{1},
          /*axis=*/2,
          /*expected_output_shape=*/{1, 1, 1, 3},
          /*expected_output=*/{4, 5, 6},
          /*params_is_tensor=*/true,
      },
      // Indices with rank>1, and output rank is rank(params)+rank(indices)-1.
      TestParams{
          /*params_shape=*/{1, 1, 2, 3},
          /*indices_shape=*/{1, 1},
          /*indices=*/{0},
          /*axis=*/3,
          /*expected_output_shape=*/{1, 1, 2, 1, 1},
          /*expected_output=*/{1, 4},
          /*params_is_tensor=*/true,
      },
      TestParams{
          /*params_shape=*/{1, 1, 2, 3},
          /*indices_shape=*/{1, 1},
          /*indices=*/{1},
          /*axis=*/3,
          /*expected_output_shape=*/{1, 1, 2, 1, 1},
          /*expected_output=*/{2, 5},
          /*params_is_tensor=*/true,
      },
      TestParams{
          /*params_shape=*/{1, 1, 2, 3},
          /*indices_shape=*/{1, 1},
          /*indices=*/{2},
          /*axis=*/-1,
          /*expected_output_shape=*/{1, 1, 2, 1, 1},
          /*expected_output=*/{3, 6},
          /*params_is_tensor=*/true,
      },
      TestParams{
          /*params_shape=*/{1, 1, 2, 3},
          /*indices_shape=*/{1, 3},
          /*indices=*/{2, 0, 1},
          /*axis=*/3,
          /*expected_output_shape=*/{1, 1, 2, 1, 3},
          /*expected_output=*/{3, 1, 2, 6, 4, 5},
          /*params_is_tensor=*/true,
      },
      TestParams{
          /*params_shape=*/{1, 3, 2},
          /*indices_shape=*/{1, 2, 2},
          /*indices=*/{0, 0, 1, 0},
          /*axis=*/2,
          /*expected_output_shape=*/{1, 3, 1, 2, 2},
          /*expected_output=*/{1, 1, 2, 1, 3, 3, 4, 3, 5, 5, 6, 5},
          /*params_is_tensor=*/true,
      },
      TestParams{
          /*params_shape=*/{1, 2, 3},
          /*indices_shape=*/{1},
          /*indices=*/{0},
          /*axis=*/0,
          /*expected_output_shape=*/{1, 2, 3},
          /*expected_output=*/{1, 2, 3, 4, 5, 6},
          /*params_is_tensor=*/false,
      },
      TestParams{
          /*params_shape=*/{3, 2},
          /*indices_shape=*/{1, 2},
          /*indices=*/{0, 1},
          /*axis=*/0,
          /*expected_output_shape=*/{1, 2, 2},
          /*expected_output=*/{1, 2, 3, 4},
          /*params_is_tensor=*/false,
      },
      TestParams{
          /*params_shape=*/{2, 3},
          /*indices_shape=*/{1, 1, 2},
          /*indices=*/{0, 1},
          /*axis=*/0,
          /*expected_output_shape=*/{1, 1, 2, 3},
          /*expected_output=*/{1, 2, 3, 4, 5, 6},
          /*params_is_tensor=*/false,
      },
      TestParams{
          /*params_shape=*/{3, 2},
          /*indices_shape=*/{2, 2},
          /*indices=*/{0, 2, 1, 0},
          /*axis=*/0,
          /*expected_output_shape=*/{2, 2, 2},
          /*expected_output=*/{1, 2, 5, 6, 3, 4, 1, 2},
          /*params_is_tensor=*/false,
      },
  };

  // Ok.
  for (int i = 0; i < ok_params.size(); i++) {
    test->Reset();
    const auto& params_shape = ok_params[i].params_shape;
    if (ok_params[i].params_is_tensor) {
      std::vector<int> params_dims(params_shape.begin() + 1,
                                   params_shape.end());
      test->AddTestTensor("params", params_dims, params_shape[0],
                          TfDataTypeToTrt(dtype));
    } else {
      test->AddTestWeights<CType>("params", params_shape, params_input);
    }

    const auto& indices_shape = ok_params[i].indices_shape;
    test->AddTestTensor(
        "indices",
        std::vector<int>(indices_shape.begin() + 1, indices_shape.end()),
        indices_shape[0], nvinfer1::DataType::kINT32);
    test->AddTestWeights<int32>("axis", {1}, {ok_params[i].axis});
    test->RunValidationAndConversion(node_def);
    TRT_TensorOrWeights output;
    TF_EXPECT_OK(test->GetTensorOrWeights("my_gather", &output));
    ASSERT_TRUE(output.is_tensor());

    const auto& expected_output_shape = ok_params[i].expected_output_shape;
    const auto& expected_output = ok_params[i].expected_output;
    ASSERT_EQ(expected_output.size(),
              TrtWeightDimsNumElements(GetTestDims(expected_output_shape)));
    const std::vector<int> expected_output_dims(
        expected_output_shape.begin() + 1, expected_output_shape.end());
    ExpectTrtDimsEqualsArray(expected_output_dims,
                             output.tensor()->getDimensions());

    // Create input in CType and convert expected output to CType.
    std::vector<CType> converted_expected_output(expected_output.begin(),
                                                 expected_output.end());

    DataVec input_data;
    if (ok_params[i].params_is_tensor) {
      input_data = {{"params", test->AsTensor<CType>(params_input)},
                    {"indices", test->AsTensor<int32>(ok_params[i].indices)}};
    } else {
      input_data = {{"indices", test->AsTensor<int32>(ok_params[i].indices)}};
    }
    DataVec output_data{
        {"my_gather", test->ConstructTensor<CType>(expected_output.size())}};
    TF_EXPECT_OK(test->BuildAndRun(input_data, &output_data,
                                   /*batch_size=*/expected_output_shape[0]));
    EXPECT_THAT(GetSpanForData<CType>(output_data[0]),
                ElementsAreArray(converted_expected_output));
  }
}

TEST_F(OpConverterTest, ConvertGather) {
  // Get the NodeDef for GatherV2.
  Scope s = Scope::NewRootScope();
  auto params = ops::Placeholder(s.WithOpName("params"), DT_FLOAT);
  auto indices = ops::Placeholder(s.WithOpName("indices"), DT_INT32);
  auto axis = ops::Placeholder(s.WithOpName("axis"), DT_INT32);
  auto gather = ops::GatherV2(s.WithOpName("my_gather"), params, indices, axis);
  const NodeDef& node_def = gather.operation.node()->def();
  {
    // Axis is a tensor, should fail.
    Reset();
    AddTestTensor("params", {1, 2, 3});
    AddTestTensor("indices", {2});
    AddTestTensor("axis", {1});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"axis\" for GatherV2 must be a constant, at my_gather");
  }
  {
    // Axis is out of bounds, should fail.
    Reset();
    AddTestTensor("params", {1, 2, 3});
    AddTestTensor("indices", {2});
    AddTestWeights<int32>("axis", {1}, {4});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "Axis value of 4 is out of bounds, must be in "
                               "range [-4, 4), at my_gather");
  }
  {
    // Axis is batch dimension, should fail.
    Reset();
    AddTestTensor("params", {1, 2, 3});
    AddTestTensor("indices", {2});
    AddTestWeights<int32>("axis", {1}, {0});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "TensorRT does not allow manipulation of the "
                               "batch dimension, at my_gather");
  }
  {
    // Axis is not zero when params is a weight, should fail.
    Reset();
    AddTestWeights<int32>("params", {1, 3}, {1, 2, 3});
    AddTestTensor("indices", {2});
    AddTestWeights<int32>("axis", {1}, {1});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input axis must be zero when params is a weight.");
  }
  {
    // Batch size of indices is not 1 when params is a tensor.
    Reset();
    AddTestTensor("params", {1, 2, 3}, /*batch_size=*/2);
    AddTestTensor("indices", {2}, /*batch_size=*/2);
    AddTestWeights<int32>("axis", {1}, {1});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "Indices must have a batch size of 1 when params is a tensor.");
  }

  Reset();
  TestConvertGather<DT_FLOAT>(this);
  TestConvertGather<DT_HALF>(this);
  TestConvertGather<DT_INT32>(this);
}

NodeDef CreateCastOp(DataType tf_dtype) {
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), DT_HALF);
  return ops::Cast(s.WithOpName("my_unary"), input, DT_FLOAT)
      .operation.node()
      ->def();
}

TEST_P(OpConverterTest1, ConvertUnary) {
  {
    // Input is weights, should fail.
    Reset();
    const NodeDef node_def = CreateUnaryOp<ops::Neg>(tf_dtype);
    AddTestWeights<float>("input", {1, 2, 3}, {-3, -2, -1, 0, 1, 2});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"x\" for Neg must be a tensor, at my_unary");
  }
  using OpFunc = std::function<NodeDef(DataType)>;
  using ValFunc = float (*)(float);
  std::map<std::string, std::pair<OpFunc, ValFunc>> op_map;
#define ADD_OP(name, op, compute) \
  op_map[name] =                  \
      std::make_pair(CreateUnaryOp<op>, static_cast<ValFunc>(compute))
  ADD_OP("Abs", ops::Abs, std::abs);
  ADD_OP("Acos", ops::Acos, std::acos);
  ADD_OP("Acosh", ops::Acosh, std::acosh);
  ADD_OP("Asin", ops::Asin, std::asin);
  ADD_OP("Asinh", ops::Asinh, std::asinh);
  ADD_OP("Atan", ops::Atan, std::atan);
  ADD_OP("Atanh", ops::Atanh, std::atanh);
  op_map["Cast"] = std::make_pair(CreateCastOp, [](float x) { return x; });
  ADD_OP("Ceil", ops::Ceil, std::ceil);
  ADD_OP("Cos", ops::Cos, std::cos);
  ADD_OP("Cosh", ops::Cosh, std::cosh);
  ADD_OP("Exp", ops::Exp, std::exp);
  ADD_OP("Floor", ops::Floor, std::floor);
  ADD_OP("Log", ops::Log, std::log);
  ADD_OP("Neg", ops::Neg, [](float x) { return -x; });
  ADD_OP("Reciprocal", ops::Reciprocal, [](float x) { return 1.0f / x; });
  ADD_OP("Rsqrt", ops::Rsqrt, [](float x) { return 1.0f / std::sqrt(x); });
  ADD_OP("Sin", ops::Sin, std::sin);
  ADD_OP("Sinh", ops::Sinh, std::sinh);
  ADD_OP("Sqrt", ops::Sqrt, std::sqrt);
  ADD_OP("Tan", ops::Tan, std::tan);
#undef ADD_OP
  // Get list of ops to test.
  std::vector<string> ops_to_test;
  // Add all ops supported by ConvertUnary.
  auto* map = UnaryOperationMap();
  ops_to_test.reserve(map->size());
  for (auto& pair : *map) {
    ops_to_test.push_back(pair.first);
  }
  // Add other unary ops to test.
  ops_to_test.push_back("Rsqrt");
  // Prepare test parameters
  auto p = TestParamBase{
      {1, 1, 2, 3},  // input dims
      {},            // input partial dims
      {1, 1, 2, 3},  // expected output dims
  };
  for (const string& op_name : ops_to_test) {
    SCOPED_TRACE(op_name);
    Reset();
    if (!op_map.count(op_name)) {
      FAIL() << "Unary op test map does not contain op " << op_name;
    }
    NodeDef node_def = op_map[op_name].first(tf_dtype);

    // TODO(bixia): we assume this test is only instantiated for DT_FLOAT for
    // now. Need to find a better way to express input and output types.
    //
    // TODO(tfeher): improve tests by defining an expected output data type and
    // check that. Currently only the shape and values of the output are
    // checked.
    DataType input_tf_dtype = op_name == "Cast" ? DT_HALF : tf_dtype;

    std::vector<float> input_values{-0.9f, 0.6f, 0.0f, -3.5f, 100.0f, 2.9f};
    AddTestTensor("input", p.input_dims, input_tf_dtype, input_values);
    std::vector<float> output;
    std::transform(input_values.begin(), input_values.end(),
                   std::back_inserter(output), op_map[op_name].second);
    TestOpConverter("my_unary", node_def, p.expected_output_dims, Status::OK(),
                    p.runtime_status, ArrayFloatNear(output, 0.0001, true));
  }
}

// Get the NodeDef for ConcatV2.
// TODO(hinsu): Consider switching this to static function.
auto get_concat_nodedef = [](DataType dtype, int num_inputs) -> NodeDef {
  Scope s = Scope::NewRootScope();
  std::vector<Input> values;
  for (int i = 0; i < num_inputs; ++i) {
    const string input_name = StrCat("values_", i);
    values.push_back(ops::Placeholder(s.WithOpName(input_name), dtype));
  }
  auto axis = ops::Placeholder(s.WithOpName("axis"), DT_INT32);
  auto concat = ops::Concat(s.WithOpName("my_concat"),
                            absl::Span<const Input>(values), axis);
  return concat.operation.node()->def();
};

template <DataType dtype>
void TestConvertConcat(OpConverterTest* test) {
  typedef typename EnumToDataType<dtype>::Type CType;

  struct TestParams {
    std::vector<std::vector<int>> input_shapes;
    std::vector<std::vector<CType>> input_values;
    int axis;
    std::vector<int> expected_output_dims;
    std::vector<CType> expected_output;
  };

  const std::vector<std::vector<CType>> common_input{
      InitTestVector<CType>(6),
      InitTestVector<CType>(6, /*start_value=*/CType(6))};
  // TODO(hinsu): Use std::vector instead of an array to avoid use of explicit
  // size.
  std::vector<TestParams> ok_params = {
      {
          /*input_shapes=*/{{1, 2, 3}, {1, 2, 3}},
          /*input_values=*/common_input,
          /*axis=*/1,
          /*expected_output_dims=*/{2, 2, 3},
          /*expected_output=*/InitTestVector<CType>(12),
      },
      {
          /*input_shapes=*/{{1, 2, 3}, {1, 2, 3}},
          /*input_values=*/common_input,
          /*axis=*/2,
          /*expected_output_dims=*/{1, 4, 3},
          /*expected_output=*/InitTestVector<CType>(12),
      },
      {
          /*input_shapes=*/{{1, 2, 3}, {1, 2, 3}},
          /*input_values=*/common_input,
          /*axis=*/3,
          /*expected_output_dims=*/{1, 2, 6},
          /*expected_output=*/
          {CType(0), CType(1), CType(2), CType(6), CType(7), CType(8), CType(3),
           CType(4), CType(5), CType(9), CType(10), CType(11)},
      },
      {
          /*input_shapes=*/{{1}, {2}, {3}, {1}, {1}, {2}},
          /*input_values=*/
          {{CType(1)},
           {CType(2), CType(3)},
           {CType(4), CType(5), CType(6)},
           {CType(7)},
           {CType(8)},
           {CType(9), CType(10)}},
          /*axis=*/1,
          /*expected_output_dims=*/{10},
          /*expected_output=*/
          InitTestVector<CType>(10, /*start_value=*/CType(1)),
      },
  };

  for (int i = 0; i < ok_params.size(); ++i) {
    test->Reset();
    const int num_inputs = ok_params[i].input_shapes.size();
    EXPECT_EQ(num_inputs, ok_params[i].input_values.size());
    NodeDef node_def = get_concat_nodedef(dtype, num_inputs);
    // Create inputs.
    for (int j = 0; j < num_inputs; ++j) {
      test->AddTestTensor(StrCat("values_", j), ok_params[i].input_shapes[j], 1,
                          TfDataTypeToTrt(dtype));
    }
    test->AddTestWeights<int32>("axis", {1}, {ok_params[i].axis});
    test->RunValidationAndConversion(node_def);

    TRT_TensorOrWeights output;
    TF_EXPECT_OK(test->GetTensorOrWeights("my_concat", &output));
    ASSERT_TRUE(output.is_tensor());
    ExpectTrtDimsEqualsArray(ok_params[i].expected_output_dims,
                             output.tensor()->getDimensions());
    // Create input data for tensors.
    DataVec input_data;
    for (int j = 0; j < num_inputs; ++j) {
      input_data.push_back(
          {StrCat("values_", j),
           test->AsTensor<CType>(ok_params[i].input_values[j])});
    }
    DataVec output_data{
        {"my_concat",
         test->ConstructTensor<CType>(ok_params[i].expected_output.size())}};
    TF_EXPECT_OK(test->BuildAndRun(input_data, &output_data));
    EXPECT_THAT(GetSpanForData<CType>(output_data[0]),
                ElementsAreArray(ok_params[i].expected_output));
  }
}

TEST_F(OpConverterTest, ConvertConcat) {
  {
    // Axis is a tensor, should fail.
    Reset();
    NodeDef node_def = get_concat_nodedef(DT_FLOAT, 2);
    AddTestTensor("values_0", {1, 2, 3});
    AddTestTensor("values_1", {1, 2, 3});
    AddTestTensor("axis", {1});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"axis\" for ConcatV2 must be a constant, at my_concat");
  }
  {
    // Axis is out of bounds, should fail.
    Reset();
    NodeDef node_def = get_concat_nodedef(DT_FLOAT, 2);
    AddTestTensor("values_0", {1, 2, 3});
    AddTestTensor("values_1", {1, 2, 3});
    AddTestWeights<int32>("axis", {1}, {4});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "Axis value of 4 is out of bounds, must be in "
                               "range [-4, 4), at my_concat");
  }
  {
    // Axis is batch dimension, should fail.
    Reset();
    NodeDef node_def = get_concat_nodedef(DT_FLOAT, 2);
    AddTestTensor("values_0", {1, 2, 3});
    AddTestTensor("values_1", {1, 2, 3});
    AddTestWeights<int32>("axis", {1}, {0});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "TensorRT does not allow manipulation of the "
                               "batch dimension, at my_concat");
  }
  {
    // Inputs have inconsistent rank, should fail.
    Reset();
    NodeDef node_def = get_concat_nodedef(DT_FLOAT, 2);
    AddTestTensor("values_0", {1, 2, 3});
    AddTestTensor("values_1", {1, 6});
    AddTestWeights<int32>("axis", {1}, {1});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Received inputs with inconsistent rank, at my_concat");
  }
  {
    // An input is a weight, should fail.
    Reset();
    NodeDef node_def = get_concat_nodedef(DT_FLOAT, 2);
    AddTestTensor("values_0", {1, 2, 3});
    AddTestWeights<float>("values_1", {1, 2, 3}, {1, 2, 3, 4, 5, 6});
    AddTestWeights<int32>("axis", {1}, {1});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"values_1\" for ConcatV2 must be a tensor, at my_concat");
  }
  {
    // Inputs have inconsistent non-axis shapes, should fail.
    Reset();
    NodeDef node_def = get_concat_nodedef(DT_FLOAT, 2);
    AddTestTensor("values_0", {1, 2, 3});
    AddTestTensor("values_1", {1, 3, 2});
    AddTestWeights<int32>("axis", {1}, {1});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Received inputs with inconsistent shape, at my_concat");
  }

  TestConvertConcat<DT_FLOAT>(this);
  TestConvertConcat<DT_HALF>(this);
  // TODO(tmorris): Enable once TRT adds support.
  // TestConvertConcat<DT_INT32>(this);
}

// Get the NodeDef for Split.
auto get_split_nodedef = [](DataType dtype, int num_split) -> NodeDef {
  Scope s = Scope::NewRootScope();
  auto axis = ops::Placeholder(s.WithOpName("axis"), DT_INT32);
  auto value = ops::Placeholder(s.WithOpName("value"), dtype);
  auto split = ops::Split(s.WithOpName("my_split"), axis, value, num_split);
  return split.operation.node()->def();
};

template <DataType dtype>
void TestConvertSplit(OpConverterTest* test) {
  typedef typename EnumToDataType<dtype>::Type CType;

  struct TestParams {
    std::vector<int> input_shape;
    std::vector<CType> value;
    int axis;
    int num_split;
    std::vector<int> expected_output_dims;
    std::vector<std::vector<CType>> expected_outputs;
  };

  const std::vector<CType> common_input = InitTestVector<CType>(6);
  std::vector<TestParams> ok_params = {
      // Identity (num_split = 1)
      {/*input_shape=*/{1, 2, 3}, /*value=*/common_input, /*axis=*/1,
       /*num_split=*/1, /*expected_output_dims=*/{1, 2, 3},
       /*expected_outputs=*/{InitTestVector<CType>(6)}},
      {/*input_shape=*/{1, 2, 3},
       /*value=*/common_input,
       /*axis=*/3,
       /*num_split=*/3,
       /*expected_output_dims=*/{1, 2, 1},
       /*expected_outputs=*/
       {{CType(0), CType(3)}, {CType(1), CType(4)}, {CType(2), CType(5)}}},
      {/*input_shape=*/{1, 6},
       /*value=*/common_input,
       /*axis=*/2,
       /*num_split=*/6,
       /*expected_output_dims=*/{1, 1},
       /*expected_outputs=*/
       {{CType(0)},
        {CType(1)},
        {CType(2)},
        {CType(3)},
        {CType(4)},
        {CType(5)}}},
      {/*input_shape=*/{1, 6},
       /*value=*/common_input,
       /*axis=*/-1,
       /*num_split=*/2,
       /*expected_output_dims=*/{1, 3},
       /*expected_outputs=*/
       {InitTestVector<CType>(3), InitTestVector<CType>(3, CType(3))}},
  };

  for (int i = 0; i < ok_params.size(); ++i) {
    test->Reset();
    NodeDef node_def = get_split_nodedef(dtype, ok_params[i].num_split);
    // Create inputs.
    test->AddTestWeights<int32>("axis", {1}, {ok_params[i].axis});
    test->AddTestTensor("value", ok_params[i].input_shape, 1,
                        TfDataTypeToTrt(dtype));
    // Convert.
    test->RunValidationAndConversion(node_def);

    // Get output tensors and verify output dims.
    EXPECT_EQ(ok_params[i].expected_outputs.size(), ok_params[i].num_split);
    std::vector<TRT_TensorOrWeights> outputs(ok_params[i].num_split);
    DataVec output_data;
    for (int j = 0; j < outputs.size(); ++j) {
      const string name = j == 0 ? StrCat("my_split") : StrCat("my_split:", j);
      TF_EXPECT_OK(test->GetTensorOrWeights(name, &outputs[j]));
      EXPECT_TRUE(outputs[j].is_tensor());
      ExpectTrtDimsEqualsArray(ok_params[i].expected_output_dims,
                               outputs[j].tensor()->getDimensions());
      // Create buffer to store output.
      output_data.push_back(
          {name, test->ConstructTensor<CType>(
                     ok_params[i].expected_outputs[j].size())});
    }

    // Verify output values are correct.
    const DataVec input_data{
        {"value", test->AsTensor<CType>(ok_params[i].value)}};
    TF_EXPECT_OK(test->BuildAndRun(input_data, &output_data));
    for (int j = 0; j < outputs.size(); ++j) {
      EXPECT_THAT(GetSpanForData<CType>(output_data[j]),
                  ElementsAreArray(ok_params[i].expected_outputs[j]));
    }
  }
}

TEST_F(OpConverterTest, ConvertSplit) {
  {
    // Axis is a tensor, should fail.
    Reset();
    NodeDef node_def = get_split_nodedef(DT_FLOAT, 1);
    AddTestTensor("axis", {1});
    AddTestTensor("value", {1, 2, 3});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"axis\" for Split must be a constant, at my_split");
  }
  {
    // Axis is out of bounds, should fail.
    Reset();
    NodeDef node_def = get_split_nodedef(DT_FLOAT, 1);
    AddTestWeights<int32>("axis", {1}, {4});
    AddTestTensor("value", {1, 2, 3});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "Axis value of 4 is out of bounds, must be in "
                               "range [-4, 4), at my_split");
  }
  {
    // Axis is out of bounds (negative), should fail.
    Reset();
    NodeDef node_def = get_split_nodedef(DT_FLOAT, 1);
    AddTestWeights<int32>("axis", {1}, {-5});
    AddTestTensor("value", {1, 2, 3});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "Axis value of -5 is out of bounds, must be in "
                               "range [-4, 4), at my_split");
  }
  {
    // Axis is batch dimension, should fail.
    Reset();
    NodeDef node_def = get_split_nodedef(DT_FLOAT, 1);
    AddTestWeights<int32>("axis", {1}, {0});
    AddTestTensor("value", {1, 2, 3});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "TensorRT does not allow manipulation of the "
                               "batch dimension, at my_split");
  }
  {
    // Value is a weight, should fail.
    Reset();
    NodeDef node_def = get_split_nodedef(DT_FLOAT, 1);
    AddTestWeights<int32>("axis", {1}, {1});
    AddTestWeights<float>("value", {1, 2, 3}, {1, 2, 3, 4, 5, 6});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"value\" for Split must be a tensor, at my_split");
  }
  {
    // Dim is not evenly divisibly by num_split, should fail.
    Reset();
    NodeDef node_def = get_split_nodedef(DT_FLOAT, 2);
    AddTestWeights<int32>("axis", {1}, {3});
    AddTestTensor("value", {1, 2, 3});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Dimension 3 of size 3 is not evenly divisble by 2, at my_split");
  }
  {
    // num_split > dim size, should fail.
    Reset();
    NodeDef node_def = get_split_nodedef(DT_FLOAT, 4);
    AddTestWeights<int32>("axis", {1}, {3});
    AddTestTensor("value", {1, 2, 3});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Dimension 3 of size 3 is not evenly divisble by 4, at my_split");
  }

  TestConvertSplit<DT_FLOAT>(this);
  TestConvertSplit<DT_HALF>(this);
#if IS_TRT_VERSION_GE(5, 1, 3, 1)
  TestConvertSplit<DT_INT32>(this);
#endif
}

// Get the NodeDef for Unpack (Unstack in TF API).
auto get_unpack_nodedef = [](DataType dtype, int num, int axis) -> NodeDef {
  Scope s = Scope::NewRootScope();
  auto value = ops::Placeholder(s.WithOpName("value"), dtype);
  auto unstack_attrs = ops::Unstack::Axis(axis);
  auto unstack =
      ops::Unstack(s.WithOpName("my_unpack"), value, num, unstack_attrs);
  return unstack.operation.node()->def();
};

template <DataType dtype>
void TestConvertUnpack(OpConverterTest* test) {
  typedef typename EnumToDataType<dtype>::Type CType;

  struct TestParams {
    std::vector<int> input_shape;
    std::vector<CType> value;
    int axis;
    int num;
    std::vector<int> expected_output_dims;
    std::vector<std::vector<CType>> expected_outputs;
  };

  const std::vector<CType> common_input = InitTestVector<CType>(6);
  std::vector<TestParams> ok_params = {
      {/*input_shape=*/{1, 2, 3}, /*value=*/common_input, /*axis=*/1,
       /*num=*/1, /*expected_output_dims=*/{2, 3},
       /*expected_outputs=*/{InitTestVector<CType>(6)}},
      {/*input_shape=*/{1, 2, 3},
       /*value=*/common_input,
       /*axis=*/3,
       /*num=*/3,
       /*expected_output_dims=*/{1, 2},
       /*expected_outputs=*/
       {{CType(0), CType(3)}, {CType(1), CType(4)}, {CType(2), CType(5)}}},
      {/*input_shape=*/{6, 1},
       /*value=*/common_input,
       /*axis=*/-2,
       /*num=*/6,
       /*expected_output_dims=*/{1},
       /*expected_outputs=*/
       {{CType(0)},
        {CType(1)},
        {CType(2)},
        {CType(3)},
        {CType(4)},
        {CType(5)}}},
      {/*input_shape=*/{6},
       /*value=*/common_input,
       /*axis=*/1,
       /*num=*/6,
       /*expected_output_dims=*/{},
       /*expected_outputs=*/
       {{CType(0)},
        {CType(1)},
        {CType(2)},
        {CType(3)},
        {CType(4)},
        {CType(5)}}},
  };

  for (int i = 0; i < ok_params.size(); ++i) {
    test->Reset();
    NodeDef node_def =
        get_unpack_nodedef(dtype, ok_params[i].num, ok_params[i].axis);
    // Create inputs.
    test->AddTestTensor("value", ok_params[i].input_shape, 1,
                        TfDataTypeToTrt(dtype));
    // Convert.
    test->RunValidationAndConversion(node_def);

    // Get output tensors and verify output dims.
    EXPECT_EQ(ok_params[i].expected_outputs.size(), ok_params[i].num);
    std::vector<TRT_TensorOrWeights> outputs(ok_params[i].num);
    DataVec output_data;
    for (int j = 0; j < outputs.size(); ++j) {
      const string name = j == 0 ? "my_unpack" : StrCat("my_unpack:", j);
      TF_EXPECT_OK(test->GetTensorOrWeights(name, &outputs[j]));
      EXPECT_TRUE(outputs[j].is_tensor());
      ExpectTrtDimsEqualsArray(ok_params[i].expected_output_dims,
                               outputs[j].tensor()->getDimensions());
      // Create buffer to store output.
      output_data.push_back(
          {name, test->ConstructTensor<CType>(
                     ok_params[i].expected_outputs[j].size())});
    }

    // Verify output values are correct.
    const DataVec input_data{
        {"value", test->AsTensor<CType>(ok_params[i].value)}};
    TF_EXPECT_OK(test->BuildAndRun(input_data, &output_data));
    for (int j = 0; j < outputs.size(); ++j) {
      EXPECT_THAT(GetSpanForData<CType>(output_data[j]),
                  ElementsAreArray(ok_params[i].expected_outputs[j]));
    }
  }
}

TEST_F(OpConverterTest, ConvertUnpack) {
  {
    // Value is weights, should fail.
    Reset();
    NodeDef node_def = get_unpack_nodedef(DT_FLOAT, /*num=*/3, /*axis=*/3);
    AddTestWeights<float>("value", {1, 2, 3}, {1, 2, 3, 4, 5, 6});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"value\" for Unpack must be a tensor, at my_unpack");
  }
  {
    // Axis is out of bounds, should fail.
    Reset();
    NodeDef node_def = get_unpack_nodedef(DT_FLOAT, /*num=*/1, /*axis=*/4);
    AddTestTensor("value", {1, 2, 3});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "Axis value of 4 is out of bounds, must be in "
                               "range [-4, 4), at my_unpack");
  }
  {
    // Axis is out of bounds (negative), should fail.
    Reset();
    NodeDef node_def = get_unpack_nodedef(DT_FLOAT, /*num=*/1, /*axis=*/-5);
    AddTestTensor("value", {1, 2, 3});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "Axis value of -5 is out of bounds, must be in "
                               "range [-4, 4), at my_unpack");
  }
  {
    // Axis is batch dimension, should fail.
    Reset();
    NodeDef node_def = get_unpack_nodedef(DT_FLOAT, /*num=*/1, /*axis=*/0);
    AddTestTensor("value", {1, 2, 3});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "TensorRT does not allow manipulation of the "
                               "batch dimension, at my_unpack");
  }
  {
    // Dim size does not match num, should fail.
    Reset();
    NodeDef node_def = get_unpack_nodedef(DT_FLOAT, /*num=*/5, /*axis=*/2);
    AddTestTensor("value", {1, 6});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Dimension 2 has size 6 which is not equal to num of 5, at my_unpack");
  }
  {
    // Output would be TF scalar, should fail.
    Reset();
    NodeDef node_def = get_unpack_nodedef(DT_FLOAT, /*num=*/1, /*axis=*/0);
    AddTestTensor("value", {});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "Input \"value\" for Unpack must be rank 2 or greater, at my_unpack");
  }

  TestConvertUnpack<DT_FLOAT>(this);
  TestConvertUnpack<DT_HALF>(this);
#if IS_TRT_VERSION_GE(5, 1, 3, 1)
  TestConvertUnpack<DT_INT32>(this);
#endif
}

// Get the NodeDef for Pack.
NodeDef GetPackNodeDef(DataType dtype, int num_inputs, int axis) {
  Scope s = Scope::NewRootScope();
  std::vector<Input> values;
  for (int i = 0; i < num_inputs; ++i) {
    const string input_name = StrCat("values_", i);
    values.push_back(ops::Placeholder(s.WithOpName(input_name), dtype));
  }
  // Pack op is renamed to Stack in APIs.
  auto pack =
      ops::Stack(s.WithOpName("my_pack"), absl::Span<const Input>(values),
                 ops::Stack::Axis(axis));
  return pack.operation.node()->def();
}

template <DataType dtype>
void TestConvertPack(OpConverterTest* test) {
  typedef typename EnumToDataType<dtype>::Type CType;

  struct TestParams {
    std::vector<std::vector<int>> input_shapes;
    std::vector<std::vector<CType>> input_values;
    int axis;
    std::vector<int> expected_output_dims;
    std::vector<CType> expected_output;
  };

  const std::vector<std::vector<CType>> common_input{
      InitTestVector<CType>(6),
      InitTestVector<CType>(6, /*start_value=*/CType(6))};
  std::vector<TestParams> params = {
      {
          /*input_shapes=*/{{2, 3}, {2, 3}},
          /*input_values=*/common_input,
          /*axis=*/1,
          /*expected_output_dims=*/{2, 2, 3},
          /*expected_output=*/InitTestVector<CType>(12),
      },
      {
          /*input_shapes=*/{{2, 3}, {2, 3}},
          /*input_values=*/common_input,
          /*axis=*/2,
          /*expected_output_dims=*/{2, 2, 3},
          /*expected_output=*/
          {CType(0), CType(1), CType(2), CType(6), CType(7), CType(8), CType(3),
           CType(4), CType(5), CType(9), CType(10), CType(11)},
      },
      {
          /*input_shapes=*/{{2, 3}, {2, 3}},
          /*input_values=*/common_input,
          /*axis=*/3,
          /*expected_output_dims=*/{2, 3, 2},
          /*expected_output=*/
          {CType(0), CType(6), CType(1), CType(7), CType(2), CType(8), CType(3),
           CType(9), CType(4), CType(10), CType(5), CType(11)},
      },
      {
          /*input_shapes=*/{{2, 3}},
          /*input_values=*/{InitTestVector<CType>(6)},
          /*axis=*/1,
          /*expected_output_dims=*/{1, 2, 3},
          /*expected_output=*/InitTestVector<CType>(6),
      },
      {
          /*input_shapes=*/{{2, 3}},
          /*input_values=*/{InitTestVector<CType>(6)},
          /*axis=*/2,
          /*expected_output_dims=*/{2, 1, 3},
          /*expected_output=*/InitTestVector<CType>(6),
      },
  };

  for (int i = 0; i < params.size(); ++i) {
    test->Reset();
    const int num_inputs = params[i].input_shapes.size();
    EXPECT_EQ(num_inputs, params[i].input_values.size());

    NodeDef node_def = GetPackNodeDef(dtype, num_inputs, params[i].axis);
    // Create inputs.
    for (int j = 0; j < num_inputs; ++j) {
      test->AddTestTensor(StrCat("values_", j), params[i].input_shapes[j], 1,
                          TfDataTypeToTrt(dtype));
    }
    test->RunValidationAndConversion(node_def);

    TRT_TensorOrWeights output;
    TF_EXPECT_OK(test->GetTensorOrWeights("my_pack", &output));
    EXPECT_TRUE(output.is_tensor());
    ExpectTrtDimsEqualsArray(params[i].expected_output_dims,
                             output.tensor()->getDimensions());
    // Create input data for tensors.
    DataVec input_data;
    for (int j = 0; j < num_inputs; ++j) {
      input_data.push_back({StrCat("values_", j),
                            test->AsTensor<CType>(params[i].input_values[j])});
    }
    DataVec output_data{{"my_pack", test->ConstructTensor<CType>(
                                        params[i].expected_output.size())}};
    TF_EXPECT_OK(test->BuildAndRun(input_data, &output_data));
    EXPECT_THAT(GetSpanForData<CType>(output_data[0]),
                ElementsAreArray(params[i].expected_output));
  }
}

TEST_F(OpConverterTest, ConvertPack) {
  {
    // An input is a weight, should fail.
    Reset();
    NodeDef node_def = GetPackNodeDef(DT_FLOAT, 2, /*axis=*/1);
    AddTestTensor("values_0", {1, 2, 3});
    AddTestWeights<float>("values_1", {1, 2, 3}, {1, 2, 3, 4, 5, 6});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"values_1\" for Pack must be a tensor, at my_pack");
  }
  {
    // Axis is out of bounds, should fail.
    Reset();
    NodeDef node_def = GetPackNodeDef(DT_FLOAT, 2, /*axis=*/-5);
    AddTestTensor("values_0", {2, 3});
    AddTestTensor("values_1", {2, 3});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "Axis value of -5 is out of bounds, must be in "
                               "range [-4, 4), at my_pack");
  }
  {
    // Axis is batch dimension, should fail.
    Reset();
    NodeDef node_def = GetPackNodeDef(DT_FLOAT, 2, /*axis=*/-4);
    AddTestTensor("values_0", {2, 3});
    AddTestTensor("values_1", {2, 3});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "TensorRT does not allow manipulation of the "
                               "batch dimension, at my_pack");
  }
  {
    // Inputs have inconsistent rank, should fail.
    Reset();
    NodeDef node_def = GetPackNodeDef(DT_FLOAT, 2, /*axis=*/1);
    AddTestTensor("values_0", {1, 2, 3});
    AddTestTensor("values_1", {1, 6});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Received inputs with inconsistent rank, at my_pack");
  }
  {
    // Inputs have inconsistent shapes, should fail.
    Reset();
    NodeDef node_def = GetPackNodeDef(DT_FLOAT, 2, /*axis=*/1);
    AddTestTensor("values_0", {1, 2});
    AddTestTensor("values_1", {2, 2});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Received inputs with inconsistent shape, at my_pack");
  }

  TestConvertPack<DT_FLOAT>(this);
  TestConvertPack<DT_HALF>(this);

  // TODO(hinsu): Enable INT32 with TensorRT version 5.1.3 after testing.
  // TestConvertPack<DT_INT32>(this);
}

// Get the NodeDef for ArgMin or ArgMax.
template <typename OpType>
NodeDef GetArgMinMaxNodeDef(DataType input_dtype, DataType output_dtype) {
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), input_dtype);
  auto dimension = ops::Placeholder(s.WithOpName("dimension"), DT_INT32);
  auto attrs = OpType::OutputType(output_dtype);
  auto arg = OpType(s.WithOpName("my_arg"), input, dimension, attrs);
  return arg.operation.node()->def();
}

template <typename OpType, DataType dtype>
void TestConvertArgMinMax(OpConverterTest* test) {
  typedef typename EnumToDataType<dtype>::Type CType;

  struct TestParams {
    std::vector<int> input_shape;
    std::vector<CType> input_value;
    int axis;
    std::vector<int> expected_output_dims;
    std::vector<int> expected_argmax_output;
    std::vector<int> expected_argmin_output;
  };

  const std::vector<CType> common_input = InitTestVector<CType>(6);
  std::vector<TestParams> params = {
      {
          /*input_shape=*/{2, 3},
          /*input_value=*/common_input,
          /*axis=*/2,
          /*expected_output_dims=*/{2},
          /*expected_argmax_output=*/{2, 2},
          /*expected_argmin_output=*/{0, 0},
      },
      {
          /*input_shape=*/{2, 3},
          /*input_value=*/common_input,
          /*axis=*/-2,
          /*expected_output_dims=*/{3},
          /*expected_argmax_output=*/{1, 1, 1},
          /*expected_argmin_output=*/{0, 0, 0},
      },
      {
          /*input_shape=*/{6},
          /*input_value=*/common_input,
          /*axis=*/1,
          /*expected_output_dims=*/{},
          /*expected_argmax_output=*/{5},
          /*expected_argmin_output=*/{0},
      },
      {
          /*input_shape=*/{10},
          /*input_value=*/
          {CType(-5), CType(3), CType(5), CType(1), CType(6), CType(-9),
           CType(7), CType(1), CType(0), CType(-1)},
          /*axis=*/-1,
          /*expected_output_dims=*/{},
          /*expected_argmax_output=*/{6},
          /*expected_argmin_output=*/{5},
      },
  };

  for (int i = 0; i < params.size(); ++i) {
    test->Reset();

    NodeDef node_def = GetArgMinMaxNodeDef<OpType>(dtype, DT_INT32);
    // Create inputs.
    test->AddTestTensor("input", params[i].input_shape, /*batch_size=*/1,
                        /*trt_dtype=*/TfDataTypeToTrt(dtype));
    test->AddTestWeights<int32>("dimension", {1}, {params[i].axis});
    test->RunValidationAndConversion(node_def);

    TRT_TensorOrWeights output;
    TF_EXPECT_OK(test->GetTensorOrWeights("my_arg", &output));
    EXPECT_TRUE(output.is_tensor());
    ExpectTrtDimsEqualsArray(params[i].expected_output_dims,
                             output.tensor()->getDimensions());
    // Create input data for tensors.
    const DataVec input_data{
        {"input", test->AsTensor<CType>(params[i].input_value)}};
    DataVec output_data{
        {"my_arg", test->ConstructTensor<int32>(
                       params[i].expected_argmax_output.size())}};
    TF_EXPECT_OK(test->BuildAndRun(input_data, &output_data));

    if (node_def.op() == "ArgMax") {
      EXPECT_THAT(GetSpanForData<int32>(output_data[0]),
                  ElementsAreArray(params[i].expected_argmax_output));
    } else if (node_def.op() == "ArgMin") {
      EXPECT_THAT(GetSpanForData<int32>(output_data[0]),
                  ElementsAreArray(params[i].expected_argmin_output));
    } else {
      ASSERT_TRUE(false);
    }
  }
}

TEST_F(OpConverterTest, ConvertArgMinMax) {
  {
    // Dimension is a tensor, should fail.
    Reset();
    NodeDef node_def = GetArgMinMaxNodeDef<ops::ArgMax>(DT_FLOAT, DT_INT32);
    AddTestTensor("input", {1, 2, 3});
    AddTestTensor("dimension", {1});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"dimension\" for ArgMax must be a constant, at my_arg");
  }
  {
    // Output type is INT64, should fail.
    Reset();
    NodeDef node_def = GetArgMinMaxNodeDef<ops::ArgMax>(DT_FLOAT, DT_INT64);
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<int32>("dimension", {1}, {3});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "Output type int64 is not supported, at my_arg");
  }
  {
    // Axis is batch dimension, should fail
    Reset();
    NodeDef node_def = GetArgMinMaxNodeDef<ops::ArgMax>(DT_FLOAT, DT_INT32);
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<int32>("dimension", {1}, {0});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "TensorRT does not allow manipulation of the batch dimension, at "
        "my_arg");
  }

  TestConvertArgMinMax<ops::ArgMin, DT_FLOAT>(this);
  TestConvertArgMinMax<ops::ArgMax, DT_FLOAT>(this);
  TestConvertArgMinMax<ops::ArgMin, DT_HALF>(this);
  TestConvertArgMinMax<ops::ArgMax, DT_HALF>(this);
  // TRT does not support int32 for TopK layer which is used to implement ArgMin
  // and ArgMax.
  // TestConvertArgMinMax<ops::ArgMin, DT_INT32>(this);
  // TestConvertArgMinMax<ops::ArgMax, DT_INT32>(this);
}

// Get the NodeDef for DepthToSpace or SpaceToSpace.
template <typename OpType>
NodeDef GetDepthSpaceShuffleNodeDef(DataType dtype, int block_size,
                                    string data_format) {
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), dtype);
  auto attrs = OpType::DataFormat(data_format);
  auto shuffle = OpType(s.WithOpName("my_shuffle"), input, block_size, attrs);
  return shuffle.operation.node()->def();
}

template <typename CType>
struct DepthSpaceShuffleTestParams {
  std::vector<int> input_dims;
  std::vector<CType> input_value;
  int block_size;
  string data_format;
  std::vector<int> expected_output_dims;
  std::vector<CType> expected_output;
};

template <typename OpType, DataType dtype, typename CType>
void TestConvertDepthSpaceShuffle(
    OpConverterTest* test,
    const std::vector<DepthSpaceShuffleTestParams<CType>>& params) {
  for (int i = 0; i < params.size(); ++i) {
    test->Reset();

    NodeDef node_def = GetDepthSpaceShuffleNodeDef<OpType>(
        dtype, params[i].block_size, params[i].data_format);
    test->AddTestTensor("input", params[i].input_dims, 1,
                        TfDataTypeToTrt(dtype));
    test->RunValidationAndConversion(node_def);

    TRT_TensorOrWeights output;
    TF_EXPECT_OK(test->GetTensorOrWeights("my_shuffle", &output));
    EXPECT_TRUE(output.is_tensor());
    ExpectTrtDimsEqualsArray(params[i].expected_output_dims,
                             output.tensor()->getDimensions());

    DataVec input_data{{"input", test->AsTensor<CType>(params[i].input_value)}};
    DataVec output_data{{"my_shuffle", test->ConstructTensor<CType>(
                                           params[i].expected_output.size())}};
    TF_EXPECT_OK(test->BuildAndRun(input_data, &output_data));
    EXPECT_THAT(GetSpanForData<CType>(output_data[0]),
                ElementsAreArray(params[i].expected_output));
  }
}

template <DataType dtype>
void TestConvertDepthToSpace(OpConverterTest* test) {
  typedef typename EnumToDataType<dtype>::Type CType;
  const std::vector<CType> common_input = InitTestVector<CType>(16);
  std::vector<DepthSpaceShuffleTestParams<CType>> params = {
      {
          /*input_shape=*/{4, 2, 2},
          /*input_value=*/common_input,
          /*block_size=*/2,
          /*data_format=*/"NCHW",
          /*expected_output_dims=*/{1, 4, 4},
          /*expected_output=*/
          CastTestVector<int, CType>(
              {0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15}),
      },
      {
          /*input_shape=*/{2, 2, 4},
          /*input_value=*/common_input,
          /*block_size=*/2,
          /*data_format=*/"NHWC",
          /*expected_output_dims=*/{4, 4, 1},
          /*expected_output=*/
          CastTestVector<int, CType>(
              {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15}),
      },
      {
          /*input_shape=*/{16, 1, 1},
          /*input_value=*/common_input,
          /*block_size=*/4,
          /*data_format=*/"NCHW",
          /*expected_output_dims=*/{1, 4, 4},
          /*expected_output=*/InitTestVector<CType>(16),
      },
      {
          /*input_shape=*/{2, 2, 8},
          /*input_value=*/InitTestVector<CType>(32),
          /*block_size=*/2,
          /*data_format=*/"NHWC",
          /*expected_output_dims=*/{4, 4, 2},
          /*expected_output=*/CastTestVector<int, CType>({0,  1,  2,  3,  8,
                                                          9,  10, 11, 4,  5,
                                                          6,  7,  12, 13, 14,
                                                          15, 16, 17, 18, 19,
                                                          24, 25, 26, 27, 20,
                                                          21, 22, 23, 28, 29,
                                                          30, 31}),
      },
  };

  TestConvertDepthSpaceShuffle<ops::DepthToSpace, dtype, CType>(test, params);
}

TEST_F(OpConverterTest, ConvertDepthToSpace) {
  {
    // Input is a weight, should fail.
    Reset();
    NodeDef node_def =
        GetDepthSpaceShuffleNodeDef<ops::DepthToSpace>(DT_FLOAT, 2, "NCHW");
    AddTestWeights<float>("input", {4, 1, 1}, {1, 2, 3, 4});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "The input \"input\" for DepthToSpace must be a "
                               "tensor, at my_shuffle");
  }
  {
    // Input rank != 4
    Reset();
    NodeDef node_def =
        GetDepthSpaceShuffleNodeDef<ops::DepthToSpace>(DT_FLOAT, 2, "NCHW");
    AddTestTensor("input", {16, 32});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "The input to DepthToSpace must be rank 4, at my_shuffle");
  }
  {
    // Channels not divisible by block_size, should fail.
    Reset();
    NodeDef node_def =
        GetDepthSpaceShuffleNodeDef<ops::DepthToSpace>(DT_FLOAT, 3, "NCHW");
    AddTestTensor("input", {16, 32, 32});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "Number of channels must be divisible by "
                               "block_size*block_size, at my_shuffle");
  }
  {
    // Unsupported format, should fail.
    Reset();
    NodeDef node_def = GetDepthSpaceShuffleNodeDef<ops::DepthToSpace>(
        DT_FLOAT, 2, "NCHW_VECT_C");
    AddTestTensor("input", {16, 32, 32});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "Data format NCHW_VECT_C is not supported, at my_shuffle");
  }

  TestConvertDepthToSpace<DT_FLOAT>(this);
  TestConvertDepthToSpace<DT_HALF>(this);
  TestConvertDepthToSpace<DT_INT32>(this);
}

template <DataType dtype>
void TestConvertSpaceToDepth(OpConverterTest* test) {
  typedef typename EnumToDataType<dtype>::Type CType;
  const std::vector<CType> common_input = InitTestVector<CType>(16);
  std::vector<DepthSpaceShuffleTestParams<CType>> params = {
      {
          /*input_shape=*/{1, 4, 4},
          /*input_value=*/common_input,
          /*block_size=*/2,
          /*data_format=*/"NCHW",
          /*expected_output_dims=*/{4, 2, 2},
          /*expected_output=*/
          CastTestVector<int, CType>(
              {0, 2, 8, 10, 1, 3, 9, 11, 4, 6, 12, 14, 5, 7, 13, 15}),
      },
      {
          /*input_shape=*/{4, 4, 1},
          /*input_value=*/common_input,
          /*block_size=*/2,
          /*data_format=*/"NHWC",
          /*expected_output_dims=*/{2, 2, 4},
          /*expected_output=*/
          CastTestVector<int, CType>(
              {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15}),
      },
      {
          /*input_shape=*/{1, 4, 4},
          /*input_value=*/common_input,
          /*block_size=*/4,
          /*data_format=*/"NCHW",
          /*expected_output_dims=*/{16, 1, 1},
          /*expected_output=*/InitTestVector<CType>(16),
      },
      {
          /*input_shape=*/{4, 4, 2},
          /*input_value=*/InitTestVector<CType>(32),
          /*block_size=*/2,
          /*data_format=*/"NHWC",
          /*expected_output_dims=*/{2, 2, 8},
          /*expected_output=*/CastTestVector<int, CType>({0,  1,  2,  3,  8,
                                                          9,  10, 11, 4,  5,
                                                          6,  7,  12, 13, 14,
                                                          15, 16, 17, 18, 19,
                                                          24, 25, 26, 27, 20,
                                                          21, 22, 23, 28, 29,
                                                          30, 31}),
      },
  };

  TestConvertDepthSpaceShuffle<ops::SpaceToDepth, dtype, CType>(test, params);
}

TEST_F(OpConverterTest, ConvertSpaceToDepth) {
  {
    // Input is a weight, should fail.
    Reset();
    NodeDef node_def =
        GetDepthSpaceShuffleNodeDef<ops::SpaceToDepth>(DT_FLOAT, 2, "NCHW");
    AddTestWeights<float>("input", {4, 1, 1}, {1, 2, 3, 4});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "The input \"input\" for SpaceToDepth must be a "
                               "tensor, at my_shuffle");
  }
  {
    // Input rank != 4
    Reset();
    NodeDef node_def =
        GetDepthSpaceShuffleNodeDef<ops::SpaceToDepth>(DT_FLOAT, 2, "NCHW");
    AddTestTensor("input", {16, 32});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "The input to SpaceToDepth must be rank 4, at my_shuffle");
  }
  {
    // Width not divisble by block_size, should fail.
    Reset();
    NodeDef node_def =
        GetDepthSpaceShuffleNodeDef<ops::SpaceToDepth>(DT_FLOAT, 3, "NCHW");
    AddTestTensor("input", {16, 9, 32});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "Width and height must be divisible by "
                               "block_size, at my_shuffle");
  }
  {
    // Height not divisble by block_size, should fail.
    Reset();
    NodeDef node_def =
        GetDepthSpaceShuffleNodeDef<ops::SpaceToDepth>(DT_FLOAT, 3, "NCHW");
    AddTestTensor("input", {16, 32, 9});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "Width and height must be divisible by "
                               "block_size, at my_shuffle");
  }
  {
    // Unsupported format, should fail.
    Reset();
    NodeDef node_def = GetDepthSpaceShuffleNodeDef<ops::SpaceToDepth>(
        DT_FLOAT, 2, "NCHW_VECT_C");
    AddTestTensor("input", {16, 32, 32});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "Data format NCHW_VECT_C is not supported, at my_shuffle");
  }

  TestConvertSpaceToDepth<DT_FLOAT>(this);
  TestConvertSpaceToDepth<DT_HALF>(this);
  TestConvertSpaceToDepth<DT_INT32>(this);
}

#if IS_TRT_VERSION_GE(5, 1, 2, 0)
// Get the NodeDef for ClipByValue.
NodeDef GetClipByValueNodeDef(DataType dtype) {
  Scope s = Scope::NewRootScope();
  auto t = ops::Placeholder(s.WithOpName("t"), dtype);
  auto clip_value_min = ops::Placeholder(s.WithOpName("clip_value_min"), dtype);
  auto clip_value_max = ops::Placeholder(s.WithOpName("clip_value_max"), dtype);
  auto clip = ops::ClipByValue(s.WithOpName("my_clip"), t, clip_value_min,
                               clip_value_max);
  return clip.operation.node()->def();
}

template <DataType dtype>
void TestConvertClipByValue(OpConverterTest* test) {
  typedef typename EnumToDataType<dtype>::Type CType;

  struct TestParams {
    std::vector<int> dims;
    std::vector<CType> input_value;
    CType clip_value_min;
    CType clip_value_max;
    std::vector<CType> expected_output;
  };

  const std::vector<CType> common_input = InitTestVector<CType>(6);
  std::vector<TestParams> params = {
      {
          /*dims=*/{1, 2, 3},
          /*input_value=*/common_input,
          /*clip_value_min=*/CType(2),
          /*clip_value_max=*/CType(5),
          /*expected_output=*/
          {CType(2), CType(2), CType(2), CType(3), CType(4), CType(5)},
      },
      {
          /*dims=*/{2, 1, 3},
          /*input_value=*/common_input,
          /*clip_value_min=*/CType(-1),
          /*clip_value_max=*/CType(8),
          /*expected_output=*/common_input,
      },
  };

  for (int i = 0; i < params.size(); ++i) {
    test->Reset();

    NodeDef node_def = GetClipByValueNodeDef(dtype);
    test->AddTestTensor("t", params[i].dims, 1, TfDataTypeToTrt(dtype));
    test->AddTestWeights<CType>("clip_value_min", {1},
                                {params[i].clip_value_min});
    test->AddTestWeights<CType>("clip_value_max", {1},
                                {params[i].clip_value_max});
    test->RunValidationAndConversion(node_def);

    TRT_TensorOrWeights output;
    TF_EXPECT_OK(test->GetTensorOrWeights("my_clip", &output));
    EXPECT_TRUE(output.is_tensor());
    ExpectTrtDimsEqualsArray(params[i].dims, output.tensor()->getDimensions());

    DataVec input_data{{"t", test->AsTensor<CType>(params[i].input_value)}};
    DataVec output_data{{"my_clip", test->ConstructTensor<CType>(
                                        params[i].expected_output.size())}};
    TF_EXPECT_OK(test->BuildAndRun(input_data, &output_data));
    EXPECT_THAT(GetSpanForData<CType>(output_data[0]),
                ElementsAreArray(params[i].expected_output));
  }
}

TEST_F(OpConverterTest, ConvertClipByValue) {
  {
    // Input is a weight, should fail.
    Reset();
    NodeDef node_def = GetClipByValueNodeDef(DT_FLOAT);
    AddTestWeights<float>("t", {1, 2, 3}, {1, 2, 3, 4, 5, 6});
    AddTestWeights<float>("clip_value_min", {1}, {1});
    AddTestWeights<float>("clip_value_max", {1}, {5});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "The input \"t\" for ClipByValue must be a "
                               "tensor, at my_clip");
  }
  {
    // Clip min is a tensor, should fail.
    Reset();
    NodeDef node_def = GetClipByValueNodeDef(DT_FLOAT);
    AddTestTensor("t", {1, 2, 3});
    AddTestTensor("clip_value_min", {1});
    AddTestWeights<float>("clip_value_max", {1}, {1});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "The input \"clip_value_min\" for ClipByValue "
                               "must be a constant, at my_clip");
  }
  {
    // Clip max is a tensor, should fail.
    Reset();
    NodeDef node_def = GetClipByValueNodeDef(DT_FLOAT);
    AddTestTensor("t", {1, 2, 3});
    AddTestWeights<float>("clip_value_min", {1}, {1});
    AddTestTensor("clip_value_max", {1});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "The input \"clip_value_max\" for ClipByValue "
                               "must be a constant, at my_clip");
  }

  TestConvertClipByValue<DT_FLOAT>(this);
  TestConvertClipByValue<DT_HALF>(this);
}
#endif  // IS_TRT_VERSION_GE(5, 1, 2, 0)

// Get the NodeDef for SquaredDifference.
NodeDef GetSquaredDifferenceNodeDef(DataType dtype) {
  Scope s = Scope::NewRootScope();
  auto x = ops::Placeholder(s.WithOpName("x"), dtype);
  auto y = ops::Placeholder(s.WithOpName("y"), dtype);
  auto squared_diff =
      ops::SquaredDifference(s.WithOpName("my_squared_diff"), x, y);
  return squared_diff.operation.node()->def();
}

template <DataType dtype>
void TestConvertSquaredDifference(OpConverterTest* test) {
  typedef typename EnumToDataType<dtype>::Type CType;

  struct TestParams {
    std::vector<int> dims_x;
    std::vector<int> dims_y;
    std::vector<CType> value_x;
    std::vector<CType> value_y;
    std::vector<int> expected_output_dims;
    std::vector<CType> expected_output;
  };

  const std::vector<CType> common_input = InitTestVector<CType>(6);
  std::vector<TestParams> params = {
      {
          /*dims_x=*/{1, 2, 3},
          /*dims_y=*/{1, 2, 3},
          /*value_x=*/common_input,
          /*value_y=*/CastTestVector<int, CType>({0, -1, 3, 0, 10, -7}),
          /*expected_output_dims=*/{1, 2, 3},
          /*expected_output=*/CastTestVector<int, CType>({0, 4, 1, 9, 36, 144}),
      },
      {
          /*dims_x=*/{1, 2, 3},
          /*dims_y=*/{1, 1, 3},
          /*value_x=*/common_input,
          /*value_y=*/CastTestVector<int, CType>({0, 1, 2}),
          /*expected_output_dims=*/{1, 2, 3},
          /*expected_output=*/CastTestVector<int, CType>({0, 0, 0, 9, 9, 9}),
      },
  };

  for (int i = 0; i < params.size(); ++i) {
    test->Reset();

    NodeDef node_def = GetSquaredDifferenceNodeDef(dtype);
    test->AddTestTensor("x", params[i].dims_x, 1, TfDataTypeToTrt(dtype));
    test->AddTestTensor("y", params[i].dims_y, 1, TfDataTypeToTrt(dtype));
    test->RunValidationAndConversion(node_def);

    TRT_TensorOrWeights output;
    TF_EXPECT_OK(test->GetTensorOrWeights("my_squared_diff", &output));
    EXPECT_TRUE(output.is_tensor());
    ExpectTrtDimsEqualsArray(params[i].expected_output_dims,
                             output.tensor()->getDimensions());

    DataVec input_data{{"x", test->AsTensor<CType>(params[i].value_x)},
                       {"y", test->AsTensor<CType>(params[i].value_y)}};
    DataVec output_data{
        {"my_squared_diff",
         test->ConstructTensor<CType>(params[i].expected_output.size())}};
    TF_EXPECT_OK(test->BuildAndRun(input_data, &output_data));
    EXPECT_THAT(GetSpanForData<CType>(output_data[0]),
                ElementsAreArray(params[i].expected_output));
  }
}

TEST_F(OpConverterTest, ConvertSquaredDifference) {
  {
    // Input is a weight, should fail.
    Reset();
    NodeDef node_def = GetSquaredDifferenceNodeDef(DT_FLOAT);
    AddTestWeights<float>("x", {1, 2, 3}, {1, 2, 3, 4, 5, 6});
    AddTestTensor("y", {1, 2, 3});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "The input \"x\" for SquaredDifference must be "
                               "a tensor, at my_squared_diff");
  }
  {
    // Shapes are not broadcastable, should fail.
    Reset();
    NodeDef node_def = GetSquaredDifferenceNodeDef(DT_FLOAT);
    AddTestTensor("x", {2, 3});
    AddTestTensor("y", {7, 5});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "Infeasible broadcast scheme");
  }

  TestConvertSquaredDifference<DT_FLOAT>(this);
  TestConvertSquaredDifference<DT_HALF>(this);
}

#if IS_TRT_VERSION_GE(6, 0, 0, 0)
template <typename OpType>
NodeDef MakeResizeNodeDef(std::string name, DataType dtype,
                          bool align_corners) {
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), dtype);
  auto size = ops::Placeholder(s.WithOpName("size"), DT_INT32);
  auto attrs = typename OpType::Attrs().AlignCorners(align_corners);
  auto resize = OpType(s.WithOpName(name), input, size, attrs);
  return resize.operation.node()->def();
}

template <typename CType>
struct ResizeTestParams {
  std::vector<int> input_dims;
  std::vector<int> output_resize_dims;
  std::vector<CType> input_values;
  bool align_corners;
  std::vector<int> expected_output_dims;
  std::vector<CType> expected_nearest_output_values;
  std::vector<CType> expected_bilinear_output_values;
};

template <typename OpType, DataType dtype>
void TestConvertResize(OpConverterTest* test) {
  typedef typename EnumToDataType<dtype>::Type CType;

  std::vector<ResizeTestParams<CType>> params{
      {
          /*input_dims=*/{1, 2, 1},       // H, W, C
          /*output_resize_dims=*/{2, 3},  // H_out, W_out
          /*input_values=*/CastTestVector<float, CType>({2.0f, -1.0f}),
          /*align_corners=*/false,
          /*expected_output_dims=*/{2, 3, 1},  // H, W, C
          /*expected_nearest_output_values=*/
          CastTestVector<float, CType>({2.0f, 2.0f, -1.0f, 2.0f, 2.0f, -1.0f}),
          /*expected_bilinear_output_values=*/
          CastTestVector<float, CType>({2.0f, 0.f, -1.0f, 2.0f, 0.f, -1.0f}),
      },
      {
          /*input_dims=*/{1, 2, 1},       // H, W, C
          /*output_resize_dims=*/{2, 3},  // H_out, W_out
          /*input_values=*/CastTestVector<float, CType>({2.0f, -1.0f}),
          /*align_corners=*/true,
          /*expected_output_dims=*/{2, 3, 1},  // H, W, C
          /*expected_nearest_output_values=*/
          CastTestVector<float, CType>({2.0f, 2.0f, -1.0f, 2.0f, 2.0f, -1.0f}),
          /*expected_bilinear_output_values=*/
          CastTestVector<float, CType>({2.0f, 0.5f, -1.0f, 2.0f, 0.5f, -1.0f}),
      }};

  for (int i = 0; i < params.size(); ++i) {
    test->Reset();
    // Create resize node.
    NodeDef node_def =
        MakeResizeNodeDef<OpType>("my_resize", dtype, params[i].align_corners);
    // Create input tensor
    test->AddTestTensor("input", params[i].input_dims, /*batch_size=*/1,
                        /*trt_dtype=*/TfDataTypeToTrt(dtype));
    // Create output size.
    test->AddTestWeights<int32>("size", {2}, params[i].output_resize_dims);

    test->RunValidationAndConversion(node_def);

    TRT_TensorOrWeights output;
    TF_EXPECT_OK(test->GetTensorOrWeights("my_resize", &output));

    // Create input data for tensors.
    const DataVec input_data{
        {"input", test->AsTensor<CType>(params[i].input_values)}};
    DataVec output_data{
        {"my_resize", test->ConstructTensor<CType>(
                          params[i].expected_nearest_output_values.size())}};

    TF_EXPECT_OK(test->BuildAndRun(input_data, &output_data));

    if (node_def.op() == "ResizeBilinear") {
      ExpectArrayAlmostEqual(params[i].expected_bilinear_output_values,
                             GetSpanForData<CType>(output_data[0]),
                             CType(1e-3));
    } else if (node_def.op() == "ResizeNearestNeighbor") {
      ExpectArrayAlmostEqual(params[i].expected_nearest_output_values,
                             GetSpanForData<CType>(output_data[0]),
                             CType(1e-3));
    }
  }
}

TEST_F(OpConverterTest, ConvertResize) {
  {
    // First input is weight, should fail.
    Reset();
    NodeDef node_def =
        MakeResizeNodeDef<ops::ResizeBilinear>("my_resize", DT_FLOAT, false);
    AddTestWeights<float>("input", {1, 2}, {1, 2});
    AddTestWeights<int>("size", {1, 2}, {1, 2});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"input\" for ResizeBilinear must be a "
        "tensor, at my_resize");
  }
  {
    // output dimension is a tensor, should fail.
    Reset();
    NodeDef node_def =
        MakeResizeNodeDef<ops::ResizeBilinear>("my_resize", DT_FLOAT, false);
    AddTestTensor("input", {1, 2});
    AddTestTensor("size", {1, 2});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"size\" for ResizeBilinear must be a "
        "constant, at my_resize");
  }
  TestConvertResize<ops::ResizeBilinear, DT_FLOAT>(this);
  TestConvertResize<ops::ResizeBilinear, DT_HALF>(this);
  TestConvertResize<ops::ResizeNearestNeighbor, DT_FLOAT>(this);
  TestConvertResize<ops::ResizeNearestNeighbor, DT_HALF>(this);
}
#endif  // IS_TRT_VERSION_GE(6, 0, 0, 0)

NodeDef MakePadNodeDef(std::string name, DataType dtype) {
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), dtype);
  auto padding = ops::Placeholder(s.WithOpName("padding"), DT_INT32);
  auto pad = ops::Pad(s.WithOpName(name), input, padding);
  return pad.operation.node()->def();
}

template <typename CType>
struct PadTestParams {
  std::vector<int> input_dims;
  std::vector<int> pad_dims;
  std::vector<CType> input_values;
  std::vector<int> expected_output_dims;
  std::vector<CType> expected_output_values;
};

template <DataType dtype>
void TestConvertPad(OpConverterTest* test) {
  typedef typename EnumToDataType<dtype>::Type CType;

  std::vector<PadTestParams<CType>> params{
      {
          /*input_dims=*/{1, 2, 1},  // H, W, C
          /*pad_dims=*/{4, 2},       // #dims, {pad_before, pad_after}
          /*input_values=*/CastTestVector<float, CType>({2.0f, -1.0f}),
          /*expected_output_dims=*/{2, 3, 1},  // H, W, C
          /*expected_output_values=*/
          CastTestVector<float, CType>({0.0, 0.0, 0.0, 2.0f, -1.0f, 0.0}),
      },
  };

  for (int i = 0; i < params.size(); ++i) {
    test->Reset();
    // Create pad node.
    NodeDef node_def = MakePadNodeDef("my_pad", dtype);
    // Create input tensor
    test->AddTestTensor("input", params[i].input_dims, /*batch_size=*/1,
                        /*trt_dtype=*/TfDataTypeToTrt(dtype));
    // Create output size.
    test->AddTestWeights<int32>("padding", params[i].pad_dims,
                                {0, 0, 1, 0, 0, 1, 0, 0});
    test->RunValidationAndConversion(node_def);

    TRT_TensorOrWeights output;
    TF_EXPECT_OK(test->GetTensorOrWeights("padding", &output));

    // Create input data for tensors.
    const DataVec input_data{
        {"input", test->AsTensor<CType>(params[i].input_values)}};
    DataVec output_data{
        {"my_pad", test->ConstructTensor<CType>(
                       params[i].expected_output_values.size())}};

    TF_EXPECT_OK(test->BuildAndRun(input_data, &output_data));
    ExpectArrayAlmostEqual(params[i].expected_output_values,
                           GetSpanForData<CType>(output_data[0]), CType(1e-5));
  }
}

TEST_F(OpConverterTest, ConvertPad) {
  {
    // First input is weight, should fail.
    Reset();
    NodeDef node_def = MakePadNodeDef("my_pad", DT_FLOAT);
    AddTestWeights<float>("input", {1, 2}, {1, 2});
    AddTestWeights<int>("padding", {1, 2}, {1, 2});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "The input \"tensor\" for Pad must be a "
                               "tensor");
  }
  {
    // padding is a tensor, should fail.
    Reset();
    NodeDef node_def = MakePadNodeDef("my_pad", DT_FLOAT);
    AddTestTensor("input", {1, 2});
    AddTestTensor("padding", {1, 2});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "The input \"paddings\" for Pad must be a "
                               "constant");
  }
  TestConvertPad<DT_FLOAT>(this);
  TestConvertPad<DT_HALF>(this);
  {
    // Make sure that ranges are inferred across a Pad.
    Reset();
    NodeDef node_def = MakePadNodeDef("my_pad", DT_FLOAT);
    AddTestTensor("input", {1, 2, 1});
    AddTestWeights<int>("padding", {4, 2}, {0, 0, 1, 0, 0, 1, 0, 0});
    TRT_TensorOrWeights input;
    TRT_TensorOrWeights output;
    RunValidationAndConversion(node_def);
    TF_EXPECT_OK(GetTensorOrWeights("input", &input));
    TF_EXPECT_OK(GetTensorOrWeights("my_pad", &output));
    converter_->ProvideQuantizationRange(input.tensor(), -5.0f, 5.0f);
    // Input range should be inferred across pad.
    PropagateQuantizationRanges();
    auto ranges = quantization_ranges();
    EXPECT_EQ(5.0f, ranges[input.tensor()]);
    EXPECT_EQ(5.0f, ranges[output.tensor()]);
  }
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
