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
#include <cmath>
#include <functional>
#include <iterator>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/tf2tensorrt/common/datavec.h"
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_engine_utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_testutils.h"
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
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"  // NOLINT
#include "tensorflow/core/public/session.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

// TensorRT modes for testing. We define the following three modes:
// 1. Implicit batch mode: The tensors have static (known) input shape and the
//    the batch dimension (first dim) is removed from the TRT tensor shape. In
//    a loose notation: trt_shape = tf_shape[1:].
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

string DebugString(const TrtTestMode mode) {
  switch (mode) {
    case TrtTestMode::kImplicitBatch:
      return "kImplicitBatch";
    case TrtTestMode::kExplicitBatch:
      return "kExplicitBatch";
    case TrtTestMode::kDynamicShape:
      return "kDynamicShape";
    default:
      return "Invalid TrtTestMode";
  }
}

namespace convert {

using absl::StrCat;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::Matcher;
using ::testing::PrintToString;

using ::tensorflow::testing::IsOk;
using ::tensorflow::testing::StatusIs;

constexpr std::array<TrtTestMode, 3> ValidTrtModes = {
    TrtTestMode::kImplicitBatch, TrtTestMode::kExplicitBatch,
    TrtTestMode::kDynamicShape};

bool TrtShapedWeightsEquals(const TRT_ShapedWeights& lhs,
                            const TRT_ShapedWeights& rhs) {
  return lhs.Shape() == rhs.Shape() && lhs.TrtDType() == rhs.TrtDType() &&
         lhs.GetPointer<int8>() == rhs.GetPointer<int8>();
}

template <typename T>
void ValidateWeights(const TRT_ShapedWeights& weights,
                     const std::vector<int>& expected_dims,
                     const std::vector<T>& expected_value) {
  EXPECT_EQ(weights.Shape(), DimsAdapter(expected_dims));
  ASSERT_EQ(expected_value.size(), weights.count()) << weights.DebugString();
  const T* actual_values = weights.GetPointer<T>();
  for (int i = 0; i < expected_value.size(); ++i) {
    EXPECT_EQ(expected_value[i], actual_values[i]);
  }
}

// TRT >= 8.2 optimizes memory management in the builder. When all builders
// are destroyed, it unloads many resources. This test fixture will create and
// destroy hundreds of builders when run sequentially for parameterized
// tests. We can hold open an IBuilder in order to prevent TRT from unloading
// shared resources between engine builds when using TRT shared library. This
// greatly speeds up unit tests and is safe to do.
void PreventUnloadBuilderResources() {
#if IS_TRT_VERSION_GE(8, 2, 0, 0)
  static thread_local absl::once_flag once;
  static TrtUniquePtrType<nvinfer1::IBuilder> hold_builder = nullptr;
  absl::call_once(
      once,
      [](TrtUniquePtrType<nvinfer1::IBuilder>& builder) {
        if (!builder) {
          builder.reset(nvinfer1::createInferBuilder(*Logger::GetLogger()));
        }
      },
      hold_builder);
#endif
}

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

      EXPECT_EQ(nullptr, ptr->GetPointer<int8>());
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

      EXPECT_EQ(nullptr, ptr->GetPointer<int8>());
      EXPECT_EQ(0, ptr->count());
      EXPECT_EQ(0, ptr->size_bytes());
    }
  }
  // Test constructor with DataType and nvinfer1::Dims arguments.
  {
    TrtWeightStore store;
    TRT_ShapedWeights weights =
        store.GetTempWeights(nvinfer1::DataType::kFLOAT, CreateDims({2, 5}))
            .ValueOrDie();
    TRT_ShapedWeights copy(weights);
    for (auto ptr : {&weights, &copy}) {
      nvinfer1::Weights trt_weights = ptr->GetTrtWeights();
      EXPECT_EQ(nvinfer1::DataType::kFLOAT, trt_weights.type);
      EXPECT_NE(nullptr, trt_weights.values);
      EXPECT_EQ(10, trt_weights.count);

      EXPECT_EQ(trt_weights.values, ptr->GetPointer<int8>());
      EXPECT_EQ(10, ptr->count());
      EXPECT_EQ(40, ptr->size_bytes());
    }
    // Test that it doesn't copy the underlying buffer.
    EXPECT_EQ(weights.GetPointer<int8>(), copy.GetPointer<int8>());
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
    ITensorProxyPtr itensor(dims);
    TRT_TensorOrWeights tw(itensor);
    TRT_TensorOrWeights tw1(itensor, /*batch_size=*/1);

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
        EXPECT_EQ(itensor->simple_tensor(), ptr->tensor()->simple_tensor());
        EXPECT_THAT(ptr->GetTrtDims(), DimsAreArray({1}));
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
      EXPECT_NE(nullptr, ptr->tensor()->simple_tensor());
      EXPECT_THAT(ptr->GetTrtDims(), DimsAreArray({1}));
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
      std::vector<int> empty_dims;
      EXPECT_THAT(ptr->GetTrtDims(), DimsAreArray(empty_dims));
    }
  }
}

class ValidatorTest : public ::testing::Test {
 public:
  ValidatorTest() { PreventUnloadBuilderResources(); }
  Status ConvertToTensorOrWeights(const Scope& scope, const Node* node,
                                  int output_port,
                                  TRT_TensorOrWeights* tensor_or_weights) {
    grappler::GrapplerItem item;
    TF_EXPECT_OK(scope.ToGraphDef(&item.graph));
    grappler::GraphProperties graph_properties(item);
    TF_EXPECT_OK(graph_properties.InferStatically(true));

    TrtNodeValidator validator(graph_properties, TrtPrecisionMode::FP32,
                               /*use_calibration=*/false,
                               /*use_implicit_batch=*/true,
                               /*use_explicit_precision=*/false);
    return validator.ConvertToTensorOrWeights(node->def(), output_port,
                                              tensor_or_weights);
  }
};

TEST_F(ValidatorTest, ConvertToTensorOrWeights) {
  // Convert Const.
  {
    Scope s = Scope::NewRootScope();
    auto node =
        ops::Const(s.WithOpName("my_const"), {1.0f, 2.0f}, TensorShape({2}));
    TRT_TensorOrWeights output;
    EXPECT_THAT(ConvertToTensorOrWeights(s, node.op().node(),
                                         /*output_port=*/0, &output),
                IsOk());
    ValidateWeights<float>(output.weights(), {2}, {1.0, 2.0});
  }

  // Helper method to run ConvertToTensorOrWeights() with predefined parameters.
  auto convert_to_tensor_or_weights = [this](const std::vector<int64_t>& dims,
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
    EXPECT_THAT(
        convert_to_tensor_or_weights(
            std::vector<int64_t>(nvinfer1::Dims::MAX_DIMS + 2, 1), &output),
        StatusIs(error::OUT_OF_RANGE,
                 HasSubstr("Input tensor rank is greater than 9")));
  }
  // Convert non-Const with #dims < 1.
  {
    TRT_TensorOrWeights output;
    EXPECT_THAT(convert_to_tensor_or_weights({}, &output),
                StatusIs(error::INVALID_ARGUMENT,
                         HasSubstr("Scalar input tensor is not supported since "
                                   "the first dimension "
                                   "is treated as batch dimension by TRT")));
  }
  // Convert non-Const. We test the case where the non-batch dimension is
  // unknown as well, to make sure the validator allows that.
  for (const int32 non_batch_dim : {-1, 2}) {
    const int32 batch_size = 12;
    TRT_TensorOrWeights output;
    EXPECT_THAT(
        convert_to_tensor_or_weights({batch_size, non_batch_dim}, &output),
        IsOk());
    ASSERT_TRUE(output.is_tensor());
    EXPECT_EQ(batch_size, output.batch_size());
    EXPECT_NE(nullptr, output.tensor()->simple_tensor());
    EXPECT_THAT(output.GetTrtDims(), DimsAreArray({non_batch_dim}));
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
                             /*use_implicit_batch=*/true,
                             /*use_explicit_precision=*/false);

  // Override the Add converter.
  bool start_conversion = false;
  bool should_fail = false;
  auto op_converter = [&start_conversion,
                       &should_fail](OpConverterParams* params) -> Status {
    if (should_fail) return errors::InvalidArgument("");
    if (!params->validation_only) start_conversion = true;
    return Status::OK();
  };

  // Validator not registered.
  auto original_op_converter = GetOpConverterRegistry()->LookUp("Add");
  ASSERT_TRUE(original_op_converter.ok());
  GetOpConverterRegistry()->Clear("Add");
  EXPECT_THAT(validator.IsTensorRTCandidate(add_node),
              StatusIs(error::UNIMPLEMENTED,
                       HasSubstr("Op type Add is not supported.")));
  GetOpConverterRegistry()->Register("Add", kDefaultConverterPriority + 1,
                                     op_converter);
  TF_EXPECT_OK(validator.IsTensorRTCandidate(add_node));
  EXPECT_EQ(false, start_conversion);

  // Let the converter return error.
  should_fail = true;
  EXPECT_THAT(validator.IsTensorRTCandidate(add_node),
              StatusIs(error::INVALID_ARGUMENT));
  GetOpConverterRegistry()->Clear("Add");
  GetOpConverterRegistry()->Register("Add", kDefaultConverterPriority,
                                     *original_op_converter);
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
  auto unsupported_op = ops::Erfc(s.WithOpName("sin"), feed);

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
                               /*use_implicit_batch=*/true,
                               /*use_explicit_precision=*/false);
    TF_EXPECT_OK(validator.IsTensorRTCandidate(matmul.operation.node()));
    EXPECT_THAT(
        validator.IsTensorRTCandidate(incompatible_matmul.operation.node()),
        StatusIs(error::INVALID_ARGUMENT,
                 HasSubstr("MatMul with 2D tensors requires explicit batch "
                           "mode, or that tensor A "
                           "is not transposed and B is a constant tensor.")));
    EXPECT_THAT(validator.IsTensorRTCandidate(unsupported_op.operation.node()),
                StatusIs(error::UNIMPLEMENTED,
                         HasSubstr("Op type Erfc is not supported")));
    EXPECT_THAT(validator.IsTensorRTCandidate(
                    matmul_with_incompatible_input.operation.node()),
                StatusIs(error::INTERNAL,
                         HasSubstr("Failed to convert at least one input to a "
                                   "TRT_TensorOrWeights:")));
    if (precision_mode == TrtPrecisionMode::INT8) {
      TF_EXPECT_OK(validator.IsTensorRTCandidate(quantize.operation.node()));
    } else {
      EXPECT_THAT(
          validator.IsTensorRTCandidate(quantize.operation.node()),
          StatusIs(
              error::UNIMPLEMENTED,
              HasSubstr("Op type FakeQuantWithMinMaxArgs is not supported")));
    }
  }
}

class ConverterTest : public ::testing::Test {
 public:
  ConverterTest() {
    PreventUnloadBuilderResources();
    Reset();
  }

  void Reset() {
    GetOpConverterRegistry()->Clear("MyOp");
    GetOpConverterRegistry()->Clear("DummyOp");
    converter_ =
        std::move(Converter::Create(TrtPrecisionMode::FP32,
                                    /*use_calibration=*/false, &logger_,
                                    /*use_implicit_batch=*/true,
                                    /*engine_name=*/"TRTEngineOp_000_000",
                                    /*use_explicit_precision=*/false)
                      .ValueOrDie());
    weight_store_ = &converter_->weight_store_;
  }

  // TODO(cbate): These should be removed or changed to public per black-box
  // testing principle.
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

  int batch_size() const { return converter_->batch_size_; }

  std::unordered_map<ITensorProxyPtr*, float>& quantization_ranges_proxy() {
    return converter_->quantization_ranges_proxy_;
  }

  std::unordered_map<nvinfer1::ITensor*, float>& quantization_ranges() {
    return converter_->quantization_ranges_;
  }

 private:
  Logger& logger_ = *Logger::GetLogger();

 protected:
  std::unique_ptr<Converter> converter_;
  TrtWeightStore* weight_store_;
};

TEST_F(ConverterTest, ConvertNode) {
  ITensorProxyPtr output_tensors[2];
  auto op_converter = [&output_tensors](OpConverterParams* params) -> Status {
    nvinfer1::Dims dims = params->inputs[0].tensor()->getDimensions();
    for (int i = 0; i < 2; ++i) {
      dims.d[0] += 1;
      output_tensors[i]->setDimensions(dims);
      params->outputs->push_back(TRT_TensorOrWeights(output_tensors[i]));
    }
    return Status::OK();
  };
  NodeDef node_def = MakeNodeDef("my_op", "MyOp", {"my_input"});

  TF_ASSERT_OK(converter_->AddInputTensor(
      "my_input", nvinfer1::DataType::kFLOAT, CreateDims({123}), 1));

  // Converter not registered.
  EXPECT_THAT(
      converter_->ConvertNode(node_def),
      StatusIs(error::NOT_FOUND, HasSubstr("No converter for op MyOp")));

  // Register the converter and retry.
  GetOpConverterRegistry()->Register("MyOp", kDefaultConverterPriority,
                                     op_converter);
  TF_ASSERT_OK(converter_->ConvertNode(node_def));

  TRT_TensorOrWeights actual_output_1;
  TF_EXPECT_OK(GetTensorOrWeights("my_op", &actual_output_1));
  EXPECT_EQ(output_tensors[0]->simple_tensor(),
            actual_output_1.tensor()->simple_tensor());
  EXPECT_EQ(124, actual_output_1.tensor()->getDimensions().d[0]);

  TRT_TensorOrWeights actual_output_2;
  TF_EXPECT_OK(GetTensorOrWeights("my_op:1", &actual_output_2));
  EXPECT_EQ(output_tensors[1]->simple_tensor(),
            actual_output_2.tensor()->simple_tensor());
  EXPECT_EQ(125, actual_output_2.tensor()->getDimensions().d[0]);

  EXPECT_THAT(converter_->network(), LayerNamesNonEmpty());
}

TEST_F(ConverterTest, AddAndGetInputs) {
  NodeDef node_def;
  node_def.add_input("^control_input");
  node_def.add_input("input");
  node_def.add_input("input:0");
  node_def.add_input("input:1");
  node_def.add_input("weird_input:2:3:4:0");

  TF_EXPECT_OK(converter_->AddInputTensor("input", nvinfer1::DataType::kFLOAT,
                                          CreateDims({1}), 1));
  TF_EXPECT_OK(converter_->AddInputTensor("input:1", nvinfer1::DataType::kINT32,
                                          CreateDims({2, 3}), 1));
  TF_EXPECT_OK(converter_->AddInputTensor(
      "weird_input:2:3:4", nvinfer1::DataType::kHALF, CreateDims({5, 3}), 1));

  std::vector<TRT_TensorOrWeights> inputs;
  TF_EXPECT_OK(GetInputs(node_def, &inputs));

  EXPECT_EQ(4, inputs.size());
  EXPECT_EQ(inputs[0].tensor()->trt_tensor(), inputs[1].tensor()->trt_tensor());

  EXPECT_EQ(nvinfer1::DataType::kFLOAT, inputs[0].tensor()->getType());
  EXPECT_EQ(nvinfer1::DataType::kINT32, inputs[2].tensor()->getType());
  EXPECT_EQ(nvinfer1::DataType::kHALF, inputs[3].tensor()->getType());
  EXPECT_THAT(inputs[0].tensor()->getDimensions(), DimsAreArray({1}));
  EXPECT_THAT(inputs[2].tensor()->getDimensions(), DimsAreArray({2, 3}));
  EXPECT_THAT(inputs[3].tensor()->getDimensions(), DimsAreArray({5, 3}));

  EXPECT_THAT(converter_->network(), LayerNamesNonEmpty());
}

TEST_F(ConverterTest, RenameAndMarkOutputTensors) {
  // Test that the tensor are actually named and marked as output after
  // Converter::RenameAndMarkOutputTensors() is called.

  // Register a custom converter which shuffles the input. We use it to build a
  // TRT network whose output will be later marked.
  std::vector<ITensorProxyPtr> output_tensors;
  auto op_converter = [&output_tensors](OpConverterParams* params) -> Status {
    nvinfer1::Permutation perm;
    perm.order[0] = 1;
    perm.order[1] = 0;
    for (int i = 0; i < 2; ++i) {
      ITensorProxyPtr input_tensor = params->inputs[0].tensor();
      nvinfer1::IShuffleLayer* layer =
          params->converter->network()->addShuffle(*input_tensor->trt_tensor());
      layer->setFirstTranspose(perm);
      ITensorProxyPtr output_tensor = layer->getOutput(0);
      params->outputs->emplace_back(output_tensor);
      output_tensors.push_back(output_tensor);
    }
    TRT_ShapedWeights output_weights(nvinfer1::DataType::kFLOAT);
    params->outputs->emplace_back(output_weights);
    return Status::OK();
  };
  GetOpConverterRegistry()->Register("MyOp", kDefaultConverterPriority,
                                     op_converter);

  // Run the conversion.
  NodeDef node_def = MakeNodeDef("my_op", "MyOp", {"my_input"});
  TF_EXPECT_OK(converter_->AddInputTensor(
      "my_input", nvinfer1::DataType::kFLOAT, CreateDims({1, 2}), 1));
  TF_EXPECT_OK(converter_->ConvertNode(node_def));

  // Mark a weight as output, should fail.
  EXPECT_THAT(
      converter_->RenameAndMarkOutputTensors({{"my_op:2", "my_output"}}),
      StatusIs(error::INVALID_ARGUMENT,
               HasSubstr("Output my_op:2 is weights not tensor")));

  // Mark tensors as output, should pass.
  TF_EXPECT_OK(converter_->RenameAndMarkOutputTensors(
      {{"my_op", "my_output"}, {"my_op:1", "my_output_1"}}));
  EXPECT_EQ(2, output_tensors.size());
  for (auto output_tensor : output_tensors) {
    EXPECT_THAT(output_tensor->getDimensions(), DimsAreArray({2, 1}));
  }
  EXPECT_EQ("my_output", string(output_tensors[0]->getName()));
  EXPECT_EQ("my_output_1", string(output_tensors[1]->getName()));

  EXPECT_THAT(converter_->network(), LayerNamesNonEmpty());
}

TEST_F(ConverterTest, TransposeTensor) {
  ITensorProxyPtr input_tensor = converter_->network()->addInput(
      "", nvinfer1::DataType::kFLOAT, CreateDims({2, 3, 5}));
  ITensorProxyPtr output_tensor = nullptr;
  NodeDef dummy_node_def = MakeNodeDef("dummy_op", "DummyOp", {});
  // Rank doesn't match.
  EXPECT_THAT(converter_->TransposeTensor(input_tensor, {0, 1}, &output_tensor,
                                          dummy_node_def, "sub1"),
              StatusIs(error::INVALID_ARGUMENT,
                       HasSubstr("Rank of perm for transpose does not match "
                                 "with that of the input")));

  // Transpose at batch dimension.
  EXPECT_THAT(
      converter_->TransposeTensor(input_tensor, {1, 0, 2, 3}, &output_tensor,
                                  dummy_node_def, "sub2"),
      StatusIs(error::UNIMPLEMENTED,
               HasSubstr("Transpose at batch dimension is not supported.")));

  // OK.
  TF_EXPECT_OK(converter_->TransposeTensor(
      input_tensor, {0, 3, 1, 2}, &output_tensor, dummy_node_def, "sub3"));
  EXPECT_THAT(output_tensor->getDimensions(), DimsAreArray({5, 2, 3}));
  EXPECT_THAT(
      converter_->network(),
      LayerNamesAreArray({"TRTEngineOp_000_000/dummy_op-sub3:SHUFFLE"}));
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
        "", nvinfer1::DataType::kFLOAT, CreateDims(input_dims)));
  } else {
    input = TRT_TensorOrWeights(
        weight_store
            ->GetTempWeights(nvinfer1::DataType::kFLOAT, CreateDims(input_dims))
            .ValueOrDie());
  }
  ITensorProxyPtr output_tensor = nullptr;

  NodeDef dummy_node_def = MakeNodeDef("dummy_op", "DummyOp", {});
  for (bool validation_only : {false, true}) {
    const Status status =
        PrepareTensorForShape(converter, input, DimsAdapter(reshape_dims),
                              validation_only, &output_tensor, dummy_node_def);
    if (expected_code == error::OK) {
      TF_EXPECT_OK(status);
      if (validation_only) {
        EXPECT_EQ(nullptr, *output_tensor);
      } else {
        EXPECT_THAT(output_tensor->getDimensions(),
                    DimsAreArray(expected_tensor_dims));
      }
    } else {
      EXPECT_THAT(status, StatusIs(expected_code,
                                   HasSubstr(expected_error_msg_substr)));
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

  EXPECT_THAT(converter_->network(), LayerNamesNonEmpty());
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

  EXPECT_THAT(
      MaybeUpdateBatchSize(124),
      StatusIs(error::INVALID_ARGUMENT,
               HasSubstr(
                   "Provided batch size does not match converter batch size")));
}

TEST_F(ConverterTest, AddAndGetTensorOrWeights) {
  // Add a tensor.
  ITensorProxyPtr simple_tensor;
  TRT_TensorOrWeights tensor(simple_tensor);
  EXPECT_EQ(-1, tensor.batch_size());
  TF_EXPECT_OK(MaybeUpdateBatchSize(123));
  TF_EXPECT_OK(AddTensorOrWeights("my_tensor", tensor));

  // Get the added tensor.
  TRT_TensorOrWeights added_tensor;
  TF_EXPECT_OK(GetTensorOrWeights("my_tensor", &added_tensor));
  EXPECT_EQ(123, added_tensor.batch_size());

  // Add the same tensor again.
  EXPECT_THAT(AddTensorOrWeights("my_tensor", tensor),
              StatusIs(error::ALREADY_EXISTS,
                       HasSubstr("tensor/weights my_tensor already exist")));
}

template <typename T>
void TestGetWeightRange(ConverterTest* test, TrtWeightStore* weight_store) {
  nvinfer1::DataType trt_type;
  TF_ASSERT_OK(TfTypeToTrtType(DataTypeToEnum<T>::v(), &trt_type));
  TRT_ShapedWeights weights =
      weight_store->GetTempWeights(trt_type, CreateDims({2, 3})).ValueOrDie();
  const std::vector<T> values = {T(3), T(1), T(2), T(6), T(5), T(4)};
  absl::c_copy(values, weights.GetPointer<T>());
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
  ITensorProxyPtr simple_tensor;
  // Asymmetric range
  converter_->ProvideQuantizationRange(&simple_tensor, 0.0f, 6.0f);
  EXPECT_EQ(6.0f, quantization_ranges_proxy()[&simple_tensor]);
  converter_->ProvideQuantizationRange(&simple_tensor, 1.0f, 6.0f);
  EXPECT_EQ(6.0f, quantization_ranges_proxy()[&simple_tensor]);
  converter_->ProvideQuantizationRange(&simple_tensor, -8.0f, 6.0f);
  EXPECT_EQ(8.0f, quantization_ranges_proxy()[&simple_tensor]);
  converter_->ProvideQuantizationRange(&simple_tensor, -8.123f, -6.123f);
  EXPECT_EQ(8.123f, quantization_ranges_proxy()[&simple_tensor]);
  // Symmetric range
  converter_->ProvideQuantizationRange(&simple_tensor, -6.123f, 6.123f);
  EXPECT_EQ(6.123f, quantization_ranges_proxy()[&simple_tensor]);

  EXPECT_THAT(converter_->network(), LayerNamesNonEmpty());
}

TEST_F(ConverterTest, MaybeApplyQuantizationRanges) {
  ITensorProxyPtr input;
  ITensorProxyPtr not_infer;
  Logger& logger = *Logger::GetLogger();
  auto int8_converter = Converter::Create(TrtPrecisionMode::INT8,
                                          /*use_calibration=*/true, &logger,
                                          /*use_implicit_batch=*/true,
                                          /*engine_name=*/"")
                            .ValueOrDie();
  int8_converter->ProvideQuantizationRange(&input, -5.0f, 5.0f);
  int8_converter->ProvideQuantizationRange(&not_infer, -100.0f, 100.0f);

  int8_converter->MaybeApplyQuantizationRanges();
  EXPECT_EQ(input->getDynamicRangeMax(), 5.0f);
  EXPECT_EQ(not_infer->getDynamicRangeMax(), 100.0f);

  EXPECT_THAT(int8_converter->network(), LayerNamesNonEmpty());
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
                               const char* expected_error_msg_substr = "",
                               const int operand_1_batch_size = -1,
                               const int operand_2_batch_size = -1) {
    auto create_tensor_or_weights = [](const std::vector<int>& shape,
                                       bool is_tensor, int batch_size = -1) {
      if (is_tensor) {
        return TRT_TensorOrWeights(nvinfer1::DataType::kFLOAT,
                                   CreateDims(shape), batch_size);
      }
      TRT_ShapedWeights weights;
      weights.Shape() = CreateDims(shape);
      return TRT_TensorOrWeights(weights);
    };

    nvinfer1::Dims operand_1_new_dims, operand_2_new_dims;
    TRT_TensorOrWeights operand_1 = create_tensor_or_weights(
        operand_1_shape, operand_1_is_tensor, operand_1_batch_size);
    TRT_TensorOrWeights operand_2 = create_tensor_or_weights(
        operand_2_shape, operand_2_is_tensor, operand_2_batch_size);

    // operand_1 broadcast operand_2
    EXPECT_THAT(
        GetTrtBroadcastShape(operand_1, operand_2, /*check_feasibility=*/true,
                             /*use_implicit_batch=*/true, &operand_1_new_dims,
                             &operand_2_new_dims),
        StatusIs(expected_code, HasSubstr(expected_error_msg_substr)));
    if (expected_code == error::OK) {
      EXPECT_THAT(operand_1_new_dims, DimsAreArray(expected_operand_1_shape));
      EXPECT_THAT(operand_2_new_dims, DimsAreArray(expected_operand_2_shape));
    }
    // operand_2 broadcast operand_1
    EXPECT_THAT(
        GetTrtBroadcastShape(operand_2, operand_1, /*check_feasibility=*/true,
                             /*use_implicit_batch=*/true, &operand_2_new_dims,
                             &operand_1_new_dims),
        StatusIs(expected_code, HasSubstr(expected_error_msg_substr)));
    if (expected_code == error::OK) {
      EXPECT_THAT(operand_1_new_dims, DimsAreArray(expected_operand_1_shape));
      EXPECT_THAT(operand_2_new_dims, DimsAreArray(expected_operand_2_shape));
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

  EXPECT_THAT(converter_->network(), LayerNamesNonEmpty());
}

TEST_F(ConverterTest, CreateConstantLayer) {
  for (auto dtype : {nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT32}) {
    TRT_ShapedWeights weights =
        weight_store_->GetTempWeights(dtype, CreateDims({2, 3, 5}))
            .ValueOrDie();
    ITensorProxyPtr tensor =
        converter_->CreateConstantLayer(weights, CreateDims({3, 10}));
    ASSERT_NE(nullptr, tensor->trt_tensor());
    EXPECT_EQ(dtype, tensor->getType())
        << "Expected " << DebugString(dtype) << " vs. actual "
        << DebugString(tensor->getType());
    EXPECT_THAT(tensor->getDimensions(), DimsAreArray({3, 10}));
  }

  EXPECT_THAT(converter_->network(), LayerNamesNonEmpty());
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
        /*convert_successfully=*/nullptr, /*profiles=*/nullptr,
        "TRTEngineOp_000_000", /*use_explicit_precision=*/false);
  }

 protected:
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine_;

 private:
  Logger& logger_ = *Logger::GetLogger();
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
  const auto dType = data.tensor.dtype();
  if (dType == DT_FLOAT) {
    auto span = GetSpanForData<float>(data);
    return std::vector<float>(span.begin(), span.end());
  }
  if (dType == DT_HALF) {
    return CastVector<Eigen::half, float>(GetSpanForData<Eigen::half>(data));
  }
  if (dType == DT_INT32) {
    return CastVector<int32, float>(GetSpanForData<int32>(data));
  }
#if IS_TRT_VERSION_GE(8, 2, 0, 0)
  if (dType == DT_BOOL) {
    return CastVector<bool, float>(GetSpanForData<bool>(data));
  }
#endif
  LOG(FATAL) << "DataType not supported for testing " << DataTypeString(dType);
  return {};
}

// Class to test various op converters, using both a TrtNodeValidator and
// Converter.
class OpConverterTest : public ::testing::Test {
 public:
  OpConverterTest()
      : tensor_buffer_allocator_(new GpuManagedAllocator()),
        scope_(Scope::NewRootScope()) {
    PreventUnloadBuilderResources();
    QCHECK_EQ(0, cudaStreamCreate(&stream_));
    Reset();
  }

  ~OpConverterTest() noexcept override {
    QCHECK_EQ(0, cudaStreamDestroy(stream_));
  }

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
                                        TrtTestMode::kImplicitBatch,
                                    /*engine_name=*/"")
                      .ValueOrDie());

    // Reset other related artifacts.
    scope_ = Scope::NewRootScope();
  }

  // Constructs a flat tensor with 'vals' in Unified Memory.
  template <typename T>
  Tensor AsTensor(gtl::ArraySlice<T> vals) {  // non-absl ok
    Tensor ret(tensor_buffer_allocator_.get(), DataTypeToEnum<T>::value,
               {static_cast<int64_t>(vals.size())});
    std::copy_n(vals.data(), vals.size(), ret.flat<T>().data());
    return ret;
  }

  // Constructs a tensor of "shape" with values "vals" in Unified Memory.
  template <typename T>
  Tensor AsTensor(gtl::ArraySlice<T> vals,  // non-absl ok
                  const TensorShape& shape) {
    Tensor ret(tensor_buffer_allocator_.get(), DataTypeToEnum<T>::value,
               {static_cast<int64_t>(vals.size())});
    CHECK(ret.CopyFrom(AsTensor(vals), shape));
    return ret;
  }

  template <typename T, typename S>
  void transformTensor(const std::vector<T>& vals, Tensor& ret) {
    std::transform(vals.begin(), vals.end(), ret.flat<S>().data(),
                   [](const T in_val) -> S { return static_cast<S>(in_val); });
  }

  template <typename T, typename S>
  void transformWeights(const std::vector<T>& vals,
                        TRT_ShapedWeights& weights) {
    std::transform(vals.begin(), vals.end(), weights.GetPointer<S>(),
                   [](const T in_val) -> S { return static_cast<S>(in_val); });
  }

  // Constructs a tensor with given values (vals). The tensor type is defined by
  // the tf_type argument, its shape is given by input_dims. The tensor is
  // constructed using the allocator of OpConverterTest in Unified Memory.
  template <typename T>
  Tensor AsTensor(const std::vector<T>& vals,
                  const std::vector<int>& input_dims, DataType tf_type) {
    Tensor ret(tensor_buffer_allocator_.get(), tf_type,
               {static_cast<int64_t>(vals.size())});
    if (tf_type == DT_FLOAT) {
      transformTensor<T, float>(vals, ret);
    } else if (tf_type == DT_HALF) {
      transformTensor<T, Eigen::half>(vals, ret);
    } else if (tf_type == DT_INT32) {
      transformTensor<T, int32>(vals, ret);
#if IS_TRT_VERSION_GE(8, 2, 0, 0)
    } else if (tf_type == DT_BOOL) {
      transformTensor<T, bool>(vals, ret);
#endif
    } else {
      LOG(FATAL) << "Cannot create tensor with type "
                 << DataTypeString(tf_type);
    }
    TensorShape shape;
    TF_EXPECT_OK(TensorShapeUtils::MakeShape(input_dims, &shape));
    CHECK(ret.CopyFrom(ret, shape));
    return ret;
  }

  template <typename T>
  Tensor AsTensor(const std::vector<int>& vals,
                  const std::vector<int>& input_dims, DataType tf_type) {
    const auto& conv_vals = CastVector<int, T>(vals);
    return AsTensor(conv_vals, input_dims, tf_type);
  }

  // Constructs a flat tensor in Unified Memory.
  template <typename T>
  Tensor ConstructTensor(int data_size, const T& value = T()) {
    std::vector<T> values(data_size, value);
    return AsTensor<T>(values);
  }

  // Constructs a flat tensor in Unified Memory.
  template <typename T>
  Tensor ConstructTensor(int data_size, const T& value, DataType tf_type) {
    std::vector<T> values(data_size, value);
    return AsTensor<T>(values, {data_size}, tf_type);
  }

  void CheckDataTypeMatches(const DataVec& datas) {
    if (VLOG_IS_ON(2)) {
      int nbBindings = engine_->getNbBindings();
      VLOG(2) << "Number of engine bindings: " << nbBindings;
      for (int i = 0; i < nbBindings; i++) {
        VLOG(2) << "Binding " << i << " name: " << engine_->getBindingName(i);
      }
    }
    for (const auto& data : datas) {
      VLOG(2) << "Checking if data type matches for tensor " << data.name;
      const int input_index = engine_->getBindingIndex(data.name.c_str());
      ASSERT_NE(-1, input_index);
      const nvinfer1::DataType trt_dtype =
          engine_->getBindingDataType(input_index);
      DataType tf_type;
      TF_ASSERT_OK(TrtTypeToTfType(trt_dtype, &tf_type));
      ASSERT_EQ(data.tensor.dtype(), tf_type)
          << DataTypeString(data.tensor.dtype()) << " vs. "
          << DataTypeString(tf_type);
    }
  }

  Status BuildAndRun(const DataVec& input_data, DataVec* output_data,
                     const int batch_size = 1) {
    // Mark the output tensor as TRT engine output.
    std::vector<Converter::EngineOutputInfo> output_info;
    for (const auto& data : *output_data) {
      nvinfer1::DataType trt_type;
      TF_RETURN_IF_ERROR(TfTypeToTrtType(data.tensor.dtype(), &trt_type));
      output_info.push_back({data.name, data.name, trt_type});
    }
    TF_RETURN_IF_ERROR(converter_->RenameAndMarkOutputTensors(output_info));

    // Build the TRT engine.
    if (engine_.get() != nullptr) {
      return errors::Internal("Engine already exists");
    }
    TrtShapeOptimizationProfile profiles;
    if (!converter_->use_implicit_batch()) {
      profiles.SetShapeTensorMask(converter_->network());
      TF_RETURN_IF_ERROR(profiles.CollectShapeValues(input_data));
      // Create a single optimization profile for explicit batch mode
      std::vector<TensorShape> input_shapes;
      TF_RETURN_IF_ERROR(GetShapeFromDataVec(input_data, &input_shapes));
      profiles.AddShape(input_shapes);
      std::vector<PartialTensorShape> input_partial_shapes;
      TF_RETURN_IF_ERROR(
          GetNetworkInputShapes(converter_->network(), &input_partial_shapes));
      profiles.InitProfiles(input_partial_shapes,
                            ProfileStrategy::kImplicitBatchModeCompatible);
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
    TF_RETURN_IF_ERROR(
        SetTrtEngineInputs(engine_.get(), execution_context.get(), 0, buffers,
                           converter_->use_implicit_batch(), batch_size,
                           profiles, nullptr, &input_data));
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

  // Adds ITensor for both validation and conversion, assuming explicit batch
  // dimension is included in dims (ie for an NCHW tensor dims = {N, C, H, W}).
  void AddTestTensorWithTFDims(
      const string& name, const std::vector<int32>& dims,
      nvinfer1::DataType trt_type = nvinfer1::DataType::kFLOAT,
      Status add_input_status = Status::OK()) {
    DataType tf_type;
    TF_ASSERT_OK(TrtTypeToTfType(trt_type, &tf_type));
    ops::Placeholder::Attrs attrs;
    TF_EXPECT_OK(TensorShapeUtils::MakeShape(dims, &attrs.shape_));

    auto input = ops::Placeholder(scope_.WithOpName(name), tf_type, attrs);
    node_inputs_[name] = input.output;

    // Add a real ITensor for conversion conditionally.

    auto dims_adap =
        DimsAdapter::Create(attrs.shape_, converter_->use_implicit_batch());
    if (converter_->use_implicit_batch() && !dims_adap.ok()) {
      ASSERT_EQ(add_input_status, dims_adap.status());
      return;
    } else {
      TF_EXPECT_OK(dims_adap.status());
    }
    if (!converter_->use_implicit_batch() || dims_adap->IsStatic()) {
      int batch_size = dims.size() > 0 ? dims[0] : 0;
      Status status = converter_->AddInputTensor(
          name, trt_type, dims_adap->AsTrtDims(), batch_size);
      ASSERT_EQ(add_input_status, status);
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
    DimsAdapter adap(dims);
    std::vector<int32_t> dims_vec;
    TF_CHECK_OK(adap.Prepend(batch_size).Vector(&dims_vec));
    AddTestTensorWithTFDims(name, dims_vec, trt_dtype);
    if (adap.IsStatic()) {
      ASSERT_EQ(batch_size, converter_->batch_size_);
    }
  }

  // Adds weights for both validation and conversion. The type of the weight is
  // determined by tf_type. The initial value vector (values) can have any
  // type (T) that can be statically casted to tf_type.
  template <typename T = int32>
  void AddTestWeights(const string& name, const std::vector<int>& dims,
                      const std::vector<T>& values_inp, DataType tf_type,
                      bool fix_values = true) {
    const DimsAdapter dims_adap(dims);
    const int64_t num_elements = dims_adap.Volume();

    std::vector<T> values(values_inp);
    if (num_elements != values.size()) {
      if (fix_values) {
        AdjustVectorByDims<T>(values, num_elements, name, "AddTestWeights");
      } else {
        FAIL() << "Unable to create test weights: "
               << (num_elements > values.size() ? "not enough" : "to many")
               << " values specified: " << values.size() << " vs. "
               << num_elements << " defined by dims";
      }
    }
    // Add weights for validation.
    Tensor t = AsTensor<T>(values, dims, tf_type);
    node_inputs_[name] = ops::Const(scope_.WithOpName(name), t);

    // Add weights for conversion.
    nvinfer1::DataType dtype;
    TF_ASSERT_OK(TfTypeToTrtType(tf_type, &dtype));
    QCHECK_EQ(num_elements, values.size())
        << num_elements << " vs " << values.size();
    TRT_ShapedWeights weights(dtype);
    if (num_elements) {
      weights =
          converter_->weight_store_.GetTempWeights(dtype, dims_adap.AsTrtDims())
              .ConsumeValueOrDie();

      if (tf_type == DT_FLOAT) {
        transformWeights<T, float>(values, weights);
      } else if (tf_type == DT_HALF) {
        transformWeights<T, Eigen::half>(values, weights);
      } else if (tf_type == DT_INT32) {
        transformWeights<T, int32>(values, weights);
#if IS_TRT_VERSION_GE(8, 2, 0, 0)
      } else if (tf_type == DT_BOOL) {
        transformWeights<T, bool>(values, weights);
#endif
      } else {
        LOG(FATAL) << "Cannot create tensor with type "
                   << DataTypeString(tf_type);
      }
    }
    TF_EXPECT_OK(
        converter_->AddTensorOrWeights(name, TRT_TensorOrWeights{weights}));
  }

  // Adds test weight without specifying tf_type arg. In this case the initial
  // value type (T) will determine the type of the weights.
  template <typename T = int32>
  void AddTestWeights(const string& name, const std::vector<int>& dims,
                      const std::vector<T>& value, bool fix_values = true) {
    AddTestWeights(name, dims, value, DataTypeToEnum<T>::value, fix_values);
  }

  // Test validation in validation-only mode.
  Status RunValidation(const Node* node) {
    grappler::GrapplerItem item;
    TF_EXPECT_OK(scope_.ToGraphDef(&item.graph));
    grappler::GraphProperties graph_properties(item);
    TF_EXPECT_OK(graph_properties.InferStatically(true));

    TrtNodeValidator validator(
        graph_properties, converter_->precision_mode(),
        /*use_calibration=*/false,
        /*use_implicit_batch=*/converter_->use_implicit_batch(),
        /*use_explicit_precision=*/false);
    return validator.IsTensorRTCandidate(node);
  }

  void RunConversion(const Node* node, error::Code expected_code = error::OK,
                     const std::string& expected_msg_substr = "") {
    EXPECT_THAT(converter_->ConvertNode(node->def()),
                StatusIs(expected_code, HasSubstr(expected_msg_substr)));
    if (expected_code == error::OK) {
      EXPECT_THAT(converter_->network(), LayerNamesNonEmpty());
    }
  }

  // Helper method to run both validation and conversion, when the expected
  // output are same.
  void RunValidationAndConversion(const NodeDef& node_def,
                                  error::Code expected_code = error::OK,
                                  const std::string& expected_msg_substr = "",
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

    status = RunValidation(node);
    if (should_run_conversion && status.ok()) {
      RunConversion(node, expected_code, expected_msg_substr);
    } else {
      EXPECT_THAT(status,
                  StatusIs(expected_code, HasSubstr(expected_msg_substr)));
    }
  }

  // Helper method to run both validation and conversion, and check the output
  // shapes.
  void RunValidationAndConversion(
      const NodeDef& node_def, const Status& status,
      const std::string& output_name,
      const std::vector<std::vector<int>>& exp_out_dims) {
    RunValidationAndConversion(node_def, status.code(), status.error_message(),
                               true);

    if (status.ok()) {
      // TODO(tfeher): Enable this check in explicit_batch_mode.
      // In dynamic shape mode the output dims cannot be tested here. In that
      // case we need to wait for the concrate input shapes to be defined (by
      // setBindingDimensions before enqueue) before we can check the output
      // dims.
      if (converter_->use_implicit_batch()) {
        for (int i = 0; i < exp_out_dims.size(); i++) {
          TRT_TensorOrWeights output;
          string name = i == 0 ? output_name : StrCat(output_name, ":", i);
          TF_EXPECT_OK(GetTensorOrWeights(name.c_str(), &output));
          ASSERT_TRUE(output.is_tensor());
          if (!exp_out_dims[i].empty()) {
            // Removing batch dim.
            auto out_dims = std::vector<int>(exp_out_dims[i].begin() + 1,
                                             exp_out_dims[i].end());
            VLOG(2) << "Testing output shape for tensor " << name;
            EXPECT_THAT(output.tensor()->getDimensions(),
                        DimsAreArray(out_dims));
          }
        }
      }
    }
  }

  // Expose quantization_ranges_ for tests
  std::unordered_map<ITensorProxyPtr*, float>& quantization_ranges_proxy() {
    return converter_->quantization_ranges_proxy_;
  }

  // Expose quantization_ranges_ for tests
  std::unordered_map<nvinfer1::ITensor*, float>& quantization_ranges() {
    return converter_->quantization_ranges_;
  }

 protected:
  template <typename T>
  void AdjustVectorByDims(std::vector<T>& values, size_t num_elements,
                          const string& name, const char* callingFunc) {
    const auto old_size = values.size();
    if (num_elements > old_size) {
      // Expending vector with 0's.
      const std::vector<T> zeros(num_elements - old_size, 0);
      values.reserve(num_elements);
      values.insert(values.end(), zeros.begin(), zeros.end());
      VLOG(2) << "In function " << callingFunc << " the vector '" << name
              << "' was extended by " << num_elements - old_size << " zeros";
    } else {
      // Removing unnecessary elements.
      values.resize(num_elements);
      VLOG(2) << "Only first " << num_elements << " out of " << old_size
              << " elements of the vector '" << name
              << "' will be used in function" << callingFunc;
    }
  }

 public:
  std::unique_ptr<Converter> converter_;

 private:
  Logger& logger_ = *Logger::GetLogger();
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine_;
  cudaStream_t stream_;
  std::unique_ptr<Allocator> tensor_buffer_allocator_;
  // The scope that contains the graph being converted. Because
  // tensor_buffer_allocator_ provides the storage for tensor contents that are
  // represented as attributes for graph nodes within scope_,
  // tensor_buffer_allocator_ needs to be available when destructing scope_.
  // Therefore, scope_ comes after tensor_buffer_allocator_ in the class member
  // field list.
  Scope scope_;
  std::unordered_map<string, Output> node_inputs_;
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
  os << "input_dims" << PrintToString(p.input_dims);
  if (!p.partial_input_dims.empty()) {
    os << ", partial_input_dims" << PrintToString(p.partial_input_dims);
  }
  if (!p.expected_output_dims.empty()) {
    os << ", exp_out_dims" << PrintToString(p.expected_output_dims);
  }
  if (!p.param.empty()) {
    os << ", param" << PrintToString(p.param);
  }
  os << ", " << p.status;
  return os;
}

// Printing vector with the numbers of type T which defines tensor or shape.
template <typename T>
const std::string get_debug_string_for_vector(const std::vector<T>& vector,
                                              absl::string_view pComment,
                                              absl::string_view name,
                                              absl::string_view type = "") {
  const std::string t1 = absl::StrCat(pComment, name, " Dims(nbDims=");
  const std::string t2 = absl::StrJoin(vector, ",");
  const std::string t3 = type != "" ? absl::StrCat(") of type ", type) : ")";
  std::stringstream stream;
  stream << t1 << vector.size() << ", d=" << t2 << t3;
  return stream.str();
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
      : trt_mode_(std::get<0>(GetParam())),
        tf_type_(std::get<1>(GetParam())),
        converter_precision_(std::get<2>(GetParam())) {
    LOG(INFO) << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";
    LOG(INFO) << "tf_type_: " << DebugString(tf_type_);
    LOG(INFO) << "trt_mode_: " << DebugString(trt_mode_);
    LOG(INFO) << "converter_precision_: " << DebugString(converter_precision_);
    LOG(INFO) << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";
  }

  void Reset() {
    OpConverterTest::Reset(converter_precision_, trt_mode_);
    input_data_.clear();
  }

  void Reset(TrtPrecisionMode precision) {
    OpConverterTest::Reset(precision, trt_mode_);
    input_data_.clear();
  }

  // Getters of protected attributes
  DataType get_tf_type() { return tf_type_; }
  TrtTestMode get_trt_mode() { return trt_mode_; }
  TrtPrecisionMode get_converter_precision() { return converter_precision_; }

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
  // - partial_input_shape dimensions which can include unknown shapes. This can
  //   be empty, in that case the partial_input_shape will be set automatically
  //   depending on the trt_mode argument. (This argument also includes explicit
  //   batch dim).
  // - add_input_status adding ITensor to the network can fail in implicit batch
  //   mode if the batch size is inconsistent. Using the add_input_status arg we
  //   can test such errors.
  //
  template <typename T = int>
  void AddTestTensor(const string& name, const std::vector<int32>& dims,
                     DataType tf_type, const std::vector<T>& values_inp,
                     const std::vector<int32>& partial_input_shape_dims = {},
                     Status add_input_status = Status::OK(),
                     bool fix_values = true) {
    std::vector<T> values(values_inp);
    VLOG(2) << "**** AddTestTensor for " << name
            << " ***** dims empty() = " << dims.empty()
            << "  tf_type = " << DebugString(tf_type);
    if (!dims.empty()) {
      const auto num_elements = std::accumulate(
          std::begin(dims), std::end(dims), 1, std::multiplies<double>());
      if (!values.empty() && num_elements != values.size()) {
        if (fix_values) {
          AdjustVectorByDims(values, num_elements, name, "AddTestTensor");
        } else {
          // Note: for conversion only tests, it is valid to have empty values,
          // otherwise the number of elements should match.
          LOG(WARNING) << "Expected Test Tensor Shape: " << DebugString(dims)
                       << ", Received Input Tensor: " << DebugString(values);
        }
      }
    }

    std::vector<int32> partial_shape;
    if (!partial_input_shape_dims.empty()) {
      partial_shape = partial_input_shape_dims;
    } else {
      if (trt_mode_ == TrtTestMode::kDynamicShape) {
        // In dynamic shape mode we make all dims unknown.
        partial_shape = std::vector<int32>(dims.size(), -1);
      } else {
        // Use static (known) input shapes.
        partial_shape = dims;
      }
      if (VLOG_IS_ON(2)) {
        VLOG(2) << get_debug_string_for_vector(
            partial_shape, "Using partial_shape: for ", name);
      }
    }
    nvinfer1::DataType trt_type;
    TF_ASSERT_OK(TfTypeToTrtType(tf_type, &trt_type));
    AddTestTensorWithTFDims(name, partial_shape, trt_type, add_input_status);
    if (!values.empty()) {
      if (VLOG_IS_ON(2)) {
        VLOG(2) << get_debug_string_for_vector(
            values, "Adding test tensor: for ", name, DataTypeString(tf_type));
      }
      InputOutputData data{name, AsTensor(values, dims, tf_type)};
      VLOG(2) << "Added tensor: " << data.name << " with dtype "
              << DataTypeString(data.tensor.dtype());
      input_data_.push_back(data);
    }
  }

  // Adds test tensor (same as above) but with the default tf_type defined by
  // the test params.
  template <typename T = int>
  void AddTestTensor(const string& name, const std::vector<int32>& dims,
                     const std::vector<T>& values = {},
                     const std::vector<int32>& partial_input_shape_dims = {}) {
    AddTestTensor<T>(name, dims, tf_type_, values, partial_input_shape_dims);
  }

  // Builds and runs the converted network. Checks output tensor shape. Tests
  // output values using a matcher. The network can have multiple input and
  // output tensors. The inputs are defined by the input_data_ member variable.
  void BuildAndRun(const string& name,
                   const std::vector<std::vector<int>>& expected_output_dims,
                   const Status& expected_runtime_status,
                   const std::vector<Matcher<std::vector<float>>>& matcher,
                   const std::vector<DataType>& out_tf_types = {}) {
    TensorShape shape;
    const int n_output = expected_output_dims.size();
    ASSERT_EQ(n_output, matcher.size());
    DataVec output_data;
    for (int i = 0; i < n_output; i++) {
      TF_EXPECT_OK(
          TensorShapeUtils::MakeShape(expected_output_dims[i], &shape));
      string out_name = (i == 0) ? name : StrCat(name, ":", i);
      DataType out_tf_type =
          out_tf_types.size() > i ? out_tf_types[i] : tf_type_;
      InputOutputData data{
          out_name, ConstructTensor(shape.num_elements(), 0, out_tf_type)};
      output_data.push_back(data);
    }
    const int batch_size =
        input_data_.empty() ||
                TensorShapeUtils::IsScalar(input_data_[0].tensor.shape())
            ? 1
            : input_data_[0].tensor.shape().dim_size(0);
    Status stat =
        OpConverterTest::BuildAndRun(input_data_, &output_data, batch_size);
    ASSERT_EQ(expected_runtime_status.ok(), stat.ok())
        << "expected status: " << expected_runtime_status
        << ", actual status: " << stat;
    if (expected_runtime_status.ok() && stat.ok()) {
      for (int i = 0; i < n_output; i++) {
        // Check the shape of the actual output tensors
        TF_EXPECT_OK(
            TensorShapeUtils::MakeShape(expected_output_dims[i], &shape));
        EXPECT_TRUE(output_data[i].tensor.shape() == shape)
            << "Expected shape: " << shape.DebugString() << ", actual shape: "
            << output_data[i].tensor.shape().DebugString();
        EXPECT_THAT(GetDataAsFloat(output_data[i]), matcher[i]);
      }
    }
  }

  // Runs validation and conversion. If conversion is successfull then builds
  // the TRT network, executes it and checks the output. Handles multiple output
  // tensors.
  void TestOpConverterMultiOut(
      const string& name, const NodeDef node_def,
      const std::vector<std::vector<int>>& expected_output_dims,
      const Status& expected_conversion_status,
      const Status& expected_runtime_status,
      const std::vector<Matcher<std::vector<float>>>& matcher,
      const std::vector<DataType>& out_tf_type = {}) {
    RunValidationAndConversion(node_def, expected_conversion_status, name,
                               expected_output_dims);
    if (expected_conversion_status.ok()) {
      BuildAndRun(name, expected_output_dims, expected_runtime_status, matcher,
                  out_tf_type);
    }
  }

  // Runs validation and conversion. If conversion is successfull then builds
  // the TRT network, executes it and checks the output.
  void TestOpConverter(const string& name, const NodeDef node_def,
                       const std::vector<int>& expected_output_dims,
                       const Status& expected_conversion_status,
                       const Status& expected_runtime_status,
                       const Matcher<std::vector<float>>& matcher,
                       const std::vector<DataType>& out_tf_types = {}) {
    const auto& exp_dims =
        std::vector<std::vector<int>>({expected_output_dims});
    RunValidationAndConversion(node_def, expected_conversion_status, name,
                               exp_dims);
    if (expected_conversion_status.ok()) {
      BuildAndRun(name, exp_dims, expected_runtime_status,
                  std::vector<Matcher<std::vector<float>>>({matcher}),
                  out_tf_types);
    }
  }

 protected:
  const TrtTestMode trt_mode_;
  const DataType tf_type_;
  const TrtPrecisionMode converter_precision_;
  DataVec input_data_;
};

template <typename T>
class OpConverter_UnaryTest : public ParameterizedOpConverterTestBase {
 public:
  template <typename S>
  void RunTests(
      const string& testName, const operationMap<S>& map,
      std::map<std::string,
               std::pair<std::function<NodeDef(DataType)>, T (*)(T)>>& op_map,
      const std::vector<T> input_values, const std::string input_name = "input",
      float max_abs_error = 0.0001, bool nan_sensitive = true) {
    // Prepare test parameters.
    auto p = TestParamBase{
        {1, 1, 2, 3},  // input dims
        {},            // input partial dims
        {1, 1, 2, 3},  // expected output dims
    };

    // Get list of ops to test.
    std::vector<string> ops_to_test;
    for (auto& pair : map) {
      ops_to_test.push_back(pair.first);
    }

    for (const string& op_name : ops_to_test) {
      SCOPED_TRACE(op_name);
      if (!op_map.count(op_name)) {
        FAIL() << testName << " op test map does not contain op " << op_name;
      }

      const DataType tf_type = get_tf_type();
      const NodeDef& node_def = op_map[op_name].first(tf_type);
      runExpectedToFailTest(node_def, input_name, input_values, op_name);

      Status conv_status = Status::OK();
      if (trt_mode_ == TrtTestMode::kImplicitBatch &&
          (op_name == "Sign" || op_name == "Round" ||
           op_name == "LogicalNot")) {
        conv_status =
            errors::Unimplemented("Unary op: '", op_name,
                                  "' is not supported in implicit batch mode");
      }

      Reset();
      const DataType input_tf_type = op_name == "Cast" ? DT_HALF : tf_type;
      const DataType output_tf_type = op_name == "Cast" ? DT_FLOAT : tf_type;

      AddTestTensor("input", p.input_dims, input_tf_type, input_values);

      std::vector<float> output;
      std::transform(input_values.begin(), input_values.end(),
                     std::back_inserter(output), op_map[op_name].second);

      TestOpConverter("my_unary", node_def, p.expected_output_dims, conv_status,
                      Status::OK(),
                      ArrayFloatNear(output, max_abs_error, nan_sensitive),
                      {output_tf_type});
    }
  }
  void runExpectedToFailTest(const NodeDef& node_def,
                             const std::string& input_name,
                             const std::vector<T>& input_values,
                             const std::string& op_name) {
    // Input is weights, should fail.
    Reset();
    std::string error =
        "The input \"" + input_name + "\" for " + op_name + " must be a tensor";
    AddTestWeights("input", {1, 2, 3}, input_values, get_tf_type());
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED, error);

    // Input has 0 dimensions, should fail.
    Reset();
    std::vector<int32> dims = {};
    if (trt_mode_ == TrtTestMode::kImplicitBatch) {
      dims = {1};
    }
    error = "At least 1 dimension is required for UNARY operation '" + op_name +
            "'";
    AddTestTensor("input", dims);
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT, error);
  }
};

template <typename T>
class OpConverter_BinaryTest : public ParameterizedOpConverterTestBase {
 public:
  template <typename S>
  void RunTests(
      const operationMap<S>& map,
      std::map<std::string,
               std::pair<std::function<NodeDef(DataType)>, std::vector<T>>>&
          op_test_info,
      const std::vector<std::vector<T>>& data) {
    // Test combinations of tensor vs weight inputs (except when both inputs are
    // weights).
    const DataType tf_type = get_tf_type();
    for (const bool operand_1_is_tensor : {true, false}) {
      for (const bool operand_2_is_tensor : {true, false}) {
        for (auto& iter : map) {
          const string& op_name = iter.first;
          SCOPED_TRACE(StrCat(op_name, "_", operand_1_is_tensor ? "T" : "W",
                              operand_2_is_tensor ? "T" : "W"));

          if (!op_test_info.count(op_name)) {
            FAIL() << "Binary op test map does not contain op " << op_name;
          }

          if (!operand_1_is_tensor && !operand_2_is_tensor) {
            runExpectedToFailTest(op_name);
            continue;
          }
          auto conv_status = Status::OK();
          if (tf_type == DT_BOOL) {
            if (trt_mode_ == TrtTestMode::kImplicitBatch) {
              conv_status = errors::Unimplemented(
                  "Binary op: '", op_name,
                  "' is not supported in implicit batch mode");
            } else if (!operand_1_is_tensor || !operand_2_is_tensor) {
              conv_status = errors::InvalidArgument(
                  "Both inputs  of '", op_name, "' are expected to be tensors");
            }
          }

          Reset();
          if (operand_1_is_tensor) {
            AddTestTensor("input1", {2, 1, 2}, data[0]);
          } else {
            AddTestWeights("input1", {1, 2}, data[1], tf_type);
          }
          if (operand_2_is_tensor) {
            AddTestTensor("input2", {2, 2, 1}, data[2]);
          } else {
            AddTestWeights("input2", {2, 1}, data[3], tf_type);
          }

          const NodeDef& node_def = op_test_info[op_name].first(tf_type);
          TestOpConverter("my_binary", node_def, {2, 2, 2}, conv_status,
                          Status::OK(),
                          ElementsAreArray(op_test_info[op_name].second));
        }
      }
    }
  }

  void runExpectedToFailTest(const std::string& op_name) {
    Reset();
    AttrValue dtype;
    dtype.set_type(tf_type_);

    const auto& node_def = MakeNodeDef(
        "my_oper", op_name, {"weights1", "weights2"}, {{"T", dtype}});
    AddTestWeights("weights1", {1}, {1}, tf_type_);
    AddTestWeights("weights2", {1}, {1}, tf_type_);
    const string error =
        "Constant folding is falled back to TensorFlow, "
        "binary op '" +
        op_name + "' received both input as constant";
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED, error);
  }
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
typedef ParameterizedOpConverterTestBase OpConverter_FP32_Test;
// Base class for tests that need to be tested for both FP32 and FP16.
typedef ParameterizedOpConverterTestBase OpConverter_FP32_FP16_Test;
// Base class for Binary tests that need to be tested
typedef OpConverter_BinaryTest<float> OpConverter_FP32_FP16_BinaryTest;
typedef OpConverter_BinaryTest<int> OpConverter_BOOL_BinaryTest;
// Base class for tests that need to be tested for FP32, FP16, and INT32
typedef ParameterizedOpConverterTestBase OpConverter_FP32_FP16_INT32_Test;
// Base class for tests that need to be tested for INT32
typedef ParameterizedOpConverterTestBase OpConverter_INT32_Test;
// Base class for Unary tests that need to be tested
typedef OpConverter_UnaryTest<float> OpConverter_FP32_UnaryTest;
typedef OpConverter_UnaryTest<int> OpConverter_BOOL_Test;

// Instantiate parameter combinations to OpConverter_<DT_X...>_Test
INSTANTIATE_TEST_CASE_P(
    OpConvTestInstantiation, OpConverter_FP32_Test,
    ::testing::Combine(::testing::ValuesIn(ValidTrtModes),
                       ::testing::Values(DT_FLOAT),
                       ::testing::Values(TrtPrecisionMode::FP32)));

INSTANTIATE_TEST_CASE_P(
    OpConvTestInstantiation, OpConverter_FP32_FP16_Test,
    ::testing::Combine(::testing::ValuesIn(ValidTrtModes),
                       ::testing::Values(DT_FLOAT, DT_HALF),
                       ::testing::Values(TrtPrecisionMode::FP32)));

INSTANTIATE_TEST_CASE_P(
    OpConvTestInstantiation, OpConverter_FP32_FP16_INT32_Test,
    ::testing::Combine(::testing::ValuesIn(ValidTrtModes),
                       ::testing::Values(DT_FLOAT, DT_HALF, DT_INT32),
                       ::testing::Values(TrtPrecisionMode::FP32)));

INSTANTIATE_TEST_CASE_P(
    OpConvTestInstantiation, OpConverter_INT32_Test,
    ::testing::Combine(::testing::ValuesIn(ValidTrtModes),
                       ::testing::Values(DT_INT32),
                       ::testing::Values(TrtPrecisionMode::FP32)));

INSTANTIATE_TEST_CASE_P(
    OpConvTestInstantiation, OpConverter_FP32_UnaryTest,
    ::testing::Combine(::testing::ValuesIn(ValidTrtModes),
                       ::testing::Values(DT_FLOAT),
                       ::testing::Values(TrtPrecisionMode::FP32)));

INSTANTIATE_TEST_CASE_P(
    OpConvTestInstantiation, OpConverter_BOOL_Test,
    ::testing::Combine(::testing::ValuesIn(ValidTrtModes),
                       ::testing::Values(DT_BOOL),
                       ::testing::Values(TrtPrecisionMode::FP32)));

INSTANTIATE_TEST_CASE_P(
    OpConvTestInstantiation, OpConverter_FP32_FP16_BinaryTest,
    ::testing::Combine(::testing::ValuesIn(ValidTrtModes),
                       ::testing::Values(DT_FLOAT, DT_HALF),
                       ::testing::Values(TrtPrecisionMode::FP32)));

INSTANTIATE_TEST_CASE_P(
    OpConvTestInstantiation, OpConverter_BOOL_BinaryTest,
    ::testing::Combine(::testing::ValuesIn(ValidTrtModes),
                       ::testing::Values(DT_BOOL),
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
    EXPECT_THAT(output.weights(), ShapedWeightsHasDimsAndValues<OutputCType>(
                                      expected_dims, expected_value));
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
    std::vector<int> expected_dims{1};
    // Scalars are represented as rank 0 tensors.
    expected_dims.clear();
    reset_and_test(t, false, expected_dims, {12});
    reset_and_test(t, true, expected_dims, {12});
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
                               "Unsupported tensorflow data type double");
  }
  {
    Reset();
    Tensor tensor =
        AsTensor<int64_t>({1, std::numeric_limits<int64_t>::max(), 1, 1, 1,
                           std::numeric_limits<int64_t>::lowest()},
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

template <typename T>
NodeDef CreateFusedBatchNormOp(DataType tf_type, std::string data_format,
                               bool is_training, float epsilon) {
  Scope s = Scope::NewRootScope();
  auto x = ops::Placeholder(s.WithOpName("x"), tf_type);
  auto scale = ops::Placeholder(s.WithOpName("scale"), tf_type);
  auto offset = ops::Placeholder(s.WithOpName("offset"), tf_type);
  auto mean = ops::Placeholder(s.WithOpName("mean"), tf_type);
  auto variance = ops::Placeholder(s.WithOpName("variance"), tf_type);
  typename T::Attrs attrs;
  attrs.data_format_ = data_format;
  attrs.is_training_ = is_training;
  if (epsilon > 0) {
    attrs.epsilon_ = epsilon;
  } else {
    EXPECT_GE(epsilon, 0);
  }
  return T(s.WithOpName("my_batchnorm"), x, scale, offset, mean, variance,
           attrs)
      .operation.node()
      ->def();
}

TEST_P(OpConverter_FP32_Test, ConvertFusedBatchNorm) {
  using OpFunc = std::function<NodeDef(DataType, std::string, bool, float)>;
  std::vector<OpFunc> get_node_def_vec{
      CreateFusedBatchNormOp<ops::FusedBatchNorm>,
      CreateFusedBatchNormOp<ops::FusedBatchNormV2>,
      CreateFusedBatchNormOp<ops::FusedBatchNormV3>};

  struct TestParam {
    std::string data_format;
    int tensor_input_idx;  // Index of an input that will be provided as tensor.
    bool is_training;
    float epsilon;
    Status conversion_status;
    bool keep_channel_unknown;
  };

  struct NodeInput {
    std::string name;
    std::vector<int> dims;
    std::vector<float> val;
  };
  std::vector<NodeInput> node_input_nchw{
      {"x", {2, 3, 2, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
      {"scale", {3}, {7, 8, 9}},
      {"offset", {3}, {10, 20, 30}},
      {"mean", {3}, {1, 2, 3}},
      {"variance", {3}, {4, 5, 6}}};

  std::vector<NodeInput> node_input_nhwc{
      {"x", {2, 2, 1, 3}, {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12}},
      {"scale", {3}, {7, 8, 9}},
      {"offset", {3}, {10, 20, 30}},
      {"mean", {3}, {1, 2, 3}},
      {"variance", {3}, {4, 5, 6}}};

  std::vector<float> expected_output_nchw{
      10.0,    13.495633, 23.574135, 27.148273, 37.342354, 41.013527,
      30.9738, 34.469433, 45.018955, 48.59309,  59.369415, 63.04059};

  std::vector<float> expected_output_nhwc{
      10.0,    23.574135, 37.342354, 13.495633, 27.148273, 41.013527,
      30.9738, 45.018955, 59.369415, 34.469433, 48.59309,  63.04059};

  for (auto get_node_def : get_node_def_vec) {
    NodeDef tmp_node_def = get_node_def(tf_type_, "NCHW", true, 0);
    std::string op_name = tmp_node_def.op();
    std::vector<TestParam> test_param{
        {"NCHW", 0, true, 0,
         errors::Unimplemented(
             StrCat(op_name, " only supports is_training=false"))},
        {"NCHW", 1, false, 0,
         errors::Unimplemented(StrCat("The input \"scale\" for ", op_name,
                                      " must be a constant"))},
        {"NCHW", 2, false, 0,
         errors::Unimplemented(StrCat("The input \"offset\" for ", op_name,
                                      " must be a constant"))},
        {"NCHW", 3, false, 0,
         errors::Unimplemented(StrCat("The input \"mean\" for ", op_name,
                                      " must be a constant"))},
        {"NCHW", 4, false, 0,
         errors::Unimplemented(StrCat("The input \"variance\" for ", op_name,
                                      " must be a constant"))},
        {"NCHW", 0, false, 0.01},
        {"NHWC", 0, false, 0.01}};
    if (trt_mode_ == TrtTestMode::kDynamicShape) {
      test_param.push_back(
          {"NCHW", 0, false, 0.01,
           errors::InvalidArgument("Channel dimension must be static"), true});
      test_param.push_back(
          {"NHWC", 0, false, 0.01,
           errors::InvalidArgument("Channel dimension must be static"), true});
    }
    for (auto p : test_param) {
      Reset();
      NodeDef node_def =
          get_node_def(tf_type_, p.data_format, p.is_training, p.epsilon);
      std::vector<NodeInput> node_input =
          p.data_format == "NCHW" ? node_input_nchw : node_input_nhwc;
      std::vector<float> expected_output =
          p.data_format == "NCHW" ? expected_output_nchw : expected_output_nhwc;
      for (int i = 0; i < node_input.size(); i++) {
        if (i == 0 || i == p.tensor_input_idx) {
          // The first input (x) is always added as a tensor, and it has shape
          // NCHW/NHWC. The other inputs are per channel values (1D, size C).
          //
          // In implicit batch mode, it is not possible to add any of the 1D
          // inputs as a tensor: the first dim is always treated as batch dim in
          // implicit batch mode, and that has to agree for all tensors. We have
          // two input tensors with shapes NCHW and C and in general N != C.
          // The converter already picked up N from the fist input, and reports
          // an error when we try to add any other tensors with not matching
          // first dim.
          //
          // This restriction does not apply in explicit batch mode: the tensors
          // can have different first dim. The converter still expects that only
          // the first arg is a tensor. TODO(tfeher) Check if one can relax this
          // restriction.
          Status expected_status =
              (i != 0 && trt_mode_ == TrtTestMode::kImplicitBatch)
                  ? errors::InvalidArgument(
                        StrCat("Batch size doesn't match for tensor ",
                               node_input[i].name,
                               ": Provided batch size does not match "
                               "converter batch size: 3 vs 2"))
                  : Status::OK();
          std::vector<int> partial_input_shape;
          if (i == 0 && trt_mode_ == TrtTestMode::kDynamicShape &&
              !p.keep_channel_unknown) {
            // keep channel dim static (known)
            partial_input_shape.resize(4, -1);
            int channel_dim = (p.data_format == "NCHW" ? 1 : 3);
            partial_input_shape[channel_dim] = node_input[i].dims[channel_dim];
          }
          AddTestTensor(node_input[i].name, node_input[i].dims, tf_type_,
                        node_input[i].val, partial_input_shape,
                        expected_status);

        } else {
          AddTestWeights(node_input[i].name, node_input[i].dims,
                         node_input[i].val, tf_type_);
        }
      }
      TestOpConverter("my_batchnorm", node_def, node_input[0].dims,
                      p.conversion_status, Status::OK(),
                      ArrayFloatNear(expected_output));
    }
  }
}

TEST_P(OpConverter_FP32_Test, ConvertTranspose) {
  // Get the NodeDef for Transpose.
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), tf_type_);
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
                           "constant")},
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
          (trt_mode_ == TrtTestMode::kImplicitBatch)
              ? Status(error::UNIMPLEMENTED,
                       "Transpose at batch dimension is not supported")
              : Status::OK()},
      TestParamBase{{1, 1, 2, 3}, {}, {1, 3, 1, 2}, {0, 3, 1, 2}},
  };
  if (trt_mode_ == TrtTestMode::kDynamicShape) {
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

TEST_P(OpConverter_FP32_Test, ConvertTile) {
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), tf_type_);
  auto weights = ops::Placeholder(s.WithOpName("weights"), DT_INT32);
  auto tile = ops::Tile(s.WithOpName("my_tile"), input, weights);
  const NodeDef& node_def = tile.operation.node()->def();

  struct TileParam {
    std::vector<int> input_dims;
    std::vector<int> multiplier;
    std::vector<float> tensor;
    // Concrete (static) output dimensions, including batch size as first dim.
    std::vector<int> expected_output_dims;
    std::vector<int> expected_results;
    int test_ID;
    // Expected status of conversion (with concrete error message).
    Status status;
    // Expected status of BuildAndRun.
    Status runtime_status;
  };

  std::vector<TileParam> test_params = {
      TileParam{{1, 2, 3},   // input_dims
                {1, -2, 1},  // multiplier
                {},          // tensor
                {},          // expected_output_dims
                {},          // expected_results
                1,           // test_ID
                Status(error::INVALID_ARGUMENT,
                       "All replications of the Tile operation in "
                       "my_tile should be positive, got (1, -2, 1).")},
      TileParam{
          {1, 2, 3},           // input_dims
          {1, 2, 1, 3},        // multiplier
          {0, 1, 2, 3, 4, 5},  // tensor
          {},                  // expected_output_dims
          {},                  // expected_results
          2,                   // test_ID
          Status(
              error::INVALID_ARGUMENT,
              "The length of the replication vector (4) of the Tile operation "
              "in 'my_tile' "
              "is expected to be equal to the rank of the input vector (3).")},
      TileParam{{1, 2},                                 // input_dims
                {1, 3},                                 // multiplier
                {2, 3},                                 // tensor
                {1, 6},                                 // expected_output_dims
                {2, 3, 2, 3, 2, 3}},                    // out values
      TileParam{{1, 2, 3},                              // input_dims
                {1, 2, 1},                              // multiplier
                {0, 1, 2, 3, 4, 5},                     // tensor
                {1, 4, 3},                              // output dims
                {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5}},  // expected_results
      TileParam{{1, 2, 3},                              // input_dims
                {1, 1, 2},                              // multiplier
                {0, 1, 2, 3, 4, 5},                     // tensor
                {1, 2, 6},                              // expected_output_dims
                {0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5}},  // expected_results
      TileParam{{1, 2, 3},                              // input_dims
                {1, 2, 2},                              // multiplier
                {0, 1, 2, 3, 4, 5},                     // tensor
                {1, 4, 6},                              // expected_output_dims
                {0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5,
                 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5}}  // expected_results
  };

  for (bool multiplier_is_tensor : {true, false}) {
    for (bool input_is_tensor : {true, false}) {
      for (auto p : test_params) {
        std::vector<int> num_mults = {static_cast<int>(p.multiplier.size())};
        std::vector<std::vector<int>> partial_input_dims_options = {{}};
        if (multiplier_is_tensor) {
          if (trt_mode_ == TrtTestMode::kImplicitBatch) {
            p.status =
                Status(error::INVALID_ARGUMENT,
                       "Conversion for Tile is not implemented for multipliers "
                       "passed as a tensor in implicit batch mode");
            num_mults = {1, static_cast<int>(p.multiplier.size())};
          } else {
            if (p.test_ID == 1) {
              // Replacement of statuses, since when multiplier is a tensor AND
              // trt_mode_ != TrtTestMode::kImplicitBatch we cannot define these
              // statuses in ConvertTile::Validate() for first two tests from
              // test_params.
              p.status = Status::OK();
              p.runtime_status =
                  Status(error::INTERNAL,
                         "Incorrect number of profile config parameters");
            }

            if (trt_mode_ == TrtTestMode::kDynamicShape) {
              if (p.test_ID == 1)
                partial_input_dims_options = {num_mults};
              else
                partial_input_dims_options = {{}, num_mults};
            }
          }
        }

        for (auto partial_input_dims : partial_input_dims_options) {
          if (multiplier_is_tensor &&
              trt_mode_ != TrtTestMode::kImplicitBatch) {
            if (trt_mode_ == TrtTestMode::kDynamicShape) {
              if (p.test_ID != 1) {
                p.runtime_status =
                    partial_input_dims.empty()
                        ? Status(error::INTERNAL,
                                 "Failed to build TensorRT engine")
                        : Status::OK();
                p.status = Status::OK();
              }
            }

            if (p.test_ID == 2 && (trt_mode_ == TrtTestMode::kExplicitBatch ||
                                   !partial_input_dims.empty())) {
              p.status = Status(error::INVALID_ARGUMENT,
                                "When replications are defined as a tensor, "
                                "the number of its elements (4) must be equal "
                                "to the rank of the input tensor (3).");
            }
          }

          Reset();
          if (input_is_tensor) {
            AddTestTensor("input", p.input_dims, p.tensor);
          } else {
            AddTestWeights("input", p.input_dims, p.tensor, tf_type_);
          }

          if (multiplier_is_tensor) {
            AddTestTensor<int>("weights", num_mults, DT_INT32, p.multiplier,
                               partial_input_dims);
          } else {
            AddTestWeights<int32>("weights", num_mults, p.multiplier);
          }

          TestOpConverter("my_tile", node_def, p.expected_output_dims, p.status,
                          p.runtime_status,
                          ElementsAreArray(p.expected_results));
        }
      }
    }
  }
}

TEST_P(OpConverter_FP32_Test, ConvertReshape) {
  // Get the NodeDef for Reshape.
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), tf_type_);
  auto weights = ops::Placeholder(s.WithOpName("weights"), DT_INT32);
  auto reshape = ops::Reshape(s.WithOpName("my_reshape"), input, weights);
  const NodeDef& node_def = reshape.operation.node()->def();

  if (trt_mode_ == TrtTestMode::kImplicitBatch) {
    // Shape is a tensor, should fail in implicit batch mode.
    Reset();
    AddTestTensor("input", {3, 2, 1});
    AddTestTensor("weights", {3});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "The input \"shape\" for Reshape must be a constant in implicit batch "
        "mode");
  } else if (!IS_TRT_VERSION_GE(7, 1, 3, 0)) {
    // Shape is a tensor, should fail before TRT 7.1.3 even in explicit batch /
    // dynamic shape mode.
    Reset();
    AddTestTensor("input", {3, 2, 1});
    AddTestTensor("weights", {3});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Non constant shape input tensor for Reshape requires minimum TRT "
        "7.1.3");
  }

  Status reshape_from_scalar_status =
      trt_mode_ == TrtTestMode::kImplicitBatch
          ? errors::Internal(
                "Failed to convert at least one input to a TRT_TensorOrWeights:"
                " Scalar input tensor is not supported since the first "
                "dimension is treated as batch dimension by TRT")
          : Status::OK();
  Status add_scalar_tensor_status =
      trt_mode_ == TrtTestMode::kImplicitBatch
          ? errors::InvalidArgument(
                "removing first dim requires explicit batch dimension")
          : Status::OK();
  Status reshape_to_scalar_status =
      trt_mode_ == TrtTestMode::kImplicitBatch
          ? errors::Unimplemented("Reshape to shape=[] is not supported")
          : Status::OK();
  Status reshape_batch_status =
      trt_mode_ == TrtTestMode::kImplicitBatch
          ? errors::Unimplemented("Reshape on batch dimension is not supported")
          : Status::OK();

  struct TestParams {
    std::vector<int> tensor_dims;
    std::vector<int> shape;
    std::vector<int> expected_shape;
    Status conversion_status;
    Status runtime_status;
    std::vector<int> shape_prof;  // needed concrete values if shape == -1.
    Status add_test_tensor_status;
  };

  std::vector<TestParams> params = {
      // Reshape scalar to tensor, should fail in implicit batch mode.
      TestParams{{},
                 {1, 1},
                 {},
                 reshape_from_scalar_status,
                 {},
                 {},
                 add_scalar_tensor_status},
      // Reshape tensor to scalar, should fail in implicit batch mode.
      // - In explicit batch mode if shape is set as weight it works.
      // - In explicit batch mode && using shape as tensor input it should
      //   fail. In that case we set the expected conversion status in the
      //   test loop.
      TestParams{{1, 1}, {}, {}, reshape_to_scalar_status},
      // Reshape at batch dimension, should fail in implicit batch mode.
      TestParams{{1, 1, 2, 3}, {3, 1, 1, 2}, {}, reshape_batch_status},
      TestParams{{2, 1, 2, 3}, {-1, 1, 4}, {3, 1, 4}, reshape_batch_status},
      // Tests that should succeed in every trt_mode.
      TestParams{{1, 1, 2, 3}, {-1, 1, 3, 2}, {1, 1, 3, 2}},
      TestParams{{1, 1, 2, 3}, {1, 1, -1}, {1, 1, 6}},
      TestParams{{1, 1, 2, 3}, {1, 1, 3, 2}},
      TestParams{{2, 1, 2, 3}, {2, 1, 3, 2}},
      TestParams{{1, 1, 1}, {1}},
      TestParams{{1}, {1, 1}},
      TestParams{{2, 1, 1}, {2}},
      TestParams{{2}, {2, 1}},
  };
  if (trt_mode_ == TrtTestMode::kImplicitBatch) {
    // Reshape tensor with zero rank using an empty shape tensor, should fail in
    // implicit batch mode. In explicit batch mode this is an identity operation
    // and does not add a reshape layer therefore we do not test it.
    params.push_back(TestParams{{},
                                {},
                                {},
                                reshape_from_scalar_status,
                                {},
                                {},
                                add_scalar_tensor_status});
  }
  // Testing the methods for representing the reshape shape for IShuffleLayer:
  // as a weight (true) or as a tensor (false).
  std::vector<bool> shape_input_options(1, true);

  if (trt_mode_ != TrtTestMode::kImplicitBatch &&
      IS_TRT_VERSION_GE(7, 1, 3, 0)) {
    shape_input_options.push_back(false);
  }

  for (auto p : params) {
    for (auto shape_as_weight : shape_input_options) {
      std::ostringstream oss;
      oss << "shape " << PrintToString(p.shape);
      SCOPED_TRACE(StrCat(oss.str(), shape_as_weight ? " weight" : " tensor"));
      if (!shape_as_weight && p.shape.empty()) {
        p.conversion_status = errors::Unimplemented(
            "Reshape with dynamic input requires 1D input tensor");
      }
      Reset();
      const int n_elements =
          std::accumulate(p.tensor_dims.begin(), p.tensor_dims.end(), 1,
                          std::multiplies<int>());
      std::vector<float> input_vec(n_elements);
      std::iota(input_vec.begin(), input_vec.end(), 1);
      AddTestTensor("input", p.tensor_dims, tf_type_, input_vec, {},
                    p.add_test_tensor_status);
      if (shape_as_weight) {
        AddTestWeights<int32>("weights", {static_cast<int>(p.shape.size())},
                              p.shape);
      } else {
        std::vector<int32> dims;
        std::vector<int32> values{p.shape};
        if (!p.shape.empty()) {
          dims.push_back(p.shape.size());
        } else {
          // If the shape is empty we use a dummy value to ensure that
          // AddTestTensor creates the corresponding entry in InputOutputData.
          values.push_back(1);
        }
        AddTestTensor("weights", dims, DT_INT32, values, dims);
      }
      std::vector<int> expected_shape =
          p.expected_shape.empty() ? p.shape : p.expected_shape;
      VLOG(2) << "Calling TestOpConverter";
      TestOpConverter("my_reshape", node_def, expected_shape,
                      p.conversion_status, p.runtime_status,
                      ElementsAreArray(input_vec));
    }
  }
}

TEST_P(OpConverter_FP32_Test, ConvertShape) {
  // Get the NodeDef for Shape op.
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), tf_type_);
  auto shape = ops::Shape(s.WithOpName("my_shape"), input);
  const NodeDef& node_def = shape.operation.node()->def();

  Status conversion_status =
      (trt_mode_ == TrtTestMode::kImplicitBatch)
          ? errors::Unimplemented(
                "Shape is only supported for explicit batch mode.")
          : Status::OK();
  std::vector<TestParamBase> test_params = {
// TODO(b/166274212): Enable the test parameter for TensorRT 7.1.3.
#if !IS_TRT_VERSION_GE(7, 1, 3, 0)
    TestParamBase{{1, 2, 3}, {}, {3}, {}, conversion_status},
#endif
    // Add input as weight (we use non empty param ({1}) to trigger this).
    TestParamBase{{1, 2, 3}, {}, {3}, {1}, conversion_status},
  };

  auto input_is_weight = [](const TestParamBase p) { return !p.param.empty(); };
  for (auto p : test_params) {
    SCOPED_TRACE(p);
    Reset();
    // The number of elements of the input tensor. We leave it 0 in case we do
    // not need to add an input tensor. This happens in explicit batch mode: the
    // shape is known at conversion time and therefore the shape is added to the
    // network as a constant layer. In this case the single node network that
    // we use for the unit test have no actual input tensor when it is converted
    // to a TensorRT network.
    int n_elements = 0;
    if (input_is_weight(p) || trt_mode_ != TrtTestMode::kExplicitBatch) {
      // Calculate the number of elements for adding input data.
      n_elements = std::accumulate(p.input_dims.begin(), p.input_dims.end(), 1,
                                   std::multiplies<int>());
    }
    std::vector<float> input_val(n_elements, 1);
    if (!input_is_weight(p)) {
      AddTestTensor("input", p.input_dims, input_val);
    } else {
      AddTestWeights("input", p.input_dims, input_val, tf_type_);
    }
    TestOpConverter("my_shape", node_def, p.expected_output_dims, p.status,
                    p.runtime_status, ElementsAreArray(p.input_dims),
                    {DT_INT32});
  }
}

struct MatMulTestParams {
  std::vector<int> shape_a;
  std::vector<int> values_a;
  bool transpose_a;
  std::vector<int> shape_b;
  std::vector<int> values_b;
  bool transpose_b;
  std::vector<int> expected_shape;
  std::vector<int> expected_output;
};

// Helper function for testing MatMul and BatchMatMul. get_matmul is a function
// used to generate the node. It accepts (DataType, transpose_a, transpose_b) as
// parameters.
void TestMatMulHelper(
    ParameterizedOpConverterTestBase* test,
    const std::function<NodeDef(DataType, bool, bool)>& get_matmul,
    const std::vector<MatMulTestParams>& params) {
  {
    // Unsupported data type.
    test->Reset();
    NodeDef node_def = get_matmul(DT_INT32, false, false);
    test->AddTestTensor("input", {1, 2}, DT_INT32, {});
    test->AddTestWeights<int32>("weights", {2, 1}, {3, 5});
    test->RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        StrCat("Data type int32 is not supported for ", node_def.op(),
               ", must be one of [float, half]"));
  }

  // FC conversion depends on whether the last dim of A is known or not. In
  // Dynamic shape mode, we will check whether A is handled correctly if it has
  // a partially known input shape (last dim known).
  std::vector<bool> a_test_partial_shape_values{false};
  if (test->get_trt_mode() == TrtTestMode::kDynamicShape) {
    a_test_partial_shape_values.push_back(true);
  }

  for (auto p : params) {
    for (bool a_is_tensor : {true, false}) {
      for (bool b_is_tensor : {true, false}) {
        for (bool a_partial_shape : a_test_partial_shape_values) {
          if (a_partial_shape && !a_is_tensor) {
            // Only tensors can have partial shape.
            continue;
          }
          if (!a_is_tensor && !b_is_tensor) {
            // Skip test when both args are weights. We do not convert this
            // since const folding eliminates this case.
            continue;
          }
          SCOPED_TRACE(StrCat("A", p.transpose_a ? ".T" : "", " is ",
                              a_is_tensor ? "tensor" : "weight", ", B",
                              p.transpose_b ? ".T" : "", " is ",
                              b_is_tensor ? "tensor " : "weight, rank A ",
                              p.shape_a.size(), ", rank B ", p.shape_b.size()));
          test->Reset();

          NodeDef node_def =
              get_matmul(test->get_tf_type(), p.transpose_a, p.transpose_b);
          const bool is_batch_matmul = node_def.op() == "BatchMatMul";

          if (a_is_tensor) {
            if (a_partial_shape) {
              // Prepare a partial shape for A where only the last dim is known.
              std::vector<int> partial_shape(p.shape_a.size(), -1);
              int k = p.shape_a.size() - 1;
              partial_shape.at(k) = p.shape_a.at(k);
              test->AddTestTensor("input", p.shape_a, test->get_tf_type(),
                                  p.values_a, partial_shape);
            } else {
              test->AddTestTensor("input", p.shape_a, p.values_a);
            }
          } else {
            test->AddTestWeights("input", p.shape_a, p.values_a,
                                 test->get_tf_type());
          }
          if (b_is_tensor) {
            if (a_is_tensor && p.shape_a[0] != p.shape_b[0] &&
                test->get_trt_mode() == TrtTestMode::kImplicitBatch) {
              VLOG(2) << "Skipping test with inpcompatible batch dimensions";
              continue;
            }
            test->AddTestTensor("weights", p.shape_b, p.values_b);
          } else {
            test->AddTestWeights("weights", p.shape_b, p.values_b,
                                 test->get_tf_type());
          }

          Status conversion_status = Status::OK();
          if (test->get_trt_mode() == TrtTestMode::kImplicitBatch) {
            // Implicit batch mode has several restriction. We change conversion
            // status accordingly.
            if (is_batch_matmul) {
              if (a_is_tensor && p.shape_a.size() < p.shape_b.size()) {
                conversion_status = errors::InvalidArgument(
                    "Broadcasting beyond batch dimension is not supported "
                    "(tensor #dims ",
                    p.shape_a.size(), " vs broadcast #dims ", p.shape_b.size(),
                    ")");
              }
              if (b_is_tensor && p.shape_b.size() < p.shape_a.size()) {
                conversion_status = errors::InvalidArgument(
                    "Broadcasting beyond batch dimension is not supported "
                    "(tensor #dims ",
                    p.shape_b.size(), " vs broadcast #dims ", p.shape_a.size(),
                    ")");
              }
              if ((!a_is_tensor || !b_is_tensor) && p.shape_a[0] != 1) {
                conversion_status = errors::Unimplemented(
                    "TensorRT does not support batched constants in implicit "
                    "batch mode.");
              }
            } else if ((a_is_tensor && p.shape_a.size() <= 2 &&
                        (p.transpose_a || b_is_tensor)) ||
                       (b_is_tensor && p.shape_b.size() <= 2)) {
              conversion_status = errors::InvalidArgument(
                  "MatMul with 2D tensors requires explicit batch mode, or that"
                  " tensor A is not transposed and B is a constant tensor.");
            }
          }

          test->TestOpConverter("my_matmul", node_def, p.expected_shape,
                                conversion_status, Status::OK(),
                                ElementsAreArray(p.expected_output));
          if (!conversion_status.ok()) {
            VLOG(2) << "Converted with status " << conversion_status;
          }
          VLOG(2) << "== Finished test iteration ==";
        }
      }
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

std::vector<MatMulTestParams> GetMatMulTestParams() {
  std::vector<MatMulTestParams> params{
      // clang-format off
      MatMulTestParams{{2, 2}, {0, 1, 2, 3}, false,  // A (shape, val, T?)
                       {2, 2}, {0, 1, 2, 3}, false,  // B (shape, val, T?)
                       {2, 2}, {2, 3, 6, 11}},       // result (shape, val)
      MatMulTestParams{{2, 2}, {0, 1, 2, 3}, false,
                       {2, 2}, {0, 1, 2, 3},  true,
                       {2, 2}, {1, 3, 3, 13}},
      MatMulTestParams{{2, 2}, {0, 1, 2, 3},  true,
                       {2, 2}, {0, 1, 2, 3}, false,
                       {2, 2}, {4, 6, 6, 10}},
      MatMulTestParams{{2, 2}, {0, 1, 2, 3}, true,
                       {2, 2}, {0, 1, 2, 3}, true,
                       {2, 2}, {2, 6, 3, 11}},
      MatMulTestParams{{2, 3}, {0, 1, 2, 3, 4, 5}, false,
                       {2, 3}, {1, 2, 3, 4, 5, 6}, true,
                       {2, 2}, {8, 17, 26, 62}},
      MatMulTestParams{{2, 3}, {0, 1, 2, 3, 4, 5}, true,
                       {2, 3}, {1, 2, 3, 4, 5, 6}, false,
                       {3, 3}, {12, 15, 18, 17, 22, 27, 22, 29, 36}},
      MatMulTestParams{{3, 2}, {0, 1, 2, 3, 4, 5}, false,
                       {2, 3}, {1, 2, 3, 4, 5, 6}, false,
                       {3, 3}, {4, 5, 6, 14, 19, 24, 24, 33, 42}},
      MatMulTestParams{{3, 2}, {0, 1, 2, 3, 4, 5}, true,
                       {2, 3}, {1, 2, 3, 4, 5, 6}, true,
                       {2, 2}, {16, 34, 22, 49}},
      // clang-format on
  };
  return params;
}

TEST_P(OpConverter_FP32_Test, ConvertMatMul) {
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

  TestMatMulHelper(this, get_matmul_nodedef, GetMatMulTestParams());
}

TEST_P(OpConverter_FP32_Test, ConvertBatchMatMul) {
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

  // We derive test data from the MatMul test params by adding extra leading
  // dimensions.
  std::vector<MatMulTestParams> params_2d = GetMatMulTestParams();
  std::vector<MatMulTestParams> params;
  params.reserve(params_2d.size() * 3 + 1);

  auto insert_ones = [](std::vector<int> v, int n) {
    std::vector<int> ones(n, 1);
    ones.insert(ones.end(), v.begin(), v.end());
    return ones;
  };

  // Add a leading 1 dimension to A, B and result.
  std::transform(params_2d.begin(), params_2d.end(), std::back_inserter(params),
                 [](MatMulTestParams p) {
                   p.shape_a.insert(p.shape_a.begin(), 1);
                   p.shape_b.insert(p.shape_b.begin(), 1);
                   p.expected_shape.insert(p.expected_shape.begin(), 1);
                   return p;
                 });

  // Test with N > 1: weights cannot be batched in implicit batch mode.
  // clang-format off
  params.push_back(
      MatMulTestParams{{2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}, false,  // A
                       {2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}, false,  // B
                       {2, 2, 2}, {2, 3, 6, 11, 2, 3, 6, 11}}       // result
  );

  params.push_back(
      MatMulTestParams{{2, 2, 3}, {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5},
      false,
                       {2, 2, 3}, {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}, true,
                       {2, 2, 2}, {8, 17, 26, 62, 8, 17, 26, 62}});
  // clang-format on

  // Add two leading 1 dimensions to A, B and result.
  std::transform(params_2d.begin(), params_2d.end(), std::back_inserter(params),
                 [insert_ones](MatMulTestParams p) {
                   p.shape_a = insert_ones(p.shape_a, 2);
                   p.shape_b = insert_ones(p.shape_b, 2);
                   p.expected_shape = insert_ones(p.expected_shape, 2);
                   return p;
                 });

  // Test broadcast: add two leading 1 dimensions to A, but not to B.
  std::transform(params_2d.begin(), params_2d.end(), std::back_inserter(params),
                 [insert_ones](MatMulTestParams p) {
                   p.shape_a = insert_ones(p.shape_a, 2);
                   p.expected_shape = insert_ones(p.expected_shape, 2);
                   return p;
                 });

  // Test broadcast: add a leading 1 dimension to A and two leading 1s to B.
  // Broadcasting A need a dynamic brodacast which will be incompatible with
  // FC layer.
  std::transform(params_2d.begin(), params_2d.end(), std::back_inserter(params),
                 [insert_ones](MatMulTestParams p) {
                   p.shape_a = insert_ones(p.shape_a, 1);
                   p.shape_b = insert_ones(p.shape_b, 2);
                   p.expected_shape = insert_ones(p.expected_shape, 2);
                   return p;
                 });

  // Test with N > 1: since weights cannot be batched in implicit batch mode.
  // We tests with batch size 2.
  std::transform(params_2d.begin(), params_2d.end(), std::back_inserter(params),
                 [insert_ones](MatMulTestParams p) {
                   p.shape_a.insert(p.shape_a.begin(), 2);
                   p.values_a.reserve(p.values_a.size() * 2);
                   p.values_a.insert(p.values_a.end(), p.values_a.begin(),
                                     p.values_a.end());

                   p.shape_b.insert(p.shape_b.begin(), 2);
                   p.values_b.reserve(p.values_b.size() * 2);
                   p.values_b.insert(p.values_b.end(), p.values_b.begin(),
                                     p.values_b.end());

                   p.expected_shape.insert(p.expected_shape.begin(), 2);
                   p.expected_output.reserve(p.expected_output.size() * 2);
                   p.expected_output.insert(p.expected_output.end(),
                                            p.expected_output.begin(),
                                            p.expected_output.end());
                   return p;
                 });

  // 4D tensor where the second "batch dim" is not 1
  params.push_back(MatMulTestParams{
      {1, 2, 4, 5},
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
       14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
       28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39},
      false,  // A
      {1, 2, 3, 5},
      {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30},
      true,  // B
      {1, 2, 4, 3},
      {40,   90,   140,  115,  290,  465,  190,  490,
       790,  265,  690,  1115, 1990, 2540, 3090, 2440,
       3115, 3790, 2890, 3690, 4490, 3340, 4265, 5190}});  // result

  TestMatMulHelper(this, get_batch_matmul_nodedef, params);
}

#if IS_TRT_VERSION_GE(7, 1, 3, 0)
TEST_P(OpConverter_FP32_Test, ConvertEinsum) {
  // Get the NodeDef for Einsum.
  auto get_einsum_nodedef = [](DataType dtype, std::string eq,
                               int n_inputs = 2) -> NodeDef {
    Scope s = Scope::NewRootScope();
    auto a = ops::Placeholder(s.WithOpName("input_a"), dtype);
    std::vector<Input> input_vec{a};
    if (n_inputs > 1) {
      auto b = ops::Placeholder(s.WithOpName("input_b"), dtype);
      input_vec.push_back(b);
    }
    InputList inputs(input_vec);
    auto einsum = ops::Einsum(s.WithOpName("my_einsum"), inputs, eq);
    return einsum.operation.node()->def();
  };

  if (trt_mode_ == TrtTestMode::kImplicitBatch) {
    Reset();
    NodeDef node_def = get_einsum_nodedef(tf_type_, "ab,cb->ac");
    AddTestTensor("input_a", {2, 3});
    AddTestTensor("input_b", {2, 3});
    TestOpConverter(
        "my_einsum", node_def, {2, 2},
        errors::Unimplemented("Einsum converter requires dynamic shape mode"),
        Status::OK(), ElementsAreArray({13, 16, 40, 52}));
    // No further tests.
    return;
  }

  struct TestParams {
    std::string equation;
    std::vector<int> shape_a;
    std::vector<int> values_a;
    std::vector<int> shape_b;
    std::vector<int> values_b;
    std::vector<int> expected_shape;
    std::vector<int> expected_output;
    Status conv_status;
  };

  Status unimplemented_eq = errors::Unimplemented("");
  Status internal_eq = errors::Internal("");

  std::vector<TestParams> params{
      // Dot product.
      TestParams{"i,i->", {2}, {2, 3}, {2}, {1, 2}, {1}, {8}, unimplemented_eq},
      // Outer product.
      TestParams{"i,k->ik",
                 {2},
                 {1, 2},
                 {3},
                 {1, 2, 3},
                 {2, 3},
                 {1, 2, 3, 2, 4, 6},
                 unimplemented_eq},
      // Transpose.
      TestParams{"ik->ki",
                 {2, 3},
                 {0, 1, 2, 3, 4, 5},
                 {},
                 {},
                 {3, 2},
                 {0, 3, 1, 4, 2, 5},
                 internal_eq},
      // Diag.
      TestParams{"ii->i",
                 {3, 3},
                 {0, 1, 2, 3, 4, 5, 6, 7, 8},
                 {},
                 {},
                 {3},
                 {0, 4, 8},
                 internal_eq},
      // Trace.
      TestParams{"ii",
                 {3, 3},
                 {0, 1, 2, 3, 4, 5, 6, 7, 8},
                 {},
                 {},
                 {},
                 {12},
                 internal_eq},
      // MatMul with reduction.
      TestParams{"abbc,dc->ad",
                 {1, 2, 2, 3},
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                 {2, 3},
                 {1, 2, 3, 4, 5, 6},
                 {2, 3},
                 {1, 2, 3, 2, 4, 6},
                 unimplemented_eq},
      // Ellipsis with broadcast.
      TestParams{"...ik,...jk->...ij",
                 {1, 3, 1, 4},
                 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                 {2, 1, 1, 4},
                 {1, 2, 3, 4, 5, 6, 7, 8},
                 {2, 3, 1, 1},
                 {20, 60, 100, 44, 148, 252},
                 unimplemented_eq},
      // MatMul
      TestParams{"ab,bc->ac",
                 {2, 3},
                 {0, 1, 2, 3, 4, 5},
                 {3, 2},
                 {1, 2, 3, 4, 5, 6},
                 {2, 2},
                 {13, 16, 40, 52}},
      // Batched MatMul
      TestParams{"abc,cde->abde",
                 /*shape_a=*/{1, 2, 3},
                 /*values_a=*/{0, 1, 2, 3, 4, 5},
                 /*shape_b=*/{3, 2, 2},
                 /*values_v=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                 /*expected_shape=*/{1, 2, 2, 2},
                 /*expected_output=*/{23, 26, 29, 32, 68, 80, 92, 104}},
      TestParams{"abcd,cde->abe",
                 {1, 2, 2, 3},
                 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                 {2, 3, 2},
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                 {1, 2, 2},
                 {125, 140, 341, 392}},
      TestParams{"abc,cd->abd",
                 {1, 2, 3},
                 {0, 1, 2, 3, 4, 5},
                 {3, 2},
                 {1, 2, 3, 4, 5, 6},
                 {1, 2, 2},
                 {13, 16, 40, 52}},
      TestParams{"acbe,aecd->abcd",
                 {1, 2, 3, 4},
                 {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
                 {1, 4, 2, 3},
                 {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                  13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
                 {1, 3, 2, 3},
                 {90, 96, 102, 732, 786, 840, 250, 272, 294, 940, 1010, 1080,
                  410, 448, 486, 1148, 1234, 1320}},
      TestParams{"aecd,abcd->acbe",
                 {1, 2, 3, 4},
                 {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
                 {1, 2, 3, 4},
                 {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                  13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
                 {1, 3, 2, 2},
                 {20, 140, 92, 788, 148, 460, 412, 1300, 404, 908, 860, 1940}},
      TestParams{"acd,dce->ae",
                 {1, 2, 3},
                 {0, 1, 2, 3, 4, 5},
                 {3, 2, 2},
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                 {1, 2},
                 {115, 130}},
      TestParams{"abcd,bace->bade",
                 {2, 3, 2, 1},
                 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                 {3, 2, 2, 1},
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                 {3, 2, 1, 1},
                 {2, 46, 28, 128, 86, 242}},
      TestParams{
          "cebfad,fageb->abcdg",
          {1, 1, 3, 3, 2, 2},
          {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
           12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
           24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35},
          {3, 2, 2, 1, 3},
          {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
           13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
           25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36},
          {2, 3, 1, 2, 2},
          {252, 288, 291, 336, 768,  912,  810,  963,  1356, 1608, 1401, 1662,
           438, 492, 495, 558, 1176, 1338, 1236, 1407, 1986, 2256, 2049, 2328}},
  };

  for (auto p : params) {
    for (bool a_is_tensor : {true, false}) {
      for (bool b_is_tensor : {true, false}) {
        if (!a_is_tensor && !b_is_tensor) {
          // Skip test when both args are weights. We do not convert this
          // since const folding eliminates this case.
          continue;
        }
        Reset();
        int n_inputs = p.shape_b.empty() ? 1 : 2;
        NodeDef node_def = get_einsum_nodedef(tf_type_, p.equation, n_inputs);
        if (a_is_tensor) {
          AddTestTensor("input_a", p.shape_a, p.values_a);
        } else {
          AddTestWeights("input_a", p.shape_a, p.values_a, tf_type_);
        }
        if (!p.shape_b.empty()) {
          if (b_is_tensor) {
            AddTestTensor("input_b", p.shape_b, p.values_b);
          } else {
            AddTestWeights("input_b", p.shape_b, p.values_b, tf_type_);
          }
        }
        TestOpConverter("my_einsum", node_def, p.expected_shape, p.conv_status,
                        Status::OK(), ElementsAreArray(p.expected_output));
      }
    }
  }
}
#endif  // IS_TRT_VERSION_GE(7, 1, 3, 0)

TEST_P(OpConverter_FP32_FP16_Test, ConvertBiasAdd) {
  // Note that kINT32 is not supported by IScaleLayer, so we don't test
  // DT_INT32 type here. DT_FLOAT and DT_HALF are tested.
  // Get the NodeDef for BiasAdd.
  auto get_biasadd_nodedef = [](const string& data_format,
                                DataType tf_type) -> NodeDef {
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), tf_type);
    auto weights = ops::Placeholder(s.WithOpName("weights"), tf_type);
    const auto biasadd_attrs = ops::BiasAdd::DataFormat(data_format);
    auto biasadd =
        ops::BiasAdd(s.WithOpName("my_biasadd"), input, weights, biasadd_attrs);
    return biasadd.operation.node()->def();
  };

  for (const string& data_format : {"NHWC", "NCHW"}) {
    for (const int trt_input_rank : {1, 2, 3, 4}) {
      Reset();
      NodeDef node_def = get_biasadd_nodedef(data_format, tf_type_);

      // Add input, dims_array will be like {2, 1, ..., 1, 3}
      std::vector<int32> dims_array(trt_input_rank + 1, 1);
      if (trt_input_rank == 1) {
        dims_array[1] = (data_format == "NHWC" ? 3 : 2);
      } else {
        dims_array[1] = 2;
        dims_array[trt_input_rank] = 3;
      }
      const int64_t num_input = DimsAdapter(dims_array).Volume();
      ASSERT_EQ(trt_input_rank > 1 ? 6 : (data_format == "NHWC" ? 3 : 2),
                num_input);
      std::vector<float> input_data(num_input, 0);

      AddTestTensor("input", dims_array, input_data);

      const int channel_size = (data_format == "NHWC" ? 3 : 2);
      std::vector<float> bias(channel_size);
      for (int i = 0; i < channel_size; ++i) {
        bias[i] = i + 1;  // bias will be {1, 2, 3, ...}
      }
      AddTestWeights("weights", {channel_size}, bias, tf_type_);

      // Build and run the engine.
      std::vector<float> output_data;

      if (trt_input_rank == 1) {
        if (data_format == "NHWC") {
          output_data = {1, 2, 3};
        } else {
          output_data = {1, 2};
        }
      } else {
        if (data_format == "NHWC") {
          output_data = {1, 2, 3, 1, 2, 3};
        } else {
          output_data = {1, 1, 1, 2, 2, 2};
        }
      }
      TestOpConverter("my_biasadd", node_def, dims_array, Status::OK(),
                      Status::OK(), ElementsAreArray(output_data));
    }
  }
}

template <typename OpType>
NodeDef GetBinaryOpNodeDef(DataType dtype) {
  Scope s = Scope::NewRootScope();
  auto input_l = ops::Placeholder(s.WithOpName("input1"), dtype);
  auto input_r = ops::Placeholder(s.WithOpName("input2"), dtype);
  auto op = OpType(s.WithOpName("my_binary"), input_l, input_r);
  return op.operation.node()->def();
}

TEST_P(OpConverter_FP32_FP16_BinaryTest, ConvertBinary) {
  using OpFunc = std::function<NodeDef(DataType)>;
  std::map<std::string, std::pair<OpFunc, std::vector<float>>> op_test_info;
#define ADD_OP(name, op, v1, v2, v3, v4, v5, v6, v7, v8) \
  op_test_info[name] =                                   \
      std::make_pair(GetBinaryOpNodeDef<op>,             \
                     std::vector<float>(v1, v2, v3, v4, v5, v6, v7, v8))
  ADD_OP("Add", ops::Add, {5, 8, 6, 9, 5, 8, 6, 9});
  ADD_OP("AddV2", ops::AddV2, {5, 8, 6, 9, 5, 8, 6, 9});
  ADD_OP("Sub", ops::Sub, {1, 4, 0, 3, 1, 4, 0, 3});
  ADD_OP("Mul", ops::Mul, {6, 12, 9, 18, 6, 12, 9, 18});
  ADD_OP("Div", ops::Div, {1.5, 3, 1, 2, 1.5, 3, 1, 2});
  ADD_OP("RealDiv", ops::RealDiv, {1.5, 3, 1, 2, 1.5, 3, 1, 2});
  ADD_OP("FloorDiv", ops::FloorDiv, {1, 3, 1, 2, 1, 3, 1, 2});
  ADD_OP("Minimum", ops::Minimum, {2, 2, 3, 3, 2, 2, 3, 3});
  ADD_OP("Maximum", ops::Maximum, {3, 6, 3, 6, 3, 6, 3, 6});
  ADD_OP("Pow", ops::Pow, {9, 36, 27, 216, 9, 36, 27, 216});
#undef ADD_OP
  std::vector<std::vector<float>> data = {
      {3, 6, 3, 6}, {3, 6}, {2, 3, 2, 3}, {2, 3}};
  RunTests(*BinaryOperationMap(), op_test_info, data);
}

TEST_P(OpConverter_BOOL_BinaryTest, ConvertBooleanBinary) {
  using OpFunc = std::function<NodeDef(DataType)>;
  std::map<std::string, std::pair<OpFunc, std::vector<int>>> op_test_info;
#define ADD_OP(name, op, v1, v2, v3, v4, v5, v6, v7, v8) \
  op_test_info[name] =                                   \
      std::make_pair(GetBinaryOpNodeDef<op>,             \
                     std::vector<int>(v1, v2, v3, v4, v5, v6, v7, v8))
  ADD_OP("LogicalOr", ops::LogicalOr, {1, 1, 0, 1, 1, 1, 0, 1});
  ADD_OP("LogicalAnd", ops::LogicalAnd, {0, 1, 0, 0, 0, 1, 0, 0});
#undef ADD_OP
#if IS_TRT_VERSION_GE(8, 2, 0, 0)
  std::vector<std::vector<int>> data = {
      {0, 1, 0, 1}, {0, 1}, {1, 0, 1, 0}, {1, 0}};
  RunTests(*BinaryBooleanOperationMap(), op_test_info, data);
#endif
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

struct AddNTestParams {
  std::vector<float> input_values;
  std::vector<string> input_names;
  std::vector<int> dimensions;
  std::vector<float> expected_output;
  Status status;
};

void TestAddN(ParameterizedOpConverterTestBase* test, AddNTestParams& p) {
  // All inputs are tensors.
  test->Reset();
  const NodeDef node_def = GetAddNNodeDef(p.input_names, test->get_tf_type());

  if (p.input_values.size() % p.input_names.size() != 0) {
    LOG(ERROR) << "The number of input values: `" << p.input_values.size()
               << "` is not a multiple of the number of inputs: `"
               << p.input_names.size() << "`";
    ASSERT_TRUE(false);
  }

  DataVec input_data;
  int input_offset = 0;
  const int window_size = p.input_values.size() / p.input_names.size();
  for (const string& name : p.input_names) {
    std::vector<float>::const_iterator start_pos =
        p.input_values.begin() + input_offset;
    std::vector<float>::const_iterator end_pos = start_pos + window_size;
    std::vector<float> sub_input_val(start_pos, end_pos);
    input_offset += window_size;

    test->AddTestTensor(name, p.dimensions, test->get_tf_type(), sub_input_val);
  }

  test->TestOpConverter("my_addn", node_def, p.dimensions,
                        /*expected_conversion_status=*/p.status,
                        /*expected_runtime_status=*/p.status,
                        /*matcher=*/ElementsAreArray(p.expected_output),
                        /*out_tf_types=*/{test->get_tf_type()});
}

TEST_P(OpConverter_FP32_FP16_Test, ConvertAddN) {
  {
    // Weights with batch dim that is not 1.
    Reset();
    const NodeDef node_def = GetAddNNodeDef({"tensor", "weights"}, tf_type_);
    AddTestTensor("tensor", /*dims=*/{1, 2});
    AddTestWeights<float>("weights", {2, 1, 2}, {0, 1, 2, 3});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Weights input to AddN is required to have batch dimension 1.");
  }

  const std::vector<float> common_input = CreateVectorIota<float>(6);

  std::vector<AddNTestParams> params = {
      {/*input_values=*/common_input,
       /*input_names=*/{"inp1", "inp2", "inp3"},
       /*dimensions=*/{1, 1, 2, 1, 1},
       /*expected_output=*/{6, 9},
       /*status=*/Status::OK()},
      {/*input_values=*/common_input,
       /*input_names=*/{"inp1", "inp2"},
       /*dimensions=*/{1, 1, 3, 1, 1},
       /*expected_output=*/{3, 5, 7},
       /*status=*/Status::OK()},
      {/*input_values=*/common_input,
       /*input_names=*/{"inp1", "inp2", "inp3"},
       /*dimensions=*/{1, 2, 1, 1},
       /*expected_output=*/{6, 9},
       /*status=*/Status::OK()},
      {/*input_values=*/common_input,
       /*input_names=*/{"inp1", "inp2"},
       /*dimensions=*/{1, 1, 3, 1},
       /*expected_output=*/{3, 5, 7},
       /*status=*/Status::OK()},
      {/*input_values=*/common_input,
       /*input_names=*/{"inp1", "inp2", "inp3"},
       /*dimensions=*/{1, 2, 1},
       /*expected_output=*/{6, 9},
       /*status=*/Status::OK()},
      {/*input_values=*/common_input,
       /*input_names=*/{"inp1", "inp2"},
       /*dimensions=*/{1, 1, 3},
       /*expected_output=*/{3, 5, 7},
       /*status=*/Status::OK()},
      {/*input_value=*/common_input,
       /*input_names=*/{"inp1", "inp2", "inp3"},
       /*dimensions=*/{2, 1},
       /*expected_output=*/{6, 9},
       /*status=*/Status::OK()},
      {/*input_values=*/common_input,
       /*input_names=*/{"inp1", "inp2"},
       /*dimensions=*/{1, 3},
       /*expected_output=*/{3, 5, 7},
       /*status=*/Status::OK()},
      {/*input_values=*/common_input,
       /*input_names=*/{"inp1", "inp2", "inp3"},
       /*dimensions=*/{2},
       /*expected_output=*/{6, 9},
       /*status=*/Status::OK()},
      {/*input_values=*/common_input,
       /*input_names=*/{"inp1", "inp2"},
       /*dimensions=*/{3},
       /*expected_output=*/{3, 5, 7},
       /*status=*/Status::OK()},
      {/*input_values=*/common_input,
       /*input_names=*/{"inp1", "inp2", "inp3", "inp4", "inp5", "inp6"},
       /*dimensions=*/{1},
       /*expected_output=*/{15},
       /*status=*/Status::OK()},
  };

  for (auto p : params) {
    TestAddN(this, p);
  }
}

TEST_P(OpConverter_FP32_Test, ConvertQDQDynamicRangeMode) {
  {
    // FakeQuantWithMinMaxArgs attributes are empty, should fail.
    Reset(TrtPrecisionMode::INT8);
    NodeDef node_def =
        MakeNodeDef("my_quantize", "FakeQuantWithMinMaxArgs", {"input"});
    AddTestTensor("input", {1, 2, 3});
    RunValidationAndConversion(node_def, error::NOT_FOUND,
                               "No attr named 'min'");
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
    EXPECT_EQ(1, ranges.count(output.tensor()->trt_tensor()));
    EXPECT_EQ(6.0f, ranges[output.tensor()->trt_tensor()]);
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
    EXPECT_EQ(1, ranges.count(output.tensor()->trt_tensor()));
    EXPECT_EQ(6.0f, ranges[output.tensor()->trt_tensor()]);
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
    EXPECT_EQ(1, ranges.count(output.tensor()->trt_tensor()));
    EXPECT_EQ(6.0f, ranges[output.tensor()->trt_tensor()]);
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
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "The input \"input_min\" for "
                               "QuantizeAndDequantizeV2 must be a constant");
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
    EXPECT_EQ(1, ranges.count(output.tensor()->trt_tensor()));
    EXPECT_EQ(6.0f, ranges[output.tensor()->trt_tensor()]);
  }
}

TEST_P(OpConverter_FP32_FP16_Test, ConvertSquare) {
  {
    // Input is weights, should fail.
    Reset();
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), tf_type_);
    auto square = ops::Square(s.WithOpName("my_square"), input);
    NodeDef node_def = square.operation.node()->def();
    AddTestWeights("input", {1, 2, 3}, {1, 2, 3, 4, -5, 6}, tf_type_);
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "The input \"x\" for Square must be a tensor");
  }

  Reset();

  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), tf_type_);
  auto square = ops::Square(s.WithOpName("my_square"), input);
  NodeDef node_def = square.operation.node()->def();

  const int num_inputs = 20;
  std::vector<float> inputs(num_inputs);
  std::vector<float> expected_outputs(num_inputs);

  for (int i = 0; i < num_inputs; ++i) {
    const float value = (i - 9);
    inputs[i] = value;
    expected_outputs[i] = value * value;
  }
  AddTestTensor("input", {1, 1, 20}, tf_type_, inputs);

  TestOpConverter("my_square", node_def, {1, 1, 20}, Status::OK(), Status::OK(),
                  ArrayFloatNear(expected_outputs, 0));
}

#if IS_TRT_VERSION_GE(8, 2, 0, 0)

TEST_P(OpConverter_FP32_FP16_INT32_Test, ConvertFill) {
  Scope s = Scope::NewRootScope();
  auto dims = ops::Placeholder(s.WithOpName("dims"), DT_INT32);
  auto value = ops::Placeholder(s.WithOpName("value"), tf_type_);
  auto fill = ops::Fill(s.WithOpName("my_fill"), dims, value);
  const NodeDef& node_def = fill.operation.node()->def();

  if (trt_mode_ == TrtTestMode::kImplicitBatch) {
    Reset();
    // random data
    AddTestWeights("dims", {2}, {2, 2}, DT_INT32);
    AddTestWeights("value", {1}, {42.0}, tf_type_);
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "Conversion for Fill is not implemented in "
                               "implicit batch mode");
    return;
  }

  std::vector<std::vector<int>> output_dims_params = {
      {8}, {8, 2, 4}, {32, 32, 3200}};
  std::vector<std::vector<int>> value_dims_params = {{}, {1}};

  float val = 42.0;
  Status status = Status::OK();
  for (bool dims_is_tensor : {true, false}) {
    for (bool value_is_tensor : {true, false}) {
      for (auto output_dims : output_dims_params) {
        for (auto value_dims : value_dims_params) {
          Reset();
          std::vector<int32> dims_dims = {output_dims.size()};
          if (dims_is_tensor) {
            AddTestTensor("dims", dims_dims, DT_INT32, output_dims, dims_dims);
          } else {
            AddTestWeights("dims", dims_dims, output_dims, DT_INT32);
          }
          if (value_is_tensor) {
            AddTestTensor("value", value_dims, tf_type_, {val});
          } else {
            AddTestWeights("value", value_dims, {val}, tf_type_);
          }
          size_t nb_el = 1;
          for (auto d : output_dims) {
            nb_el *= d;
          }
          std::vector<float> expected_output(nb_el, val);
          TestOpConverter("my_fill", node_def, output_dims, status, status,
                          ElementsAreArray(expected_output));
        }
      }
    }
  }
}

TEST_P(OpConverter_FP32_FP16_INT32_Test, ConvertRange) {
  auto get_casted_value = [this](const float value, const DataType dtype) {
    return dtype == DT_INT32 ? static_cast<int32>(value) : value;
  };

  auto set_parameters = [this](const std::array<const char*, 3>& name,
                               const std::array<std::vector<float>, 3>& value,
                               const std::array<DataType, 3>& type,
                               bool all_tensors = false, int shape_idx = -1) {
    Reset();
    for (int i = 0; i < 3; i++) {
      if (all_tensors) {
        std::vector<int32> partial_shape_dims = {};
        // The correct partial shape will be provided
        // (a) for all parameters, when shape_idx > 3
        // (b) for all parameters, except shape_idx, when shape_idx >= 0
        // (c) for none of the shape_idx < 0
        if (shape_idx > 3 || shape_idx >= 0 && shape_idx != i) {
          partial_shape_dims = {1};
        }
        AddTestTensor(name[i], {1}, type[i], value[i], partial_shape_dims);
      } else {
        AddTestWeights(name[i], {1}, value[i], type[i]);
      }
    }
  };

  const float start = 1.0;
  const float limit = 43.0;
  const float delta = 2.0;

  const std::array<const char*, 3> param_name = {"start", "limit", "delta"};
  std::array<std::vector<float>, 3> param_value;
  param_value[0] = {start};
  param_value[1] = {limit};
  param_value[2] = {delta};
  const auto start_type = tf_type_;
  std::array<DataType, 3> param_type = {tf_type_, tf_type_, tf_type_};

  Scope s = Scope::NewRootScope();
  const auto range =
      ops::Range(s.WithOpName("my_range"),
                 ops::Placeholder(s.WithOpName(param_name[0]), param_type[0]),
                 ops::Placeholder(s.WithOpName(param_name[1]), param_type[1]),
                 ops::Placeholder(s.WithOpName(param_name[2]), param_type[2]));

  const NodeDef& node_def = range.operation.node()->def();
  const std::vector<DataType> param_types{DT_FLOAT, DT_HALF, DT_INT32};

  // ConverterRange is not implemented for Implicite batch mode.
  if (trt_mode_ == TrtTestMode::kImplicitBatch) {
    for (bool all_tensors : {false, true}) {
      set_parameters(param_name, param_value, param_type, all_tensors);
      RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                                 "Conversion for Range is not implemented in "
                                 "implicit batch mode");
    }
    return;
  }

  const std::string expected_msg = convert_range_expected_msg(node_def);
  {
    // We expect that all three (start, limit and delta) are passed as weights
    // OR tensors and we reject parameters, if it's not true.
    Reset();
    // Passing (start, limit) as weights
    for (int i = 0; i < 2; i++) {
      AddTestWeights(param_name[i], {1}, param_value[i], param_type[i]);
    }
    // ... and delta as a tensor
    AddTestTensor(param_name[2], {1}, param_type[2], param_value[2]);

    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               expected_msg + "passed as weights OR tensors");
  }

  nvinfer1::DataType trt_type;
  TF_ASSERT_OK(TfTypeToTrtType(tf_type_, &trt_type));
  const std::string expected = DebugString(trt_type);

  // Reject invalid parameters if delta = 0  (for weights only).
  for (auto limit_type : param_types) {
    param_type[1] = limit_type;
    for (auto delta_type : param_types) {
      param_type[2] = delta_type;
      param_value[2] = {0};

      set_parameters(param_name, param_value, param_type);
      RunValidationAndConversion(
          node_def, error::INVALID_ARGUMENT,
          "The delta parameter of Range operation cannot be equal to 0");

      // Reject invalid parameters preventing the limit from
      // being reached for fixed values of start and delta.
      for (int j = 0; j <= 1; j++) {
        param_value[j] = {get_casted_value(start, tf_type_)};
        param_value[1 - j] = {get_casted_value(limit, limit_type)};
        param_value[2] = {(2 * j - 1) * get_casted_value(delta, delta_type)};
        set_parameters(param_name, param_value, param_type);
        const auto error = convert_range_error_msg(
            param_value[0][0], param_value[1][0], param_value[2][0]);
        RunValidationAndConversion(node_def, error::INVALID_ARGUMENT, error);
      }

      param_value[0] = {start};
      // When passed as tensors, all parameters should be of DT_INT32 type.
      if (start_type == DT_INT32 && limit_type == DT_INT32 &&
          delta_type == DT_INT32) {
        if (trt_mode_ == TrtTestMode::kDynamicShape) {
          // Wrong dimension for one of parameters.
          for (int j = 0; j < 3; j++) {
            const string err =
                StrCat("Dimension for '", param_name[j],
                       "' of Range operator should be equal to 1");
            set_parameters(param_name, param_value, param_type, true, j);
            RunValidationAndConversion(node_def, error::INVALID_ARGUMENT, err);
          }
        }
      } else {
        // When at least one parameter is set as non-integer tensors,
        // the following test should fail.
        set_parameters(param_name, param_value, param_type, true);
        RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                                   expected_msg + "tensors");
      }
    }
  }

  // The tests that pass all checks in ConvertRange::Validate().
  const Status status = Status::OK();
  const std::vector<DataType> int_type{DT_INT32};
  for (bool all_tensors : {false, true}) {
    // For now when (start, limit, delta) are passed as tensors
    // these tensors should be of DT_INT32 type.
    int partial_shape_idx = -1;
    if (all_tensors) {
      if (start_type != DT_INT32) {
        continue;
      }
      if (trt_mode_ == TrtTestMode::kDynamicShape) {
        // The correct partial shape will be provided for all parameters
        partial_shape_idx = 3;
      }
    }

    // For now only parameters of DT_INT32 type could be used when
    // they are pased as tensors.
    const auto& types = all_tensors ? int_type : param_types;
    const auto jEnd = all_tensors ? 0 : 1;
    for (auto limit_type : types) {
      param_type[1] = limit_type;
      for (auto delta_type : types) {
        param_type[2] = delta_type;
        // Loop for positive and negative deltas.
        for (int j = 0; j <= jEnd; j++) {
          // Define the expected result which should match the usage
          // of DT_INT32 for one of (start, limit, delta).
          const int mult = (1 - 2 * j);
          param_value[j] = {get_casted_value(start, tf_type_)};
          param_value[1 - j] = {get_casted_value(limit, limit_type)};
          param_value[2] = {mult * get_casted_value(delta, delta_type)};

          // Create expected output.
          std::vector<float> expected_output;
          const float limit_curr = param_value[1][0];
          const float delta_curr = param_value[2][0];
          float value = param_value[0][0];
          int num_values = 0;
          while (mult * (limit_curr - value) > 0) {
            num_values++;
            expected_output.push_back(value);
            value += delta_curr;
          }

          set_parameters(param_name, param_value, param_type, all_tensors,
                         partial_shape_idx);
          const std::vector<int> output_dims = {num_values};
          TestOpConverter("my_range", node_def, output_dims, status, status,
                          ElementsAreArray(expected_output));
        }
      }
    }
  }
}

TEST_P(OpConverter_FP32_FP16_INT32_Test, ConvertLikeOps) {
  auto get_node = [&](int value) -> NodeDef {
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), tf_type_);
    if (value == 0) {
      auto zeros_like = ops::ZerosLike(s.WithOpName("Zeros"), input);
      return zeros_like.operation.node()->def();
    }
    auto ones_like = ops::OnesLike(s.WithOpName("Ones"), input);
    return ones_like.operation.node()->def();
  };

  for (int value : {0, 1}) {
    Reset();
    const NodeDef& node_def = get_node(value);
    const std::string name = value ? "Ones" : "Zeros";

    if (trt_mode_ == TrtTestMode::kImplicitBatch) {
      std::vector<float> input_data(8, 42.0f);
      AddTestTensor("input", {8}, tf_type_, input_data);
      RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                                 "Conversion for " + name + "Like is not " +
                                     "implemented in implicit batch mode");
      continue;
    }

    std::vector<std::vector<int>> output_dims_params = {
        {8}, {8, 2, 4}, {32, 32, 3200}};

    float val = 42.0;
    Status status = Status::OK();
    for (bool input_is_tensor : {true, false}) {
      for (auto output_dims : output_dims_params) {
        Reset();
        size_t nb_el = 1;
        for (auto d : output_dims) {
          nb_el *= d;
        }
        std::vector<float> input_data(nb_el, val);
        if (input_is_tensor) {
          AddTestTensor("input", output_dims, tf_type_, input_data);
        } else {
          AddTestWeights("input", output_dims, input_data, tf_type_);
        }
        std::vector<float> expected_output(nb_el, value);
        TestOpConverter(name, node_def, output_dims, status, status,
                        ElementsAreArray(expected_output));
      }
    }
  }
}

#endif  // IS_TRT_VERSION_GE(8, 2, 0, 0)

#if IS_TRT_VERSION_GE(8, 2, 1, 6) || defined(TF_TRT_USE_EFFICIENT_NMS_PLUGIN)

TEST_P(OpConverter_FP32_Test, ConvertCombinedNMS) {
  // Get the NodeDef for CombinedNMS.
  auto get_nms_nodedef = [](DataType tf_type, bool clip_boxes = true,
                            bool pad_per_class = false) -> NodeDef {
    Scope s = Scope::NewRootScope();
    auto boxes_tensor = ops::Placeholder(s.WithOpName("boxes"), tf_type);
    auto scores_tensor = ops::Placeholder(s.WithOpName("scores"), tf_type);
    auto max_output_size_per_class =
        ops::Placeholder(s.WithOpName("max_output_size_per_class"), DT_INT32);
    auto max_total_size =
        ops::Placeholder(s.WithOpName("max_total_size"), DT_INT32);
    auto iou_threshold =
        ops::Placeholder(s.WithOpName("iou_threshold"), tf_type);
    auto score_threshold =
        ops::Placeholder(s.WithOpName("score_threshold"), tf_type);
    auto nms_attrs = ops::CombinedNonMaxSuppression::Attrs()
                         .PadPerClass(pad_per_class)
                         .ClipBoxes(clip_boxes);

    auto nms_op = ops::CombinedNonMaxSuppression(
        s.WithOpName("my_nms"), boxes_tensor, scores_tensor,
        max_output_size_per_class, max_total_size, iou_threshold,
        score_threshold, nms_attrs);
    return nms_op.operation.node()->def();
  };

  struct TestParams {
    const std::string description;
    const std::vector<int32> boxes_tensor_dims;
    const std::vector<int32> scores_tensor_dims;
    const std::vector<float> boxes_values;
    const std::vector<float> scores_values;
    const int32 max_output_size_per_class;
    const int32 max_total_size;
    const float iou_threshold;
    const float score_threshold;
    bool pad_per_class;
    bool clip_boxes;
    const std::vector<std::vector<int32>> expected_output_dims;
    const std::vector<float> exp_boxes;
    const std::vector<float> exp_scores;
    const std::vector<float> exp_classes;
    const std::vector<float> exp_num_detections;
    Status conversion_status;
    Status runtime_status;
  };

  Status conv_status =
      trt_mode_ == TrtTestMode::kImplicitBatch
          ? errors::Unimplemented(
                "Implict batch mode not supported with CombinedNMS")
          : Status::OK();

  std::vector<TestParams> params = {
      TestParams{"Test 1: clip boxes",
                 {1, 1, 3, 4},  // boxes dims
                 {1, 1, 3},     // scores dims
                                // boxes values:
                 {0, 0, 0.3, 1.4, 0, 0, 0.3, 1.4, 0, 0, 0.3, 1.4},
                 {0.4, 0.7, 0.3},  // scores values
                 3,                // max_output_size_per_class
                 2,                // max_total_size
                 0.1,              // IOU threshold
                 0,                // score_threshold
                 false,            // pad_per_class
                 true,             // clip_boxes
                 {{1, 2, 4},       // expected_nmsed_boxes_dims
                  {1, 2},          // expected_nmsed_scores_dims
                  {1, 2},          // expected_nmsed_classes_dims
                  {1}},            // expected_valid_detections_dims
                                   // exp_boxes_values:
                 {0, 0, 0.3, 1.0, 0, 0, 0.3, 1.0},
                 {0.7, 0.4},  // exp_scores
                 {1, 0},      // exp_classes
                 {2},         // exp_num_detections
                 conv_status},
      TestParams{
          "Test 2: iou threshold",
          {1, 5, 1, 4},  // boxes dims
          {1, 5, 1},     // scores dims
                         // boxes values:
          {0, 0, 5, 10, 0, 1, 5, 11, 8, 0, 12, 4, 6, 2, 10, 6, 8, 9, 11, 12},
          {5, 4, 3, 2, 1},  // scores values
          4,                // max_output_size_per_class
          4,                // max_total_size
          0.7,              // IOU threshold
          0,                // score threshold
          false,            // pad_per_class
          false,            // clip_boxes
          {{1, 4, 4},       // expected nmsed_boxes_dims
           {1, 4},          // expected nmsed_scores_dims
           {1, 4},          // expected_nmsed_classes_dims
           {1}},            // expected_valid_detections_dims
                            // exp_boxes_values:
          {0, 0, 5, 10, 8, 0, 12, 4, 6, 2, 10, 6, 8, 9, 11, 12},
          {5, 3, 2, 1},  // exp_scores
          {0, 0, 0, 0},  // exp_classes
          {4},           // exp_num_detections
          conv_status},
      TestParams{
          "Test 3: score threshold",
          {1, 5, 1, 4},  // boxes dims
          {1, 5, 1},     // scores dims
                         // boxes values:
          {0, 0, 5, 10, 0, 1, 5, 11, 8, 0, 12, 4, 6, 2, 10, 6, 8, 9, 11, 12},
          {5, 4, 3, 2, 1},  // scores values
          4,                // max_output_size_per_class
          4,                // max_total_size
          0.1,              // IOU threshold
          2,                // score threshold
          false,            // pad_per_class
          false,            // clip_boxes
          {{1, 4, 4},       // expected nmsed_boxes_dims
           {1, 4},          // expected nmsed_scores_dims
           {1, 4},          // expected_nmsed_classes_dims
           {1}},            // expected_valid_detections_dims
                            // exp_boxes_values:
          {0, 0, 5, 10, 8, 0, 12, 4, 0, 0, 0, 0, 0, 0, 0, 0},
          {5, 3, 0, 0},  // exp_scores
          {0, 0, 0, 0},  // exp_classes
          {2},           // exp_num_detections
          conv_status},
      TestParams{
          "Test 4: per class size and pad",
          {1, 5, 1, 4},  // boxes dims
          {1, 5, 2},     // scores dims
                         // boxes values:
          {0, 0, 5, 10, 0, 1, 5, 11, 8, 0, 12, 4, 6, 2, 10, 6, 8, 9, 11, 12},
          // scores values:
          {5, 0, 0, 4, 3, 0, 2, 0, 1, 0},
          1,           // max_output_size_per_class
          4,           // max_total_size
          0.1,         // IOU threshold
          0,           // score threshold
          true,        // pad_per_class
          false,       // clip_boxes
          {{1, 2, 4},  // expected nmsed_boxes_dims
           {1, 2},     // expected nmsed_scores_dims
           {1, 2},     // expected_nmsed_classes_dims
           {1}},       // expected_valid_detections_dims
                       // exp_boxes_values:
          {0, 0, 5, 10, 0, 1, 5, 11},
          {5, 4},  // exp_scores
          {0, 1},  // exp_classes
          {2},     // exp_num_detections
          conv_status},
      TestParams{
          "Test 5: different box coordinate order",
          {1, 5, 1, 4},  // boxes dims
          {1, 5, 2},     // scores dims
                         // boxes values:
          {5, 10, 0, 0, 5, 11, 0, 1, 12, 4, 8, 0, 10, 6, 6, 2, 11, 12, 8, 9},
          // scores values:
          {5, 0, 0, 4, 3, 0, 2, 0, 1, 0},
          1,           // max_output_size_per_class
          4,           // max_total_size
          0.1,         // IOU threshold
          0,           // score threshold
          true,        // pad_per_class
          false,       // clip_boxes
          {{1, 2, 4},  // expected nmsed_boxes_dims
           {1, 2},     // expected nmsed_scores_dims
           {1, 2},     // expected_nmsed_classes_dims
           {1}},       // expected_valid_detections_dims
                       // exp_boxes_values:
          {5, 10, 0, 0, 5, 11, 0, 1},
          {5, 4},  // exp_scores
          {0, 1},  // exp_classes
          {2},     // exp_num_detections
          conv_status},
  };

  for (auto p : params) {
    Reset();
    SCOPED_TRACE(p.description);
    AddTestTensor("boxes", p.boxes_tensor_dims, p.boxes_values);
    AddTestTensor("scores", p.scores_tensor_dims, p.scores_values);
    AddTestWeights<int32>("max_output_size_per_class", {1},
                          {p.max_output_size_per_class});
    AddTestWeights<int32>("max_total_size", {1}, {p.max_total_size});
    AddTestWeights<float>("iou_threshold", {1}, {p.iou_threshold}, tf_type_);
    AddTestWeights<float>("score_threshold", {1}, {p.score_threshold},
                          tf_type_);

    auto node_def = get_nms_nodedef(tf_type_, p.clip_boxes, p.pad_per_class);

    TestOpConverterMultiOut("my_nms", node_def, p.expected_output_dims,
                            p.conversion_status, p.runtime_status,
                            {
                                ElementsAreArray(p.exp_boxes),
                                ElementsAreArray(p.exp_scores),
                                ElementsAreArray(p.exp_classes),
                                ElementsAreArray(p.exp_num_detections),
                            },
                            {tf_type_, tf_type_, tf_type_, DT_INT32});
  }
}

#elif IS_TRT_VERSION_GE(7, 1, 3, 0)

TEST_P(OpConverter_FP32_Test, ConvertCombinedNMS) {
  // Get the NodeDef for CombinedNMS.
  auto get_nms_nodedef = [](DataType tf_type, bool clip_boxes = true,
                            bool pad_per_class = false) -> NodeDef {
    Scope s = Scope::NewRootScope();
    auto boxes_tensor = ops::Placeholder(s.WithOpName("boxes"), tf_type);
    auto scores_tensor = ops::Placeholder(s.WithOpName("scores"), tf_type);
    auto max_output_size_per_class =
        ops::Placeholder(s.WithOpName("max_output_size_per_class"), DT_INT32);
    auto max_total_size =
        ops::Placeholder(s.WithOpName("max_total_size"), DT_INT32);
    auto iou_threshold =
        ops::Placeholder(s.WithOpName("iou_threshold"), tf_type);
    auto score_threshold =
        ops::Placeholder(s.WithOpName("score_threshold"), tf_type);
    auto nms_attrs = ops::CombinedNonMaxSuppression::Attrs()
                         .PadPerClass(pad_per_class)
                         .ClipBoxes(clip_boxes);

    auto nms_op = ops::CombinedNonMaxSuppression(
        s.WithOpName("my_nms"), boxes_tensor, scores_tensor,
        max_output_size_per_class, max_total_size, iou_threshold,
        score_threshold, nms_attrs);
    return nms_op.operation.node()->def();
  };

  struct TestParams {
    const std::string description;
    const std::vector<int32> boxes_tensor_dims;
    const std::vector<int32> scores_tensor_dims;
    const std::vector<float> boxes_values;
    const std::vector<float> scores_values;
    const int32 max_output_size_per_class;
    const int32 max_total_size;
    const float iou_threshold;
    const float score_threshold;
    bool pad_per_class;
    bool clip_boxes;
    const std::vector<std::vector<int32>> expected_output_dims;
    const std::vector<float> exp_boxes;
    const std::vector<float> exp_scores;
    const std::vector<float> exp_classes;
    const std::vector<float> exp_num_detections;
    Status conversion_status;
    Status runtime_status;
  };

  Status conv_status =
      trt_mode_ == TrtTestMode::kDynamicShape
          ? errors::Unimplemented(
                "TensorRT BatchedNMS Plugin requires input with static shape")
          : Status::OK();

  std::vector<TestParams> params = {
      // TODO(aaroey): there is a bug in TRT's CombinedNonMaxSuppression
      // implementation that, the extra output classes that are outside of the
      // range specified by valid_detections[i] are not zeros but -1s.
      TestParams{
          "Test 1: Original test",
          {1, 1, 3, 4},                                      // boxes dims
          {1, 1, 3},                                         // scores dims
          {0, 0, 0.3, 0.4, 0, 0, 0.3, 0.4, 0, 0, 0.3, 0.4},  // boxes values
          {0.4, 0.7, 0.3},                                   // scores values
          3,                                 // max_output_size_per_class
          2,                                 // max_total_size
          .5f,                               // IOU threshold
          0,                                 // score_threshold
          false,                             // pad_per_class
          true,                              // clip_boxes
          {{1, 2, 4},                        // expected_nmsed_boxes_dims
           {1, 2},                           // expected_nmsed_scores_dims
           {1, 2},                           // expected_nmsed_classes_dims
           {1}},                             // expected_valid_detections_dims
          {0, 0, 0.3, 0.4, 0, 0, 0.3, 0.4},  // exp_boxes_values
          {0.7, 0.4},                        // exp_scores
          {1, 0},                            // exp_classes
          {2},                               // exp_num_detections
          conv_status},
      // Test with clip_boxes = False
      TestParams{
          "Test 2: clip_boxes",
          {1, 5, 1, 4},  // boxes dims
          {1, 5, 1},     // scores dims
          // boxes values:
          {0, 0, 5, 10, 0, 4, 5, 14, 8, 0, 12, 4, 6, 2, 10, 6, 8, 9, 11, 12},
          {5, 4, 3, 2, 1},  // scores values
          4,                // max_output_size_per_class
          4,                // max_total_size
          0.1,              // IOU threshold
          0,                // score threshold
          false,            // pad_per_class
          false,            // clip_boxes
          {{1, 4, 4},       // expected nmsed_boxes_dims
           {1, 4},          // expected nmsed_scores_dims
           {1, 4},          // expected_nmsed_classes_dims
           {1}},            // expected_valid_detections_dims
                            // exp_boxes_values:
          {0, 0, 5, 10, 8, 0, 12, 4, 8, 9, 11, 12, 0, 0, 0, 0},
          {5, 3, 1, 0},   // exp_scores
          {0, 0, 0, -1},  // exp_classes
          {3},            // exp_num_detections
          conv_status},
      // Test with clip_boxes = False, and nonzero score threshold
      TestParams{
          "Test 3: score threshold",
          {1, 5, 1, 4},  // boxes dims
          {1, 5, 1},     // scores dims
          // boxes values:
          {0, 0, 5, 10, 0, 4, 5, 14, 8, 0, 12, 4, 6, 2, 10, 6, 8, 9, 11, 12},
          {5, 4, 3, 2, 1},  // scores values
          4,                // max_output_size_per_class
          4,                // max_total_size
          0.1,              // IOU threshold
          2,                // score threshold
          false,            // pad_per_class
          false,            // clip_boxes
          {{1, 4, 4},       // expected nmsed_boxes_dims
           {1, 4},          // expected nmsed_scores_dims
           {1, 4},          // expected_nmsed_classes_dims
           {1}},            // expected_valid_detections_dims
                            // exp_boxes_values:
          {0, 0, 5, 10, 8, 0, 12, 4, 0, 0, 0, 0, 0, 0, 0, 0},
          {5, 3, 0, 0},    // exp_scores
          {0, 0, -1, -1},  // exp_classes
          {2},             // exp_num_detections
          conv_status},
      // Test where the boxes are defined as with max value first for the box
      // coordinates. This test fails before TRT 7.1.3.
      TestParams{
          "Test 4: max coord first",
          {1, 5, 1, 4},  // boxes dims
          {1, 5, 1},     // scores dims
                         // boxes values:
          {5, 10, 0, 0, 5, 14, 0, 4, 12, 4, 8, 0, 10, 6, 6, 2, 11, 12, 8, 9},
          {5, 4, 3, 2, 1},  // scores values
          4,                // max_output_size_per_class
          4,                // max_total_size
          0.1,              // IOU threshold
          0,                // score threshold
          false,            // pad_per_class
          false,            // clip_boxes
          {{1, 4, 4},       // expected nmsed_boxes_dims
           {1, 4},          // expected nmsed_scores_dims
           {1, 4},          // expected_nmsed_classes_dims
           {1}},            // expected_valid_detections_dims
                            // exp_boxes_values:
          {5, 10, 0, 0, 12, 4, 8, 0, 11, 12, 8, 9, 0, 0, 0, 0},
          {5, 3, 1, 0},   // exp_scores
          {0, 0, 0, -1},  // exp_classes
          {3},            // exp_num_detections
          conv_status},
      TestParams{"Test 5: TopK error",
                 {1, 5000, 1, 4},  // boxes dims
                 {1, 5000, 1},     // scores dims
                 {},               // boxes values:
                 {},               // scores values
                 4,                // max_output_size_per_class
                 4,                // max_total_size
                 0.1,              // IOU threshold
                 0,                // score threshold
                 false,            // pad_per_class
                 false,            // clip_boxes
                 {},               // expected_valid_detections_dims
                 {},               // exp_boxes_values
                 {},               // exp_scores
                 {},               // exp_classes
                 {},               // exp_num_detections
                 conv_status.ok()
                     ? errors::InvalidArgument(
                           "TRT NMS plugin allow top_k<=4096, where top_k = "
                           "max(num_boxes, max_total_size). You can override "
                           "this by setting TF_TRT_ALLOW_NMS_TOPK_OVERRIDE=1 "
                           "environment variable, but this can result in a "
                           "loss of accuracy.")
                     : conv_status},
  };

  for (auto p : params) {
    Reset();
    SCOPED_TRACE(p.description);
    AddTestTensor("boxes", p.boxes_tensor_dims, p.boxes_values);
    AddTestTensor("scores", p.scores_tensor_dims, p.scores_values);
    AddTestWeights<int32>("max_output_size_per_class", {1},
                          {p.max_output_size_per_class});
    AddTestWeights<int32>("max_total_size", {1}, {p.max_total_size});
    AddTestWeights<float>("iou_threshold", {1}, {p.iou_threshold}, tf_type_);
    AddTestWeights<float>("score_threshold", {1}, {p.score_threshold},
                          tf_type_);

    auto node_def = get_nms_nodedef(tf_type_, p.clip_boxes, p.pad_per_class);

    TestOpConverterMultiOut("my_nms", node_def, p.expected_output_dims,
                            p.conversion_status, p.runtime_status,
                            {
                                ElementsAreArray(p.exp_boxes),
                                ElementsAreArray(p.exp_scores),
                                ElementsAreArray(p.exp_classes),
                                ElementsAreArray(p.exp_num_detections),
                            },
                            {tf_type_, tf_type_, tf_type_, DT_INT32});
  }
}

#endif  // IS_TRT_VERSION_GE(7, 1, 3, 0)

template <typename T>
NodeDef CreateUnaryOp(DataType tf_type) {
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), tf_type);
  return T(s.WithOpName("my_unary"), input).operation.node()->def();
}

constexpr float kLeakyReluAlpha = 0.2f;
template <>
NodeDef CreateUnaryOp<ops::internal::LeakyRelu>(DataType tf_type) {
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), tf_type);
  return ops::internal::LeakyRelu(
             s.WithOpName("my_unary"), input,
             ops::internal::LeakyRelu::Alpha(kLeakyReluAlpha))
      .operation.node()
      ->def();
}

TEST_P(OpConverter_FP32_UnaryTest, ConvertActivation) {
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

  // std::exp in Softplus will overflow for input > 88
  const std::vector<float> input = {-100, -2, -1, 0, 1, 88};
  const bool nan_sensitive = false;

#if IS_TRT_VERSION_GE(8, 0, 0, 0)
  // NVBug # 3322482 - Known bug with TRT 8.0 on specific GPU architectures
  const float max_abs_error = 1e-4;
#else
  const float max_abs_error = 0.;
#endif
  RunTests("Activation", *ActivationTypeMap(), op_map, input, "input",
           max_abs_error, nan_sensitive);
}

TEST_P(OpConverter_FP32_Test, ConvertExpandDims) {
  // Get the NodeDef for ExpandDims.
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), tf_type_);
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
                               "tensor");
  }
  {
    // Axis is a tensor, should fail.
    Reset();
    AddTestTensor("input", {3, 2, 1});
    AddTestTensor("weights", {3});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "The input \"axis\" for ExpandDims must be a "
                               "constant");
  }
  std::vector<TestParamBase> test_params = {
      TestParamBase{{1, 1, 2, 3},
                    {},
                    {1, 1, 1, 2, 3},
                    {0},
                    trt_mode_ == TrtTestMode::kImplicitBatch
                        ? Status(error::UNIMPLEMENTED,
                                 "TensorRT does not allow manipulation of the "
                                 "batch dimension")
                        : Status::OK()},
      TestParamBase{{1, 1, 2, 3},
                    {},
                    {1, 1, 1, 2, 3},
                    {-5},
                    trt_mode_ == TrtTestMode::kImplicitBatch
                        ? Status(error::UNIMPLEMENTED,
                                 "TensorRT does not allow manipulation of the "
                                 "batch dimension")
                        : Status::OK()},
      TestParamBase{{1, 1, 2, 3},
                    {},
                    {},
                    {5},
                    Status(error::INVALID_ARGUMENT,
                           "Axis value of 5 is out of bounds, must be in range"
                           " [-5, 5)")},
      TestParamBase{{1, 1, 2, 3},
                    {},
                    {},
                    {-6},
                    Status(error::INVALID_ARGUMENT,
                           "Axis value of -6 is out of bounds, must be in range"
                           " [-5, 5)")},
      TestParamBase{{1, 2, 3}, {}, {1, 1, 2, 3}, {1}},
      TestParamBase{{1, 2, 3}, {}, {1, 1, 2, 3}, {-3}},
      TestParamBase{{1, 2, 3}, {}, {1, 2, 3, 1}, {3}},
      TestParamBase{{1, 2, 3}, {}, {1, 2, 3, 1}, {-1}},
      TestParamBase{{1, 2, 3}, {}, {1, 2, 1, 3}, {2}},
      TestParamBase{{1, 2, 3}, {}, {1, 2, 1, 3}, {-2}},
      TestParamBase{{1, 6}, {}, {1, 1, 6}, {1}},
      TestParamBase{{1, 6}, {}, {1, 6, 1}, {-1}},
  };
  for (auto p : test_params) {
    Reset();
    AddTestTensor("input", p.input_dims, {1, 2, 3, 4, 5, 6});
    AddTestWeights<int32>("weights", {1}, {p.param[0]});
    TestOpConverter("my_expanddims", node_def, p.expected_output_dims, p.status,
                    p.runtime_status, ElementsAreArray({1, 2, 3, 4, 5, 6}));
  }
}

TEST_P(OpConverter_FP32_FP16_Test, ConvertSoftmax) {
  // Get the NodeDef for SoftMax.
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("logits"), tf_type_);
  auto softmax = ops::Softmax(s.WithOpName("my_softmax"), input);
  const NodeDef& node_def = softmax.operation.node()->def();

  struct TestParams {
    std::vector<int> input_dims;
    std::vector<float> expected_values;
  };
  std::vector<TestParams> test_params = {
      TestParams{{2, 3},
                 {0.09003057, 0.24472848, 0.66524094, 0.09003057, 0.24472848,
                  0.66524094}},
      TestParams{{6, 1}, {1, 1, 1, 1, 1, 1}},  // works with std input
      TestParams{{1, 6},  // this works with arange(1,7) input
                 {0.00426978, 0.01160646, 0.03154963, 0.08576079, 0.23312202,
                  0.6336913}},
  };
  std::vector<float> input_values{1, 2, 3, 4, 5, 6};
  for (auto p : test_params) {
    Reset();
    AddTestTensor("logits", p.input_dims, input_values);
    TestOpConverter("my_softmax", node_def, p.input_dims, Status::OK(),
                    Status::OK(), ArrayFloatNear(p.expected_values, 1e-3));
  }
}

TEST_P(OpConverter_FP32_Test, ConvertSqueeze) {
  const bool use_implicit_batch = (trt_mode_ == TrtTestMode::kImplicitBatch);
  // Get the NodeDef for Squeeze.
  auto get_squeeze_nodedef = [](std::vector<int> axes,
                                DataType tf_type) -> NodeDef {
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), tf_type);
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
          trt_mode_ == TrtTestMode::kExplicitBatch
              ? Status::OK()
              : Status{error::UNIMPLEMENTED,
                       "Squeeze is not implemented for empty squeeze_dims"}},
      TestParamBase{{1, 2, 1, 3},
                    {},
                    {2, 1, 3},
                    {0},
                    use_implicit_batch
                        ? Status{error::UNIMPLEMENTED,
                                 "TensorRT does not allow manipulation of the "
                                 "batch dimension"}
                        : Status::OK()},
      TestParamBase{{1, 2, 1, 3},
                    {},
                    {2, 1, 3},
                    {-4},
                    use_implicit_batch
                        ? Status{error::UNIMPLEMENTED,
                                 "TensorRT does not allow manipulation of the "
                                 "batch dimension"}
                        : Status::OK()},
      TestParamBase{
          {1, 1, 2, 3},
          {},
          {},
          {4},
          Status{error::INVALID_ARGUMENT,
                 "Axis value of 4 is out of bounds, must be in range [-4, 4)"}},
      TestParamBase{
          {1, 1, 2, 3},
          {},
          {},
          {-5},
          Status{
              error::INVALID_ARGUMENT,
              "Axis value of -5 is out of bounds, must be in range [-4, 4)"}},
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
             "size 1"}};

  if (trt_mode_ == TrtTestMode::kDynamicShape) {
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
    NodeDef node_def = get_squeeze_nodedef(p.param, tf_type_);
    AddTestTensor("input", p.input_dims, {1, 2, 3, 4, 5, 6},
                  p.partial_input_dims);
    TestOpConverter("my_squeeze", node_def, p.expected_output_dims, p.status,
                    p.runtime_status, ElementsAreArray({1, 2, 3, 4, 5, 6}));
  }
}

TEST_P(OpConverter_FP32_FP16_INT32_Test, ConvertStridedSlice) {
  // Get nodedef for StridedSlice layer.
  auto get_strided_slice_nodedef =
      [](DataType tf_type, int64 begin_mask = 0, int64 end_mask = 0,
         int64 ellipsis_mask = 0, int64 new_axis_mask = 0,
         int64 shrink_axis_mask = 0) -> NodeDef {
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), tf_type);
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
    NodeDef node_def = get_strided_slice_nodedef(tf_type_);
    AddTestWeights<int32>("input", {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6});
    AddTestWeights<int32>("begin", {4}, {0, 0, 0, 0});
    AddTestWeights<int32>("end", {4}, {1, 1, 2, 3});
    AddTestWeights<int32>("strides", {4}, {1, 1, 1, 1});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "The input \"input\" for StridedSlice must "
                               "be a tensor");
  }
  {
    // Begin, end, strides are tensors, should fail.
    Reset();
    NodeDef node_def = get_strided_slice_nodedef(tf_type_);
    AddTestTensor("input", {4, 1, 1, 1});
    AddTestTensor("begin", {4});
    AddTestTensor("end", {4});
    AddTestTensor("strides", {4});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"begin\" for StridedSlice must be a constant");
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
    Status conversion_status;
    Status runtime_status;
    std::vector<int> partial_input_dims;
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

  Status modified_batch_dim_status =
      (trt_mode_ == TrtTestMode::kImplicitBatch)
          ? errors::Unimplemented(
                "TensorRT does not allow modifications to "
                "the batch dimension")
          : Status::OK();
  std::vector<TestParams> params = {
      // Modify batch dim, should fail in implicit batch mode.
      TestParams{/*input_dims=*/{2, 1, 1, 3},
                 /*begin=*/{0, 0, 0, 0},
                 /*end=*/{1, 1, 1, 2},
                 /*strides=*/{1, 1, 1, 1},
                 /*begin_mask=*/get_mask({0, 0, 0, 0}),
                 /*end_mask=*/get_mask({0, 0, 0, 0}),
                 /*ellipsis_mask=*/0,
                 /*new_axis_mask=*/0,
                 /*shrink_axis_mask=*/0,
                 /*expected_output_dims=*/{1, 1, 1, 2},
                 /*expected_output=*/{1, 2},
                 /*conversion_status=*/modified_batch_dim_status,
                 /*runtime_status=*/Status::OK(),
                 /*partial_input_dims=*/{}},
      // Unknown batch size without end_mask.
      TestParams{
          /*input_dims=*/{2, 1, 1, 3},
          /*begin=*/{0, 0, 0, 0},
          /*end=*/{1, 1, 1, 2},
          /*strides=*/{1, 1, 1, 1},
          /*begin_mask=*/get_mask({0, 0, 0, 0}),
          /*end_mask=*/get_mask({0, 0, 0, 0}),
          /*ellipsis_mask=*/0,
          /*new_axis_mask=*/0,
          /*shrink_axis_mask=*/0,
          /*expected_output_dims=*/{1, 1, 1, 2},
          /*expected_output=*/{1, 2},
          modified_batch_dim_status,
          Status::OK(),
          /*partial_input_dims=*/{-1, 1, 1, 3},
      },
      // Test Case 2: Unknown batch size with end_mask.
      TestParams{
          /*input_dims=*/{2, 1, 1, 3},
          /*begin=*/{0, 0, 0, 0},
          /*end=*/{0, 1, 1, 2},
          /*strides=*/{1, 1, 1, 1},
          /*begin_mask=*/get_mask({1, 0, 0, 0}),
          /*end_mask=*/get_mask({1, 0, 0, 0}),
          /*ellipsis_mask=*/0,
          /*new_axis_mask=*/0,
          /*shrink_axis_mask=*/0,
          /*expected_output_dims=*/{2, 1, 1, 2},
          /*expected_output=*/{1, 2, 4, 5},
          Status::OK(),
          Status::OK(),
          /*partial_input_dims=*/{-1, 1, 1, 3},
      },
      // Invalid parameters: end[2] < begin[2]
      TestParams{/*input_dims=*/{1, 1, 2, 3},
                 /*begin=*/{0, 0, 2, 0},
                 /*end=*/{1, 1, 0, 3},
                 /*strides=*/{1, 1, 1, 1},
                 /*begin_mask=*/0,
                 /*end_mask=*/0,
                 /*ellipsis_mask=*/0,
                 /*new_axis_mask=*/0,
                 /*shrink_axis_mask=*/0,
                 /*expected_output_dims=*/{},
                 /*expected_output=*/{},
                 errors::InvalidArgument("\"size\" cannot be negative for "
                                         "StridedSlice"),
                 Status::OK(),
                 /*partial_input_dims=*/{}},
      // Slice on the last two dimensions. All dimensions are static.
      TestParams{
          /*input_dims=*/{1, 1, 2, 3},
          /*begin=*/{0, 0, 0, 0},
          /*end=*/{0, 0, 1, 2},
          /*strides=*/{1, 1, 1, 1},
          /*begin_mask=*/get_mask({0, 0, 0, 0}),
          /*end_mask=*/get_mask({1, 1, 0, 0}),
          /*ellipsis_mask=*/0,
          /*new_axis_mask=*/0,
          /*shrink_axis_mask=*/0,
          /*expected_output_dims=*/{1, 1, 1, 2},
          /*expected_output=*/{1, 2},
      },
      // Slice on the last two dimensions. The slice is fully
      // specified for the dynamic dimensions.
      TestParams{
          /*input_dims=*/{1, 1, 2, 3},
          /*begin=*/{0, 0, 0, 0},
          /*end=*/{0, 0, 1, 2},
          /*strides=*/{1, 1, 1, 1},
          /*begin_mask=*/get_mask({0, 0, 0, 0}),
          /*end_mask=*/get_mask({1, 1, 0, 0}),
          /*ellipsis_mask=*/0,
          /*new_axis_mask=*/0,
          /*shrink_axis_mask=*/0,
          /*expected_output_dims=*/{1, 1, 1, 2},
          /*expected_output=*/{1, 2},
          Status::OK(),
          Status::OK(),
          /*partial_input_dims=*/{1, 1, -1, -1},
      },
      // End mask is provided on all dimensions. This should override the fact
      // that the end value is 0. For dynamic shape, it tests
      // that we can infer tensor size when "end mask" is provided.
      TestParams{
          /*input_dims=*/{1, 1, 2, 3},
          /*begin=*/{0, 0, 1, 1},
          /*end=*/{0, 0, 0, 0},
          /*strides=*/{1, 1, 1, 1},
          /*begin_mask=*/get_mask({0, 0, 0, 0}),
          /*end_mask=*/get_mask({1, 1, 1, 1}),
          /*ellipsis_mask=*/0,
          /*new_axis_mask=*/0,
          /*shrink_axis_mask=*/0,
          /*expected_output_dims=*/{1, 1, 1, 2},
          /*expected_output=*/{5, 6},
          Status::OK(),
          Status::OK(),
          /*partial_input_dims=*/{1, 1, -1, -1},
      },
      // End mask is provided for the batch dimension to overwrite the end value
      // 0 for that dimension.
      TestParams{
          /*input_dims=*/{1, 1, 2, 3},
          /*begin=*/{0, 0, 1, 1},
          /*end=*/{0, 1, 2, 3},
          /*strides=*/{1, 1, 1, 1},
          /*begin_mask=*/get_mask({0, 0, 0, 0}),
          /*end_mask=*/get_mask({1, 1, 0, 0}),
          /*ellipsis_mask=*/0,
          /*new_axis_mask=*/0,
          /*shrink_axis_mask=*/0,
          /*expected_output_dims=*/{1, 1, 1, 2},
          /*expected_output=*/{5, 6},
      },
      // Test slice on two dimensions with negative stride, without end_mask set
      // on crop dimensions.
      TestParams{/*input_dims=*/{1, 1, 2, 3},
                 /*begin=*/{0, 0, 1, 2},
                 /*end=*/{0, 0, 0, 0},
                 /*strides=*/{1, 1, -1, -1},
                 /*begin_mask=*/get_mask({0, 0, 0, 0}),
                 /*end_mask=*/get_mask({1, 1, 0, 0}),
                 /*ellipsis_mask=*/0,
                 /*new_axis_mask=*/0,
                 /*shrink_axis_mask=*/0,
                 /*expected_output_dims=*/{1, 1, 1, 2},
                 /*expected_output=*/{6, 5},
                 /*conversion_status=*/Status::OK(),
                 /*runtime_status=*/Status::OK(),
                 /*partial_input_dims=*/{1, 1, -1, -1}},
      // Test slice on two dimensions with negative stride, with end_mask set on
      // crop dimensions. In dynamic shape mode, this tests the runtime size
      // computation.
      TestParams{/*input_dims=*/{1, 1, 2, 3},
                 /*begin=*/{0, 0, 1, 1},
                 /*end=*/{0, 0, 0, 0},
                 /*strides=*/{1, 1, -1, -1},
                 /*begin_mask=*/get_mask({0, 0, 0, 0}),
                 /*end_mask=*/get_mask({1, 1, 1, 1}),
                 /*ellipsis_mask=*/0,
                 /*new_axis_mask=*/0,
                 /*shrink_axis_mask=*/0,
                 /*expected_output_dims=*/{1, 1, 2, 2},
                 /*expected_output=*/{5, 4, 2, 1},
                 /*conversion_status=*/Status::OK(),
                 /*runtime_status=*/Status::OK(),
                 /*partial_input_dims=*/{1, 1, -1, -1}},
      // Test slice on two dimensions with negative stride, with begin_mask set
      // on the crop dimensions. In dynamic shape mode, this tests the runtime
      // size computation.
      TestParams{/*input_dims=*/{1, 1, 2, 3},
                 /*begin=*/{0, 0, 0, 0},
                 /*end=*/{0, 0, 0, 0},
                 /*strides=*/{1, 1, -1, -1},
                 /*begin_mask=*/get_mask({0, 0, 1, 1}),
                 /*end_mask=*/get_mask({1, 1, 0, 0}),
                 /*ellipsis_mask=*/0,
                 /*new_axis_mask=*/0,
                 /*shrink_axis_mask=*/0,
                 /*expected_output_dims=*/{1, 1, 1, 2},
                 /*expected_output=*/{6, 5},
                 /*conversion_status=*/Status::OK(),
                 /*runtime_status=*/Status::OK(),
                 /*partial_input_dims=*/{1, 1, -1, -1}},
      // Test the reversal of all non-batch dimensions by providing the begin
      // masks, end masks, and -1 as strides.
      TestParams{/*input_dims=*/{1, 1, 2, 3},
                 /*begin=*/{0, 0, 0, 0},
                 /*end=*/{0, 0, 0, 0},
                 /*strides=*/{1, -1, -1, -1},
                 /*begin_mask=*/get_mask({1, 1, 1, 1}),
                 /*end_mask=*/get_mask({1, 1, 1, 1}),
                 /*ellipsis_mask=*/0,
                 /*new_axis_mask=*/0,
                 /*shrink_axis_mask=*/0,
                 /*expected_output_dims=*/{1, 1, 2, 3},
                 /*expected_output=*/{6, 5, 4, 3, 2, 1},
                 /*conversion_status=*/Status::OK(),
                 /*runtime_status=*/Status::OK(),
                 /*partial_input_dims=*/{1, -1, -1, -1}},
      // Slice on dimensions 1 and 2.
      TestParams{
          /*input_dims=*/{1, 2, 3, 1},
          /*begin=*/{0, 0, 0, 0},
          /*end=*/{0, 1, 2, 1},
          /*strides=*/{1, 1, 1, 1},
          /*begin_mask=*/get_mask({0, 0, 0, 0}),
          /*end_mask=*/get_mask({1, 0, 0, 0}),
          /*ellipsis_mask=*/0,
          /*new_axis_mask=*/0,
          /*shrink_axis_mask=*/0,
          /*expected_output_dims=*/{1, 1, 2, 1},
          /*expected_output=*/{1, 2},
      },
      // Slice on dimensions 1 and 2.
      TestParams{
          /*input_dims=*/{1, 2, 3, 1},
          /*begin=*/{0, 1, 1, 0},
          /*end=*/{0, 2, 3, 1},
          /*strides=*/{1, 1, 1, 1},
          /*begin_mask=*/get_mask({0, 0, 0, 0}),
          /*end_mask=*/get_mask({1, 0, 0, 0}),
          /*ellipsis_mask=*/0,
          /*new_axis_mask=*/0,
          /*shrink_axis_mask=*/0,
          /*expected_output_dims=*/{1, 1, 2, 1},
          /*expected_output=*/{5, 6},
      },
      // Slice on dimensions 1 and 3.
      TestParams{
          /*input_dims=*/{1, 2, 1, 3},
          /*begin=*/{0, 0, 0, 0},
          /*end=*/{0, 1, 1, 2},
          /*strides=*/{1, 1, 1, 1},
          /*begin_mask=*/get_mask({0, 0, 0, 0}),
          /*end_mask=*/get_mask({1, 0, 0, 0}),
          /*ellipsis_mask=*/0,
          /*new_axis_mask=*/0,
          /*shrink_axis_mask=*/0,
          /*expected_output_dims=*/{1, 1, 1, 2},
          /*expected_output=*/{1, 2},
      },
      // Slice on dimensions 1 and 3 with non-zero slice start.
      TestParams{
          /*input_dims=*/{1, 2, 1, 3},
          /*begin=*/{0, 1, 0, 1},
          /*end=*/{0, 2, 1, 3},
          /*strides=*/{1, 1, 1, 1},
          /*begin_mask=*/get_mask({0, 0, 0, 0}),
          /*end_mask=*/get_mask({1, 0, 0, 0}),
          /*ellipsis_mask=*/0,
          /*new_axis_mask=*/0,
          /*shrink_axis_mask=*/0,
          /*expected_output_dims=*/{1, 1, 1, 2},
          /*expected_output=*/{5, 6},
      },
      // Slice on 3D tensor.
      TestParams{
          /*input_dims=*/{1, 2, 3},
          /*begin=*/{0, 0, 0},
          /*end=*/{0, 1, 2},
          /*strides=*/{1, 1, 1},
          /*begin_mask=*/get_mask({0, 0, 0}),
          /*end_mask=*/get_mask({1, 0, 0}),
          /*ellipsis_mask=*/0,
          /*new_axis_mask=*/0,
          /*shrink_axis_mask=*/0,
          /*expected_output_dims=*/{1, 1, 2},
          /*expected_output=*/{1, 2},
      },
      // Slice on 3D tensor using end_mask. For dynamic shape, all
      // dimensions are dynamic.
      TestParams{/*input_dims=*/{1, 2, 3},
                 /*begin=*/{0, 1, 1},
                 /*end=*/{0, 0, 0},
                 /*strides=*/{1, 1, 1},
                 /*begin_mask=*/get_mask({0, 0, 0}),
                 /*end_mask=*/get_mask({1, 1, 1}),
                 /*ellipsis_mask=*/0,
                 /*new_axis_mask=*/0,
                 /*shrink_axis_mask=*/0,
                 /*expected_output_dims=*/{1, 1, 2},
                 /*expected_output=*/{5, 6},
                 /*conversion_status=*/Status::OK(),
                 /*runtime_status=*/Status::OK(),
                 /*partial_input_dims=*/{-1, -1, -1}},
      // Slice on 3D tensor using end_mask. For dynamic shape, all
      // dimensions are dynamic.
      TestParams{/*input_dims=*/{1, 1, 2, 3},
                 /*begin=*/{0, 0, 0, 0},
                 /*end=*/{0, 0, 0, 2},
                 /*strides=*/{1, 1, 1, 1},
                 /*begin_mask=*/get_mask({0, 0, 0, 0}),
                 /*end_mask=*/get_mask({1, 1, 1, 0}),
                 /*ellipsis_mask=*/0,
                 /*new_axis_mask=*/0,
                 /*shrink_axis_mask=*/0,
                 /*expected_output_dims=*/{1, 1, 2, 2},
                 /*expected_output=*/{1, 2, 4, 5},
                 /*conversion_status=*/Status::OK(),
                 /*runtime_status=*/Status::OK(),
                 /*partial_input_dims=*/{-1, -1, -1, -1}},
      TestParams{
          /*input_dims=*/{1, 1, 2, 3},
          /*begin=*/{0, 0, 1, 0},
          /*end=*/{0, 0, 0, 0},
          /*strides=*/{1, 1, 1, 1},
          /*begin_mask=*/get_mask({0, 0, 0, 0}),
          /*end_mask=*/get_mask({1, 1, 1, 1}),
          /*ellipsis_mask=*/0,
          /*new_axis_mask=*/0,
          /*shrink_axis_mask=*/0,
          /*expected_output_dims=*/{1, 1, 1, 3},
          /*expected_output=*/{4, 5, 6},
      },
      // 1D simple slice.
      TestParams{/*input_dims=*/{1, 2, 3, 1},
                 /*begin=*/{0, 0, 0, 0},
                 /*end=*/{0, 1, 0, 0},
                 /*strides=*/{1, 1, 1, 1},
                 /*begin_mask=*/get_mask({0, 0, 0, 0}),
                 /*end_mask=*/get_mask({1, 0, 1, 1}),
                 /*ellipsis_mask=*/0,
                 /*new_axis_mask=*/0,
                 /*shrink_axis_mask=*/0,
                 /*expected_output_dims=*/{1, 1, 3, 1},
                 /*expected_output=*/{1, 2, 3},
                 /*conversion_status=*/Status::OK(),
                 /*runtime_status=*/Status::OK(),
                 /*partial_input_dims=*/{-1, -1, -1, -1}},
      TestParams{
          /*input_dims=*/{1, 2, 3, 1},
          /*begin=*/{0, 1, 0, 0},
          /*end=*/{0, 0, 0, 0},
          /*strides=*/{1, 1, 1, 1},
          /*begin_mask=*/get_mask({0, 0, 0, 0}),
          /*end_mask=*/get_mask({1, 1, 1, 1}),
          /*ellipsis_mask=*/0,
          /*new_axis_mask=*/0,
          /*shrink_axis_mask=*/0,
          /*expected_output_dims=*/{1, 1, 3, 1},
          /*expected_output=*/{4, 5, 6},
      },
      // Simple 1D slice on 2D input.
      TestParams{/*input_dims=*/{1, 6},
                 /*begin=*/{0, 0},
                 /*end=*/{0, 3},
                 /*strides=*/{1, 1},
                 /*begin_mask=*/get_mask({0, 0}),
                 /*end_mask=*/get_mask({1, 0}),
                 /*ellipsis_mask=*/0,
                 /*new_axis_mask=*/0,
                 /*shrink_axis_mask=*/0,
                 /*expected_output_dims=*/{1, 3},
                 /*expected_output=*/{1, 2, 3},
                 /*conversion_status=*/Status::OK(),
                 /*runtime_status=*/Status::OK(),
                 /*partial_input_dims=*/{-1, -1}},
      TestParams{
          /*input_dims=*/{1, 1, 6},
          /*begin=*/{0, 0, 2},
          /*end=*/{0, 0, 5},
          /*strides=*/{1, 1, 1},
          /*begin_mask=*/get_mask({0, 0, 0}),
          /*end_mask=*/get_mask({1, 1, 0}),
          /*ellipsis_mask=*/0,
          /*new_axis_mask=*/0,
          /*shrink_axis_mask=*/0,
          /*expected_output_dims=*/{1, 1, 3},
          /*expected_output=*/{3, 4, 5},
      },
      TestParams{
          /*input_dims=*/{1, 6, 1},
          /*begin=*/{0, 2, 0},
          /*end=*/{0, 5, 0},
          /*strides=*/{1, 1, 1},
          /*begin_mask=*/get_mask({0, 0, 0}),
          /*end_mask=*/get_mask({1, 0, 1}),
          /*ellipsis_mask=*/0,
          /*new_axis_mask=*/0,
          /*shrink_axis_mask=*/0,
          /*expected_output_dims=*/{1, 3, 1},
          /*expected_output=*/{3, 4, 5},
      },
      // Negative axis.
      TestParams{
          /*input_dims=*/{1, 6, 1},
          /*begin=*/{0, -6, 0},
          /*end=*/{0, -3, 0},
          /*strides=*/{1, 1, 1},
          /*begin_mask=*/get_mask({0, 0, 0}),
          /*end_mask=*/get_mask({1, 0, 1}),
          /*ellipsis_mask=*/0,
          /*new_axis_mask=*/0,
          /*shrink_axis_mask=*/0,
          /*expected_output_dims=*/{1, 3, 1},
          /*expected_output=*/{1, 2, 3},
      },
      TestParams{
          /*input_dims=*/{1, 6, 1},
          /*begin=*/{0, 0, 0},
          /*end=*/{0, -1, 0},
          /*strides=*/{1, 1, 1},
          /*begin_mask=*/get_mask({0, 0, 0}),
          /*end_mask=*/get_mask({1, 0, 1}),
          /*ellipsis_mask=*/0,
          /*new_axis_mask=*/0,
          /*shrink_axis_mask=*/0,
          /*expected_output_dims=*/{1, 5, 1},
          /*expected_output=*/{1, 2, 3, 4, 5},
      },
      // Clamp out of bounds begin and end.
      TestParams{
          /*input_dims=*/{1, 1, 2, 3},
          /*begin=*/{0, 0, -9999, -9},
          /*end=*/{0, 1, 1000, 4},
          /*strides=*/{1, 1, 1, 1},
          /*begin_mask=*/get_mask({0, 0, 0, 0}),
          /*end_mask=*/get_mask({1, 0, 0, 0}),
          /*ellipsis_mask=*/0,
          /*new_axis_mask=*/0,
          /*shrink_axis_mask=*/0,
          /*expected_output_dims=*/{1, 1, 2, 3},
          /*expected_output=*/{1, 2, 3, 4, 5, 6},
      },
      // Stride values >= 2.
      TestParams{/*input_dims=*/{1, 6},
                 /*begin=*/{0, 0},
                 /*end=*/{0, 5},
                 /*strides=*/{1, 2},
                 /*begin_mask=*/get_mask({0, 0}),
                 /*end_mask=*/get_mask({1, 0}),
                 /*ellipsis_mask=*/0,
                 /*new_axis_mask=*/0,
                 /*shrink_axis_mask=*/0,
                 /*expected_output_dims=*/{1, 3},
                 /*expected_output=*/{1, 3, 5},
                 /*conversion_status=*/Status::OK(),
                 /*runtime_status=*/Status::OK(),
                 /*partial_input_dims=*/{-1, -1}},
      TestParams{/*input_dims=*/{1, 6},
                 /*begin=*/{0, 0},
                 /*end=*/{0, 6},
                 /*strides=*/{1, 2},
                 /*begin_mask=*/get_mask({0, 0}),
                 /*end_mask=*/get_mask({1, 0}),
                 /*ellipsis_mask=*/0,
                 /*new_axis_mask=*/0,
                 /*shrink_axis_mask=*/0,
                 /*expected_output_dims=*/{1, 3},
                 /*expected_output=*/{1, 3, 5},
                 /*conversion_status=*/Status::OK(),
                 /*runtime_status=*/Status::OK(),
                 /*partial_input_dims=*/{-1, -1}},
      TestParams{/*input_dims=*/{1, 6},
                 /*begin=*/{0, 1},
                 /*end=*/{0, 6},
                 /*strides=*/{1, 2},
                 /*begin_mask=*/get_mask({0, 0}),
                 /*end_mask=*/get_mask({1, 0}),
                 /*ellipsis_mask=*/0,
                 /*new_axis_mask=*/0,
                 /*shrink_axis_mask=*/0,
                 /*expected_output_dims=*/{1, 3},
                 /*expected_output=*/{2, 4, 6},
                 /*conversion_status=*/Status::OK(),
                 /*runtime_status=*/Status::OK(),
                 /*partial_input_dims=*/{-1, -1}},
      TestParams{/*input_dims=*/{1, 6},
                 /*begin=*/{0, 2},
                 /*end=*/{0, 6},
                 /*strides=*/{1, 3},
                 /*begin_mask=*/get_mask({0, 0}),
                 /*end_mask=*/get_mask({1, 0}),
                 /*ellipsis_mask=*/0,
                 /*new_axis_mask=*/0,
                 /*shrink_axis_mask=*/0,
                 /*expected_output_dims=*/{1, 2},
                 /*expected_output=*/{3, 6},
                 /*conversion_status=*/Status::OK(),
                 /*runtime_status=*/Status::OK(),
                 /*partial_input_dims=*/{-1, -1}},
      // Stride values <= -2.
      TestParams{/*input_dims=*/{1, 6},
                 /*begin=*/{0, 5},
                 /*end=*/{0, 0},
                 /*strides=*/{1, -2},
                 /*begin_mask=*/get_mask({0, 0}),
                 /*end_mask=*/get_mask({1, 1}),
                 /*ellipsis_mask=*/0,
                 /*new_axis_mask=*/0,
                 /*shrink_axis_mask=*/0,
                 /*expected_output_dims=*/{1, 3},
                 /*expected_output=*/{6, 4, 2},
                 /*conversion_status=*/Status::OK(),
                 /*runtime_status=*/Status::OK(),
                 /*partial_input_dims=*/{-1, -1}},
      TestParams{/*input_dims=*/{1, 6},
                 /*begin=*/{0, 5},
                 /*end=*/{0, 0},
                 /*strides=*/{1, -2},
                 /*begin_mask=*/get_mask({0, 0}),
                 /*end_mask=*/get_mask({1, 0}),
                 /*ellipsis_mask=*/0,
                 /*new_axis_mask=*/0,
                 /*shrink_axis_mask=*/0,
                 /*expected_output_dims=*/{1, 3},
                 /*expected_output=*/{6, 4, 2},
                 /*conversion_status=*/Status::OK(),
                 /*runtime_status=*/Status::OK(),
                 /*partial_input_dims=*/{-1, -1}},
      TestParams{/*input_dims=*/{1, 6},
                 /*begin=*/{0, 5},
                 /*end=*/{0, 1},
                 /*strides=*/{1, -3},
                 /*begin_mask=*/get_mask({0, 0}),
                 /*end_mask=*/get_mask({1, 0}),
                 /*ellipsis_mask=*/0,
                 /*new_axis_mask=*/0,
                 /*shrink_axis_mask=*/0,
                 /*expected_output_dims=*/{1, 2},
                 /*expected_output=*/{6, 3},
                 /*conversion_status=*/Status::OK(),
                 /*runtime_status=*/Status::OK(),
                 /*partial_input_dims=*/{-1, -1}},
      // Ellipsis_mask causes leading dimensions to be ignored. Begin, end,
      // stride, and mask values of size 2 should be interpreted as applying to
      // the last 2 dimensions, while the ellipsis applies to the first 2 (for a
      // 4D input tensor).
      TestParams{/*input_dims=*/{1, 1, 2, 3},
                 /*begin=*/{0, 1},
                 /*end=*/{0, 2},
                 /*strides=*/{1, 1},
                 /*begin_mask=*/get_mask({0, 0}),
                 /*end_mask=*/get_mask({0, 0}),
                 /*ellipsis_mask=*/get_mask({1, 0, 0}),
                 /*new_axis_mask=*/0,
                 /*shrink_axis_mask=*/0,
                 /*expected_output_dims=*/{1, 1, 2, 1},
                 /*expected_output=*/{2, 5},
                 /*conversion_status=*/Status::OK(),
                 /*runtime_status=*/Status::OK(),
                 /*partial_input_dims=*/{-1, -1, -1, -1}},
      // Ellipsis_mask on single inner dimension.
      TestParams{
          /*input_dims=*/{1, 1, 2, 3},
          /*begin=*/{0, 0, 1},
          /*end=*/{0, 0, 2},
          /*strides=*/{1, 1, 1},
          /*begin_mask=*/get_mask({1, 0, 0, 0}),
          /*end_mask=*/get_mask({1, 0, 0, 0}),
          /*ellipsis_mask=*/get_mask({0, 1, 0, 0}),
          /*new_axis_mask=*/0,
          /*shrink_axis_mask=*/0,
          /*expected_output_dims=*/{1, 1, 2, 1},
          /*expected_output=*/{2, 5},
      },
      // Ellipsis_mask on single leading dimension.
      TestParams{/*input_dims=*/{1, 1, 2, 3},
                 /*begin=*/{0, 0, 0, 1},
                 /*end=*/{0, 1, 2, 2},
                 /*strides=*/{1, 1, 1, 1},
                 /*begin_mask=*/get_mask({0, 0, 0, 0}),
                 /*end_mask=*/get_mask({0, 0, 0, 0}),
                 /*ellipsis_mask=*/get_mask({1, 0, 0, 0}),
                 /*new_axis_mask=*/0,
                 /*shrink_axis_mask=*/0,
                 /*expected_output_dims=*/{1, 1, 2, 1},
                 /*expected_output=*/{2, 5},
                 /*conversion_status=*/Status::OK(),
                 /*runtime_status=*/Status::OK(),
                 /*partial_input_dims=*/{-1, -1, -1, -1}},
      // Ellipsis_mask on single inner dimension overrides that dimensions'
      // begin/end values.
      TestParams{/*input_dims=*/{1, 1, 2, 3},
                 /*begin=*/{0, 1, 0, 1},
                 /*end=*/{1, 1, 2, 2},
                 /*strides=*/{1, 1, 1, 1},
                 /*begin_mask=*/get_mask({0, 0, 0, 0}),
                 /*end_mask=*/get_mask({0, 0, 0, 0}),
                 /*ellipsis_mask=*/get_mask({0, 1, 0, 0}),
                 /*new_axis_mask=*/0,
                 /*shrink_axis_mask=*/0,
                 /*expected_output_dims=*/{1, 1, 2, 1},
                 /*expected_output=*/{2, 5},
                 /*conversion_status=*/Status::OK(),
                 /*runtime_status=*/Status::OK(),
                 /*partial_input_dims=*/{-1, -1, -1, -1}},
      // Ellipsis mask on single leading dimension should throw out extra
      // leading values of begin/end vectors so that only the last N-1 values of
      // each remain.
      TestParams{/*input_dims=*/{1, 1, 2, 3},
                 /*begin=*/{0, 0, 0, 0, 1},
                 /*end=*/{0, 1, 1, 2, 2},
                 /*strides=*/{1, 1, 1, 1, 1},
                 /*begin_mask=*/get_mask({0, 0, 0, 0}),
                 /*end_mask=*/get_mask({0, 0, 0, 0}),
                 /*ellipsis_mask=*/get_mask({1, 0, 0, 0}),
                 /*new_axis_mask=*/0,
                 /*shrink_axis_mask=*/0,
                 /*expected_output_dims=*/{1, 1, 2, 1},
                 /*expected_output=*/{2, 5},
                 /*conversion_status=*/Status::OK(),
                 /*runtime_status=*/Status::OK(),
                 /*partial_input_dims=*/{-1, -1, -1, -1}},
      // Shrink-axis mask set for the final dimension of final size 1 should
      // remove that dimension from the final shape.
      TestParams{/*input_dims=*/{1, 1, 2, 3},
                 /*begin=*/{0, 0, 0, 1},
                 /*end=*/{0, 0, 0, 2},
                 /*strides=*/{1, 1, 1, 1},
                 /*begin_mask=*/get_mask({1, 1, 1, 0}),
                 /*end_mask=*/get_mask({1, 1, 1, 0}),
                 /*ellipsis_mask=*/0,
                 /*new_axis_mask=*/0,
                 /*shrink_axis_mask=*/get_mask({0, 0, 0, 1}),
                 /*expected_output_dims=*/{1, 1, 2},
                 /*expected_output=*/{2, 5},
                 /*conversion_status=*/Status::OK(),
                 /*runtime_status=*/Status::OK(),
                 /*partial_input_dims=*/{1, 1, 2, -1}},
      // Shrink-axis mask set for multiple dimensions that have a final size of
      // 1 should remove those dimensions from the final shape.
      TestParams{/*input_dims=*/{1, 1, 2, 3},
                 /*begin=*/{0, 0, 0, 1},
                 /*end=*/{0, 1, 2, 2},
                 /*strides=*/{1, 1, 1, 1},
                 /*begin_mask=*/get_mask({1, 0, 0, 0}),
                 /*end_mask=*/get_mask({1, 0, 0, 0}),
                 /*ellipsis_mask=*/0,
                 /*new_axis_mask=*/0,
                 /*shrink_axis_mask=*/get_mask({0, 1, 0, 1}),
                 /*expected_output_dims=*/{1, 2},
                 /*expected_output=*/{2, 5},
                 /*conversion_status=*/Status::OK(),
                 /*runtime_status=*/Status::OK(),
                 /*partial_input_dims=*/{1, 1, 2, -1}},
      // Shrink-axis mask set for multiple sequential dimensions of final size 1
      // should
      // remove those dimensions from the final shape.
      TestParams{/*input_dims=*/{6, 1, 1},
                 /*begin=*/{0, 0, 0},
                 /*end=*/{0, 0, 0},
                 /*strides=*/{1, 1, 1},
                 /*begin_mask=*/get_mask({1, 1, 1}),
                 /*end_mask=*/get_mask({1, 1, 1}),
                 /*ellipsis_mask=*/0,
                 /*new_axis_mask=*/0,
                 /*shrink_axis_mask=*/get_mask({0, 1, 1}),
                 /*expected_output_dims=*/{6},
                 /*expected_output=*/{1, 2, 3, 4, 5, 6},
                 /*conversion_status=*/Status::OK(),
                 /*runtime_status=*/Status::OK(),
                 /*partial_input_dims=*/{-1, -1, -1}},
      // The new_axis_mask parameter is not supported.
      TestParams{/*input_dims=*/{1, 6},
                 /*begin=*/{0, 0, 0},
                 /*end=*/{0, 0, 0},
                 /*strides=*/{1, 1, 1},
                 /*begin_mask=*/
                 get_mask({0, 1, 1}),
                 /*end_mask=*/get_mask({0, 1, 1}),
                 /*ellipsis_mask=*/0,
                 /*new_axis_mask=*/get_mask({1, 0, 0}),
                 /*shrink_axis_mask=*/get_mask({0, 0, 0}),
                 /*expected_output_dims=*/{1, 1, 6},
                 /*expected_output=*/{1, 1, 6},
                 /*conversion_status=*/
                 errors::Unimplemented(
                     "new_axis_mask is not supported for StridedSlice"),
                 /*runtime_status=*/Status::OK(),
                 /*partial_input_dims=*/{1, 6}},
  };

  int i = 0;
  for (auto p : params) {
    Reset();
    NodeDef node_def = get_strided_slice_nodedef(
        tf_type_, p.begin_mask, p.end_mask, p.ellipsis_mask, p.new_axis_mask,
        p.shrink_axis_mask);

    VLOG(2) << "Preparing test case " << i++ << " with dims "
            << DebugString(p.input_dims);

    switch (trt_mode_) {
      case TrtTestMode::kImplicitBatch: {
        AddTestTensor("input", p.input_dims, ok_input);
        break;
      }
      case TrtTestMode::kExplicitBatch: {
        AddTestTensor("input", p.input_dims, ok_input);
        break;
      }
      case TrtTestMode::kDynamicShape: {
        if (p.partial_input_dims.size() > 0) {
          AddTestTensor("input", p.input_dims, tf_type_, ok_input,
                        p.partial_input_dims);

        } else {
          AddTestTensor("input", p.input_dims, tf_type_, ok_input,
                        p.input_dims);
        }
        break;
      }
    }

    VLOG(2) << "Adding weights begin: " << DebugString(p.begin)
            << ", end: " << DebugString(p.end)
            << ", strides: " << DebugString(p.strides);
    AddTestWeights<int32>("begin", {static_cast<int>(p.begin.size())}, p.begin);
    AddTestWeights<int32>("end", {static_cast<int>(p.end.size())}, p.end);
    AddTestWeights<int32>("strides", {static_cast<int>(p.strides.size())},
                          p.strides);

    TestOpConverter("my_strided_slice", node_def, p.expected_output_dims,
                    p.conversion_status, p.runtime_status,
                    ElementsAreArray(p.expected_output));
  }
}

TEST_P(OpConverter_FP32_FP16_INT32_Test, ConvertSlice) {
  // Get nodedef for Slice layer.
  auto get_slice_nodedef = [](DataType tf_type) -> NodeDef {
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), tf_type);
    auto begin = ops::Placeholder(s.WithOpName("begin"), DT_INT32);
    auto size = ops::Placeholder(s.WithOpName("size"), DT_INT32);
    auto slice = ops::Slice(s.WithOpName("my_slice"), input, begin, size);
    return slice.operation.node()->def();
  };

  struct TestParams {
    std::vector<int> input_dims;
    std::vector<int>
        partial_input_dims;  // Symbolic shape in dynamic shape mode.
    std::vector<int> begin;
    std::vector<int> size;
    std::vector<int> expected_output_dims;
    std::vector<int> expected_output;
    Status conversion_status;
    Status runtime_status;
  };

  std::vector<TestParams> params = {
      // Slice start points must always be >= 0.
      TestParams{/*input_dims=*/{1, 1, 2, 3},
                 /*partial_input_dims=*/{-1, -1, -1, -1},
                 /*begin=*/{0, 0, -1, 0},
                 /*size=*/{1, 1, 2, 3},
                 /*expected_output_dims=*/{},
                 /*expected_output=*/{},
                 /*conversion_status=*/
                 errors::InvalidArgument("\"begin\" in Slice "
                                         "is out of range")},
      // In implicit batch mode, slicing the batch dimension is not allowed.
      TestParams{/*input_dims=*/{2, 1, 1, 3},
                 /*partial_input_dims=*/{-1, -1, -1, -1},
                 /*begin=*/{0, 0, 0, 0},
                 /*size=*/{1, 1, 1, 3},
                 /*expected_output_dims=*/{1, 1, 1, 3},
                 /*expected_output=*/{1, 2, 3},
                 /*conversion_status=*/trt_mode_ == TrtTestMode::kImplicitBatch
                     ? errors::Unimplemented(
                           "TensorRT does not allow modifications to the batch "
                           "dimension in implicit batch mode")
                     : Status::OK()},
      // Dynamic batch size but using size[0] of -1, ok.
      TestParams{{1, 1, 2, 3},
                 /*partial_input_dims=*/{-1, -1, -1, -1},
                 {0, 0, 0, 0},
                 {-1, 1, 2, 2},
                 {1, 1, 2, 2},
                 {1, 2, 4, 5},
                 Status::OK()},
      TestParams{{1, 1, 2, 3},
                 /*partial_input_dims=*/{-1, -1, -1, -1},
                 {0, 0, 0, 0},
                 {-1, -1, -1, -1},
                 {1, 1, 2, 3},
                 {1, 2, 3, 4, 5, 6},
                 Status::OK()},
      TestParams{{1, 1, 2, 3},
                 /*partial_input_dims=*/{-1, -1, -1, -1},
                 {0, 0, 0, 0},
                 {1, 1, 2, 3},
                 {1, 1, 2, 3},
                 {1, 2, 3, 4, 5, 6}},
      TestParams{{1, 1, 2, 3},
                 /*partial_input_dims=*/{-1, -1, -1, -1},
                 /*begin=*/{0, 0, 0, 0},
                 /*size=*/{1, -1, 2, 2},
                 /*expected_output_dims=*/{1, 1, 2, 2},
                 /*expected_output=*/{1, 2, 4, 5},
                 Status::OK()},
      TestParams{/*input_dims=*/{1, 6},
                 /*partial_input_dims=*/{-1, -1},
                 /*being=*/{0, 1},
                 /*size=*/{1, 5},
                 /*expected_output_dims=*/{1, 5},
                 /*expected_output=*/{2, 3, 4, 5, 6}},
      TestParams{/*input_dims=*/{1, 6},
                 /*partial_input_dims=*/{-1, -1},
                 /*begin=*/{0, 1},
                 /*size=*/{-1, 3},
                 /*expected_output_dims=*/{1, 3},
                 /*expected_output=*/{2, 3, 4}, Status::OK()},
      // In dynamic shape mode we do not know the input shape during
      // conversion, therfore we cannot check out of bound access.
      TestParams{
          {1, 1, 2, 3},
          /*partial_input_dims=*/{-1, -1, -1, -1},
          /*begin=*/{0, 0, 3, 0},
          /*end=*/{1, 1, 2, 3},
          {},
          {},
          trt_mode_ == TrtTestMode::kDynamicShape
              ? Status::OK()
              : errors::InvalidArgument("\"begin\" + \"size\" for dimension "
                                        "2 in Slice is out of range"),
          errors::Internal("Internal: Failed to build TensorRT engine")},
      // The slice operation should expect that the "size[i]" values are not
      // less than -1.
      TestParams{/*input_dims=*/{1, 1, 2, 3},
                 /*partial_input_dims=*/{-1, -1, -1, -1},
                 /*begin=*/{0, 0, 0, 0},
                 /*size=*/{1, 1, 2, -2},
                 {},
                 {},
                 errors::InvalidArgument("\"size\" in Slice is out of range")},
      TestParams{
          /*input_dims=*/{1, 1, 2, 3},
          /*partial_input_dims=*/{-1, -1, -1, -1},
          /*begin=*/{0, 0, 0, 0},
          /*size=*/{1, 1, 3, 2},
          /*expected_output_dims=*/{},
          /*expected_output=*/{},
          /*conversion_status=*/trt_mode_ == TrtTestMode::kDynamicShape
              ? Status::OK()
              : errors::InvalidArgument("\"begin\" + \"size\" for dimension "
                                        "2 in Slice is out of range"),
          errors::Internal("Internal: Failed to build TensorRT engine")},
  };

  int i = 0;
  for (auto p : params) {
    Reset();
    NodeDef node_def = get_slice_nodedef(tf_type_);

    VLOG(2) << "Preparing test case " << i++ << " with dims "
            << DebugString(p.input_dims);

    // The input tensor always has size 6.
    std::vector<int> input_vals = {1, 2, 3, 4, 5, 6};

    switch (trt_mode_) {
      case TrtTestMode::kImplicitBatch: {
        AddTestTensor("input", p.input_dims, input_vals);
        break;
      }
      case TrtTestMode::kExplicitBatch: {
        AddTestTensor("input", p.input_dims, input_vals);
        break;
      }
      case TrtTestMode::kDynamicShape: {
        if (p.partial_input_dims.size() > 0) {
          AddTestTensor("input", p.input_dims, tf_type_, input_vals,
                        p.partial_input_dims);

        } else {
          AddTestTensor("input", p.input_dims, tf_type_, input_vals,
                        p.input_dims);
        }
        break;
      }
    }

    AddTestWeights<int32>("begin", {static_cast<int>(p.begin.size())}, p.begin);
    AddTestWeights<int32>("size", {static_cast<int>(p.size.size())}, p.size);

    TestOpConverter("my_slice", node_def, p.expected_output_dims,
                    p.conversion_status, p.runtime_status,
                    ElementsAreArray(p.expected_output));
  }
}

TEST_P(OpConverter_FP32_Test, ConvertConv2D) {
  // Get nodedef for Conv2D layer.
  DataType tf_type = tf_type_;
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
        "The input \"input\" for Conv2D must be a tensor");
  }
  {
    // Filter is tensor, should fail.
    Reset();
    NodeDef node_def = get_conv2d_nodedef();
    AddTestTensor("input", {3, 1, 2, 1});
    AddTestTensor("weights", {3, 3, 1, 1});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"filter\" for Conv2D must be a constant");
  }
  {
    // Filter is not 4D, should fail.
    Reset();
    NodeDef node_def = get_conv2d_nodedef();
    AddTestTensor("input", {1, 1, 2, 3});
    AddTestWeights<float>("weights", {3, 3, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "Conv2D expects kernel of dimension 4");
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
        "Convolution dilations field must specify 4 dimensions");
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
                               "dimensions");
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
                               "dimensions");
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
        "Convolution strides field must specify 4 dimensions");
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
        "Stride must be 1 for batch and channel dimensions");
  }
  if (trt_mode_ == TrtTestMode::kDynamicShape) {
    Reset();
    NodeDef node_def = get_conv2d_nodedef();
    // Channel dim unknown, should fail.
    nvinfer1::DataType trt_type;
    TF_ASSERT_OK(TfTypeToTrtType(tf_type_, &trt_type));
    AddTestTensorWithTFDims("input", {-1, -1, -1, -1}, trt_type);
    AddTestWeights<float>("weights", {1, 2, 1, 1}, {-1, 1});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "Channel dimension must be static");
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
    if (trt_mode_ == TrtTestMode::kDynamicShape) {
      // The channel dim cannot have unknown size, fix that.
      partial_input_shape.resize(ok_params[i].input_dims.size(), -1);
      int channel_id = (ok_params[i].data_format == "NCHW") ? 1 : 3;
      partial_input_shape[channel_id] = ok_params[i].input_dims[channel_id];
    }

    AddTestTensor("input", ok_params[i].input_dims, tf_type_,
                  ok_params[i].input, partial_input_shape);
    AddTestWeights<float>("weights", ok_params[i].filter_dims,
                          ok_params[i].filter);

    TestOpConverter("my_conv2d", node_def, ok_params[i].expected_output_dims,
                    Status::OK(), Status::OK(),
                    ElementsAreArray(ok_params[i].expected_output));
  }
}

TEST_P(OpConverter_FP32_Test, ConvertConv2DBackpropInput) {
  // Get nodedef for Conv2D layer.
  auto get_conv2d_backprop_input_nodedef =
      [](DataType tf_type, std::vector<int> strides = {1, 1, 1, 1},
         string padding = "SAME", string data_format = "NCHW",
         std::vector<int> dilations = {1, 1, 1, 1}) -> NodeDef {
    Scope s = Scope::NewRootScope();
    auto input = ops::Placeholder(s.WithOpName("input"), tf_type);
    auto filter = ops::Placeholder(s.WithOpName("weights"), tf_type);
    auto input_sizes = ops::Placeholder(s.WithOpName("input_sizes"), DT_INT32);
    ops::Conv2DBackpropInput::Attrs attrs = ops::Conv2DBackpropInput::Attrs()
                                                .DataFormat(data_format)
                                                .Dilations(dilations);
    auto conv2d = ops::Conv2DBackpropInput(
        s.WithOpName("my_conv2d_backprop_input"), input_sizes, filter, input,
        strides, padding, attrs);
    return conv2d.operation.node()->def();
  };

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
    Status conversion_status;
    // For dynamic shape mode, we must use the partial_input_dims for
    // creating the test tensor if any of the input_dims are -1.
    std::vector<int> partial_input_dims;
  };

  // Ok.
  std::vector<TestParams> params = {
      // Transpose Strided
      TestParams{/*input_dims=*/{1, 1, 2, 2},
                 /*input=*/{0, 1, 2, 3},
                 /*filter_dims=*/{1, 2, 1, 1},
                 /*filter=*/{-1, 1},
                 /*strides=*/{1, 1, 1, 2},
                 /*padding=*/"SAME",
                 /*data_format=*/"NCHW",
                 /*dilations=*/{1, 1, 1, 1},
                 /*expected_output_dims=*/{1, 1, 2, 4},
                 /*expected_output=*/{0, 0, -1, 1, -2, 2, -3, 3}},
      // Transpose Strided NHWC
      TestParams{/*input_dims=*/{1, 2, 2, 1},
                 /*input=*/{0, 1, 2, 3},
                 /*filter_dims=*/{1, 2, 1, 1},
                 /*filter=*/{-1, 1},
                 /*strides=*/{1, 1, 2, 1},
                 /*padding=*/"SAME",
                 /*data_format=*/"NHWC",
                 /*dilations=*/{1, 1, 1, 1},
                 /*expected_output_dims=*/{1, 2, 4, 1},
                 /*expected_output=*/{0, 0, -1, 1, -2, 2, -3, 3}},
      // Transpose Strided NHWC with VALID padding
      TestParams{/*input_dims=*/{1, 3, 1, 1},
                 /*input=*/{0, 1, 2},
                 /*filter_dims=*/{2, 1, 1, 1},
                 /*filter=*/{-1, 1},
                 /*strides=*/{1, 2, 1, 1},
                 /*padding=*/"VALID",
                 /*data_format=*/"NHWC",
                 /*dilations=*/{1, 1, 1, 1},
                 /*expected_output_dims=*/{1, 7, 1, 1},
                 /*expected_output=*/{0, 0, -1, 1, -2, 2, 0}},
      TestParams{/*input_dims=*/{1, 1, 2, 2},
                 /*input=*/{0, 1, 2, 3},
                 /*filter_dims=*/{1, 2, 1, 1},
                 /*filter=*/{-1, 1},
                 /*strides=*/{1, 1, 1, 2},
                 /*padding=*/"EXPLICIT",
                 /*data_format=*/"NCHW",
                 /*dilations=*/{1, 1, 1, 1},
                 /*expected_output_dims=*/{1, 1, 2, 4},
                 /*expected_output=*/{0, 0, -1, 1, -2, 2, -3, 3},
                 errors::Unimplemented("EXPLICIT padding type not "
                                       "implemented, only VALID and SAME are"
                                       " supported")},
      // Dilation + Conv2DBackpropInput, should fail.
      TestParams{/*input_dims=*/{1, 1, 2, 2},
                 /*input=*/{0, 1, 2, 3},
                 /*filter_dims=*/{1, 2, 1, 1},
                 /*filter=*/{-1, 1},
                 /*strides=*/{1, 1, 1, 1},
                 /*padding=*/"SAME",
                 /*data_format=*/"NCHW",
                 /*dilations=*/{1, 1, 1, 2},
                 {1, 1, 2, 2},
                 {},
                 errors::Unimplemented("Dilation with Conv2DBackpropInput "
                                       "(conv2d_transpose) is not supported")},
  };
  if (trt_mode_ == TrtTestMode::kDynamicShape) {
    params.push_back(
        TestParams{/*input_dims=*/{1, 1, 2, 2},
                   /*input=*/{0, 1, 2, 3},
                   /*filter_dims=*/{1, 2, 1, 1},
                   /*filter=*/{-1, 1},
                   /*strides=*/{1, 1, 1, 2},
                   /*padding=*/"SAME",
                   /*data_format=*/"NCHW",
                   /*dilations=*/{1, 1, 1, 1},
                   /*expected_output_dims=*/{1, 1, 2, 4},
                   /*expected_output=*/{0, 0, -1, 1, -2, 2, -3, 3},
                   errors::InvalidArgument("Channel dimension must be static"),
                   /*partial input dims=*/{1, -1, 2, 2}});
    // Test dynamic  batch dimension.
    params.push_back(
        TestParams{/*input_dims=*/{2, 1, 2, 2},
                   /*input=*/
                   // clang-format off
                      {0, 1, 2, 3,
                       3, 2, 1, 0},
                   // clang-format on
                   /*filter_dims=*/{1, 2, 1, 1},
                   /*filter=*/{-1, 1},
                   /*strides=*/{1, 1, 1, 2},
                   /*padding=*/"SAME",
                   /*data_format=*/"NCHW",
                   /*dilations=*/{1, 1, 1, 1},
                   /*expected_output_dims=*/{2, 1, 2, 4},
                   /*expected_output=*/
                   // clang-format off
                   { 0, 0, -1, 1, -2, 2, -3, 3,
                    -3, 3, -2, 2, -1, 1, 0, 0},
                   // clang-format on
                   /*conversion_status=*/Status::OK(),
                   /*partial input dims=*/{-1, 1, 2, 2}});

    // Test dynamic height and width.
    params.push_back(TestParams{
        /*input_dims=*/{1, 1, 2, 2},
        /*input=*/{0, 1, 2, 3},
        /*filter_dims=*/{1, 2, 1, 1},
        /*filter=*/{-1, 1},
        /*strides=*/{1, 1, 1, 2},
        /*padding=*/"SAME",
        /*data_format=*/"NCHW",
        /*dilations=*/{1, 1, 1, 1},
        /*expected_output_dims=*/{1, 1, 2, 4},
        /*expected_output=*/
        {0, 0, -1, 1, -2, 2, -3, 3},
        /*conversion_status=*/
        errors::Unimplemented(
            "Conv2dBackpropInput does not support input with unknown spatial "
            "shape"),
        /*partial input dims=*/{1, 1, -1, -1}});
  }
  for (auto p : params) {
    for (int input_sizes_length : {2, 4}) {
      Reset();
      NodeDef node_def = get_conv2d_backprop_input_nodedef(
          tf_type_, p.strides, p.padding, p.data_format, p.dilations);

      switch (trt_mode_) {
        case TrtTestMode::kImplicitBatch: {
          AddTestTensor("input", p.input_dims, p.input);
          break;
        }
        case TrtTestMode::kExplicitBatch: {
          AddTestTensor("input", p.input_dims, p.input);
          break;
        }
        case TrtTestMode::kDynamicShape: {
          AddTestTensor("input", p.input_dims, tf_type_, p.input,
                        p.partial_input_dims.size() > 0 ? p.partial_input_dims
                                                        : p.input_dims);
          break;
        }
        default: {
          ASSERT_TRUE(false) << "unknown test mode";
        }
      }

      AddTestWeights<float>("weights", p.filter_dims, p.filter, tf_type_);

      if (input_sizes_length == 4) {
        AddTestWeights<int>("input_sizes", {4}, p.expected_output_dims);
      } else {
        std::vector<int> tf_input_sizes(2);
        // Remove the channel and batch dimensions.
        if (p.data_format == "NHWC") {
          std::copy(p.expected_output_dims.begin() + 1,
                    p.expected_output_dims.end() - 1, tf_input_sizes.begin());
        } else {
          std::copy(p.expected_output_dims.begin() + 2,
                    p.expected_output_dims.end(), tf_input_sizes.begin());
        }
        QCHECK_EQ(2, tf_input_sizes.size());
        AddTestWeights<int>("input_sizes", {2}, tf_input_sizes);
      }

      TestOpConverter("my_conv2d_backprop_input", node_def,
                      p.expected_output_dims, p.conversion_status, Status::OK(),
                      ElementsAreArray(p.expected_output));
    }
  }
}

// Get the NodeDef for Pack.
NodeDef GetConv3DNodeDef(std::vector<int> strides = {1, 1, 1, 1, 1},
                         string padding = "SAME", string data_format = "NCDHW",
                         std::vector<int> dilations = {1, 1, 1, 1, 1},
                         bool is_conv3d_backprop_input = false) {
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT);
  auto filter = ops::Placeholder(s.WithOpName("weights"), DT_FLOAT);

  if (is_conv3d_backprop_input) {
    auto input_sizes = ops::Placeholder(s.WithOpName("input_sizes"), DT_INT32);
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
    auto conv3d = ops::Conv3D(s.WithOpName("my_conv3d"), input, filter, strides,
                              padding, attrs);
    return conv3d.operation.node()->def();
  }
}

struct Conv3DTestParams {
  std::vector<int> input_dims;
  std::vector<float> input;
  std::vector<int> filter_dims;
  std::vector<float> filter;
  std::vector<int> strides;
  string padding;
  string data_format;
  std::vector<int> dilations;
  bool is_conv3d_backprop;
  std::vector<int> expected_output_dims;
  std::vector<float> expected_output;
  bool allow_dynamic_channel_dim;
  Status validation_status;
};

void TestConv3D(ParameterizedOpConverterTestBase* test, Conv3DTestParams& p) {
  test->Reset();
  NodeDef node_def = GetConv3DNodeDef(p.strides, p.padding, p.data_format,
                                      p.dilations, p.is_conv3d_backprop);

  std::vector<int> partial_input_shape;
  if (!p.allow_dynamic_channel_dim &&
      test->get_trt_mode() == TrtTestMode::kDynamicShape) {
    // The channel dim cannot have unknown size, fix that.
    partial_input_shape.resize(p.input_dims.size(), -1);
    int channel_id = (p.data_format == "NCDHW") ? 1 : 4;
    partial_input_shape[channel_id] = p.input_dims[channel_id];
  }

  test->AddTestTensor("input", p.input_dims, test->get_tf_type(), p.input,
                      partial_input_shape);
  test->AddTestWeights<float>("weights", p.filter_dims, p.filter);

  if (p.is_conv3d_backprop) {
    test->AddTestWeights<float>("input_sizes",
                                {static_cast<int>(p.expected_output.size())},
                                p.expected_output);
  }

  test->TestOpConverter("my_conv3d", node_def, p.expected_output_dims,
                        /*expected_conversion_status=*/p.validation_status,
                        /*expected_runtime_status=*/Status::OK(),
                        /*matcher=*/ElementsAreArray(p.expected_output),
                        /*out_tf_types=*/{test->get_tf_type()});
}

TEST_P(OpConverter_FP32_FP16_Test, ConvertConv3D) {
  {
    // Input is weights, should fail.
    Reset();
    NodeDef node_def = GetConv3DNodeDef();

    AddTestWeights<float>("input", {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6});
    AddTestWeights<float>("weights", {1, 3, 3, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"input\" for Conv3D must be a tensor");
  }
  {
    // Filter is tensor, should fail.
    Reset();
    NodeDef node_def = GetConv3DNodeDef();
    AddTestTensor("input", {1, 1, 2, 3}, tf_type_, CreateVectorIota<float>(6));
    AddTestTensor("weights", {1, 3, 3, 1}, tf_type_,
                  CreateVectorIota<float>(9));
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"filter\" for Conv3D must be a constant");
  }
  {
    // Filter is not 5D, should fail.
    Reset();
    NodeDef node_def = GetConv3DNodeDef();
    AddTestTensor("input", {1, 1, 2, 3}, tf_type_, CreateVectorIota<float>(6));
    AddTestWeights<float>("weights", {3, 3, 1, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "Conv3D expects kernel of dimension 5");
  }
  {
    // Dilations is not 5D, should fail.
    Reset();
    NodeDef node_def =
        GetConv3DNodeDef({1, 1, 1, 1, 1}, "SAME", "NCDHW", {1, 1, 1, 1});
    AddTestTensor("input", {1, 1, 2, 3}, tf_type_, CreateVectorIota<float>(6));
    AddTestWeights<float>(
        "weights", {3, 3, 1, 1, 1},
        {1, 2, 3, 4, 5, 6, 7, 8, 9});  // Dimensions, then values
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Convolution dilations field must specify 5 dimensions");
  }
  {
    // Dilation value is not 1 for channel, should fail.
    Reset();
    NodeDef node_def =
        GetConv3DNodeDef({1, 1, 1, 1, 1}, "SAME", "NCDHW", {1, 2, 1, 1, 1});
    AddTestTensor("input", {1, 1, 2, 3}, tf_type_, CreateVectorIota<float>(6));
    AddTestWeights<float>("weights", {3, 3, 1, 1, 1},
                          {1, 2, 3, 4, 5, 6, 7, 8, 9});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "Dilation rate must be 1 for batch and channel "
                               "dimensions");
  }
  {
    // Dilation value is not 1 for channel (NDHWC), should fail.
    Reset();
    NodeDef node_def =
        GetConv3DNodeDef({1, 1, 1, 1, 1}, "SAME", "NDHWC", {1, 1, 1, 1, 2});
    AddTestTensor("input", {1, 2, 3, 1}, tf_type_, CreateVectorIota<float>(6));
    AddTestWeights<float>("weights", {3, 3, 1, 1, 1},
                          {1, 2, 3, 4, 5, 6, 7, 8, 9});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "Dilation rate must be 1 for batch and channel "
                               "dimensions");
  }
  {
    // Dilation + Conv3DBackpropInputV2, should fail.
    Reset();
    NodeDef node_def = GetConv3DNodeDef({1, 1, 1, 1, 1}, "SAME", "NDHWC",
                                        {1, 1, 2, 1, 1}, true);
    AddTestTensor("input", {1, 2, 3, 1}, tf_type_, CreateVectorIota<float>(6));
    AddTestWeights<float>("weights", {3, 3, 1, 1, 1},
                          {1, 2, 3, 4, 5, 6, 7, 8, 9});
    AddTestWeights<int>("input_sizes", {4}, {1, 2, 3, 1});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "Dilation with Conv3DBackpropInputV2 "
                               "(conv3d_transpose) is not supported");
  }
  {
    // Asymmetric+ Conv3DBackpropInputV2, should fail.
    Reset();
    NodeDef node_def = GetConv3DNodeDef({1, 1, 1, 1, 1}, "SAME", "NDHWC",
                                        {1, 1, 1, 1, 1}, true);
    AddTestTensor("input", {1, 2, 2, 2}, tf_type_, CreateVectorIota<float>(8));
    AddTestWeights<float>("weights", {1, 1, 2, 1, 1}, {1, 1});
    AddTestWeights<int>("input_sizes", {8}, {1, 2, 3, 4, 5, 6, 7, 8});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "Asymmetric padding with Conv3DBackpropInputV2 "
                               "(conv3d_transpose) is not supported");
  }
  {
    // Strides is not 5D, should fail.
    Reset();
    NodeDef node_def =
        GetConv3DNodeDef({1, 1, 1, 1, 1, 1}, "SAME", "NCDHW", {1, 1, 1, 1, 1});
    AddTestTensor("input", {1, 2, 2, 2}, tf_type_, CreateVectorIota<float>(8));
    AddTestWeights<float>("weights", {1, 1, 2, 1, 1}, {1, 1});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Convolution strides field must specify 5 dimensions");
  }
  {
    // Stride value is not 1 for channel, should fail.
    Reset();
    NodeDef node_def =
        GetConv3DNodeDef({1, 2, 1, 1, 1}, "SAME", "NCDHW", {1, 1, 1, 1, 1});
    AddTestTensor("input", {1, 1, 2, 3}, tf_type_, CreateVectorIota<float>(6));
    AddTestWeights<float>("weights", {3, 3, 1, 1, 1},
                          {1, 2, 3, 4, 5, 6, 7, 8, 9});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "Stride must be 1 for batch and channel dimensions");
  }

  // Start here
  std::vector<Conv3DTestParams> ok_params = {
      // Basic - just 1x1 conv - input = output
      {/*input_dims=*/{1, 1, 3, 3, 3},  // CDHW
       /*input=*/{1, 2,  15,  3, 6,  -3, 22, 1, 88, 56, 36, 1,  1, 105,
                  1, 16, -28, 1, 42, 9,  3,  1, 7,  1,  11, 61, 5},
       /*filter_dims=*/{1, 1, 1, 1, 1},  // DRSCK
       /*filter=*/{1},
       /*strides=*/{1, 1, 1, 1, 1},
       /*padding=*/"VALID",
       /*data_format=*/"NCDHW",
       /*dilations=*/{1, 1, 1, 1, 1},
       /*is_conv3d_backprop=*/false,
       /*expected_output_dims=*/{1, 1, 3, 3, 3},
       /*expected_output=*/{1,  2,  15, 3, 6,   -3, 22, 1,   88,
                            56, 36, 1,  1, 105, 1,  16, -28, 1,
                            42, 9,  3,  1, 7,   1,  11, 61,  5},
       /*allow_dynamic_channel_dim=*/false,
       /*validation_status=*/Status::OK()},
      // Basic - 2x1 filter
      {/*input_dims=*/{1, 1, 3, 3, 3},  // CDHW
       /*input=*/{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6},
       /*filter_dims=*/{2, 1, 1, 1, 1},  // DRSCK
       /*filter=*/{1, 1},
       /*strides=*/{1, 1, 1, 1, 1},
       /*padding=*/"VALID",
       /*data_format=*/"NCDHW",
       /*dilations=*/{1, 1, 1, 1, 1},
       /*is_conv3d_backprop=*/false,
       /*expected_output_dims=*/{1, 1, 2, 3, 3},
       /*expected_output=*/
       {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 7},
       /*allow_dynamic_channel_dim=*/false,
       /*validation_status=*/Status::OK()},
      // SAME padding (Asymmetric)
      {/*input_dims=*/{1, 1, 2, 3, 2},  // CDHW
       /*input=*/{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
       /*filter_dims=*/{2, 1, 1, 1, 1},  // DRSCK
       /*filter=*/{-1, 1},
       /*strides=*/{1, 1, 1, 1, 1},
       /*padding=*/"SAME",
       /*data_format=*/"NCDHW",
       /*dilations=*/{1, 1, 1, 1, 1},
       /*is_conv3d_backprop=*/false,
       /*expected_output_dims=*/{1, 1, 2, 3, 2},
       // Diff in first 2 depths is const 6.
       /*expected_output=*/{6, 6, 6, 6, 6, 6, -6, -7, -8, -9, -10, -11},
       /*allow_dynamic_channel_dim=*/false,
       /*validation_status=*/Status::OK()},
      // SAME padding (Symmetric)
      {/*input_dims=*/{1, 1, 2, 3, 2},  // CDHW
       /*input=*/{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
       /*filter_dims=*/{3, 1, 1, 1, 1},  // DRSCK
       /*filter=*/{-1, 0, 1},
       /*strides=*/{1, 1, 1, 1, 1},
       /*padding=*/"SAME",
       /*data_format=*/"NCDHW",
       /*dilations=*/{1, 1, 1, 1, 1},
       /*is_conv3d_backprop=*/false,
       /*expected_output_dims=*/{1, 1, 2, 3, 2},
       // Swaps front two depths, negates
       /*expected_output=*/{6, 7, 8, 9, 10, 11, 0, -1, -2, -3, -4, -5},
       /*allow_dynamic_channel_dim=*/false,
       /*validation_status=*/Status::OK()

      },
      // NDHWC (multi-channel)
      {/*input_dims=*/{1, 2, 3, 2, 2},  // DHWC
       /*input=*/{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
       /*filter_dims=*/{2, 1, 1, 2, 1},  // DRSCK
       /*filter=*/{-1, 1, 1, -1},
       /*strides=*/{1, 1, 1, 1, 1},
       /*padding=*/"VALID",
       /*data_format=*/"NDHWC",
       /*dilations=*/{1, 1, 1, 1, 1},
       /*is_conv3d_backprop=*/false,
       /*expected_output_dims=*/{1, 1, 3, 2, 1},
       /*expected_output=*/{0, 0, 0, 0, 0, 0},  // Filters oppose each-other
       /*allow_dynamic_channel_dim=*/false,
       /*validation_status=*/Status::OK()},
      // Dilated
      {/*input_dims=*/{1, 1, 3, 3, 3},  // CDHW
       /*input=*/{1,   1,   1,   1,   1, 1, 1, 1, 1, -10, -10, -10, -10, -10,
                  -10, -10, -10, -10, 7, 7, 7, 7, 7, 7,   7,   7,   7},
       /*filter_dims=*/{2, 1, 1, 1, 1},  // DRSCK
       /*filter=*/{1, 1},
       /*strides=*/{1, 1, 1, 1, 1},
       /*padding=*/"VALID",
       /*data_format=*/"NCDHW",
       /*dilations=*/{1, 1, 2, 1, 1},
       /*is_conv3d_backprop=*/false,
       /*expected_output_dims=*/{1, 1, 1, 3, 3},
       // Only front depth is valid, skips neg values
       /*expected_output=*/{8, 8, 8, 8, 8, 8, 8, 8, 8},
       /*allow_dynamic_channel_dim=*/false,
       /*validation_status=*/Status::OK()},
      // Strided
      {/*input_dims=*/{1, 1, 3, 3, 3},
       /*input=*/{1, 0, 2, 0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 5, 0, 6, 0, 0, 0, 7, 0, 8},
       /*filter_dims=*/{1, 1, 1, 1, 1},
       /*filter=*/{1},
       /*strides=*/{1, 1, 2, 2, 2},
       /*padding=*/"VALID",
       /*data_format=*/"NCDHW",
       /*dilations=*/{1, 1, 1, 1, 1},
       /*is_conv3d_backprop=*/false,
       /*expected_output_dims=*/{1, 1, 2, 2, 2},
       // Should only pick up the corners
       /*expected_output=*/{1, 2, 3, 4, 5, 6, 7, 8},
       /*allow_dynamic_channel_dim=*/false,
       /*validation_status=*/Status::OK()},
      // Transpose Strided
      {/*input_dims=*/{1, 1, 2, 2, 2},  // CDHW
       /*input=*/{1, 2, 3, 4, 5, 6, 7, 8},
       /*filter_dims=*/{1, 1, 1, 1, 1},
       /*filter=*/{1},
       /*strides=*/{1, 1, 2, 2, 2},
       /*padding=*/"VALID",
       /*data_format=*/"NCDHW",
       /*dilations=*/{1, 1, 1, 1, 1},
       /*is_conv3d_backprop=*/true,
       /*expected_output_dims=*/{1, 1, 3, 3, 3},
       /*expected_output=*/{1, 0, 2, 0, 0, 0, 3, 0, 4,   // Cube expands and
                            0, 0, 0, 0, 0, 0, 0, 0, 0,   // fills center
                            5, 0, 6, 0, 0, 0, 7, 0, 8},  // with zeroes
       /*allow_dynamic_channel_dim=*/false,
       /*validation_status=*/Status::OK()},
  };

  if (trt_mode_ == TrtTestMode::kDynamicShape) {
    ok_params.reserve(ok_params.size() + 2);
    const std::vector<float> common_input = CreateVectorIota<float>(3 * 3 * 3);
    // NCDHW - Dynamic Channel - Should fail in kDynamicShape
    ok_params.push_back(Conv3DTestParams{
        /*input_dims=*/{1, 1, 3, 3, 3},
        /*input=*/common_input,
        /*filter_dims=*/{1, 1, 1, 1, 1},
        /*filter=*/{1},
        /*strides=*/{1, 1, 2, 2, 2},
        /*padding=*/"VALID",
        /*data_format=*/"NCDHW",
        /*dilations=*/{1, 1, 1, 1, 1},
        /*is_conv3d_backprop=*/false,
        /*expected_output_dims=*/{},  // ignore, will fail anyway
        /*expected_output=*/{},       // ignore, will fail anyway
        /*allow_dynamic_channel_dim=*/true,
        /*validation_status=*/
        Status{error::INVALID_ARGUMENT, "Channel dimension must be static"}});
    // NDHWC - Dynamic Channel - Should fail in kDynamicShape
    ok_params.push_back(Conv3DTestParams{
        /*input_dims=*/{1, 3, 3, 3, 1},
        /*input=*/common_input,
        /*filter_dims=*/{1, 1, 1, 1, 1},
        /*filter=*/{1},
        /*strides=*/{1, 2, 2, 2, 1},
        /*padding=*/"VALID",
        /*data_format=*/"NDHWC",
        /*dilations=*/{1, 1, 1, 1, 1},
        /*is_conv3d_backprop=*/false,
        /*expected_output_dims=*/{},  // ignore, will fail anyway
        /*expected_output=*/{},       // ignore, will fail anyway
        /*allow_dynamic_channel_dim=*/true,
        /*validation_status=*/
        Status{error::INVALID_ARGUMENT, "Channel dimension must be static"}});
  }

  for (auto p : ok_params) {
    TestConv3D(this, p);
  }
}

template <typename T>
NodeDef CreatePoolOp(DataType tf_type, std::vector<int> ksize,
                     std::vector<int> strides, string padding,
                     string data_format) {
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), tf_type);
  typename T::Attrs attrs;
  attrs.data_format_ = data_format;
  return T(s.WithOpName("my_pool"), input, ksize, strides, padding, attrs)
      .operation.node()
      ->def();
}
TEST_P(OpConverter_FP32_Test, ConvertPool) {
  // Get nodedef for MaxPool and AvgPool layers (2D or 3D).
  auto get_pool_nodedef =
      [](DataType tf_type, int nDim, std::vector<int> ksize = {},
         std::vector<int> strides = {}, string padding = "SAME",
         string data_format = "", const bool is_max_pooling = true) -> NodeDef {
    if (ksize.empty()) {
      ksize = nDim == 2 ? std::vector<int>{1, 1, 1, 1}
                        : std::vector<int>{1, 1, 1, 1, 1};
    }
    if (strides.empty()) {
      strides = nDim == 2 ? std::vector<int>{1, 1, 1, 1}
                          : std::vector<int>{1, 1, 1, 1, 1};
    }
    if (data_format == "") {
      data_format = nDim == 2 ? "NCHW" : "NCDHW";
    }
    if (is_max_pooling) {
      if (nDim == 3) {
        return CreatePoolOp<ops::MaxPool3D>(tf_type, ksize, strides, padding,
                                            data_format);
      } else {
        return CreatePoolOp<ops::MaxPool>(tf_type, ksize, strides, padding,
                                          data_format);
      }
    } else {
      if (nDim == 3) {
        return CreatePoolOp<ops::AvgPool3D>(tf_type, ksize, strides, padding,
                                            data_format);
      } else {
        return CreatePoolOp<ops::AvgPool>(tf_type, ksize, strides, padding,
                                          data_format);
      }
    }
  };

  std::vector<int> test_nDims{2, 3};

  for (int nDim : test_nDims) {
    // Input is weights, should fail.
    Reset();
    NodeDef node_def = get_pool_nodedef(tf_type_, nDim);

    AddTestWeights<float>("input", {1, 1, 1, 2, 3}, {1, 2, 3, 4, 5, 6});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        StrCat("The input \"input\" for ", node_def.op(), " must be a tensor"));
  }

  struct TestParams {
    std::vector<int> input_dims;
    std::vector<float> input;
    std::vector<int> ksize;
    std::vector<int> strides;
    string padding;
    string data_format;
    std::vector<int> expected_output_dims;
    // The expected outputs for the following operations: MaxPool2D, AvgPool2D,
    // MaxPool3D, AvgPool3D
    std::vector<std::vector<float>> expected_outputs;
  };

  // We use common_input as the input to test both 2D and 3D pooling operations,
  // to simplify TestParams. For 2D operations, only the first 1/3 of the values
  // are used.
  const std::vector<float> common_input{-4, 2,  15, 3, 6,   -3, 22, 1,   88,
                                        56, 36, 1,  1, 105, 1,  16, -28, 1,
                                        42, 9,  3,  1, 7,   1,  11, 61,  5};
  // The output of 2D ops for the case where the op is equivalent to the
  // identity op.
  const std::vector<float> common_2d_output{-4, 2, 15, 3, 6, -3, 22, 1, 88};
  std::vector<TestParams> ok_params = {
      // Basic - just 1x1 max pooling - input = output
      TestParams{
          /*input_dims=*/{1, 1, 3, 3, 3},
          /*input=*/common_input,
          /*ksize=*/{1, 1, 1, 1, 1},
          /*strides=*/{1, 1, 1, 1, 1},
          /*padding=*/"VALID",
          /*data_format=*/"NCDHW",
          /*expected_output_dims=*/{1, 1, 3, 3, 3},
          /*expected_outputs=*/
          {common_2d_output, common_2d_output, common_input, common_input}},
      // Basic - just 1x1 max pooling - input = output, SAME padding
      TestParams{
          /*input_dims=*/{1, 1, 3, 3, 3},
          /*input=*/common_input,
          /*ksize=*/{1, 1, 1, 1, 1},
          /*strides=*/{1, 1, 1, 1, 1},
          /*padding=*/"SAME",
          /*data_format=*/"NCDHW",
          /*expected_output_dims=*/{1, 1, 3, 3, 3},
          /*expected_outputs=*/
          {common_2d_output, common_2d_output, common_input, common_input}},
      // 3x3 pooling NCDHW
      TestParams{/*input_dims=*/{1, 1, 3, 3, 3},
                 /*input=*/common_input,
                 /*ksize=*/{1, 1, 3, 3, 3},
                 /*strides=*/{1, 1, 1, 1, 1},
                 /*padding=*/"VALID",
                 /*data_format=*/"NCDHW",
                 /*expected_output_dims=*/{1, 1, 1, 1, 1},
                 /*expected_outputs=*/{{88}, {14.444445}, {105}, {17}}},
      // 3x3 pooling, NDHWC
      TestParams{/*input_dims=*/{1, 3, 3, 3, 1},
                 /*input=*/common_input,
                 /*ksize=*/{1, 3, 3, 3, 1},
                 /*strides=*/{1, 1, 1, 1, 1},
                 /*padding=*/"VALID",
                 /*data_format=*/"NDHWC",
                 /*expected_output_dims=*/{1, 1, 1, 1, 1},
                 /*expected_outputs=*/{{88}, {14.444445}, {105}, {17}}},
      // Strided
      TestParams{/*input_dims=*/{1, 1, 3, 3, 3},
                 /*input=*/{1, 0, 2, 0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 5, 0, 6, 0, 0, 0, 7, 0, 8},
                 /*ksize=*/{1, 1, 1, 1, 1},
                 /*strides=*/{1, 1, 2, 2, 2},
                 /*padding=*/"VALID",
                 /*data_format=*/"NCDHW",
                 /*expected_output_dims=*/{1, 1, 2, 2, 2},
                 /*expected_outputs=*/
                 {{1, 2, 3, 4},  // Should only pick up the corners
                  {1, 2, 3, 4},
                  {1, 2, 3, 4, 5, 6, 7, 8},
                  {1, 2, 3, 4, 5, 6, 7, 8}}},
  };

  for (auto p : ok_params) {
    int test_counter = 0;
    for (int nDim : test_nDims) {
      auto input = p.input;
      auto input_dims = p.input_dims;
      auto ksize = p.ksize;
      auto strides = p.strides;
      auto expected_output_dims = p.expected_output_dims;
      std::string data_format = p.data_format;
      if (nDim == 2) {
        input.resize(9);
        data_format = p.data_format == "NDHWC" ? "NHWC" : "NCHW";
        // Remove one of the spatial dimensions
        input_dims.erase(input_dims.begin() + 2);
        ksize.erase(ksize.begin() + 2);
        strides.erase(strides.begin() + 2);
        expected_output_dims.erase(expected_output_dims.begin() + 2);
      }
      for (bool is_max_pooling : {true, false}) {
        Reset();
        NodeDef node_def =
            get_pool_nodedef(tf_type_, nDim, ksize, strides, p.padding,
                             data_format, is_max_pooling);
        AddTestTensor("input", input_dims, input);
        TestOpConverter("my_pool", node_def, expected_output_dims, Status::OK(),
                        Status::OK(),
                        ElementsAreArray(p.expected_outputs.at(test_counter)));
        test_counter++;
      }
    }
  }
}

TEST_P(OpConverter_FP32_FP16_Test, ConvertTopK) {
  // Get the NodeDef for TopKV2.
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), tf_type_);
  auto weights = ops::Placeholder(s.WithOpName("weights"), DT_INT32);
  auto topk = ops::TopK(s.WithOpName("my_topk"), input, weights);
  const NodeDef& node_def = topk.operation.node()->def();
  {
    // K is a tensor, should fail.
    Reset();
    AddTestTensor("input", {1, 1, 2, 3});
    AddTestTensor("weights", {1}, DT_INT32, {});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "The input \"k\" for TopKV2 must be a constant");
  }
  {
    // Ok.
    Reset();
    AddTestTensor("input", {1, 1, 2, 5}, {-9, 3, 5, 1, 6, -5, 7, 1, 0, -1});
    AddTestWeights<int32>("weights", {1}, {2});
    std::vector<std::vector<int>> expected_output_dims{{1, 1, 2, 2},
                                                       {1, 1, 2, 2}};
    TestOpConverterMultiOut("my_topk", node_def, expected_output_dims,
                            Status::OK(), Status::OK(),
                            {ElementsAre(6, 5, 7, 1), ElementsAre(4, 2, 1, 2)},
                            {tf_type_, DT_INT32});
  }
}

struct DataFormatVecPermuteTestParams {
  string dst_format;
  string src_format;
  std::vector<int> x_shape;
  std::vector<int> x;
  bool x_is_tensor;
  std::vector<int> expected_output;
  Status conversion_status;
};

NodeDef GetDataFormatVecPermuteNodeDef(string dst_format, string src_format,
                                       std::vector<int>& x_shape) {
  Scope s = Scope::NewRootScope();
  PartialTensorShape tensor_shape;
  auto x = ops::Placeholder(s.WithOpName("x"), DT_INT32);
  const auto attrs = ops::DataFormatVecPermute::Attrs()
                         .DstFormat(dst_format)
                         .SrcFormat(src_format);
  auto dfvp = ops::DataFormatVecPermute(s.WithOpName("my_dfvp"), x, attrs);
  return dfvp.operation.node()->def();
}

TEST_P(OpConverter_INT32_Test, ConvertDataFormatVecPermute) {
  Status implicit_error = Status{
      error::UNIMPLEMENTED, "Implicit batch mode not supported, at my_dfvp"};

  std::vector<DataFormatVecPermuteTestParams> test_params = {
      // 1D case with tensor.
      DataFormatVecPermuteTestParams{
          /*dst_format=*/"NCHW",
          /*src_format=*/"NHWC",
          /*x_shape=*/{4},
          /*x=*/{1, 2, 3, 4},
          /*x_is_tensor=*/true,
          /*expected_output=*/{1, 4, 2, 3},
          /*conversion_status=*/trt_mode_ == TrtTestMode::kImplicitBatch
              ? implicit_error
              : Status::OK()},
      // 1D case with weights.
      DataFormatVecPermuteTestParams{
          /*dst_format=*/"NCHW",
          /*src_format=*/"NHWC",
          /*x_shape=*/{4},
          /*x=*/{1, 2, 3, 4},
          /*x_is_tensor=*/false,
          /*expected_output=*/{1, 4, 2, 3},
          /*conversion_status=*/trt_mode_ == TrtTestMode::kImplicitBatch
              ? implicit_error
              : Status::OK()},
      // 2D case with tensor.
      DataFormatVecPermuteTestParams{
          /*dst_format=*/"NCHW",
          /*src_format=*/"NHWC",
          /*x_shape=*/{4, 2},
          /*x=*/{1, 2, 3, 4, 5, 6, 7, 8},
          /*x_is_tensor=*/true,
          /*expected_output=*/{1, 2, 7, 8, 3, 4, 5, 6},
          /*conversion_status=*/trt_mode_ == TrtTestMode::kImplicitBatch
              ? implicit_error
              : Status::OK()},
      // 2D case with weights.
      DataFormatVecPermuteTestParams{
          /*dst_format=*/"NCHW",
          /*src_format=*/"NHWC",
          /*x_shape=*/{4, 2},
          /*x=*/{1, 2, 3, 4, 5, 6, 7, 8},
          /*x_is_tensor=*/false,
          /*expected_output=*/{1, 2, 7, 8, 3, 4, 5, 6},
          /*conversion_status=*/trt_mode_ == TrtTestMode::kImplicitBatch
              ? implicit_error
              : Status::OK()},
      // Format of size 5.
      DataFormatVecPermuteTestParams{
          /*dst_format=*/"NCDHW",
          /*src_format=*/"NDHWC",
          /*x_shape=*/{5, 2},
          /*x=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
          /*x_is_tensor=*/true,
          /*expected_output=*/{1, 2, 9, 10, 3, 4, 5, 6, 7, 8},
          /*conversion_status=*/trt_mode_ == TrtTestMode::kImplicitBatch
              ? implicit_error
              : Status::OK()},
      // Input of size 2: treat the elements as spatial dimensions.
      DataFormatVecPermuteTestParams{
          /*dst_format=*/"NCWH",
          /*src_format=*/"NHWC",
          /*x_shape=*/{2, 2},
          /*x=*/{1, 2, 3, 4},
          /*x_is_tensor=*/true,
          /*expected_output=*/{3, 4, 1, 2},
          /*conversion_status=*/trt_mode_ == TrtTestMode::kImplicitBatch
              ? implicit_error
              : Status::OK()},
      // Input of size 3: treat the elements as spatial dimensions.
      DataFormatVecPermuteTestParams{
          /*dst_format=*/"NCHWD",
          /*src_format=*/"NDHWC",
          /*x_shape=*/{3},
          /*x=*/{1, 2, 3},
          /*x_is_tensor=*/true,
          /*expected_output=*/{2, 3, 1},
          /*conversion_status=*/trt_mode_ == TrtTestMode::kImplicitBatch
              ? implicit_error
              : Status::OK()},
      // Invalid rank, should fail.
      DataFormatVecPermuteTestParams{
          /*dst_format=*/"NCHW",
          /*src_format=*/"NHWC",
          /*x_shape=*/{2, 2, 2},
          /*x=*/{1, 2, 3, 4, 5, 6, 7, 8},
          /*x_is_tensor=*/true,
          /*expected_output=*/{},
          /*conversion_status=*/trt_mode_ == TrtTestMode::kImplicitBatch
              ? implicit_error
              : Status{error::INVALID_ARGUMENT,
                       "Input must be a vector or matrix, but got rank 3, at "
                       "my_dfvp"}},
      // Invalid size for 1D input, should fail.
      DataFormatVecPermuteTestParams{
          /*dst_format=*/"NCHW",
          /*src_format=*/"NHWC",
          /*x_shape=*/{3},
          /*x=*/{1, 2, 3},
          /*x_is_tensor=*/true,
          /*expected_output=*/{},
          /*conversion_status=*/trt_mode_ == TrtTestMode::kImplicitBatch
              ? implicit_error
              : Status{error::INVALID_ARGUMENT,
                       "1D input must be of size 2 or 4, but got size 3, at "
                       "my_dfvp"}},
      // Invalid first dim for 2D input, should fail.
      DataFormatVecPermuteTestParams{
          /*dst_format=*/"NCDHW",
          /*src_format=*/"NDHWC",
          /*x_shape=*/{4, 2},
          /*x=*/{1, 2, 3, 4, 5, 6, 7, 8},
          /*x_is_tensor=*/true,
          /*expected_output=*/{},
          /*conversion_status=*/trt_mode_ == TrtTestMode::kImplicitBatch
              ? implicit_error
              : Status{error::INVALID_ARGUMENT,
                       "First dimension of 2D input must be of size 3 or 5, "
                       "but got shape (4, 2), at my_dfvp"}},
      // Invalid second dim for 2D input, should fail.
      DataFormatVecPermuteTestParams{
          /*dst_format=*/"NCHW",
          /*src_format=*/"NHWC",
          /*x_shape=*/{4, 3},
          /*x=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
          /*x_is_tensor=*/true,
          /*expected_output=*/{},
          /*conversion_status=*/trt_mode_ == TrtTestMode::kImplicitBatch
              ? implicit_error
              : Status{error::INVALID_ARGUMENT,
                       "Second dimension of 2D input must be of size 2, but "
                       "got shape (4, 3), at my_dfvp"}},
  };

  for (auto p : test_params) {
    Reset();
    const NodeDef node_def =
        GetDataFormatVecPermuteNodeDef(p.dst_format, p.src_format, p.x_shape);

    if (p.x_is_tensor) {
      AddTestTensor("x", p.x_shape, DT_INT32, p.x, p.x_shape);
    } else {
      AddTestWeights("x", p.x_shape, p.x, DT_INT32);
    }

    TestOpConverter("my_dfvp", node_def, p.x_shape, p.conversion_status,
                    Status::OK(), ElementsAreArray(p.expected_output));
  }
}

TEST_P(OpConverter_FP32_FP16_INT32_Test, ConvertGather) {
  // Get the NodeDef for GatherV2.
  Scope s = Scope::NewRootScope();
  auto params = ops::Placeholder(s.WithOpName("params"), tf_type_);
  auto indices = ops::Placeholder(s.WithOpName("indices"), DT_INT32);
  auto axis = ops::Placeholder(s.WithOpName("axis"), DT_INT32);
  auto gather = ops::GatherV2(s.WithOpName("my_gather"), params, indices, axis);
  const NodeDef& node_def = gather.operation.node()->def();
  {
    // Axis is a tensor, should fail.
    Reset();
    AddTestTensor("params", {1, 1, 2, 3}, tf_type_, {});
    AddTestTensor("indices", {1, 2}, DT_INT32, {});
    AddTestTensor("axis", {1}, DT_INT32, {});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"axis\" for GatherV2 must be a constant");
  }
  {
    // Axis is out of bounds, should fail.
    Reset();
    AddTestTensor("params", {1, 1, 2, 3});
    AddTestTensor("indices", {1, 2}, DT_INT32, {});
    AddTestWeights<int32>("axis", {1}, {4});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "Axis value of 4 is out of bounds, must be in "
                               "range [-4, 4)");
  }

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
    bool indices_is_tensor;
    Status conversion_status;
    Status runtime_status;
    Status add_index_status;
  };

  // Input is the same {1, 2, 3, 4, 5, 6} for all cases.
  const std::vector<int> params_input = {1, 2, 3, 4, 5, 6};

  std::vector<TestParams> test_params = {
      // Axis is batch dimension, should fail in implicit batch mode.
      TestParams{/*params_shape=*/{2, 1, 1, 3},
                 /*indices_shape=*/{2},
                 /*indices=*/{1, 0},
                 /*axis=*/0,
                 /*expected_output_shape=*/{2, 1, 1, 3},
                 /*expected_output=*/{4, 5, 6, 1, 2, 3},
                 /*params_is_tensor=*/true,
                 /*indices_is_tensor=*/true,
                 /*conversion_status=*/trt_mode_ == TrtTestMode::kImplicitBatch
                     ? Status{error::UNIMPLEMENTED,
                              "TensorRT does not allow "
                              "manipulation of the batch dimension"}
                     : Status::OK()},
      // Batch size of indices is not 1 when params and indices are tensors.
      TestParams{/*params_shape=*/{2, 1, 3},
                 /*indices_shape=*/{2, 1},
                 /*indices=*/{2, 0},
                 /*axis=*/2,
                 /*expected_output_shape=*/{2, 1, 2, 1},
                 /*expected_output=*/{3, 1, 6, 4},
                 /*params_is_tensor=*/true,
                 /*indices_is_tensor=*/true,
                 /*conversion_status=*/trt_mode_ == TrtTestMode::kImplicitBatch
                     ? Status{error::UNIMPLEMENTED,
                              "Params and indices must have a"
                              " batch size of 1 when params and indices are "
                              "both tensors or both"
                              " constants."}
                     : Status::OK()},
      // Batch size of indices is not 1 when params is tensor and indices are
      // constant.
      TestParams{/*params_shape=*/{2, 1, 3},
                 /*indices_shape=*/{2, 1},
                 /*indices=*/{2, 0},
                 /*axis=*/2,
                 /*expected_output_shape=*/{2, 1, 2, 1},
                 /*expected_output=*/{3, 1, 6, 4},
                 /*params_is_tensor=*/true,
                 /*indices_is_tensor=*/false,
                 /*conversion_status=*/Status::OK()},
      // Axis is not zero when params is a weight, should fail in implicit batch
      // mode.
      TestParams{/*params_shape=*/{2, 1, 3},
                 /*indices_shape=*/{2},
                 /*indices=*/{1, 2},
                 /*axis=*/2,
                 /*expected_output_shape=*/{2, 1, 2},
                 /*expected_output=*/{2, 3, 5, 6},
                 /*params_is_tensor=*/false,
                 /*indices_is_tensor=*/true,
                 /*conversion_status=*/trt_mode_ == TrtTestMode::kImplicitBatch
                     ? Status{error::UNIMPLEMENTED,
                              "The input axis must be zero when "
                              "params is a weight."}
                     : Status::OK()},
      // Params with only batch dimension.
      TestParams{
          /*params_shape=*/{6},
          /*indices_shape=*/{2},
          /*indices=*/{1, 3},
          /*axis=*/0,
          /*expected_output_shape=*/{2},
          /*expected_output=*/{2, 4},
          /*params_is_tensor=*/true,
          /*indices_is_tensor=*/true,
          /*conversion_status=*/trt_mode_ == TrtTestMode::kImplicitBatch
              ? Status{error::UNIMPLEMENTED,
                       "TensorRT does not allow "
                       "manipulation of the batch dimension"}
              : Status::OK(),
          /*runtime_status=*/Status::OK(),
          /*add_index_status=*/trt_mode_ == TrtTestMode::kImplicitBatch
              ? Status{error::INVALID_ARGUMENT,
                       "Batch size doesn't match for "
                       "tensor indices: Provided batch size does not match "
                       "converter batch size: 2 vs 6"}
              : Status::OK()},
      // Vector indices, and output rank is rank(params).
      TestParams{
          /*params_shape=*/{1, 1, 2, 3},
          /*indices_shape=*/{1},
          /*indices=*/{0},
          /*axis=*/3,
          /*expected_output_shape=*/{1, 1, 2, 1},
          /*expected_output=*/{1, 4},
          /*params_is_tensor=*/true,
          /*indices_is_tensor=*/true,
      },
      TestParams{
          /*params_shape=*/{1, 1, 2, 3},
          /*indices_shape=*/{1},
          /*indices=*/{1},
          /*axis=*/2,
          /*expected_output_shape=*/{1, 1, 1, 3},
          /*expected_output=*/{4, 5, 6},
          /*params_is_tensor=*/true,
          /*indices_is_tensor=*/true,
      },
      // Indices with rank>1, and output rank is rank(params) + rank(indices) -
      // 1
      TestParams{
          /*params_shape=*/{1, 1, 2, 3},
          /*indices_shape=*/{1, 1},
          /*indices=*/{0},
          /*axis=*/3,
          /*expected_output_shape=*/{1, 1, 2, 1, 1},
          /*expected_output=*/{1, 4},
          /*params_is_tensor=*/true,
          /*indices_is_tensor=*/true,
      },
      TestParams{
          /*params_shape=*/{1, 1, 2, 3},
          /*indices_shape=*/{1, 1},
          /*indices=*/{1},
          /*axis=*/3,
          /*expected_output_shape=*/{1, 1, 2, 1, 1},
          /*expected_output=*/{2, 5},
          /*params_is_tensor=*/true,
          /*indices_is_tensor=*/true,
      },
      TestParams{
          /*params_shape=*/{1, 1, 2, 3},
          /*indices_shape=*/{1, 1},
          /*indices=*/{2},
          /*axis=*/-1,
          /*expected_output_shape=*/{1, 1, 2, 1, 1},
          /*expected_output=*/{3, 6},
          /*params_is_tensor=*/true,
          /*indices_is_tensor=*/true,
      },
      TestParams{
          /*params_shape=*/{1, 1, 2, 3},
          /*indices_shape=*/{1, 3},
          /*indices=*/{2, 0, 1},
          /*axis=*/3,
          /*expected_output_shape=*/{1, 1, 2, 1, 3},
          /*expected_output=*/{3, 1, 2, 6, 4, 5},
          /*params_is_tensor=*/true,
          /*indices_is_tensor=*/true,
      },
      TestParams{
          /*params_shape=*/{1, 3, 2},
          /*indices_shape=*/{1, 2, 2},
          /*indices=*/{0, 0, 1, 0},
          /*axis=*/2,
          /*expected_output_shape=*/{1, 3, 1, 2, 2},
          /*expected_output=*/{1, 1, 2, 1, 3, 3, 4, 3, 5, 5, 6, 5},
          /*params_is_tensor=*/true,
          /*indices_is_tensor=*/true,
      },
      TestParams{
          /*params_shape=*/{1, 2, 3},
          /*indices_shape=*/{1},
          /*indices=*/{0},
          /*axis=*/0,
          /*expected_output_shape=*/{1, 2, 3},
          /*expected_output=*/{1, 2, 3, 4, 5, 6},
          /*params_is_tensor=*/false,
          /*indices_is_tensor=*/true,
      },
      TestParams{
          /*params_shape=*/{3, 2},
          /*indices_shape=*/{1, 2},
          /*indices=*/{0, 1},
          /*axis=*/0,
          /*expected_output_shape=*/{1, 2, 2},
          /*expected_output=*/{1, 2, 3, 4},
          /*params_is_tensor=*/false,
          /*indices_is_tensor=*/true,
      },
      TestParams{
          /*params_shape=*/{2, 3},
          /*indices_shape=*/{1, 1, 2},
          /*indices=*/{0, 1},
          /*axis=*/0,
          /*expected_output_shape=*/{1, 1, 2, 3},
          /*expected_output=*/{1, 2, 3, 4, 5, 6},
          /*params_is_tensor=*/false,
          /*indices_is_tensor=*/true,
      },
      TestParams{
          /*params_shape=*/{3, 2},
          /*indices_shape=*/{2, 2},
          /*indices=*/{0, 2, 1, 0},
          /*axis=*/0,
          /*expected_output_shape=*/{2, 2, 2},
          /*expected_output=*/{1, 2, 5, 6, 3, 4, 1, 2},
          /*params_is_tensor=*/false,
          /*indices_is_tensor=*/true,
      },
      // Test cases in which indices constant
      TestParams{
          /*params_shape=*/{1, 1, 2, 3},
          /*indices_shape=*/{1, 1},
          /*indices=*/{0},
          /*axis=*/3,
          /*expected_output_shape=*/{1, 1, 2, 1, 1},
          /*expected_output=*/{1, 4},
          /*params_is_tensor=*/true,
          /*indices_is_tensor=*/false,
      },
      // Test cases in which both input and indices constant
      TestParams{/*params_shape=*/{1, 2, 3},
                 /*indices_shape=*/{1},
                 /*indices=*/{0},
                 /*axis=*/0,
                 /*expected_output_shape=*/{1, 2, 3},
                 /*expected_output=*/{1, 2, 3, 4, 5, 6},
                 /*params_is_tensor=*/false,
                 /*indices_is_tensor=*/false,
                 /*conversion_status=*/trt_mode_ == TrtTestMode::kImplicitBatch
                     ? Status{error::UNIMPLEMENTED,
                              "Params and indices must have a"
                              " batch size of 1 when params and indices are "
                              "both tensors or both"
                              " constants."}
                     : Status::OK()},
      TestParams{/*params_shape=*/{3, 2},
                 /*indices_shape=*/{2, 2},
                 /*indices=*/{0, 2, 1, 0},
                 /*axis=*/0,
                 /*expected_output_shape=*/{2, 2, 2},
                 /*expected_output=*/{1, 2, 5, 6, 3, 4, 1, 2},
                 /*params_is_tensor=*/false,
                 /*indices_is_tensor=*/false,
                 /*conversion_status=*/trt_mode_ == TrtTestMode::kImplicitBatch
                     ? Status{error::UNIMPLEMENTED,
                              "Params and indices must have a"
                              " batch size of 1 when params and indices are "
                              "both tensors or both"
                              " constants."}
                     : Status::OK()},
  };

  for (auto p : test_params) {
    Reset();

    if (p.params_is_tensor) {
      AddTestTensor("params", p.params_shape, params_input);
    } else {
      AddTestWeights("params", p.params_shape, params_input, tf_type_);
    }

    if (p.indices_is_tensor) {
      AddTestTensor("indices", p.indices_shape, DT_INT32, p.indices, {},
                    p.add_index_status);
    } else {
      std::vector<int> indices_shape(p.indices_shape);
      AddTestWeights("indices", indices_shape, p.indices, DT_INT32);
    }

    AddTestWeights<int32>("axis", {1}, {p.axis});
    TestOpConverter("my_gather", node_def, p.expected_output_shape,
                    p.conversion_status, p.runtime_status,
                    ElementsAreArray(p.expected_output));
  }
}

template <typename OpType>
NodeDef CreateReduceOp(DataType tf_type, bool keep_dims) {
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), tf_type);
  auto axis = ops::Placeholder(s.WithOpName("axis"), DT_INT32);
  typename OpType::Attrs op_attrs;
  op_attrs.keep_dims_ = keep_dims;
  auto op = OpType(s.WithOpName("my_reduce"), input, axis, op_attrs);
  return op.operation.node()->def();
}

// Applies reduction op on sub-sequences of input
// output[i] = reduce(input[m * i : m * (i +1)])
std::vector<float> CalcReduce(string op_name, std::vector<float> input, int m,
                              float (*op)(float, float), float init) {
  std::vector<float> output(input.size() / m);
  for (int i = 0; i < output.size(); i++) {
    auto begin = input.begin() + i * m;
    auto end = input.begin() + (i + 1) * m;
    output[i] = std::accumulate(begin, end, init, op);
    if (op_name == "Mean") {
      output[i] /= m;
    }
  }
  return output;
}
TEST_P(OpConverter_FP32_FP16_INT32_Test, ConvertReduce) {
  {
    // Input is weights, should fail.
    Reset();
    const NodeDef node_def = CreateReduceOp<ops::Sum>(tf_type_, false);
    AddTestWeights<float>("input", {1, 2, 3}, {-3, -2, -1, 0, 1, 2});
    AddTestWeights<int32>("axis", {1}, {1});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "The input \"input\" for Sum must be a tensor");
  }
  {
    // Axis is weights, should fail.
    Reset();
    const NodeDef node_def = CreateReduceOp<ops::Sum>(tf_type_, false);
    AddTestTensor("input", {1, 2, 3}, {-3, -2, -1, 0, 1, 2});
    AddTestTensor("axis", {1}, DT_INT32, {1});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "The input \"axis\" for Sum must be a constant");
  }
  using OpFunc = std::function<NodeDef(DataType, bool)>;
  using ValFunc = float (*)(float, float);
  struct ReduceTestDescriptor {
    string name;
    OpFunc get_node;
    ValFunc val_func;
    float init_val;
  };
  std::vector<ReduceTestDescriptor> op_test_info{
      {"Sum", CreateReduceOp<ops::Sum>, [](float x, float y) { return x + y; },
       0},
      {"Prod", CreateReduceOp<ops::Prod>,
       [](float x, float y) { return x * y; }, 1},
      {"Mean", CreateReduceOp<ops::Mean>,
       [](float x, float y) { return x + y; }, 0},
      {"Min", CreateReduceOp<ops::Min>,
       [](float x, float y) { return y < x ? y : x; }, 1000},
      {"Max", CreateReduceOp<ops::Max>,
       [](float x, float y) { return x < y ? y : x; }, -1000}};

  std::vector<float> input_values{1, 2, 3, 4, 5, 6};
  struct TestParams {
    std::vector<int> input_dims;
    std::vector<float> input_values;
    // Helper array contains the same elements as input but permuted in a way
    // that the reduction can be calculated over contiguous elements using
    // CalcReduce
    std::vector<float> helper_array;
    std::vector<int> axis;
    int stride;  // product of input_dims along axis
    Status conversion_status;
  };
  std::vector<TestParams> params{
      // Out of range tests
      TestParams{{2, 3, 1}, input_values, input_values, {3}, 3},
      TestParams{{2, 3, 1}, input_values, input_values, {-4}, 3},
      // Ok tests
      TestParams{{2, 3, 1}, input_values, {1, 4, 2, 5, 3, 6}, {0}, 2},
      TestParams{{2, 3, 1}, input_values, input_values, {1}, 3},
      TestParams{{2, 3, 1}, input_values, input_values, {2}, 1},
      TestParams{{2, 3, 1}, input_values, input_values, {0, 1}, 6},
      // Ok tests with negative axis values
      TestParams{{2, 3, 1}, input_values, {1, 4, 2, 5, 3, 6}, {-3}, 2},
      TestParams{{2, 3, 1}, input_values, input_values, {-2}, 3},
      TestParams{{2, 3, 1}, input_values, input_values, {-1}, 1},
      TestParams{{2, 3, 1}, input_values, input_values, {-3, 1}, 6},
  };

  for (bool keep_dims : {false, true}) {
    for (auto& op : op_test_info) {
      VLOG(2) << "Processing " << op.name << " with keep_dims=" << keep_dims;
      for (auto p : params) {
        SCOPED_TRACE(StrCat(op.name, keep_dims ? " & keep_dims" : ""));
        Reset();
        NodeDef node_def = op.get_node(tf_type_, keep_dims);

        AddTestTensor("input", p.input_dims, p.input_values);
        AddTestWeights<int32>("axis", {static_cast<int>(p.axis.size())},
                              p.axis);
        std::vector<int> expected_output_dims(p.input_dims);

        // Set expected output dim and conversion error messages
        for (int ax : p.axis) {
          int rank = p.input_dims.size();
          if (ax >= rank || ax < -rank) {
            p.conversion_status =
                errors::InvalidArgument("Axis value of ", ax,
                                        " is out of bounds, must be in "
                                        "range [",
                                        -rank, ", ", rank, ")");
          } else {
            int ax_positive = ax >= 0 ? ax : ax + rank;
            // Zero marks elements that we will remove later.
            expected_output_dims[ax_positive] = keep_dims ? 1 : 0;
            if (trt_mode_ == TrtTestMode::kImplicitBatch &&
                (ax == 0 || ax == -rank)) {
              p.conversion_status = errors::Unimplemented(
                  "TensorRT does not allow manipulation of the batch "
                  "dimension");
            }
          }
        }
        expected_output_dims.erase(std::remove(expected_output_dims.begin(),
                                               expected_output_dims.end(), 0),
                                   expected_output_dims.end());
        VLOG(2) << "out dims "
                << absl::StrCat("[", absl::StrJoin(expected_output_dims, ","),
                                "]");
        std::vector<float> expected_values = CalcReduce(
            op.name, p.helper_array, p.stride, op.val_func, op.init_val);

        if (tf_type_ == DT_INT32) {
          // We need to floor the float values in the `expected_values` vector.
          std::for_each(expected_values.begin(), expected_values.end(),
                        [](float& _n) { _n = std::floor(_n); });
        }

        TestOpConverter("my_reduce", node_def, expected_output_dims,
                        p.conversion_status, Status::OK(),
                        ArrayFloatNear(expected_values));
      }
    }
  }
}

NodeDef CreateCastOp(DataType tf_type) {
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), DT_HALF);
  return ops::Cast(s.WithOpName("my_unary"), input, DT_FLOAT)
      .operation.node()
      ->def();
}

TEST_P(OpConverter_FP32_UnaryTest, ConvertUnary) {
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
  ADD_OP("Erf", ops::Erf, std::erf);
  ADD_OP("Floor", ops::Floor, std::floor);
  ADD_OP("Log", ops::Log, std::log);
  ADD_OP("Neg", ops::Neg, [](float x) { return -x; });
  ADD_OP("Reciprocal", ops::Reciprocal, [](float x) { return 1.0f / x; });
#if IS_TRT_VERSION_GE(8, 2, 0, 0)
  ADD_OP("Round", ops::Round, [](float x) { return (float)std::round(x); });
  ADD_OP("Sign", ops::Sign,
         [](float x) { return x > 0 ? 1.0f : (x < 0 ? -1.0f : 0.0f); });
#endif
  ADD_OP("Rsqrt", ops::Rsqrt, [](float x) { return 1.0f / std::sqrt(x); });
  ADD_OP("Sin", ops::Sin, std::sin);
  ADD_OP("Sinh", ops::Sinh, std::sinh);
  ADD_OP("Sqrt", ops::Sqrt, std::sqrt);
  ADD_OP("Tan", ops::Tan, std::tan);
#undef ADD_OP

  std::vector<float> input_values{-0.9f, 0.6f, 0.0f, -3.5f, 100.0f, 2.9f};
  RunTests("Unary", *UnaryOperationMap(), op_map, input_values, "x");
}

TEST_P(OpConverter_BOOL_Test, ConvertBoolean) {
  std::vector<int> input_values{1, 0, 1, 0, 0, 1};
  using OpFunc = std::function<NodeDef(DataType)>;

  using ValFunc = int (*)(int);
  std::map<std::string, std::pair<OpFunc, ValFunc>> op_map;
#define ADD_OP(name, op, compute) \
  op_map[name] =                  \
      std::make_pair(CreateUnaryOp<op>, static_cast<ValFunc>(compute))
  ADD_OP("LogicalNot", ops::LogicalNot, [](int x) { return 1 - x; });
#undef ADD_OP

#if IS_TRT_VERSION_GE(8, 2, 0, 0)
  // The test does not actually run for TPT versions less than 8.2
  RunTests("LogicalUnary", *UnaryBooleanOperationMap(), op_map, input_values,
           "x");
#endif
}

// Get the NodeDef for ConcatV2.
// TODO(hinsu): Consider switching this to static function.
auto get_concat_nodedef = [](DataType dtype, int num_inputs) -> NodeDef {
  Scope s = Scope::NewRootScope();
  std::vector<Input> values;
  values.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    const string input_name = StrCat("values_", i);
    values.push_back(ops::Placeholder(s.WithOpName(input_name), dtype));
  }
  auto axis = ops::Placeholder(s.WithOpName("axis"), DT_INT32);
  auto concat = ops::Concat(s.WithOpName("my_concat"),
                            absl::Span<const Input>(values), axis);
  return concat.operation.node()->def();
};

TEST_P(OpConverter_FP32_FP16_INT32_Test, ConvertConcat) {
  {
    // Axis is a tensor, should fail.
    Reset();
    NodeDef node_def = get_concat_nodedef(tf_type_, 2);
    AddTestTensor("values_0", {1, 1, 2, 3});
    AddTestTensor("values_1", {1, 1, 2, 3});
    AddTestTensor("axis", {1});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"axis\" for ConcatV2 must be a constant");
  }
  {
    // Axis is out of bounds, should fail.
    Reset();
    NodeDef node_def = get_concat_nodedef(tf_type_, 2);
    AddTestTensor("values_0", {1, 1, 2, 3});
    AddTestTensor("values_1", {1, 1, 2, 3});
    AddTestWeights<int32>("axis", {1}, {4});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "Axis value of 4 is out of bounds, must be in "
                               "range [-4, 4)");
  }
  {
    // Inputs have inconsistent ranks, should fail.
    Reset();
    NodeDef node_def = get_concat_nodedef(tf_type_, 2);
    AddTestTensor("values_0", {1, 1, 2, 3});
    AddTestTensor("values_1", {1, 1, 6});
    AddTestWeights<int32>("axis", {1}, {1});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "Received inputs with inconsistent rank");
  }

  struct TestParams {
    std::vector<std::vector<int>> input_shapes;
    std::vector<std::vector<int>> input_values;
    std::vector<bool> inputs_are_tensors;
    int axis;
    std::vector<int> expected_output_dims;
    std::vector<int> expected_output;
    Status conversion_status;
    Status run_status;
  };

  const std::vector<std::vector<int>> common_input{CreateVectorIota<int>(6),
                                                   CreateVectorIota<int>(6, 6)};

  std::vector<TestParams> params = {
      {
          /*input_shapes=*/{{1, 1, 2, 3}, {1, 1, 2, 3}},
          /*input_values=*/common_input,
          /*inputs_are_tensors=*/{true, true},
          /*axis=*/1,
          /*expected_output_dims=*/{1, 2, 2, 3},
          /*expected_output=*/CreateVectorIota<int>(12),
      },
      {
          /*input_shapes=*/{{1, 1, 2, 3}, {1, 1, 2, 3}},
          /*input_values=*/common_input,
          /*inputs_are_tensors=*/{true, true},
          /*axis=*/2,
          /*expected_output_dims=*/{1, 1, 4, 3},
          /*expected_output=*/CreateVectorIota<int>(12),
      },
      {
          /*input_shapes=*/{{1, 1, 2, 3}, {1, 1, 2, 3}},
          /*input_values=*/common_input,
          /*inputs_are_tensors=*/{true, true},
          /*axis=*/3,
          /*expected_output_dims=*/{1, 1, 2, 6},
          /*expected_output=*/
          {0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11},
      },
      {
          /*input_shapes=*/{{1, 1}, {1, 2}, {1, 3}, {1, 1}, {1, 1}, {1, 2}},
          /*input_values=*/
          {{1}, {2, 3}, {4, 5, 6}, {7}, {8}, {9, 10}},
          /*inputs_are_tensors=*/{true, true, true, true, true, true},
          /*axis=*/1,
          /*expected_output_dims=*/{1, 10},
          /*expected_output=*/
          CreateVectorIota<int>(10, /*start_value=*/1),
      },
      {
          // An input is a weight
          /*input_shapes=*/{{1, 1, 2, 3}, {1, 1, 2, 3}},
          /*input_values=*/common_input,
          /*inputs_are_tensors=*/{true, false},
          /*axis=*/1,
          /*expected_output_dims=*/{1, 2, 2, 3},
          /*expected_output=*/CreateVectorIota<int>(12),
          /*conversion_status=*/trt_mode_ == TrtTestMode::kImplicitBatch
              ? errors::Unimplemented(
                    "The input \"values_1\" for ConcatV2 must be a tensor")
              : Status::OK(),
          /*run_status=*/Status::OK(),
      },
      {
          // An input is a weight
          /*input_shapes=*/{{1, 1, 2, 3}, {1, 1, 2, 3}},
          /*input_values=*/common_input,
          /*inputs_are_tensors=*/{false, false},
          /*axis=*/1,
          /*expected_output_dims=*/{1, 2, 2, 3},
          /*expected_output=*/CreateVectorIota<int>(12),
          /*conversion_status=*/trt_mode_ == TrtTestMode::kImplicitBatch
              ? errors::Unimplemented(
                    "The input \"values_0\" for ConcatV2 must be a tensor")
              : Status::OK(),
          /*run_status=*/Status::OK(),
      },
      {
          // Axis is batch dimension, should fail in implicit batch mode.
          /*input_shapes=*/{{1, 1, 2, 3}, {1, 1, 2, 3}},
          /*input_values=*/common_input,
          /*inputs_are_tensors=*/{true, true},
          /*axis=*/0,
          /*expected_output_dims=*/{2, 1, 2, 3},
          /*expected_output=*/CreateVectorIota<int>(12),
          /*conversion_status=*/trt_mode_ == TrtTestMode::kImplicitBatch
              ? errors::Unimplemented(
                    "TensorRT does not allow manipulation of the "
                    "batch dimension")
              : Status::OK(),
      },
      {
          // Inconsistent input shape, runtime error in dynamic shape mode.
          /*input_shapes=*/{{1, 1, 2, 3}, {1, 1, 3, 2}},
          /*input_values=*/common_input,
          /*inputs_are_tensors=*/{true, true},
          /*axis=*/1,
          /*expected_output_dims=*/{2, 1, 2, 3},
          /*expected_output=*/CreateVectorIota<int>(12),
          trt_mode_ != TrtTestMode::kDynamicShape
              ? errors::InvalidArgument(
                    "Received inputs with inconsistent shape")
              : Status::OK(),
          errors::InvalidArgument(""),
      }};

  for (auto p : params) {
    Reset();
    const int num_inputs = p.input_shapes.size();
    EXPECT_EQ(num_inputs, p.input_values.size());

    NodeDef node_def = get_concat_nodedef(tf_type_, num_inputs);

    // Create inputs.
    for (int j = 0; j < num_inputs; ++j) {
      string name = StrCat("values_", j);

      if (!p.inputs_are_tensors[j]) {
        AddTestWeights(name, p.input_shapes[j], p.input_values[j], tf_type_);
      } else {
        AddTestTensor(name, p.input_shapes[j], p.input_values[j]);
      }
    }
    AddTestWeights<int32>("axis", {1}, {p.axis});

    TestOpConverter("my_concat", node_def, p.expected_output_dims,
                    p.conversion_status, p.run_status,
                    ElementsAreArray(p.expected_output));
  }
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

  const std::vector<CType> common_input = CreateVectorIota<CType>(6);
  std::vector<TestParams> ok_params = {
      // Identity (num_split = 1)
      {/*input_shape=*/{1, 2, 3}, /*value=*/common_input, /*axis=*/1,
       /*num_split=*/1, /*expected_output_dims=*/{1, 2, 3},
       /*expected_outputs=*/{CreateVectorIota<CType>(6)}},
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
       {CreateVectorIota<CType>(3), CreateVectorIota<CType>(3, CType(3))}},
  };

  for (int i = 0; i < ok_params.size(); ++i) {
    test->Reset();
    NodeDef node_def = get_split_nodedef(dtype, ok_params[i].num_split);
    // Create inputs.
    test->AddTestWeights<int32>("axis", {1}, {ok_params[i].axis});
    nvinfer1::DataType trt_type;
    TF_ASSERT_OK(TfTypeToTrtType(dtype, &trt_type));
    test->AddTestTensor("value", ok_params[i].input_shape, 1, trt_type);
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
      EXPECT_THAT(outputs[j].tensor()->getDimensions(),
                  DimsAreArray(ok_params[i].expected_output_dims));
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
        "The input \"axis\" for Split must be a constant");
  }
  {
    // Axis is out of bounds, should fail.
    Reset();
    NodeDef node_def = get_split_nodedef(DT_FLOAT, 1);
    AddTestWeights<int32>("axis", {1}, {4});
    AddTestTensor("value", {1, 2, 3});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "Axis value of 4 is out of bounds, must be in "
                               "range [-4, 4)");
  }
  {
    // Axis is out of bounds (negative), should fail.
    Reset();
    NodeDef node_def = get_split_nodedef(DT_FLOAT, 1);
    AddTestWeights<int32>("axis", {1}, {-5});
    AddTestTensor("value", {1, 2, 3});
    RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                               "Axis value of -5 is out of bounds, must be in "
                               "range [-4, 4)");
  }
  {
    // Axis is batch dimension, should fail.
    Reset();
    NodeDef node_def = get_split_nodedef(DT_FLOAT, 1);
    AddTestWeights<int32>("axis", {1}, {0});
    AddTestTensor("value", {1, 2, 3});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "TensorRT does not allow manipulation of the "
                               "batch dimension");
  }
  {
    // Value is a weight, should fail.
    Reset();
    NodeDef node_def = get_split_nodedef(DT_FLOAT, 1);
    AddTestWeights<int32>("axis", {1}, {1});
    AddTestWeights<float>("value", {1, 2, 3}, {1, 2, 3, 4, 5, 6});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"value\" for Split must be a tensor");
  }
  {
    // Dim is not evenly divisibly by num_split, should fail.
    Reset();
    NodeDef node_def = get_split_nodedef(DT_FLOAT, 2);
    AddTestWeights<int32>("axis", {1}, {3});
    AddTestTensor("value", {1, 2, 3});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Dimension 3 of size 3 is not evenly divisible by 2");
  }
  {
    // num_split > dim size, should fail.
    Reset();
    NodeDef node_def = get_split_nodedef(DT_FLOAT, 4);
    AddTestWeights<int32>("axis", {1}, {3});
    AddTestTensor("value", {1, 2, 3});
    RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        "Dimension 3 of size 3 is not evenly divisible by 4");
  }

  TestConvertSplit<DT_FLOAT>(this);
  TestConvertSplit<DT_HALF>(this);
  TestConvertSplit<DT_INT32>(this);
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

struct UnpackTestParams {
  std::vector<int> input_shape;
  std::vector<float> input_value;
  int axis;
  int num;
  std::vector<int> expected_output_dims;
  std::vector<std::vector<float>> expected_outputs;
  Status run_status;
};

void TestConvertUnpack(ParameterizedOpConverterTestBase* test,
                       UnpackTestParams& p) {
  test->Reset();
  NodeDef node_def = get_unpack_nodedef(test->get_tf_type(), p.num, p.axis);
  // Create inputs.
  test->AddTestTensor("value", p.input_shape, test->get_tf_type(),
                      p.input_value);

  std::vector<Matcher<std::vector<float>>> matcher_vec;
  std::vector<DataType> datatype_vec;
  std::vector<std::vector<int>> expected_output_dims;

  for (int j = 0; j < p.expected_outputs.size(); ++j) {
    matcher_vec.push_back(ElementsAreArray(p.expected_outputs[j]));
    datatype_vec.push_back(test->get_tf_type());
    expected_output_dims.push_back(p.expected_output_dims);
  }

  test->TestOpConverterMultiOut(/*name=*/"my_unpack",
                                /*node_def=*/node_def,
                                /*expected_output_dims=*/expected_output_dims,
                                /*expected_conversion_status=*/p.run_status,
                                /*expected_runtime_status=*/p.run_status,
                                /*matcher=*/matcher_vec,
                                /*out_tf_type=*/datatype_vec);
}

// TODO: Reactivate when INT32 Segfault fixed
TEST_P(OpConverter_FP32_FP16_INT32_Test, ConvertUnpack) {
  // We need to skip error testing for Dynamic Shape mode, as it is impossible
  // to convert Unpack in Dynamic Shape Mode.
  if (trt_mode_ != TrtTestMode::kDynamicShape) {
    {
      // Value is weights, should fail.
      Reset();
      NodeDef node_def = get_unpack_nodedef(tf_type_, /*num=*/3, /*axis=*/3);
      AddTestWeights<float>("value", {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6});
      RunValidationAndConversion(
          node_def, error::UNIMPLEMENTED,
          "The input \"value\" for Unpack must be a tensor");
    }
    {
      // Axis is out of bounds, should fail.
      Reset();
      NodeDef node_def = get_unpack_nodedef(tf_type_, /*num=*/1, /*axis=*/4);
      AddTestTensor("value", {1, 1, 2, 3});
      RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                                 "Axis value of 4 is out of bounds, must be in "
                                 "range [-4, 4)");
    }
    {
      // Axis is out of bounds (negative), should fail.
      Reset();
      NodeDef node_def = get_unpack_nodedef(tf_type_, /*num=*/1, /*axis=*/-5);
      AddTestTensor("value", {1, 1, 2, 3});
      RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                                 "Axis value of -5 is out of bounds, must be "
                                 "in range [-4, 4)");
    }
    {
      if (trt_mode_ != TrtTestMode::kExplicitBatch) {
        // Axis is batch dimension, should fail.
        Reset();
        NodeDef node_def = get_unpack_nodedef(tf_type_, /*num=*/1, /*axis=*/0);
        AddTestTensor("value", {1, 2, 3});
        RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                                   "TensorRT does not allow manipulation of "
                                   "the batch dimension");
      }
    }
    {
      // Dim size does not match num, should fail.
      Reset();
      NodeDef node_def = get_unpack_nodedef(tf_type_, /*num=*/5, /*axis=*/2);
      AddTestTensor("value", {1, 1, 6});
      RunValidationAndConversion(
          node_def, error::INVALID_ARGUMENT,
          "Dimension 2 has size 6 which is not equal to num of 5");
    }
    {
      // Output would be TF scalar, should fail.
      Reset();
      NodeDef node_def = get_unpack_nodedef(tf_type_, /*num=*/1, /*axis=*/0);
      AddTestTensor(
          "value", {}, tf_type_, {}, {},
          trt_mode_ == TrtTestMode::kImplicitBatch
              ? errors::InvalidArgument(
                    "removing first dim requires explicit batch dimension")
              : Status::OK());
      if (trt_mode_ == TrtTestMode::kImplicitBatch) {
        RunValidationAndConversion(
            node_def, error::INTERNAL,
            "Failed to convert at least one input to a TRT_TensorOrWeights: "
            "Scalar input tensor is not supported since the first dimension is "
            "treated as batch dimension by TRT");
      } else {
        RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                                   "Input \"value\" for Unpack must be rank 2 "
                                   "or greater");
      }
    }
  }

  const std::vector<float> common_input = CreateVectorIota<float>(6);

  Status run_status =
      trt_mode_ == TrtTestMode::kDynamicShape
          ? errors::InvalidArgument(
                "The argument `strided_slice_spec` is "
                "`absl::nullopt` with `dynamic_input_size_indices` non empty.")
          : Status::OK();

  std::vector<UnpackTestParams> params = {
      {/*input_shape=*/{1, 1, 2, 1, 3, 1},
       /*input_value=*/common_input,
       /*axis=*/4,
       /*num=*/3,
       /*expected_output_dims=*/{1, 1, 2, 1, 1},
       /*expected_outputs=*/{{0, 3}, {1, 4}, {2, 5}},
       /*run_status=*/run_status},
      {/*input_shape=*/{1, 1, 2, 1, 3},
       /*input_value=*/common_input,
       /*axis=*/4,
       /*num=*/3,
       /*expected_output_dims=*/{1, 1, 2, 1},
       /*expected_outputs=*/{{0, 3}, {1, 4}, {2, 5}},
       /*run_status=*/run_status},
      {/*input_shape=*/{1, 1, 2, 3},
       /*input_value=*/common_input,
       /*axis=*/1,
       /*num=*/1,
       /*expected_output_dims=*/{1, 2, 3},
       /*expected_outputs=*/{CreateVectorIota<float>(6)},
       /*run_status=*/run_status},
      {/*input_shape=*/{1, 6, 1},
       /*input_value=*/common_input,
       /*axis=*/-2,
       /*num=*/6,
       /*expected_output_dims=*/{1, 1},
       /*expected_outputs=*/{{0}, {1}, {2}, {3}, {4}, {5}},
       /*run_status=*/run_status},
      {/*input_shape=*/{1, 6},
       /*input_value=*/common_input,
       /*axis=*/1,
       /*num=*/6,
       /*expected_output_dims=*/{1},
       /*expected_outputs=*/{{0}, {1}, {2}, {3}, {4}, {5}},
       /*run_status=*/run_status},
  };
  for (auto p : params) {
    TestConvertUnpack(this, p);
  }
}

// Get the NodeDef for Pack.
NodeDef GetPackNodeDef(DataType dtype, int num_inputs, int axis) {
  Scope s = Scope::NewRootScope();
  std::vector<Input> values;
  values.reserve(num_inputs);
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

TEST_P(OpConverter_FP32_FP16_INT32_Test, ConvertPack) {
  struct TestParams {
    std::vector<std::vector<int>> input_shapes;
    std::vector<std::vector<int>> partial_input_shapes;
    std::vector<std::vector<float>> input_values;
    int axis;
    std::vector<int> expected_output_dims;
    std::vector<float> expected_output;
    Status conversion_status;
    Status runtime_status;
    bool input_1_is_weight;
  };

  const std::vector<std::vector<float>> common_input{
      CreateVectorIota<float>(6),
      CreateVectorIota<float>(6, /*start_value=*/6)};
  std::vector<TestParams> params = {
      // Second input is weight, should fail in implicit batch mode
      {/*input_shapes=*/{{1, 2, 3}, {1, 2, 3}},
       /*partial_input_shapes=*/{{}, {}},
       /*input_values=*/common_input,
       /*axis=*/1,
       /*expected_output_dims=*/{1, 2, 2, 3},
       /*expected_output=*/CreateVectorIota<float>(12),
       trt_mode_ == TrtTestMode::kImplicitBatch
           ? Status{error::UNIMPLEMENTED,
                    "The input \"values_1\" for Pack must be a tensor"}
           : Status::OK(),
       /*runtime_status*/ Status::OK(),
       /*weight_input*/ true},
      // Axis is out of bounds, should fail.
      {
          /*input_shapes=*/{{1, 2, 3}, {1, 2, 3}},
          /*partial_input_shapes=*/{{}, {}},
          /*input_values=*/common_input,
          /*axis=*/-5,
          /*expected_output_dims=*/{},
          /*expected_output=*/{},
          Status{error::INVALID_ARGUMENT,
                 "Axis value of -5 is out of bounds, must be in"
                 " range [-4, 4)"},
      },
      // Axis is batch dimension, should fail in implicit batch mode.
      {/*input_shapes=*/{{1, 2, 3}, {1, 2, 3}},
       /*partial_input_shapes=*/{{}, {}},
       /*input_values=*/common_input,
       /*axis=*/-4,
       /*expected_output_dims=*/{2, 1, 2, 3},
       /*expected_output=*/CreateVectorIota<float>(12),
       trt_mode_ == TrtTestMode::kImplicitBatch
           ? Status{error::UNIMPLEMENTED,
                    "TensorRT does not allow manipulation of the batch "
                    "dimension"}
           : Status::OK()},
      // Inconsistent rank, should fail.
      {
          /*input_shapes=*/{{1, 2, 3}, {1, 6}},
          /*partial_input_shapes=*/{{}, {}},
          /*input_values=*/common_input,
          /*axis=*/1,
          /*expected_output_dims=*/{},
          /*expected_output=*/{},
          Status{error::INVALID_ARGUMENT,
                 "Received inputs with inconsistent rank"},
      },
      {
          /*input_shapes=*/{{1, 2, 3}, {1, 2, 3}},
          /*partial_input_shapes=*/{{}, {}},
          /*input_values=*/common_input,
          /*axis=*/1,
          /*expected_output_dims=*/{1, 2, 2, 3},
          /*expected_output=*/CreateVectorIota<float>(12),
      },
      {
          /*input_shapes=*/{{1, 2, 3}, {1, 2, 3}},
          /*partial_input_shapes=*/{{}, {}},
          /*input_values=*/common_input,
          /*axis=*/2,
          /*expected_output_dims=*/{1, 2, 2, 3},
          /*expected_output=*/
          {0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11},
      },
      {
          /*input_shapes=*/{{1, 2, 3}, {1, 2, 3}},
          /*partial_input_shapes=*/{{}, {}},
          /*input_values=*/common_input,
          /*axis=*/3,
          /*expected_output_dims=*/{1, 2, 3, 2},
          /*expected_output=*/
          {0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11},
      },
      {
          /*input_shapes=*/{{1, 2, 3}},
          /*partial_input_shapes=*/{{}},
          /*input_values=*/{CreateVectorIota<float>(6)},
          /*axis=*/1,
          /*expected_output_dims=*/{1, 1, 2, 3},
          /*expected_output=*/CreateVectorIota<float>(6),
      },
      {
          /*input_shapes=*/{{1, 2, 3}},
          /*partial_input_shapes=*/{{}},
          /*input_values=*/{CreateVectorIota<float>(6)},
          /*axis=*/2,
          /*expected_output_dims=*/{1, 2, 1, 3},
          /*expected_output=*/CreateVectorIota<float>(6),
      },
  };
  // Inputs have inconsistent shapes, should fail.
  if (trt_mode_ != TrtTestMode::kDynamicShape) {
    params.push_back(
        TestParams{/*input_shapes=*/{{1, 2, 3}, {1, 3, 2}},
                   /*partial_input_shapes=*/{{}, {}},
                   /*input_values=*/common_input,
                   /*axis=*/1,
                   /*expected_output_dims=*/{},
                   /*expected_output=*/CreateVectorIota<float>(12),
                   Status{error::INVALID_ARGUMENT,
                          "Received inputs with inconsistent shape"}});
  } else {
    // In dynamic shape mode we cannot catch inconsistent shapes at conversion
    // time, only during runtime. But TensorRT does not raise a proper runtime
    // error, instead it aborts the program with the following message:
    //  Assertion failed: t->start.d[i] + t->extent.d[i] <= r.dims.d[i]
    // ../builder/cudnnBuilderGraph.cpp:862
    // Aborting...
    // TODO(tfeher) Add dynamic shapes test once TRT handles shape error
    // decently
  }
  if (trt_mode_ == TrtTestMode::kDynamicShape) {
    // Test with mixed dynamic / static shape input tensors
    params.push_back(
        TestParams{/*input_shapes=*/{{1, 2, 3}, {1, 2, 3}},
                   /*partial_input_shapes=*/{{-1, -1, -1}, {1, 2, 3}},
                   /*input_values=*/common_input,
                   /*axis=*/2,
                   /*expected_output_dims=*/{1, 2, 2, 3},
                   /*expected_output=*/
                   {0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11}});
  }
  for (auto p : params) {
    Reset();
    const int num_inputs = p.input_shapes.size();
    EXPECT_EQ(num_inputs, p.input_values.size());

    NodeDef node_def = GetPackNodeDef(tf_type_, num_inputs, p.axis);
    // Create inputs.
    for (int j = 0; j < num_inputs; ++j) {
      if (j == 1 && p.input_1_is_weight) {
        AddTestWeights(StrCat("values_", j), p.input_shapes[j],
                       p.input_values[j], tf_type_);
      } else {
        AddTestTensor(StrCat("values_", j), p.input_shapes[j], tf_type_,
                      p.input_values[j], p.partial_input_shapes[j]);
      }
    }
    TestOpConverter("my_pack", node_def, p.expected_output_dims,
                    p.conversion_status, p.runtime_status,
                    ElementsAreArray(p.expected_output));
  }
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

struct ArgMinMaxTestParams {
  std::vector<int> input_shape;
  std::vector<float> input_value;
  int axis;
  std::vector<int> expected_output_dims;
  std::vector<int> expected_argmax_output;
  std::vector<int> expected_argmin_output;
  Status status;
};

template <typename OpType>
void TestConvertArgMinMax(ParameterizedOpConverterTestBase* test,
                          DataType _tf_type, ArgMinMaxTestParams& p) {
  test->Reset();

  NodeDef node_def = GetArgMinMaxNodeDef<OpType>(_tf_type,
                                                 /*output_dtype=*/DT_INT32);

  std::vector<int> expected_out;
  if (node_def.op() == "ArgMax") {
    expected_out = p.expected_argmax_output;
  } else if (node_def.op() == "ArgMin") {
    expected_out = p.expected_argmin_output;
  } else {
    ASSERT_TRUE(false);
  }

  test->AddTestTensor("input", p.input_shape, _tf_type, p.input_value);
  test->AddTestWeights("dimension", {1}, {p.axis}, DT_INT32);

  test->TestOpConverter("my_arg", node_def, p.expected_output_dims,
                        /*expected_conversion_status=*/p.status,
                        /*expected_runtime_status=*/Status::OK(),
                        /*matcher=*/ElementsAreArray(expected_out), {DT_INT32});
}

TEST_P(OpConverter_FP32_FP16_Test, ConvertArgMinMax) {
  {
    // Dimension is a tensor, should fail.
    Reset();
    NodeDef node_def =
        GetArgMinMaxNodeDef<ops::ArgMax>(tf_type_,
                                         /*output_dtype=*/DT_INT32);
    AddTestTensor("input", {1, 2, 3});
    AddTestTensor("dimension", {1});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"dimension\" for ArgMax must be a constant");
  }
  {
    // Output type is INT64, should fail.
    Reset();
    NodeDef node_def =
        GetArgMinMaxNodeDef<ops::ArgMax>(tf_type_,
                                         /*output_dtype=*/DT_INT64);
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights("dimension", {1}, {3}, DT_INT32);
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "Output type int64 is not supported");
  }

  const std::vector<float> common_input = CreateVectorIota<float>(6);
  std::vector<ArgMinMaxTestParams> params = {
      {/*input_shape=*/{2, 3},
       /*input_value=*/common_input,
       /*axis=*/0,
       /*expected_output_dims=*/{3},
       /*expected_argmax_output=*/{1, 1, 1},
       /*expected_argmin_output=*/{0, 0, 0},
       trt_mode_ == TrtTestMode::kImplicitBatch
           ? errors::Unimplemented("TensorRT does not allow manipulation of "
                                   "the batch dimension")
           : Status::OK()},
      {
          /*input_shape=*/{1, 6},
          /*input_value=*/common_input,
          /*axis=*/1,
          /*expected_output_dims=*/{1},
          /*expected_argmax_output=*/{5},
          /*expected_argmin_output=*/{0},
      },
      {
          /*input_shape=*/{1, 10},
          /*input_value=*/
          {-5.0f, 3.0f, 5.0f, 1.0f, 6.0f, -9.0f, 7.0f, 1.0f, 0.0f, -1.0f},
          /*axis=*/-1,
          /*expected_output_dims=*/{1},
          /*expected_argmax_output=*/{6},
          /*expected_argmin_output=*/{5},
      },
      {
          /*input_shape=*/{1, 2, 3},
          /*input_value=*/common_input,
          /*axis=*/2,
          /*expected_output_dims=*/{1, 2},
          /*expected_argmax_output=*/{2, 2},
          /*expected_argmin_output=*/{0, 0},
      },
      {
          /*input_shape=*/{1, 2, 3},
          /*input_value=*/common_input,
          /*axis=*/-2,
          /*expected_output_dims=*/{1, 3},
          /*expected_argmax_output=*/{1, 1, 1},
          /*expected_argmin_output=*/{0, 0, 0},
      },
      {
          /*input_shape=*/{1, 2, 1, 3},
          /*input_value=*/common_input,
          /*axis=*/3,
          /*expected_output_dims=*/{1, 2, 1},
          /*expected_argmax_output=*/{2, 2},
          /*expected_argmin_output=*/{0, 0},
      },
      {
          /*input_shape=*/{1, 2, 1, 3},
          /*input_value=*/common_input,
          /*axis=*/-3,
          /*expected_output_dims=*/{1, 1, 3},
          /*expected_argmax_output=*/{1, 1, 1},
          /*expected_argmin_output=*/{0, 0, 0},
      },
      {/*input_shape=*/{1, 2, 1, 1, 3},
       /*input_value=*/common_input,
       /*axis=*/4,
       /*expected_output_dims=*/{1, 2, 1, 1},
       /*expected_argmax_output=*/{2, 2},
       /*expected_argmin_output=*/{0, 0},
#if !IS_TRT_VERSION_GE(7, 0, 0, 11)
       errors::Unimplemented("op is not able to support tensors with 4+"
                             " dimensions (excluding batch size)")
#else
       Status::OK()
#endif
      },
      {/*input_shape=*/{1, 2, 1, 1, 3},
       /*input_value=*/common_input,
       /*axis=*/-4,
       /*expected_output_dims=*/{1, 1, 1, 3},
       /*expected_argmax_output=*/{1, 1, 1},
       /*expected_argmin_output=*/{0, 0, 0},
#if !IS_TRT_VERSION_GE(7, 0, 0, 11)
       errors::Unimplemented("op is not able to support tensors with 4+"
                             " dimensions (excluding batch size)")
#else
       Status::OK()
#endif
      },
  };

  for (auto p : params) {
    TestConvertArgMinMax<ops::ArgMin>(this, tf_type_, p);
    TestConvertArgMinMax<ops::ArgMax>(this, tf_type_, p);
  }
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

struct DepthSpaceShuffleTestParams {
  std::vector<int> input_dims;
  std::vector<int> input_value;
  int block_size;
  string data_format;
  std::vector<int> expected_output_dims;
  std::vector<int> expected_output;
};

template <typename OpType>
void TestConvertDepthSpaceShuffle(
    ParameterizedOpConverterTestBase* test,
    const std::vector<DepthSpaceShuffleTestParams>& params) {
  Status status = Status::OK();

  {
    // Input is a weight, should fail.
    test->Reset();
    NodeDef node_def = GetDepthSpaceShuffleNodeDef<ops::DepthToSpace>(
        test->get_tf_type(), 2, "NCHW");
    test->AddTestWeights<float>("input", {1, 4, 1, 1}, {1, 2, 3, 4});
    test->RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        StrCat("The input \"input\" for ", node_def.op(), " must be a tensor"));
  }
  {
    // Input rank != 4
    test->Reset();
    NodeDef node_def = GetDepthSpaceShuffleNodeDef<ops::DepthToSpace>(
        test->get_tf_type(), 2, "NCHW");
    test->AddTestTensor("input", {1, 16, 32});
    test->RunValidationAndConversion(
        node_def, error::INVALID_ARGUMENT,
        StrCat("The input to ", node_def.op(), " must be rank 4"));
  }
  {
    // Unsupported format, should fail.
    test->Reset();
    NodeDef node_def = GetDepthSpaceShuffleNodeDef<ops::DepthToSpace>(
        test->get_tf_type(), 2, "NCHW_VECT_C");
    test->AddTestTensor("input", {1, 16, 32, 32});
    test->RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "Data format NCHW_VECT_C is not supported");
  }
  if (test->get_trt_mode() != TrtTestMode::kDynamicShape) {
    // In dynamic shape mode, we cannot check input dimension values at
    // conversion time therefore we cannot confirm block_size vs input dim
    // consistency. We rely on the user to provide a valid TF graph. Otherwise
    // TRT will fail with a runtime error.
    if (std::is_same<OpType, ops::DepthToSpace>::value) {
      // Channels not divisible by block_size, should fail.
      test->Reset();
      NodeDef node_def = GetDepthSpaceShuffleNodeDef<ops::DepthToSpace>(
          test->get_tf_type(), 3, "NCHW");
      test->AddTestTensor("input", {1, 16, 32, 32});
      test->RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                                       "Number of channels must be divisible by"
                                       " block_size*block_size");
    } else {
      {  // Width not divisible by block_size, should fail.
        test->Reset();
        NodeDef node_def = GetDepthSpaceShuffleNodeDef<ops::SpaceToDepth>(
            test->get_tf_type(), 3, "NCHW");
        test->AddTestTensor("input", {1, 16, 9, 32});
        test->RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                                         "Width and height must be divisible by"
                                         " block_size");
      }
      {
        // Height not divisible by block_size, should fail.
        test->Reset();
        NodeDef node_def = GetDepthSpaceShuffleNodeDef<ops::SpaceToDepth>(
            test->get_tf_type(), 3, "NCHW");
        test->AddTestTensor("input", {1, 16, 32, 9});
        test->RunValidationAndConversion(node_def, error::INVALID_ARGUMENT,
                                         "Width and height must be divisible by"
                                         " block_size");
      }
    }
  }

  for (auto p : params) {
    test->Reset();
    NodeDef node_def = GetDepthSpaceShuffleNodeDef<OpType>(
        test->get_tf_type(), p.block_size, p.data_format);
    test->AddTestTensor("input", p.input_dims, p.input_value);
    test->TestOpConverter("my_shuffle", node_def, p.expected_output_dims,
                          status, Status::OK(),
                          ElementsAreArray(p.expected_output));
  }
}

TEST_P(OpConverter_FP32_FP16_INT32_Test, ConvertDepthToSpace) {
  const std::vector<int> common_input = CreateVectorIota<int>(16);
  std::vector<DepthSpaceShuffleTestParams> params = {
      {
          /*input_shape=*/{1, 4, 2, 2},
          /*input_value=*/common_input,
          /*block_size=*/2,
          /*data_format=*/"NCHW",
          /*expected_output_dims=*/{1, 1, 4, 4},
          /*expected_output=*/
          {0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15},
      },
      {
          /*input_shape=*/{1, 2, 2, 4},
          /*input_value=*/common_input,
          /*block_size=*/2,
          /*data_format=*/"NHWC",
          /*expected_output_dims=*/{1, 4, 4, 1},
          /*expected_output=*/
          {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15},
      },
      {
          /*input_shape=*/{1, 16, 1, 1},
          /*input_value=*/common_input,
          /*block_size=*/4,
          /*data_format=*/"NCHW",
          /*expected_output_dims=*/{1, 1, 4, 4},
          /*expected_output=*/CreateVectorIota<int>(16),
      },
      {
          /*input_shape=*/{1, 2, 2, 8},
          /*input_value=*/CreateVectorIota<int>(32),
          /*block_size=*/2,
          /*data_format=*/"NHWC",
          /*expected_output_dims=*/{1, 4, 4, 2},
          /*expected_output=*/{0,  1,  2,  3,  8,  9,  10, 11, 4,  5,  6,
                               7,  12, 13, 14, 15, 16, 17, 18, 19, 24, 25,
                               26, 27, 20, 21, 22, 23, 28, 29, 30, 31},
      }};

  TestConvertDepthSpaceShuffle<ops::DepthToSpace>(this, params);
}

TEST_P(OpConverter_FP32_FP16_INT32_Test, ConvertSpaceToDepth) {
  const std::vector<int> common_input = CreateVectorIota<int>(16);
  std::vector<DepthSpaceShuffleTestParams> params = {
      {
          /*input_shape=*/{1, 1, 4, 4},
          /*input_value=*/common_input,
          /*block_size=*/2,
          /*data_format=*/"NCHW",
          /*expected_output_dims=*/{1, 4, 2, 2},
          /*expected_output=*/
          {0, 2, 8, 10, 1, 3, 9, 11, 4, 6, 12, 14, 5, 7, 13, 15},
      },
      {
          /*input_shape=*/{1, 4, 4, 1},
          /*input_value=*/common_input,
          /*block_size=*/2,
          /*data_format=*/"NHWC",
          /*expected_output_dims=*/{1, 2, 2, 4},
          /*expected_output=*/
          {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15},
      },
      {
          /*input_shape=*/{1, 1, 4, 4},
          /*input_value=*/common_input,
          /*block_size=*/4,
          /*data_format=*/"NCHW",
          /*expected_output_dims=*/{1, 16, 1, 1},
          /*expected_output=*/CreateVectorIota<int>(16),
      },
      {
          /*input_shape=*/{1, 4, 4, 2},
          /*input_value=*/CreateVectorIota<int>(32),
          /*block_size=*/2,
          /*data_format=*/"NHWC",
          /*expected_output_dims=*/{1, 2, 2, 8},
          /*expected_output=*/{0,  1,  2,  3,  8,  9,  10, 11, 4,  5,  6,
                               7,  12, 13, 14, 15, 16, 17, 18, 19, 24, 25,
                               26, 27, 20, 21, 22, 23, 28, 29, 30, 31},
      },
  };
  TestConvertDepthSpaceShuffle<ops::SpaceToDepth>(this, params);
}

TEST_P(OpConverter_FP32_FP16_Test, ConvertClipByValue) {
  Scope s = Scope::NewRootScope();
  auto t = ops::Placeholder(s.WithOpName("t"), tf_type_);
  auto clip_value_min =
      ops::Placeholder(s.WithOpName("clip_value_min"), tf_type_);
  auto clip_value_max =
      ops::Placeholder(s.WithOpName("clip_value_max"), tf_type_);
  auto clip = ops::ClipByValue(s.WithOpName("my_clip"), t, clip_value_min,
                               clip_value_max);
  const NodeDef& node_def = clip.operation.node()->def();

  nvinfer1::DataType trt_type_;
  TF_ASSERT_OK(TfTypeToTrtType(tf_type_, &trt_type_));

  {
    // Input is a weight, should fail.
    Reset();
    AddTestWeights("t", {1, 2, 3}, {1, 2, 3, 4, 5, 6}, tf_type_);
    AddTestWeights("clip_value_min", {1}, {1}, tf_type_);
    AddTestWeights("clip_value_max", {1}, {5}, tf_type_);
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "The input \"t\" for ClipByValue must be a "
                               "tensor");
  }
  {
    // Clip min is a tensor, should fail.
    Reset();
    AddTestTensor("t", {1, 2, 3});
    AddTestTensor("clip_value_min", {1});
    AddTestWeights("clip_value_max", {1}, {1}, tf_type_);
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "The input \"clip_value_min\" for ClipByValue "
                               "must be a constant");
  }
  {
    // Clip max is a tensor, should fail.
    Reset();
    AddTestTensor("t", {1, 2, 3});
    AddTestWeights("clip_value_min", {1}, {1}, tf_type_);
    AddTestTensor("clip_value_max", {1});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "The input \"clip_value_max\" for ClipByValue "
                               "must be a constant");
  }

  struct TestParams {
    std::vector<int> dims;
    int clip_value_min;
    int clip_value_max;
    std::vector<float> expected_output;
  };

  const std::vector<float> common_input = CreateVectorIota<float>(6);

  std::vector<TestParams> params = {{
                                        /*dims=*/{6},
                                        /*clip_value_min=*/2,
                                        /*clip_value_max=*/4,
                                        /*expected_output=*/{2, 2, 2, 3, 4, 4},
                                    },
                                    {
                                        /*dims=*/{1, 6},
                                        /*clip_value_min=*/2,
                                        /*clip_value_max=*/4,
                                        /*expected_output=*/{2, 2, 2, 3, 4, 4},
                                    },
                                    {
                                        /*dims=*/{1, 2, 3},
                                        /*clip_value_min=*/2,
                                        /*clip_value_max=*/4,
                                        /*expected_output=*/{2, 2, 2, 3, 4, 4},
                                    },
                                    {
                                        /*dims=*/{1, 2, 3, 1},
                                        /*clip_value_min=*/2,
                                        /*clip_value_max=*/4,
                                        /*expected_output=*/{2, 2, 2, 3, 4, 4},
                                    },
                                    {
                                        /*dims=*/{1, 1, 3, 1, 2},
                                        /*clip_value_min=*/2,
                                        /*clip_value_max=*/4,
                                        /*expected_output=*/{2, 2, 2, 3, 4, 4},
                                    },
                                    {
                                        /*dims=*/{1, 1, 3, 1, 2, 1},
                                        /*clip_value_min=*/2,
                                        /*clip_value_max=*/4,
                                        /*expected_output=*/{2, 2, 2, 3, 4, 4},
                                    },
                                    {
                                        /*dims=*/{2, 1, 3},
                                        /*clip_value_min=*/-1,
                                        /*clip_value_max=*/8,
                                        /*expected_output=*/common_input,
                                    }};

  for (auto p : params) {
    Reset();

    AddTestTensor("t", p.dims, tf_type_, common_input);
    AddTestWeights("clip_value_min", {1}, {p.clip_value_min}, tf_type_);
    AddTestWeights("clip_value_max", {1}, {p.clip_value_max}, tf_type_);

    TestOpConverter("my_clip", node_def, p.dims,
                    /*expected_conversion_status=*/Status::OK(),
                    /*expected_runtime_status=*/Status::OK(),
                    /*matcher=*/ElementsAreArray(p.expected_output));
  }
}

// Get the NodeDef for SquaredDifference.
NodeDef GetSquaredDifferenceNodeDef(DataType dtype) {
  Scope s = Scope::NewRootScope();
  auto x = ops::Placeholder(s.WithOpName("x"), dtype);
  auto y = ops::Placeholder(s.WithOpName("y"), dtype);
  auto squared_diff =
      ops::SquaredDifference(s.WithOpName("my_squared_diff"), x, y);
  return squared_diff.operation.node()->def();
}

TEST_P(OpConverter_FP32_FP16_Test, ConvertSquaredDifference) {
  {
    // Input is a weight, should fail.
    Reset();
    NodeDef node_def = GetSquaredDifferenceNodeDef(tf_type_);
    AddTestWeights<float>("x", {1, 2, 3}, {1, 2, 3, 4, 5, 6});
    AddTestTensor("y", {1, 1, 2, 3});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "The input \"x\" for SquaredDifference must be "
                               "a tensor");
  }

  struct TestParams {
    std::vector<int> dims_x;
    std::vector<int> dims_y;
    std::vector<float> value_x;
    std::vector<float> value_y;
    std::vector<int> expected_output_dims;
    std::vector<float> expected_output;
    Status status;
    Status runtime_status;
  };

  const std::vector<float> common_input = CreateVectorIota<float>(6);
  std::vector<TestParams> params = {
      {/*dims_x=*/{1, 2, 3},
       /*dims_y=*/{1, 7, 5},
       /*value_x=*/common_input,
       /*value_y=*/std::vector<float>(7 * 5, 0),
       /*expected_output_dims=*/{1, 1, 2, 3},
       /*expected_output=*/common_input,
       trt_mode_ == TrtTestMode::kDynamicShape
           ? Status::OK()
           : errors::InvalidArgument("Infeasible broadcast scheme"),
       errors::Internal(
           "Binding index out of range. This can happen if profile is not set, "
           "or the network is invalid for the current profile.")},
      {
          /*dims_x=*/{1, 1, 2, 3},
          /*dims_y=*/{1, 1, 2, 3},
          /*value_x=*/common_input,
          /*value_y=*/{0, -1, 3, 0, 10, -7},
          /*expected_output_dims=*/{1, 1, 2, 3},
          /*expected_output=*/{0, 4, 1, 9, 36, 144},
      },
      {
          /*dims_x=*/{1, 1, 2, 3},
          /*dims_y=*/{1, 1, 1, 3},
          /*value_x=*/common_input,
          /*value_y=*/{0, 1, 2},
          /*expected_output_dims=*/{1, 1, 2, 3},
          /*expected_output=*/{0, 0, 0, 9, 9, 9},
      },
  };

  for (auto p : params) {
    Reset();
    NodeDef node_def = GetSquaredDifferenceNodeDef(tf_type_);
    AddTestTensor("x", p.dims_x, p.value_x);
    AddTestTensor("y", p.dims_y, p.value_y);
    TestOpConverter("my_squared_diff", node_def, p.expected_output_dims,
                    p.status, p.runtime_status,
                    ElementsAreArray(p.expected_output));
  }
}

template <typename OpType>
NodeDef MakeResizeNodeDef(DataType dtype, bool align_corners) {
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), dtype);
  auto size = ops::Placeholder(s.WithOpName("size"), DT_INT32);
  auto attrs = typename OpType::Attrs().AlignCorners(align_corners);
  auto resize = OpType(s.WithOpName("my_resize"), input, size, attrs);
  return resize.operation.node()->def();
}

struct ResizeTestParams {
  std::vector<int> input_dims;
  std::vector<int> output_resize_dims;
  std::vector<float> input_value;
  bool size_as_tensor;
  bool align_corners;
  std::vector<int> expected_output_dims;
  std::vector<float> expected_nearest_output_values;
  std::vector<float> expected_bilinear_output_values;
  Status status;
};

template <typename OpType>
void TestConvertResize(ParameterizedOpConverterTestBase* test,
                       ResizeTestParams& p) {
  test->Reset();
  // Create resize node.
  NodeDef node_def =
      MakeResizeNodeDef<OpType>(test->get_tf_type(), p.align_corners);

  test->AddTestTensor("input", p.input_dims, test->get_tf_type(),
                      p.input_value);
  // Create output size.
  if (p.size_as_tensor) {
    std::vector<int32> size_dims{2};
    std::vector<int32> size_values{p.output_resize_dims};
    test->AddTestTensor("size", size_dims, DT_INT32, size_values, size_dims);
  } else {
    test->AddTestWeights("size", {2}, p.output_resize_dims, DT_INT32);
  }

  std::vector<float> expected_out;

  if (node_def.op() == "ResizeBilinear") {
    expected_out = p.expected_bilinear_output_values;
  } else if (node_def.op() == "ResizeNearestNeighbor") {
    expected_out = p.expected_nearest_output_values;
  } else {
    ASSERT_TRUE(false);
  }

  test->TestOpConverter("my_resize", node_def, p.expected_output_dims,
                        /*expected_conversion_status=*/p.status,
                        /*expected_runtime_status=*/p.status,
                        /*matcher=*/ElementsAreArray(expected_out),
                        /*out_tf_types=*/{DT_FLOAT});
}

TEST_P(OpConverter_FP32_FP16_Test, ConvertResize) {
  {
    // First input is weight, should fail.
    Reset();
    NodeDef node_def = MakeResizeNodeDef<ops::ResizeBilinear>(tf_type_,
                                                              /*align_corners=*/
                                                              true);
    AddTestWeights<float>("input", {1, 2}, {1, 2});
    AddTestWeights<int>("size", {1, 2}, {1, 2});
    RunValidationAndConversion(
        node_def, error::UNIMPLEMENTED,
        "The input \"input\" for ResizeBilinear must be a "
        "tensor");
  }

  std::vector<ResizeTestParams> params{
      {/*input_dims=*/{1, 1, 2, 1},    // N, H, W, C
       /*output_resize_dims=*/{2, 3},  // H_out, W_out
       /*input_values=*/{2.0f, -1.0f},
       /*size_as_tensor=*/false,
       /*align_corners=*/false,
       /*expected_output_dims=*/{1, 2, 3, 1},  // N, H, W, C
       /*expected_nearest_output_values=*/
       {2.0f, 2.0f, -1.0f, 2.0f, 2.0f, -1.0f},
       /*expected_bilinear_output_values=*/
       {2.0f, 0.f, -1.0f, 2.0f, 0.f, -1.0f},
       /*status=*/Status::OK()},
      {/*input_dims=*/{1, 1, 2, 1},    // N, H, W, C
       /*output_resize_dims=*/{2, 3},  // H_out, W_out
       /*input_values=*/{2.0f, -1.0f},
       /*size_as_tensor=*/false,
       /*align_corners=*/true,
       /*expected_output_dims=*/{1, 2, 3, 1},  // N, H, W, C
       /*expected_nearest_output_values=*/
       {2.0f, 2.0f, -1.0f, 2.0f, 2.0f, -1.0f},
       /*expected_bilinear_output_values=*/
       {2.0f, 0.5f, -1.0f, 2.0f, 0.5f, -1.0f},
       /*status=*/Status::OK()}};

  if (trt_mode_ != TrtTestMode::kImplicitBatch) {
    // Size as a tensor is not supported in implicit batch mode.
    params.push_back({/*input_dims=*/{1, 1, 2, 1},    // N, H, W, C
                      /*output_resize_dims=*/{2, 3},  // H_out, W_out
                      /*input_values=*/{2.0f, -1.0f},
                      /*size_as_tensor=*/true,
                      /*align_corners=*/true,
                      /*expected_output_dims=*/{1, 2, 3, 1},  // N, H, W, C
                      /*expected_nearest_output_values=*/
                      {2.0f, 2.0f, -1.0f, 2.0f, 2.0f, -1.0f},
                      /*expected_bilinear_output_values=*/
                      {2.0f, 0.5f, -1.0f, 2.0f, 0.5f, -1.0f},
                      /*status=*/Status::OK()});
  }

  for (auto p : params) {
    TestConvertResize<ops::ResizeNearestNeighbor>(this, p);

// This use case is not supported as of TRT version 7.1
#if IS_TRT_VERSION_GE(7, 1, 0, 0)
    if (!p.align_corners) {
      p.status = errors::InvalidArgument(
          "Cannot Convert Bilinear Resize when align_corners=False");
    }
#endif

    TestConvertResize<ops::ResizeBilinear>(this, p);
  }
}

NodeDef MakePadNodeDef(std::string name, DataType dtype) {
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), dtype);
  auto padding = ops::Placeholder(s.WithOpName("padding"), DT_INT32);
  auto pad = ops::Pad(s.WithOpName(name), input, padding);
  return pad.operation.node()->def();
}

struct PadTestParams {
  std::vector<int> input_dims;
  std::vector<int> pad_dims;
  std::vector<int> pad_values;
  std::vector<float> input_values;
  std::vector<int> expected_output_dims;
  std::vector<float> expected_output_values;
  Status status;
};

TEST_P(OpConverter_FP32_FP16_Test, ConvertPad) {
  {
    // First input is weight, should fail.
    Reset();
    NodeDef node_def = MakePadNodeDef("my_pad", tf_type_);
    AddTestWeights("input", {1, 2}, {1, 2}, tf_type_);
    AddTestWeights<int>("padding", {1, 2}, {1, 2});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "The input \"tensor\" for Pad must be a "
                               "tensor");
  }
  {
    // padding is a tensor, should fail.
    Reset();
    NodeDef node_def = MakePadNodeDef("my_pad", tf_type_);
    AddTestTensor("input", {1, 2});
    AddTestTensor("padding", {1, 2});
    RunValidationAndConversion(node_def, error::UNIMPLEMENTED,
                               "The input \"paddings\" for Pad must be a "
                               "constant");
  }
  {
    // Make sure that ranges are inferred across a Pad.
    Reset();
    NodeDef node_def = MakePadNodeDef("my_pad", tf_type_);
    AddTestTensor("input", {1, 1, 2, 1});
    AddTestWeights<int>("padding", {4, 2}, {0, 0, 1, 0, 0, 1, 0, 0});
    TRT_TensorOrWeights input;
    TRT_TensorOrWeights output;
    RunValidationAndConversion(node_def);
    TF_EXPECT_OK(GetTensorOrWeights("input", &input));
    TF_EXPECT_OK(GetTensorOrWeights("my_pad", &output));
    ITensorProxyPtr input_tensor = input.tensor();
    converter_->ProvideQuantizationRange(&input_tensor, -5.0f, 5.0f);
    auto ranges = quantization_ranges();
    EXPECT_EQ(5.0f, ranges[input.tensor()->trt_tensor()]);
  }

  std::vector<PadTestParams> params{
      // 1 padding dim
      {
          /*input_dims=*/{1, 1, 3, 2},  // N, H, W, C
          /*pad_dims=*/{4, 2},          // #dims, {pad_before, pad_after}
          /*pad_values*/ {0, 0, 0, 0, 0, 1, 0, 0},
          /*input_values=*/{1, 2, 3, 4, 5, 6},
          /*expected_output_dims=*/{1, 1, 4, 2},  // N, H, W, C
          /*expected_output_values=*/
          {1, 2, 3, 4, 5, 6, 0, 0},
      },
      {
          /*input_dims=*/{1, 1, 3, 2},  // N, H, W, C
          /*pad_dims=*/{4, 2},          // #dims, {pad_before, pad_after}
          /*pad_values*/ {0, 0, 0, 0, 0, 0, 0, 1},
          /*input_values=*/{1, 2, 3, 4, 5, 6},
          /*expected_output_dims=*/{1, 1, 3, 3},  // N, H, W, C
          /*expected_output_values=*/
          {1, 2, 0, 3, 4, 0, 5, 6, 0},
      },
      {
          /*input_dims=*/{1, 1, 3, 2},  // N, H, W, C
          /*pad_dims=*/{4, 2},          // #dims, {pad_before, pad_after}
          /*pad_values*/ {0, 0, 1, 0, 0, 0, 0, 0},
          /*input_values=*/{1, 2, 3, 4, 5, 6},
          /*expected_output_dims=*/{1, 2, 3, 2},  // N, H, W, C
          /*expected_output_values=*/
          {0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6},
      },
      // 2 padding dims
      {
          /*input_dims=*/{1, 1, 2, 1},  // N, H, W, C
          /*pad_dims=*/{4, 2},          // #dims, {pad_before, pad_after}
          /*pad_values*/ {0, 0, 1, 0, 0, 1, 0, 0},
          /*input_values=*/{2.0f, -1.0f},
          /*expected_output_dims=*/{1, 2, 3, 1},  // N, H, W, C
          /*expected_output_values=*/
          {0.0, 0.0, 0.0, 2.0f, -1.0f, 0.0},
      },
      PadTestParams{
          /*input_dims=*/{1, 1, 2, 2},  // N, H, W, C
          /*pad_dims=*/{4, 2},          // #dims, {pad_before, pad_after}
          /*pad_values*/ {0, 0, 1, 0, 0, 1, 0, 0},
          /*input_values=*/{2, -1, 3., 4},
          /*expected_output_dims=*/{1, 2, 3, 2},  // N, H, W, C
          /*expected_output_values=*/
          {0, 0, 0, 0, 0, 0, 2, -1, 3, 4, 0, 0},
      },
      PadTestParams{
          /*input_dims=*/{1, 1, 2, 1, 2},  // N, C, H, W, D
          /*pad_dims=*/{5, 2},             // #dims, {pad_before, pad_after}
          /*pad_values*/ {0, 0, 1, 0, 0, 1, 0, 0, 0, 0},
          /*input_values=*/{2, -1, 3., 4},
          /*expected_output_dims=*/{1, 2, 3, 1, 2},  // N, H, W, C
          /*expected_output_values=*/
          {0, 0, 0, 0, 0, 0, 2, -1, 3, 4, 0, 0},
      },
      PadTestParams{
          /*input_dims=*/{1, 1, 2, 1, 2},  // N, C, H, W, D
          /*pad_dims=*/{5, 2},             // #dims, {pad_before, pad_after}
          /*pad_values*/ {0, 0, 0, 1, 0, 0, 1, 1, 0, 0},
          /*input_values=*/{2, -1, 3., 4},
          /*expected_output_dims=*/{1, 2, 2, 3, 2},  // N, H, W, C
          /*expected_output_values=*/
          {0., 0., 2., -1., 0., 0., 0., 0., 3., 4., 0., 0.,
           0., 0., 0., 0.,  0., 0., 0., 0., 0., 0., 0., 0},
      },
      PadTestParams{
          /*input_dims=*/{1, 1, 2, 1},  // N, H, W, C
          /*pad_dims=*/{4, 2},          // #dims, {pad_before, pad_after}
          /*pad_values*/ {1, 0, 0, 0, 0, 1, 0, 0},
          /*input_values=*/{2.0f, -1.0f},
          /*expected_output_dims=*/{2, 1, 3, 1},  // N, H, W, C
          /*expected_output_values=*/{0.0, 0.0, 0.0, 2.0f, -1.0f, 0.0},
          trt_mode_ == TrtTestMode::kImplicitBatch
              ? errors::InvalidArgument("Padding layer does not support "
                                        "padding on batch dimension")
              : Status::OK()},
      PadTestParams{
          /*input_dims=*/{1, 1, 2, 1},  // N, H, W, C
          /*pad_dims=*/{4, 2},          // #dims, {pad_before, pad_after}
          /*pad_values*/ {0, 0, 1, 0, 0, 1, 1, 1},
          /*input_values=*/{2.0f, -1.0f},
          /*expected_output_dims=*/{},  // N, H, W, C
          /*expected_output_values=*/{},
          errors::InvalidArgument("Padding layer does not support padding on "
                                  "> 2")},
      PadTestParams{
          /*input_dims=*/{1, 2, 2},  // N, H, W
          /*pad_dims=*/{3, 2},       // #dims, {pad_before, pad_after}
          /*pad_values*/ {0, 0, 1, 0, 0, 1},
          /*input_values=*/{2, -1, 3., 4},
          /*expected_output_dims=*/{1, 3, 3},  // N, H, W, C
          /*expected_output_values=*/
          {0., 0., 0., 2., -1., 0., 3., 4., 0.},
          errors::InvalidArgument("Convertpad requires at least 4D input")}};

  for (auto p : params) {
    Reset();
    // Create pad node.
    NodeDef node_def = MakePadNodeDef("my_pad", tf_type_);
    // Create input tensor.
    AddTestTensor("input", p.input_dims, p.input_values);
    // Create output size.
    AddTestWeights<int32>("padding", p.pad_dims, p.pad_values);
    TestOpConverter("my_pad", node_def, p.expected_output_dims, p.status,
                    p.status, ElementsAreArray(p.expected_output_values));
  }
}
}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
