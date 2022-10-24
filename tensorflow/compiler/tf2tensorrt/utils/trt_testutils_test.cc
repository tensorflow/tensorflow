/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "tensorflow/compiler/tf2tensorrt/utils/trt_testutils.h"

#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

using ::testing::AllOf;
using ::testing::AnyOf;
using ::testing::Eq;
using ::testing::Not;

TEST(TrtDimsMatcher, ParameterizedMatchers) {
  EXPECT_THAT(nvinfer1::Dims4(1, 2, 3, 4), DimsAreArray({1, 2, 3, 4}));
  // Check empty dims.
  EXPECT_THAT(nvinfer1::Dims{}, Not(DimsAreArray({1, 2})));
  std::vector<int> empty_dims;
  EXPECT_THAT(nvinfer1::Dims{}, DimsAreArray(empty_dims));
  // Check mismatching values.
  EXPECT_THAT(nvinfer1::Dims4(1, 2, 3, 4), Not(DimsAreArray({1, 2, 3, 5})));
  // Check mismatching number of arguments.
  EXPECT_THAT(nvinfer1::Dims4(1, 2, 3, 4), Not(DimsAreArray({1, 2, 5})));
}

TEST(TrtDimsMatcher, EqualityMatcher) {
  EXPECT_THAT(nvinfer1::Dims4(1, 2, 3, 4), Eq(nvinfer1::Dims4(1, 2, 3, 4)));
  // Check empty dims.
  EXPECT_THAT(nvinfer1::Dims{}, Eq(nvinfer1::Dims()));
  // Check empty Dims is not equal to DimsHW, since their sizes differ.
  EXPECT_THAT(nvinfer1::Dims{}, Not(Eq(nvinfer1::DimsHW())));
  // Check mismatching values.
  EXPECT_THAT(nvinfer1::Dims4(1, 2, 3, 4),
              Not(Eq(nvinfer1::Dims4(1, 2, 3, 3))));
  // Check mismatching number of arguments.
  EXPECT_THAT(nvinfer1::Dims4(1, 2, 3, 4), Not(Eq(nvinfer1::Dims2(1, 2))));
}

TEST(INetworkDefinitionMatchers, CorrectlyMatch) {
  Logger& logger = *Logger::GetLogger();
  TrtUniquePtrType<nvinfer1::IBuilder> builder(
      nvinfer1::createInferBuilder(logger));
  TrtUniquePtrType<nvinfer1::INetworkDefinition> network(
      builder->createNetworkV2(0L));

  // Empty network checks.
  EXPECT_THAT(network.get(), AllOf(Not(LayerNamesAreArray({"some layer"})),
                                   LayerNamesNonEmpty()));

  // Add the input and FC layers.
  nvinfer1::Weights weights;
  weights.type = nvinfer1::DataType::kFLOAT;
  std::array<float, 1> vals;
  weights.values = vals.data();
  weights.count = 1;
  auto input = network->addInput("input-tensor", nvinfer1::DataType::kFLOAT,
                                 nvinfer1::Dims3{1, 1, 1});
  ASSERT_NE(input, nullptr);

  const char* fc_layer_name = "my-fc-layer";
  auto layer = network->addFullyConnected(*input, 1, weights, weights);
  ASSERT_NE(layer, nullptr);
  layer->setName(fc_layer_name);

  // Check layer names.
  EXPECT_THAT(network.get(),
              AllOf(LayerNamesNonEmpty(), LayerNamesAreArray({fc_layer_name})));

  // Add layer with default name and check layer name.
  layer = network->addFullyConnected(*input, 1, weights, weights);
  EXPECT_THAT(network.get(), AllOf(LayerNamesNonEmpty(),
                                   Not(LayerNamesAreArray({fc_layer_name}))));
}

}  // namespace convert

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
