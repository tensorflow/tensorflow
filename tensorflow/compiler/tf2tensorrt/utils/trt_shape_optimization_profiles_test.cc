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

#include <string.h>

#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/test.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

std::vector<TensorShape> dimvec2shapevec(std::vector<nvinfer1::Dims3> dimvec) {
  std::vector<TensorShape> shapevec(dimvec.size());
  for (int i = 0; i < dimvec.size(); i++) {
    TensorShape shape;
    TensorShapeUtils::MakeShape(dimvec[i].d, dimvec[i].nbDims, &shape);
    shapevec[i] = shape;
  }
  return shapevec;
}

bool dimsContained(const nvinfer1::Dims& dim, const nvinfer1::Dims& min,
                   const nvinfer1::Dims& max) {
  if (dim.nbDims != min.nbDims || dim.nbDims != max.nbDims) {
    return false;
  }
  for (int i = 0; i < dim.nbDims; i++) {
    if (dim.d[i] < min.d[i] || dim.d[i] > max.d[i]) {
      return false;
    }
  }
  return true;
}

bool dimsEqual(const nvinfer1::Dims& a, const nvinfer1::Dims& b) {
  if (a.nbDims != b.nbDims) {
    return false;
  }
  for (int i = 0; i < a.nbDims; i++) {
    if (a.d[i] != b.d[i]) {
      return false;
    }
  }
  return true;
}

class TrtShapeOptimizationProfileTest : public ::testing::Test {
 protected:
  void SetUp() override {
    builder_ = TrtUniquePtrType<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger_));

    network_ = TrtUniquePtrType<nvinfer1::INetworkDefinition>(
        builder_->createNetworkV2(flags_));
    // TODO(Tamas) think over where do we need TRT version ifdefs
    builder_config_ = TrtUniquePtrType<nvinfer1::IBuilderConfig>(
        builder_->createBuilderConfig());

    builder_config_->setMaxWorkspaceSize(1 << 10);
  }

  // define a simple network: output = input1 + input2
  void defineNetwork(nvinfer1::INetworkDefinition* network,
                     nvinfer1::Dims3& dims) {
    nvinfer1::ITensor* input1 =
        network->addInput("input1", nvinfer1::DataType::kFLOAT, dims);
    EXPECT_NE(nullptr, input1);

    nvinfer1::ITensor* input2 =
        network->addInput("input2", nvinfer1::DataType::kFLOAT, dims);
    EXPECT_NE(nullptr, input1);

    auto layer = network->addElementWise(*input1, *input2,
                                         nvinfer1::ElementWiseOperation::kSUM);
    EXPECT_NE(nullptr, layer);
    // Mark the output.
    nvinfer1::ITensor* output = layer->getOutput(0);
    output->setName("output");
    network->markOutput(*output);
  }

  TrtUniquePtrType<nvinfer1::IBuilder> builder_;
  TrtUniquePtrType<nvinfer1::INetworkDefinition> network_;
  TrtUniquePtrType<nvinfer1::IBuilderConfig> builder_config_;
  std::vector<TrtUniquePtrType<nvinfer1::IExecutionContext>> exec_context_;

  Logger logger_;
  const uint32_t flags_ =
      1U << static_cast<int>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
};

TEST_F(TrtShapeOptimizationProfileTest, Basic) {
  nvinfer1::Dims3 dims(-1, -1, 10);
  defineNetwork(network_.get(), dims);

  TrtShapeOptimizationProfile profile;
  std::vector<std::vector<nvinfer1::Dims3>> input_profiles{
      {nvinfer1::Dims3(2, 2, 10), nvinfer1::Dims3(2, 2, 10)},
      {nvinfer1::Dims3(3, 3, 10), nvinfer1::Dims3(3, 3, 10)},
      {nvinfer1::Dims3(16, 16, 10), nvinfer1::Dims3(16, 16, 10)},
  };

  // Simulate a profile collection phase
  for (auto dim_vec : input_profiles) {
    std::vector<TensorShape> shape_vec = dimvec2shapevec(dim_vec);
    profile.addShape(shape_vec);
  }

  // Configure and build engine
  profile.configureBuilder(builder_.get(), builder_config_.get(),
                           network_.get());
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine(
      builder_->buildEngineWithConfig(*network_.get(), *builder_config_.get()));
  EXPECT_NE(nullptr, engine);

  profile.createExcecutionContexts(engine.get(), exec_context_);
  // Each profile has an associated execution context
  // This test depends on the profile creation strategy:
  // e.g. if we have a default context, then the sizes will not match
  EXPECT_EQ(exec_context_.size(), input_profiles.size());

  // Check if the profiles are assigned correctly
  for (auto dimvec : input_profiles) {
    std::vector<TensorShape> shape_vec = dimvec2shapevec(dimvec);
    int idx = profile.getProfileNumber(shape_vec);
    int prof_idx = exec_context_[idx]->getOptimizationProfile();
    ASSERT_GE(prof_idx, 0);

    for (int j = 0; j < dimvec.size(); j++) {
      nvinfer1::Dims min = engine->getProfileDimensions(
          j, prof_idx, nvinfer1::OptProfileSelector::kMIN);
      nvinfer1::Dims max = engine->getProfileDimensions(
          j, prof_idx, nvinfer1::OptProfileSelector::kMAX);
      nvinfer1::Dims opt = engine->getProfileDimensions(
          j, prof_idx, nvinfer1::OptProfileSelector::kOPT);

      EXPECT_TRUE(dimsContained(dimvec[j], min, max));
      EXPECT_TRUE(dimsEqual(dimvec[j], opt));
    }
  }
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
