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

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

#include <string.h>

#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/test.h"
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

std::vector<TensorShape> DimVecToShapeVec(std::vector<nvinfer1::Dims3> dimvec) {
  std::vector<TensorShape> shapevec(dimvec.size());
  for (int i = 0; i < dimvec.size(); i++) {
    TensorShape shape;
    TF_CHECK_OK(
        TensorShapeUtils::MakeShape(dimvec[i].d, dimvec[i].nbDims, &shape));
    shapevec[i] = shape;
  }
  return shapevec;
}

bool DimsContained(const nvinfer1::Dims& dim, const nvinfer1::Dims& min,
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

bool DimsEqual(const nvinfer1::Dims& a, const nvinfer1::Dims& b) {
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
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
    network_ = TrtUniquePtrType<nvinfer1::INetworkDefinition>(
        builder_->createNetworkV2(flags_));
    builder_config_ = TrtUniquePtrType<nvinfer1::IBuilderConfig>(
        builder_->createBuilderConfig());
    builder_config_->setMaxWorkspaceSize(1 << 10);
#else
    network_ = TrtUniquePtrType<nvinfer1::INetworkDefinition>(
        builder_->createNetwork());
    builder_->setMaxWorkspaceSize(1 << 10);
#endif
  }

  // Defines a simple network: output = input1 + input2.
  void DefineNetwork(nvinfer1::INetworkDefinition* network,
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

  Logger logger_;
  TrtUniquePtrType<nvinfer1::IBuilder> builder_;
  TrtUniquePtrType<nvinfer1::INetworkDefinition> network_;
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  TrtUniquePtrType<nvinfer1::IBuilderConfig> builder_config_;
#endif
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine;
  std::vector<TrtUniquePtrType<nvinfer1::IExecutionContext>> exec_context_;
  // The order is important: exec_context_ must be destroyed first, and logger
  // at last.
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  const uint32_t flags_ =
      1U << static_cast<int>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
};

TEST_F(TrtShapeOptimizationProfileTest, Static) {
  // Network with static input shape
  nvinfer1::Dims3 dims(8, 8, 10);
  DefineNetwork(network_.get(), dims);

  TrtShapeOptimizationProfile profile;

#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  // Configure and build engine - should be a no-op
  TF_CHECK_OK(profile.ConfigureBuilder(builder_.get(), builder_config_.get(),
                                       network_.get()));

  engine = TrtUniquePtrType<nvinfer1::ICudaEngine>(
      builder_->buildEngineWithConfig(*network_, *builder_config_));
#else
  engine = TrtUniquePtrType<nvinfer1::ICudaEngine>(
      builder_->buildCudaEngine(*network_));
#endif
  EXPECT_NE(nullptr, engine);
  TF_CHECK_OK(profile.CreateExecutionContexts(engine.get(), exec_context_));
  // A single execution context should be created for a graph with static input
  ASSERT_EQ(exec_context_.size(), 1);
  EXPECT_NE(nullptr, exec_context_[0]);

  std::vector<nvinfer1::Dims3> dim_vec(2, dims);
  std::vector<TensorShape> shape_vec = DimVecToShapeVec(dim_vec);
  EXPECT_EQ(-1, profile.GetProfileNumber(shape_vec));
}

#if IS_TRT_VERSION_GE(6, 0, 0, 0)
TEST_F(TrtShapeOptimizationProfileTest, Dynamic) {
  // Network with dynamic input shapes
  nvinfer1::Dims3 dims(-1, -1, 10);
  DefineNetwork(network_.get(), dims);

  TrtShapeOptimizationProfile profile;
  std::vector<std::vector<nvinfer1::Dims3>> input_profiles{
      {nvinfer1::Dims3(2, 2, 10), nvinfer1::Dims3(2, 2, 10)},
      {nvinfer1::Dims3(3, 3, 10), nvinfer1::Dims3(3, 3, 10)},
      {nvinfer1::Dims3(16, 16, 10), nvinfer1::Dims3(16, 16, 10)},
  };

  // Simulate a profile collection phase
  for (auto dim_vec : input_profiles) {
    std::vector<TensorShape> shape_vec = DimVecToShapeVec(dim_vec);
    profile.AddShape(shape_vec);
  }
  profile.InitProfiles();

  // Configure and build engine
  TF_CHECK_OK(profile.ConfigureBuilder(builder_.get(), builder_config_.get(),
                                       network_.get()));
  engine = TrtUniquePtrType<nvinfer1::ICudaEngine>(
      builder_->buildEngineWithConfig(*network_.get(), *builder_config_.get()));
  ASSERT_NE(nullptr, engine);

  TF_CHECK_OK(profile.CreateExecutionContexts(engine.get(), exec_context_));

  // Each profile has an associated execution context.
  EXPECT_EQ(exec_context_.size(), input_profiles.size());

  // Check if the profiles are assigned correctly.
  for (auto dimvec : input_profiles) {
    std::vector<TensorShape> shape_vec = DimVecToShapeVec(dimvec);
    int idx = profile.GetProfileNumber(shape_vec);
    int prof_idx = exec_context_[idx]->getOptimizationProfile();
    ASSERT_GE(prof_idx, 0);

    for (int j = 0; j < dimvec.size(); j++) {
      nvinfer1::Dims min = engine->getProfileDimensions(
          j, prof_idx, nvinfer1::OptProfileSelector::kMIN);
      nvinfer1::Dims max = engine->getProfileDimensions(
          j, prof_idx, nvinfer1::OptProfileSelector::kMAX);
      nvinfer1::Dims opt = engine->getProfileDimensions(
          j, prof_idx, nvinfer1::OptProfileSelector::kOPT);

      // This should always hold.
      EXPECT_TRUE(DimsContained(dimvec[j], min, max));

      // The following test depends on the profile creation strategy, and needs
      // to be updated (disabled) if the default trategy (defined by
      // InitProfiles) changes.
      EXPECT_TRUE(DimsEqual(dimvec[j], opt));
    }
  }
}
#endif

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
