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

#if GOOGLE_CUDA && GOOGLE_TENSORRT

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

std::vector<TensorShape> DimVecToShapeVec(
    std::vector<nvinfer1::Dims3> dimvec,
    bool expand_with_empty_shape_values = false) {
  std::vector<TensorShape> shapevec(dimvec.size());
  for (int i = 0; i < dimvec.size(); i++) {
    TensorShape shape;
    TF_CHECK_OK(
        TensorShapeUtils::MakeShape(dimvec[i].d, dimvec[i].nbDims, &shape));
    shapevec[i] = shape;
  }
  if (expand_with_empty_shape_values) {
    shapevec.resize(2 * dimvec.size());  // Append empty shape values
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

class TrtShapeOptimizationProfileTest
    : public ::testing::TestWithParam<ProfileStrategy> {
 protected:
  TrtShapeOptimizationProfileTest() {
    strategy_ = GetParam();
    builder_ = TrtUniquePtrType<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger_));
    network_ = TrtUniquePtrType<nvinfer1::INetworkDefinition>(
        builder_->createNetworkV2(flags_));
    builder_config_ = TrtUniquePtrType<nvinfer1::IBuilderConfig>(
        builder_->createBuilderConfig());
    builder_config_->setMaxWorkspaceSize(1 << 10);
  }

  // Defines a simple network: output = input1 + input2.
  void DefineNetwork(nvinfer1::INetworkDefinition* network,
                     nvinfer1::Dims3& dims) {
    ITensorProxyPtr input1 =
        network->addInput("input1", nvinfer1::DataType::kFLOAT, dims);
    EXPECT_NE(nullptr, input1->trt_tensor());

    ITensorProxyPtr input2 =
        network->addInput("input2", nvinfer1::DataType::kFLOAT, dims);
    EXPECT_NE(nullptr, input2->trt_tensor());

    auto layer =
        network->addElementWise(*input1->trt_tensor(), *input2->trt_tensor(),
                                nvinfer1::ElementWiseOperation::kSUM);
    EXPECT_NE(nullptr, layer);
    // Mark the output.
    ITensorProxyPtr output = layer->getOutput(0);
    output->setName("output");
    network->markOutput(*output->trt_tensor());
  }

  void CheckProfile(const std::vector<nvinfer1::Dims3>& dimvec,
                    TrtShapeOptimizationProfile* profile, bool has_prof,
                    bool test_optimality) {
    std::vector<TensorShape> shape_vec = DimVecToShapeVec(dimvec);
    int idx = profile->GetProfileNumber(shape_vec);
    ASSERT_EQ(idx >= 0, has_prof);
    if (idx < 0) return;
    int prof_idx = exec_contexts_[idx]->getOptimizationProfile();
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

      if (test_optimality) {
        // We shall have selected an optimal strategy.
        EXPECT_TRUE(DimsEqual(dimvec[j], opt));
      }
    }
  }

  Logger& logger_ = *Logger::GetLogger();
  TrtUniquePtrType<nvinfer1::IBuilder> builder_;
  TrtUniquePtrType<nvinfer1::INetworkDefinition> network_;
  TrtUniquePtrType<nvinfer1::IBuilderConfig> builder_config_;
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine;
  std::vector<ExecutionContext> exec_contexts_;
  // The order is important: exec_context_ must be destroyed first, and logger
  // at last.
  const uint32_t flags_ =
      1U << static_cast<int>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  ProfileStrategy strategy_;
};

INSTANTIATE_TEST_CASE_P(
    OptProfilesTestInstantiation, TrtShapeOptimizationProfileTest,
    ::testing::Values(ProfileStrategy::kRange, ProfileStrategy::kOptimal,
                      ProfileStrategy::kRangeOptimal,
                      ProfileStrategy::kImplicitBatchModeCompatible));

TEST_P(TrtShapeOptimizationProfileTest, Static) {
  // Static mode does not depend on strategies, we test only once.
  if (strategy_ != ProfileStrategy::kRange) return;

  // Network with static input shape.
  nvinfer1::Dims3 dims(8, 8, 10);
  DefineNetwork(network_.get(), dims);

  TrtShapeOptimizationProfile profile;

  // Configure and build engine - should be a no-op.
  TF_CHECK_OK(profile.ConfigureBuilder(builder_.get(), builder_config_.get(),
                                       network_.get()));

  engine = TrtUniquePtrType<nvinfer1::ICudaEngine>(
      builder_->buildEngineWithConfig(*network_, *builder_config_));
  EXPECT_NE(nullptr, engine);
  TF_CHECK_OK(profile.CreateExecutionContexts(engine.get(), &exec_contexts_));
  // A single execution context should be created for a graph with static input.
  ASSERT_EQ(exec_contexts_.size(), 1);
  EXPECT_NE(nullptr, exec_contexts_[0]);

  std::vector<nvinfer1::Dims3> dim_vec(2, dims);
  std::vector<TensorShape> shape_vec = DimVecToShapeVec(dim_vec);
  EXPECT_EQ(0, profile.GetProfileNumber(shape_vec));
}

TEST_P(TrtShapeOptimizationProfileTest, Dynamic) {
  // Network with dynamic input shapes.
  nvinfer1::Dims3 dims(-1, -1, 10);
  DefineNetwork(network_.get(), dims);

  TrtShapeOptimizationProfile profile;
  std::vector<std::vector<nvinfer1::Dims3>> input_profiles{
      {nvinfer1::Dims3(2, 2, 10), nvinfer1::Dims3(2, 2, 10)},
      {nvinfer1::Dims3(3, 3, 10), nvinfer1::Dims3(3, 3, 10)},
      {nvinfer1::Dims3(16, 16, 10), nvinfer1::Dims3(16, 16, 10)},
  };

  std::vector<nvinfer1::Dims3> unseen_shapes{nvinfer1::Dims3(5, 5, 10),
                                             nvinfer1::Dims3(9, 9, 10)};

  // Simulate a profile collection phase.
  for (auto dim_vec : input_profiles) {
    std::vector<TensorShape> shape_vec = DimVecToShapeVec(dim_vec, true);
    profile.AddShape(shape_vec);
  }
  std::vector<PartialTensorShape> input_partial_shapes;
  TF_CHECK_OK(GetNetworkInputShapes(network_.get(), &input_partial_shapes));
  profile.InitProfiles(input_partial_shapes, strategy_);

  // Configure and build engine.
  TF_CHECK_OK(profile.ConfigureBuilder(builder_.get(), builder_config_.get(),
                                       network_.get()));
  engine = TrtUniquePtrType<nvinfer1::ICudaEngine>(
      builder_->buildEngineWithConfig(*network_.get(), *builder_config_.get()));
  ASSERT_NE(nullptr, engine);

  TF_CHECK_OK(profile.CreateExecutionContexts(engine.get(), &exec_contexts_));

  int n_profiles_exp;
  switch (strategy_) {
    case (ProfileStrategy::kImplicitBatchModeCompatible):
    case (ProfileStrategy::kOptimal):
      n_profiles_exp = input_profiles.size();
      break;
    case (ProfileStrategy::kRange):
      n_profiles_exp = 1;
      break;
    case (ProfileStrategy::kRangeOptimal):
      n_profiles_exp = 1 + input_profiles.size();
      break;
  }
  // Each profile has an associated execution context.
  EXPECT_EQ(exec_contexts_.size(), n_profiles_exp);

  profile.SetShapeTensorMask(network_.get());

  EXPECT_EQ(profile.HasShapeTensor(), false);

  // Check if the profiles are assigned correctly.
  for (auto dimvec : input_profiles) {
    bool test_optimal_prof = strategy_ == ProfileStrategy::kOptimal ||
                             strategy_ == ProfileStrategy::kRangeOptimal;
    CheckProfile(dimvec, &profile, true, test_optimal_prof);
  }
  bool has_prof = (strategy_ == ProfileStrategy::kRange ||
                   strategy_ == ProfileStrategy::kRangeOptimal);
  CheckProfile(unseen_shapes, &profile, has_prof, false);
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
