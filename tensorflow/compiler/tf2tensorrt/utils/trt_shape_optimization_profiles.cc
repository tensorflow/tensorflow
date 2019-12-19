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

#include <algorithm>
#include <functional>
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_shape_optimization_profiles.h"

namespace tensorflow {
namespace tensorrt {

bool isAnyInputDynamic(const nvinfer1::INetworkDefinition *network) {
  bool is_any_dynamic = false;
  for (int i = 0; i < network->getNbInputs(); i++) {
    auto input = network->getInput(i);
    auto dims = input->getDimensions();
    is_any_dynamic |= !HasStaticShape(dims);
  }
  return is_any_dynamic;
}

/// Create optimization profise for a list of input shapes.
/// The list of input shapes are stored in shapes_.
void TrtShapeOptimizationProfile::initProfiles(int n_profiles) {
  // Can it happen that we call build twice? In that case we should decide
  // whether we want to append the new profiles, or start again?
  // profiles_.clear();
  if (n_profiles != 0) {
    VLOG(1) << "Strategy for n_profiles != 0 is not yet implemented, using "
               "default strategy";
    n_profiles = input_shapes_.size();
  }
  VLOG(1) << "Default startegy: one optimization profile for each input shape."
          <<  "We have " << input_shapes_.size() << " input shapes";

  for (auto& shape_vec: input_shapes_) {
    std::vector<nvinfer1::Dims> dimvec;
    for (auto& shape: shape_vec) {
       dimvec.push_back(TensorShapeToTrtDims(shape, false));
     }
    // We set min=opt=max
    OptimizationProfileConfig profConfig {dimvec, dimvec, dimvec};
    profiles_.push_back(std::move(profConfig));
  }
}

Status TrtShapeOptimizationProfile::addProfiles(
    nvinfer1::IBuilder *builder, nvinfer1::IBuilderConfig* config,
    const nvinfer1::INetworkDefinition *network) {
  // Create a vector of optimization profiles
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  VLOG(1) << "Adding an optimization profile to TRT network configuration";
  profile_map_.clear();
  auto shape_vec = input_shapes_.begin();
  for (auto& profile: profiles_) {
    auto* optProfile = builder->createOptimizationProfile();
    Status status = profile.setDimensions(network, optProfile);
    if (!status.ok()) {
      return status;
    }
    int idx = -1;
    if (optProfile->isValid()) {
      idx = config->addOptimizationProfile(optProfile);
    }
    if (idx >= 0) {
      VLOG(1) << "Optimization profile added ", profile.DebugString();
      profile_map_.emplace(std::make_pair(*shape_vec, idx));
    } else {
      VLOG(1) << "Failed to add optimization profile, ignoring config"
              << profile.DebugString();
    }
    shape_vec++;
  }
  if (config->getNbOptimizationProfiles() == 0) {
     return errors::Internal("Failure in adding an optimization profile.");
  }
#endif
// if TRT_VERSION < 6, then we do not need to add
  return Status::OK();
}

Status TrtShapeOptimizationProfile::configureBuilder(
  nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
  const nvinfer1::INetworkDefinition* network, int n_profiles) {
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  if (!isAnyInputDynamic(network)) {
    // we do not need profiles for static input
    return Status::OK();
  }
  //
  if (input_shapes_.size() == 0) {
    // This should not happen. If we did not have build mode, then a single
    // profile is added by GetEngine
    return errors::Internal("No TRT optimization profile found");
  }
  initProfiles(n_profiles);
  addProfiles(builder, config, network);
#endif
  return Status::OK();
}

int TrtShapeOptimizationProfile::getProfileNumber(std::vector<TensorShape> shapes) {
  if (profile_map_.size() == 0) {
    // No optimization profiles, we just return 0, the default profile number
    return 0;
  }
  auto it = profile_map_.find(shapes);
  if (it != profile_map_.end()) {
    return it->second;
  } else {
    VLOG(1) << "Profile not found for input " <<  DebugString(shapes);
    // TODO probably just abort TRT execution.
    return 0;
  }
}

Status TrtShapeOptimizationProfile::createExcecutionContexts(
  nvinfer1::ICudaEngine* engine,
  std::vector<TrtUniquePtrType<nvinfer1::IExecutionContext>>& exec_context) {
  int i=0;
  // The following loops runs once if we have static shapes, to create a single
  // execution context without profiles.
  // In dynamic mode we create one context for each profile and set the
  // corresponding optimization profile.
  do {
    VLOG(1) << "Creating execution context " <<  i;
    nvinfer1::IExecutionContext *ctx = engine->createExecutionContext();
    if (ctx == nullptr) {
      return errors::Internal("Failed to create execution context");
    }
    if (i>0) {
      // This condition is needed for two reasons:
      // - using static shapes we do not have any profiles so we cannot call
      //   set optimizationprofiles.
      // - The 0th profile is set implicitly for the first execution context
      //   therefore we do not need to set.
      bool stat = ctx->setOptimizationProfile(i);
      if (!stat) {
        ctx->destroy();
        return errors::Internal("Could not set TRT optimization profile.");
      }
    }
    exec_context.push_back(
      std::move(TrtUniquePtrType<nvinfer1::IExecutionContext>(ctx)));
    i++;
  } while (i<profile_map_.size());

  return Status::OK();
}

}  // namespace tensorrt
}  // namespace tensorflow
