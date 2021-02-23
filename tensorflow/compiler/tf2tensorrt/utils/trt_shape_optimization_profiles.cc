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

#include "tensorflow/compiler/tf2tensorrt/utils/trt_shape_optimization_profiles.h"

#include <algorithm>
#include <functional>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
namespace tensorflow {
namespace tensorrt {

// Returns a vector of nvinfer1::Dims for a vector of TensorShapes.
template <typename TensorShapeType>
std::vector<nvinfer1::Dims> GetDimVec(std::vector<TensorShapeType> shape_vec) {
  std::vector<nvinfer1::Dims> dimvec(shape_vec.size());
  absl::c_transform(shape_vec, dimvec.begin(), [](TensorShapeType shape) {
    return TensorShapeToTrtDims(shape, false);
  });
  return dimvec;
}

// In dynamic shape mode the optimization profile dims are only allowed to
// differ from the network input dims where the network input dims have -1
// values. We enforce this condition by changing prof_dims if necessary.
void EnforceCompatibility(nvinfer1::Dims* prof_dims,
                          const PartialTensorShape& input_shape) {
  for (int i = 0; i < input_shape.dims(); i++) {
    if (input_shape.dim_size(i) != -1) {
      prof_dims->d[i] = input_shape.dim_size(i);
    }
  }
}

void SetImplicitBatchModeCompatibleProfile(
    const std::vector<nvinfer1::Dims>& dimvec, std::vector<nvinfer1::Dims>* min,
    std::vector<nvinfer1::Dims>* opt, std::vector<nvinfer1::Dims>* max) {
  *min = dimvec;
  for (auto& dim : *min) {
    dim.d[0] = 1;  // Set min batch size to 1.
  }
  *opt = dimvec;
  *max = dimvec;
}

void TrtShapeOptimizationProfile::ImplicitBatchModeCompatibleStrategy() {
  for (auto& shape_vec : input_shapes_) {
    if (!shape_vec.empty()) {
      std::vector<nvinfer1::Dims> dimvec = GetDimVec(shape_vec);
      std::vector<nvinfer1::Dims> min, opt, max;
      SetImplicitBatchModeCompatibleProfile(dimvec, &min, &opt, &max);
      OptimizationProfileConfig profConfig{min, opt, max};
      profiles_.push_back(std::move(profConfig));
    }
  }
}

void TrtShapeOptimizationProfile::OptimalStrategy() {
  for (auto& shape_vec : input_shapes_) {
    if (!shape_vec.empty()) {
      std::vector<nvinfer1::Dims> min = GetDimVec(shape_vec);
      std::vector<nvinfer1::Dims> opt = min;
      std::vector<nvinfer1::Dims> max = min;
      OptimizationProfileConfig profConfig{min, opt, max};
      profiles_.push_back(std::move(profConfig));
    }
  }
}

// Adjust shape value profile to prevent TRT from removing shape value input
// bindings whose value is redundant (only a single value matches the profile).
// This should be removed once the NVIDIA bug 3153064 is fixed.
void FixShapeValueProfile(OptimizationProfileConfig* prof,
                          const std::vector<bool>& is_shape_tensor) {
  for (int i = 0; i < prof->min.size(); i++) {
    if (is_shape_tensor[i] &&
        std::equal(prof->min[i].d, prof->min[i].d + prof->min[i].nbDims,
                   prof->max[i].d)) {
      VLOG(2) << "Adjust profile for shape value tensor " << i;
      prof->max[i].d[0]++;
    }
  }
}

void TrtShapeOptimizationProfile::InitProfiles(
    const std::vector<PartialTensorShape>& input_partial_shapes) {
  if (input_shapes_.size() == 0) {
    VLOG(1) << "Not creating profiles without input_shapes. "
               "You have to enable profile generation mode first (build).";
    return;
  }
  switch (strategy_) {
    case ProfileStrategy::kImplicitBatchModeCompatible:
      VLOG(1) << "Creating profiles with ImplicitBatchModeCompatible strategy";
      ImplicitBatchModeCompatibleStrategy();
      break;
    case ProfileStrategy::kOptimal:
      VLOG(1) << "Creating profiles with Optimal strategy";
      OptimalStrategy();
      break;
  }
  // Define a mask that describe which input could be a shape tensor. Note that
  // here we can have false positives. The shape tensor mask will be updated
  // once the network is constructed.
  SetShapeTensorMask(input_partial_shapes);
  if (input_partial_shapes.size() > 0) {
    for (OptimizationProfileConfig& prof : profiles_) {
      // TODO: Remove this when the bug is fixed.
      FixShapeValueProfile(&prof, is_shape_tensor_);
      for (int i = 0; i < input_partial_shapes.size(); i++) {
        auto network_input = input_partial_shapes[i];
        EnforceCompatibility(&prof.min[i], network_input);
        EnforceCompatibility(&prof.opt[i], network_input);
        EnforceCompatibility(&prof.max[i], network_input);
      }
    }
  }
}

#if IS_TRT_VERSION_GE(6, 0, 0, 0)
Status TrtShapeOptimizationProfile::AddProfiles(
    nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
    const nvinfer1::INetworkDefinition* network) {
  // Create a vector of optimization profiles.
  for (int i = 0; i < profiles_.size(); i++) {
    auto* optProfile = builder->createOptimizationProfile();
    Status status = profiles_[i].SetDimensions(network, optProfile);
    if (!status.ok()) {
      return status;
    }
    int idx = -1;
    if (optProfile->isValid()) {
      idx = config->addOptimizationProfile(optProfile);
    }
    if (idx >= 0) {
      if (i != idx) {
        return errors::Internal(
            "Profile index of engine config is different from resource profile "
            "index: ",
            i, " != ", idx);
      }
      VLOG(1) << "Added optimization profile " << profiles_[i].DebugString()
              << " to builder config.";
    } else {
      LOG(ERROR) << "Failed to add optimization profile "
                 << profiles_[i].DebugString()
                 << ". This usually happens when profile is invalid.";
    }
  }
  if (!profiles_.empty() && config->getNbOptimizationProfiles() == 0) {
    return errors::Internal("Failure in adding an optimization profile.");
  }
  need_profiles_ = config->getNbOptimizationProfiles() > 0;
  // Update the the mask that flag shape tensors. The network is known now,
  // the mask will be correct.
  SetShapeTensorMask(network);
  // if TRT_VERSION < 6, then we do not need to add.
  return Status::OK();
}
#endif

#if IS_TRT_VERSION_GE(6, 0, 0, 0)
Status TrtShapeOptimizationProfile::ConfigureBuilder(
    nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
    const nvinfer1::INetworkDefinition* network) {
  TF_RETURN_IF_ERROR(AddProfiles(builder, config, network));
  return Status::OK();
}
#endif

// Sets the shape tensor mask using the network definition.
void TrtShapeOptimizationProfile::SetShapeTensorMask(
    const nvinfer1::INetworkDefinition* network) {
  int n_inputs = network->getNbInputs();
  is_shape_tensor_.resize(n_inputs, false);
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  for (int i = 0; i < n_inputs; i++) {
    const nvinfer1::ITensor* input = network->getInput(i);
    is_shape_tensor_[i] = input->isShapeTensor();
    if (is_shape_tensor_[i]) {
      VLOG(2) << "Found shape tensor " << input->getName() << ' at ' << i;
    }
  }
#endif
  has_shape_tensor_ =
      absl::c_any_of(is_shape_tensor_, [](bool b) { return b; });
}

// Sets the shape tensor mask using the input partial shapes. This only tells
// whether the tensors are shape value compatible, only the final network
// definition or the engine would give concrete answers.
void TrtShapeOptimizationProfile::SetShapeTensorMask(
    const std::vector<PartialTensorShape>& input_partial_shapes) {
  is_shape_tensor_.resize(input_partial_shapes.size(), false);
  for (int i = 0; i < input_partial_shapes.size(); i++) {
    is_shape_tensor_[i] = IsTrtShapeTensorCompatible(input_partial_shapes[i]);
    if (is_shape_tensor_[i]) {
      VLOG(2) << "Found shape compatible tensor at " << i;
    }
  }
  has_shape_tensor_ =
      absl::c_any_of(is_shape_tensor_, [](bool b) { return b; });
}

int TrtShapeOptimizationProfile::GetProfileNumber(
    const std::vector<TensorShape>& shapes) {
  if (!need_profiles_) return 0;
  for (int i = 0; i < profiles_.size(); i++) {
    if (profiles_[i].IncludesShapes(shapes)) {
      return i;
    }
  }
  VLOG(1) << "Profile not found for input shapes " << DebugString(shapes)
          << ".";
  return -1;
}

Status TrtShapeOptimizationProfile::CreateExecutionContexts(
    nvinfer1::ICudaEngine* engine, std::vector<ExecutionContext>& exec_context,
    TRTBaseAllocator* memory_allocator) {
  int i = 0;
  // The following loop runs once if we have static shapes, to create a single
  // execution context without profiles. In dynamic mode we create one context
  // for each profile and set the corresponding optimization profile.
  do {
    VLOG(1) << "Creating execution context " << i;
    auto exec_context_status =
        ExecutionContext::Create(engine, memory_allocator);
    if (!exec_context_status.ok()) {
      return errors::Internal("Failed to create execution context");
    }
    if (i > 0) {
      // This condition is needed for two reasons:
      // - using static shapes we do not have any profiles so we cannot call
      //   set optimizationprofiles.
      // - The 0th profile is set implicitly for the first execution context
      //   therefore we do not need to set.
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
      bool stat = exec_context_status.ValueOrDie()
                      .GetIExecutionContext()
                      ->setOptimizationProfile(i);
      if (!stat) {
        return errors::Internal("Could not set TRT optimization profile.");
      }
#endif
    }
    exec_context.push_back(std::move(exec_context_status.ValueOrDie()));
    i++;
  } while (i < profiles_.size());

  return Status::OK();
}

Status TrtShapeOptimizationProfile::RestoreProfiles(
    const nvinfer1::ICudaEngine* engine) {
  need_profiles_ = false;
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  if (!engine) {
    // We do not need to restore profiles for an empty engine.
    return Status::OK();
  }
#if IS_TRT_VERSION_GE(7, 0, 0, 0)
  if (engine->hasImplicitBatchDimension()) {
    // Nothing to do, we cannot have profiles in implicit batch mode.
    return Status::OK();
  }
#endif
  int n_profiles = engine->getNbOptimizationProfiles();
  need_profiles_ = n_profiles > 0;
  int n_inputs = GetNumberOfEngineInputs(engine);
  VLOG(2) << "Attempting to restore " << n_profiles << " profiles, each with "
          << n_inputs << " inputs";
  for (int prof_idx = 0; prof_idx < n_profiles; prof_idx++) {
    OptimizationProfileConfig cfg;
    for (int j = 0; j < n_inputs; j++) {
      nvinfer1::Dims min = engine->getProfileDimensions(
          j, prof_idx, nvinfer1::OptProfileSelector::kMIN);
      nvinfer1::Dims max = engine->getProfileDimensions(
          j, prof_idx, nvinfer1::OptProfileSelector::kMAX);
      nvinfer1::Dims opt = engine->getProfileDimensions(
          j, prof_idx, nvinfer1::OptProfileSelector::kOPT);
      cfg.min.push_back(min);
      cfg.max.push_back(max);
      cfg.opt.push_back(opt);
    }
    VLOG(2) << "Restored profile " << cfg.DebugString();
    profiles_.push_back(std::move(cfg));
  }
#endif
  return Status::OK();
}

int TrtShapeOptimizationProfile::GetNumProfiles() const {
  return profiles_.size();
}

}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
