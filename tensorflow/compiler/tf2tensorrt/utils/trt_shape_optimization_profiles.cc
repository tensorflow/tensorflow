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
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/profiler/lib/traceme.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "third_party/gpus/cuda/include/cuda_runtime_api.h"

namespace tensorflow {
namespace tensorrt {

// Returns a vector of nvinfer1::Dims for a vector of TensorShapes.
template <typename TensorShapeType>
std::vector<nvinfer1::Dims> GetDimVec(std::vector<TensorShapeType> shape_vec) {
  std::vector<nvinfer1::Dims> dimvec(shape_vec.size());
  absl::c_transform(shape_vec, dimvec.begin(), [](TensorShapeType shape) {
    auto adap = DimsAdapter::Create(shape);
    TF_CHECK_OK(adap.status());
    return adap->AsTrtDims();
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
    // Shape value tensors can have -1 value as a wildcard. We do not change
    // in that case.
    if (dim.d[0] != -1) dim.d[0] = 1;  // Set min batch size to 1.
  }
  *opt = dimvec;
  *max = dimvec;
}

void TrtShapeOptimizationProfile::ImplicitBatchModeCompatibleStrategy(
    const std::vector<std::vector<nvinfer1::Dims>>& collected_shapes) {
  for (auto& shape_vec : collected_shapes) {
    std::vector<nvinfer1::Dims> min, opt, max;
    SetImplicitBatchModeCompatibleProfile(shape_vec, &min, &opt, &max);
    VLOG(2) << "Initializing optimization profile config with min="
            << DebugString(min) << ", opt=max=" << DebugString(max);
    OptimizationProfileConfig profConfig{min, opt, max};
    profiles_.push_back(std::move(profConfig));
  }
}

// Applies a binary operation for each dimension of the input shapes.
// x[i].d[k] = op(x[i].d[k], y[i].d[k]), where i enumerates the input tensors,
// and k enumerates the dimensions of the tensors. The BinaryOperation may be
// std::min, std::max etc.
template <typename BinaryOperation>
Status ShapeProfileBinaryOp(std::vector<nvinfer1::Dims>* x,
                            const std::vector<nvinfer1::Dims>& y,
                            BinaryOperation op) {
  if (x->size() != y.size())
    return errors::InvalidArgument(
        "Number of input tensors differ during profile creation");
  for (int i = 0; i < x->size(); i++) {
    if (x->at(i).nbDims != y[i].nbDims)
      return errors::InvalidArgument(
          "Number of input dimensions differ during profile creation");
    for (int j = 0; j < x->at(i).nbDims; j++) {
      x->at(i).d[j] = op(x->at(i).d[j], y[i].d[j]);
    }
  }
  return Status::OK();
}

Status TrtShapeOptimizationProfile::RangeStrategy(
    const std::vector<std::vector<nvinfer1::Dims>>& collected_shapes) {
  if (collected_shapes.empty()) return Status::OK();

  std::vector<nvinfer1::Dims> min = collected_shapes[0];
  std::vector<nvinfer1::Dims> max = min;

  for (int i = 1; i < collected_shapes.size(); i++) {
    TF_RETURN_IF_ERROR(
        ShapeProfileBinaryOp(&min, collected_shapes[i],
                             [](int a, int b) { return std::min(a, b); }));
    TF_RETURN_IF_ERROR(
        ShapeProfileBinaryOp(&max, collected_shapes[i],
                             [](int a, int b) { return std::max(a, b); }));
  }
  VLOG(2) << "Initializing optimization profile config with min="
          << DebugString(min) << ", opt=max=" << DebugString(max);
  OptimizationProfileConfig profConfig{min, max, max};
  profiles_.push_back(std::move(profConfig));
  return Status::OK();
}

void TrtShapeOptimizationProfile::OptimalStrategy(
    const std::vector<std::vector<nvinfer1::Dims>>& collected_shapes) {
  for (auto& shape_vec : collected_shapes) {
    std::vector<nvinfer1::Dims> min = shape_vec;
    std::vector<nvinfer1::Dims> opt = min;
    std::vector<nvinfer1::Dims> max = min;
    VLOG(2) << "Initializing optimization profile config with min=opt=max="
            << DebugString(min);
    OptimizationProfileConfig profConfig{min, opt, max};
    profiles_.push_back(std::move(profConfig));
  }
}

// Collects the values of tensors that are ShapeTensorCompatible to. The values
// are stored in the actual_shape_values_ member variable.
Status TrtShapeOptimizationProfile::CollectShapeValues(OpKernelContext* ctx) {
  tensorflow::profiler::TraceMe activity(
      "TrtShapeOptimizationProfile::CollectShapeValues",
      tensorflow::profiler::TraceMeLevel::kInfo);
  const cudaStream_t* stream = CHECK_NOTNULL(
      reinterpret_cast<const cudaStream_t*>(ctx->op_device_context()
                                                ->stream()
                                                ->implementation()
                                                ->GpuStreamMemberHack()));
  actual_shape_values_.resize(ctx->num_inputs());
  if (is_shape_tensor_.empty()) {
    is_shape_tensor_.resize(ctx->num_inputs());
    for (int i = 0; i < ctx->num_inputs(); i++) {
      is_shape_tensor_[i] = IsTrtShapeTensorCompatible(ctx->input(i));
    }
  }
  int n_shape_val = 0;
  // First copy all the shape value candidates into actual_shape_values_ vector.
  for (int i = 0; i < ctx->num_inputs(); i++) {
    if (is_shape_tensor_[i]) {
      if (ctx->input_dtype(i) != DT_INT32) {
        // In case the is_shape_tensor mask was initialized with the input
        // shapes only (without knowledge of dtype) then we apply correction.
        is_shape_tensor_[i] = false;
        continue;
      }
      // We have to copy the shape values to the host, because TRT's
      // ExecutionContext::setInputShapeBinding expects a host pointer.
      n_shape_val++;
      const Tensor& input = ctx->input(i);
      actual_shape_values_[i].nbDims = input.NumElements();
      auto ret = cudaMemcpyAsync(
          actual_shape_values_[i].d, input.flat<int32>().data(),
          input.NumElements() * sizeof(int32), cudaMemcpyDeviceToHost, *stream);
      if (ret != 0) {
        return errors::Internal("Could not copy shape tensor values");
      }
      VLOG(2) << "Input " << i << " is (probably) a shape tensor, n_values="
              << input.NumElements();
    } else {
      actual_shape_values_[i] = {0, {}};
    }
  }
  if (n_shape_val > 0) {
    // If we have any shape values candidates, then wait until data is copied
    // to host.
    cudaStreamSynchronize(*stream);
  }
  return Status::OK();
}

// Collects the values of tensors that are ShapeTensorCompatible to. To be used
// for unit tests.
Status TrtShapeOptimizationProfile::CollectShapeValues(const DataVec& input) {
  actual_shape_values_.resize(input.size());
  for (int i = 0; i < input.size(); i++) {
    if (is_shape_tensor_[i]) {
      if (!IsTrtShapeTensorCompatible(input[i].tensor)) {
        return errors::Internal("Inconsistent shape tensor ", input[i].name,
                                ", ", i);
      }
      int n_elements = input[i].tensor.NumElements();
      actual_shape_values_[i].nbDims = n_elements;
      // During unit tests, the data is in unified memory
      std::copy(input[i].tensor.flat<int32>().data(),
                input[i].tensor.flat<int32>().data() + n_elements,
                actual_shape_values_[i].d);
      VLOG(2) << "Collected tensor shape values "
              << DebugString(actual_shape_values_[i]);
    } else {
      actual_shape_values_[i] = {0, {}};
    }
  }
  return Status::OK();
}

// Adjusts shape value profile to prevent TRT from removing shape value input
// bindings whose value is redundant (only a single value matches the profile).
// This should be removed once the NVIDIA bug 3153064 is fixed.
void FixShapeValueProfile(OptimizationProfileConfig* prof,
                          const std::vector<bool>& is_shape_tensor) {
  int shape_value_offset = is_shape_tensor.size();
  for (int i = 0; i < is_shape_tensor.size(); i++) {
    if (is_shape_tensor[i] &&
        std::equal(prof->min[shape_value_offset + i].d,
                   prof->min[shape_value_offset + i].d +
                       prof->min[shape_value_offset + i].nbDims,
                   prof->max[shape_value_offset + i].d)) {
      prof->max[shape_value_offset + i].d[0]++;
      VLOG(2) << "Adjusted profile for shape value tensor " << i << " "
              << DebugString(prof->max[shape_value_offset + i]);
    } else {
      VLOG(2) << i << " is not a shape tensor." << is_shape_tensor[i];
    }
  }
}

// Checks whether rhs is already contained in values.
bool AlreadyCollected(const std::vector<std::vector<nvinfer1::Dims>>& values,
                      const std::vector<nvinfer1::Dims>& rhs) {
  for (auto& lhs : values) {
    bool ret = lhs.size() == rhs.size();
    for (int i = 0; ret && i < lhs.size(); i++) {
      ret &= lhs[i].nbDims == rhs[i].nbDims;
      for (int j = 0; ret && j < lhs[i].nbDims; j++) {
        ret &= (lhs[i].d[j] == rhs[i].d[j]);
      }
    }
    if (ret) return true;
  }
  return false;
}

void TrtShapeOptimizationProfile::InitProfiles(
    const std::vector<PartialTensorShape>& input_partial_shapes,
    ProfileStrategy strategy) {
  strategy_ = strategy;
  if (input_shapes_.size() == 0) {
    VLOG(1) << "Not creating profiles without input_shapes. "
               "You have to enable profile generation mode first (build).";
    return;
  }
  // Preprocess the vector of input shapes and shape values:
  // - Converts TensorShape -> nvinfer::Dims.
  // - Concatenates the shape values after the input shapes:
  //   dimvec = [dim0, dim1,..., shapeval0, shapval1, ...]
  // - Ensures that the list is unique.
  std::vector<std::vector<nvinfer1::Dims>> collected_shapes;
  for (int i = 0; i < input_shapes_.size(); i++) {
    auto shape_vec = input_shapes_[i];
    VLOG(2) << "Initprofiles, processing shape " << i;
    if (!shape_vec.empty()) {
      std::vector<nvinfer1::Dims> dimvec = GetDimVec(shape_vec);
      dimvec.insert(dimvec.end(), input_shape_values_[i].begin(),
                    input_shape_values_[i].end());
      // TODO(tfeher): This condition should not apply for explicit profile. In
      // that case consicutive elements in collected_shapes contain the user
      // defined values of min, opt and max, and it is valid the have min = opt
      // and opt = max.
      if (!AlreadyCollected(collected_shapes, dimvec)) {
        collected_shapes.push_back(dimvec);
      }
    }
  }
  switch (strategy_) {
    case ProfileStrategy::kImplicitBatchModeCompatible:
      VLOG(1) << "Creating profiles with ImplicitBatchModeCompatible strategy";
      ImplicitBatchModeCompatibleStrategy(collected_shapes);
      break;
    // Treat all other strategies the same as kOptimal for now. Implementing
    // those is outlined in the dynamic shape support implementation plan.
    case ProfileStrategy::kRange:
      VLOG(1) << "Creating profiles with Range strategy";
      TF_CHECK_OK(RangeStrategy(collected_shapes));
      break;
    case ProfileStrategy::kRangeOptimal:
      VLOG(1) << "Creating profiles with RangeOptimal strategy";
      OptimalStrategy(collected_shapes);
      TF_CHECK_OK(RangeStrategy(collected_shapes));
      break;
    case ProfileStrategy::kOptimal:
      VLOG(1) << "Creating profiles with Optimal strategy";
      OptimalStrategy(collected_shapes);
      break;
  }
  // Define a mask that describe which input could be a shape tensor. Note
  // that here we can have false positives. The shape tensor mask will be
  // updated once the network is constructed.
  SetShapeTensorMask(input_partial_shapes);
  if (input_partial_shapes.size() > 0) {
    for (OptimizationProfileConfig& prof : profiles_) {
      // TODO: Remove this when the bug is fixed.
#if !IS_TRT_VERSION_GE(8, 0, 0, 0)
      FixShapeValueProfile(&prof, is_shape_tensor_);
#endif
      for (int i = 0; i < input_partial_shapes.size(); i++) {
        auto network_input = input_partial_shapes[i];
        EnforceCompatibility(&prof.min[i], network_input);
        EnforceCompatibility(&prof.opt[i], network_input);
        EnforceCompatibility(&prof.max[i], network_input);
      }
    }
  }
}

void TrtShapeOptimizationProfile::InitCalibProfile(
    const std::vector<TensorShape>& shapes) {
  VLOG(1) << "Collected shape(s) " << DebugString(shapes) << " for "
          << " calibration profile.";
  auto shape_vec = shapes;
  if (!shape_vec.empty()) {
    std::vector<nvinfer1::Dims> dimvec = GetDimVec(shape_vec);
    dimvec.insert(dimvec.end(), actual_shape_values_.begin(),
                  actual_shape_values_.end());
    VLOG(2) << "Initializing calibration optimization profile config with "
            << "min=opt=max " << DebugString(dimvec);

    OptimizationProfileConfig profConfig{dimvec, dimvec, dimvec};
    calib_profiles_ = std::move(profConfig);
  } else {
    VLOG(2) << "Failed to initialize calibration optimization profile.";
  }
}

Status TrtShapeOptimizationProfile::AddProfiles(
    nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
    const nvinfer1::INetworkDefinition* network) {
  // Create optimization profile for calibration if necessary.
  if (!calib_profiles_.min.empty()) {
    VLOG(2) << "Setting up calibration profies";
    auto* calibProfile = builder->createOptimizationProfile();
    Status status =
        calib_profiles_.SetDimensions(network, calibProfile, input_mask_);
    if (!status.ok()) {
      return status;
    }
    bool result = false;
    if (calibProfile->isValid()) {
      result = config->setCalibrationProfile(calibProfile);
    } else {
      VLOG(2) << "Calibration profile is not valid";
    }
    if (result) {
      VLOG(2) << "Added calibration optimization profile "
              << calib_profiles_.DebugString() << " to builder config.";
    } else {
      VLOG(2) << "FAILED TO ADD PROFILE";
      LOG(ERROR) << "Failed to add calibration optimization profile "
                 << calib_profiles_.DebugString()
                 << ". This usually happens when profile is invalid.";
    }
  }
  // Create a vector of optimization profiles.
  for (int i = 0; i < profiles_.size(); i++) {
    auto* optProfile = builder->createOptimizationProfile();
    Status status =
        profiles_[i].SetDimensions(network, optProfile, input_mask_);
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
            "Profile index of engine config is different from source profile "
            "index: ",
            i, " != ", idx);
      }
      VLOG(1) << "Added optimization profile " << profiles_[i].DebugString()
              << " with idx " << idx << " to builder config.";
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
  // Update the mask that flag shape tensors. The network is known now,
  // the mask will be correct.
  SetShapeTensorMask(network);
  is_pruned_input_.resize(network->getNbInputs());
  absl::c_fill(is_pruned_input_, false);
  return Status::OK();
}

Status TrtShapeOptimizationProfile::ConfigureBuilder(
    nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
    const nvinfer1::INetworkDefinition* network) {
  TF_RETURN_IF_ERROR(AddProfiles(builder, config, network));
  return Status::OK();
}

// Sets the shape tensor mask from the TRT engine definition.
void TrtShapeOptimizationProfile::SetShapeTensorMask(
    const nvinfer1::ICudaEngine* engine, int n_inputs) {
  is_shape_tensor_.resize(n_inputs, false);
  for (int i = 0; i < n_inputs; i++) {
    int binding_index;
    Status status = GetTrtBindingIndex(i, 0, engine, &binding_index);
    if (!status.ok()) {
      continue;
    }
    is_shape_tensor_[i] = engine->isShapeBinding(binding_index);
    if (is_shape_tensor_[i]) {
      VLOG(2) << "Found shape tensor at " << i;
    }
  }
  has_shape_tensor_ =
      absl::c_any_of(is_shape_tensor_, [](bool b) { return b; });
}

// Sets the shape tensor mask using the network definition.
void TrtShapeOptimizationProfile::SetShapeTensorMask(
    const nvinfer1::INetworkDefinition* network) {
  int n_inputs = network->getNbInputs();
  is_shape_tensor_.resize(n_inputs, false);
  for (int i = 0; i < n_inputs; i++) {
    const ITensorProxyPtr input = network->getInput(i);
    is_shape_tensor_[i] = input->isShapeTensor();
    if (is_shape_tensor_[i]) {
      VLOG(2) << "Found shape tensor " << input->getName() << " at " << i;
    }
  }
  has_shape_tensor_ =
      absl::c_any_of(is_shape_tensor_, [](bool b) { return b; });
}

// Sets the shape tensor mask using the input partial shapes. This only tells
// whether the tensors are shape value compatible, only the final network
// definition or the engine would give concrete answers.
void TrtShapeOptimizationProfile::SetShapeTensorMask(
    const std::vector<PartialTensorShape>& input_partial_shapes) {
  if (is_shape_tensor_.size() == input_partial_shapes.size()) {
    // Already initialized, e.g. by TRTEngineOp::ComputeAsync().
    return;
  }
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
  tensorflow::profiler::TraceMe activity(
      "TrtShapeOptimizationProfile::GetProfileNumber",
      tensorflow::profiler::TraceMeLevel::kInfo);
  if (!need_profiles_) return 0;
  // TODO(tfeher): Return the best profile not just the first compatible.
  for (int i = 0; i < profiles_.size(); i++) {
    if (profiles_[i].IncludesShapes(shapes, HasShapeTensor(),
                                    actual_shape_values_, is_pruned_input_)) {
      return i;
    }
  }
  VLOG(1) << "Profile not found for input shapes " << DebugString(shapes)
          << ".";
  return -1;
}

Status TrtShapeOptimizationProfile::CreateExecutionContexts(
    nvinfer1::ICudaEngine* engine,
    std::vector<ExecutionContext>* exec_contexts) {
  int i = 0;
  // The following loop runs once if we have static shapes, to create a single
  // execution context without profiles. In dynamic mode we create one context
  // for each profile and set the corresponding optimization profile.
  do {
    VLOG(1) << "Creating execution context " << i;
    ExecutionContext context = ExecutionContext::Create(engine);
    if (i > 0) {
      // This condition is needed for two reasons:
      // - using static shapes we do not have any profiles so we cannot call
      //   set optimizationprofiles.
      // - The 0th profile is set implicitly for the first execution context
      //   therefore we do not need to set.
      if (!context->setOptimizationProfile(i)) {
        return errors::Internal("Could not set TRT optimization profile.");
      }
    }
    exec_contexts->push_back(std::move(context));
    i++;
  } while (i < profiles_.size());

  return Status::OK();
}

Status TrtShapeOptimizationProfile::SetInputShapeBinding(
    int input_index, int binding_index, nvinfer1::ICudaEngine* cuda_engine,
    nvinfer1::IExecutionContext* exec_context) const {
  tensorflow::profiler::TraceMe activity(
      "TrtShapeOptimizationProfile::SetInputShapeBinding",
      tensorflow::profiler::TraceMeLevel::kInfo);
  if (cuda_engine->isShapeBinding(binding_index)) {
    // Input shape binding data has to be in host memory. That is the reason
    // we can't use input_tensor.flat().data(). which contains the same
    // values in device memory. Instead, we use data that was copied to host
    // by CollectShapeValues.
    VLOG(2) << "Setting input shape binding for idx " << binding_index
            << ", with values "
            << DebugString(actual_shape_values_.at(input_index));
    bool ret = exec_context->setInputShapeBinding(
        binding_index, actual_shape_values_.at(input_index).d);
    if (!ret) {
      return errors::Internal("Could not set input shape binding for idx ",
                              binding_index);
    }
  }
  return Status::OK();
}

// If binding_idx is a shape tensor, then returns the associated min/max/opt
// shape values from prof_idx.
nvinfer1::Dims GetDimsFromShapeVal(int prof_idx, int binding_idx,
                                   nvinfer1::OptProfileSelector selector,
                                   const nvinfer1::ICudaEngine* engine) {
  if (engine->isShapeBinding(binding_idx)) {
    const int32* shape_val_ptr =
        engine->getProfileShapeValues(binding_idx, prof_idx, selector);
    if (shape_val_ptr) {
      VLOG(2) << "Found shape value in prof " << prof_idx << ", binding "
              << binding_idx;
      nvinfer1::Dims dims = engine->getBindingDimensions(binding_idx);
      // nbDims == 0 represent scalar, -1 represents invalid dim
      int n_values = (dims.nbDims == 0) ? 1 : dims.d[0];
      if (n_values > 0) {
        dims.nbDims = n_values;
        std::copy(shape_val_ptr, shape_val_ptr + n_values, dims.d);
      }
      return dims;
    }
  }
  return {0, {0}};
}

Status TrtShapeOptimizationProfile::SetPrunedMask(
    const nvinfer1::ICudaEngine* engine, int n_network_inputs) {
  is_pruned_input_.resize(n_network_inputs);
  absl::c_fill(is_pruned_input_, false);
  for (int j = 0; j < n_network_inputs; j++) {
    int binding_idx;
    Status status = GetTrtBindingIndex(j, 0, engine, &binding_idx);
    if (!status.ok()) {
      // Before TRT 8, an input tensor can be pruned (nvbugs/3153064)
      // Resource inputs are also unknown by TRT, so we can treat them as
      // pruned (the engine includes the variable as weights).
      is_pruned_input_[j] = true;
      VLOG(2) << "Skipping pruned input " << j;
      continue;
    }
  }
  return Status::OK();
}

Status TrtShapeOptimizationProfile::RestoreProfiles(
    const nvinfer1::ICudaEngine* engine, int n_network_inputs) {
  need_profiles_ = false;
  if (!engine) {
    // We do not need to restore profiles for an empty engine.
    return Status::OK();
  }
  if (engine->hasImplicitBatchDimension()) {
    // Nothing to do, we cannot have profiles in implicit batch mode.
    return Status::OK();
  }
  int n_profiles = engine->getNbOptimizationProfiles();
  need_profiles_ = n_profiles > 0;
  int n_inputs = GetNumberOfEngineInputs(engine);
  if (n_inputs > n_network_inputs) {
    return errors::Internal("Incorrect number of engine inputs");
  }
  VLOG(2) << "Attempting to restore " << n_profiles << " profiles, each with "
          << n_inputs << " inputs";
  SetShapeTensorMask(engine, n_network_inputs);

  TF_RETURN_IF_ERROR(SetPrunedMask(engine, n_network_inputs));

  for (int prof_idx = 0; prof_idx < n_profiles; prof_idx++) {
    OptimizationProfileConfig cfg;

    cfg.min.resize(n_network_inputs * 2);
    cfg.max.resize(n_network_inputs * 2);
    cfg.opt.resize(n_network_inputs * 2);
    // restore shape values
    for (int j = 0; j < n_network_inputs; j++) {
      if (is_pruned_input_[j]) continue;
      int binding_idx;
      TF_RETURN_IF_ERROR(GetTrtBindingIndex(j, 0, engine, &binding_idx));

      nvinfer1::Dims min = engine->getProfileDimensions(
          binding_idx, prof_idx, nvinfer1::OptProfileSelector::kMIN);
      nvinfer1::Dims max = engine->getProfileDimensions(
          binding_idx, prof_idx, nvinfer1::OptProfileSelector::kMAX);
      nvinfer1::Dims opt = engine->getProfileDimensions(
          binding_idx, prof_idx, nvinfer1::OptProfileSelector::kOPT);
      cfg.min[j] = min;
      cfg.max[j] = max;
      cfg.opt[j] = opt;

      cfg.min[j + n_inputs] = GetDimsFromShapeVal(
          prof_idx, binding_idx, nvinfer1::OptProfileSelector::kMIN, engine);
      cfg.max[j + n_inputs] = GetDimsFromShapeVal(
          prof_idx, binding_idx, nvinfer1::OptProfileSelector::kMAX, engine);
      cfg.opt[j + n_inputs] = GetDimsFromShapeVal(
          prof_idx, binding_idx, nvinfer1::OptProfileSelector::kOPT, engine);
    }
    VLOG(2) << "Restored profile " << cfg.DebugString();
    profiles_.push_back(std::move(cfg));
  }
  return Status::OK();
}

int TrtShapeOptimizationProfile::GetNumProfiles() const {
  return profiles_.size();
}

}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
