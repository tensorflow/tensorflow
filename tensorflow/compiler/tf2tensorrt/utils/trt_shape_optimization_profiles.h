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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TRT_SHAPE_OPTIMIZATION_PROFILES_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TRT_SHAPE_OPTIMIZATION_PROFILES_H_

#include <list>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/compiler/tf2tensorrt/common/datavec.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_execution_context.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

// Stores optimization profile parameters (min/opt/max of each input shape).
//
// A TensorRT optimization profile describes the possible min/max values of
// each dynamic input shape along with an optimum value. These values are used
// by the TensorRT builder to select the best kernel for the optimum value among
// those kernels that are valid for all input tensors in the [min, max] range.
struct OptimizationProfileConfig {
  // Length of vector == 2*num_inputs to engine. min[0:num_inputs-1] are the min
  // input dimensions for execution tensors. If engine has shape input tensors,
  // then min[num_inputs + i] store the shape value for input i. For inputs that
  // are not shape tensors min = opt = max = {0, {}}.
  std::vector<nvinfer1::Dims> min;
  std::vector<nvinfer1::Dims> opt;
  std::vector<nvinfer1::Dims> max;

  string DebugString() const {
    using absl::StrCat;
    return StrCat("[min: ", tensorflow::tensorrt::DebugString(min),
                  ", opt: : ", tensorflow::tensorrt::DebugString(opt),
                  ", max: ", tensorflow::tensorrt::DebugString(max), "]");
  }

#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  // Sets the min/opt/max dimensions for profile.
  //
  // The given min/opt/max dimensions should satisfy the condition
  // min <= opt <= max. Additionally TRT requires that the min/opt/max values
  // are compatible with the network input. Compatibility is defined the
  // following way: let dim be the shape of an input binding and min/opt/max the
  // corresponding profile dims. TRT requires that dim.d[k] must be -1 if
  // (min.d[k] != dim.d[k] || opt.d[k] != dim.d[k] || max.d[k] != dim.d[k]).
  //
  // Parameters:
  // network - TensorRT network, used to enumerate all the input tensors
  // profile - on exit the profile information will be set for each input tensor
  Status SetDimensions(const nvinfer1::INetworkDefinition* network,
                       nvinfer1::IOptimizationProfile* profile) const {
    int n_inputs = network->getNbInputs();
    if (min.size() != 2 * n_inputs || opt.size() != 2 * n_inputs ||
        max.size() != 2 * n_inputs) {
      return errors::Internal("Incorrect number of profile config parameters");
    }
    for (int i = 0; i < n_inputs; i++) {
      const nvinfer1::ITensor* input = network->getInput(i);
      const char* name = input->getName();
      if (input->isShapeTensor()) {
        int idx = i + n_inputs;
        VLOG(2) << "Setting shape values for " << name << ", "
                << ::tensorflow::tensorrt::DebugString(opt[idx]);
        profile->setShapeValues(name, nvinfer1::OptProfileSelector::kMIN,
                                min[idx].d, min[idx].nbDims);
        profile->setShapeValues(name, nvinfer1::OptProfileSelector::kOPT,
                                opt[idx].d, opt[idx].nbDims);
        profile->setShapeValues(name, nvinfer1::OptProfileSelector::kMAX,
                                max[idx].d, max[idx].nbDims);
      }
      if (input->isExecutionTensor()) {
        VLOG(2) << "Setting input dimensions for " << name << ", "
                << ::tensorflow::tensorrt::DebugString(opt[i]);
        profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN,
                               min[i]);
        profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT,
                               opt[i]);
        profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX,
                               max[i]);
      }
    }
    return Status::OK();
  }
#endif

  // Returns true if profile range completely includes the given shapes.
  bool IncludesShapes(const std::vector<TensorShape>& shapes,
                      bool has_shape_tensor,
                      const std::vector<nvinfer1::Dims>& shape_values) const {
    // min, max, and opt must have the same size which is already verified in
    // SetDimensions.
    if (min.size() != shapes.size() * 2 ||
        (has_shape_tensor && min.size() != shape_values.size() * 2)) {
      VLOG(2) << "Profile size mismatch min size " << min.size()
              << " vs input shapes size " << shapes.size() << " "
              << shape_values.size();
      return false;
    }
    for (int i = 0; i < shapes.size(); i++) {
      auto current_shape = shapes[i];
      // min, max, and opt must have the same nbDims, which is already verified
      // in SetDimensions.
      if (min[i].nbDims != current_shape.dims()) {
        return false;
      }
      // Check if range [min, max] includes current_shape.
      for (int dim = 0; dim < current_shape.dims(); dim++) {
        if ((min[i].d[dim] > current_shape.dim_size(dim)) ||
            (max[i].d[dim] < current_shape.dim_size(dim))) {
          return false;
        }
      }
    }
    // Check shape values.
    if (has_shape_tensor) {
      int offset = shapes.size();
      for (int i = 0; i < shape_values.size(); i++) {
        auto shape_val = shape_values[i];
        // min, max, and opt must have the same nbDims, which is already
        // verified in SetDimensions.
        if (min[i + offset].nbDims != shape_val.nbDims) {
          return false;
        }
        // Check if range [min, max] includes shape_val.
        for (int dim = 0; dim < shape_val.nbDims; dim++) {
          if (min[i + offset].d[dim] > shape_val.d[dim] ||
              max[i + offset].d[dim] < shape_val.d[dim]) {
            return false;
          }
        }
      }
    }
    return true;
  }
};

// Manages Optimization profiles during TRT Engine construction.
//
// An optimization profile describes a range of dimensions for each TRT network
// input, and the optimal dimensions that the auto-tuner should use for
// optimization.
//
// This class stores the list of input shapes that were seen during the
// build/profile_generation_mode phase, and using them it creates a set of
// OptimizationProfileConfigs. These configs will be added to IBuilderConfig
// before the engine is created.
class TrtShapeOptimizationProfile {
 public:
  TrtShapeOptimizationProfile(
      ProfileStrategy strategy = ProfileStrategy::kImplicitBatchModeCompatible)
      : strategy_(strategy) {}

  // Stores input shape information during profile_generation_mode.
  void AddShape(const std::vector<TensorShape>& shapes) {
    input_shapes_.push_back(shapes);
    input_shape_values_.push_back(actual_shape_values_);
    VLOG(1) << "Collected shape(s) " << DebugString(shapes) << " for profiles.";
  }

  // Collects ShapeTensorCompatible tensor values. This is needed both during
  // profile_generation_mode and during normal inference calls.
  Status CollectShapeValues(OpKernelContext* ctx);

  // Collects ShapeTensorCompatible tensor values, used only for unit tests.
  Status CollectShapeValues(const DataVec& input);

  void clear() { profiles_.clear(); }

  // Returns the profile number that should be used to execute the network with
  // the given input shapes. Returns -1 if none of cached profiles are
  // compatible with the given input shapes.
  int GetProfileNumber(const std::vector<TensorShape>& shapes);

#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  // Creates optimization profiles and add them to the builder config.
  Status ConfigureBuilder(nvinfer1::IBuilder* builder,
                          nvinfer1::IBuilderConfig* config,
                          const nvinfer1::INetworkDefinition* network);
#endif

  // Creates execution contexts for each optimization profile.
  Status CreateExecutionContexts(nvinfer1::ICudaEngine* engine,
                                 std::vector<ExecutionContext>& exec_context,
                                 TRTBaseAllocator* memory_allocator);

  Status SetInputShapeBinding(int input_index, int binding_index,
                              nvinfer1::ICudaEngine* cuda_engine,
                              nvinfer1::IExecutionContext* exec_context) const;

  // Creates optimization profiles profiles_ for the set of concrete input
  // shapes collected in input_shapes_. The input_partial_shapes of the network
  // is used to ensure that the created optimization profiles are compatible
  // with the network.
  void InitProfiles(
      const std::vector<PartialTensorShape>& input_partial_shapes);

  // Returns number of created profiles.
  int GetNumProfiles() const;

  bool HasShape() const { return !input_shapes_.empty(); }
  bool NeedProfiles() const { return need_profiles_; }

  // Restores profiles from the engine (used after deserialization).
  Status RestoreProfiles(const nvinfer1::ICudaEngine* engine);

  // Whether the network has any shape tensors.
  bool HasShapeTensor() const { return has_shape_tensor_; }

  void SetShapeTensorMask(const nvinfer1::INetworkDefinition* network);

 private:
  // Set of input shape vetors that we collect during profile_generation_mode.
  std::vector<std::vector<TensorShape>> input_shapes_;

  // Input shape values that we collect during profile_generation_mode. If the
  // tensor is not compatible with a TRT shape tensor then an empty shape is
  // stored.
  std::vector<std::vector<nvinfer1::Dims>> input_shape_values_;

  // Shape values present in the current inference call.
  std::vector<nvinfer1::Dims> actual_shape_values_;

  // The optimization profiles generated from input_shapes_.
  std::vector<OptimizationProfileConfig> profiles_;

  // Whether the network has any shape tensors. Initially we assume that the
  // network might have a shape value input. This will be updated when the
  // network is created / engine is deserialized.
  bool has_shape_tensor_ = true;

  // Whether the network/engine requires optimization profiles.
  bool need_profiles_ = false;

  // Whether an input tensor is a shape tensor.
  std::vector<bool> is_shape_tensor_;

  // Optimization profile generation strategy.
  ProfileStrategy strategy_;

#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  // Adds optimization profiles to the builder config.
  Status AddProfiles(nvinfer1::IBuilder* builder,
                     nvinfer1::IBuilderConfig* config,
                     const nvinfer1::INetworkDefinition* network);
#endif

  void SetShapeTensorMask(const nvinfer1::ICudaEngine* engine, int n_inputs);
  void SetShapeTensorMask(
      const std::vector<PartialTensorShape>& input_partial_shapes);

  void ImplicitBatchModeCompatibleStrategy(
      const std::vector<std::vector<nvinfer1::Dims>>& collected_shapes);
  void OptimalStrategy(
      const std::vector<std::vector<nvinfer1::Dims>>& collected_shapes);
};

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TRT_SHAPE_OPTIMIZATION_PROFILES_H_
