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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_SHAPE_OPTIMIZATION_PROFILES_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_SHAPE_OPTIMIZATION_PROFILES_H_

#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"


#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

#include "third_party/tensorrt/NvInfer.h"


namespace tensorflow {
namespace tensorrt {


// Stores optimization profile parameters (min/opt/max of each input shape)
//
// A TensorRT optimization profile describes the possible min/max values of
// each dynamic input shape along with an optimum value. These values are used
// by the TensorRT builder to select the best kernel for the optimum value among
// those kernels that are valid for all input tensors in the [min, max] range.
struct OptimizationProfileConfig {
  // Length of vector == num_inputs to engine
  std::vector<nvinfer1::Dims> min;
  std::vector<nvinfer1::Dims> opt;
  std::vector<nvinfer1::Dims> max;

  string DebugString() const {
    using absl::StrCat;
    return StrCat("[min: ", tensorflow::tensorrt::DebugString(min),
                  ", opt: : ", tensorflow::tensorrt::DebugString(opt),
                  ", max: ", tensorflow::tensorrt::DebugString(max), "]");
  }

  // Set the stored min/opt/max dimensions for profile
  //
  // Parameters:
  // network - TensorRT network, used to enumerate all the input tensors
  // profile - on exit the profile information will be set for each input tensor
  Status setDimensions(const nvinfer1::INetworkDefinition *network,
                       nvinfer1::IOptimizationProfile *profile) const {
    int n_inputs = network->getNbInputs();
    if (min.size()!=n_inputs || opt.size()!=n_inputs || max.size()!=n_inputs) {
      return errors::Internal("Incorrect number of profile config parameters");
    }
    for (int i = 0; i < n_inputs; i++) {
      const char *name = network->getInput(i)->getName();
      profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, min[i]);
      profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, opt[i]);
      profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, max[i]);
    }
    return Status::OK();
  }

  // Returns true if profile range completely includes the given shapes.
  bool IncludesShapes(const std::vector<TensorShape>& shapes) const {
    // min, max, and opt must have the same size which,
    // already verified in SetDimensions.
    if (min.size() != shapes.size()) {
      return false;
    }
    for (int i = 0; i < shapes.size(); i++) {
       auto current_shape = shapes[i];
       auto trt_shape = TensorShapeToTrtDims(current_shape, false);
       // min, max, and opt must have the same nbDims, which is
       // already verified in SetDimensions.
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
// build/profile_generation_mode phase, and using them it creates a set
// of OptimizationProfileConfigs. These configs will be added to
// IBuilderConfig before the engine is created.
//
class TrtShapeOptimizationProfile {
 public:
  TrtShapeOptimizationProfile() {};

  // Stores input shape information during profile_generation_mode
  void addShape(std::vector<TensorShape> shapes) {
    input_shapes_.insert(shapes);
    VLOG(1) << "Collected shape(s) " << DebugString(shapes) << " for profiles.";
  }

  void addShapeIfEmpty(std::vector<TensorShape> shapes) {
    if (input_shapes_.size() == 0) {
      addShape(shapes);
      VLOG(1) << "Collected shape(s) " << DebugString(shapes) << " for profiles.";
    }
  }
  void clear() { profiles_.clear(); }

  // Returns the profile number that should be used to execute the network
  // with the given input shapes. Returns -1 if none of cached profiles
  // is compatible with the given input shapes.
  int getProfileNumber(std::vector<TensorShape> shapes);

  // Creates optimization profiles and add them to the builder config.
  //
  // Parameters:
  // builder -
  // config -
  // network -
  Status configureBuilder(
    nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
    const nvinfer1::INetworkDefinition* network);

  // Creates execution contexts for each optimization profile.
  //
  // Parameters:
  // engine - cuda engine
  // exec_context - we append one execution context for each element in profile_
  Status createExcecutionContexts(
    nvinfer1::ICudaEngine* engine,
    std::vector<TrtUniquePtrType<nvinfer1::IExecutionContext>>& exec_context);

  /// Map input vector shapes to TRT Optimization profiles (min, max, opt)
  // i.e. maps input_shapes_ to profiles_
  void initProfiles();

  // Returns number of created profiles.
  int GetNumProfiles() const;

 private:
  // Set of input shape vetors that we collect during profile_generation_mode
  std::unordered_set<std::vector<TensorShape>, VectorTensorShapeHasher>
      input_shapes_;

  // The optimization profiles generated from input_shapes_
  std::vector<OptimizationProfileConfig> profiles_;

  /// Add optimization profiles to the builder config
  Status addProfiles(nvinfer1::IBuilder *builder,
                     nvinfer1::IBuilderConfig* config,
                     const nvinfer1::INetworkDefinition *network);

  // Map shapes to profile number, which is also a context number, because
  // each context has only one associated profile
  std::unordered_map<std::vector<TensorShape>, int, VectorTensorShapeHasher>
      profile_map_;
};

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_SHAPE_OPTIMIZATION_PROFILES_H_
