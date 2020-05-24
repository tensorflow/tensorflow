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
#import <CoreML/CoreML.h>
#import <UIKit/UIKit.h>

#include <string>
#include <vector>

#include "external/coremltools/mlmodel/format/Model.pb.h"

// Data for input/output tensors.
struct TensorData {
  std::vector<float> data;
  const std::string name;
  std::vector<int> shape;  // only required for input tensor.
};

// Responsible for:
// - Compiling and constructing MLModel from a serialized MlModel
//   protocol buffer.
// - Invoking predictions on the built model.
// Usage: Construct object, call Build() and Invoke() for inference.
@interface CoreMlExecutor : NSObject

- (bool)invokeWithInputs:(const std::vector<TensorData>&)inputs
                 outputs:(const std::vector<TensorData>&)outputs API_AVAILABLE(ios(11));

- (NSURL*)saveModel:(CoreML::Specification::Model*)model API_AVAILABLE(ios(11));
- (bool)build:(NSURL*)modelUrl API_AVAILABLE(ios(11));

- (bool)cleanup;

@property MLModel* model API_AVAILABLE(ios(11));
@property NSString* mlModelFilePath;
@property NSString* compiledModelFilePath;
@property(nonatomic, readonly) int coreMlVersion;
@end
