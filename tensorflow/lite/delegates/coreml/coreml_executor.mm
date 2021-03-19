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
#import "tensorflow/lite/delegates/coreml/coreml_executor.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <fstream>
#include <iostream>

namespace {
// Returns NSURL for a temporary file.
NSURL* createTemporaryFile() {
  // Get temporary directory.
  NSURL* temporaryDirectoryURL = [NSURL fileURLWithPath:NSTemporaryDirectory() isDirectory:YES];
  // Generate a Unique file name to use.
  NSString* temporaryFilename = [[NSProcessInfo processInfo] globallyUniqueString];
  // Create URL to that file.
  NSURL* temporaryFileURL = [temporaryDirectoryURL URLByAppendingPathComponent:temporaryFilename];

  return temporaryFileURL;
}
}  // namespace

@interface MultiArrayFeatureProvider : NSObject <MLFeatureProvider> {
  const std::vector<TensorData>* _inputs;
  NSSet* _featureNames;
}

- (instancetype)initWithInputs:(const std::vector<TensorData>*)inputs
                 coreMlVersion:(int)coreMlVersion;
- (MLFeatureValue*)featureValueForName:(NSString*)featureName API_AVAILABLE(ios(11));
- (NSSet<NSString*>*)featureNames;

@property(nonatomic, readonly) int coreMlVersion;

@end

@implementation MultiArrayFeatureProvider

- (instancetype)initWithInputs:(const std::vector<TensorData>*)inputs
                 coreMlVersion:(int)coreMlVersion {
  self = [super init];
  _inputs = inputs;
  _coreMlVersion = coreMlVersion;
  for (auto& input : *_inputs) {
    if (input.name.empty()) {
      return nil;
    }
  }
  return self;
}

- (NSSet<NSString*>*)featureNames {
  if (_featureNames == nil) {
    NSMutableArray* names = [[NSMutableArray alloc] init];
    for (auto& input : *_inputs) {
      [names addObject:[NSString stringWithCString:input.name.c_str()
                                          encoding:[NSString defaultCStringEncoding]]];
    }
    _featureNames = [NSSet setWithArray:names];
  }
  return _featureNames;
}

- (MLFeatureValue*)featureValueForName:(NSString*)featureName {
  for (auto& input : *_inputs) {
    if ([featureName cStringUsingEncoding:NSUTF8StringEncoding] == input.name) {
      // TODO(b/141492326): Update shape handling for higher ranks
      NSArray* shape = @[
        @(input.shape[0]),
        @(input.shape[1]),
        @(input.shape[2]),
      ];
      NSArray* strides = @[
        @(input.shape[1] * input.shape[2]),
        @(input.shape[2]),
        @1,
      ];

      if ([self coreMlVersion] >= 3) {
        shape = @[
          @(input.shape[0]),
          @(input.shape[1]),
          @(input.shape[2]),
          @(input.shape[3]),
        ];
        strides = @[
          @(input.shape[1] * input.shape[2] * input.shape[3]),
          @(input.shape[2] * input.shape[3]),
          @(input.shape[3]),
          @1,
        ];
      };
      NSError* error = nil;
      MLMultiArray* mlArray = [[MLMultiArray alloc] initWithDataPointer:(float*)input.data.data()
                                                                  shape:shape
                                                               dataType:MLMultiArrayDataTypeFloat32
                                                                strides:strides
                                                            deallocator:(^(void* bytes){
                                                                        })error:&error];
      if (error != nil) {
        NSLog(@"Failed to create MLMultiArray for feature %@ error: %@", featureName,
              [error localizedDescription]);
        return nil;
      }
      auto* mlFeatureValue = [MLFeatureValue featureValueWithMultiArray:mlArray];
      return mlFeatureValue;
    }
  }

  NSLog(@"Feature %@ not found", featureName);
  return nil;
}
@end

@implementation CoreMlExecutor
- (bool)invokeWithInputs:(const std::vector<TensorData>&)inputs
                 outputs:(const std::vector<TensorData>&)outputs {
  if (_model == nil) {
    return NO;
  }
  NSError* error = nil;
  MultiArrayFeatureProvider* inputFeature =
      [[MultiArrayFeatureProvider alloc] initWithInputs:&inputs coreMlVersion:[self coreMlVersion]];
  if (inputFeature == nil) {
    NSLog(@"inputFeature is not initialized.");
    return NO;
  }
  MLPredictionOptions* options = [[MLPredictionOptions alloc] init];
  id<MLFeatureProvider> outputFeature = [_model predictionFromFeatures:inputFeature
                                                               options:options
                                                                 error:&error];
  if (error != nil) {
    NSLog(@"Error executing model: %@", [error localizedDescription]);
    return NO;
  }
  NSSet<NSString*>* outputFeatureNames = [outputFeature featureNames];
  for (auto& output : outputs) {
    NSString* outputName = [NSString stringWithCString:output.name.c_str()
                                              encoding:[NSString defaultCStringEncoding]];
    MLFeatureValue* outputValue =
        [outputFeature featureValueForName:[outputFeatureNames member:outputName]];
    auto* data = [outputValue multiArrayValue];
    float* outputData = (float*)data.dataPointer;
    if (outputData == nullptr) {
      return NO;
    }
    memcpy((float*)output.data.data(), outputData, output.data.size() * sizeof(output.data[0]));
  }
  return YES;
}

- (bool)cleanup {
  NSError* error = nil;
  [[NSFileManager defaultManager] removeItemAtPath:_mlModelFilePath error:&error];
  if (error != nil) {
    NSLog(@"Failed cleaning up model: %@", [error localizedDescription]);
    return NO;
  }
  [[NSFileManager defaultManager] removeItemAtPath:_compiledModelFilePath error:&error];
  if (error != nil) {
    NSLog(@"Failed cleaning up compiled model: %@", [error localizedDescription]);
    return NO;
  }
  return YES;
}

- (NSURL*)saveModel:(CoreML::Specification::Model*)model {
  NSURL* modelUrl = createTemporaryFile();
  NSString* modelPath = [modelUrl path];
  if (model->specificationversion() == 3) {
    _coreMlVersion = 2;
  } else if (model->specificationversion() == 4) {
    _coreMlVersion = 3;
  } else {
    NSLog(@"Only Core ML models with specification version 3 or 4 are supported");
    return nil;
  }
  // Flush data to file.
  // TODO(karimnosseir): Can we mmap this instead of actual writing it to phone ?
  std::ofstream file_stream([modelPath UTF8String], std::ios::out | std::ios::binary);
  model->SerializeToOstream(&file_stream);
  return modelUrl;
}

- (bool)build:(NSURL*)modelUrl {
  NSError* error = nil;
  NSURL* compileUrl = [MLModel compileModelAtURL:modelUrl error:&error];
  if (error != nil) {
    NSLog(@"Error compiling model %@", [error localizedDescription]);
    return NO;
  }
  _mlModelFilePath = [modelUrl path];
  _compiledModelFilePath = [compileUrl path];

  if (@available(iOS 12.0, *)) {
    MLModelConfiguration* config = [MLModelConfiguration alloc];
    config.computeUnits = MLComputeUnitsAll;
    _model = [MLModel modelWithContentsOfURL:compileUrl configuration:config error:&error];
  } else {
    _model = [MLModel modelWithContentsOfURL:compileUrl error:&error];
  }
  if (error != NULL) {
    NSLog(@"Error Creating MLModel %@", [error localizedDescription]);
    return NO;
  }
  return YES;
}
@end
