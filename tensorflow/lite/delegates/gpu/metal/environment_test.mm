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

#include "tensorflow/lite/delegates/gpu/metal/environment.h"

#import <XCTest/XCTest.h>

#include "tensorflow/lite/delegates/gpu/metal/common.h"

using ::tflite::gpu::metal::GetGpuType;

@interface EnvironmentTest : XCTestCase

@end

@implementation EnvironmentTest

- (void)testCompileTimeOSDetection {
#if IOS_VERSION > 0
  XCTAssertTrue(MACOS_VERSION == 0 && TVOS_VERSION == 0, @"IOS_VERSION: %d", int{IOS_VERSION});
#endif
#if MACOS_VERSION > 0
  XCTAssertTrue(IOS_VERSION == 0 && TVOS_VERSION == 0, @"MACOS_VERSION: %d", int{MACOS_VERSION});
#endif
#if TVOS_VERSION > 0
  XCTAssertTrue(IOS_VERSION == 0 && MACOS_VERSION == 0, @"TVOS_VERSION: %d", int{TVOS_VERSION});
#endif
}

- (void)testGetGpuType {
#if (IOS_VERSION > 0) || (TVOS_VERSION > 0)
  auto gpuType = GetGpuType();
  XCTAssertTrue(gpuType != GpuType::kUnknown);
#endif
}

@end
