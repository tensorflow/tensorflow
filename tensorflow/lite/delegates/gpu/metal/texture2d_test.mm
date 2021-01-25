/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/metal/texture2d.h"

#import <Metal/Metal.h>
#import <XCTest/XCTest.h>

#include <iostream>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/types.h"

@interface Texture2DTest : XCTestCase
@end

@implementation Texture2DTest

using tflite::gpu::half;

- (void)testTexture2DF32 {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();

  const std::vector<float> data = {1.0, 2.0, 3.0, -4.0, 5.1, 6.7, 4.1, 6.17};
  tflite::gpu::metal::Texture2D texture;
  XCTAssertTrue(tflite::gpu::metal::CreateTexture2DRGBA32F(1, 2, device, &texture).ok());
  XCTAssertTrue(texture.WriteData(device, absl::MakeConstSpan(data.data(), data.size())).ok());
  std::vector<float> gpu_data;
  XCTAssertTrue(texture.ReadData<float>(device, &gpu_data).ok());

  XCTAssertEqual(gpu_data.size(), data.size());
  for (int i = 0; i < gpu_data.size(); ++i) {
    XCTAssertEqual(gpu_data[i], data[i]);
  }
}

- (void)testTexture2DF16 {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();

  const std::vector<half> data = {half(1.4),  half(2.1),  half(2.2), half(1.34),
                                  half(20.1), half(2.24), half(0.1), half(0.2)};

  tflite::gpu::metal::Texture2D texture;
  XCTAssertTrue(tflite::gpu::metal::CreateTexture2DRGBA16F(2, 1, device, &texture).ok());
  XCTAssertTrue(texture.WriteData(device, absl::MakeConstSpan(data.data(), data.size())).ok());
  std::vector<half> gpu_data;
  XCTAssertTrue(texture.ReadData<half>(device, &gpu_data).ok());

  XCTAssertEqual(gpu_data.size(), data.size());
  for (int i = 0; i < gpu_data.size(); ++i) {
    XCTAssertEqual(gpu_data[i], data[i]);
  }
}

@end
