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

#include "tensorflow/lite/delegates/gpu/metal/buffer.h"

#include "tensorflow/lite/delegates/gpu/common/types.h"

#import <XCTest/XCTest.h>

#import <Metal/Metal.h>

#include <vector>
#include <iostream>

@interface BufferTest : XCTestCase
@end

@implementation BufferTest
- (void)setUp {
  [super setUp];
}

using tflite::gpu::half;

- (void)testBufferF32 {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();

  const std::vector<float> data = {1.0f, 2.0f, 3.0f, -4.0f, 5.1f};
  tflite::gpu::metal::Buffer buffer;
  XCTAssertTrue(tflite::gpu::metal::CreateBuffer(sizeof(float) * 5, nullptr, device, &buffer).ok());
  XCTAssertTrue(buffer.WriteData(absl::MakeConstSpan(data.data(), data.size())).ok());
  std::vector<float> gpu_data;
  XCTAssertTrue(buffer.ReadData<float>(&gpu_data).ok());

  XCTAssertEqual(gpu_data.size(), data.size());
  for (int i = 0; i < gpu_data.size(); ++i) {
    XCTAssertEqual(gpu_data[i], data[i]);
  }
}

- (void)testBufferF16 {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();

  const std::vector<half> data = {half(1.0f), half(2.0f), half(3.0f), half(-4.0f), half(5.1f)};
  tflite::gpu::metal::Buffer buffer;
  XCTAssertTrue(tflite::gpu::metal::CreateBuffer(
      sizeof(tflite::gpu::half) * 5, nullptr, device, &buffer).ok());
  XCTAssertTrue(buffer.WriteData(absl::MakeConstSpan(data.data(), data.size())).ok());
  std::vector<half> gpu_data;
  XCTAssertTrue(buffer.ReadData<half>(&gpu_data).ok());

  XCTAssertEqual(gpu_data.size(), data.size());
  for (int i = 0; i < gpu_data.size(); ++i) {
    XCTAssertEqual(gpu_data[i], data[i]);
  }
}

@end
