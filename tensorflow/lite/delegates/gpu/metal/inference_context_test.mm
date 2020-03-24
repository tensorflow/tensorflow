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

#include "tensorflow/lite/delegates/gpu/metal/inference_context.h"

#import <XCTest/XCTest.h>

#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/common.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/test_util.h"

using ::tflite::gpu::AlignByN;
using ::tflite::gpu::BHWC;
using ::tflite::gpu::metal::CompareVectors;
using ::tflite::gpu::metal::ComputeTaskDescriptor;
using ::tflite::gpu::metal::ComputeTaskDescriptorPtr;
using ::tflite::gpu::metal::GetBestSupportedMetalDevice;
using ::tflite::gpu::metal::RunGraph;
using ::tflite::gpu::metal::GetByteBuffer;
using ::tflite::gpu::TensorFloat32;
using ::tflite::gpu::uint3;
using ::tflite::gpu::ValueId;

// This is an example of simple linkable operation performing multiplication by a constant.
static std::vector<ComputeTaskDescriptorPtr> MulLinkable(int id, ValueId input_id,
                                                         ValueId output_id) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = true;
  desc->shader_source = R"(FLT4 linkable$0(FLT4 value, int linear_index, uint3 gid) {
    return value * 1.1f;
  })";
  desc->input_buffers = {{input_id}};
  desc->output_buffer = {output_id};
  return {desc};
}

// This is an example of simple non-linkable operation performing add with a constant.
static std::vector<ComputeTaskDescriptorPtr> Add(int id, ValueId input_id, ValueId output_id) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  desc->shader_source = R"(
    #include <metal_stdlib>
    using namespace metal;
    $0
    kernel void ComputeFunction(
                                $1
                                uint3 gid[[thread_position_in_grid]]) {
      if (int(gid.x) >= size.x || int(gid.y) >= size.y) {
        return;
      }
      const int linear_index = (gid.z * size.y + gid.y) * size.x + gid.x;
      FLT4 value = input_buffer[linear_index] + 1.0f;
      $2
      output_buffer[linear_index] = value;
    }
  )";

  desc->input_buffers = {
      {input_id, "device FLT4* const input_buffer"},
  };

  desc->output_buffer = {output_id, "device FLT4* output_buffer",
                         [input_id, output_id](const std::map<ValueId, BHWC>& buffers) {
                           return buffers.find(input_id)->second;
                         }};

  desc->uniform_buffers = {
      {"constant int2& size",
       [output_id](const std::map<ValueId, BHWC>& buffers) {
         std::vector<uint8_t> data;
         const auto& dimension = buffers.find(output_id)->second;
         const int temp[] = {dimension.w, dimension.h};
         data.insert(data.begin(), reinterpret_cast<const uint8_t*>(temp),
                     reinterpret_cast<const uint8_t*>(temp) + sizeof(temp));
         return data;
       }},
  };

  desc->resize_function = [output_id](const std::map<ValueId, BHWC>& buffers) {
    const auto& dimension = buffers.find(output_id)->second;
    uint3 groups_size{16, 16, 1};
    uint3 groups_count{AlignByN(dimension.w, groups_size.x), AlignByN(dimension.h, groups_size.y),
                       AlignByN(dimension.c, 4)};
    return std::make_pair(groups_size, groups_count);
  };

  return {desc};
}

// This is an example of simple linkable operation performing multiplication by a uniform
static std::vector<ComputeTaskDescriptorPtr> AddUniformLinkable(
    int id, ValueId input_id, ValueId output_id, const std::vector<float>& channel_multipliers) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = true;
  desc->shader_source = R"(FLT4 linkable$0(FLT4 value, int linear_index, uint3 gid, FLT4 multiplier)
  {
      return value + multiplier;
  })";
  desc->input_buffers = {{input_id}};
  desc->output_buffer = {output_id};
  desc->uniform_buffers = {
      {"constant FLT4&",
       [channel_multipliers](const std::map<ValueId, BHWC>& buffers) {
         return GetByteBuffer(channel_multipliers);
       }},
  };
  return {desc};
}

// This is an example of simple linkable operation performing multiplication by a constant.
static std::vector<ComputeTaskDescriptorPtr> MulArrayLinkable(
    int id, ValueId input_id, ValueId output_id, const std::vector<float>& channel_multipliers) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = true;
  desc->shader_source = R"(FLT4 linkable$0(FLT4 value, int linear_index, uint3 gid,
    device FLT4* const multiplier) {
      return value * multiplier[gid.z];
  })";
  desc->input_buffers = {{input_id}};
  desc->output_buffer = {output_id};
  desc->immutable_buffers = {
      {"device FLT4* const", GetByteBuffer(channel_multipliers)},
  };
  return {desc};
}

@interface InferenceContextTest : XCTestCase {
  id<MTLDevice> _device;
}

@end

@implementation InferenceContextTest

- (void)setUp {
  [super setUp];
  _device = GetBestSupportedMetalDevice();
  XCTAssertNotNil(_device);
}

- (void)testTwoInputsShaderOutput {
  ValueId inputBufferID = 1;
  ValueId outputBufferID = 3;
  auto graph = Add(1, inputBufferID, 2);
  auto graph2 = MulLinkable(2, 2, outputBufferID);
  graph.insert(graph.end(), graph2.begin(), graph2.end());
  TensorFloat32 input;
  input.shape = BHWC(1, 1, 1, 3);
  input.id = inputBufferID;
  input.data = {1, 2, 3};
  std::map<ValueId, TensorFloat32> inputs{{inputBufferID, input}};
  std::map<ValueId, TensorFloat32> outputs{{outputBufferID, {}}};
  auto status = RunGraph(graph, _device, inputs, &outputs);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
  status = CompareVectors({2.2f, 3.3f, 4.4f}, outputs[outputBufferID].data, 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

- (void)testImmutableShaderOutput {
  ValueId inputBufferID = 1;
  ValueId outputBufferID = 2;
  auto graph = MulArrayLinkable(1, inputBufferID, outputBufferID,
                                {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  TensorFloat32 input;
  input.shape = BHWC(1, 1, 1, 7);
  input.id = inputBufferID;
  input.data = {1, 2, 3, 4, 5, 6, 7};
  std::map<ValueId, TensorFloat32> inputs{{inputBufferID, input}};
  std::map<ValueId, TensorFloat32> outputs{{outputBufferID, {}}};
  auto status = RunGraph(graph, _device, inputs, &outputs);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
  status = CompareVectors({1, 4, 9, 16, 25, 36, 49}, outputs[outputBufferID].data, 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

- (void)testUniformShaderOutput {
  ValueId inputBufferID = 1;
  ValueId outputBufferID = 2;
  auto graph = AddUniformLinkable(1, inputBufferID, outputBufferID, {1.0f, 2.0f, 3.0f, 4.0f});
  TensorFloat32 input;
  input.shape = BHWC(1, 1, 1, 3);
  input.id = inputBufferID;
  input.data = {1, 2, 3};
  std::map<ValueId, TensorFloat32> inputs{{inputBufferID, input}};
  std::map<ValueId, TensorFloat32> outputs{{outputBufferID, {}}};
  auto status = RunGraph(graph, _device, inputs, &outputs);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
  status = CompareVectors({2, 4, 6}, outputs[outputBufferID].data, 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

- (void)testUniformAndImmutableShaderOutput {
  ValueId inputBufferID = 1;
  ValueId outputBufferID = 3;
  auto graph =
      MulArrayLinkable(1, inputBufferID, 2, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  auto graph2 = AddUniformLinkable(2, 2, outputBufferID, {1.0f, 2.0f, 3.0f, 4.0f});
  graph.insert(graph.end(), graph2.begin(), graph2.end());
  TensorFloat32 input;
  input.shape = BHWC(1, 1, 1, 7);
  input.id = inputBufferID;
  input.data = {1, 2, 3, 4, 5, 6, 7};
  std::map<ValueId, TensorFloat32> inputs{{inputBufferID, input}};
  std::map<ValueId, TensorFloat32> outputs{{outputBufferID, {}}};
  auto status = RunGraph(graph, _device, inputs, &outputs);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
  status = CompareVectors({2, 6, 12, 20, 26, 38, 52}, outputs[outputBufferID].data, 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

@end
