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

#include <string>

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
static ComputeTaskDescriptorPtr MulLinkable() {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->is_linkable = true;
  desc->shader_source = R"(FLT4 linkable$0(FLT4 value, int linear_index, uint3 gid) {
    return value * 1.1f;
  })";
  desc->AddSrcTensor("", {});
  desc->AddDstTensor("", {});
  return desc;
}

// This is an example of simple non-linkable operation performing add with a constant.
static ComputeTaskDescriptorPtr Add() {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
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

  desc->AddSrcTensor("input_buffer", {});
  desc->AddDstTensor("output_buffer", {});

  desc->uniform_buffers = {
      {"constant int2& size",
       [](const std::vector<BHWC>& src_shapes,
          const std::vector<BHWC>& dst_shapes) {
         std::vector<uint8_t> data;
         const int temp[] = {src_shapes[0].w, src_shapes[0].h};
         data.insert(data.begin(), reinterpret_cast<const uint8_t*>(temp),
                     reinterpret_cast<const uint8_t*>(temp) + sizeof(temp));
         return data;
       }},
  };

  desc->resize_function = [](const std::vector<BHWC>& src_shapes,
                             const std::vector<BHWC>& dst_shapes) {
    uint3 groups_size{16, 16, 1};
    uint3 groups_count{AlignByN(dst_shapes[0].w, groups_size.x),
                       AlignByN(dst_shapes[0].h, groups_size.y),
                       AlignByN(dst_shapes[0].c, 4)};
    return std::make_pair(groups_size, groups_count);
  };

  return desc;
}

// This is an example of simple linkable operation performing multiplication by a uniform
static ComputeTaskDescriptorPtr AddUniformLinkable(
    const std::vector<float>& channel_multipliers) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->is_linkable = true;
  desc->shader_source = R"(FLT4 linkable$0(FLT4 value, int linear_index, uint3 gid, FLT4 multiplier)
  {
      return value + multiplier;
  })";
  desc->AddSrcTensor("", {});
  desc->AddDstTensor("", {});
  desc->uniform_buffers = {
      {"constant FLT4&",
       [channel_multipliers](const std::vector<BHWC>& src_shapes,
                             const std::vector<BHWC>& dst_shapes) {
         return GetByteBuffer(channel_multipliers);
       }},
  };
  return desc;
}

// This is an example of simple linkable operation performing multiplication by a constant.
static ComputeTaskDescriptorPtr MulArrayLinkable(
    const std::vector<float>& channel_multipliers) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->is_linkable = true;
  desc->shader_source = R"(FLT4 linkable$0(FLT4 value, int linear_index, uint3 gid,
    device FLT4* const multiplier) {
      return value * multiplier[gid.z];
  })";
  desc->AddSrcTensor("", {});
  desc->AddDstTensor("", {});
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
  std::vector<tflite::gpu::metal::NodeDescriptor> nodes(2);
  nodes[0].task = Add();
  nodes[0].src_tensors_ids = {inputBufferID};
  nodes[0].dst_tensors_ids = {2};
  tflite::gpu::metal::NodeDescriptor node1;
  nodes[1].task = MulLinkable();
  nodes[1].src_tensors_ids = {2};
  nodes[1].dst_tensors_ids = {outputBufferID};
  TensorFloat32 input;
  input.shape = BHWC(1, 1, 1, 3);
  input.id = inputBufferID;
  input.data = {1, 2, 3};
  TensorFloat32 output;
  output.shape = BHWC(1, 1, 1, 3);
  std::map<ValueId, TensorFloat32> inputs{{inputBufferID, input}};
  std::map<ValueId, TensorFloat32> outputs{{outputBufferID, output}};
  auto status = RunGraph(nodes, _device, inputs, &outputs);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({2.2f, 3.3f, 4.4f}, outputs[outputBufferID].data, 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testImmutableShaderOutput {
  ValueId inputBufferID = 1;
  ValueId outputBufferID = 2;
  std::vector<tflite::gpu::metal::NodeDescriptor> nodes(1);
  nodes[0].task = MulArrayLinkable({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  nodes[0].src_tensors_ids = {inputBufferID};
  nodes[0].dst_tensors_ids = {outputBufferID};
  TensorFloat32 input;
  input.shape = BHWC(1, 1, 1, 7);
  input.id = inputBufferID;
  input.data = {1, 2, 3, 4, 5, 6, 7};
  TensorFloat32 output;
  output.shape = BHWC(1, 1, 1, 7);
  std::map<ValueId, TensorFloat32> inputs{{inputBufferID, input}};
  std::map<ValueId, TensorFloat32> outputs{{outputBufferID, output}};
  auto status = RunGraph(nodes, _device, inputs, &outputs);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({1, 4, 9, 16, 25, 36, 49}, outputs[outputBufferID].data, 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testUniformShaderOutput {
  ValueId inputBufferID = 1;
  ValueId outputBufferID = 2;
  std::vector<tflite::gpu::metal::NodeDescriptor> nodes(1);
  nodes[0].task = AddUniformLinkable({1.0f, 2.0f, 3.0f, 4.0f});
  nodes[0].src_tensors_ids = {inputBufferID};
  nodes[0].dst_tensors_ids = {outputBufferID};
  TensorFloat32 input;
  input.shape = BHWC(1, 1, 1, 3);
  input.id = inputBufferID;
  input.data = {1, 2, 3};
  TensorFloat32 output;
  output.shape = BHWC(1, 1, 1, 3);
  std::map<ValueId, TensorFloat32> inputs{{inputBufferID, input}};
  std::map<ValueId, TensorFloat32> outputs{{outputBufferID, output}};
  auto status = RunGraph(nodes, _device, inputs, &outputs);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({2, 4, 6}, outputs[outputBufferID].data, 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testUniformAndImmutableShaderOutput {
  ValueId inputBufferID = 1;
  ValueId outputBufferID = 3;
  std::vector<tflite::gpu::metal::NodeDescriptor> nodes(2);
  nodes[0].task = MulArrayLinkable({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  nodes[0].src_tensors_ids = {inputBufferID};
  nodes[0].dst_tensors_ids = {2};
  tflite::gpu::metal::NodeDescriptor node1;
  nodes[1].task = AddUniformLinkable({1.0f, 2.0f, 3.0f, 4.0f});
  nodes[1].src_tensors_ids = {2};
  nodes[1].dst_tensors_ids = {outputBufferID};
  TensorFloat32 input;
  input.shape = BHWC(1, 1, 1, 7);
  input.id = inputBufferID;
  input.data = {1, 2, 3, 4, 5, 6, 7};
  TensorFloat32 output;
  output.shape = BHWC(1, 1, 1, 7);
  std::map<ValueId, TensorFloat32> inputs{{inputBufferID, input}};
  std::map<ValueId, TensorFloat32> outputs{{outputBufferID, output}};
  auto status = RunGraph(nodes, _device, inputs, &outputs);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({2, 6, 12, 20, 26, 38, 52}, outputs[outputBufferID].data, 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

@end
