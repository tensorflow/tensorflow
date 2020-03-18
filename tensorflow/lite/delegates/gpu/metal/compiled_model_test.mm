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

#include "tensorflow/lite/delegates/gpu/metal/compiled_model.h"

#import <XCTest/XCTest.h>

#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

using ::tflite::gpu::AlignByN;
using ::tflite::gpu::BHWC;
using ::tflite::gpu::metal::ComputeTaskDescriptor;
using ::tflite::gpu::metal::ComputeTaskDescriptorPtr;
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

// An example of linkable operation performing summing of two tensors.
static std::vector<ComputeTaskDescriptorPtr> Add2(int id, ValueId input_id1, ValueId input_id2,
                                                  ValueId output_id) {
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
      FLT4 value = input_buffer1[linear_index] + input_buffer2[linear_index];
      $2
      output_buffer[linear_index] = value;
    }
  )";

  desc->input_buffers = {
      {input_id1, "device FLT4* const input_buffer1"},
      {input_id2, "device FLT4* const input_buffer2"},
  };

  desc->output_buffer = {output_id, "device FLT4* output_buffer",
                         [input_id1](const std::map<ValueId, BHWC>& buffers) {
                           return buffers.find(input_id1)->second;
                         }};

  desc->uniform_buffers = {
      {"constant int2& size",
       [input_id1](const std::map<ValueId, BHWC>& buffers) {
         std::vector<uint8_t> data;
         const auto& dimension = buffers.find(input_id1)->second;
         const int temp[] = {dimension.w, dimension.h};
         data.insert(data.begin(), reinterpret_cast<const uint8_t*>(temp),
                     reinterpret_cast<const uint8_t*>(temp) + sizeof(temp));
         return data;
       }},
  };

  desc->resize_function = [input_id1](const std::map<ValueId, BHWC>& buffers) {
    const auto& dimension = buffers.find(input_id1)->second;
    uint3 groups_size{16, 16, 1};
    uint3 groups_count{AlignByN(dimension.w, groups_size.x), AlignByN(dimension.h, groups_size.y),
                       AlignByN(dimension.c, 4)};
    return std::make_pair(groups_size, groups_count);
  };

  return {desc};
}

// An example of linkable operation performing summing of two tensors.
static std::vector<ComputeTaskDescriptorPtr> Add2Linkable(int id, ValueId input_id1,
                                                          ValueId input_id2, ValueId output_id) {
  std::vector<ComputeTaskDescriptorPtr> descriptors;
  descriptors.push_back(ComputeTaskDescriptorPtr(new ComputeTaskDescriptor({
      id,
      true,  // Is linkable?
      R"(FLT4 linkable$0(FLT4 value, int linear_index, uint3 gid, device FLT4* const buffer2) {
           return value + buffer2[linear_index];
         }
      )",
      {
          {input_id1, "device FLT4* const"},
          {input_id2, "device FLT4* const"},
      },
      {output_id},
  })));
  return descriptors;
}

@interface CompiledModelTest : XCTestCase

@end

@implementation CompiledModelTest

- (void)testSingleOperationSuccess {
  auto nodes = MulLinkable(1, 1, 2);
  std::vector<ComputeTaskDescriptorPtr> model;
  auto status = ValidateOptimizeModel({1}, {2}, nodes, &model);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

// Outputs: one missing, one unused.
- (void)testSingleOperationErrorWrongOutput {
  auto nodes = MulLinkable(1, 1, 2);
  std::vector<ComputeTaskDescriptorPtr> model;
  auto status = ValidateOptimizeModel({1}, {3}, nodes, &model);
  XCTAssertFalse(status.ok());
  std::vector<std::string> errorMessages = {"Input operations count 1", "Unused operations 1",
                                            "Unused inputs 1", "Missing output buffers 1"};
  for (const std::string& message : errorMessages) {
    bool doesContainMessage = status.error_message().find(message) != std::string::npos;
    XCTAssertTrue(doesContainMessage, @"%s", status.error_message().c_str());
  }
}

// Outputs: one ok, one missing.
- (void)testSingleOperationWarningExtraOutput {
  auto nodes = MulLinkable(1, 1, 2);
  std::vector<ComputeTaskDescriptorPtr> model;
  auto status = ValidateOptimizeModel({1}, {2, 3}, nodes, &model);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

// Unused input => empty graph, missing output.
- (void)testSingleOperationErrorWrongInput {
  auto nodes = MulLinkable(1, 1, 2);
  std::vector<ComputeTaskDescriptorPtr> model;
  auto status = ValidateOptimizeModel({3}, {2}, nodes, &model);
  std::vector<std::string> errorMessages = {"Input operations count 1", "Unused operations 0",
                                            "Unused inputs 1", "Missing output buffers 1"};
  for (const std::string& message : errorMessages) {
    bool doesContainMessage = status.error_message().find(message) != std::string::npos;
    XCTAssertTrue(doesContainMessage, @"%s", status.error_message().c_str());
  }
}

// Two sequential operations.
- (void)testTwoOperationsSuccess {
  auto nodes = MulLinkable(1, 1, 2);
  auto nodes2 = MulLinkable(2, 2, 3);
  nodes.insert(nodes.end(), nodes2.begin(), nodes2.end());
  std::vector<ComputeTaskDescriptorPtr> model;
  auto status = ValidateOptimizeModel({1}, {3}, nodes, &model);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

// Two sequential operations. Not fused.
- (void)testTwoOperationsNotFusedSuccess {
  auto nodes = Add(1, 1, 2);
  auto nodes2 = Add(2, 2, 3);
  nodes.insert(nodes.end(), nodes2.begin(), nodes2.end());
  std::vector<ComputeTaskDescriptorPtr> model;
  auto status = ValidateOptimizeModel({1}, {3}, nodes, &model);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

- (void)testAddOperationSuccess {
  auto nodes = Add2(1, 1, 2, 3);
  std::vector<ComputeTaskDescriptorPtr> model;
  auto status = ValidateOptimizeModel({1, 2}, {3}, nodes, &model);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

- (void)testAddOperationFused {
  auto graph = Add(1, 1, 3);
  auto graph2 = Add2Linkable(2, 2, 3, 4);
  graph.insert(graph.end(), graph2.begin(), graph2.end());
  std::vector<ComputeTaskDescriptorPtr> model;
  auto status = ValidateOptimizeModel({1, 2}, {4}, graph, &model);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
  XCTAssertTrue(model.size() == 1, @"Not fused, more than one task descriptor.");
}

- (void)testBinaryOperationSuccess {
  auto graph = Add(1, 1, 3);
  auto graph2 = Add(2, 2, 4);
  graph.insert(graph.end(), graph2.begin(), graph2.end());
  auto graph3 = Add2Linkable(3, 3, 4, 5);
  graph.insert(graph.end(), graph3.begin(), graph3.end());
  std::vector<ComputeTaskDescriptorPtr> model;
  auto status = ValidateOptimizeModel({1, 2}, {5}, graph, &model);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

@end
