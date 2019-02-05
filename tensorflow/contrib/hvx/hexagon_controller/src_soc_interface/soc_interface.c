/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "soc_interface.h"

#include <inttypes.h>

#include "hexagon_controller.h"
#include "hexagon_nn.h"
#include "node_data_float.h"
#include "tfm_log.h"

// to demonstrate the performance difference between ION and HLOS memory
// for sharing with ADSP.
#define USE_ION_MEMORY

const int64_t FLAG_ENABLE_INCEPTION_DUMMY_BINARY_INPUT = 0x01;
const int64_t FLAG_ENABLE_EXPERIMENTAL_DEBUG = 0x02;

static const int INCEPTION_VERSION = 3;

static hexagon_nn_input* s_node_inputs_array;
static int s_node_inputs_array_index;
static int s_node_inputs_array_max_count;

static hexagon_nn_output* s_node_outputs_array;
static int s_node_outputs_array_index;
static int s_node_outputs_array_max_count;

int soc_interface_GetWrapperVersion() {
  TFMLOGD("GetWrapperVersion");
  return hexagon_controller_GetWrapperVersion();
}

int soc_interface_GetSocControllerVersion() {
  TFMLOGD("GetSocControllerVersion");
  return hexagon_controller_GetHexagonBinaryVersion();
}

bool soc_interface_Init() {
  TFMLOGD("Init");
  hexagon_controller_InitHexagonWithMaxAttributes(
      0, 100, INCEPTION_VERSION /* version */);
  hexagon_controller_GrowMemorySize();
  return true;
}

bool soc_interface_Finalize() {
  TFMLOGD("Finalize");
  hexagon_controller_DeInitHexagon();
  return true;
}

bool soc_interface_ExecuteGraph() {
  TFMLOGD("ExecuteGraph");
  if (hexagon_controller_IsDbgUseInceptionDummyDataEnabled()) {
    hexagon_controller_InitInputNodeDataToInceptionDummyData(
        INCEPTION_VERSION /* version */);
  }
  const uint32_t graph_id = hexagon_controller_GetTargetGraphId();
  if (graph_id == 0) {
    TFMLOGE("Graph id has not been set yet.");
    return false;
  }
  hexagon_controller_ExecuteGraphWithBuffer(graph_id, true);
  return true;
}

bool soc_interface_TeardownGraph() {
  TFMLOGD("TeardownGraph");
  const uint32_t graph_id = hexagon_controller_GetTargetGraphId();
  if (graph_id == 0) {
    TFMLOGE("Graph id has not been set yet.");
    return false;
  }
  hexagon_controller_Teardown(graph_id);
  return true;
}

bool soc_interface_AllocateInOutNodeBuffers(int input_count, int* input_sizes,
                                            int output_count,
                                            int* output_sizes) {
  TFMLOGD("AllocateInOutNodeBuffers");
  return hexagon_controller_AllocateMultipleNodeDataBuffers(
      input_count, input_sizes, output_count, output_sizes);
}

bool soc_interface_FillInputNodeWithPort(int port, int x, int y, int z, int d,
                                         const uint8_t* const buf,
                                         uint64_t buf_byte_size) {
  TFMLOGD("FillInputNodeWithPort %d", port);
  struct NodeDataFloat* node_data =
      hexagon_controller_GetInputNodeDataBuffer(port);
  if (buf_byte_size > node_data->max_buf_byte_size) {
    TFMLOGE("buf size exceeds max buf size");
    return false;
  }
  memcpy(node_data->byte_array_data, buf, buf_byte_size);
  node_data->x = x;
  node_data->y = y;
  node_data->z = z;
  node_data->d = d;
  node_data->array_byte_size = buf_byte_size;
  return true;
}

bool soc_interface_FillInputNodeFloat(int x, int y, int z, int d,
                                      const uint8_t* const buf,
                                      uint64_t buf_byte_size) {
  return soc_interface_FillInputNodeWithPort(
      /*port=*/0, x, y, z, d, buf, buf_byte_size);
}

// TODO(satok): Remove and use runtime version
bool soc_interface_ReadOutputNodeWithPort(int port, uint8_t** buf,
                                          uint64_t* buf_byte_size) {
  TFMLOGD("ReadOutputNodeWithPort");
  int array_byte_size = -1;
  uint8_t* output_node_data_buffer =
      hexagon_controller_GetOutputNodeDataBuffer(port, &array_byte_size);
  if (array_byte_size < 0) {
    TFMLOGE("Failed to read data.");
    return false;
  }
  *buf = output_node_data_buffer;
  *buf_byte_size = array_byte_size;
  return true;
}

bool soc_interface_ReadOutputNodeFloat(const char* const node_name,
                                       uint8_t** buf, uint64_t* buf_byte_size) {
  return soc_interface_ReadOutputNodeWithPort(/*port=*/0, buf, buf_byte_size);
}

bool soc_interface_setupDummyGraph(int version) {
  TFMLOGD("SetupGraphDummy");
  const uint32_t graph_id = hexagon_controller_SetupGraph(version);
  if (graph_id == 0) {
    TFMLOGE("Failed to setup graph");
    return false;
  }
  hexagon_controller_SetTargetGraphId(graph_id);
  return true;
}

bool soc_interface_AllocateNodeInputAndNodeOutputArray(int total_input_count,
                                                       int total_output_count) {
  TFMLOGD("Allocate node inputs and node outputs array %d, %d",
          total_input_count, total_output_count);
  posix_memalign((void**)&s_node_inputs_array, 128,
                 total_input_count * sizeof(hexagon_nn_input));
  posix_memalign((void**)&s_node_outputs_array, 128,
                 total_output_count * sizeof(hexagon_nn_output));
  s_node_inputs_array_index = 0;
  s_node_outputs_array_index = 0;
  s_node_inputs_array_max_count = total_input_count;
  s_node_outputs_array_max_count = total_output_count;
  return true;
}

bool soc_interface_ReleaseNodeInputAndNodeOutputArray() {
  TFMLOGD("Release node inputs and node outputs array");
  free(s_node_inputs_array);
  free(s_node_outputs_array);
  return true;
}

void* soc_interface_SetOneNodeInputs(
    int input_count, const int* const node_id, const int* const port) {
  if (s_node_inputs_array_index + input_count > s_node_inputs_array_max_count) {
    TFMLOGE("input count exceeds limit");
    return 0;
  }
  for (int i = 0; i < input_count; ++i) {
    const int index = s_node_inputs_array_index + i;
    s_node_inputs_array[index].src_id = node_id[i];
    s_node_inputs_array[index].output_idx = port[i];
  }
  void* retval = (void*)(&s_node_inputs_array[s_node_inputs_array_index]);
  s_node_inputs_array_index += input_count;
  return retval;
}

void* soc_interface_SetOneNodeOutputs(int output_count, int* max_size) {
  if (s_node_outputs_array_index + output_count >
      s_node_outputs_array_max_count) {
    TFMLOGE("output count exceeds limit");
    return 0;
  }
  for (int i = 0; i < output_count; ++i) {
    const int index = s_node_outputs_array_index + i;
    s_node_outputs_array[index].max_size = max_size[i];
  }
  void* retval = (void*)(&s_node_outputs_array[s_node_outputs_array_index]);
  s_node_outputs_array_index += output_count;
  return retval;
}

// Append const node to the graph
bool soc_interface_AppendConstNode(const char* const name, int node_id,
                                   int batch, int height, int width, int depth,
                                   const uint8_t* const data, int data_length) {
  const uint32_t graph_id = hexagon_controller_GetTargetGraphId();
  const int retval = hexagon_controller_AppendConstNode(
      name, graph_id, node_id, batch, height, width, depth, data, data_length);
  if (retval != 0) {
    TFMLOGE("Failed to append const node %d", node_id);
    return false;
  }
  return true;
}

// Append node to the graph
bool soc_interface_AppendNode(const char* const name, int node_id, int ops_id,
                              int padding_id, const void* const inputs,
                              int inputs_count, const void* const outputs,
                              int outputs_count) {
  const uint32_t graph_id = hexagon_controller_GetTargetGraphId();
  const int retval = hexagon_controller_AppendNode(
      name, graph_id, node_id, ops_id, padding_id, (hexagon_nn_input*)inputs,
      inputs_count, (hexagon_nn_output*)outputs, outputs_count);
  if (retval != 0) {
    TFMLOGE("Failed to append const node %d", node_id);
    return false;
  }
  return true;
}

// Instantiate graph
bool soc_interface_InstantiateGraph() {
  const uint32_t nn_id = hexagon_controller_InstantiateGraph();
  hexagon_controller_SetTargetGraphId(nn_id);
  return true;
}

// Construct graph
bool soc_interface_ConstructGraph() {
  const uint32_t graph_id = hexagon_controller_GetTargetGraphId();
  return hexagon_controller_ConstructGraph(graph_id);
}

void soc_interface_SetLogLevel(int log_level) {
  SetLogLevel(log_level);
}

void soc_interface_SetDebugFlag(uint64_t flag) {
  TFMLOGI("Set debug flag 0x%" PRIx64, flag);
  if ((flag & FLAG_ENABLE_INCEPTION_DUMMY_BINARY_INPUT) != 0) {
    TFMLOGI("Enable always use panda data");
    hexagon_controller_EnableDbgUseInceptionDummyData(true);
  } else if ((flag & FLAG_ENABLE_EXPERIMENTAL_DEBUG) != 0) {
    SetExperimentalDebug();
  }
}
