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

// to demonstrate the performance difference between ION and HLOS memory
// for sharing with ADSP.
#define USE_ION_MEMORY

#include "hexagon_controller.h"

#include <malloc.h>
#include <stdio.h>

#include "adspmsgd.h"
#include "dspCV.h"
#include "node_data_float.h"
#include "rpcmem.h"  // helper API's for shared buffer allocation
#include "soc_interface.h"
#include "tfm_log.h"

// if false, use int data as input.  This is only for acceleration purpose.
// Also you may need to change android.min.
static const bool USE_FLOAT_DATA = true;

// if true, show id for each node
static const bool DBG_SHOW_ID = false;

static const uint32_t OUTPUT_PARAM_MAX_LINE_SIZE = 1000;

static const uint32_t PRINT_BUFSIZE = 2 * 1024 * 1024;

// extern pre-generated inception dummy data
extern uint8_t inception_dummy_int_data_224x224[];
extern uint8_t inception_dummy_int_data_299x299[];
extern float inception_dummy_float_data_299x299[];

#define HEXAGON_CONTROLLER_VERSION 101

// allocate print bufsize in advance @MB
#define PRINT_BUFSIZE (2 * 1024 * 1024)

static unsigned char s_print_buf[PRINT_BUFSIZE];

#define MAX_INPUTS 10
#define MAX_OUTPUTS 10

static struct NodeDataFloat s_input_node_data_buffer[MAX_INPUTS];
static uint8_t* s_output_node_data_buffer[MAX_OUTPUTS];
static int s_output_node_data_buffer_max_byte_size[MAX_OUTPUTS];
static int s_output_node_data_array_byte_size[MAX_OUTPUTS];
static uint32_t s_target_graph_id;

static bool s_dbg_use_inception_dummy_data = false;
static int s_dbg_inception_version = 3;

static int GetInputNodeCount() {
  for (int i = 0; i < MAX_INPUTS; ++i) {
    if (s_input_node_data_buffer[i].max_buf_byte_size == 0) {
      return i;
    }
  }
  return 0;
}

static int GetOutputNodeCount() {
  for (int i = 0; i < MAX_OUTPUTS; ++i) {
    if (s_output_node_data_buffer_max_byte_size[i] == 0) {
      return i;
    }
  }
  return 0;
}

static bool SetInputTensorDef(int port, hexagon_nn_tensordef* tensordef) {
  if (port >= GetInputNodeCount()) {
    TFMLOGE("Error exceeds input count.");
    return false;
  }
  struct NodeDataFloat* input_node_data_buffer =
      &s_input_node_data_buffer[port];
  tensordef->batches = input_node_data_buffer->x;
  tensordef->height = input_node_data_buffer->y;
  tensordef->width = input_node_data_buffer->z;
  tensordef->depth = input_node_data_buffer->d;
  tensordef->data = input_node_data_buffer->byte_array_data;
  tensordef->dataLen = input_node_data_buffer->array_byte_size;

  return true;
}

bool hexagon_controller_SetAllInputTensorDef(int node_count,
                                             hexagon_nn_tensordef* tensordef) {
  bool success = true;
  if (node_count != GetInputNodeCount()) {
    TFMLOGE("Error invalid input node count.");
    return false;
  }
  for (int i = 0; i < node_count; ++i) {
    SetInputTensorDef(i, &tensordef[i]);
  }
  return success;
}

static bool SetOutputTensorDef(int port, hexagon_nn_tensordef* tensordef) {
  if (port >= GetOutputNodeCount()) {
    TFMLOGE("Error exceeds output count.");
    return false;
  }
  tensordef->data = s_output_node_data_buffer[port];
  tensordef->dataLen = s_output_node_data_buffer_max_byte_size[port];
  return true;
}

bool hexagon_controller_SetAllOutputTensorDef(int node_count,
                                              hexagon_nn_tensordef* tensordef) {
  bool success = true;
  if (node_count != GetOutputNodeCount()) {
    TFMLOGE("Error invalid output node count. %d != %d", node_count,
            GetOutputNodeCount());
    return false;
  }
  for (int i = 0; i < node_count; ++i) {
    SetOutputTensorDef(i, &tensordef[i]);
  }
  return success;
}

void hexagon_controller_InitInputNodeDataToInceptionDummyData(int version) {
  if (version == 1) {
    if (USE_FLOAT_DATA) {
      TFMLOGE("ERROR!!!! Do not use float data for v1");
      return;
    }
    hexagon_controller_CopyByteNodeData(
        0, INCEPTION_PARAM_BATCHES, INCEPTION_PARAM_HEIGHT_V1,
        INCEPTION_PARAM_WIDTH_V1, INCEPTION_PARAM_DEPTH, 1,
        inception_dummy_int_data_224x224);
  } else if (version == 3) {
    if (USE_FLOAT_DATA) {
      hexagon_controller_CopyByteNodeData(
          0, INCEPTION_PARAM_BATCHES, INCEPTION_PARAM_HEIGHT_V3,
          INCEPTION_PARAM_WIDTH_V3, INCEPTION_PARAM_DEPTH, sizeof(float),
          (uint8_t*)inception_dummy_float_data_299x299);
    } else {
      hexagon_controller_CopyByteNodeData(
          0, INCEPTION_PARAM_BATCHES, INCEPTION_PARAM_HEIGHT_V3,
          INCEPTION_PARAM_WIDTH_V3, INCEPTION_PARAM_DEPTH, 1,
          inception_dummy_int_data_299x299);
    }
  }
}

bool hexagon_controller_ExecuteGraphWithBuffer(uint32_t nn_id,
                                               bool show_ranking) {
  const int input_node_count = GetInputNodeCount();
  hexagon_nn_tensordef inputs[input_node_count];
  const int output_node_count = GetOutputNodeCount();
  if (output_node_count <= 0) {
    TFMLOGI("Error output node count is 0.");
    return false;
  }
  hexagon_nn_tensordef outputs[output_node_count];
  hexagon_controller_SetAllInputTensorDef(input_node_count, inputs);
  hexagon_controller_SetAllOutputTensorDef(output_node_count, outputs);
  const bool success = hexagon_controller_ExecuteGraphWithMultipleInOut(
      nn_id, input_node_count, inputs, output_node_count, outputs);
  for (int i = 0; i < output_node_count; ++i) {
    s_output_node_data_array_byte_size[i] = outputs[i].data_valid_len;
  }

  const hexagon_nn_tensordef* output0 = &outputs[0];

  const uint32_t out_batches = output0->batches;
  const uint32_t out_height = output0->height;
  const uint32_t out_width = output0->width;
  const uint32_t out_depth = output0->depth;
  const uint32_t out_data_size = output0->data_valid_len;
  const uint32_t out_buf_byte_size = output0->dataLen;

  if (!success) {
    TFMLOGE("Execution failed");
    DumpNNId(nn_id);
    return false;
  } else if (!show_ranking) {
    return true;
  }

  static const int OUT_RANKING_SIZE = 5;
  int out_ranking[OUT_RANKING_SIZE];
  hexagon_controller_PrintMaxNIdx(
      (float*)s_output_node_data_buffer[0],
      out_batches * out_height * out_width * out_depth, OUT_RANKING_SIZE,
      out_ranking);
  TFMLOGD("%d x %d x %d x %d, byte size = %d, buf size = %d\n", out_batches,
          out_height, out_width, out_depth, out_data_size, out_buf_byte_size);
  if (s_dbg_use_inception_dummy_data) {
    // Check the result of inception with a dummy data. This step shouldn't
    // be passed when show_ranking != true to avoid adding unnecessary
    // additional computation cost.
    if (out_ranking[0] == 169 && out_ranking[1] == 7) {
      TFMLOGD("Result is correct! %d, %d", out_ranking[0], out_ranking[1]);
      return true;
    } else {
      TFMLOGD("Result is wrong! %d, %d", out_ranking[0], out_ranking[1]);
      return false;
    }
  }
  return true;
}

uint32_t hexagon_controller_GetTargetGraphId() { return s_target_graph_id; }

void hexagon_controller_SetTargetGraphId(uint32_t graph_id) {
  s_target_graph_id = graph_id;
}

void hexagon_controller_PrintGraph(uint32_t id) {
  int retval = hexagon_nn_snpprint(id, s_print_buf, PRINT_BUFSIZE);
  TFMLOGD("PrintGraph %s\n", s_print_buf);
  if (retval) {
    TFMLOGE("Error on print graph\n");
  }
}

int hexagon_controller_GetWrapperVersion() {
  return HEXAGON_CONTROLLER_VERSION;
}

int hexagon_controller_GetHexagonBinaryVersion() {
  int retval = 0;
  hexagon_nn_version(&retval);
  return retval;
}

bool hexagon_controller_AllocateInputNodeDataBuffers(int port,
                                                     int input_buf_byte_size) {
  TFMLOGD("Allocate memory for input node data. port = %d, size = %d", port,
          input_buf_byte_size);
  if (s_input_node_data_buffer[port].max_buf_byte_size != 0) {
    TFMLOGE("ERROR! input buffer is already allocated!!");
    return false;
  } else {
    s_input_node_data_buffer[port].max_buf_byte_size = input_buf_byte_size;
    posix_memalign((void**)&s_input_node_data_buffer[port].byte_array_data, 128,
                   input_buf_byte_size);
    TFMLOGD("allocate input node data buffers done");
  }
  return true;
}

bool hexagon_controller_AllocateOutputNodeDataBuffers(
    int port, int output_buf_byte_size) {
  TFMLOGD("Allocate memory for output node data. port = %d, size = %d", port,
          output_buf_byte_size);
  if (s_output_node_data_buffer_max_byte_size[port] != 0) {
    TFMLOGE("ERROR! input buffer is already allocated!!");
    return false;
  } else {
    // s_output_node_data_buffer = malloc(output_size * sizeof(float));
    posix_memalign((void**)&s_output_node_data_buffer[port], 128,
                   output_buf_byte_size);
    s_output_node_data_buffer_max_byte_size[port] = output_buf_byte_size;
    s_output_node_data_array_byte_size[port] = 0;
    TFMLOGD("allocate output node data buffers");
  }
  return true;
}

bool hexagon_controller_AllocateMultipleNodeDataBuffers(int input_count,
                                                        int* input_sizes,
                                                        int output_count,
                                                        int* output_sizes) {
  bool success = true;
  for (int i = 0; i < input_count; ++i) {
    success &=
        hexagon_controller_AllocateInputNodeDataBuffers(i, input_sizes[i]);
  }
  for (int i = 0; i < output_count; ++i) {
    success &=
        hexagon_controller_AllocateOutputNodeDataBuffers(i, output_sizes[i]);
  }

  if (s_dbg_use_inception_dummy_data) {
    hexagon_controller_InitInputNodeDataToInceptionDummyData(
        s_dbg_inception_version);
  }
  return success;
}

bool hexagon_controller_AllocateNodeDataBuffers(int input_size,
                                                int output_size) {
  return hexagon_controller_AllocateMultipleNodeDataBuffers(1, &input_size, 1,
                                                            &output_size);
}

bool hexagon_controller_ReleaseInputNodeDataBuffersWithPort(int port) {
  struct NodeDataFloat* input_node_data_buffer =
      &s_input_node_data_buffer[port];
  if (input_node_data_buffer->max_buf_byte_size == 0) {
    TFMLOGE("ERROR! input buffer has not been allocated yet!!");
    return false;
  } else {
    input_node_data_buffer->max_buf_byte_size = 0;
    input_node_data_buffer->array_byte_size = 0;
    free(input_node_data_buffer->byte_array_data);
  }
  return true;
}

bool hexagon_controller_ReleaseOutputNodeDataBuffersWithPort(int port) {
  if (s_output_node_data_buffer_max_byte_size[port] == 0) {
    TFMLOGE("ERROR! output buffer has not been allocated yet!!");
    return false;
  } else {
    s_output_node_data_buffer_max_byte_size[port] = 0;
    s_output_node_data_array_byte_size[port] = 0;
    free(s_output_node_data_buffer[port]);
  }
  return true;
}

bool hexagon_controller_ReleaseNodeDataBuffers() {
  bool success = true;
  for (int i = 0; i < GetInputNodeCount(); ++i) {
    success &= hexagon_controller_ReleaseInputNodeDataBuffersWithPort(i);
  }
  for (int i = 0; i < GetOutputNodeCount(); ++i) {
    success &= hexagon_controller_ReleaseOutputNodeDataBuffersWithPort(i);
  }
  return success;
}

bool hexagon_controller_CopyByteNodeData(int port, int x, int y, int z, int d,
                                         int type_byte_size,
                                         uint8_t* array_data) {
  int array_byte_size = x * y * z * d * type_byte_size;
  TFMLOGD("--- %d, %d, %d, %d, %d, %d", x, y, z, d, type_byte_size,
          array_byte_size);
  struct NodeDataFloat* input_node_data_buffer = &s_input_node_data_buffer[0];

  if (input_node_data_buffer->max_buf_byte_size < array_byte_size) {
    TFMLOGE("ERROR! input buffer size is too small! %d < %d",
            input_node_data_buffer->max_buf_byte_size, array_byte_size);
    return false;
  }
  memcpy(input_node_data_buffer->byte_array_data, array_data, array_byte_size);
  input_node_data_buffer->array_byte_size = array_byte_size;
  input_node_data_buffer->x = x;
  input_node_data_buffer->y = y;
  input_node_data_buffer->z = z;
  input_node_data_buffer->d = d;
  return true;
}

int hexagon_controller_InitHexagonWithMaxAttributes(int enable_dcvs,
                                                    int bus_usage,
                                                    int version) {
  TFMLOGI("Init hexagon with max attributes (Controller version = %d)",
          HEXAGON_CONTROLLER_VERSION);
  const int MCPS = 1000;
  const int MBPS = 12000;

  adspmsgd_start(0, RPCMEM_HEAP_DEFAULT, 4096);

  dspCV_Attribute attrib[] = {
      // The below values will result in the maximum aDSP performance,
      // at Turbo voltage.
      // Slightly more MCPS than are available on current targets
      {DSP_TOTAL_MCPS, MCPS},
      // drive the clock to MAX on known targets
      {DSP_MCPS_PER_THREAD, MCPS / 2},
      // 12 GB/sec is slightly higher than the max realistic
      // max BW on existing targets.
      {PEAK_BUS_BANDWIDTH_MBPS, MBPS},
      // This app is non-real time, and constantly reading/writing memory
      {BUS_USAGE_PERCENT, bus_usage},
  };
  int retval = 0;
  if (!enable_dcvs) {
    retval = hexagon_nn_disable_dcvs();
    if (retval) {
      TFMLOGE("Failed to disable DSP DCVS: %x\n", retval);
    }
  }

  retval =
      dspCV_initQ6_with_attributes(attrib, sizeof(attrib) / sizeof(attrib[0]));
  TFMLOGD("Return value from dspCV_initQ6() : %d\n", retval);

  s_target_graph_id = 0;
  s_dbg_inception_version = version;

  return retval;
}

int hexagon_controller_DeInitHexagon() {
  adspmsgd_stop();
  TFMLOGI("Finalize hexagon");
  const int retval = dspCV_deinitQ6();
  TFMLOGD("return value from dspCV_deinitQ6(): %d \n", retval);

  hexagon_controller_ReleaseNodeDataBuffers();

  return retval;
}

void hexagon_controller_GrowMemorySize() { hexagon_nn_config(); }

struct NodeDataFloat* hexagon_controller_GetInputNodeDataBuffer(int port) {
  if (port >= GetInputNodeCount()) {
    TFMLOGE("port should be less than 1");
  }
  return &s_input_node_data_buffer[port];
}

uint8_t* hexagon_controller_GetOutputNodeDataBuffer(int port,
                                                    int* out_array_byte_size) {
  if (port >= GetOutputNodeCount()) {
    TFMLOGE("port should be less than 1");
  }
  *out_array_byte_size = s_output_node_data_array_byte_size[port];
  return s_output_node_data_buffer[port];
}

// Append const node to the graph
int hexagon_controller_AppendConstNode(const char* const name, int graph_id,
                                       int node_id, int batch, int height,
                                       int width, int depth,
                                       const uint8_t* const data,
                                       int data_length) {
  if (DBG_SHOW_ID) {
    TFMLOGV("---(CONST) %s, %d, %d, %d, %d, %d, %d", name, node_id, batch,
            height, width, depth, data_length);
  } else {
    TFMLOGV("---(CONST) %s, %d, %d, %d, %d, %d", name, batch, height, width,
            depth, data_length);
  }
  const int retval = hexagon_nn_append_const_node(
      graph_id, node_id, batch, height, width, depth, data, data_length);
  if (retval != 0) {
    TFMLOGE("Failed to append const node %d", node_id);
    return retval;
  }
  return retval;
}

// Append node to the graph
int hexagon_controller_AppendNode(const char* const name, int graph_id,
                                  int node_id, int ops_id, int padding_id,
                                  const hexagon_nn_input* const inputs,
                                  int inputs_count,
                                  const hexagon_nn_output* const outputs,
                                  int outputs_count) {
  char input_param_buf[OUTPUT_PARAM_MAX_LINE_SIZE];
  memset(input_param_buf, 0, OUTPUT_PARAM_MAX_LINE_SIZE);
  int pos = 0;
  pos += snprintf(&input_param_buf[pos], 500, "in: ");
  for (int i = 0; i < inputs_count; ++i) {
    if (DBG_SHOW_ID) {
      pos += snprintf(&input_param_buf[pos], 500, "(%d, %d), ",
                      inputs[i].src_id, inputs[i].output_idx);
    } else {
      pos +=
          snprintf(&input_param_buf[pos], 500, "(%d), ", inputs[i].output_idx);
    }
  }

  char output_param_buf[OUTPUT_PARAM_MAX_LINE_SIZE];
  memset(output_param_buf, 0, OUTPUT_PARAM_MAX_LINE_SIZE);
  pos = 0;
  pos += snprintf(&output_param_buf[pos], 500, "out: ");
  for (int i = 0; i < outputs_count; ++i) {
    pos += snprintf(&output_param_buf[pos], 500, "(%d), ", outputs[i].max_size);
  }

  if (DBG_SHOW_ID) {
    TFMLOGV("---(OP) %s, %d, %d, %d, %d, %d, %s, %s", name, node_id, ops_id,
            padding_id, inputs_count, outputs_count, input_param_buf,
            output_param_buf);
  } else {
    TFMLOGV("---(OP) %s, %d, %d, %d, %d, %s, %s", name, ops_id, padding_id,
            inputs_count, outputs_count, input_param_buf, output_param_buf);
  }
  const int retval =
      hexagon_nn_append_node(graph_id, node_id, ops_id, padding_id, inputs,
                             inputs_count, outputs, outputs_count);
  if (retval != 0) {
    TFMLOGE("Failed to append const node %d", node_id);
    return retval;
  }
  return retval;
}

void hexagon_controller_EnableDbgUseInceptionDummyData(bool enable) {
  s_dbg_use_inception_dummy_data = enable;
}

bool hexagon_controller_IsDbgUseInceptionDummyDataEnabled() {
  return s_dbg_use_inception_dummy_data;
}
