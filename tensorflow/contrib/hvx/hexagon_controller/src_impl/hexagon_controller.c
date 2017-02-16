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
#include "rpcmem.h"    // helper API's for shared buffer allocation
#include "soc_interface.h"
#include "tfm_log.h"

// if false, use int data as input.  This is only for acceleration purpose
static const bool USE_FLOAT_DATA = true;

// if true, show id for each node
static const bool DBG_SHOW_ID = false;

static const uint32_t OUTPUT_PARAM_MAX_LINE_SIZE = 1000;

// extern pre-generated inception dummy data
extern uint8_t inception_dummy_int_data_224x224[];
extern uint8_t inception_dummy_int_data_299x299[];
extern float inception_dummy_float_data_299x299[];

#define HEXAGON_CONTROLLER_VERSION 91

// allocate print bufsize in advance @MB
#define PRINT_BUFSIZE (2 * 1024 * 1024)

static unsigned char s_print_buf[PRINT_BUFSIZE];

// input node data buffer size
// x2 1024 * 1024 * 2 > 299 * 299 * 3 * 4 > 1024 * 1024
static const int INPUT_NODE_DATA_BUFFER_SIZE = 1024 * 1024 * 2;
// output node data buffer size
// (1008 is enough for inception)
static const int OUTPUT_NODE_DATA_BUFFER_SIZE = 300 * 300 * 3 * 4;

static struct NodeDataFloat s_input_node_data_float_buffer;
static float* s_output_node_data_float_buffer;
static int s_output_node_data_float_buffer_byte_size;
static int s_output_node_data_float_array_size;
static uint32_t s_target_graph_id;

static bool s_dbg_use_inception_dummy_data = false;

void hexagon_controller_InitInputNodeDataToInceptionDummyData(int version) {
  if (version == 1) {
    if (USE_FLOAT_DATA) {
      TFMLOGE("ERROR!!!! Do not use float data for v1");
      return;
    }
    hexagon_controller_CopyByteNodeData(
        INCEPTION_PARAM_BATCHES, INCEPTION_PARAM_HEIGHT_V1,
        INCEPTION_PARAM_WIDTH_V1, INCEPTION_PARAM_DEPTH,
        1, inception_dummy_int_data_224x224);
  } else if (version == 3) {
    if (USE_FLOAT_DATA) {
      hexagon_controller_CopyByteNodeData(
          INCEPTION_PARAM_BATCHES, INCEPTION_PARAM_HEIGHT_V3,
          INCEPTION_PARAM_WIDTH_V3, INCEPTION_PARAM_DEPTH,
          sizeof(float), (uint8_t*)inception_dummy_float_data_299x299);
    } else {
      hexagon_controller_CopyByteNodeData(
          INCEPTION_PARAM_BATCHES, INCEPTION_PARAM_HEIGHT_V3,
          INCEPTION_PARAM_WIDTH_V3, INCEPTION_PARAM_DEPTH,
          1, inception_dummy_int_data_299x299);
    }
  }
}

bool hexagon_controller_ExecuteGraphWithBuffer(
    uint32_t nn_id, bool show_ranking) {
  uint32_t out_batches, out_height, out_width, out_depth;
  uint32_t out_data_size;
  int x = s_input_node_data_float_buffer.x;
  int y = s_input_node_data_float_buffer.y;
  int z = s_input_node_data_float_buffer.z;
  int d = s_input_node_data_float_buffer.d;
  uint8_t *byte_data = s_input_node_data_float_buffer.byte_array_data;
  int array_size = s_input_node_data_float_buffer.array_size;
  const bool success = hexagon_controller_ExecuteGraph(
      nn_id, x, y, z, d, byte_data, array_size,
      &out_batches, &out_height, &out_width, &out_depth,
      (uint8_t *)s_output_node_data_float_buffer,
      s_output_node_data_float_buffer_byte_size,
      &out_data_size);
  s_output_node_data_float_array_size =
      out_batches * out_height * out_width * out_depth;
  if (!success) {
    TFMLOGE("Execution failed");
    return false;
  } else if (!show_ranking) {
    return true;
  }

  static const int OUT_RANKING_SIZE = 5;
  int out_ranking[OUT_RANKING_SIZE];
  hexagon_controller_PrintMaxNIdx(
      s_output_node_data_float_buffer,
      out_batches * out_height * out_width * out_depth,
      OUT_RANKING_SIZE, out_ranking);
  TFMLOGD("%d x %d x %d x %d, byte size = %d\n",
          out_batches,
          out_height,
          out_width,
          out_depth,
          out_data_size);
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

uint32_t hexagon_controller_GetTargetGraphId() {
  return s_target_graph_id;
}

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

bool hexagon_controller_AllocateNodeDataBuffers(
    int input_size, int output_size) {
  TFMLOGD("Allocate memory for input / output node data float");
  if (s_input_node_data_float_buffer.buf_size != 0) {
    TFMLOGE("ERROR! input buffer is already allocated!!");
    return false;
  } else {
    int byte_array_data_size = USE_FLOAT_DATA ?
        input_size * sizeof(float) : input_size; /* sizeof(uint8_t) ? */
    s_input_node_data_float_buffer.buf_size = input_size;
    // unused? remove?
    s_input_node_data_float_buffer.array_data =
        malloc(input_size * sizeof(float));
    s_input_node_data_float_buffer.byte_array_data =
        malloc(byte_array_data_size);

    s_output_node_data_float_buffer = malloc(output_size * sizeof(float));
    s_output_node_data_float_buffer_byte_size = output_size * sizeof(float);
    s_output_node_data_float_array_size = 0;
    TFMLOGD("allocate node data buffers");
  }
  return true;
}

bool hexagon_controller_ReleaseNodeDataBuffers() {
  if (s_input_node_data_float_buffer.buf_size == 0) {
    TFMLOGE("ERROR! input buffer has not been allocated yet!!");
    return false;
  } else {
    s_input_node_data_float_buffer.buf_size = 0;
    free(s_input_node_data_float_buffer.array_data);
  }
  if (s_output_node_data_float_buffer_byte_size == 0) {
    TFMLOGE("ERROR! output buffer has not been allocated yet!!");
    return false;
  } else {
    s_output_node_data_float_buffer_byte_size = 0;
    free(s_input_node_data_float_buffer.byte_array_data);
  }
  return true;
}

bool hexagon_controller_CopyByteNodeData(
    int x, int y, int z, int d, int type_byte_size, uint8_t* array_data) {
  int array_byte_size = x * y * z * d * type_byte_size;
  TFMLOGD("--- %d, %d, %d, %d, %d, %d",x,y,z,d,type_byte_size,array_byte_size);
  if (s_input_node_data_float_buffer.buf_size < array_byte_size) {
    TFMLOGE("ERROR! input buffer size is too small! %d < %d",
            s_input_node_data_float_buffer.buf_size, array_byte_size);
    return false;
  }
  memcpy(s_input_node_data_float_buffer.byte_array_data,
         array_data, array_byte_size);
  s_input_node_data_float_buffer.array_size = array_byte_size;
  s_input_node_data_float_buffer.x = x;
  s_input_node_data_float_buffer.y = y;
  s_input_node_data_float_buffer.z = z;
  s_input_node_data_float_buffer.d = d;
  return true;
}

int hexagon_controller_InitHexagonWithMaxAttributes(
    int enable_dcvs, int bus_usage, int version) {
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

  hexagon_controller_AllocateNodeDataBuffers(
      INPUT_NODE_DATA_BUFFER_SIZE, OUTPUT_NODE_DATA_BUFFER_SIZE);

  if (s_dbg_use_inception_dummy_data) {
    hexagon_controller_InitInputNodeDataToInceptionDummyData(version);
  }
  s_target_graph_id = 0;

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

void hexagon_controller_GrowMemorySize() {
  hexagon_nn_config();
}

struct NodeDataFloat* hexagon_controller_GetInputNodeDataFloatBuffer() {
  return &s_input_node_data_float_buffer;
}

float* hexagon_controller_GetOutputNodeDataFloatBuffer(
    const char *const node_name, int* out_array_size) {
  *out_array_size = s_output_node_data_float_array_size;
  return s_output_node_data_float_buffer;
}

// Append const node to the graph
int hexagon_controller_AppendConstNode(
    const char* const name, int graph_id, int node_id,
    int batch, int height, int width, int depth,
    const uint8_t* const data, int data_length) {
  if (DBG_SHOW_ID) {
    TFMLOGV("---(CONST) %s, %d, %d, %d, %d, %d, %d",
            name, node_id, batch, height, width, depth, data_length);
  } else {
    TFMLOGV("---(CONST) %s, %d, %d, %d, %d, %d",
            name, batch, height, width, depth, data_length);
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
int hexagon_controller_AppendNode(
    const char* const name, int graph_id, int node_id, int ops_id,
    int padding_id, const hexagon_nn_input* const inputs,
    int inputs_count, const hexagon_nn_output* const outputs,
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
      pos += snprintf(&input_param_buf[pos], 500, "(%d), ",
                      inputs[i].output_idx);
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
    TFMLOGV("---(OP) %s, %d, %d, %d, %d, %d, %s, %s", name, node_id,
            ops_id, padding_id, inputs_count, outputs_count, input_param_buf,
            output_param_buf);
  } else {
    TFMLOGV("---(OP) %s, %d, %d, %d, %d, %s, %s", name,
            ops_id, padding_id, inputs_count, outputs_count, input_param_buf,
            output_param_buf);
  }
  const int retval = hexagon_nn_append_node(
      graph_id, node_id, ops_id, padding_id,
      inputs, inputs_count,
      outputs, outputs_count);
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
