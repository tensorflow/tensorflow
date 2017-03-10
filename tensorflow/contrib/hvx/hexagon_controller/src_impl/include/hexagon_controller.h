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

#ifndef GEMM_WRAPPER_H
#define GEMM_WRAPPER_H

#include <stdbool.h>
#include <stdlib.h>

#include "hexagon_nn.h"
#include "node_data_float.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define INCEPTION_PARAM_BATCHES 1
#define INCEPTION_PARAM_HEIGHT_V1 224
#define INCEPTION_PARAM_WIDTH_V1 224
#define INCEPTION_PARAM_HEIGHT_V3 299
#define INCEPTION_PARAM_WIDTH_V3 299
#define INCEPTION_PARAM_DEPTH 3

// General functions
void hexagon_controller_PrintGraph(uint32_t nn_id);

int hexagon_controller_GetWrapperVersion();

int hexagon_controller_GetHexagonBinaryVersion();

// Hexagon perf functions
int hexagon_controller_InitHexagonWithMaxAttributes(int enable_dcvs,
                                                    int bus_usage, int version);

bool hexagon_controller_AllocateNodeDataBuffers(int input_size,
                                                int output_size);

bool hexagon_controller_ReleaseNodeDataBuffers();

bool hexagon_controller_CopyByteNodeData(int x, int y, int z, int d,
                                         int type_byte_size,
                                         uint8_t* array_data);

int hexagon_controller_DeInitHexagon();

uint32_t hexagon_controller_GetTargetGraphId();

void hexagon_controller_SetTargetGraphId(uint32_t graph_id);

// Hexagon config functions
void hexagon_controller_GrowMemorySize();

// Graph data transfer functions
struct NodeDataFloat* hexagon_controller_GetInputNodeDataFloatBuffer();

float* hexagon_controller_GetOutputNodeDataFloatBuffer(
    const char* const node_name, int* out_array_size);

// Graph functions
uint32_t hexagon_controller_InstantiateGraph();

void hexagon_controller_InitGraph(int version, uint32_t nn_id);

bool hexagon_controller_ConstructGraph(uint32_t nn_id);

uint32_t hexagon_controller_SetupGraph(int version);

bool hexagon_controller_ExecuteInceptionDummyData(uint32_t nn_id);

bool hexagon_controller_ExecuteGraph(
    const uint32_t nn_id, const uint32_t batches, const uint32_t height,
    const uint32_t width, const uint32_t depth, uint8_t* int_data,
    const uint32_t int_data_size, uint32_t* out_batches, uint32_t* out_height,
    uint32_t* out_width, uint32_t* out_depth, uint8_t* out_vals,
    const uint32_t output_val_byte_size, uint32_t* out_data_byte_size);

bool hexagon_controller_ExecuteGraphWithBuffer(uint32_t nn_id,
                                               bool show_ranking);

void hexagon_controller_DumpPerf(uint32_t nn_id);

void hexagon_controller_DumpNodeName(uint32_t nn_id);

void hexagon_controller_Teardown(uint32_t nn_id);

void hexagon_controller_PrintMaxNIdx(const float* data, const uint32_t entries,
                                     const int n, int* out_ranking);

void hexagon_controller_InitInputNodeDataToInceptionDummyData(int version);

int hexagon_controller_AppendNode(const char* const name, int graph_id,
                                  int node_id, int op_id, int padding_id,
                                  const hexagon_nn_input* const inputs,
                                  int inputs_count,
                                  const hexagon_nn_output* const outputs,
                                  int outputs_count);

int hexagon_controller_AppendConstNode(const char* const name, int graph_id,
                                       int node_id, int batch, int height,
                                       int width, int depth,
                                       const uint8_t* const data,
                                       int data_length);

void hexagon_controller_EnableDbgUseInceptionDummyData(bool enable);

bool hexagon_controller_IsDbgUseInceptionDummyDataEnabled();

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // GEMM_WRAPPER_H
