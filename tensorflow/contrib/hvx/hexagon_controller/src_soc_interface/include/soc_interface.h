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

#ifndef TENSORFLOW_PLATFORM_HEXAGON_SOC_INTERFACE_H_
#define TENSORFLOW_PLATFORM_HEXAGON_SOC_INTERFACE_H_

#include <inttypes.h>

// Declaration of APIs provided by hexagon shared library. This header is shared
// with both hexagon library built with qualcomm SDK and tensorflow.
// All functions defined here must have prefix "soc_interface" to avoid
// naming conflicts.
#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif  // __cplusplus
// Returns the version of loaded hexagon wrapper shared library.
// You should assert that the version matches the expected version before
// calling APIs defined in this header.
int soc_interface_GetWrapperVersion();
// Returns the version of hexagon binary.
// You should assert that the version matches the expected version before
// calling APIs defined in this header.
int soc_interface_GetSocControllerVersion();
// Initialize SOC
bool soc_interface_Init();
// Finalize SOC
bool soc_interface_Finalize();
// Execute graph on SOC
bool soc_interface_ExecuteGraph();
// Teardown graph setup
bool soc_interface_TeardownGraph();

// Allocate buffers for input node and output node
bool soc_interface_AllocateInOutNodeBuffers(int input_count, int* input_sizes,
                                            int output_count,
                                            int* output_sizes);

// Send input data to SOC with port
bool soc_interface_FillInputNodeWithPort(int port, int x, int y, int z, int d,
                                         const uint8_t* const buf,
                                         uint64_t buf_byte_size);

// Send input data to SOC
bool soc_interface_FillInputNodeFloat(int x, int y, int z, int d,
                                      const uint8_t* const buf,
                                      uint64_t buf_byte_size);

// Load output data from SOC with port
bool soc_interface_ReadOutputNodeWithPort(int port, uint8_t** buf,
                                          uint64_t* buf_byte_size);

// Load output data from SOC
bool soc_interface_ReadOutputNodeFloat(const char* const node_name,
                                       uint8_t** buf, uint64_t* buf_byte_size);

// Setup graph
// TODO(satok): Remove and use runtime version
bool soc_interface_setupDummyGraph(int version);

// Allocate memory for params of node inputs and node outputs
bool soc_interface_AllocateNodeInputAndNodeOutputArray(int total_input_count,
                                                       int total_output_count);

// Release memory for params of node inputs and node outputs
bool soc_interface_ReleaseNodeInputAndNodeOutputArray();

// Set one node's inputs and return pointer to that struct
void* soc_interface_SetOneNodeInputs(int input_count, const int* const node_id,
                                     const int* const port);

// Set one node's outputs and return pointer to that struct
void* soc_interface_SetOneNodeOutputs(int output_count, int* max_size);

// Append const node to the graph
bool soc_interface_AppendConstNode(const char* const name, int node_id,
                                   int batch, int height, int width, int depth,
                                   const uint8_t* const data, int data_length);

// Append node to the graph
bool soc_interface_AppendNode(const char* const name, int node_id, int op_id,
                              int padding_id, const void* const inputs,
                              int inputs_count, const void* const outputs,
                              int outputs_count);

// Instantiate graph
bool soc_interface_InstantiateGraph();

// Construct graph
bool soc_interface_ConstructGraph();

// Set log level
void soc_interface_SetLogLevel(int log_level);

// Set debug flag
void soc_interface_SetDebugFlag(uint64_t flag);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_PLATFORM_HEXAGON_SOC_INTERFACE_H_
