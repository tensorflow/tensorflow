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

int soc_interface_GetWrapperVersion() {
  // TODO(satok): implement
  return -1;
}

int soc_interface_GetSocControllerVersion() {
  // TODO(satok): implement
  return -1;
}

bool soc_interface_Init() {
  // TODO(satok): implement
  return false;
}

bool soc_interface_Finalize() {
  // TODO(satok): implement
  return false;
}

bool soc_interface_ExecuteGraph() {
  // TODO(satok): implement
  return false;
}

bool soc_interface_TeardownGraph() {
  // TODO(satok): implement
  return false;
}

bool soc_interface_FillInputNodeFloat(
    int x, int y, int z, int d, const uint8_t* const buf, uint64_t buf_size) {
  // TODO(satok): implement
  return false;
}

// TODO(satok): Remove and use runtime version
bool soc_interface_ReadOutputNodeFloat(
    const char* const node_name, uint8_t** buf, uint64_t *buf_size) {
  // TODO(satok): implement
  return false;
}

bool soc_interface_SetupGraphDummy(int version) {
  // TODO(satok): implement
  return false;
}

bool soc_interface_AllocateNodeInputAndNodeOutputArray(
    int total_input_count, int total_output_count) {
  // TODO(satok): implement
  return false;
}

bool soc_interface_ReleaseNodeInputAndNodeOutputArray() {
  // TODO(satok): implement
  return false;
}

void* soc_interface_SetOneNodeInputs(
    int input_count, const int* const node_id, const int* const port) {
  // TODO(satok): implement
  return 0;
}

void* soc_interface_SetOneNodeOutputs(int output_count, int* max_size) {
  // TODO(satok): implement
  return 0;
}

// Append const node to the graph
bool soc_interface_AppendConstNode(
    const char* const name, int node_id, int batch, int height, int width,
    int depth, const uint8_t* const data, int data_length) {
  // TODO(satok): implement
  return false;
}

// Append node to the graph
bool soc_interface_AppendNode(
    const char* const name, int node_id, int ops_id, int padding_id,
    const void* const inputs, int inputs_count, const void* const outputs,
    int outputs_count) {
  // TODO(satok): implement
  return false;
}


// Instantiate graph
bool soc_interface_InstantiateGraph() {
  // TODO(satok): implement
  return false;
}

// Construct graph
bool soc_interface_ConstructGraph() {
  // TODO(satok): implement
  return false;
}

void soc_interface_SetLogLevel(int log_level) {
  // TODO(satok): implement
}

void soc_interface_SetDebugFlag(uint64_t flag) {
  // TODO(satok): implement
}
