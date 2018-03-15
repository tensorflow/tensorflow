/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
vcyou may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/kernels/hexagon/soc_interface.h"

// Dummy implementation of soc_interface.

int soc_interface_GetWrapperVersion() { return -1; }
int soc_interface_GetSocControllerVersion() { return -1; }
bool soc_interface_Init() { return false; }
bool soc_interface_Finalize() { return false; }
bool soc_interface_ExecuteGraph() { return false; }
bool soc_interface_TeardownGraph() { return false; }
bool soc_interface_AllocateInOutNodeBuffers(int /*input_count*/,
                                            int* /*input_sizes*/,
                                            int /*output_count*/,
                                            int* /*output_sizes*/) {
  return false;
}
bool soc_interface_FillInputNodeWithPort(int /*port*/, int /*x*/, int /*y*/,
                                         int /*z*/, int /*d*/,
                                         const uint8_t* const /*buf*/,
                                         uint64_t /*buf_byte_size*/) {
  return false;
}
bool soc_interface_FillInputNodeFloat(int /*x*/, int /*y*/, int /*z*/,
                                      int /*d*/, const uint8_t* const /*buf*/,
                                      uint64_t /*buf_byte_size*/) {
  return false;
}
bool soc_interface_ReadOutputNodeWithPort(int /*port*/, uint8_t** /*buf*/,
                                          uint64_t* /*buf_byte_size*/) {
  return false;
}
bool soc_interface_ReadOutputNodeFloat(const char* const /*node_name*/,
                                       uint8_t** /*buf*/,
                                       uint64_t* /*buf_byte_size*/) {
  return false;
}
bool soc_interface_setupDummyGraph(int /*version*/) { return false; }
bool soc_interface_AllocateNodeInputAndNodeOutputArray(
    int /*total_input_count*/, int /*total_output_count*/) {
  return false;
}
bool soc_interface_ReleaseNodeInputAndNodeOutputArray() { return false; }
void* soc_interface_SetOneNodeInputs(int /*input_count*/,
                                     const int* const /*node_id*/,
                                     const int* const /*port*/) {
  return nullptr;
}
void* soc_interface_SetOneNodeOutputs(int /*output_count*/, int* /*max_size*/) {
  return nullptr;
}
bool soc_interface_AppendConstNode(const char* const /*name*/, int /*node_id*/,
                                   int /*batch*/, int /*height*/, int /*width*/,
                                   int /*depth*/, const uint8_t* const /*data*/,
                                   int /*data_length*/) {
  return false;
}
bool soc_interface_AppendNode(const char* const /*name*/, int /*node_id*/,
                              int /*op_id*/, int /*padding_id*/,
                              const void* const /*inputs*/,
                              int /*inputs_count*/,
                              const void* const /*outputs*/,
                              int /*outputs_count*/) {
  return false;
}
bool soc_interface_InstantiateGraph() { return false; }
bool soc_interface_ConstructGraph() { return false; }
void soc_interface_SetLogLevel(int /*log_level*/) {}
void soc_interface_SetDebugFlag(uint64_t /*flag*/) {}
