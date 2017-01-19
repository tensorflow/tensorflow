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

#ifndef NODE_DATA_FLOAT_H
#define NODE_DATA_FLOAT_H

#ifdef __cplusplus
extern "C" {
#else
#include <inttypes.h>
#endif
#define NODE_DATA_FLOAT_NODE_NAME_BUF_SIZE 100

struct NodeDataFloat {
  int x;
  int y;
  int z;
  int d;
  int buf_size;
  int array_size;
  float* array_data;
  uint8_t* byte_array_data;
  char node_name[NODE_DATA_FLOAT_NODE_NAME_BUF_SIZE];
};
#ifdef __cplusplus
}
#endif

#endif  // NODE_DATA_FLOAT_H
