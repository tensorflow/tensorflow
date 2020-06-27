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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_HELLO_WORLD_MAIN_FUNCTIONS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_HELLO_WORLD_MAIN_FUNCTIONS_H_

//#include <stdint.h>
//#include <stdbool.h>

#include "tensorflow/lite/micro/examples/hello_world/main_functions.h"
// Initializes all data needed for the example. The name is important, and needs
// to be setup() for Arduino compatibility.

struct model_info * setup_NN(const unsigned char *model_data);
float * loop_NN(float * input_data);
void print_string(const char * str);
*/
/*
void * getHandle();
void setup_PDM();
void pdm_desable();


void pdm_config_print(void);
void pdm_data_get(void);
bool is_data_ready();
void clear_data_ready();
int16_t * getBuffer();
void getBuffer_ext(int16_t ** buf);
*/
#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_HELLO_WORLD_MAIN_FUNCTIONS_H_
