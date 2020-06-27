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

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MAIN_FUNCTIONS_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MAIN_FUNCTIONS_H_

// Expose a C friendly interface for main functions.
#ifdef __cplusplus
extern "C" {
#endif
struct tensor_info{
	char *name;
	int  dim;
	int  dtypes_index;
	int  sizes[4];
};
struct model_info{
	struct tensor_info input;
	struct tensor_info output;
};
static const char* const types_names[] = {"kTfLiteNoType", "kTfLiteFloat32","kTfLiteInt32","kTfLiteUInt8"
	"kTfLiteInt64","kTfLiteString","kTfLiteBool","kTfLiteInt16","kTfLiteComplex64","kTfLiteInt8","kTfLiteFloat16"};
struct model_info * setup_NN_gcc(const unsigned char *model_data);

float * loop_NN_gcc(float * input_data);

void print_string_gcc(const char * str);
void print_string_f_gcc(const char * str,float f);
void print_string_f2_gcc(const char * str,float f1,float f2);

#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MAIN_FUNCTIONS_H_
