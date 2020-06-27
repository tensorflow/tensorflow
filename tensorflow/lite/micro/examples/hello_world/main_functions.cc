/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
//#define MC 

#ifdef MC
#include "am_mcu_apollo.h"
#include "am_bsp.h"
#include "am_util.h"
#endif

#include "tensorflow/lite/micro/examples/hello_world/main_functions.h"

#include "tensorflow/lite/micro/examples/hello_world/constants.h"
//#include "tensorflow/lite/micro/examples/hello_world/model.h"
#include "tensorflow/lite/micro/examples/hello_world/output_handler.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// Minimum arena size, at the time of writing. After allocating tensors
// you can retrieve this value by invoking interpreter.arena_used_bytes().
const int kModelArenaSize = 20000;
// Extra headroom for model + alignment + future interpreter changes.
const int kExtraArenaSize = 560 + 16 + 1000;
const int kTensorArenaSize = kModelArenaSize + kExtraArenaSize;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace
///////////////////////////////////////////////////////////////////////




// The name of this function is important for Arduino compatibility.
struct model_info * setup_NN_gcc(const unsigned char *model_data) {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  bool debug=1;

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
//error_reporter =NULL;
  static struct model_info info;
#ifdef MC
  am_bsp_uart_printf_enable(); 
  am_util_stdio_terminal_clear(); 
#endif
  error_reporter->Report("\rsetup_NN invoked \n");

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return NULL;
  }
  else 
	error_reporter->Report("Model TFLITE_SCHEMA_VERSION OK\n");

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  error_reporter->Report("interpreter OK");

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return NULL;
  }

  else
	error_reporter->Report("AllocateTensors OK\n");

 // get_model_info(interpreter,error_reporter);

	input = interpreter->input(0);


	info.input.name="input";
  	info.input.dim =input->dims->size;
	info.input.dtypes_index= input->type;

  	error_reporter->Report("\rin_dim= %d\n",info.input.dim);
  	error_reporter->Report("\rinput_type = %s\n",types_names[info.input.dtypes_index]);
	for(int ii=0;ii<info.input.dim;ii++)
	{
		info.input.sizes[ii]=input->dims->data[ii];
		error_reporter->Report("\rin_size[%d]= %d\n",ii,info.input.sizes[ii]);
	}

  	output = interpreter->output(0);
	info.output.name="output";	
	info.output.dim =output->dims->size;
	info.output.dtypes_index=output->type;

	error_reporter->Report("\r\nout_dim= %d\n",info.output.dim);
  	error_reporter->Report("\routput_type = %s\n",types_names[info.output.dtypes_index]);
	for(int ii=0;ii<info.output.dim;ii++)
	{
		info.output.sizes[ii]=output->dims->data[ii];
		error_reporter->Report("\rout_size[%d]= %d\n",ii,info.output.sizes[ii]);
  	}
  
  	error_reporter->Report("\r\nSetup OK\n");
	
	input = interpreter->input(0);
	output = interpreter->output(0);

	inference_count = 0;
	return &info;
  // Keep track of how many inferences we have performed.
  
}

// The name of this function is important for Arduino compatibility.
float * loop_NN_gcc(float * input_data) {
	
	input->data.f=input_data;
	TfLiteStatus invoke_status = interpreter->Invoke();
  	if (invoke_status != kTfLiteOk) 
	{
    		error_reporter->Report("Invoke failed on x_val:\n");
    		return NULL;
  	}
	float *y = output->data.f;
	//error_reporter->Report("sin(%f)=%f\n",x,y);
	return y;
}


void print_string_gcc(const char * str){  error_reporter->Report(str);}
void print_string_f_gcc(const char * str,float f){  error_reporter->Report(str,f);}
void print_string_f2_gcc(const char * str,float f1,float f2){  error_reporter->Report(str,f1,f2);}
