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

#include "tensorflow/lite/micro/kernels/arc/scratch_buffers.h"
#include <limits.h>

/* by default use all the XY memory, and half of the DCCM because DCCM is also used
 * for the data section and the stack.
 * the values can be overruled by adding a -D option to the makefile of the application
 */
#ifndef SCRATCH_MEM_X_SIZE
#ifdef core_config_xy_size
#define SCRATCH_MEM_X_SIZE (core_config_xy_size)
#else
#define SCRATCH_MEM_X_SIZE (0)
#endif
#endif

#ifndef SCRATCH_MEM_Y_SIZE
#ifdef core_config_xy_size
#define SCRATCH_MEM_Y_SIZE (core_config_xy_size)
#else
#define SCRATCH_MEM_Y_SIZE (0)
#endif
#endif

#ifndef SCRATCH_MEM_Z_SIZE
#ifdef core_config_dccm_size
#define SCRATCH_MEM_Z_SIZE ((core_config_dccm_size) / 2)
#else
#define SCRATCH_MEM_Z_SIZE (0)
#endif
#endif

namespace {
#pragma Bss(".Xdata")
    static int8_t scratch_mem_x[SCRATCH_MEM_X_SIZE];
#pragma Bss()

#pragma Bss(".Ydata")
    static int8_t scratch_mem_y[SCRATCH_MEM_Y_SIZE];
#pragma Bss()

#pragma Bss(".Zdata")
    static int8_t scratch_mem_z[SCRATCH_MEM_Z_SIZE];
#pragma Bss()
}

static inline
bool inside_arc_dccm(void* p) {
#if core_config_dccm_present
  return ((unsigned)p >= core_config_dccm_base) && ((unsigned)p < core_config_dccm_base + core_config_dccm_size);
#else
  return false;
#endif
}
static inline
bool inside_arc_xccm(void* p) {
#if core_config_xy
  return ((unsigned)p >= core_config_xy_x_base) && ((unsigned)p < core_config_xy_x_base + core_config_xy_size);
#else
  return false;
#endif
}
static inline
bool inside_arc_yccm(void* p) {
#if core_config_xy
  return ((unsigned)p >= core_config_xy_y_base) && ((unsigned)p < core_config_xy_y_base + core_config_xy_size);
#else
  return false;
#endif
}

static inline
bool inside_arc_ccm(void* p) {
  return inside_arc_dccm(p) || inside_arc_xccm(p) || inside_arc_yccm(p);
}

TfLiteStatus get_arc_scratch_buffer_for_conv_tensors(TfLiteContext* context,
    mli_tensor* in, 
    mli_tensor* weights, 
    mli_tensor* bias, 
    mli_tensor* out) {
#ifdef __Xxy
  // Function to assign fast memory from one of 3 scratch buffers.
  // Best Fit strategy - memory is asigned to those tensor which leave less memory of bank unused
  mli_tensor* tensors[3] = { weights, in, out };
  uint32_t tensor_sizes[3] = {
    mli_hlp_count_elem_num(tensors[0], 0), mli_hlp_count_elem_num(tensors[1], 0), mli_hlp_count_elem_num(tensors[2], 0) };
  bool mem_is_free[3] = { true, true, true };
  int8_t* scratch_mem[] = {scratch_mem_x, scratch_mem_y, scratch_mem_z};
  uint32_t scratch_sizes[] = {SCRATCH_MEM_X_SIZE, SCRATCH_MEM_Y_SIZE, SCRATCH_MEM_Z_SIZE};

  for (int i = 0; i < 3; ++i) {
    int best_mem_idx = -1;
    int best_mem_delta = INT_MAX;
	// only for tensors that are not already located in one of the ccm memories, find a local memory that fits the data size.
	if (inside_arc_ccm(tensors[i]->data)) continue;
    for (int j = 0; j < 3; ++j) {
       // Best Fit
       if (mem_is_free[j] && tensor_sizes[i] <= scratch_sizes[j] && scratch_sizes[j] - tensor_sizes[i] < best_mem_delta) {
          best_mem_idx = j;
          best_mem_delta = scratch_sizes[j] - tensor_sizes[i];
       }
    }
    if (best_mem_idx >= 0) {
      tensors[i]->data = static_cast<void*>(scratch_mem[best_mem_idx]);
      tensors[i]->capacity = scratch_sizes[best_mem_idx];
      mem_is_free[best_mem_idx] = false;
    } else {
        return kTfLiteError;
    }
  }

  // Bias is expected to be much smaller than other operands, not affect performance and can be placed 
  // in the end of some of already used memory bank (to occupy free space of it)
  bool is_bias_allocated = inside_arc_ccm(bias->data);
  if (!is_bias_allocated) {
    uint32_t bias_mem_requirements = mli_hlp_count_elem_num(bias, 0) * mli_hlp_tensor_element_size(bias);
    for (int i = 0; i < 3; ++i) {
      if (tensors[i]->capacity - tensor_sizes[i] > bias_mem_requirements) {
        bias->data = &((char*)tensors[i]->data)[tensor_sizes[i]];
        bias->capacity = bias_mem_requirements;
        tensors[i]->capacity = tensor_sizes[i];
        is_bias_allocated = true;
        break;
      }
    }
  }
  if (!is_bias_allocated) {
    uint32_t bias_mem_requirements = mli_hlp_count_elem_num(bias, 0) * mli_hlp_tensor_element_size(bias);
    for (int i = 0; i < 3; ++i) {
      if (mem_is_free[i]) {
		  bias->data = static_cast<void*>(scratch_mem[i]);
		  bias->capacity = bias_mem_requirements;
        is_bias_allocated = true;
        break;
	  }
    }
  }
  return (is_bias_allocated) ? kTfLiteOk : kTfLiteError;
#else
  return kTfLiteOk;
#endif
}

TfLiteStatus get_arc_scratch_buffer_for_io_tensors(TfLiteContext* context,
    mli_tensor* in, 
    mli_tensor* out) {
#ifdef __Xxy
  // Function to assign fast memory from one of 3 scratch buffers.
  // Best Fit strategy - memory is asigned to those tensor which leave less memory of bank unused
  mli_tensor* tensors[2] = { in, out };
  uint32_t tensor_sizes[2] = {
    mli_hlp_count_elem_num(tensors[0], 0), mli_hlp_count_elem_num(tensors[1], 0)};
  bool mem_is_free[3] = { true, true, true };
  int8_t* scratch_mem[] = {scratch_mem_x, scratch_mem_y, scratch_mem_z};
  uint32_t scratch_sizes[] = {SCRATCH_MEM_X_SIZE, SCRATCH_MEM_Y_SIZE, SCRATCH_MEM_Z_SIZE};
  int num_tensors = 2;
  int num_memories = 3;
  

  for (int i = 0; i < num_tensors; ++i) {
    int best_mem_idx = -1;
    int best_mem_delta = INT_MAX;
	// only for tensors that are not already located in one of the ccm memories, find a local memory that fits the data size.
	if (inside_arc_ccm(tensors[i]->data)) continue;
    for (int j = 0; j < num_memories; ++j) {
       // Best Fit
       if (mem_is_free[j] && tensor_sizes[i] <= scratch_sizes[j] && scratch_sizes[j] - tensor_sizes[i] < best_mem_delta) {
          best_mem_idx = j;
          best_mem_delta = scratch_sizes[j] - tensor_sizes[i];
       }
    }
    if (best_mem_idx >= 0) {
      tensors[i]->data = static_cast<void*>(scratch_mem[best_mem_idx]);
      tensors[i]->capacity = scratch_sizes[best_mem_idx];
      mem_is_free[best_mem_idx] = false;
    } else {
        return kTfLiteError;
    }
  }
#endif
  return kTfLiteOk;
}