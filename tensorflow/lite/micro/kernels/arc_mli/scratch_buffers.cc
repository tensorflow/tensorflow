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

#include "tensorflow/lite/micro/kernels/arc_mli/scratch_buffers.h"

#include <limits.h>

namespace tflite {
namespace ops {
namespace micro {

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

static int8_t* scratch_mem[] = {scratch_mem_x, scratch_mem_y, scratch_mem_z};
static uint32_t scratch_sizes[] = {SCRATCH_MEM_X_SIZE, SCRATCH_MEM_Y_SIZE, SCRATCH_MEM_Z_SIZE};


void *get_arc_scratch_buffer(int size) {
  // Function to asign fast memory from one of 3 scratch buffers.
  // Best Fit strategy - memory is allocated from that memory bank that leaves the least unused memory.
  void *buf = NULL;
  int best_mem_idx = -1;
  int best_mem_delta = INT_MAX;
  const int num_mem = sizeof(scratch_mem)/sizeof(scratch_mem[0]);
  // find a local memory that fits the data size.
  for (int mem_idx = 0; mem_idx < num_mem; ++mem_idx) {
    // Best Fit
    if ((size <= scratch_sizes[mem_idx]) && (scratch_sizes[mem_idx] - size < best_mem_delta)) {
      best_mem_idx = mem_idx;
      best_mem_delta = scratch_sizes[mem_idx] - size;
    }
  }
  if (best_mem_idx >= 0) {
    buf = static_cast<void*>(scratch_mem[best_mem_idx]);
    scratch_mem[best_mem_idx] += size;
    scratch_sizes[best_mem_idx] -= size;
  }
  return buf;
}

void get_arc_scratch_buffer_max_size(int *size) {
  int maxavailable = 0;
  const int num_mem = sizeof(scratch_mem)/sizeof(scratch_mem[0]);
  // find the largest available buffer.
  for (int i = 0; i < num_mem; i++) {
    if (scratch_sizes[i] > maxavailable) {
      maxavailable = scratch_sizes[i];
    }
  }
  *size = maxavailable;
}

void get_arc_scratch_buffer_two_max_sizes(int *size1, int *size2) {
  int maxavailable = 0;
  int secondavail = 0;
  const int num_mem = sizeof(scratch_mem)/sizeof(scratch_mem[0]);
  // find the two largest available buffers.
  for (int i = 0; i < num_mem; i++) {
    if (scratch_sizes[i] > maxavailable) {
      secondavail = maxavailable;
      maxavailable = scratch_sizes[i];
    } else if (scratch_sizes[i] > secondavail) {
      secondavail = scratch_sizes[i];
    }
  }
  *size1 = maxavailable;
  *size2 = secondavail;
}

void init_arc_scratch_buffers(void) {
  scratch_mem[0] = scratch_mem_x;
  scratch_mem[1] = scratch_mem_y;
  scratch_mem[2] = scratch_mem_z;
  scratch_sizes[0] = SCRATCH_MEM_X_SIZE;
  scratch_sizes[1] = SCRATCH_MEM_Y_SIZE;
  scratch_sizes[2] = SCRATCH_MEM_Z_SIZE;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite