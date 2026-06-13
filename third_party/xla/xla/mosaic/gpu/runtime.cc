/* Copyright 2024 The JAX Authors.

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

#include <cassert>
#include <cstdint>
#include <cstdio>

#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/mosaic/gpu/nvshmem.h"

extern "C" {

void mosaic_gpu_init_tma_desc(CUtensorMap *tma_desc, void *base_addr,
                              int64_t elem_type, int64_t rank, int64_t *sizes,
                              int64_t *strides, int64_t swizzle_bytes,
                              int64_t *window_shape) {
  if (((uintptr_t)tma_desc) % 64 != 0) {
    fprintf(stderr,
            "TMA descriptor address must be 64 byte aligned, but got: %p\n",
            tma_desc);
    abort();
  }

  CUtensorMapDataType data_type;
  int64_t elem_bitwidth;
  // types are defined in: LaunchContext._get_tma_desc()
  if (elem_type == 0) {
    // this is for int4s
    data_type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    elem_bitwidth = 4;
  } else if (elem_type == 1) {
    data_type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    elem_bitwidth = 8;
  } else if (elem_type == 2) {
    data_type = CU_TENSOR_MAP_DATA_TYPE_UINT16;
    elem_bitwidth = 16;
  } else if (elem_type == 3) {
    data_type = CU_TENSOR_MAP_DATA_TYPE_UINT32;
    elem_bitwidth = 32;
  } else if (elem_type == 4) {
    data_type = CU_TENSOR_MAP_DATA_TYPE_UINT64;
    elem_bitwidth = 64;
  } else if (elem_type == 5) {
    data_type = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    elem_bitwidth = 16;
  } else if (elem_type == 6) {
    data_type = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    elem_bitwidth = 32;
  } else if (elem_type == 7) {
    data_type = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    elem_bitwidth = 16;
  } else {
    fprintf(stderr, "Unsupported element type: %ld \n", elem_type);
    abort();
  }

  // Pack 4 bit types in 8 bit pairs.
  int64_t elem_bytewidth;
  if (elem_bitwidth < 8) {
    // Check that it's a power of 2.
    assert((elem_bitwidth & (elem_bitwidth - 1)) == 0);
    int packing = 8 / elem_bitwidth;
    assert(sizes[rank - 1] % packing == 0);
    assert(window_shape[rank - 1] % packing == 0);
    assert(strides[rank - 1] == 1);

    // TMA requires that the last dimension be the contiguous one so we pack the
    // elements under that assumption.
    sizes[rank - 1] /= packing;
    window_shape[rank - 1] /= packing;
    for (int i = 0; i < rank - 1; i++) {
      strides[i] /= packing;
    }
    elem_bytewidth = 1;
  } else {
    elem_bytewidth = elem_bitwidth / 8;
  }

  if (rank < 1 || rank > 5) {
    fprintf(stderr, "Rank must be in [1, 5], but got %ld\n", rank);
    abort();
  }
  cuuint64_t tma_sizes[5] = {1, 1, 1, 1, 1};
  for (int i = 0; i < rank; ++i) {
    cuuint64_t tma_size_i = static_cast<cuuint64_t>(sizes[rank - i - 1]);
    if (tma_size_i > static_cast<cuuint64_t>(1) << 32) {
      fprintf(stderr,
              "TMA size must be less than 2**32, but got %ld at index %ld\n",
              tma_size_i, rank - i - 1);
      abort();
    }
    tma_sizes[i] = tma_size_i;
  }
  cuuint64_t tma_strides[5] = {1, 1, 1, 1, 1};
  if (strides[rank - 1] != 1) {
    fprintf(stderr, "Minormost stride must be 1, but got %ld\n",
            strides[rank - 1]);
    abort();
  }
  for (int i = 0; i < rank - 1; ++i) {  // We skip the implicit minor stride.
    cuuint64_t tma_stride_i =
        static_cast<cuuint64_t>(strides[rank - i - 2] * elem_bytewidth);
    if (tma_stride_i % 16 != 0 || tma_stride_i >= static_cast<cuuint64_t>(1)
                                                      << 40) {
      fprintf(stderr,
              "Byte strides must be divisible by 16 and less than 2**40, but "
              "got %ld (item stride = %ld, item size = %ld) at index %ld\n",
              tma_stride_i, strides[rank - 1], elem_bytewidth, rank - i - 2);
      abort();
    }
    tma_strides[i] = tma_stride_i;
  }
  cuuint32_t tma_window_shape[5] = {1, 1, 1, 1, 1};
  for (int64_t i = 0; i < rank; ++i) {
    cuuint32_t tma_window_shape_i =
        static_cast<cuuint32_t>(window_shape[rank - i - 1]);
    if (tma_window_shape_i > 256) {
      fprintf(stderr,
              "Window shape must be in [0, 256], but got %d at index %ld\n",
              tma_window_shape_i, rank - i - 1);
      abort();
    }
    if (i == 0 && (tma_window_shape_i * elem_bytewidth) % 16 != 0) {
      fprintf(stderr,
              "The last dimension of window shape must have a bytewidth "
              "divisible by 16, but got %d*%ld at index %ld\n",
              tma_window_shape_i, elem_bytewidth, rank - i - 1);
      abort();
    }
    tma_window_shape[i] = tma_window_shape_i;
  }
  cuuint32_t element_strides[5] = {1, 1, 1, 1, 1};
  CUtensorMapSwizzle swizzle;
  if (swizzle_bytes == 16) {
    swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
  } else if (swizzle_bytes == 32) {
    swizzle = CU_TENSOR_MAP_SWIZZLE_32B;
  } else if (swizzle_bytes == 64) {
    swizzle = CU_TENSOR_MAP_SWIZZLE_64B;
  } else if (swizzle_bytes == 128) {
    swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
  } else {
    fprintf(stderr, "Unsupported swizzle: %ld\n", swizzle_bytes);
    abort();
  }
  CUresult result = cuTensorMapEncodeTiled(
      tma_desc, data_type, rank, base_addr, tma_sizes, tma_strides,
      tma_window_shape, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (result != CUDA_SUCCESS) {
    const char *ptr = nullptr;
    cuGetErrorString(result, &ptr);
    fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", ptr);
    abort();
  }
}

void *mosaic_gpu_module_load(void *data) {
  CUmodule module = nullptr;
  if (auto result = cuModuleLoadData(&module, data); result != CUDA_SUCCESS) {
    const char *ptr = nullptr;
    cuGetErrorString(result, &ptr);
    fprintf(stderr, "cuModuleLoadData failed: %s\n", ptr);
    abort();
  }

  {  // Set the NVSHMEM state if it's used by the module.
    CUdeviceptr ptr = 0;
    size_t size = 0;
    if (cuModuleGetGlobal(&ptr, &size, module,
                          "nvshmemi_device_lib_version_d") == CUDA_SUCCESS) {
      if (mosaic::gpu::NvshmemApi::Default().cumodule_init(module) !=
          NVSHMEM_SUCCESS) {
        fprintf(stderr, "nvshmemx_cumodule_init failed.\n");
        abort();
      }
    }
  }

  return module;
}

// cluster_size can be -1 when it's not statically known.
void *mosaic_gpu_get_function(CUmodule module, const char *name,
                              int32_t smem_bytes, int32_t cluster_size) {
  CUfunction function = nullptr;
  CUresult result = cuModuleGetFunction(&function, module, name);
  if (result != CUDA_SUCCESS) {
    const char *ptr = nullptr;
    cuGetErrorString(result, &ptr);
    fprintf(stderr, "cuModuleGetFunction failed: %s\n", ptr);
    abort();
  }
  if (smem_bytes) {
    result = cuFuncSetAttribute(
        function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes);
    if (result != CUDA_SUCCESS) {
      const char *ptr = nullptr;
      cuGetErrorString(result, &ptr);
      fprintf(stderr, "cuFuncSetAttribute failed: %s\n", ptr);
      abort();
    }
  }
  if (cluster_size > 8) {
    result = cuFuncSetAttribute(
        function, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1);
    if (result != CUDA_SUCCESS) {
      const char *ptr = nullptr;
      cuGetErrorString(result, &ptr);
      fprintf(stderr, "cuFuncSetAttribute failed: %s\n", ptr);
      abort();
    }
  }
  return function;
}

void mosaic_gpu_launch_kernel(CUfunction function, uint32_t grid_x,
                              uint32_t grid_y, uint32_t grid_z,
                              uint32_t cluster_x, uint32_t cluster_y,
                              uint32_t cluster_z, uint32_t block_x,
                              uint32_t block_y, uint32_t block_z,
                              uint32_t smem_bytes, CUstream stream,
                              void **params) {
  CUlaunchConfig config{
      .gridDimX = grid_x,
      .gridDimY = grid_y,
      .gridDimZ = grid_z,
      .blockDimX = block_x,
      .blockDimY = block_y,
      .blockDimZ = block_z,
      .sharedMemBytes = smem_bytes,
      .hStream = stream,
      .attrs = nullptr,
      .numAttrs = 0,
  };
  CUlaunchAttribute cluster_attr;
  if (cluster_x != 0) {
    cluster_attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    cluster_attr.value.clusterDim = {
        .x = cluster_x,
        .y = cluster_y,
        .z = cluster_z,
    };
    config.attrs = &cluster_attr;
    config.numAttrs = 1;
  }
  CUresult result = cuLaunchKernelEx(&config, function, params, nullptr);
  if (result != CUDA_SUCCESS) {
    const char *ptr = nullptr;
    cuGetErrorString(result, &ptr);
    fprintf(stderr, "cuLaunchKernel failed: %s\n", ptr);
    abort();
  }
}
}
