/* Copyright 2020 The OpenXLA Authors.

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

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "xla/stream_executor/cuda/cuda_diagnostics.h"
#include "xla/stream_executor/cuda/cuda_status.h"

namespace stream_executor {
namespace cuda {

void CheckCuda(CUresult result, const char* file, int line) {
  CHECK_OK(cuda::ToStatus(result));
}

void CheckCuda(cudaError_t result, const char* file, int line) {
  if (result == cudaSuccess) {
    return;
  }
  const char* name = cudaGetErrorName(result);
  const char* message = cudaGetErrorString(result);
  LOG(FATAL) << file << "(" << line << "): " << name << ", " << message;
}

#define CHECK_CUDA(result) CheckCuda(result, __FILE__, __LINE__)

class CudaDriverTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { CHECK_CUDA(cuInit(0)); }
};

TEST_F(CudaDriverTest, DriverVersionParsingTest) {
  // Tests that the driver version can be right after 'Kernel Module',
  // or later as well.
  auto driver_version = Diagnostician::FindKernelModuleVersion(
      "... NVIDIA UNIX Open Kernel Module for x86_64  570.00  Release Build  "
      "...  Mon Aug 12 04:17:20 UTC 2024");
  CHECK_OK(driver_version.status());
  EXPECT_EQ("570.0.0", cuda::DriverVersionToString(driver_version.value()));

  driver_version = Diagnostician::FindKernelModuleVersion(
      "... NVIDIA UNIX Open Kernel Module  571.00  Release Build  "
      "...  Mon Aug 12 04:17:20 UTC 2024");
  CHECK_OK(driver_version.status());
  EXPECT_EQ("571.0.0", cuda::DriverVersionToString(driver_version.value()));
}

namespace {

// The outcome of a driver query that returns its result through a `void*`.
struct QueryResult {
  CUresult result;
  // The number of bytes that the driver wrote into the result buffer (more
  // precisely: the offset of the last written byte plus one).
  size_t write_size;
};

// Measures how many bytes the driver writes when running `query`, which is a
// callable that takes a `void*` result buffer and returns a `CUresult`.
//
// We run the query twice: once with a buffer filled with 0x00 and once with a
// buffer filled with 0xFF. A byte counts as written if it differs from the
// fill pattern in at least one of the two buffers.
template <typename Query>
QueryResult MeasureQuery(Query query) {
  constexpr size_t kBufferSize = 32;
  alignas(uint64_t) std::array<uint8_t, kBufferSize> zeros;
  alignas(uint64_t) std::array<uint8_t, kBufferSize> ones;
  zeros.fill(0x00);
  ones.fill(0xFF);
  CUresult result_zeros = query(zeros.data());
  CUresult result_ones = query(ones.data());
  EXPECT_EQ(result_ones, result_zeros);

  size_t write_size = 0;
  for (size_t i = 0; i < kBufferSize; ++i) {
    if (zeros[i] != 0x00 || ones[i] != 0xFF) {
      write_size = i + 1;
    }
  }
  return {result_zeros, write_size};
}

// A pointer attribute that stream executor queries, together with the size of
// the variable that stream executor passes to the driver for the result.
struct PointerAttribute {
  const char* name;
  CUpointer_attribute attribute;
  size_t size;
};

// All pointer attributes that stream executor queries with
// `cuPointerGetAttribute` or `cuPointerGetAttributes`. When you add a new query
// to stream executor, please also add the attribute here.
constexpr PointerAttribute kPointerAttributes[] = {
    // Queried by `IsFabricMemory` in cuda_stream.cc. The documentation doesn't
    // mention the type of the result, and all CUmemAllocationHandleType values
    // fit into 32 bits. But the driver writes a 64-bit value, so querying the
    // attribute into a 32-bit variable overwrites the 4 bytes after it. (This
    // caused a stack buffer overflow in `IsFabricMemory`.)
    {"AllowedHandleTypes", CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES,
     sizeof(uint64_t)},
    // Queried by `CudaContext::GetContextMap` in cuda_context.cc.
    {"DeviceOrdinal", CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, sizeof(int)},
    // Queried by `CudaExecutor::GetPointerMemorySpace` in cuda_executor.cc.
    {"IsManaged", CU_POINTER_ATTRIBUTE_IS_MANAGED, sizeof(unsigned int)},
    // Queried by `CudaExecutor::GetPointerMemorySpace` and
    // `CudaExecutor::IsHostMemoryPinned` in cuda_executor.cc.
    {"MemoryType", CU_POINTER_ATTRIBUTE_MEMORY_TYPE, sizeof(unsigned int)},
    // Queried by `CudaExecutor::IsHostMemoryPinned` in cuda_executor.cc.
    {"RangeStartAddr", CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, sizeof(uint64_t)},
    {"RangeSize", CU_POINTER_ATTRIBUTE_RANGE_SIZE, sizeof(uint64_t)},
    // Queried by `CudaExecutor::GetAllocationRange` in cuda_executor.cc.
    {"MappingBaseAddr", CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR,
     sizeof(uint64_t)},
    // Queried by `CudaExecutor::GetAllocationRange` and
    // `CudaExecutor::ExportFabricHandle` in cuda_executor.cc.
    {"MappingSize", CU_POINTER_ATTRIBUTE_MAPPING_SIZE, sizeof(uint64_t)},
};

// Allocates all kinds of memory that stream executor might query pointer
// attributes for.
class PointerAttributeTest
    : public CudaDriverTest,
      public ::testing::WithParamInterface<PointerAttribute> {
 protected:
  struct Allocation {
    const char* kind;
    CUdeviceptr ptr;
  };

  static constexpr size_t kSize = 1024;

  void SetUp() override {
    CHECK_CUDA(cuDeviceGet(&device_, 0));
    CUcontext context;
    CHECK_CUDA(cuDevicePrimaryCtxRetain(&context, device_));
    CHECK_CUDA(cuCtxSetCurrent(context));

    CHECK_CUDA(cuMemAlloc(&device_memory_, kSize));
    allocations_.push_back({"device memory (cuMemAlloc)", device_memory_});

    CHECK_CUDA(cuMemHostAlloc(&pinned_host_memory_, kSize, 0));
    allocations_.push_back(
        {"pinned host memory (cuMemHostAlloc)",
         reinterpret_cast<CUdeviceptr>(pinned_host_memory_)});

    // The driver doesn't know about pageable host memory, but
    // `CudaExecutor::IsHostMemoryPinned` queries arbitrary host pointers.
    pageable_host_memory_.resize(kSize);
    allocations_.push_back(
        {"pageable host memory",
         reinterpret_cast<CUdeviceptr>(pageable_host_memory_.data())});

    if (GetDeviceAttribute(CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY)) {
      CHECK_CUDA(
          cuMemAllocManaged(&managed_memory_, kSize, CU_MEM_ATTACH_GLOBAL));
      allocations_.push_back(
          {"managed memory (cuMemAllocManaged)", managed_memory_});
    }

    if (GetDeviceAttribute(
            CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED)) {
      AllocateVmmMemory();
      allocations_.push_back({"VMM memory (cuMemCreate)", vmm_memory_});
    }
  }

  void TearDown() override {
    if (vmm_memory_ != 0) {
      CHECK_CUDA(cuMemUnmap(vmm_memory_, vmm_size_));
      CHECK_CUDA(cuMemAddressFree(vmm_memory_, vmm_size_));
      CHECK_CUDA(cuMemRelease(vmm_handle_));
    }
    if (managed_memory_ != 0) {
      CHECK_CUDA(cuMemFree(managed_memory_));
    }
    CHECK_CUDA(cuMemFreeHost(pinned_host_memory_));
    CHECK_CUDA(cuMemFree(device_memory_));
    CHECK_CUDA(cuDevicePrimaryCtxRelease(device_));
  }

  std::vector<Allocation> allocations_;

 private:
  int GetDeviceAttribute(CUdevice_attribute attribute) {
    int value = 0;
    CHECK_CUDA(cuDeviceGetAttribute(&value, attribute, device_));
    return value;
  }

  void AllocateVmmMemory() {
    CUmemAllocationProp properties = {};
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = device_;
    CHECK_CUDA(cuMemGetAllocationGranularity(&vmm_size_, &properties,
                                             CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    CHECK_CUDA(cuMemCreate(&vmm_handle_, vmm_size_, &properties, 0));
    CHECK_CUDA(cuMemAddressReserve(&vmm_memory_, vmm_size_, 0, 0, 0));
    CHECK_CUDA(cuMemMap(vmm_memory_, vmm_size_, 0, vmm_handle_, 0));
    CUmemAccessDesc access = {};
    access.location = properties.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_CUDA(cuMemSetAccess(vmm_memory_, vmm_size_, &access, 1));
  }

  CUdevice device_ = 0;
  CUdeviceptr device_memory_ = 0;
  void* pinned_host_memory_ = nullptr;
  std::vector<char> pageable_host_memory_;
  CUdeviceptr managed_memory_ = 0;
  CUdeviceptr vmm_memory_ = 0;
  size_t vmm_size_ = 0;
  CUmemGenericAllocationHandle vmm_handle_ = 0;
};

TEST_P(PointerAttributeTest, WriteSizeMatchesStreamExecutorVariable) {
  CUpointer_attribute attribute = GetParam().attribute;
  const size_t expected_size = GetParam().size;
  int num_written_results = 0;
  for (const Allocation& allocation : allocations_) {
    SCOPED_TRACE(allocation.kind);
    for (bool use_get_attributes : {false, true}) {
      SCOPED_TRACE(use_get_attributes ? "cuPointerGetAttributes"
                                      : "cuPointerGetAttribute");
      QueryResult query = MeasureQuery([&](void* data) {
        if (use_get_attributes) {
          return cuPointerGetAttributes(1, &attribute, &data, allocation.ptr);
        }
        return cuPointerGetAttribute(data, attribute, allocation.ptr);
      });

      // The driver must never write past the end of the variable, not even if
      // the query fails.
      EXPECT_LE(query.write_size, expected_size);

      // If the query succeeds, the driver either writes the whole variable or
      // doesn't touch it at all. (The latter happens for RANGE_START_ADDR and
      // RANGE_SIZE when querying pageable host memory with
      // `cuPointerGetAttributes`. `CudaExecutor::IsHostMemoryPinned` only reads
      // them if the memory type is CU_MEMORYTYPE_HOST, and for pageable host
      // memory the driver reports a memory type of 0.)
      if (query.result == CUDA_SUCCESS && query.write_size > 0) {
        ++num_written_results;
        EXPECT_EQ(query.write_size, expected_size);
      }
    }
  }
  // Make sure that the test doesn't pass just because the driver never wrote
  // a result.
  EXPECT_GT(num_written_results, 0);
}

INSTANTIATE_TEST_SUITE_P(
    StreamExecutorPointerAttributes, PointerAttributeTest,
    ::testing::ValuesIn(kPointerAttributes),
    [](const ::testing::TestParamInfo<PointerAttribute>& info) {
      return std::string(info.param.name);
    });

// A memory pool attribute that stream executor queries.
struct MemPoolAttribute {
  const char* name;
  CUmemPool_attribute attribute;
};

TEST_F(CudaDriverTest, MemPoolAttributeWriteSizeMatchesStreamExecutorVariable) {
  CUdevice device;
  CHECK_CUDA(cuDeviceGet(&device, 0));
  CUcontext context;
  CHECK_CUDA(cuDevicePrimaryCtxRetain(&context, device));
  CHECK_CUDA(cuCtxSetCurrent(context));
  CUmemoryPool pool;
  CHECK_CUDA(cuDeviceGetDefaultMemPool(&pool, device));

  // All memory pool attributes that stream executor queries with
  // `cuMemPoolGetAttribute` (in gpu_cudamallocasync_allocator.cc). The
  // documentation says that all of them are `cuuint64_t`, and that's also what
  // stream executor uses.
  constexpr MemPoolAttribute kMemPoolAttributes[] = {
      {"RELEASE_THRESHOLD", CU_MEMPOOL_ATTR_RELEASE_THRESHOLD},
      {"RESERVED_MEM_CURRENT", CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT},
      {"RESERVED_MEM_HIGH", CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH},
      {"USED_MEM_CURRENT", CU_MEMPOOL_ATTR_USED_MEM_CURRENT},
      {"USED_MEM_HIGH", CU_MEMPOOL_ATTR_USED_MEM_HIGH},
  };
  for (const MemPoolAttribute& attribute : kMemPoolAttributes) {
    SCOPED_TRACE(attribute.name);
    QueryResult query = MeasureQuery([&](void* data) {
      return cuMemPoolGetAttribute(pool, attribute.attribute, data);
    });
    EXPECT_EQ(query.result, CUDA_SUCCESS);
    EXPECT_EQ(query.write_size, sizeof(cuuint64_t));
  }

  CHECK_CUDA(cuDevicePrimaryCtxRelease(device));
}

}  // namespace

}  // namespace cuda

}  // namespace stream_executor
