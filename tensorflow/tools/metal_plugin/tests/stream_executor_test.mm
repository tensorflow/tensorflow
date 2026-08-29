/* Copyright 2026 The TensorFlow Metal Plugin Authors. All Rights Reserved.

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

// Drives the StreamExecutor C API directly, without TensorFlow.
//
// Some of what this backend implements is reachable only from TensorFlow's
// runtime and not from any op: memset32 with a pattern that is not four equal
// bytes has exactly one caller in the whole tree, in a CUDA-only kernel, so a
// Python test cannot exercise it on this device at all. Rather than ship it
// unverified, this stands where TensorFlow would stand and calls the same
// entry points in the same order.

#include <dlfcn.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/tf_status.h"

namespace {

int failures = 0;

void Check(const std::string& name, bool condition,
           const std::string& detail = "") {
  std::printf("  %-44s %s%s%s\n", name.c_str(), condition ? "ok" : "FAILED",
              detail.empty() ? "" : "  ", detail.c_str());
  if (!condition) ++failures;
}

// One device, one stream, and the pieces needed to reach them, torn down in
// reverse on scope exit so a failing check does not leak a device.
struct Backend {
  void* handle = nullptr;
  SP_Platform platform = {};
  SP_PlatformFns platform_fns = {};
  void (*destroy_platform)(SP_Platform*) = nullptr;
  void (*destroy_platform_fns)(SP_PlatformFns*) = nullptr;
  SP_Device device = {};
  SP_DeviceFns device_fns = {};
  SP_StreamExecutor executor = {};
  SP_Stream stream = nullptr;

  ~Backend() {
    if (stream != nullptr && executor.destroy_stream != nullptr) {
      executor.destroy_stream(&device, stream);
    }
    if (platform_fns.destroy_device != nullptr) {
      platform_fns.destroy_device(&platform, &device);
    }
    if (platform_fns.destroy_device_fns != nullptr) {
      platform_fns.destroy_device_fns(&platform, &device_fns);
    }
    if (platform_fns.destroy_stream_executor != nullptr) {
      platform_fns.destroy_stream_executor(&platform, &executor);
    }
    if (destroy_platform_fns != nullptr) destroy_platform_fns(&platform_fns);
    if (destroy_platform != nullptr) destroy_platform(&platform);
    if (handle != nullptr) dlclose(handle);
  }
};

bool Open(Backend* backend, const char* path, TF_Status* status) {
  backend->handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (backend->handle == nullptr) {
    std::printf("could not load %s: %s\n", path, dlerror());
    return false;
  }
  auto init = reinterpret_cast<void (*)(SE_PlatformRegistrationParams*,
                                        TF_Status*)>(
      dlsym(backend->handle, "SE_InitPlugin"));
  if (init == nullptr) {
    std::printf("the library exports no SE_InitPlugin\n");
    return false;
  }

  SE_PlatformRegistrationParams params = {};
  params.struct_size = SE_PLATFORM_REGISTRATION_PARAMS_STRUCT_SIZE;
  params.platform = &backend->platform;
  params.platform_fns = &backend->platform_fns;
  init(&params, status);
  if (TF_GetCode(status) != TF_OK) {
    std::printf("SE_InitPlugin failed: %s\n", TF_Message(status));
    return false;
  }
  backend->destroy_platform = params.destroy_platform;
  backend->destroy_platform_fns = params.destroy_platform_fns;

  SE_CreateDeviceParams device_params = {};
  device_params.struct_size = SE_CREATE_DEVICE_PARAMS_STRUCT_SIZE;
  device_params.device = &backend->device;
  backend->device.struct_size = SP_DEVICE_STRUCT_SIZE;
  backend->platform_fns.create_device(&backend->platform, &device_params,
                                      status);
  if (TF_GetCode(status) != TF_OK) {
    std::printf("create_device failed: %s\n", TF_Message(status));
    return false;
  }

  SE_CreateDeviceFnsParams fns_params = {};
  fns_params.struct_size = SE_CREATE_DEVICE_FNS_PARAMS_STRUCT_SIZE;
  fns_params.device_fns = &backend->device_fns;
  backend->device_fns.struct_size = SP_DEVICE_FNS_STRUCT_SIZE;
  backend->platform_fns.create_device_fns(&backend->platform, &fns_params,
                                          status);
  if (TF_GetCode(status) != TF_OK) return false;

  SE_CreateStreamExecutorParams executor_params = {};
  executor_params.struct_size = SE_CREATE_STREAM_EXECUTOR_PARAMS_STRUCT_SIZE;
  executor_params.stream_executor = &backend->executor;
  backend->executor.struct_size = SP_STREAMEXECUTOR_STRUCT_SIZE;
  backend->platform_fns.create_stream_executor(&backend->platform,
                                               &executor_params, status);
  if (TF_GetCode(status) != TF_OK) return false;

  backend->executor.create_stream(&backend->device, &backend->stream, status);
  return TF_GetCode(status) == TF_OK;
}

// Fills `words` 32-bit words with `pattern` and reports what came back.
//
// Every allocation this backend makes is shared storage, so the host can read
// the result without a copy; that is what makes the fill checkable at all.
std::vector<uint32_t> Fill32(Backend* backend, uint32_t pattern, size_t words,
                             size_t word_offset, TF_Status* status) {
  const uint64_t bytes = (words + word_offset) * sizeof(uint32_t);
  SP_DeviceMemoryBase memory = {};
  memory.struct_size = SP_DEVICE_MEMORY_BASE_STRUCT_SIZE;
  backend->executor.allocate(&backend->device, bytes, 0, &memory);
  if (memory.opaque == nullptr) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED, "allocation failed");
    return {};
  }
  std::memset(memory.opaque, 0xAB, bytes);

  SP_DeviceMemoryBase target = memory;
  target.opaque = static_cast<uint32_t*>(memory.opaque) + word_offset;
  target.size = words * sizeof(uint32_t);
  backend->executor.memset32(&backend->device, backend->stream, &target,
                             pattern, words * sizeof(uint32_t), status);
  std::vector<uint32_t> out;
  if (TF_GetCode(status) == TF_OK) {
    backend->executor.block_host_until_done(&backend->device, backend->stream,
                                            status);
  }
  if (TF_GetCode(status) == TF_OK) {
    const auto* data = static_cast<const uint32_t*>(memory.opaque);
    out.assign(data, data + words + word_offset);
  }
  backend->executor.deallocate(&backend->device, &memory);
  return out;
}

bool All(const std::vector<uint32_t>& values, size_t from, size_t to,
         uint32_t want) {
  for (size_t i = from; i < to; ++i) {
    if (values[i] != want) return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "build/libmetal_plugin.dylib";
  TF_Status* status = TF_NewStatus();
  Backend backend;
  if (!Open(&backend, path, status)) {
    TF_DeleteStatus(status);
    return 1;
  }
  std::printf("platform %s, type %s\n", backend.platform.name,
              backend.platform.type);

  // Four equal bytes: the case a blit fill covers natively.
  auto uniform = Fill32(&backend, 0x7F7F7F7Fu, 1024, 0, status);
  Check("memset32 fills a uniform pattern",
        TF_GetCode(status) == TF_OK && uniform.size() == 1024 &&
            All(uniform, 0, 1024, 0x7F7F7F7Fu),
        TF_GetCode(status) == TF_OK ? "" : TF_Message(status));

  // Four different bytes: the case that has to go through the shader, and the
  // one that a byte fill would silently get wrong.
  TF_SetStatus(status, TF_OK, "");
  auto mixed = Fill32(&backend, 0x12345678u, 4097, 0, status);
  Check("memset32 fills a mixed pattern",
        TF_GetCode(status) == TF_OK && mixed.size() == 4097 &&
            All(mixed, 0, 4097, 0x12345678u),
        TF_GetCode(status) == TF_OK ? "" : TF_Message(status));

  // The same, starting part way into an allocation, because the BFC allocator
  // hands out interior pointers and a fill that ignored the offset would
  // corrupt whatever tensor sits in front of this one.
  TF_SetStatus(status, TF_OK, "");
  auto offset = Fill32(&backend, 0x0000FFFFu, 300, 7, status);
  Check("memset32 respects an interior offset",
        TF_GetCode(status) == TF_OK && offset.size() == 307 &&
            All(offset, 0, 7, 0xABABABABu) &&
            All(offset, 7, 307, 0x0000FFFFu),
        TF_GetCode(status) == TF_OK ? "" : TF_Message(status));

  // INT_MAX is the one non-uniform pattern TensorFlow itself passes.
  TF_SetStatus(status, TF_OK, "");
  auto int_max = Fill32(&backend, 0x7FFFFFFFu, 64, 0, status);
  Check("memset32 fills INT_MAX",
        TF_GetCode(status) == TF_OK && int_max.size() == 64 &&
            All(int_max, 0, 64, 0x7FFFFFFFu),
        TF_GetCode(status) == TF_OK ? "" : TF_Message(status));

  // Reported rather than asserted: the point of the shader is that a large
  // fill does not go through a host loop, and a wall clock is the only place
  // that shows. Times the fill alone, with the allocation and the readback
  // outside the measurement, since both dwarf it.
  if (argc > 2 && std::strcmp(argv[2], "--time") == 0) {
    const size_t words = 64u << 20;  // 256 MB
    const uint64_t bytes = words * sizeof(uint32_t);
    SP_DeviceMemoryBase memory = {};
    memory.struct_size = SP_DEVICE_MEMORY_BASE_STRUCT_SIZE;
    backend.executor.allocate(&backend.device, bytes, 0, &memory);
    if (memory.opaque != nullptr) {
      double best = 1e9;
      for (int i = 0; i < 5; ++i) {
        TF_SetStatus(status, TF_OK, "");
        const auto start = std::chrono::steady_clock::now();
        backend.executor.memset32(&backend.device, backend.stream, &memory,
                                  0x12345678u, bytes, status);
        backend.executor.block_host_until_done(&backend.device, backend.stream,
                                               status);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - start).count();
        if (i > 0 && ms < best) best = ms;
      }
      // The loop this replaced, for the comparison the change rests on.
      double host_best = 1e9;
      for (int i = 0; i < 5; ++i) {
        const auto start = std::chrono::steady_clock::now();
        uint32_t* out = static_cast<uint32_t*>(memory.opaque);
        for (size_t w = 0; w < words; ++w) out[w] = 0x12345678u;
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - start).count();
        if (i > 0 && ms < host_best) host_best = ms;
      }
      std::printf("\n256 MB mixed-pattern fill: %.2f ms (%.0f GB/s) on device,"
                  " %.2f ms (%.0f GB/s) in a host loop\n",
                  best, (bytes / 1e9) / (best / 1e3), host_best,
                  (bytes / 1e9) / (host_best / 1e3));
      backend.executor.deallocate(&backend.device, &memory);
    }
  }

  TF_DeleteStatus(status);
  std::printf("\n%s\n", failures == 0
                            ? "all checks passed"
                            : (std::to_string(failures) + " FAILED").c_str());
  return failures == 0 ? 0 : 1;
}
