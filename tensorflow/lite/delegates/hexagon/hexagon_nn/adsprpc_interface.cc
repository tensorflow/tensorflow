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
#include <dlfcn.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/stat.h>

#include <cstdio>
#include <cstring>

#include "tensorflow/lite/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_nn/soc_model.h"

namespace {

void* LoadLibadsprpc() {
  void* lib = dlopen("libadsprpc.so", RTLD_LAZY | RTLD_LOCAL);
  if (lib) {
    fprintf(stdout, "loaded libadsprpc.so\n");
    return lib;
  }

  return nullptr;
}

void* LoadLibcdsprpc() {
  void* lib = dlopen("libcdsprpc.so", RTLD_LAZY | RTLD_LOCAL);
  if (lib) {
    fprintf(stdout, "loaded libcdsprpc.so\n");
    return lib;
  }

  return nullptr;
}

void* LoadDsprpc() {
  SocSkelTable soc_model = tflite::delegates::getsoc_model();
  // Use aDSP for 835 and 820, otherwise cDSP.
  if (soc_model.mode == NON_DOMAINS ||
      (soc_model.dsp_type != nullptr &&
       strcmp(soc_model.dsp_type, "adsp") == 0)) {
    return LoadLibadsprpc();
  }
  return LoadLibcdsprpc();
}

void* LoadFunction(const char* name) {
  static void* libadsprpc = LoadDsprpc();
  if (libadsprpc == nullptr) {
    fprintf(stderr, "libadsprpc handle is NULL\n");
    return nullptr;
  }
  auto* func_pt = dlsym(libadsprpc, name);
  if (func_pt == nullptr) {
    fprintf(stderr, "Func %s not available on this device (NULL).\n", name);
  }
  return func_pt;
}

using remote_handle_open_fn = decltype(remote_handle_open);
using remote_handle64_open_fn = decltype(remote_handle64_open);
using remote_handle_invoke_fn = decltype(remote_handle_invoke);
using remote_handle64_invoke_fn = decltype(remote_handle64_invoke);
using remote_handle_close_fn = decltype(remote_handle_close);
using remote_handle64_close_fn = decltype(remote_handle64_close);
using remote_mmap_fn = decltype(remote_mmap);
using remote_mmap64_fn = decltype(remote_mmap64);
using remote_munmap_fn = decltype(remote_munmap);
using remote_munmap64_fn = decltype(remote_munmap64);
using remote_register_buf_fn = decltype(remote_register_buf);
using remote_set_mode_fn = decltype(remote_set_mode);
using remote_handle_control_fn = decltype(remote_handle_control);

struct AdsprpcInterface {
  remote_handle_open_fn* handle_open_fn =
      reinterpret_cast<remote_handle_open_fn*>(
          LoadFunction("remote_handle_open"));
  remote_handle64_open_fn* handle64_open_fn =
      reinterpret_cast<remote_handle64_open_fn*>(
          LoadFunction("remote_handle64_open"));
  remote_handle_invoke_fn* handle_invoke_fn =
      reinterpret_cast<remote_handle_invoke_fn*>(
          LoadFunction("remote_handle_invoke"));
  remote_handle64_invoke_fn* handle64_invoke_fn =
      reinterpret_cast<remote_handle64_invoke_fn*>(
          LoadFunction("remote_handle64_invoke"));
  remote_handle_close_fn* handle_close_fn =
      reinterpret_cast<remote_handle_close_fn*>(
          LoadFunction("remote_handle_close"));
  remote_handle64_close_fn* handle64_close_fn =
      reinterpret_cast<remote_handle64_close_fn*>(
          LoadFunction("remote_handle64_close"));
  remote_handle_control_fn* handle_control_fn =
      reinterpret_cast<remote_handle_control_fn*>(
          LoadFunction("remote_handle_control"));
  remote_mmap_fn* mmap_fn =
      reinterpret_cast<remote_mmap_fn*>(LoadFunction("remote_mmap"));
  remote_munmap_fn* munmap_fn =
      reinterpret_cast<remote_munmap_fn*>(LoadFunction("remote_munmap"));
  remote_mmap64_fn* mmap64_fn =
      reinterpret_cast<remote_mmap64_fn*>(LoadFunction("remote_mmap64"));
  remote_munmap64_fn* munmap64_fn =
      reinterpret_cast<remote_munmap64_fn*>(LoadFunction("remote_munmap64"));
  remote_register_buf_fn* register_buf_fn =
      reinterpret_cast<remote_register_buf_fn*>(
          LoadFunction("remote_register_buf"));
  remote_set_mode_fn* set_mode_fn =
      reinterpret_cast<remote_set_mode_fn*>(LoadFunction("remote_set_mode"));

  // Returns singleton instance.
  static AdsprpcInterface* Singleton() {
    static AdsprpcInterface* instance = new AdsprpcInterface();
    return instance;
  }
};

}  // namespace

extern "C" {
int remote_handle_open(const char* name, remote_handle* h) {
  return AdsprpcInterface::Singleton()->handle_open_fn
             ? AdsprpcInterface::Singleton()->handle_open_fn(name, h)
             : -1;
}

int remote_handle64_open(const char* name, remote_handle64* h) {
  return AdsprpcInterface::Singleton()->handle64_open_fn
             ? AdsprpcInterface::Singleton()->handle64_open_fn(name, h)
             : -1;
}

int remote_handle_invoke(remote_handle h, uint32_t scalars, remote_arg* args) {
  return AdsprpcInterface::Singleton()->handle_invoke_fn
             ? AdsprpcInterface::Singleton()->handle_invoke_fn(h, scalars, args)
             : -1;
}

int remote_handle64_invoke(remote_handle64 h, uint32_t scalars,
                           remote_arg* args) {
  return AdsprpcInterface::Singleton()->handle64_invoke_fn
             ? AdsprpcInterface::Singleton()->handle64_invoke_fn(h, scalars,
                                                                 args)
             : -1;
}

int remote_handle_close(remote_handle h) {
  return AdsprpcInterface::Singleton()->handle_close_fn
             ? AdsprpcInterface::Singleton()->handle_close_fn(h)
             : -1;
}

int remote_handle64_close(remote_handle64 h) {
  return AdsprpcInterface::Singleton()->handle64_close_fn
             ? AdsprpcInterface::Singleton()->handle64_close_fn(h)
             : -1;
}

int remote_handle_control(uint32_t req, void* data, uint32_t datalen) {
  return AdsprpcInterface::Singleton()->handle_control_fn
             ? AdsprpcInterface::Singleton()->handle_control_fn(req, data,
                                                                datalen)
             : -1;
}

int remote_mmap(int fd, uint32_t flags, uint32_t addr, int size,
                uint32_t* result) {
  return AdsprpcInterface::Singleton()->mmap_fn
             ? AdsprpcInterface::Singleton()->mmap_fn(fd, flags, addr, size,
                                                      result)
             : -1;
}

int remote_mmap64(int fd, uint32_t flags, uintptr_t vaddrin, int64_t size,
                  uintptr_t* vaddrout) {
  return AdsprpcInterface::Singleton()->mmap64_fn
             ? AdsprpcInterface::Singleton()->mmap64_fn(fd, flags, vaddrin,
                                                        size, vaddrout)
             : -1;
}

int remote_munmap(uint32_t addr, int size) {
  return AdsprpcInterface::Singleton()->munmap_fn
             ? AdsprpcInterface::Singleton()->munmap_fn(addr, size)
             : -1;
}

int remote_munmap64(uintptr_t vaddrout, int64_t size) {
  return AdsprpcInterface::Singleton()->munmap64_fn
             ? AdsprpcInterface::Singleton()->munmap64_fn(vaddrout, size)
             : -1;
}

void remote_register_buf(void* buf, int size, int fd) {
  if (AdsprpcInterface::Singleton()->register_buf_fn) {
    AdsprpcInterface::Singleton()->register_buf_fn(buf, size, fd);
  }
}

int remote_set_mode(uint32_t mode) {
  return AdsprpcInterface::Singleton()->set_mode_fn
             ? AdsprpcInterface::Singleton()->set_mode_fn(mode)
             : -1;
}

}  // extern "C"
