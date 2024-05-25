/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_OP_KERNEL_HELPER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_OP_KERNEL_HELPER_H_

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c_plugin_op_kernel.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/direct_plugin_op_kernel.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/flags.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/plugin_op_kernel.h"

namespace tensorflow {

inline PluginOpKernelConstruction* CreatePluginOpKernelConstruction(void* ctx) {
  if (!absl::GetFlag(FLAGS_next_pluggable_device_use_c_api)) {
    return new DirectPluginOpKernelConstruction(ctx);
  } else {
    return new CPluginOpKernelConstruction(ctx);
  }
}

inline void DeletePluginOpKernelConstruction(
    PluginOpKernelConstruction* wrapper) {
  delete wrapper;
}

inline PluginOpKernelContext* CreatePluginOpKernelContext(void* ctx) {
  if (!absl::GetFlag(FLAGS_next_pluggable_device_use_c_api)) {
    return new DirectPluginOpKernelContext(ctx);
  } else {
    return new CPluginOpKernelContext(ctx);
  }
}

inline void DeletePluginOpKernelContext(PluginOpKernelContext* wrapper) {
  delete wrapper;
}

#define PLUGIN_OP_REQUIRES_OK(CTX, ...)          \
  do {                                           \
    absl::Status _s(__VA_ARGS__);                \
    if (!TF_PREDICT_TRUE(_s.ok())) {             \
      (CTX)->CtxFailure(__FILE__, __LINE__, _s); \
      return;                                    \
    }                                            \
  } while (0)

// A helper to register C OpKernel. CREATE_FN, COMPUTE_FN, and DELETE_FN are
// expected to be defined in the same file where this macro is used.
//
// HOST_MEMORY_ARGS a string containing names of args to be placed on host
// memory. Names are expected to be comma separated.
//
// TODO(chuanhao): simplify the registration macro. reference:
// REGISTER_KERNEL_BUILDER
#define REGISTER_WRAPPED_C_OPKERNEL_HOST_MEM_ARGS(                            \
    KERNEL_NAME, CREATE_FN, COMPUTE_FN, DELETE_FN, DEVICE, PRIORITY,          \
    HOST_MEMORY_ARGS)                                                         \
  {                                                                           \
    typedef void* (*wrapped_create_func)(TF_OpKernelConstruction*);           \
    typedef void (*wrapped_compute_func)(void*, TF_OpKernelContext*);         \
                                                                              \
    TF_StatusPtr status_ptr(TF_NewStatus());                                  \
                                                                              \
    wrapped_create_func create_func =                                         \
        [](TF_OpKernelConstruction* ctx) -> void* {                           \
      PluginOpKernelConstruction* ctx_wrapper =                               \
          CreatePluginOpKernelConstruction(ctx);                              \
      void* kernel = CREATE_FN(ctx_wrapper);                                  \
      delete ctx_wrapper;                                                     \
      return kernel;                                                          \
    };                                                                        \
                                                                              \
    wrapped_compute_func compute_func = [](void* kernel,                      \
                                           TF_OpKernelContext* ctx) -> void { \
      PluginOpKernelContext* ctx_wrapper = CreatePluginOpKernelContext(ctx);  \
      COMPUTE_FN(kernel, ctx_wrapper);                                        \
      delete ctx_wrapper;                                                     \
    };                                                                        \
                                                                              \
    auto* builder = TF_NewKernelBuilder(KERNEL_NAME, DEVICE, create_func,     \
                                        compute_func, &DELETE_FN);            \
                                                                              \
    /* NOTE: We explicitly set the priority to 1 to overwrite the */          \
    /* StreamExecutor based OpKernel of the same op.              */          \
    TF_KernelBuilder_Priority(builder, PRIORITY);                             \
                                                                              \
    std::stringstream s_stream(HOST_MEMORY_ARGS);                             \
    while (s_stream.good()) {                                                 \
      std::string host_mem_arg;                                               \
      std::getline(s_stream, host_mem_arg, ',');                              \
      if (host_mem_arg.empty()) break;                                        \
      TF_KernelBuilder_HostMemory(builder, host_mem_arg.c_str());             \
    }                                                                         \
                                                                              \
    TF_RegisterKernelBuilder(KERNEL_NAME, builder, status_ptr.get());         \
    CHECK_EQ(TF_OK, TF_GetCode(status_ptr.get()))                             \
        << "Error while registering " << KERNEL_NAME << " kernel.";           \
  }

#define REGISTER_WRAPPED_C_OPKERNEL(KERNEL_NAME, CREATE_FN, COMPUTE_FN, \
                                    DELETE_FN, DEVICE, PRIORITY)        \
  REGISTER_WRAPPED_C_OPKERNEL_HOST_MEM_ARGS(                            \
      KERNEL_NAME, CREATE_FN, COMPUTE_FN, DELETE_FN, DEVICE, PRIORITY, "")

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_OP_KERNEL_HELPER_H_
