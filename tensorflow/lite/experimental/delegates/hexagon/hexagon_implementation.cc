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

#include "tensorflow/lite/experimental/delegates/hexagon/hexagon_implementation.h"

#include <dlfcn.h>
#include <fcntl.h>

#include <cstdio>

#include "tensorflow/lite/experimental/delegates/hexagon/hexagon_nn_interface.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace {

void* LoadFunction(void* dl_handle, const char* name) {
  TFLITE_DCHECK(dl_handle != nullptr);
  auto* func_pt = dlsym(dl_handle, name);
  if (func_pt == nullptr) {
    TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Function %s is  NULL", name);
  }
  return func_pt;
}

#define LOAD_FUNCTION(dl_handle, method_name, hexagon_obj)           \
  hexagon_obj.method_name = reinterpret_cast<method_name##_fn*>(     \
      LoadFunction(dl_handle, #method_name));                        \
  if ((hexagon_obj.method_name) == nullptr) {                        \
    TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "%s is NULL", (#method_name)); \
    return hexagon_obj;                                              \
  }

HexagonNN CreateNewHexagonInterface() {
  HexagonNN hexagon_nn;
  void* libhexagon_interface =
      dlopen("libhexagon_interface.so", RTLD_LAZY | RTLD_LOCAL);
  if (libhexagon_interface == nullptr) {
    TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Failed to load libhexagon_interface.so");
    return hexagon_nn;
  }
  LOAD_FUNCTION(libhexagon_interface, hexagon_nn_config, hexagon_nn);
  LOAD_FUNCTION(libhexagon_interface, hexagon_nn_init, hexagon_nn);
  LOAD_FUNCTION(libhexagon_interface, hexagon_nn_prepare, hexagon_nn);
  LOAD_FUNCTION(libhexagon_interface, hexagon_nn_set_powersave_level,
                hexagon_nn);
  LOAD_FUNCTION(libhexagon_interface, hexagon_nn_set_debug_level, hexagon_nn);
  LOAD_FUNCTION(libhexagon_interface, hexagon_nn_append_node, hexagon_nn);
  LOAD_FUNCTION(libhexagon_interface, hexagon_nn_append_const_node, hexagon_nn);
  LOAD_FUNCTION(libhexagon_interface, hexagon_nn_execute, hexagon_nn);
  LOAD_FUNCTION(libhexagon_interface, hexagon_nn_execute_new, hexagon_nn);
  LOAD_FUNCTION(libhexagon_interface, hexagon_nn_teardown, hexagon_nn);
  LOAD_FUNCTION(libhexagon_interface, hexagon_nn_snpprint, hexagon_nn);
  LOAD_FUNCTION(libhexagon_interface, hexagon_nn_getlog, hexagon_nn);
  LOAD_FUNCTION(libhexagon_interface, hexagon_nn_get_perfinfo, hexagon_nn);
  LOAD_FUNCTION(libhexagon_interface, hexagon_nn_reset_perfinfo, hexagon_nn);
  LOAD_FUNCTION(libhexagon_interface, hexagon_nn_op_id_to_name, hexagon_nn);
  LOAD_FUNCTION(libhexagon_interface, hexagon_nn_global_teardown, hexagon_nn);
  LOAD_FUNCTION(libhexagon_interface, hexagon_nn_global_init, hexagon_nn);
  LOAD_FUNCTION(libhexagon_interface, hexagon_nn_is_device_supported,
                hexagon_nn);
  hexagon_nn.interface_loaded = true;
  return hexagon_nn;
}

}  // namespace

const HexagonNN* HexagonNNImplementation() {
  static HexagonNN hexagon_nn = CreateNewHexagonInterface();
  if (!hexagon_nn.interface_loaded) {
    return nullptr;
  }
  return &hexagon_nn;
}

}  // namespace tflite
