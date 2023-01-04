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
#include "tensorflow/lite/delegates/hexagon/hexagon_nn/hexagon_nn_init.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

#include "hexagon/remote.h"  // NOLINT
#include "hexagon/rpcmem.h"  // NOLINT
#include "tensorflow/lite/delegates/hexagon/hexagon_nn/soc_model.h"

extern "C" {

// Version 1.20
static const int kHexagonNNVersion = 137729;
#pragma weak remote_handle_control  // Declare it as a weak symbol
void hexagon_nn_global_init() {
  rpcmem_init();
  // Non-domains QoS invocation
  struct remote_rpc_control_latency data;
  data.enable = RPC_PM_QOS;
  if (remote_handle_control) {  // Check if API is available before invoking
    remote_handle_control(DSPRPC_CONTROL_LATENCY, (void*)&data, sizeof(data));
  }
}

void hexagon_nn_global_teardown() { rpcmem_deinit(); }

bool hexagon_nn_is_device_supported() {
  return tflite::delegates::getsoc_model().mode != UNSPECIFIED_MODE;
}

int hexagon_nn_hexagon_interface_version() { return kHexagonNNVersion; }

}
