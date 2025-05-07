/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_plugin_init.h"

#include <cstddef>
#include <cstdint>

#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow/c/experimental/pluggable_profiler/pluggable_profiler.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/tf_buffer.h"
#include "tensorflow/c/tf_status.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/plugin_c_api.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

extern "C" {
// from test_pluggable_device.cc
void SE_InitPlugin(SE_PlatformRegistrationParams* params, TF_Status* status);
void TF_InitKernel();

// from test_next_pluggable_device_plugin.cc
const TFNPD_Api* TFNPD_InitPlugin(TFNPD_PluginParams* params,
                                  TF_Status* tf_status);
const PJRT_Api* GetPjrtApi();
}

TEST(PluggableDevicePluginInitTest, StaticInitTest) {
  static bool init_plugin_fn_called = false;

  auto init_plugin_fn = +[](SE_PlatformRegistrationParams* const platform,
                            TF_Status* const status) {
    init_plugin_fn_called = true;
    SE_InitPlugin(platform, status);
  };

  PluggableDeviceInit_Api api;
  // All initialization functions are optional
  TF_ASSERT_OK(RegisterPluggableDevicePlugin(&api));

  init_plugin_fn_called = false;
  api.init_plugin_fn = reinterpret_cast<void*>(init_plugin_fn);
  TF_ASSERT_OK(RegisterPluggableDevicePlugin(&api));
  ASSERT_TRUE(init_plugin_fn_called);
}

TEST(PluggableDevicePluginInitTest, StaticNPInitTest) {
  static bool init_np_plugin_fn_called = false;
  static bool init_pjrt_fn_called = false;

  auto init_np_plugin_fn = +[](TFNPD_PluginParams* plugin_params,
                               TF_Status* status) -> const TFNPD_Api* {
    init_np_plugin_fn_called = true;
    return TFNPD_InitPlugin(plugin_params, status);
  };

  auto init_pjrt_fn = +[]() -> const PJRT_Api* {
    init_pjrt_fn_called = true;
    return GetPjrtApi();
  };

  PluggableDeviceInit_Api api;
  init_np_plugin_fn_called = false;
  init_pjrt_fn_called = false;
  api.init_np_plugin_fn = reinterpret_cast<void*>(init_np_plugin_fn);
  api.get_pjrt_api_fn = reinterpret_cast<void*>(init_pjrt_fn);
  TF_ASSERT_OK(RegisterPluggableDevicePlugin(&api));
  ASSERT_TRUE(init_np_plugin_fn_called);
  ASSERT_TRUE(init_pjrt_fn_called);
}

TEST(PluggableDevicePluginInitTest, StaticKernelInitTest) {
  static bool init_kernel_fn_called = false;

  auto init_kernel_fn = +[]() {
    init_kernel_fn_called = true;
    TF_InitKernel();
  };

  PluggableDeviceInit_Api api;
  init_kernel_fn_called = false;
  api.init_kernel_fn = reinterpret_cast<void*>(init_kernel_fn);
  TF_ASSERT_OK(RegisterPluggableDevicePlugin(&api));
  ASSERT_TRUE(init_kernel_fn_called);
}

TEST(PluggableDevicePluginInitTest, StaticGraphInitTest) {
  static bool init_graph_fn_called = false;

  auto init_graph_fn =
      +[](TP_OptimizerRegistrationParams* const params, TF_Status* const) {
        init_graph_fn_called = true;
        params->device_type = "GPU";
        params->optimizer->optimize_func =
            +[](void*, const TF_Buffer*, const TF_GrapplerItem*, TF_Buffer*,
                TF_Status*) {};
      };

  PluggableDeviceInit_Api api;
  init_graph_fn_called = false;
  api.init_graph_fn = reinterpret_cast<void*>(init_graph_fn);
  TF_ASSERT_OK(RegisterPluggableDevicePlugin(&api));
  ASSERT_TRUE(init_graph_fn_called);
}

TEST(PluggableDevicePluginInitTest, StaticProfilerInitTest) {
  static bool init_profiler_fn_called = false;

  auto init_profiler_fn =
      +[](TF_ProfilerRegistrationParams* const params, TF_Status* const) {
        init_profiler_fn_called = true;
        params->destroy_profiler = +[](TP_Profiler*) {};
        params->destroy_profiler_fns = +[](TP_ProfilerFns*) {};
        params->profiler->device_type = "GPU";
        params->profiler_fns->start =
            +[](const TP_Profiler* profiler, TF_Status* status) {};
        params->profiler_fns->stop =
            +[](const TP_Profiler* profiler, TF_Status* status) {};
        params->profiler_fns->collect_data_xspace =
            +[](const TP_Profiler* profiler, uint8_t* buffer,
                size_t* size_in_bytes, TF_Status* status) {};
      };

  PluggableDeviceInit_Api api;
  init_profiler_fn_called = false;
  api.init_profiler_fn = reinterpret_cast<void*>(init_profiler_fn);
  TF_ASSERT_OK(RegisterPluggableDevicePlugin(&api));
  ASSERT_TRUE(init_profiler_fn_called);
}

}  // namespace
}  // namespace tensorflow
