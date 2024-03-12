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

#include "tensorflow/compiler/jit/xla_platform_info.h"

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/jit/device_executable_persistor.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/pjrt_device_compiler_client.h"
#include "tensorflow/compiler/jit/xla_compile_util.h"
#include "tensorflow/compiler/jit/xla_device_compiler_client.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/platform_manager.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/common/create_pjrt_client_util.h"
#include "tensorflow/core/tfrt/common/global_state.h"
#include "tensorflow/core/tfrt/common/pjrt_util.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tsl/framework/device_type.h"

namespace tensorflow {
namespace {
using XlaDeviceCompiler =
    DeviceCompiler<xla::LocalExecutable, xla::LocalClient>;
using PjRtDeviceCompiler =
    DeviceCompiler<xla::PjRtLoadedExecutable, xla::PjRtClient>;
using XlaDeviceExecutablePersistor =
    DeviceExecutablePersistor<xla::LocalExecutable, xla::LocalClient>;
using PjRtDeviceExecutablePersistor =
    DeviceExecutablePersistor<xla::PjRtLoadedExecutable, xla::PjRtClient>;

XlaDeviceCompiler* CreateXlaDeviceCompiler(
    const XlaDeviceExecutablePersistor::Config& persistor_config,
    DeviceType compilation_device_type, xla::LocalClient* local_client) {
  return new XlaDeviceCompiler(
      std::make_unique<XlaDeviceExecutablePersistor>(
          std::move(persistor_config), compilation_device_type),
      std::make_unique<XlaDeviceCompilerClient>(local_client));
}

PjRtDeviceCompiler* CreatePjRtDeviceCompiler(DeviceType compilation_device_type,
                                             xla::PjRtClient* pjrt_client) {
  std::string persistent_cache_directory =
      GetPersistentCacheDirectory(compilation_device_type);

  PjRtDeviceExecutablePersistor::Config persistor_config(
      persistent_cache_directory,
      GetMarkForCompilationPassFlags()->tf_xla_disable_strict_signature_checks,
      GetMarkForCompilationPassFlags()->tf_xla_persistent_cache_prefix,
      GetMarkForCompilationPassFlags()->tf_xla_persistent_cache_read_only);

  return new PjRtDeviceCompiler(
      std::make_unique<PjRtDeviceExecutablePersistor>(
          std::move(persistor_config), compilation_device_type),
      std::make_unique<PjRtDeviceCompilerClient>(pjrt_client));
}

absl::StatusOr<std::optional<std::set<int>>> GetAllowedGpus(
    FunctionLibraryRuntime* flr) {
  std::optional<std::set<int>> gpu_ids = std::nullopt;

  if (flr->config_proto()) {
    string allowed_gpus =
        flr->config_proto()->gpu_options().visible_device_list();
    TF_ASSIGN_OR_RETURN(gpu_ids, ParseVisibleDeviceList(allowed_gpus));
  }

  return gpu_ids;
}

Status GetCompilationDeviceTypeAndPjRtClient(
    const XlaPlatformInfo& platform_info, FunctionLibraryRuntime* flr,
    DeviceType* compilation_device_type, xla::PjRtClient** pjrt_client) {
  DeviceType device_type = platform_info.device_type();

  if (platform_info.xla_device_metadata()) {
    VLOG(2) << "Building PjRtDeviceCompiler using "
               "platform_info.xla_device_metadata().";

    *compilation_device_type =
        platform_info.xla_device_metadata()->jit_device_type();
    TF_ASSIGN_OR_RETURN(*pjrt_client, GetOrCreatePjRtClient(device_type));
    return absl::OkStatus();
  }

  if (platform_info.pjrt_device_metadata()) {
    VLOG(2) << "Building PjRtDeviceCompiler using "
               "platform_info.pjrt_device_metadata().";

    *compilation_device_type =
        platform_info.pjrt_device_metadata()->jit_device_type();
    TF_ASSIGN_OR_RETURN(*pjrt_client, GetOrCreatePjRtClient(device_type));
    return absl::OkStatus();
  }

  // TFRT-TPU is used if device_type is `DEVICE_TPU` and platform_info does not
  // have `xla_device_metadata`.
  if (device_type == DEVICE_TPU) {
    *compilation_device_type = DeviceType(DEVICE_TPU_XLA_JIT);
    TF_ASSIGN_OR_RETURN(*pjrt_client, GetOrCreatePjRtClient(device_type));
    return absl::OkStatus();
  }

  VLOG(2) << "platform_info.xla_device_metadata not found and "
             "platform_info.device_type() != DEVICE_TPU. Building "
             "PjRtDeviceCompiler for non-XLA device.";

  const XlaOpRegistry::DeviceRegistration* registration;
  if (!XlaOpRegistry::GetCompilationDevice(device_type.type(), &registration)) {
    return errors::InvalidArgument("No JIT device registered for ",
                                   device_type.type());
  }
  *compilation_device_type = DeviceType(registration->compilation_device_name);

  TF_ASSIGN_OR_RETURN(auto allowed_gpus, GetAllowedGpus(flr));
  // TODO(b/255826209): Set platform, intra op parallelism threads if required
  // and when supported by GetOrCreatePjRtClient().
  // The `allowed_gpus` argument is used only if the `device_type` is GPU.
  TF_ASSIGN_OR_RETURN(*pjrt_client,
                      GetOrCreatePjRtClient(device_type, allowed_gpus));

  return absl::OkStatus();
}
}  // namespace

std::string GetPersistentCacheDirectory(
    const DeviceType& compilation_device_type) {
  // If a persistent cache device type is specified, ensure it matches
  // compilation device type.
  if (!GetMarkForCompilationPassFlags()
           ->tf_xla_persistent_cache_device_types.empty() &&
      !absl::c_any_of(absl::StrSplit(GetMarkForCompilationPassFlags()
                                         ->tf_xla_persistent_cache_device_types,
                                     ','),
                      [&](absl::string_view device) {
                        return compilation_device_type == DeviceType(device);
                      })) {
    return "";
  }
  return GetMarkForCompilationPassFlags()->tf_xla_persistent_cache_directory;
}

absl::StatusOr<std::optional<std::set<int>>> ParseVisibleDeviceList(
    absl::string_view visible_device_list) {
  std::set<int> gpu_ids;
  if (visible_device_list.empty()) {
    return {{std::nullopt}};
  }
  const std::vector<string> visible_devices =
      absl::StrSplit(visible_device_list, ',');
  for (const string& platform_device_id_str : visible_devices) {
    int32_t platform_device_id;
    if (!absl::SimpleAtoi(platform_device_id_str, &platform_device_id)) {
      return errors::InvalidArgument(
          "Could not parse entry in 'visible_device_list': '",
          platform_device_id_str,
          "'. visible_device_list = ", visible_device_list);
    }
    gpu_ids.insert(platform_device_id);
  }
  return {{gpu_ids}};
}

absl::StatusOr<DeviceType> GetCompilationDeviceType(
    const DeviceType& platform_device_type) {
  DeviceType compilation_device_type = platform_device_type;
  const XlaOpRegistry::DeviceRegistration* registration = nullptr;
  if (!XlaOpRegistry::GetCompilationDevice(platform_device_type.type(),
                                           &registration)) {
    return errors::InvalidArgument("No JIT device registered for ",
                                   platform_device_type.type());
  }
  compilation_device_type = DeviceType(registration->compilation_device_name);
  return compilation_device_type;
}

Status BuildXlaDeviceCompiler(DeviceBase* device, FunctionLibraryRuntime* flr,
                              const XlaPlatformInfo& platform_info,
                              DeviceType compilation_device_type,
                              XlaDeviceCompiler** xla_device_compiler) {
  if (platform_info.platform_id() == nullptr &&
      platform_info.device_type() == DEVICE_GPU) {
    // We do not need to (and cannot) build a real device compiler for GPU
    // if the platform ID is null. We just return a placeholder for supporting
    // cross platform lowering.
    *xla_device_compiler = new XlaDeviceCompiler(/*persistor=*/nullptr,
                                                 /*compiler_client=*/nullptr);
    return absl::OkStatus();
  }
  std::string persistent_cache_directory =
      GetPersistentCacheDirectory(platform_info.device_type());

  XlaDeviceExecutablePersistor::Config persistor_config(
      persistent_cache_directory,
      GetMarkForCompilationPassFlags()->tf_xla_disable_strict_signature_checks,
      GetMarkForCompilationPassFlags()->tf_xla_persistent_cache_prefix,
      GetMarkForCompilationPassFlags()->tf_xla_persistent_cache_read_only);

  if (platform_info.xla_device_metadata()) {
    *xla_device_compiler = CreateXlaDeviceCompiler(
        persistor_config,
        platform_info.xla_device_metadata()->jit_device_type(),
        platform_info.xla_device_metadata()->client());
    return absl::OkStatus();
  }

  // TFRT-TPU is used if device type is `DEVICE_TPU` and platform_info does not
  // have `xla_device_metadata`. This is used for TFRT-TPU when
  // BuildXlaDeviceCompiler() is called in GetCompilerIr(). Currently only
  // lowering to HLO is needed there and xla::LocalClient doesn't support
  // building the executable for TFRT-TPU and hence, is set to nullptr here.
  if (platform_info.device_type() == DEVICE_TPU) {
    *xla_device_compiler = CreateXlaDeviceCompiler(
        persistor_config, DeviceType(DEVICE_TPU_XLA_JIT), nullptr);
    return absl::OkStatus();
  }

  if (platform_info.platform_id() == nullptr) {
    return errors::InvalidArgument("platform_id is null.");
  }
  auto platform =
      se::PlatformManager::PlatformWithId(platform_info.platform_id());
  if (!platform.ok()) {
    return platform.status();
  }

  absl::StatusOr<xla::Compiler*> compiler_for_platform =
      xla::Compiler::GetForPlatform(platform.value());
  if (!compiler_for_platform.ok()) {
    // In some rare cases (usually in unit tests with very small clusters) we
    // may end up transforming an XLA cluster with at least one GPU operation
    // (which would normally force the cluster to be compiled using XLA:GPU)
    // into an XLA cluster with no GPU operations (i.e. containing only CPU
    // operations).  Such a cluster can fail compilation (in way that
    // MarkForCompilation could not have detected) if the CPU JIT is not
    // linked in.
    //
    // So bail out of _XlaCompile in this case, and let the executor handle
    // the situation for us.
    const Status& status = compiler_for_platform.status();
    if (status.code() == error::NOT_FOUND) {
      return errors::Unimplemented("Could not find compiler for platform ",
                                   platform.value()->Name(), ": ",
                                   status.ToString());
    }
  }

  xla::LocalClientOptions client_options;
  client_options.set_platform(platform.value());
  if (device != nullptr) {
    client_options.set_intra_op_parallelism_threads(
        device->tensorflow_cpu_worker_threads()->num_threads);
  }

  if (flr != nullptr) {
    TF_ASSIGN_OR_RETURN(auto allowed_gpus, GetAllowedGpus(flr));
    client_options.set_allowed_devices(allowed_gpus);
  }

  TF_ASSIGN_OR_RETURN(
      auto client, xla::ClientLibrary::GetOrCreateLocalClient(client_options));

  *xla_device_compiler = CreateXlaDeviceCompiler(
      persistor_config, compilation_device_type, client);
  return absl::OkStatus();
}

Status GetOrCreatePjRtDeviceCompilerAndProfiler(
    const XlaPlatformInfo& platform_info, ResourceMgr* rm,
    FunctionLibraryRuntime* flr, PjRtDeviceCompiler** pjrt_device_compiler,
    DeviceCompilationProfiler** profiler) {
  const auto& device_type = platform_info.device_type();
  const std::string& compiler_name =
      GetPjRtDeviceCompilerResourceName(device_type);
  const std::string& profiler_name =
      GetPjRtDeviceCompilationProfilerResourceName(device_type);
  bool deleted_old_device_compiler = false;

  // Lookup the DeviceCompiler, create one if not found.
  Status s = rm->Lookup<PjRtDeviceCompiler>(
      rm->default_container(), compiler_name, pjrt_device_compiler);
  if (s.ok() && device_type == DEVICE_TPU) {
    auto* existing_pjrt_client = (*pjrt_device_compiler)->client();
    TF_ASSIGN_OR_RETURN(auto* latest_pjrt_client, GetPjRtClient(device_type));

    if (existing_pjrt_client != latest_pjrt_client) {
      // PjRtClient has changed. Delete the PjRtDeviceCompiler (and the cache
      // within) and create a new one.
      TF_RETURN_IF_ERROR(rm->Delete<PjRtDeviceCompiler>(rm->default_container(),
                                                        compiler_name));
      TF_RETURN_IF_ERROR(rm->Delete<DeviceCompilationProfiler>(
          rm->default_container(), profiler_name));

      deleted_old_device_compiler = true;
    }
  }

  // TODO(b/308698131): Try consolidating all PJRT-related state into one class
  // instead of directly storing it in the ResourceMgr.
  if (!s.ok() || deleted_old_device_compiler) {
    DeviceType compilation_device_type("");
    xla::PjRtClient* pjrt_client = nullptr;
    TF_RETURN_IF_ERROR(GetCompilationDeviceTypeAndPjRtClient(
        platform_info, flr, &compilation_device_type, &pjrt_client));

    TF_RETURN_IF_ERROR(rm->LookupOrCreate<PjRtDeviceCompiler>(
        rm->default_container(), compiler_name, pjrt_device_compiler,
        [&](PjRtDeviceCompiler** pjrt_device_compiler) {
          *pjrt_device_compiler =
              CreatePjRtDeviceCompiler(compilation_device_type, pjrt_client);
          return absl::OkStatus();
        }));
  }

  TF_RETURN_IF_ERROR(rm->LookupOrCreate<DeviceCompilationProfiler>(
      rm->default_container(), profiler_name, profiler,
      [](DeviceCompilationProfiler** profiler) {
        *profiler = new DeviceCompilationProfiler();
        return absl::OkStatus();
      }));

  return absl::OkStatus();
}

Status GetOrCreatePjRtDeviceCompilerAndProfiler(
    const OpKernelContext& ctx, const XlaPlatformInfo& platform_info,
    FunctionLibraryRuntime* flr,
    DeviceCompiler<xla::PjRtLoadedExecutable, xla::PjRtClient>**
        pjrt_device_compiler,
    DeviceCompilationProfiler** profiler) {
  TF_ASSIGN_OR_RETURN(ResourceMgr * rm, GetResourceMgrForDeviceCompiler(
                                            ctx, platform_info.device_type()));
  return GetOrCreatePjRtDeviceCompilerAndProfiler(
      platform_info, rm, flr, pjrt_device_compiler, profiler);
}

XlaPlatformInfo XlaPlatformInfoFromDevice(DeviceBase* device_base) {
  se::Platform::Id platform_id = nullptr;
  const XlaDevice::Metadata* xla_device_metadata = nullptr;
  const PjRtBaseDevice::Metadata* pjrt_device_metadata = nullptr;
  std::shared_ptr<se::DeviceMemoryAllocator> custom_allocator;

  const std::string& device_type = device_base->device_type();
  if (device_type == DEVICE_CPU) {
    platform_id = se::host::kHostPlatformId;
  } else if (device_type == DEVICE_GPU) {
    auto device = static_cast<Device*>(device_base);
    platform_id = device->tensorflow_accelerator_device_info()
                      ->stream->parent()
                      ->platform()
                      ->id();
  } else if (XlaDevice::GetMetadataFromDevice(device_base, &xla_device_metadata)
                 .ok()) {
    // If we are on an XlaDevice, use the underlying XLA platform's allocator
    // directly. We could use the StreamExecutor's allocator which may
    // theoretically be more correct, but XLA returns a nice OOM message in a
    // Status and StreamExecutor does not.
    //
    // Importantly we can't use ctx->device()->GetAllocator() as the allocator
    // (which xla_allocator above uses) as on an XlaDevice, this is a dummy
    // allocator that returns XlaTensor objects. The XlaCompiler needs a real
    // allocator to allocate real buffers.
    platform_id = xla_device_metadata->platform()->id();
    custom_allocator =
        xla_device_metadata->client()->backend().shared_memory_allocator();
  } else if (auto metadata = PjRtBaseDevice::GetMetadataFromDevice(device_base);
             metadata.ok()) {
    pjrt_device_metadata = *metadata;
  }

  return XlaPlatformInfo(DeviceType(device_type), platform_id,
                         xla_device_metadata, pjrt_device_metadata,
                         custom_allocator);
}

std::shared_ptr<se::DeviceMemoryAllocator> GetAllocator(
    DeviceBase* device, se::Stream* stream,
    const XlaPlatformInfo& platform_info) {
  if (platform_info.custom_allocator()) {
    return platform_info.custom_allocator();
  }
  auto* alloc = device->GetAllocator({});
  if (!stream) {
    // Stream is not set for the host platform.
    se::Platform* platform =
        se::PlatformManager::PlatformWithId(platform_info.platform_id())
            .value();
    return std::make_shared<se::TfAllocatorAdapter>(alloc, platform);
  }
  return std::make_shared<se::TfAllocatorAdapter>(alloc, stream);
}

}  // namespace tensorflow
