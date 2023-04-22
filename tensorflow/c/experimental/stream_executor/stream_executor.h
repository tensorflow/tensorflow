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
#ifndef TENSORFLOW_C_EXPERIMENTAL_STREAM_EXECUTOR_STREAM_EXECUTOR_H_
#define TENSORFLOW_C_EXPERIMENTAL_STREAM_EXECUTOR_STREAM_EXECUTOR_H_
#include <stddef.h>
#include <stdint.h>

#include "tensorflow/c/c_api_macros.h"
#include "tensorflow/c/tf_status.h"

// --------------------------------------------------------------------------
// C API for StreamExecutor. The API is under active development and eventually
// should allow registering a pluggable device with TensorFlow.
//
// Conventions:
//   * Struct prefix indicates whether struct fields should be filled by the
//     plugin or core implementation:
//     * SE_ : set/filled by core unless explicitly marked otherwise.
//     * SP_ : set/filled by plugin unless explicitly marked otherwise.
//   * We use `struct_size` for version checking. It is exempt from the `SE/SP`
//     rule above and should be set both by core and the plugin.
//     * For example, `create_device` function receives `SP_Device*` as input
//       with `struct_size` populated by core. The plugin is responsible for
//       setting `struct_size` as well, along with all other fields.
//     * Refer to "TensorFlow Versioning Strategy" section at
//       https://github.com/tensorflow/community/pull/257/files.
//     * Note that the API is still under active development and doesn't have
//       versioning guarantees yet.
//   * `void* ext` is a free-form field that can be populated by
//     a plugin in `SP_*` structs or potential future extension points in `SE_`
//     structs.
//
// Example usage:
//
//   /* Sample TensorFlow code below, exact implementation might differ. */
//   // Version checking uses `struct_size`. It is exempt from the `SE/SP` rule
//   // above and should be set both by core and the plugin."
//   SP_Device device { SP_DEVICE_STRUCT_SIZE };
//   SE_CreateDeviceParams params { SE_CREATE_DEVICE_PARAMS_STRUCT_SIZE } ;
//   params.device = &device;
//
//   /* Plugin code below */
//   constexpr char DEVICE_NAME[] = "MY_DEVICE";
//   constexpr char DEVICE_TYPE[] = "GPU";
//
//   void create_device(const SP_Platform* platform,
//                      SE_CreateDeviceParams* params, TF_Status* status) {
//     // Custom actions based on TensorFlow's view of SP_Device.
//     OnTFDeviceView(params->device->struct_size);
//     params->device = { SP_DEVICE_STRUCT_SIZE };
//     params->device->device_handle = get_my_device_handle(device->ordinal);
//     params->device->ordinal = params->ordinal;
//     ...
//   }
//
//   void destroy_device(const SP_Platform* platform, SP_Device* device) {
//     delete_my_device_handle(device->device_handle);
//   }
//
//   void SE_InitPlugin(
//       SE_PlatformRegistrationParams* params,
//       TF_Status* status) {
//     params->platform = { SP_PLATFORM_STRUCT_SIZE };
//     // Values such as `name` and `type` must outlive SE_InitPlugin call.
//     params->platform->name = DEVICE_NAME;
//     params->platform->type = DEVICE_TYPE;
//     params->platform_fns->get_device_count = get_device_count;
//     params->platform_fns->create_device = create_device;
//     params->platform_fns->destroy_device = destroy_device;
//     ...
//   }

#define SE_MAJOR 0
#define SE_MINOR 0
#define SE_PATCH 1

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SP_Stream_st* SP_Stream;
typedef struct SP_Event_st* SP_Event;
typedef struct SP_Timer_st* SP_Timer;
// Takes `callback_arg` passed to `host_callback` as the first argument.
typedef void (*SE_StatusCallbackFn)(void* const, TF_Status* const);

typedef struct SP_TimerFns {
  size_t struct_size;
  void* ext;  // reserved for future use
  uint64_t (*nanoseconds)(SP_Timer timer);
} SP_TimerFns;

#define SP_TIMER_FNS_STRUCT_SIZE TF_OFFSET_OF_END(SP_TimerFns, nanoseconds)

typedef struct SP_AllocatorStats {
  size_t struct_size;
  int64_t num_allocs;
  int64_t bytes_in_use;
  int64_t peak_bytes_in_use;
  int64_t largest_alloc_size;

  int8_t has_bytes_limit;
  int64_t bytes_limit;

  int64_t bytes_reserved;
  int64_t peak_bytes_reserved;

  int8_t has_bytes_reservable_limit;
  int64_t bytes_reservable_limit;

  int64_t largest_free_block_bytes;
} SP_AllocatorStats;

#define SP_ALLOCATORSTATS_STRUCT_SIZE \
  TF_OFFSET_OF_END(SP_AllocatorStats, largest_free_block_bytes)

// Potential states for an SP_Event. If `poll_for_status` returns anything aside
// from kPending or kComplete, an error has occurred; kUnknown is a bad state.
typedef enum SE_EventStatus {
  SE_EVENT_UNKNOWN,
  SE_EVENT_ERROR,
  SE_EVENT_PENDING,
  SE_EVENT_COMPLETE,
} SE_EventStatus;

// Memory allocation information.
// This matches DeviceMemoryBase defined here:
// https://cs.opensource.google/tensorflow/tensorflow/+/refs/tags/v2.3.0:tensorflow/stream_executor/device_memory.h;l=57
typedef struct SP_DeviceMemoryBase {
  size_t struct_size;
  void* ext;  // Reserved for future use
  // Platform-dependent value representing allocated memory.
  // Note that the pointer does not have to be to the virtual address itself.
  void* opaque;
  uint64_t size;     // Size in bytes of this allocation.
  uint64_t payload;  // Value for plugin's use
} SP_DeviceMemoryBase;

#define SP_DEVICE_MEMORY_BASE_STRUCT_SIZE \
  TF_OFFSET_OF_END(SP_DeviceMemoryBase, payload)

typedef struct SP_Device {
  size_t struct_size;
  void* ext;        // free-form data set by plugin
  int32_t ordinal;  // device index

  // Device vendor can store handle to their device representation
  // here.
  void* device_handle;

  // [Optional]
  // Device hardware name. Used for printing.
  // Must be null-terminated.
  const char* hardware_name;

  // [Optional]
  // Device vendor name. Used for printing.
  // Must be null-terminated.
  const char* device_vendor;

  // [Optional]
  // Returns the PCI bus identifier for this device, of the form
  // [domain]:[bus]:[device].[function]
  // where domain number is usually 0000.
  // Example: 0000:00:02.1
  // For more information see:
  // https://en.wikipedia.org/wiki/PCI_configuration_space
  // https://www.oreilly.com/library/view/linux-device-drivers/0596005903/ch12.html
  // Used for printing. Must be null-terminated.
  const char* pci_bus_id;
} SP_Device;

#define SP_DEVICE_STRUCT_SIZE TF_OFFSET_OF_END(SP_Device, pci_bus_id)

typedef struct SE_CreateDeviceParams {
  size_t struct_size;
  void* ext;        // reserved for future use
  int32_t ordinal;  // device index

  SP_Device* device;  // Input/output, struct_size set by TF for plugin to read.
                      // Subsequently plugin fills the entire struct.
} SE_CreateDeviceParams;

#define SE_CREATE_DEVICE_PARAMS_STRUCT_SIZE \
  TF_OFFSET_OF_END(SE_CreateDeviceParams, device)

typedef struct SP_DeviceFns {
  size_t struct_size;
  void* ext;  // reserved for future use

  // [Optional]
  // Returns the NUMA node associated with this device, for use in
  // determining socket locality. If the NUMA node could not be determined, -1
  // is returned.
  // Negative values are treated as "unset".
  int32_t (*get_numa_node)(const SP_Device* device);

  // [Optional]
  // Device's memory bandwidth in bytes/sec.  (This is for reads/writes to/from
  // the device's own memory, not for transfers between the host and device.)
  // Negative values are treated as "unset".
  int64_t (*get_memory_bandwidth)(const SP_Device* device);

  // [Optional]
  // Estimate of average number of floating point operations per second for
  // this device * 10e-9.
  // Negative values are treated as "unset".
  double (*get_gflops)(const SP_Device* device);
} SP_DeviceFns;

#define SP_DEVICE_FNS_STRUCT_SIZE TF_OFFSET_OF_END(SP_DeviceFns, get_gflops)

typedef struct SE_CreateDeviceFnsParams {
  size_t struct_size;
  void* ext;  // reserved for future use

  SP_DeviceFns* device_fns;  // output, to be filled by plugin
} SE_CreateDeviceFnsParams;

#define SE_CREATE_DEVICE_FNS_PARAMS_STRUCT_SIZE \
  TF_OFFSET_OF_END(SE_CreateDeviceFnsParams, device_fns)

typedef struct SP_StreamExecutor {
  size_t struct_size;
  void* ext;  // reserved for future use

  /*** ALLOCATION CALLBACKS ***/
  // Synchronously allocates `size` bytes on the underlying platform and returns
  // `SP_DeviceMemoryBase` representing that allocation. In the case of failure,
  // nullptr is returned.
  // `memory_space` is reserved for a potential future usage and should be set
  // to 0.
  void (*allocate)(const SP_Device* device, uint64_t size, int64_t memory_space,
                   SP_DeviceMemoryBase* mem);

  // Deallocate the device memory previously allocated via this interface.
  // Deallocation of a nullptr-representative value is permitted.
  void (*deallocate)(const SP_Device* device, SP_DeviceMemoryBase* memory);

  // Allocates a region of host memory and registers it with the platform API.
  // Memory allocated in this manner is required for use in asynchronous memcpy
  // operations, such as `memcpy_dtoh`.
  void* (*host_memory_allocate)(const SP_Device* device, uint64_t size);

  // Deallocates a region of host memory allocated by `host_memory_allocate`.
  void (*host_memory_deallocate)(const SP_Device* device, void* mem);

  // Allocates unified memory space of the given size, if supported. Unified
  // memory support should be added by setting `supports_unified_memory` field
  // in `SP_Platform`.
  void* (*unified_memory_allocate)(const SP_Device* device, uint64_t bytes);

  // Deallocates unified memory space previously allocated with
  // `unified_memory_allocate`. Unified
  // memory support should be added by setting `supports_unified_memory` field
  // in `SP_Platform`.
  void (*unified_memory_deallocate)(const SP_Device* device, void* location);

  // Fills SP_AllocatorStats with allocator statistics, if it is available.
  // If it is not available, return false.
  TF_Bool (*get_allocator_stats)(const SP_Device* device,
                                 SP_AllocatorStats* stats);
  // Fills the underlying device memory usage information, if it is
  // available. If it is not available (false is returned), free/total need not
  // be initialized.
  TF_Bool (*device_memory_usage)(const SP_Device* device, int64_t* free,
                                 int64_t* total);

  /*** STREAM CALLBACKS ***/
  // Creates SP_Stream. This call should also allocate stream
  // resources on the underlying platform and initializes its
  // internals.
  void (*create_stream)(const SP_Device* device, SP_Stream* stream,
                        TF_Status* status);

  // Destroys SP_Stream and deallocates any underlying resources.
  void (*destroy_stream)(const SP_Device* device, SP_Stream stream);

  // Causes `dependent` to not begin execution until `other` has finished its
  // last-enqueued work.
  void (*create_stream_dependency)(const SP_Device* device, SP_Stream dependent,
                                   SP_Stream other, TF_Status* status);

  // Without blocking the device, retrieve the current stream status.
  void (*get_stream_status)(const SP_Device* device, SP_Stream stream,
                            TF_Status* status);

  /*** EVENT CALLBACKS ***/
  // Create SP_Event. Performs platform-specific allocation and initialization
  // of an event.
  void (*create_event)(const SP_Device* device, SP_Event* event,
                       TF_Status* status);

  // Destroy SE_Event and perform any platform-specific deallocation and
  // cleanup of an event.
  void (*destroy_event)(const SP_Device* device, SP_Event event);

  // Requests the current status of the event from the underlying platform.
  SE_EventStatus (*get_event_status)(const SP_Device* device, SP_Event event);
  // Inserts the specified event at the end of the specified stream.
  void (*record_event)(const SP_Device* device, SP_Stream stream,
                       SP_Event event, TF_Status* status);

  // Wait for the specified event at the end of the specified stream.
  void (*wait_for_event)(const SP_Device* const device, SP_Stream stream,
                         SP_Event event, TF_Status* const status);

  /*** TIMER CALLBACKS ***/
  // Creates SP_Timer. Allocates timer resources on the underlying platform
  // and initializes its internals, setting `timer` output variable. Sets
  // values in `timer_fns` struct.
  void (*create_timer)(const SP_Device* device, SP_Timer* timer,
                       TF_Status* status);

  // Destroy timer and deallocates timer resources on the underlying platform.
  void (*destroy_timer)(const SP_Device* device, SP_Timer timer);

  // Records a start event for an interval timer.
  void (*start_timer)(const SP_Device* device, SP_Stream stream, SP_Timer timer,
                      TF_Status* status);

  // Records a stop event for an interval timer.
  void (*stop_timer)(const SP_Device* device, SP_Stream stream, SP_Timer timer,
                     TF_Status* status);

  /*** MEMCPY CALLBACKS ***/
  // Enqueues a memcpy operation onto stream, with a host destination location
  // `host_dst` and a device memory source, with target size `size`.
  void (*memcpy_dtoh)(const SP_Device* device, SP_Stream stream, void* host_dst,
                      const SP_DeviceMemoryBase* device_src, uint64_t size,
                      TF_Status* status);

  // Enqueues a memcpy operation onto stream, with a device destination
  // location and a host memory source, with target size `size`.
  void (*memcpy_htod)(const SP_Device* device, SP_Stream stream,
                      SP_DeviceMemoryBase* device_dst, const void* host_src,
                      uint64_t size, TF_Status* status);

  // Enqueues a memcpy operation onto stream, with a device destination
  // location and a device memory source, with target size `size`.
  void (*memcpy_dtod)(const SP_Device* device, SP_Stream stream,
                      SP_DeviceMemoryBase* device_dst,
                      const SP_DeviceMemoryBase* device_src, uint64_t size,
                      TF_Status* status);

  // Blocks the caller while a data segment of the given size is
  // copied from the device source to the host destination.
  void (*sync_memcpy_dtoh)(const SP_Device* device, void* host_dst,
                           const SP_DeviceMemoryBase* device_src, uint64_t size,
                           TF_Status* status);

  // Blocks the caller while a data segment of the given size is
  // copied from the host source to the device destination.
  void (*sync_memcpy_htod)(const SP_Device* device,
                           SP_DeviceMemoryBase* device_dst,
                           const void* host_src, uint64_t size,
                           TF_Status* status);

  // Blocks the caller while a data segment of the given size is copied from the
  // device source to the device destination.
  void (*sync_memcpy_dtod)(const SP_Device* device,
                           SP_DeviceMemoryBase* device_dst,
                           const SP_DeviceMemoryBase* device_src, uint64_t size,
                           TF_Status* status);

  // Causes the host code to synchronously wait for the event to complete.
  void (*block_host_for_event)(const SP_Device* device, SP_Event event,
                               TF_Status* status);

  // [Optional]
  // Causes the host code to synchronously wait for operations entrained onto
  // stream to complete. Effectively a join on the asynchronous device
  // operations enqueued on the stream before this program point.
  // If not set, then corresponding functionality will be implemented
  // by registering an event on the `stream` and waiting for it using
  // `block_host_for_event`.
  void (*block_host_until_done)(const SP_Device* device, SP_Stream stream,
                                TF_Status* status);

  // Synchronizes all activity occurring in the StreamExecutor's context (most
  // likely a whole device).
  void (*synchronize_all_activity)(const SP_Device* device, TF_Status* status);

  // Enqueues on a stream a user-specified function to be run on the host.
  // `callback_arg` should be passed as the first argument to `callback_fn`.
  TF_Bool (*host_callback)(const SP_Device* device, SP_Stream stream,
                           SE_StatusCallbackFn callback_fn, void* callback_arg);
} SP_StreamExecutor;

#define SP_STREAMEXECUTOR_STRUCT_SIZE \
  TF_OFFSET_OF_END(SP_StreamExecutor, host_callback)

typedef struct SE_CreateStreamExecutorParams {
  size_t struct_size;
  void* ext;  // reserved for future use

  SP_StreamExecutor* stream_executor;  // output, to be filled by plugin
} SE_CreateStreamExecutorParams;

#define SE_CREATE_STREAM_EXECUTOR_PARAMS_STRUCT_SIZE \
  TF_OFFSET_OF_END(SE_CreateStreamExecutorParams, stream_executor)

typedef struct SP_Platform {
  size_t struct_size;

  void* ext;  // free-form data set by plugin

  // Platform name (also referred to as subtype), for example MY_DEVICE.
  // The name must start with a capital letter and consist of
  // capital letters and underscores.
  // Must be null-terminated.
  const char* name;

  // Device type name, for example GPU. Must be null-terminated.
  // The name must start with a capital letter and consist of
  // capital letters and underscores.
  const char* type;

  // Whether this platform supports unified memory.
  // Unified memory is a single memory address space accessible from any device.
  TF_Bool supports_unified_memory;

  // Whether to wrap allocator for this device with an allocator that uses BFC
  // (best-fit with coalescing) strategy.
  TF_Bool use_bfc_allocator;
} SP_Platform;

#define SP_PLATFORM_STRUCT_SIZE TF_OFFSET_OF_END(SP_Platform, use_bfc_allocator)

typedef struct SP_PlatformFns {
  size_t struct_size;

  void* ext;  // reserved for future use

  // Callbacks for getting device count
  void (*get_device_count)(const SP_Platform* platform, int* device_count,
                           TF_Status* status);
  // Callbacks for creating/destroying SP_Device.
  void (*create_device)(const SP_Platform* platform,
                        SE_CreateDeviceParams* params, TF_Status* status);

  // Clean up fields inside SP_Device that were allocated
  // by the plugin. `device` itself should not be deleted here.
  void (*destroy_device)(const SP_Platform* platform, SP_Device* device);

  // Callbacks for creating/destroying SP_DeviceFns.
  void (*create_device_fns)(const SP_Platform* platform,
                            SE_CreateDeviceFnsParams* params,
                            TF_Status* status);

  // Clean up fields inside SP_DeviceFns that were allocated
  // by the plugin. `device_fns` itself should not be deleted here.
  void (*destroy_device_fns)(const SP_Platform* platform,
                             SP_DeviceFns* device_fns);

  // Callbacks for creating/destroying SP_StreamExecutor.
  void (*create_stream_executor)(const SP_Platform* platform,
                                 SE_CreateStreamExecutorParams* params,
                                 TF_Status* status);
  // Clean up fields inside SP_StreamExecutor that were allocated
  // by the plugin. `stream_executor` itself should not be deleted here.
  void (*destroy_stream_executor)(const SP_Platform* platform,
                                  SP_StreamExecutor* stream_executor);

  // Callbacks for creating/destroying SP_TimerFns.
  void (*create_timer_fns)(const SP_Platform* platform, SP_TimerFns* timer,
                           TF_Status* status);

  void (*destroy_timer_fns)(const SP_Platform* platform,
                            SP_TimerFns* timer_fns);
} SP_PlatformFns;

#define SP_PLATFORM_FNS_STRUCT_SIZE \
  TF_OFFSET_OF_END(SP_PlatformFns, destroy_timer_fns)

typedef struct SE_PlatformRegistrationParams {
  size_t struct_size;
  void* ext;  // reserved for future use

  // StreamExecutor C API version.
  int32_t major_version;
  int32_t minor_version;
  int32_t patch_version;

  SP_Platform* platform;         // output, set by plugin
  SP_PlatformFns* platform_fns;  // output, set by plugin
  // Clean up fields inside SP_Platform that were allocated
  // by the plugin. `platform` itself should not be deleted here.
  void (*destroy_platform)(SP_Platform* platform);  // out, set by plugin
  void (*destroy_platform_fns)(
      SP_PlatformFns* platform_fns);  // out, set by plugin
} SE_PlatformRegistrationParams;

#define SE_PLATFORM_REGISTRATION_PARAMS_STRUCT_SIZE \
  TF_OFFSET_OF_END(SE_PlatformRegistrationParams, destroy_platform_fns)

void SE_InitPlugin(SE_PlatformRegistrationParams* params, TF_Status* status);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_C_EXPERIMENTAL_STREAM_EXECUTOR_STREAM_EXECUTOR_H_
