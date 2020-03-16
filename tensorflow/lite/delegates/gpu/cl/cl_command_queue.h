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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_COMMAND_QUEUE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_COMMAND_QUEUE_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/time/time.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_event.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_kernel.h"
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

struct ProfilingInfo {
  struct DispatchInfo {
    std::string label;
    absl::Duration duration;
  };

  std::vector<DispatchInfo> dispatches;

  absl::Duration GetTotalTime() const;

  // Returns report (string of lines delimited by \n)
  // This method uses GPU counters and measure GPU time only.
  // Report has next structure:
  // Per kernel timing(K kernels):
  //   conv2d 3.2ms
  //   ...
  // --------------------
  // Accumulated time per operation type:
  //   conv2d - 14.5ms
  //   ....
  // --------------------
  // Ideal total time: 23.4ms // Total time for all kernels
  std::string GetDetailedReport() const;
};

// A wrapper around opencl command queue
class CLCommandQueue {
 public:
  CLCommandQueue() {}
  CLCommandQueue(cl_command_queue queue, bool has_ownership);

  // Move only
  CLCommandQueue(CLCommandQueue&& queue);
  CLCommandQueue& operator=(CLCommandQueue&& queue);
  CLCommandQueue(const CLCommandQueue&) = delete;
  CLCommandQueue& operator=(const CLCommandQueue&) = delete;

  virtual ~CLCommandQueue();

  cl_command_queue queue() const { return queue_; }

  virtual Status DispatchImplicit(const CLKernel& kernel, int3 grid,
                                  int3 work_group_size);

  Status EnqueueEvent(CLEvent* event);

  Status DispatchImplicit(const CLKernel& kernel, int3 grid,
                          int3 work_group_size, CLEvent* event);

  Status EnqueueWriteImage(cl_mem memory, int3 region, const void* data);
  Status EnqueueReadImage(cl_mem memory, int3 region, void* data);

  Status EnqueueWriteBuffer(cl_mem memory, size_t size_in_bytes,
                            const void* data);
  Status EnqueueReadBuffer(cl_mem memory, size_t size_in_bytes, void* data);

  Status WaitForCompletion();

 protected:
  void Release();

  cl_command_queue queue_ = nullptr;
  bool has_ownership_ = false;
};

class ProfilingCommandQueue : public CLCommandQueue {
 public:
  ProfilingCommandQueue() {}
  explicit ProfilingCommandQueue(cl_command_queue queue);

  // Move only
  ProfilingCommandQueue(ProfilingCommandQueue&& queue);
  ProfilingCommandQueue& operator=(ProfilingCommandQueue&& queue);
  ProfilingCommandQueue(const ProfilingCommandQueue&) = delete;
  ProfilingCommandQueue& operator=(const ProfilingCommandQueue&) = delete;

  Status DispatchImplicit(const CLKernel& kernel, int3 grid,
                          int3 work_group_size) override;

  // will write index for fastest work_group among work_group_sizes
  Status GetBestWorkGroupIndex(const CLKernel& kernel,
                               const DeviceInfo& device_info, const int3& grid,
                               const std::vector<int3>& work_group_sizes,
                               int* index);

  // call ResetMeasurements() to start new seriese of measurements
  void ResetMeasurements();

  double GetQueueExecutionTimeMs() const;

  // Difference from GetQueueExecutionTimeMs is that this number doesn't include
  // time between kernels(kernels launches or preparing) on GPU. Usually, this
  // time should be 5-10% better than GetQueueExecutionTimeMs, because 5-10%
  // spend on something else(maybe kernels launches or preparing)
  double GetSumOfEventsTimeMs() const;

  // This label will be used for all subsequent dispatches.
  void SetEventsLabel(const std::string& name);

  ProfilingInfo GetProfilingInfo() const;

 private:
  std::vector<CLEvent> events_;
  std::string current_label_;
};

Status CreateCLCommandQueue(const CLDevice& device, const CLContext& context,
                            CLCommandQueue* result);

Status CreateProfilingCommandQueue(const CLDevice& device,
                                   const CLContext& context,
                                   ProfilingCommandQueue* result);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_COMMAND_QUEUE_H_
