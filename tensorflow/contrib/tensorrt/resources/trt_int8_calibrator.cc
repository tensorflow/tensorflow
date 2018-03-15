/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/tensorrt/resources/trt_int8_calibrator.h"

#include <atomic>
#include <chrono>
#include <unordered_map>

#include "tensorflow/core/platform/logging.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "cuda_runtime_api.h"

namespace tensorflow {
namespace tensorrt {

// set the batch size before constructing the thread to execute engine
int TRTInt8Calibrator::getBatchSize() const { return batch_size_; }

TRTInt8Calibrator::TRTInt8Calibrator(
    const std::unordered_map<string, std::pair<void*, size_t>>& dev_buffers,
    int batch_size, string engine_name)
    : batch_size_(batch_size),
      done_(false),
      dev_buffers_(dev_buffers),
      calib_running_(false),
      engine_name_(engine_name) {}

bool TRTInt8Calibrator::setBatch(
    const std::unordered_map<string, void*>& data) {
  // TODO(aaroey): make sure that in future PR:
  // 1. the mutex_lock is outside of the loop
  // 2. wait() is used instead of wait_for()
  // 3. done_ is to be protected by the mutex
  // 4. the first batch is not missed
  if (done_) return false;
  while (calib_running_.load(
      std::memory_order_acquire)) {  // wait while calibration is running
    tensorflow::mutex_lock l(cond_mtx_);
    cond_.wait_for(l, std::chrono::milliseconds(50));
    if (done_) return false;
  }
  VLOG(1) << "Set Batch Waiting finished";
  for (const auto it : data) {
    auto devptr = dev_buffers_.find(it.first);
    if (devptr == dev_buffers_.end()) {
      LOG(FATAL) << "FATAL " << engine_name_ << " input name '" << it.first
                 << "' does not match with the buffer names";
    }
    const auto& d = devptr->second;

    // TODO(aaroey): we should not use sync copy on default stream. Make sure
    // stream->ThenMemcpy() is used in future PRs.
    auto status =
        cudaMemcpy(d.first, it.second, d.second, cudaMemcpyDeviceToDevice);
    if (status != cudaSuccess) {
      LOG(FATAL) << "cudaMemcpy " << engine_name_ << " for '" << it.first
                 << "' failed with " << status;
    }
  }
  calib_running_.store(true, std::memory_order_release);  // release builder
  cond_.notify_all();
  return true;
}

bool TRTInt8Calibrator::getBatch(void** bindings, const char** names,
                                 int num_bindings) {
  calib_running_.store(false, std::memory_order_release);  // wait for new batch
  cond_.notify_all();
  while (!calib_running_.load(
      std::memory_order_acquire)) {  // wait until new batch arrives
    tensorflow::mutex_lock l(cond_mtx_);
    cond_.wait_for(l, std::chrono::milliseconds(50));
    if (done_) return false;
  }
  if (done_) {
    return false;
  }

  for (int i = 0; i < num_bindings; i++) {
    auto it = dev_buffers_.find(names[i]);
    if (it == dev_buffers_.end()) {
      LOG(FATAL) << "Calibration engine asked for unknown tensor name '"
                 << names[i] << "' at position " << i;
    }

    bindings[i] = it->second.first;
  }
  return true;
}

const void* TRTInt8Calibrator::readCalibrationCache(std::size_t& length) {
  return nullptr;
}

void TRTInt8Calibrator::writeCalibrationCache(const void* ptr,
                                              std::size_t length) {}
TRTInt8Calibrator::~TRTInt8Calibrator() {
  VLOG(1) << "Destroying calibrator for " << engine_name_;
}

}  // namespace tensorrt
}  // namespace tensorflow
#endif
#endif
