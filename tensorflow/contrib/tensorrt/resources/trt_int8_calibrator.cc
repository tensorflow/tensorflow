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
#include "cuda/include/cuda_runtime_api.h"

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
      batch_is_set_(false),
      engine_name_(engine_name) {}

TRTInt8Calibrator::TRTInt8Calibrator(const string& calib_data)
    : batch_size_(0),
      done_(false),
      calib_running_(false),
      batch_is_set_(false),
      calibration_table_(calib_data) {}

bool TRTInt8Calibrator::setBatch(const std::unordered_map<string, void*>& data,
                                 const cudaStream_t stream) {
  tensorflow::mutex_lock lock(cond_mtx_);
  while ((calib_running_ || batch_is_set_) &&
         !done_) {  // wait while calibration is running
    cond_.wait(lock);
  }
  if (done_) return false;
  CHECK(!calib_running_ && !batch_is_set_);
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
    // TODO(sami,aaroey): Need to figure out a way to ensure synchronization
    // between stream, perhaps using a tensor?
    auto status = cudaMemcpyAsync(d.first, it.second, d.second,
                                  cudaMemcpyDeviceToDevice, stream);
    if (status != cudaSuccess) {
      LOG(FATAL) << "cudaMemcpy " << engine_name_ << " for '" << it.first
                 << "' failed with " << status;
    }
  }

  // TODO(Sami, aaorey): Find an alternative way!
  cudaStreamSynchronize(
      stream);  // we have to wait for the stream before returning!
  batch_is_set_ = true;
  cond_.notify_all();
  return true;
}

bool TRTInt8Calibrator::getBatch(void** bindings, const char** names,
                                 int num_bindings) {
  tensorflow::mutex_lock lock(cond_mtx_);
  calib_running_ = false;
  cond_.notify_all();
  while ((!batch_is_set_ && !done_)) {  // wait until new batch arrives
    cond_.wait(lock);
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
  batch_is_set_ = false;
  calib_running_ = true;
  return true;
}

const void* TRTInt8Calibrator::readCalibrationCache(std::size_t& length) {
  if (calibration_table_.empty()) return nullptr;
  length = calibration_table_.size();
  return calibration_table_.data();
}

void TRTInt8Calibrator::setDone() {
  tensorflow::mutex_lock lock(cond_mtx_);
  done_ = true;
  cond_.notify_all();
}

void TRTInt8Calibrator::writeCalibrationCache(const void* ptr,
                                              std::size_t length) {
  calibration_table_ = string((const char*)ptr, length);
  VLOG(1) << "Got calibration data for " << engine_name_ << " @" << ptr
          << " length=" << length;
}
TRTInt8Calibrator::~TRTInt8Calibrator() {
  VLOG(1) << "Destroying calibrator for " << engine_name_;
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif
#endif
