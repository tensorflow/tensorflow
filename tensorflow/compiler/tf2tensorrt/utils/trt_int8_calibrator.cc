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

#include "tensorflow/compiler/tf2tensorrt/utils/trt_int8_calibrator.h"

#include <atomic>
#include <unordered_map>

#include "tensorflow/core/platform/logging.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"

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
      // Make sure setBatch() waits until getBatch() is called (the first time).
      calib_running_(true),
      batch_is_set_(false),
      engine_name_(engine_name) {}

TRTInt8Calibrator::TRTInt8Calibrator(const string& calib_data)
    : batch_size_(0),
      done_(true),
      calib_running_(false),
      batch_is_set_(false),
      calibration_table_(calib_data) {}

bool TRTInt8Calibrator::setBatch(const std::unordered_map<string, void*>& data,
                                 const cudaStream_t stream) {
  mutex_lock lock(cond_mtx_);

  // Wait while the queue is full or calibration is running.
  while ((calib_running_ || batch_is_set_) && !done_) cond_.wait(lock);
  if (done_) return false;
  CHECK(!calib_running_ && !batch_is_set_);
  VLOG(1) << "Set Batch Waiting finished";

  // Sets the batch.
  for (const auto& it : data) {
    auto devptr = dev_buffers_.find(it.first);
    if (devptr == dev_buffers_.end()) {
      LOG(FATAL) << "FATAL " << engine_name_ << " input name '" << it.first
                 << "' does not match with the buffer names";
    }
    const auto& d = devptr->second;

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
  // we have to wait for the stream before returning!
  cudaStreamSynchronize(stream);
  batch_is_set_ = true;
  cond_.notify_all();
  return true;
}

bool TRTInt8Calibrator::getBatch(void** bindings, const char** names,
                                 int num_bindings) {
  mutex_lock lock(cond_mtx_);
  // Notify finish of last round of calibration.
  calib_running_ = false;
  cond_.notify_all();

  // Wait until new batch arrives
  while ((!batch_is_set_ && !done_)) cond_.wait(lock);
  if (done_) return false;

  // Gets the batch
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

void TRTInt8Calibrator::waitAndSetDone() {
  mutex_lock lock(cond_mtx_);
  // Wait while the queue is full or calibration is running, so we don't miss
  // the last batch.
  while ((calib_running_ || batch_is_set_) && !done_) cond_.wait(lock);
  if (!done_) {
    done_ = true;
    cond_.notify_all();
    dev_buffers_.clear();
  }
}

const void* TRTInt8Calibrator::readCalibrationCache(std::size_t& length) {
  if (calibration_table_.empty()) return nullptr;
  length = calibration_table_.size();
  return calibration_table_.data();
}

void TRTInt8Calibrator::setDone() {
  mutex_lock lock(cond_mtx_);
  done_ = true;
  cond_.notify_all();
}

void TRTInt8Calibrator::writeCalibrationCache(const void* ptr,
                                              std::size_t length) {
  calibration_table_ = string(static_cast<const char*>(ptr), length);
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
