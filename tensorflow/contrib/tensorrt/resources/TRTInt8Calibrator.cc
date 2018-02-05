//
// Created by skama on 1/24/18.
//

#include "tensorflow/contrib/tensorrt/resources/TRTInt8Calibrator.h"

#include <cuda_runtime_api.h>
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace trt {
// set the batch size before constructing the thread to execute engine
int TRTInt8Calibrator::getBatchSize() const { return batch_size_; }

bool TRTInt8Calibrator::setBatch(
    const std::unordered_map<std::string, void*>& data) {
  while (calib_running_.load(
      std::memory_order_acquire)) {  // wait while calibration is running
    tensorflow::mutex_lock l(cond_mtx_);
    cond_.wait_for(l, std::chrono::milliseconds(50));
  }
  for (const auto it : data) {
    auto devptr = dev_buffers_.find(it.first);
    if (devptr == dev_buffers_.end()) {
      LOG(FATAL) << "FATAL input name '" << it.first
                 << "' does not match with the buffer names";
    }
    const auto& d = devptr->second;
    auto status =
        cudaMemcpy(d.first, it.second, d.second, cudaMemcpyHostToDevice);
    if (status != 0) {
      LOG(FATAL) << "cudaMemcpy for '" << it.first << "' failed with "
                 << status;
    }
  }
  calib_running_.store(true, std::memory_order_release);  // release builder
  cond_.notify_all();
  return true;
}

bool TRTInt8Calibrator::getBatch(void** bindings, const char** names,
                                 int nbBindings) {
  calib_running_.store(false, std::memory_order_release);  // wait for new batch
  cond_.notify_all();
  while (!calib_running_.load(
      std::memory_order_acquire)) {  // wait until new batch arrives
    tensorflow::mutex_lock l(cond_mtx_);
    cond_.wait_for(l, std::chrono::milliseconds(50));
  }
  if (done_) {
    return false;
  }
  for (int i = 0; i < nbBindings; i++) {
    auto it = dev_buffers_.find(names[i]);
    if (it == dev_buffers_.end()) {
      LOG(FATAL) << "Calibration engine asked for unknown tensor name '"
                 << names[i] << "' at position " << i;
    }
    bindings[i] = it->second.first;
  }
  return true;
}

}  // namespace trt
}  // namespace tensorflow