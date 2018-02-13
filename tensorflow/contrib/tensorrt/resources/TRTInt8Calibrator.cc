//
// Created by skama on 1/24/18.
//

#include "tensorflow/contrib/tensorrt/resources/TRTInt8Calibrator.h"

#include <cuda_runtime_api.h>
#include <atomic>
#include <chrono>
#include <unordered_map>

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace trt {
// set the batch size before constructing the thread to execute engine
int TRTInt8Calibrator::getBatchSize() const { return batch_size_; }

TRTInt8Calibrator::TRTInt8Calibrator(const std::unordered_map<
    std::string, std::pair<void*, size_t>>& dev_buffers,
                  int batch_size)
    : batch_size_(batch_size),
      done_(false),
      dev_buffers_(dev_buffers),
      calib_running_(false){
  cudaPointerAttributes pa;
  int devid=-1;
  cudaGetDevice(&devid);
  VLOG(0)<<"Constructing calibrator with batch size "<<batch_size<<" on device"<<devid;
  for(auto b : dev_buffers_) {
    if(cudaPointerGetAttributes(&pa,b.second.first)==cudaSuccess){
      VLOG(1) << "CALIBRATOR Device buffer name " << b.first << " size" << b.second.second
              << " @ " << b.second.first << " onDevice "<<((pa.memoryType==cudaMemoryTypeHost)?"HOST":"DEVICE");
    }else {
      VLOG(1) << "CALIBRATOR Device buffer name " << b.first << " size" << b.second.second << " @ " << b.second.first;
    }
  }
}

bool TRTInt8Calibrator::setBatch(
    const std::unordered_map<std::string, void*>& data) {
  VLOG(1)<<"SAMI SAMI Waiting to set new batch";
  if(done_)return false;
  while (calib_running_.load(
      std::memory_order_acquire)) {  // wait while calibration is running
    tensorflow::mutex_lock l(cond_mtx_);
    cond_.wait_for(l, std::chrono::milliseconds(50));
    if(done_)return false;
  }
  VLOG(1)<<"Set Batch Waiting finished";
  for (const auto it : data) {

    auto devptr = dev_buffers_.find(it.first);
    if (devptr == dev_buffers_.end()) {
      LOG(FATAL) << "FATAL input name '" << it.first
                 << "' does not match with the buffer names";
    }
    cudaPointerAttributes pa;
    const auto& d = devptr->second;
    VLOG(1)<<"cuda memcopy buff name= "<<it.first<<" dst= "
           <<d.first<<" size= "<<d.second<<" inp= "<<it.second;
    if(cudaPointerGetAttributes(&pa,it.second)==cudaSuccess) {
      VLOG(1) << "CALIBRATOR Device buffer name " << it.first << " size" << d.second
          << " @ " << d.first << " onDevice " << ((pa.memoryType == cudaMemoryTypeHost) ? "HOST" : "DEVICE");
    }

    auto status =
        cudaMemcpy(d.first, it.second, d.second, cudaMemcpyDeviceToDevice);
    if (status != cudaSuccess) {
      LOG(FATAL) << "cudaMemcpy for '" << it.first << "' failed with "
                 << status;
    }
    float f[2];
    f[0]=3.;
    f[1]=0.14159;
    status=cudaMemcpy(f,d.first,sizeof(float)*2,cudaMemcpyDeviceToHost);
    int devid=-1;
    cudaGetDevice(&devid);
    VLOG(0)<<"SAMI ORDER SETTING Data in perm storage [0]="<<f[0]<<" [1]="<<f[1]<<" current device="<<devid;
  }
  calib_running_.store(true, std::memory_order_release);  // release builder
  cond_.notify_all();
  return true;
}

bool TRTInt8Calibrator::getBatch(void** bindings, const char** names,
                                 int nbBindings) {
  calib_running_.store(false, std::memory_order_release);  // wait for new batch
  VLOG(1)<<"SAMI SAMI Calibrator is waiting for new batch";
  cond_.notify_all();
  while (!calib_running_.load(
      std::memory_order_acquire)) {  // wait until new batch arrives
    tensorflow::mutex_lock l(cond_mtx_);
    cond_.wait_for(l, std::chrono::milliseconds(50));
    if(done_)return false;
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
    VLOG(1)<<"Setting buffer "<< i <<" named=" << names[i] <<" @ "<<it->second.first;
    bindings[i] = it->second.first;
    float f[2];
    f[0]=3.;
    f[1]=0.14159;
    auto status=cudaMemcpy(f,bindings[i],sizeof(float)*2,cudaMemcpyDeviceToHost);
    int devid=-1;
    cudaGetDevice(&devid);
    VLOG(0)<<"SAMI ORDER GETTING, Data in perm storage [0]="<<f[0]<<" [1]="
           <<f[1]<<" on device="<<devid;

  }
  return true;
}
const void *TRTInt8Calibrator::readCalibrationCache(std::size_t &length) {
  return nullptr;
}
void TRTInt8Calibrator::writeCalibrationCache(const void *ptr, std::size_t length) {

}
TRTInt8Calibrator::~TRTInt8Calibrator() {
  VLOG(1)<<"Destroying calibrator";
}

}  // namespace trt
}  // namespace tensorflow