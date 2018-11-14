/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/device_base.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

DeviceBase::~DeviceBase() { gtl::STLDeleteElements(&eigen_cpu_devices_); }

const DeviceAttributes& DeviceBase::attributes() const {
  LOG(FATAL) << "Device does not implement attributes()";
}

const string& DeviceBase::name() const {
  LOG(FATAL) << "Device does not implement name()";
}

void DeviceBase::set_eigen_cpu_device(Eigen::ThreadPoolDevice* d) {
  // Eigen::ThreadPoolDevice is a very cheap struct (two pointers and
  // an int).  Therefore, we can afford a pre-allocated array of
  // Eigen::ThreadPoolDevice.  Here, we ensure that
  // Eigen::ThreadPoolDevices in eigen_cpu_devices_ has increasingly
  // larger numThreads.
  for (int i = 1; i <= d->numThreads(); ++i) {
    eigen_cpu_devices_.push_back(new Eigen::ThreadPoolDevice(
        d->getPool(), i /* numThreads() */, d->allocator()));
  }
}

const Eigen::ThreadPoolDevice* DeviceBase::eigen_cpu_device() {
  // Based on GetPerThreadMaxParallelism(), we return a different
  // pre-allocated Eigen::ThreadPoolDevice. All these ThreadPoolDevice
  // use the same underlying threadpool. But they use different
  // nominal numThreads() hoping that the user of the returned
  // Eigen::ThreadPoolDevice may not aggressively occupy all the
  // threads in the underlying threadpool.
  const int parallelism = std::max<int>(
      1,
      std::min<int>(GetPerThreadMaxParallelism(), eigen_cpu_devices_.size()));
  return eigen_cpu_devices_[parallelism - 1];
}

}  // namespace tensorflow
