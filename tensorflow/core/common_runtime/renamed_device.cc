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

#include "tensorflow/core/common_runtime/renamed_device.h"

#include <utility>

#include "absl/memory/memory.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {

/* static */
std::unique_ptr<Device> RenamedDevice::NewRenamedDevice(
    const string& new_base, Device* underlying, bool owns_underlying,
    bool isolate_session_state,
    thread::ThreadPoolInterface* underlying_threadpool) {
  DeviceNameUtils::ParsedName parsed_name;
  CHECK(DeviceNameUtils::ParseFullName(new_base, &parsed_name));
  DeviceNameUtils::ParsedName underlying_parsed_name =
      underlying->parsed_name();
  CHECK(underlying_parsed_name.has_type);
  CHECK(underlying_parsed_name.has_id);
  parsed_name.type = underlying_parsed_name.type;
  parsed_name.id = underlying_parsed_name.id;
  string name = DeviceNameUtils::FullName(parsed_name.job, parsed_name.replica,
                                          parsed_name.task, parsed_name.type,
                                          parsed_name.id);
  DeviceAttributes attributes(underlying->attributes());
  attributes.set_name(name);
  // Call absl::WrapUnique to access private constructor.
  return absl::WrapUnique(
      new RenamedDevice(underlying, std::move(attributes), owns_underlying,
                        isolate_session_state, underlying_threadpool));
}

RenamedDevice::RenamedDevice(Device* underlying, DeviceAttributes attributes,
                             bool owns_underlying_device,
                             bool isolate_session_state,
                             thread::ThreadPoolInterface* underlying_threadpool)
    : Device(underlying->env(), std::move(attributes)),
      underlying_device_(underlying),
      owns_underlying_device_(owns_underlying_device),
      isolate_session_state_(isolate_session_state) {
  if (underlying_threadpool != nullptr) {
    underlying_threadpool_.reset(new thread::ThreadPool(underlying_threadpool));
    eigen_worker_threads_.workers = underlying_threadpool_.get();
    eigen_worker_threads_.num_threads = underlying_threadpool->NumThreads();
    set_tensorflow_cpu_worker_threads(&eigen_worker_threads_);
    set_tensorflow_device_thread_pool(underlying_threadpool_.get());

    Eigen::ThreadPoolDevice eigen_threadpool_device(
        underlying_threadpool, underlying_threadpool->NumThreads());
    set_eigen_cpu_device(&eigen_threadpool_device);
  }
}

RenamedDevice::~RenamedDevice() {
  if (owns_underlying_device_) {
    delete underlying_device_;
  }
}

}  // namespace tensorflow
