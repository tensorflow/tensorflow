/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_TESTS_PJRT_CLIENT_REGISTRY_H_
#define TENSORFLOW_COMPILER_XLA_TESTS_PJRT_CLIENT_REGISTRY_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"

namespace xla {

class PjRtClientTestFactoryRegistry {
 public:
  typedef std::function<Shape(const Shape&)> DeviceShapeRepresentationFn;
  typedef std::function<DeviceShapeRepresentationFn(PjRtClient*)>
      DeviceShapeRepresentationFnFactory;
  typedef std::function<StatusOr<std::unique_ptr<PjRtClient>>()>
      PjRtClientFactory;

  static DeviceShapeRepresentationFn DefaultShapeRepresentationRegisteredFn(
      StatusOr<PjRtClient*> client) {
    return [](const Shape& host_shape) { return host_shape; };
  }

  void Register(PjRtClientFactory factory,
                DeviceShapeRepresentationFnFactory
                    registered_device_shape_representation_fn) {
    if (HasRegisteredFactory()) {
      LOG(FATAL) << "A PjRtClient has already been registered.";
      return;
    }

    absl::MutexLock lock(&mu_);
    factory_ = std::move(factory);
    registered_device_shape_representation_fn_ =
        std::move(registered_device_shape_representation_fn);
  }

  // Return the device shape representation of 'host_shape'.
  DeviceShapeRepresentationFn GetDeviceShapeRepresentationFn(
      PjRtClient* pjrt_client) {
    absl::MutexLock lock(&mu_);
    return registered_device_shape_representation_fn_(pjrt_client);
  }

  bool HasRegisteredFactory() {
    absl::MutexLock lock(&mu_);
    return factory_ != nullptr;
  }

  std::function<StatusOr<std::unique_ptr<PjRtClient>>()> Get() const {
    absl::MutexLock lock(&mu_);
    return factory_;
  }

 private:
  mutable absl::Mutex mu_;
  std::function<StatusOr<std::unique_ptr<PjRtClient>>()> factory_
      ABSL_GUARDED_BY(mu_);
  DeviceShapeRepresentationFnFactory registered_device_shape_representation_fn_;
};

PjRtClientTestFactoryRegistry& GetGlobalPjRtClientTestFactory();

void RegisterPjRtClientTestFactory(
    PjRtClientTestFactoryRegistry::PjRtClientFactory factory,
    PjRtClientTestFactoryRegistry::DeviceShapeRepresentationFnFactory
        registered_device_shape_representation_fn =
            PjRtClientTestFactoryRegistry::
                DefaultShapeRepresentationRegisteredFn);

bool ShouldUsePjRt();

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_PJRT_CLIENT_REGISTRY_H_
