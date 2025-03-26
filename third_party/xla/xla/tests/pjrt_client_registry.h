/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_TESTS_PJRT_CLIENT_REGISTRY_H_
#define XLA_TESTS_PJRT_CLIENT_REGISTRY_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "xla/pjrt/pjrt_client.h"

namespace xla {

class PjRtClientTestFactoryRegistry {
 public:
  using DeviceShapeRepresentationFn = std::function<Shape(const Shape&)>;
  using DeviceShapeRepresentationFnFactory =
      std::function<DeviceShapeRepresentationFn(PjRtClient*)>;
  using DeviceShapeSizeFn = std::function<int64_t(const Shape&)>;
  using DeviceShapeSizeFnFactory =
      std::function<DeviceShapeSizeFn(PjRtClient*)>;
  using PjRtClientFactory =
      std::function<absl::StatusOr<std::unique_ptr<PjRtClient>>()>;

  static DeviceShapeRepresentationFn DefaultShapeRepresentationRegisteredFn(
      PjRtClient* client) {
    return [](const Shape& host_shape) { return host_shape; };
  }
  static DeviceShapeSizeFn DefaultDeviceShapeSizeRegisteredFn(
      PjRtClient* client) {
    return [](const Shape& shape) -> int64_t {
      if (shape.IsOpaque()) {
        return sizeof(void*);
      }
      return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
    };
  }

  void Register(PjRtClientFactory factory,
                DeviceShapeRepresentationFnFactory
                    registered_device_shape_representation_fn,
                DeviceShapeSizeFnFactory registered_device_shape_size_fn)
      ABSL_LOCKS_EXCLUDED(mu_) {
    if (HasRegisteredFactory()) {
      LOG(FATAL) << "A PjRtClient has already been registered.";
      return;
    }

    absl::MutexLock lock(&mu_);
    factory_ = std::move(factory);
    registered_device_shape_representation_fn_ =
        std::move(registered_device_shape_representation_fn);
    registered_device_shape_size_fn_ =
        std::move(registered_device_shape_size_fn);
  }

  // Return the device shape representation of 'host_shape'.
  DeviceShapeRepresentationFn GetDeviceShapeRepresentationFn(
      PjRtClient* pjrt_client) ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(&mu_);
    return registered_device_shape_representation_fn_(pjrt_client);
  }

  // Return the device shape size of 'host_shape'.
  // This function is used e.g. to create a VerifiedHloModule. It returns an
  // integer representing the size of the shape in bytes as opposed to a Shape.
  DeviceShapeSizeFn GetDeviceShapeSizeFn(PjRtClient* pjrt_client)
      ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(&mu_);
    return registered_device_shape_size_fn_(pjrt_client);
  }

  bool HasRegisteredFactory() ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(&mu_);
    return factory_ != nullptr;
  }

  std::function<absl::StatusOr<std::unique_ptr<PjRtClient>>()> Get() const {
    absl::MutexLock lock(&mu_);
    return factory_;
  }

 private:
  mutable absl::Mutex mu_;
  std::function<absl::StatusOr<std::unique_ptr<PjRtClient>>()> factory_
      ABSL_GUARDED_BY(mu_);
  DeviceShapeRepresentationFnFactory registered_device_shape_representation_fn_
      ABSL_GUARDED_BY(mu_);
  DeviceShapeSizeFnFactory registered_device_shape_size_fn_
      ABSL_GUARDED_BY(mu_);
};

PjRtClientTestFactoryRegistry& GetGlobalPjRtClientTestFactory();

void RegisterPjRtClientTestFactory(
    PjRtClientTestFactoryRegistry::PjRtClientFactory factory,
    PjRtClientTestFactoryRegistry::DeviceShapeRepresentationFnFactory
        registered_device_shape_representation_fn =
            PjRtClientTestFactoryRegistry::
                DefaultShapeRepresentationRegisteredFn,
    PjRtClientTestFactoryRegistry::DeviceShapeSizeFnFactory
        registered_device_shape_size_fn_ =
            PjRtClientTestFactoryRegistry::DefaultDeviceShapeSizeRegisteredFn);

bool ShouldUsePjRt();

}  // namespace xla

#endif  // XLA_TESTS_PJRT_CLIENT_REGISTRY_H_
