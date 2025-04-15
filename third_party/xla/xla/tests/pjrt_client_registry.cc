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

#include "xla/tests/pjrt_client_registry.h"

#include <functional>
#include <memory>
#include <utility>

namespace xla {

PjRtClientTestFactoryRegistry& GetGlobalPjRtClientTestFactory() {
  static auto* const factory = new PjRtClientTestFactoryRegistry;
  return *factory;
}

void RegisterPjRtClientTestFactory(
    PjRtClientTestFactoryRegistry::PjRtClientFactory factory,
    PjRtClientTestFactoryRegistry::DeviceShapeRepresentationFnFactory
        registered_device_shape_representation_fn,
    PjRtClientTestFactoryRegistry::DeviceShapeSizeFnFactory
        registered_device_shape_size_fn) {
  GetGlobalPjRtClientTestFactory().Register(
      std::move(factory), registered_device_shape_representation_fn,
      registered_device_shape_size_fn);
}

bool ShouldUsePjRt() {
  return GetGlobalPjRtClientTestFactory().HasRegisteredFactory();
}

}  // namespace xla
