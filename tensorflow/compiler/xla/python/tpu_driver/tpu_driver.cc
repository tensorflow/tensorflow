// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.h"

#include "absl/strings/match.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/util.h"

namespace tpu_driver {

namespace {

typedef absl::flat_hash_map<
    std::string, std::function<xla::StatusOr<std::unique_ptr<TpuDriver>>(
                     const TpuDriverConfig&)>>
    DriverRegistryMap;

DriverRegistryMap* GetDriverRegistryMap() {
  static DriverRegistryMap* driver_registry = new DriverRegistryMap();
  return driver_registry;
}

int64_t ByteSizeOfPrimitiveType(xla::PrimitiveType primitive_type) {
  switch (primitive_type) {
    case xla::PrimitiveType::PRED:
      return sizeof(int8_t);
    case xla::PrimitiveType::S8:
      return sizeof(int8_t);
    case xla::PrimitiveType::S16:
      return sizeof(int16_t);
    case xla::PrimitiveType::S32:
      return sizeof(int32_t);
    case xla::PrimitiveType::S64:
      return sizeof(int64_t);
    case xla::PrimitiveType::U8:
      return sizeof(uint8_t);
    case xla::PrimitiveType::U16:
      return sizeof(uint16_t);
    case xla::PrimitiveType::U32:
      return sizeof(uint32_t);
    case xla::PrimitiveType::U64:
      return sizeof(uint64_t);
    case xla::PrimitiveType::BF16:
      return sizeof(float) / 2;
    case xla::PrimitiveType::F16:
      return sizeof(float) / 2;
    case xla::PrimitiveType::F32:
      return sizeof(float);
    case xla::PrimitiveType::F64:
      return sizeof(double);
    case xla::PrimitiveType::C64:
      return sizeof(std::complex<float>);
    case xla::PrimitiveType::C128:
      return sizeof(std::complex<double>);
    case xla::PrimitiveType::TOKEN:
    case xla::PrimitiveType::TUPLE:
    case xla::PrimitiveType::OPAQUE_TYPE:
      LOG(FATAL) << PrimitiveType_Name(primitive_type)
                 << " primitive type has no definitive size";
    default:
      LOG(FATAL) << "Unhandled primitive type " << primitive_type;
  }
}

}  // namespace

/*static*/ int TpuDriverRegistry::RegisterDriver(
    const std::string& prefix,
    const std::function<xla::StatusOr<std::unique_ptr<TpuDriver>>(
        const TpuDriverConfig&)>& creator) {
  (*GetDriverRegistryMap())[prefix] = creator;
  return GetDriverRegistryMap()->size();
}

/*static*/ xla::StatusOr<std::unique_ptr<TpuDriver>> TpuDriverRegistry::Open(
    const TpuDriverConfig& config) {
  for (const auto& driver : *GetDriverRegistryMap()) {
    if (absl::StartsWith(config.worker(), driver.first)) {
      return driver.second(config);
    }
  }
  return xla::NotFound("Unable to find driver in registry given worker: %s",
                       config.worker());
}

int64_t ComputeBytesFromShape(const xla::ShapeProto& shape) {
  if (shape.tuple_shapes_size() > 0) {
    LOG(FATAL) << "Tuples are not supported at the moment.";
  }

  int64_t num_elems = 1;
  for (auto dim : shape.dimensions()) {
    num_elems *= dim;
  }

  return ByteSizeOfPrimitiveType(shape.element_type()) * num_elems;
}

}  // namespace tpu_driver
