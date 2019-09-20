#include "native_type_test_impl.h"
#include "native_type_test_generated.h"

namespace flatbuffers {
  Geometry::Vector3D Pack(const Native::Vector3D &obj) {
    return Geometry::Vector3D(obj.x, obj.y, obj.z);
  }

  const Native::Vector3D UnPack(const Geometry::Vector3D &obj) {
    return Native::Vector3D(obj.x(), obj.y(), obj.z());
  }
}

