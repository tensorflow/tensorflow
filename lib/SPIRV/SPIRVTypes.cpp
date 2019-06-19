//===- SPIRVTypes.cpp - MLIR SPIR-V Types ---------------------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file defines the types in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/SPIRV/SPIRVTypes.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace mlir::spirv;

// Pull in all enum utility function definitions
#include "mlir/SPIRV/SPIRVEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

struct spirv::detail::ArrayTypeStorage : public TypeStorage {
  using KeyTy = std::pair<Type, int64_t>;

  static ArrayTypeStorage *construct(TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<ArrayTypeStorage>()) ArrayTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, elementCount);
  }

  ArrayTypeStorage(const KeyTy &key)
      : elementType(key.first), elementCount(key.second) {}

  Type elementType;
  int64_t elementCount;
};

ArrayType ArrayType::get(Type elementType, int64_t elementCount) {
  return Base::get(elementType.getContext(), TypeKind::Array, elementType,
                   elementCount);
}

Type ArrayType::getElementType() { return getImpl()->elementType; }

int64_t ArrayType::getElementCount() { return getImpl()->elementCount; }

//===----------------------------------------------------------------------===//
// ImageType
//===----------------------------------------------------------------------===//

template <typename T> static constexpr unsigned getNumBits() { return 0; }
template <> constexpr unsigned getNumBits<Dim>() {
  static_assert((1 << 3) > getMaxEnumValForDim(),
                "Not enough bits to encode Dim value");
  return 3;
}
template <> constexpr unsigned getNumBits<ImageDepthInfo>() {
  static_assert((1 << 2) > getMaxEnumValForImageDepthInfo(),
                "Not enough bits to encode ImageDepthInfo value");
  return 2;
}
template <> constexpr unsigned getNumBits<ImageArrayedInfo>() {
  static_assert((1 << 1) > getMaxEnumValForImageArrayedInfo(),
                "Not enough bits to encode ImageArrayedInfo value");
  return 1;
}
template <> constexpr unsigned getNumBits<ImageSamplingInfo>() {
  static_assert((1 << 1) > getMaxEnumValForImageSamplingInfo(),
                "Not enough bits to encode ImageSamplingInfo value");
  return 1;
}
template <> constexpr unsigned getNumBits<ImageSamplerUseInfo>() {
  static_assert((1 << 2) > getMaxEnumValForImageSamplerUseInfo(),
                "Not enough bits to encode ImageSamplerUseInfo value");
  return 2;
}
template <> constexpr unsigned getNumBits<ImageFormat>() {
  static_assert((1 << 6) > getMaxEnumValForImageFormat(),
                "Not enough bits to encode ImageFormat value");
  return 6;
}

struct spirv::detail::ImageTypeStorage : public TypeStorage {
private:
  /// Define a bit-field struct to pack the enum values
  union EnumPack {
    struct {
      Dim dim : getNumBits<Dim>();
      ImageDepthInfo depthInfo : getNumBits<ImageDepthInfo>();
      ImageArrayedInfo arrayedInfo : getNumBits<ImageArrayedInfo>();
      ImageSamplingInfo samplingInfo : getNumBits<ImageSamplingInfo>();
      ImageSamplerUseInfo samplerUseInfo : getNumBits<ImageSamplerUseInfo>();
      ImageFormat format : getNumBits<ImageFormat>();
    } data;
    unsigned storage;
  };

public:
  using KeyTy = std::tuple<Type, Dim, ImageDepthInfo, ImageArrayedInfo,
                           ImageSamplingInfo, ImageSamplerUseInfo, ImageFormat>;

  static ImageTypeStorage *construct(TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<ImageTypeStorage>()) ImageTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, getDim(), getDepthInfo(), getArrayedInfo(),
                        getSamplingInfo(), getSamplerUseInfo(),
                        getImageFormat());
  }

  Dim getDim() const {
    EnumPack v;
    v.storage = getSubclassData();
    return v.data.dim;
  }
  void setDim(Dim dim) {
    EnumPack v;
    v.storage = getSubclassData();
    v.data.dim = dim;
    setSubclassData(v.storage);
  }

  ImageDepthInfo getDepthInfo() const {
    EnumPack v;
    v.storage = getSubclassData();
    return v.data.depthInfo;
  }
  void setDepthInfo(ImageDepthInfo depthInfo) {
    EnumPack v;
    v.storage = getSubclassData();
    v.data.depthInfo = depthInfo;
    setSubclassData(v.storage);
  }

  ImageArrayedInfo getArrayedInfo() const {
    EnumPack v;
    v.storage = getSubclassData();
    return v.data.arrayedInfo;
  }
  void setArrayedInfo(ImageArrayedInfo arrayedInfo) {
    EnumPack v;
    v.storage = getSubclassData();
    v.data.arrayedInfo = arrayedInfo;
    setSubclassData(v.storage);
  }

  ImageSamplingInfo getSamplingInfo() const {
    EnumPack v;
    v.storage = getSubclassData();
    return v.data.samplingInfo;
  }
  void setSamplingInfo(ImageSamplingInfo samplingInfo) {
    EnumPack v;
    v.storage = getSubclassData();
    v.data.samplingInfo = samplingInfo;
    setSubclassData(v.storage);
  }

  ImageSamplerUseInfo getSamplerUseInfo() const {
    EnumPack v;
    v.storage = getSubclassData();
    return v.data.samplerUseInfo;
  }
  void setSamplerUseInfo(ImageSamplerUseInfo samplerUseInfo) {
    EnumPack v;
    v.storage = getSubclassData();
    v.data.samplerUseInfo = samplerUseInfo;
    setSubclassData(v.storage);
  }

  ImageFormat getImageFormat() const {
    EnumPack v;
    v.storage = getSubclassData();
    return v.data.format;
  }
  void setImageFormat(ImageFormat format) {
    EnumPack v;
    v.storage = getSubclassData();
    v.data.format = format;
    setSubclassData(v.storage);
  }

  ImageTypeStorage(const KeyTy &key) : elementType(std::get<0>(key)) {
    static_assert(sizeof(EnumPack) <= sizeof(getSubclassData()),
                  "EnumPack size greater than subClassData type size");
    setDim(std::get<1>(key));
    setDepthInfo(std::get<2>(key));
    setArrayedInfo(std::get<3>(key));
    setSamplingInfo(std::get<4>(key));
    setSamplerUseInfo(std::get<5>(key));
    setImageFormat(std::get<6>(key));
  }

  Type elementType;
};

ImageType
ImageType::get(std::tuple<Type, Dim, ImageDepthInfo, ImageArrayedInfo,
                          ImageSamplingInfo, ImageSamplerUseInfo, ImageFormat>
                   value) {
  return Base::get(std::get<0>(value).getContext(), TypeKind::ImageType, value);
}

Type ImageType::getElementType() { return getImpl()->elementType; }

Dim ImageType::getDim() { return getImpl()->getDim(); }

ImageDepthInfo ImageType::getDepthInfo() { return getImpl()->getDepthInfo(); }

ImageArrayedInfo ImageType::getArrayedInfo() {
  return getImpl()->getArrayedInfo();
}

ImageSamplingInfo ImageType::getSamplingInfo() {
  return getImpl()->getSamplingInfo();
}

ImageSamplerUseInfo ImageType::getSamplerUseInfo() {
  return getImpl()->getSamplerUseInfo();
}

ImageFormat ImageType::getImageFormat() { return getImpl()->getImageFormat(); }

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

struct spirv::detail::PointerTypeStorage : public TypeStorage {
  // (Type, StorageClass) as the key: Type stored in this struct, and
  // StorageClass stored as TypeStorage's subclass data.
  using KeyTy = std::pair<Type, StorageClass>;

  static PointerTypeStorage *construct(TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    return new (allocator.allocate<PointerTypeStorage>())
        PointerTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(pointeeType, getStorageClass());
  }

  PointerTypeStorage(const KeyTy &key)
      : TypeStorage(static_cast<unsigned>(key.second)), pointeeType(key.first) {
  }

  StorageClass getStorageClass() const {
    return static_cast<StorageClass>(getSubclassData());
  }

  Type pointeeType;
};

PointerType PointerType::get(Type pointeeType, StorageClass storageClass) {
  return Base::get(pointeeType.getContext(), TypeKind::Pointer, pointeeType,
                   storageClass);
}

Type PointerType::getPointeeType() { return getImpl()->pointeeType; }

StorageClass PointerType::getStorageClass() {
  return getImpl()->getStorageClass();
}

StringRef PointerType::getStorageClassStr() {
  return stringifyStorageClass(getStorageClass());
}

//===----------------------------------------------------------------------===//
// RuntimeArrayType
//===----------------------------------------------------------------------===//

struct spirv::detail::RuntimeArrayTypeStorage : public TypeStorage {
  using KeyTy = Type;

  static RuntimeArrayTypeStorage *construct(TypeStorageAllocator &allocator,
                                            const KeyTy &key) {
    return new (allocator.allocate<RuntimeArrayTypeStorage>())
        RuntimeArrayTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const { return elementType == key; }

  RuntimeArrayTypeStorage(const KeyTy &key) : elementType(key) {}

  Type elementType;
};

RuntimeArrayType RuntimeArrayType::get(Type elementType) {
  return Base::get(elementType.getContext(), TypeKind::RuntimeArray,
                   elementType);
}

Type RuntimeArrayType::getElementType() { return getImpl()->elementType; }
