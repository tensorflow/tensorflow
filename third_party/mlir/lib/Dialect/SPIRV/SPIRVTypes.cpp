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

#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace mlir::spirv;

// Pull in all enum utility function definitions
#include "mlir/Dialect/SPIRV/SPIRVEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

struct spirv::detail::ArrayTypeStorage : public TypeStorage {
  using KeyTy = std::pair<Type, unsigned>;

  static ArrayTypeStorage *construct(TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<ArrayTypeStorage>()) ArrayTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, getSubclassData());
  }

  ArrayTypeStorage(const KeyTy &key)
      : TypeStorage(key.second), elementType(key.first) {}

  Type elementType;
};

ArrayType ArrayType::get(Type elementType, unsigned elementCount) {
  return Base::get(elementType.getContext(), TypeKind::Array, elementType,
                   elementCount);
}

unsigned ArrayType::getNumElements() const {
  return getImpl()->getSubclassData();
}

Type ArrayType::getElementType() const { return getImpl()->elementType; }

//===----------------------------------------------------------------------===//
// CompositeType
//===----------------------------------------------------------------------===//

Type CompositeType::getElementType(unsigned index) const {
  switch (getKind()) {
  case spirv::TypeKind::Array:
    return cast<ArrayType>().getElementType();
  case spirv::TypeKind::Struct:
    return cast<StructType>().getElementType(index);
  case StandardTypes::Vector:
    return cast<VectorType>().getElementType();
  default:
    llvm_unreachable("invalid composite type");
  }
}

unsigned CompositeType::getNumElements() const {
  switch (getKind()) {
  case spirv::TypeKind::Array:
    return cast<ArrayType>().getNumElements();
  case spirv::TypeKind::Struct:
    return cast<StructType>().getNumElements();
  case StandardTypes::Vector:
    return cast<VectorType>().getNumElements();
  default:
    llvm_unreachable("invalid composite type");
  }
}

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
      unsigned dimEncoding : getNumBits<Dim>();
      unsigned depthInfoEncoding : getNumBits<ImageDepthInfo>();
      unsigned arrayedInfoEncoding : getNumBits<ImageArrayedInfo>();
      unsigned samplingInfoEncoding : getNumBits<ImageSamplingInfo>();
      unsigned samplerUseInfoEncoding : getNumBits<ImageSamplerUseInfo>();
      unsigned formatEncoding : getNumBits<ImageFormat>();
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
    return static_cast<Dim>(v.data.dimEncoding);
  }
  void setDim(Dim dim) {
    EnumPack v;
    v.storage = getSubclassData();
    v.data.dimEncoding = static_cast<unsigned>(dim);
    setSubclassData(v.storage);
  }

  ImageDepthInfo getDepthInfo() const {
    EnumPack v;
    v.storage = getSubclassData();
    return static_cast<ImageDepthInfo>(v.data.depthInfoEncoding);
  }
  void setDepthInfo(ImageDepthInfo depthInfo) {
    EnumPack v;
    v.storage = getSubclassData();
    v.data.depthInfoEncoding = static_cast<unsigned>(depthInfo);
    setSubclassData(v.storage);
  }

  ImageArrayedInfo getArrayedInfo() const {
    EnumPack v;
    v.storage = getSubclassData();
    return static_cast<ImageArrayedInfo>(v.data.arrayedInfoEncoding);
  }
  void setArrayedInfo(ImageArrayedInfo arrayedInfo) {
    EnumPack v;
    v.storage = getSubclassData();
    v.data.arrayedInfoEncoding = static_cast<unsigned>(arrayedInfo);
    setSubclassData(v.storage);
  }

  ImageSamplingInfo getSamplingInfo() const {
    EnumPack v;
    v.storage = getSubclassData();
    return static_cast<ImageSamplingInfo>(v.data.samplingInfoEncoding);
  }
  void setSamplingInfo(ImageSamplingInfo samplingInfo) {
    EnumPack v;
    v.storage = getSubclassData();
    v.data.samplingInfoEncoding = static_cast<unsigned>(samplingInfo);
    setSubclassData(v.storage);
  }

  ImageSamplerUseInfo getSamplerUseInfo() const {
    EnumPack v;
    v.storage = getSubclassData();
    return static_cast<ImageSamplerUseInfo>(v.data.samplerUseInfoEncoding);
  }
  void setSamplerUseInfo(ImageSamplerUseInfo samplerUseInfo) {
    EnumPack v;
    v.storage = getSubclassData();
    v.data.samplerUseInfoEncoding = static_cast<unsigned>(samplerUseInfo);
    setSubclassData(v.storage);
  }

  ImageFormat getImageFormat() const {
    EnumPack v;
    v.storage = getSubclassData();
    return static_cast<ImageFormat>(v.data.formatEncoding);
  }
  void setImageFormat(ImageFormat format) {
    EnumPack v;
    v.storage = getSubclassData();
    v.data.formatEncoding = static_cast<unsigned>(format);
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
  return Base::get(std::get<0>(value).getContext(), TypeKind::Image, value);
}

Type ImageType::getElementType() const { return getImpl()->elementType; }

Dim ImageType::getDim() const { return getImpl()->getDim(); }

ImageDepthInfo ImageType::getDepthInfo() const {
  return getImpl()->getDepthInfo();
}

ImageArrayedInfo ImageType::getArrayedInfo() const {
  return getImpl()->getArrayedInfo();
}

ImageSamplingInfo ImageType::getSamplingInfo() const {
  return getImpl()->getSamplingInfo();
}

ImageSamplerUseInfo ImageType::getSamplerUseInfo() const {
  return getImpl()->getSamplerUseInfo();
}

ImageFormat ImageType::getImageFormat() const {
  return getImpl()->getImageFormat();
}

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

Type PointerType::getPointeeType() const { return getImpl()->pointeeType; }

StorageClass PointerType::getStorageClass() const {
  return getImpl()->getStorageClass();
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

Type RuntimeArrayType::getElementType() const { return getImpl()->elementType; }

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

struct spirv::detail::StructTypeStorage : public TypeStorage {
  StructTypeStorage(unsigned numMembers, Type const *memberTypes,
                    StructType::LayoutInfo const *layoutInfo)
      : TypeStorage(numMembers), memberTypes(memberTypes),
        layoutInfo(layoutInfo) {}

  using KeyTy = std::pair<ArrayRef<Type>, ArrayRef<StructType::LayoutInfo>>;
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(getMemberTypes(), getLayoutInfo());
  }

  static StructTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    ArrayRef<Type> keyTypes = key.first;

    // Copy the member type and layout information into the bump pointer
    auto typesList = allocator.copyInto(keyTypes).data();

    const StructType::LayoutInfo *layoutInfoList = nullptr;
    if (!key.second.empty()) {
      ArrayRef<StructType::LayoutInfo> keyLayoutInfo = key.second;
      assert(keyLayoutInfo.size() == keyTypes.size() &&
             "size of layout information must be same as the size of number of "
             "elements");
      layoutInfoList = allocator.copyInto(keyLayoutInfo).data();
    }

    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(keyTypes.size(), typesList, layoutInfoList);
  }

  ArrayRef<Type> getMemberTypes() const {
    return ArrayRef<Type>(memberTypes, getSubclassData());
  }

  ArrayRef<StructType::LayoutInfo> getLayoutInfo() const {
    if (layoutInfo) {
      return ArrayRef<StructType::LayoutInfo>(layoutInfo, getSubclassData());
    }
    return ArrayRef<StructType::LayoutInfo>(nullptr, size_t(0));
  }

  Type const *memberTypes;
  StructType::LayoutInfo const *layoutInfo;
};

StructType StructType::get(ArrayRef<Type> memberTypes) {
  assert(!memberTypes.empty() && "Struct needs at least one member type");
  ArrayRef<StructType::LayoutInfo> noLayout(nullptr, size_t(0));
  return Base::get(memberTypes[0].getContext(), TypeKind::Struct, memberTypes,
                   noLayout);
}

StructType StructType::get(ArrayRef<Type> memberTypes,
                           ArrayRef<StructType::LayoutInfo> layoutInfo) {
  assert(!memberTypes.empty() && "Struct needs at least one member type");
  return Base::get(memberTypes.vec().front().getContext(), TypeKind::Struct,
                   memberTypes, layoutInfo);
}

unsigned StructType::getNumElements() const {
  return getImpl()->getSubclassData();
}

Type StructType::getElementType(unsigned index) const {
  assert(
      getNumElements() > index &&
      "element index is more than number of members of the SPIR-V StructType");
  return getImpl()->memberTypes[index];
}

bool StructType::hasLayout() const { return getImpl()->layoutInfo; }

uint64_t StructType::getOffset(unsigned index) const {
  assert(
      getNumElements() > index &&
      "element index is more than number of members of the SPIR-V StructType");
  return getImpl()->layoutInfo[index];
}
