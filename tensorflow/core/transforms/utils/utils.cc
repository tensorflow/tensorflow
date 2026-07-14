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

#include "tensorflow/core/transforms/utils/utils.h"

#include <cassert>
#include <cstdint>
#include <numeric>
#include <optional>
#include <string>

#include "absl/strings/match.h"
#include "llvm/ADT/BitVector.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/interfaces.h"
#include "tensorflow/core/ir/tf_op_wrapper.h"
#include "tensorflow/core/ir/utility.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace mlir {
namespace tfg {
namespace util {

bool OpHasDevice(Operation *op, const char *device_name) {
  std::string task, device;
  return tensorflow::DeviceNameUtils::SplitDeviceName(TFOp(op).device().data(),
                                                      &task, &device) &&
         absl::StartsWith(device, device_name);
}

void EraseRegularNodeAttributes(NamedAttrList &attr_list) {
  NamedAttrList new_attr_list;
  for (NamedAttribute attr : attr_list) {
    if (attr.getName().strref().starts_with("_")) new_attr_list.append(attr);
  }
  attr_list = new_attr_list;
}

void ForwardNonIntrinsicAttributes(Operation *src, Operation *dst) {
  NamedAttrList dst_attrs(dst->getAttrDictionary());
  DenseSet<StringAttr> name_set;

  // Forward all non-intrinsic attributes. If the source op is unregistered,
  // forward all attributes.
  if (std::optional<RegisteredOperationName> src_name =
          src->getRegisteredInfo()) {
    ArrayRef<StringAttr> src_attr_names = src_name->getAttributeNames();
    name_set.insert(src_attr_names.begin(), src_attr_names.end());
  }
  for (const NamedAttribute &attr : src->getAttrs()) {
    if (!name_set.contains(attr.getName())) dst_attrs.append(attr);
  }

  dst->setAttrs(dst_attrs.getDictionary(dst->getContext()));
}

static void UpdateIfPresent(Region &region,
                            function_ref<RegionAttr(RegionAttr)> copy_update) {
  unsigned index = region.getRegionNumber();
  auto iface = cast<PreservedAttributesInterface>(region.getParentOp());
  if (auto attrs = iface.getPreservedAttrs(index))
    iface.setPreservedAttrs(index, copy_update(attrs));
}

static void UpdateArgAttrsIfPresent(
    Region &region, function_ref<void(SmallVectorImpl<Attribute> &)> update) {
  UpdateIfPresent(region, [&](RegionAttr attrs) {
    SmallVector<Attribute> args = llvm::to_vector(attrs.getArgAttrs());
    update(args);
    return RegionAttr::get(attrs.getAttrs(),
                           ArrayAttr::get(attrs.getContext(), args),
                           attrs.getResAttrs());
  });
}

static void UpdateResultAttrsIfPresent(
    Region &region, function_ref<void(SmallVectorImpl<Attribute> &)> update) {
  UpdateIfPresent(region, [&](RegionAttr attrs) {
    SmallVector<Attribute> results = llvm::to_vector(attrs.getResAttrs());
    update(results);
    return RegionAttr::get(attrs.getAttrs(), attrs.getArgAttrs(),
                           ArrayAttr::get(attrs.getContext(), results));
  });
}

LoopRegionArgumentUpdate LoopRegionAddArgument(Region &region, Type type) {
  // Add the arguments.
  BlockArgument data = region.insertArgument(
      GetLoopRegionDataArgs(region).size(), type, region.getLoc());
  BlockArgument ctl =
      region.addArgument(ControlType::get(type.getContext()), region.getLoc());

  UpdateArgAttrsIfPresent(region, [&](SmallVectorImpl<Attribute> &arg_attrs) {
    arg_attrs.push_back(DictionaryAttr::get(type.getContext(), {}));
  });

  return {data, ctl};
}

void LoopRegionEraseArgument(Region &region, unsigned index) {
  Block::BlockArgListType args = GetLoopRegionDataArgs(region);
  assert(index < args.size());

  // Erase the arguments.
  BitVector indices(region.front().getNumArguments());
  indices.set(args[index].getArgNumber());
  indices.set(GetLoopRegionControlOf(args[index]).getArgNumber());
  region.front().eraseArguments(indices);

  UpdateArgAttrsIfPresent(region, [&](SmallVectorImpl<Attribute> &arg_attrs) {
    arg_attrs.erase(arg_attrs.begin() + index);
  });
}

void LoopRegionResultAdded(Region &region, unsigned num) {
  UpdateResultAttrsIfPresent(
      region, [&](SmallVectorImpl<Attribute> &res_attrs) {
        res_attrs.append(num, DictionaryAttr::get(region.getContext(), {}));
      });
}

void LoopRegionResultErased(Region &region, unsigned index) {
  UpdateResultAttrsIfPresent(region,
                             [&](SmallVectorImpl<Attribute> &res_attrs) {
                               res_attrs.erase(res_attrs.begin() + index);
                             });
}

void SizedOperandSegmentsEraseOperands(Operation *op,
                                       ArrayRef<unsigned> indices) {
  llvm::BitVector erase(op->getNumOperands());
  for (unsigned index : indices) erase.set(index);
  SizedOperandSegmentsEraseOperands(op, erase);
}

void SizedOperandSegmentsEraseOperands(Operation *op,
                                       const llvm::BitVector &erase) {
  // Update the segment sizes if present.
  Builder b(op->getContext());
  StringAttr attr_name = b.getStringAttr(
      OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr());
  auto segment_sizes = op->getAttrOfType<DenseI32ArrayAttr>(attr_name);
  if (segment_sizes) {
    auto values = segment_sizes.asArrayRef();
    SmallVector<int32_t> new_sizes = llvm::to_vector(values);

    unsigned base = 0;
    for (auto it : llvm::zip(values, new_sizes)) {
      int32_t size = std::get<0>(it);
      int32_t &new_size = std::get<1>(it);
      for (int32_t i = 0; i < size; ++i)
        if (erase.test(base + i)) --new_size;
      base += size;
    }
    assert(llvm::all_of(new_sizes, [](int32_t size) { return size >= 0; }));
    assert(std::accumulate(new_sizes.begin(), new_sizes.end(), 0) ==
           op->getNumOperands() - erase.count());
    segment_sizes = b.getDenseI32ArrayAttr(new_sizes);
  }

  op->eraseOperands(erase);
  if (segment_sizes) op->setAttr(attr_name, segment_sizes);
}

}  // namespace util
}  // namespace tfg
}  // namespace mlir
