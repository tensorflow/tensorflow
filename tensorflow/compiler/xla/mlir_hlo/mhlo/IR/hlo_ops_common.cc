/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "mhlo/IR/hlo_ops_common.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace hlo {
// Verifies the source target pairs attached to collective permute.
LogicalResult verifyCollectivePermuteSourceTargetPairs(
    Operation *op, DenseIntElementsAttr attr) {
  auto type = attr.getType().cast<RankedTensorType>();
  if (type.getRank() != 2)
    return op->emitError() << "expect source_target_pairs attribute to be of "
                              "rank 2, but got rank "
                           << type.getRank();
  if (type.getShape()[1] != 2)
    return op->emitError()
           << "expect source_target_pairs attribute of shape (N, 2), but got ("
           << type.getShape() << ")";
  // Check source target pairs for duplicate sources or targets.
  llvm::DenseSet<int64_t> sources;
  llvm::DenseSet<int64_t> targets;
  for (auto i = attr.begin(), e = attr.end(); i != e; ++i) {
    auto val = (*i).getSExtValue();
    if (val < 0)
      return op->emitError()
             << "replica ids in source_target_pairs must be >= 0.";

    if (i.getIndex() % 2 == 0) {
      bool isUnique = sources.insert(val).second;
      if (!isUnique) return op->emitError() << "duplicate sources not allowed.";
    } else {
      bool isUnique = targets.insert(val).second;
      if (!isUnique) return op->emitError() << "duplicate targets not allowed.";
    }
  }
  return success();
}

LogicalResult verifyReduceScatter(Operation *op, TypeRange operandTypes,
                                  TypeRange resultTypes,
                                  uint64_t scatterDimension) {
  // If operand and result are both ranked, then the size of the scatter
  // dimension in the operand should be a multiple of the size of the scatter
  // dimension in the result.

  // TODO(zhouxin) Change the ODS definition to return int64_t.
  if (static_cast<int64_t>(scatterDimension) < 0) {
    return op->emitOpError("expects scatter_dimension >= 0");
  }

  for (auto it : llvm::zip(operandTypes, resultTypes)) {
    auto operandType = std::get<0>(it).cast<ShapedType>();
    auto resultType = std::get<1>(it).cast<ShapedType>();
    if (!operandType.hasRank() || !resultType.hasRank()) continue;
    if (operandType.getRank() != resultType.getRank())
      return op->emitOpError() << "operand and result should have same rank";
    if (static_cast<int64_t>(scatterDimension) >= operandType.getRank())
      return op->emitOpError()
             << "scatter dim should be less than operand/result rank";
    if (operandType.isDynamicDim(scatterDimension) ||
        resultType.isDynamicDim(scatterDimension))
      continue;
    if (operandType.getDimSize(scatterDimension) == 0)
      return op->emitOpError() << "operand scatter dimension cannot be zero";
    if (resultType.getDimSize(scatterDimension) == 0)
      return op->emitOpError() << "result scatter dimension cannot be zero";
    if ((operandType.getDimSize(scatterDimension) %
         resultType.getDimSize(scatterDimension)) != 0)
      return op->emitOpError()
             << "operand scatter dimension has size "
             << operandType.getDimSize(scatterDimension)
             << ", expected to be a multiple of result scatter dimension size "
             << resultType.getDimSize(scatterDimension);

    // Non scatter dimensions should be equal.
    for (uint64_t index : llvm::seq<uint64_t>(0, operandType.getRank())) {
      if (index == scatterDimension || operandType.isDynamicDim(index) ||
          resultType.isDynamicDim(index))
        continue;
      if (operandType.getDimSize(index) != resultType.getDimSize(index))
        return op->emitOpError()
               << "non scatter dimensions should be same for operand ("
               << operandType.getDimSize(index) << ") and result ("
               << resultType.getDimSize(index) << ")";
    }
  }
  return success();
}

namespace {
// Custom formatting for convolution window attributes.
void printWindowAttribute(OpAsmPrinter &p, DenseElementsAttr attribute) {
  if (attribute.getElementType().isInteger(/*width=*/1)) {
    // boolean attribute.
    llvm::interleaveComma(attribute.getValues<bool>(), p,
                          [&](bool b) { p << (b ? 1 : 0); });
    return;
  }
  if (attribute.getType().getRank() == 2) {
    // Padding is Nx2 attribute.
    auto it = attribute.value_begin<int64_t>();
    std::vector<std::pair<int64_t, int64_t>> values(attribute.getNumElements() /
                                                    2);
    for (auto &item : values) {
      int64_t first = *it;
      ++it;
      int64_t second = *it;
      ++it;
      item = {first, second};
    }
    llvm::interleaveComma(
        values, p, [&](const std::pair<int64_t, int64_t> pair) {
          p << '[' << pair.first << ", " << pair.second << ']';
        });
  } else {
    llvm::interleaveComma(attribute.getValues<int64_t>(), p);
  }
}
}  // namespace

void printWindowAttributes(OpAsmPrinter &p, Operation * /*op*/,
                           llvm::Optional<DenseIntElementsAttr> windowStrides,
                           llvm::Optional<DenseIntElementsAttr> padding,
                           llvm::Optional<DenseIntElementsAttr> lhsDilation,
                           llvm::Optional<DenseIntElementsAttr> rhsDilation,
                           llvm::Optional<DenseElementsAttr> windowReversal) {
  using pair_t = std::pair<DenseElementsAttr, StringRef>;
  std::array<pair_t, 5> printedAttributes = {{
      {windowStrides ? *windowStrides : nullptr, "stride"},
      {padding ? *padding : nullptr, "pad"},
      {lhsDilation ? *lhsDilation : nullptr, "lhs_dilate"},
      {rhsDilation ? *rhsDilation : nullptr, "rhs_dilate"},
      {windowReversal ? *windowReversal : nullptr, "reverse"},
  }};

  // Do not print attributes that do no exist.
  auto nonNullAttributes = llvm::make_filter_range(
      printedAttributes,
      [](const pair_t &a) { return static_cast<bool>(a.first); });

  llvm::interleaveComma(nonNullAttributes, p, [&](const pair_t &a) {
    p << a.second << " = [";
    printWindowAttribute(p, a.first);
    p << "]";
  });
}

ParseResult parseWindowAttributes(OpAsmParser &parser,
                                  DenseIntElementsAttr &windowStrides,
                                  DenseIntElementsAttr &padding,
                                  DenseIntElementsAttr &lhsDilation,
                                  DenseIntElementsAttr &rhsDilation,
                                  DenseElementsAttr &windowReversal) {
  StringRef attributeName;

  llvm::StringSet<> allowedAttributeNames{
      {"stride", "pad", "lhs_dilate", "rhs_dilate", "reverse"}};

  while (parser.parseOptionalKeyword(&attributeName).succeeded()) {
    // Verify that the attribute name is valid and erase it.
    if (!allowedAttributeNames.erase(attributeName)) {
      return parser.emitError(parser.getCurrentLocation(),
                              "Unexpected keyword ")
             << attributeName;
    }

    if (parser.parseEqual()) {
      return failure();
    }

    // parse the attribute value. We need to support either 1D and Nx2 array of
    // integers to parse.
    llvm::SmallVector<int64_t> values;
    auto int64Parser = [&]() {
      return parser.parseInteger(values.emplace_back(0));
    };

    if (attributeName == "pad") {
      // Parse 2D array of integers.
      // Helper to parse an array of two integer elements such as [e0, e1].
      auto innerParser = [&]() -> ParseResult {
        size_t numOldElements = values.size();
        if (parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Square,
                                           int64Parser))
          return failure();
        size_t numParsedElements = values.size() - numOldElements;
        constexpr size_t kExpectedElements = 2;
        if (numParsedElements != kExpectedElements)
          return parser.emitError(parser.getCurrentLocation())
                 << "Expected array with " << kExpectedElements
                 << " elements, got " << numParsedElements
                 << " elements instead";
        return success();
      };

      if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                         innerParser)) {
        return failure();
      }
      const int64_t size = static_cast<int64_t>(values.size());
      // values should be filled with the Nx2 padding values.
      assert(size % 2 == 0);
      auto ty = RankedTensorType::get({size / 2, 2},
                                      parser.getBuilder().getIntegerType(64));
      padding = DenseIntElementsAttr::get(ty, values);
    } else {
      // Parse 1D array of integers.
      if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                         int64Parser)) {
        return failure();
      }
      const int64_t size = static_cast<int64_t>(values.size());
      if (attributeName == "reverse") {
        auto ty = RankedTensorType::get({size},
                                        parser.getBuilder().getIntegerType(1));
        auto boolVector = llvm::to_vector<4>(
            llvm::map_range(values, [](int64_t v) { return v != 0; }));
        windowReversal = DenseElementsAttr::get(ty, boolVector);
      } else {
        auto attr = parser.getBuilder().getI64TensorAttr(values);

        if (attributeName == "stride") {
          windowStrides = attr;
        } else if (attributeName == "lhs_dilate") {
          lhsDilation = attr;
        } else if (attributeName == "rhs_dilate") {
          rhsDilation = attr;
        } else {
          llvm_unreachable("Unexpected attribute name");
        }
      }
    }
    // continue parsing if there is a comma at the end.
    if (parser.parseOptionalComma().failed()) break;
  }
  return success();
}

}  // namespace hlo
}  // namespace mlir
