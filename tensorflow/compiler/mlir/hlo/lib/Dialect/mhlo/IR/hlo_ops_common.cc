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

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_common.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace hlo {
// Verifies the source target pairs attached to collective permute.
LogicalResult VerifyCollectivePermuteSourceTargetPairs(
    Operation *op, DenseIntElementsAttr attr) {
  auto type = attr.getType().dyn_cast<RankedTensorType>();
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
    if (i.getIndex() % 2 == 0) {
      bool is_unique = sources.insert(val).second;
      if (!is_unique)
        return op->emitError() << "duplicate sources not allowed.";
    } else {
      bool is_unique = targets.insert(val).second;
      if (!is_unique)
        return op->emitError() << "duplicate targets not allowed.";
    }
  }
  return success();
}

namespace {
// Custom formatting for convolution window attributes.
void printWindowAttribute(OpAsmPrinter &p, DenseElementsAttr attribute) {
  if (attribute.getType().getElementType().isInteger(/*width=*/1)) {
    // boolean attribute.
    llvm::interleaveComma(attribute.getBoolValues(), p,
                          [&](bool b) { p << (b ? 1 : 0); });
    return;
  }
  if (attribute.getType().getRank() == 2) {
    // Padding is Nx2 attribute.
    auto it = attribute.getValues<int64_t>().begin();
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

void printWindowAttributes(OpAsmPrinter &p, Operation *op,
                           llvm::Optional<DenseIntElementsAttr> window_strides,
                           llvm::Optional<DenseIntElementsAttr> padding,
                           llvm::Optional<DenseIntElementsAttr> lhs_dilation,
                           llvm::Optional<DenseIntElementsAttr> rhs_dilation,
                           llvm::Optional<DenseElementsAttr> window_reversal) {
  using pair_t = std::pair<DenseElementsAttr, StringRef>;
  std::array<pair_t, 5> printed_attributes = {{
      {window_strides ? *window_strides : nullptr, "stride"},
      {padding ? *padding : nullptr, "pad"},
      {lhs_dilation ? *lhs_dilation : nullptr, "lhs_dilate"},
      {rhs_dilation ? *rhs_dilation : nullptr, "rhs_dilate"},
      {window_reversal ? *window_reversal : nullptr, "reverse"},
  }};

  // Do not print attributes that do no exist.
  auto non_null_attributes = llvm::make_filter_range(
      printed_attributes,
      [](const pair_t &a) { return static_cast<bool>(a.first); });

  llvm::interleaveComma(non_null_attributes, p, [&](const pair_t &a) {
    p << a.second << " = [";
    printWindowAttribute(p, a.first);
    p << "]";
  });
}

ParseResult parseWindowAttributes(OpAsmParser &parser,
                                  DenseIntElementsAttr &window_strides,
                                  DenseIntElementsAttr &padding,
                                  DenseIntElementsAttr &lhs_dilation,
                                  DenseIntElementsAttr &rhs_dilation,
                                  DenseElementsAttr &window_reversal) {
  StringRef attribute_name;

  // Helper to parse an array of the form [ e0, e1, .. ]
  auto parse_array = [&](std::function<ParseResult(void)> parse_element,
                         llvm::Optional<size_t> expected_size =
                             llvm::None) -> ParseResult {
    if (parser.parseLSquare()) {
      return failure();
    }
    size_t size = 0;
    do {
      if (parse_element()) {
        return failure();
      }
      size++;
    } while (parser.parseOptionalComma().succeeded());
    if (parser.parseRSquare()) {
      return failure();
    }
    if (expected_size && size != *expected_size) {
      return parser.emitError(parser.getCurrentLocation(),
                              "Expected array with")
             << *expected_size << " elements, got " << size
             << " elements instead";
    }
    return success();
  };

  llvm::StringSet<> allowed_attribute_names{
      {"stride", "pad", "lhs_dilate", "rhs_dilate", "reverse"}};

  while (parser.parseOptionalKeyword(&attribute_name).succeeded()) {
    // Verify that the attribute name is valid and erase it.
    if (!allowed_attribute_names.erase(attribute_name)) {
      return parser.emitError(parser.getCurrentLocation(),
                              "Unexpected keyword ")
             << attribute_name;
    }

    if (parser.parseEqual()) {
      return failure();
    }

    // parse the attribute value. We need to support either 1D and Nx2 array of
    // integers to parse.
    llvm::SmallVector<int64_t> values;
    auto int64_parser = [&]() {
      return parser.parseInteger(values.emplace_back(0));
    };

    if (attribute_name == "pad") {
      // Parse a 2D array of integers.
      auto inner_parser = [&]() {
        return parse_array(int64_parser, /*expected_size=*/2);
      };
      if (parse_array(inner_parser)) {
        return failure();
      }
      const int64_t size = static_cast<int64_t>(values.size());
      // values should be filled with the Nx2 padding values.
      auto ty = RankedTensorType::get({size / 2, 2},
                                      parser.getBuilder().getIntegerType(64));
      padding = DenseIntElementsAttr::get(ty, values);
    } else {
      // Parse 1D array of integers.
      if (parse_array(int64_parser)) {
        return failure();
      }
      const int64_t size = static_cast<int64_t>(values.size());
      if (attribute_name == "reverse") {
        auto ty = RankedTensorType::get({size},
                                        parser.getBuilder().getIntegerType(1));
        auto bool_vector = llvm::to_vector<4>(
            llvm::map_range(values, [](int64_t v) { return v != 0; }));
        window_reversal = DenseElementsAttr::get(ty, bool_vector);
      } else {
        auto attr = parser.getBuilder().getI64TensorAttr(values);

        if (attribute_name == "stride") {
          window_strides = attr;
        } else if (attribute_name == "lhs_dilate") {
          lhs_dilation = attr;
        } else if (attribute_name == "rhs_dilate") {
          rhs_dilation = attr;
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
