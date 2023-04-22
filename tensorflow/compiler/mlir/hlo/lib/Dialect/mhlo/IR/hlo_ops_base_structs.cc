/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_structs.h"

#include <set>
#include <unordered_map>

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_structs.cc.inc"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace mhlo {

namespace {
enum NonSpatialDim : int64_t {
  IOBatch = -1,    // Input or output batch dimension
  IOFeature = -2,  // Input or output feature dimension
  KIFeature = -3,  // Kernel input feature dimension
  KOFeature = -4,  // Kernel output feature dimensions.
};

char NonSpatialDimToString(NonSpatialDim dim) {
  switch (dim) {
    case IOBatch:
      return 'b';
    case IOFeature:
      return 'f';
    case KIFeature:
      return 'i';
    case KOFeature:
      return 'o';
  }
}
}  // namespace

// Custom printer and parser for struct attributes.
void printConvolutionDimensions(OpAsmPrinter &p, Operation * /*op*/,
                                ConvDimensionNumbers dnums) {
  auto print_dim =
      [&p](DenseIntElementsAttr spatial_dims,
           ArrayRef<std::pair<IntegerAttr, NonSpatialDim>> non_spatial_dims) {
        llvm::SmallVector<int64_t> dims(non_spatial_dims.size() +
                                        spatial_dims.size());
        // Fill each element of dims with a (< 0) NonSpatialDim enum or a (>=0)
        // spatial dimension index.
        for (const std::pair<IntegerAttr, NonSpatialDim> &non_spatial_dim :
             non_spatial_dims) {
          dims[non_spatial_dim.first.getInt()] = non_spatial_dim.second;
        }
        for (auto spatial_dim :
             llvm::enumerate(spatial_dims.getValues<int64_t>())) {
          dims[spatial_dim.value()] = static_cast<int64_t>(spatial_dim.index());
        }

        // Each dimension numbers will be printed as a comma separated list
        // surrounded by square brackets, e.g., [b, 0, 1, 2, f]
        p << '[';
        llvm::interleaveComma(dims, p, [&](int64_t dim) {
          if (dim >= 0) {
            p << dim;
          } else {
            p << NonSpatialDimToString(static_cast<NonSpatialDim>(dim));
          }
        });
        p << ']';
      };

  print_dim(dnums.input_spatial_dimensions(),
            {{dnums.input_batch_dimension(), IOBatch},
             {dnums.input_feature_dimension(), IOFeature}});
  p << "x";
  print_dim(dnums.kernel_spatial_dimensions(),
            {{dnums.kernel_input_feature_dimension(), KIFeature},
             {dnums.kernel_output_feature_dimension(), KOFeature}});
  p << "->";
  print_dim(dnums.output_spatial_dimensions(),
            {{dnums.output_batch_dimension(), IOBatch},
             {dnums.output_feature_dimension(), IOFeature}});
}

ParseResult parseConvolutionDimensions(OpAsmParser &parser,
                                       ConvDimensionNumbers &dnums) {
  // Parsing a single set of dim numbers gives the spatial dimensions as a
  // single DenseIntElementsAttr and a list of non-spatial dimensions as
  // IntegerAttrs (indexed by the NonSpatialDim enum).
  using parse_dim_result_t = std::pair<
      DenseIntElementsAttr,
      std::unordered_map<NonSpatialDim, IntegerAttr, std::hash<int64_t>>>;

  // Note that the allowed_non_spatial_dims is a set (as opposed to unordered
  // set) because its used to print a list of allowed non spatial dims in the
  // error messages, so making it a set keeps the error messages deterministic.
  auto parse_dims =
      [&](std::set<NonSpatialDim, std::greater<>> allowed_non_spatial_dims,
          parse_dim_result_t &parsed_dims) -> ParseResult {
    // Parse the starting [
    if (parser.parseLSquare()) {
      return failure();
    }
    llvm::SmallVector<int64_t> spatial_dims;
    std::unordered_map<NonSpatialDim, IntegerAttr, std::hash<int64_t>>
        non_spatial_dims;

    int64_t index = 0;
    do {
      int64_t spatial_dim;
      OptionalParseResult parseResult =
          parser.parseOptionalInteger(spatial_dim);
      if (parseResult.hasValue()) {
        if (parseResult.getValue().failed()) {
          return failure();
        }
        // We were successful in parsing an integer. Add its index to the
        // spatial dims.
        spatial_dims.push_back(index);
      } else {
        // We did not parse an integer. We expect a keyword token.
        StringRef keyword;
        if (parser.parseKeyword(&keyword)) {
          return failure();
        }
        if (keyword.size() != 1 || allowed_non_spatial_dims.empty()) {
          return parser.emitError(parser.getCurrentLocation(),
                                  "Unexpected keyword ")
                 << keyword;
        }
        // Check if the keyword matches one of the allowed non-spatial dims.
        // If so, add it to the non_spatial dims and remove it from the
        // allowed set so that it won't be allowed again.
        bool is_allowed = false;
        for (NonSpatialDim allowed : allowed_non_spatial_dims) {
          if (keyword[0] == NonSpatialDimToString(allowed)) {
            non_spatial_dims.insert(
                {allowed, parser.getBuilder().getI64IntegerAttr(index)});
            allowed_non_spatial_dims.erase(allowed);
            is_allowed = true;
            break;
          }
        }

        if (!is_allowed) {
          mlir::InFlightDiagnostic diag = parser.emitError(
              parser.getCurrentLocation(), "Unexpected dimension ");
          diag << keyword << ", expecting ";
          llvm::interleaveComma(
              allowed_non_spatial_dims, diag,
              [&](NonSpatialDim dim) { diag << NonSpatialDimToString(dim); });
          return diag;
        }
      }
      index++;
    } while (parser.parseOptionalComma().succeeded());

    // Make sure all expected non-spatial dimensions are parsed.
    if (!allowed_non_spatial_dims.empty()) {
      mlir::InFlightDiagnostic diag =
          parser.emitError(parser.getCurrentLocation(), "Expected dimensions ");
      llvm::interleaveComma(
          allowed_non_spatial_dims, diag,
          [&](NonSpatialDim dim) { diag << NonSpatialDimToString(dim); });
      diag << " not specified";
      return diag;
    }

    // parse ending ]
    if (parser.parseRSquare()) {
      return failure();
    }

    parsed_dims = std::make_pair(
        parser.getBuilder().getI64TensorAttr(spatial_dims), non_spatial_dims);
    return success();
  };

  parse_dim_result_t parsed_dims;
  if (parse_dims({IOBatch, IOFeature}, parsed_dims)) {
    return failure();
  }
  DenseIntElementsAttr input_spatial_dimensions = parsed_dims.first;
  IntegerAttr input_batch_dimension = parsed_dims.second[IOBatch];
  IntegerAttr input_feature_dimension = parsed_dims.second[IOFeature];
  if (parser.parseKeyword("x")) return failure();
  if (parse_dims({KIFeature, KOFeature}, parsed_dims)) {
    return failure();
  }
  DenseIntElementsAttr kernel_spatial_dimensions = parsed_dims.first;
  IntegerAttr kernel_input_feature_dimension = parsed_dims.second[KIFeature];
  IntegerAttr kernel_output_feature_dimension = parsed_dims.second[KOFeature];
  if (parser.parseArrow()) {
    return failure();
  }
  if (parse_dims({IOBatch, IOFeature}, parsed_dims)) {
    return failure();
  }
  DenseIntElementsAttr output_spatial_dimensions = parsed_dims.first;
  IntegerAttr output_batch_dimension = parsed_dims.second[IOBatch];
  IntegerAttr output_feature_dimension = parsed_dims.second[IOFeature];
  dnums = ConvDimensionNumbers::get(
      input_batch_dimension, input_feature_dimension, input_spatial_dimensions,
      kernel_input_feature_dimension, kernel_output_feature_dimension,
      kernel_spatial_dimensions, output_batch_dimension,
      output_feature_dimension, output_spatial_dimensions,
      parser.getBuilder().getContext());

  return success();
}

}  // namespace mhlo
}  // namespace mlir
