/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.
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

#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "stablehlo/integrations/c/StablehloAttributes.h"
#include "stablehlo/integrations/c/StablehloDialect.h"
#include "stablehlo/integrations/c/StablehloTypes.h"

namespace py = pybind11;

namespace {
// Returns a vector containing integers extracted from an attribute using the
// two provided callbacks.
std::vector<int64_t> attributePropertyVector(
    MlirAttribute attr, llvm::function_ref<intptr_t(MlirAttribute)> sizeFn,
    llvm::function_ref<int64_t(MlirAttribute, intptr_t)> getFn) {
  std::vector<int64_t> result;
  intptr_t size = sizeFn(attr);
  result.reserve(size);
  for (intptr_t i = 0; i < size; ++i) {
    result.push_back(getFn(attr, i));
  }
  return result;
}

auto toPyString(MlirStringRef mlirStringRef) {
  return py::str(mlirStringRef.data, mlirStringRef.length);
}

}  // namespace

PYBIND11_MODULE(_stablehlo, m) {
  m.doc() = "stablehlo main python extension";

  //
  // Dialects.
  //

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__stablehlo__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      py::arg("context"), py::arg("load") = true);

  //
  // Types.
  //

  mlir::python::adaptors::mlir_type_subclass(m, "TokenType",
                                             stablehloTypeIsAToken)
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctx) {
            return cls(stablehloTokenTypeGet(ctx));
          },
          py::arg("cls"), py::arg("context") = py::none(),
          "Creates a Token type.");

  //
  // Attributes.
  //

  auto scatteredDimsToOperandDimsFunc = [](MlirAttribute self) {
    return attributePropertyVector(
        self, stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsSize,
        stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsElem);
  };

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "ScatterDimensionNumbers",
      stablehloAttributeIsAScatterDimensionNumbers)
      .def_classmethod(
          "get",
          [](py::object cls, const std::vector<int64_t> &updateWindowDims,
             const std::vector<int64_t> &insertedWindowDims,
             const std::vector<int64_t> &scatteredDimsToOperandDims,
             int64_t indexVectorDim, MlirContext ctx) {
            return cls(stablehloScatterDimensionNumbersGet(
                ctx, updateWindowDims.size(), updateWindowDims.data(),
                insertedWindowDims.size(), insertedWindowDims.data(),
                scatteredDimsToOperandDims.size(),
                scatteredDimsToOperandDims.data(), indexVectorDim));
          },
          py::arg("cls"), py::arg("update_window_dims"),
          py::arg("inserted_window_dims"),
          py::arg("scattered_dims_to_operand_dims"),
          py::arg("index_vector_dim"), py::arg("context") = py::none(),
          "Creates a ScatterDimensionNumbers with the given dimension "
          "configuration.")
      .def_property_readonly(
          "update_window_dims",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self, stablehloScatterDimensionNumbersGetUpdateWindowDimsSize,
                stablehloScatterDimensionNumbersGetUpdateWindowDimsElem);
          })
      .def_property_readonly(
          "inserted_window_dims",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self, stablehloScatterDimensionNumbersGetInsertedWindowDimsSize,
                stablehloScatterDimensionNumbersGetInsertedWindowDimsElem);
          })
      .def_property_readonly("scattered_dims_to_operand_dims",
                             scatteredDimsToOperandDimsFunc)
      .def_property_readonly("index_vector_dim", [](MlirAttribute self) {
        return stablehloDimensionNumbersGetIndexVectorDim(self);
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "GatherDimensionNumbers", stablehloAttributeIsAGatherDimensionNumbers)
      .def_classmethod(
          "get",
          [](py::object cls, const std::vector<int64_t> &offsetDims,
             const std::vector<int64_t> &collapsedSliceDims,
             const std::vector<int64_t> &startIndexMap, int64_t indexVectorDim,
             MlirContext ctx) {
            return cls(stablehloGatherDimensionNumbersGet(
                ctx, offsetDims.size(), offsetDims.data(),
                collapsedSliceDims.size(), collapsedSliceDims.data(),
                startIndexMap.size(), startIndexMap.data(), indexVectorDim));
          },
          py::arg("cls"), py::arg("offset_dims"),
          py::arg("collapsed_slice_dims"), py::arg("start_index_map"),
          py::arg("index_vector_dim"), py::arg("context") = py::none(),
          "Creates a GatherDimensionNumbers attribute with the given dimension "
          "configuration.")
      .def_property_readonly(
          "offset_dims",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self, stablehloGatherDimensionNumbersGetOffsetDimsSize,
                stablehloGatherDimensionNumbersGetOffsetDimsElem);
          })
      .def_property_readonly(
          "collapsed_slice_dims",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self, stablehloGatherDimensionNumbersGetCollapsedSliceDimsSize,
                stablehloGatherDimensionNumbersGetCollapsedSliceDimsElem);
          })
      .def_property_readonly(
          "start_index_map",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self, stablehloGatherDimensionNumbersGetStartIndexMapSize,
                stablehloGatherDimensionNumbersGetStartIndexMapElem);
          })
      .def_property_readonly("index_vector_dim", [](MlirAttribute self) {
        return stablehloGatherDimensionNumbersGetIndexVectorDim(self);
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "DotDimensionNumbers", stablehloAttributeIsADotDimensionNumbers)
      .def_classmethod(
          "get",
          [](py::object cls, const std::vector<int64_t> &lhsBatchingDims,
             const std::vector<int64_t> &rhsBatchingDims,
             const std::vector<int64_t> &lhsContractingDims,
             const std::vector<int64_t> &rhsContractingDims, MlirContext ctx) {
            return cls(stablehloDotDimensionNumbersGet(
                ctx, lhsBatchingDims.size(), lhsBatchingDims.data(),
                rhsBatchingDims.size(), rhsBatchingDims.data(),
                lhsContractingDims.size(), lhsContractingDims.data(),
                rhsContractingDims.size(), rhsContractingDims.data()));
          },
          py::arg("cls"), py::arg("lhs_batching_dimensions"),
          py::arg("rhs_batching_dimensions"),
          py::arg("lhs_contracting_dimensions"),
          py::arg("rhs_contracting_dimensions"),
          py::arg("context") = py::none(),
          "Creates a DotDimensionNumbers attribute with the given dimension "
          "configuration.")
      .def_property_readonly(
          "lhs_batching_dimensions",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self, stablehloDotDimensionNumbersGetLhsBatchingDimensionsSize,
                stablehloDotDimensionNumbersGetLhsBatchingDimensionsElem);
          })
      .def_property_readonly(
          "rhs_batching_dimensions",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self, stablehloDotDimensionNumbersGetRhsBatchingDimensionsSize,
                stablehloDotDimensionNumbersGetRhsBatchingDimensionsElem);
          })
      .def_property_readonly(
          "lhs_contracting_dimensions",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self,
                stablehloDotDimensionNumbersGetLhsContractingDimensionsSize,
                stablehloDotDimensionNumbersGetLhsContractingDimensionsElem);
          })
      .def_property_readonly(
          "rhs_contracting_dimensions", [](MlirAttribute self) {
            return attributePropertyVector(
                self,
                stablehloDotDimensionNumbersGetRhsContractingDimensionsSize,
                stablehloDotDimensionNumbersGetRhsContractingDimensionsElem);
          });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "ConvDimensionNumbers", stablehloAttributeIsAConvDimensionNumbers)
      .def_classmethod(
          "get",
          [](py::object cls, int64_t inputBatchDimension,
             int64_t inputFeatureDimension,
             const std::vector<int64_t> inputSpatialDimensions,
             int64_t kernelInputFeatureDimension,
             int64_t kernelOutputFeatureDimension,
             const std::vector<int64_t> kernelSpatialDimensions,
             int64_t outputBatchDimension, int64_t outputFeatureDimension,
             const std::vector<int64_t> outputSpatialDimensions,
             MlirContext ctx) {
            return cls(stablehloConvDimensionNumbersGet(
                ctx, inputBatchDimension, inputFeatureDimension,
                inputSpatialDimensions.size(), inputSpatialDimensions.data(),
                kernelInputFeatureDimension, kernelOutputFeatureDimension,
                kernelSpatialDimensions.size(), kernelSpatialDimensions.data(),
                outputBatchDimension, outputFeatureDimension,
                outputSpatialDimensions.size(),
                outputSpatialDimensions.data()));
          },
          py::arg("cls"), py::arg("input_batch_dimension"),
          py::arg("input_feature_dimension"),
          py::arg("input_spatial_dimensions"),
          py::arg("kernel_input_feature_dimension"),
          py::arg("kernel_output_feature_dimension"),
          py::arg("kernel_spatial_dimensions"),
          py::arg("output_batch_dimension"),
          py::arg("output_feature_dimension"),
          py::arg("output_spatial_dimensions"), py::arg("ctx") = py::none(),
          "Creates a ConvDimensionNumbers attribute with the given dimension "
          "configuration.")
      .def_property_readonly(
          "input_batch_dimension",
          [](MlirAttribute self) {
            return stablehloConvDimensionNumbersGetInputBatchDimension(self);
          })
      .def_property_readonly(
          "input_feature_dimension",
          [](MlirAttribute self) {
            return stablehloConvDimensionNumbersGetInputFeatureDimension(self);
          })
      .def_property_readonly(
          "input_spatial_dimensions",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self,
                stablehloConvDimensionNumbersGetInputSpatialDimensionsSize,
                stablehloConvDimensionNumbersGetInputSpatialDimensionsElem);
          })
      .def_property_readonly(
          "kernel_input_feature_dimension",
          [](MlirAttribute self) {
            return stablehloConvDimensionNumbersGetKernelInputFeatureDimension(
                self);
          })
      .def_property_readonly(
          "kernel_output_feature_dimension",
          [](MlirAttribute self) {
            return stablehloConvDimensionNumbersGetKernelOutputFeatureDimension(
                self);
          })
      .def_property_readonly(
          "kernel_spatial_dimensions",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self,
                stablehloConvDimensionNumbersGetKernelSpatialDimensionsSize,
                stablehloConvDimensionNumbersGetKernelSpatialDimensionsElem);
          })
      .def_property_readonly(
          "output_batch_dimension",
          [](MlirAttribute self) {
            return stablehloConvDimensionNumbersGetOutputBatchDimension(self);
          })
      .def_property_readonly(
          "output_feature_dimension",
          [](MlirAttribute self) {
            return stablehloConvDimensionNumbersGetOutputFeatureDimension(self);
          })
      .def_property_readonly(
          "output_spatial_dimensions", [](MlirAttribute self) {
            return attributePropertyVector(
                self,
                stablehloConvDimensionNumbersGetOutputSpatialDimensionsSize,
                stablehloConvDimensionNumbersGetOutputSpatialDimensionsElem);
          });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "ComparisonDirectionAttr",
      stablehloAttributeIsAComparisonDirectionAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &direction, MlirContext ctx) {
            return cls(stablehloComparisonDirectionAttrGet(
                ctx, mlirStringRefCreate(direction.c_str(), direction.size())));
          },
          py::arg("cls"), py::arg("comparison_direction"),
          py::arg("context") = py::none(),
          "Creates a ComparisonDirection attribute with the given direction.")
      .def_property_readonly("comparison_direction", [](MlirAttribute self) {
        return toPyString(stablehloComparisonDirectionAttrGetDirection(self));
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "ComparisonTypeAttr", stablehloAttributeIsAComparisonTypeAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &type, MlirContext ctx) {
            return cls(stablehloComparisonTypeAttrGet(
                ctx, mlirStringRefCreate(type.c_str(), type.size())));
          },
          py::arg("cls"), py::arg("comparison_type"),
          py::arg("context") = py::none(),
          "Creates a ComparisonType attribute with the given type.")
      .def_property_readonly("comparison_type", [](MlirAttribute self) {
        return toPyString(stablehloComparisonTypeAttrGetType(self));
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "PrecisionAttr", stablehloAttributeIsAPrecisionAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &type, MlirContext ctx) {
            return cls(stablehloPrecisionAttrGet(
                ctx, mlirStringRefCreate(type.c_str(), type.size())));
          },
          py::arg("cls"), py::arg("precision_type"),
          py::arg("context") = py::none(),
          "Creates a Precision attribute with the given type.")
      .def_property_readonly("precision_type", [](MlirAttribute self) {
        return toPyString(stablehloPrecisionAttrGetPrecision(self));
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "FftTypeAttr", stablehloAttributeIsAFftTypeAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &type, MlirContext ctx) {
            return cls(stablehloFftTypeAttrGet(
                ctx, mlirStringRefCreate(type.c_str(), type.size())));
          },
          py::arg("cls"), py::arg("fft_type"), py::arg("context") = py::none(),
          "Creates a FftType attribute with the given type.")
      .def_property_readonly("fft_type", [](MlirAttribute self) {
        return toPyString(stablehloFftTypeAttrGetFftType(self));
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "TransposeAttr", stablehloAttributeIsATransposeAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &type, MlirContext ctx) {
            return cls(stablehloTransposeAttrGet(
                ctx, mlirStringRefCreate(type.c_str(), type.size())));
          },
          py::arg("cls"), py::arg("transpose_type"),
          py::arg("context") = py::none(),
          "Creates a Transpose attribute with the given type.")
      .def_property_readonly("transpose_type", [](MlirAttribute self) {
        return toPyString(stablehloTransposeAttrGetTranspose(self));
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "RngDistributionAttr", stablehloAttributeIsARngDistributionAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &distribution, MlirContext ctx) {
            return cls(stablehloRngDistributionAttrGet(
                ctx, mlirStringRefCreate(distribution.c_str(),
                                         distribution.size())));
          },
          py::arg("cls"), py::arg("rng_distribution"),
          py::arg("context") = py::none(),
          "Creates a RngDistribution attribute with the given rng "
          "distribution.")
      .def_property_readonly("rng_distribution", [](MlirAttribute self) {
        auto distribution =
            stablehloRngDistributionAttrGetRngDistribution(self);
        return py::str(distribution.data, distribution.length);
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "RngAlgorithmAttr", stablehloAttributeIsARngAlgorithmAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &algorithm, MlirContext ctx) {
            return cls(stablehloRngAlgorithmAttrGet(
                ctx, mlirStringRefCreate(algorithm.c_str(), algorithm.size())));
          },
          py::arg("cls"), py::arg("rng_algorithm"),
          py::arg("context") = py::none(),
          "Creates a RngAlgorithm attribute with the given rng algorithm.")
      .def_property_readonly("rng_algorithm", [](MlirAttribute self) {
        auto algorithm = stablehloRngAlgorithmAttrGetRngAlgorithm(self);
        return py::str(algorithm.data, algorithm.length);
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "ChannelHandle", stablehloAttributeIsChannelHandle)
      .def_classmethod(
          "get",
          [](py::object cls, int64_t handle, int64_t type, MlirContext ctx) {
            return cls(stablehloChannelHandleGet(ctx, handle, type));
          },
          py::arg("cls"), py::arg("handle"), py::arg("type"),
          py::arg("context") = py::none(), "Creates a ChannelHandle attribute.")
      .def_property_readonly("handle",
                             [](MlirAttribute self) {
                               return stablehloChannelHandleGetHandle(self);
                             })
      .def_property_readonly("channel_type", [](MlirAttribute self) {
        return stablehloChannelHandleGetType(self);
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "TypeExtensions", stablehloAttributeIsTypeExtensions)
      .def_classmethod(
          "get",
          [](py::object cls, const std::vector<int64_t> &bounds,
             MlirContext ctx) {
            return cls(
                stablehloTypeExtensionsGet(ctx, bounds.size(), bounds.data()));
          },
          py::arg("cls"), py::arg("bounds"), py::arg("context") = py::none(),
          "Creates a TypeExtensions with the given bounds.")
      .def_property_readonly("bounds", [](MlirAttribute self) {
        return attributePropertyVector(self,
                                       stablehloTypeExtensionsGetBoundsSize,
                                       stablehloTypeExtensionsGetBoundsElem);
      });
}
