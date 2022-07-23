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

#include "mlir-c/Registration.h"
#include "mlir-hlo-c/Attributes.h"
#include "mlir-hlo-c/Dialects.h"
#include "mlir-hlo-c/Passes.h"
#include "mlir-hlo-c/Types.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

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

PYBIND11_MODULE(_mlirHlo, m) {
  m.doc() = "mlir-hlo main python extension";

  //
  // Dialects.
  //

  m.def(
      "register_mhlo_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle mhloDialect = mlirGetDialectHandle__mhlo__();
        mlirDialectHandleRegisterDialect(mhloDialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(mhloDialect, context);
        }
      },
      py::arg("context"), py::arg("load") = true);

  m.def(
      "register_chlo_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle chloDialect = mlirGetDialectHandle__chlo__();
        mlirDialectHandleRegisterDialect(chloDialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(chloDialect, context);
        }
      },
      py::arg("context"), py::arg("load") = true);

  //
  // Passes.
  //

  m.def("register_mhlo_passes", []() { mlirRegisterAllMhloPasses(); });

  //
  // Types.
  //

  mlir::python::adaptors::mlir_type_subclass(m, "TokenType",
                                             mlirMhloTypeIsAToken)
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctx) {
            return cls(mlirMhloTokenTypeGet(ctx));
          },
          py::arg("cls"), py::arg("context") = py::none(),
          "Creates a Token type.");

  //
  // Attributes.
  //

  auto scatteredDimsToOperandDimsFunc = [](MlirAttribute self) {
    return attributePropertyVector(
        self, mlirMhloScatterDimensionNumbersGetScatteredDimsToOperandDimsSize,
        mlirMhloScatterDimensionNumbersGetScatteredDimsToOperandDimsElem);
  };

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "ScatterDimensionNumbers", mlirMhloAttributeIsAScatterDimensionNumbers)
      .def_classmethod(
          "get",
          [](py::object cls, const std::vector<int64_t> &updateWindowDims,
             const std::vector<int64_t> &insertedWindowDims,
             const std::vector<int64_t> &scatteredDimsToOperandDims,
             int64_t indexVectorDim, MlirContext ctx) {
            return cls(mlirMhloScatterDimensionNumbersGet(
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
                self, mlirMhloScatterDimensionNumbersGetUpdateWindowDimsSize,
                mlirMhloScatterDimensionNumbersGetUpdateWindowDimsElem);
          })
      .def_property_readonly(
          "inserted_window_dims",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self, mlirMhloScatterDimensionNumbersGetInsertedWindowDimsSize,
                mlirMhloScatterDimensionNumbersGetInsertedWindowDimsElem);
          })
      .def_property_readonly("scattered_dims_to_operand_dims",
                             scatteredDimsToOperandDimsFunc)
      .def_property_readonly("index_vector_dim", [](MlirAttribute self) {
        return mlirMhloDimensionNumbersGetIndexVectorDim(self);
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "GatherDimensionNumbers", mlirMhloAttributeIsAGatherDimensionNumbers)
      .def_classmethod(
          "get",
          [](py::object cls, const std::vector<int64_t> &offsetDims,
             const std::vector<int64_t> &collapsedSliceDims,
             const std::vector<int64_t> &startIndexMap, int64_t indexVectorDim,
             MlirContext ctx) {
            return cls(mlirMhloGatherDimensionNumbersGet(
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
                self, mlirMhloGatherDimensionNumbersGetOffsetDimsSize,
                mlirMhloGatherDimensionNumbersGetOffsetDimsElem);
          })
      .def_property_readonly(
          "collapsed_slice_dims",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self, mlirMhloGatherDimensionNumbersGetCollapsedSliceDimsSize,
                mlirMhloGatherDimensionNumbersGetCollapsedSliceDimsElem);
          })
      .def_property_readonly(
          "start_index_map",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self, mlirMhloGatherDimensionNumbersGetStartIndexMapSize,
                mlirMhloGatherDimensionNumbersGetStartIndexMapElem);
          })
      .def_property_readonly("index_vector_dim", [](MlirAttribute self) {
        return mlirMhloGatherDimensionNumbersGetIndexVectorDim(self);
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "DotDimensionNumbers", mlirMhloAttributeIsADotDimensionNumbers)
      .def_classmethod(
          "get",
          [](py::object cls, const std::vector<int64_t> &lhsBatchingDims,
             const std::vector<int64_t> &rhsBatchingDims,
             const std::vector<int64_t> &lhsContractingDims,
             const std::vector<int64_t> &rhsContractingDims, MlirContext ctx) {
            return cls(mlirMhloDotDimensionNumbersGet(
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
                self, mlirMhloDotDimensionNumbersGetLhsBatchingDimensionsSize,
                mlirMhloDotDimensionNumbersGetLhsBatchingDimensionsElem);
          })
      .def_property_readonly(
          "rhs_batching_dimensions",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self, mlirMhloDotDimensionNumbersGetRhsBatchingDimensionsSize,
                mlirMhloDotDimensionNumbersGetRhsBatchingDimensionsElem);
          })
      .def_property_readonly(
          "lhs_contracting_dimensions",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self,
                mlirMhloDotDimensionNumbersGetLhsContractingDimensionsSize,
                mlirMhloDotDimensionNumbersGetLhsContractingDimensionsElem);
          })
      .def_property_readonly(
          "rhs_contracting_dimensions", [](MlirAttribute self) {
            return attributePropertyVector(
                self,
                mlirMhloDotDimensionNumbersGetRhsContractingDimensionsSize,
                mlirMhloDotDimensionNumbersGetRhsContractingDimensionsElem);
          });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "ConvDimensionNumbers", mlirMhloAttributeIsAConvDimensionNumbers)
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
            return cls(mlirMhloConvDimensionNumbersGet(
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
            return mlirMhloConvDimensionNumbersGetInputBatchDimension(self);
          })
      .def_property_readonly(
          "input_feature_dimension",
          [](MlirAttribute self) {
            return mlirMhloConvDimensionNumbersGetInputFeatureDimension(self);
          })
      .def_property_readonly(
          "input_spatial_dimensions",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self, mlirMhloConvDimensionNumbersGetInputSpatialDimensionsSize,
                mlirMhloConvDimensionNumbersGetInputSpatialDimensionsElem);
          })
      .def_property_readonly(
          "kernel_input_feature_dimension",
          [](MlirAttribute self) {
            return mlirMhloConvDimensionNumbersGetKernelInputFeatureDimension(
                self);
          })
      .def_property_readonly(
          "kernel_output_feature_dimension",
          [](MlirAttribute self) {
            return mlirMhloConvDimensionNumbersGetKernelOutputFeatureDimension(
                self);
          })
      .def_property_readonly(
          "kernel_spatial_dimensions",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self,
                mlirMhloConvDimensionNumbersGetKernelSpatialDimensionsSize,
                mlirMhloConvDimensionNumbersGetKernelSpatialDimensionsElem);
          })
      .def_property_readonly(
          "output_batch_dimension",
          [](MlirAttribute self) {
            return mlirMhloConvDimensionNumbersGetOutputBatchDimension(self);
          })
      .def_property_readonly(
          "output_feature_dimension",
          [](MlirAttribute self) {
            return mlirMhloConvDimensionNumbersGetOutputFeatureDimension(self);
          })
      .def_property_readonly(
          "output_spatial_dimensions", [](MlirAttribute self) {
            return attributePropertyVector(
                self,
                mlirMhloConvDimensionNumbersGetOutputSpatialDimensionsSize,
                mlirMhloConvDimensionNumbersGetOutputSpatialDimensionsElem);
          });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "ComparisonDirectionAttr", mlirMhloAttributeIsAComparisonDirectionAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &direction, MlirContext ctx) {
            return cls(mlirMhloComparisonDirectionAttrGet(
                ctx, mlirStringRefCreate(direction.c_str(), direction.size())));
          },
          py::arg("cls"), py::arg("comparison_direction"),
          py::arg("context") = py::none(),
          "Creates a ComparisonDirection attribute with the given direction.")
      .def_property_readonly("comparison_direction", [](MlirAttribute self) {
        return toPyString(mlirMhloComparisonDirectionAttrGetDirection(self));
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "ComparisonTypeAttr", mlirMhloAttributeIsAComparisonTypeAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &type, MlirContext ctx) {
            return cls(mlirMhloComparisonTypeAttrGet(
                ctx, mlirStringRefCreate(type.c_str(), type.size())));
          },
          py::arg("cls"), py::arg("comparison_type"),
          py::arg("context") = py::none(),
          "Creates a ComparisonType attribute with the given type.")
      .def_property_readonly("comparison_type", [](MlirAttribute self) {
        return toPyString(mlirMhloComparisonTypeAttrGetType(self));
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "PrecisionAttr", mlirMhloAttributeIsAPrecisionAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &type, MlirContext ctx) {
            return cls(mlirMhloPrecisionAttrGet(
                ctx, mlirStringRefCreate(type.c_str(), type.size())));
          },
          py::arg("cls"), py::arg("precision_type"),
          py::arg("context") = py::none(),
          "Creates a Precision attribute with the given type.")
      .def_property_readonly("precision_type", [](MlirAttribute self) {
        return toPyString(mlirMhloPrecisionAttrGetPrecision(self));
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "FftTypeAttr", mlirMhloAttributeIsAFftTypeAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &type, MlirContext ctx) {
            return cls(mlirMhloFftTypeAttrGet(
                ctx, mlirStringRefCreate(type.c_str(), type.size())));
          },
          py::arg("cls"), py::arg("fft_type"), py::arg("context") = py::none(),
          "Creates a FftType attribute with the given type.")
      .def_property_readonly("fft_type", [](MlirAttribute self) {
        return toPyString(mlirMhloFftTypeAttrGetFftType(self));
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "DequantizeModeAttr", mlirMhloAttributeIsADequantizeModeAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &type, MlirContext ctx) {
            return cls(mlirMhloDequantizeModeAttrGet(
                ctx, mlirStringRefCreate(type.c_str(), type.size())));
          },
          py::arg("cls"), py::arg("dequantize_mode"),
          py::arg("context") = py::none(),
          "Creates a DequantizeMode attribute with the given mode.")
      .def_property_readonly("dequantize_mode", [](MlirAttribute self) {
        return toPyString(mlirMhloDequantizeModeAttrGetDequantizeMode(self));
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "TransposeAttr", mlirMhloAttributeIsATransposeAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &type, MlirContext ctx) {
            return cls(mlirMhloTransposeAttrGet(
                ctx, mlirStringRefCreate(type.c_str(), type.size())));
          },
          py::arg("cls"), py::arg("transpose_type"),
          py::arg("context") = py::none(),
          "Creates a Transpose attribute with the given type.")
      .def_property_readonly("transpose_type", [](MlirAttribute self) {
        return toPyString(mlirMhloTransposeAttrGetTranspose(self));
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "FusionKindAttr", mlirMhloAttributeIsAFusionKindAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &kind, MlirContext ctx) {
            return cls(mlirMhloFusionKindAttrGet(
                ctx, mlirStringRefCreate(kind.c_str(), kind.size())));
          },
          py::arg("cls"), py::arg("fusion_kind"),
          py::arg("context") = py::none(),
          "Creates a FusionKind attribute with the given kind.")
      .def_property_readonly("fusion_kind", [](MlirAttribute self) {
        return toPyString(mlirMhloFusionKindAttrGetFusionKind(self));
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "RngDistributionAttr", mlirMhloAttributeIsARngDistributionAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &distribution, MlirContext ctx) {
            return cls(mlirMhloRngDistributionAttrGet(
                ctx, mlirStringRefCreate(distribution.c_str(),
                                         distribution.size())));
          },
          py::arg("cls"), py::arg("rng_distribution"),
          py::arg("context") = py::none(),
          "Creates a RngDistribution attribute with the given rng "
          "distribution.")
      .def_property_readonly("rng_distribution", [](MlirAttribute self) {
        auto distribution = mlirMhloRngDistributionAttrGetRngDistribution(self);
        return py::str(distribution.data, distribution.length);
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "RngAlgorithmAttr", mlirMhloAttributeIsARngAlgorithmAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &algorithm, MlirContext ctx) {
            return cls(mlirMhloRngAlgorithmAttrGet(
                ctx, mlirStringRefCreate(algorithm.c_str(), algorithm.size())));
          },
          py::arg("cls"), py::arg("rng_algorithm"),
          py::arg("context") = py::none(),
          "Creates a RngAlgorithm attribute with the given rng algorithm.")
      .def_property_readonly("rng_algorithm", [](MlirAttribute self) {
        auto algorithm = mlirMhloRngAlgorithmAttrGetRngAlgorithm(self);
        return py::str(algorithm.data, algorithm.length);
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "ChannelHandle", mlirMhloAttributeIsChannelHandle)
      .def_classmethod(
          "get",
          [](py::object cls, int64_t handle, int64_t type, MlirContext ctx) {
            return cls(mlirMhloChannelHandleGet(ctx, handle, type));
          },
          py::arg("cls"), py::arg("handle"), py::arg("type"),
          py::arg("context") = py::none(), "Creates a ChannelHandle attribute.")
      .def_property_readonly("handle",
                             [](MlirAttribute self) {
                               return mlirMhloChannelHandleGetHandle(self);
                             })
      .def_property_readonly("channel_type", [](MlirAttribute self) {
        return mlirMhloChannelHandleGetType(self);
      });
}
