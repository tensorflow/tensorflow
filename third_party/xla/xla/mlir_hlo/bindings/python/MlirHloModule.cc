/* Copyright 2021 The OpenXLA Authors.
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

#include <string>
#include <vector>

#include "bindings/c/Attributes.h"
#include "bindings/c/Dialects.h"
#include "bindings/c/Passes.h"
#include "bindings/c/Types.h"
#include "mlir-c/IR.h"
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
             const std::vector<int64_t> &inputBatchingDims,
             const std::vector<int64_t> &scatterIndicesBatchingDims,
             const std::vector<int64_t> &scatteredDimsToOperandDims,
             int64_t indexVectorDim, MlirContext ctx) {
            return cls(mlirMhloScatterDimensionNumbersGet(
                ctx, updateWindowDims.size(), updateWindowDims.data(),
                insertedWindowDims.size(), insertedWindowDims.data(),
                inputBatchingDims.size(), inputBatchingDims.data(),
                scatterIndicesBatchingDims.size(),
                scatterIndicesBatchingDims.data(),
                scatteredDimsToOperandDims.size(),
                scatteredDimsToOperandDims.data(), indexVectorDim));
          },
          py::arg("cls"), py::arg("update_window_dims"),
          py::arg("inserted_window_dims"), py::arg("input_batching_dims"),
          py::arg("scatter_indices_batching_dims"),
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
      .def_property_readonly(
          "input_batching_dims",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self, mlirMhloScatterDimensionNumbersGetInputBatchingDimsSize,
                mlirMhloScatterDimensionNumbersGetInputBatchingDimsElem);
          })
      .def_property_readonly(
          "scatter_indices_batching_dims",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self,
                mlirMhloScatterDimensionNumbersGetScatterIndicesBatchingDimsSize,  // NOLINT(whitespace/line_length)
                mlirMhloScatterDimensionNumbersGetScatterIndicesBatchingDimsElem  // NOLINT(whitespace/line_length)
            );
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
             const std::vector<int64_t> &operandBatchingDims,
             const std::vector<int64_t> &startIndicesBatchingDims,
             const std::vector<int64_t> &startIndexMap, int64_t indexVectorDim,
             MlirContext ctx) {
            return cls(mlirMhloGatherDimensionNumbersGet(
                ctx, offsetDims.size(), offsetDims.data(),
                collapsedSliceDims.size(), collapsedSliceDims.data(),
                operandBatchingDims.size(), operandBatchingDims.data(),
                startIndicesBatchingDims.size(),
                startIndicesBatchingDims.data(), startIndexMap.size(),
                startIndexMap.data(), indexVectorDim));
          },
          py::arg("cls"), py::arg("offset_dims"),
          py::arg("collapsed_slice_dims"), py::arg("operand_batching_dims"),
          py::arg("start_indices_batching_dims"), py::arg("start_index_map"),
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
          "operand_batching_dims",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self, mlirMhloGatherDimensionNumbersGetOperandBatchingDimsSize,
                mlirMhloGatherDimensionNumbersGetOperandBatchingDimsElem);
          })
      .def_property_readonly(
          "start_indices_batching_dims",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self,
                mlirMhloGatherDimensionNumbersGetStartIndicesBatchingDimsSize,
                mlirMhloGatherDimensionNumbersGetStartIndicesBatchingDimsElem);
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
      m, "OutputOperandAlias", mlirMhloAttributeIsAOutputOperandAlias)
      .def_classmethod(
          "get",
          [](py::object cls, const std::vector<int64_t> outputTupleIndices,
             int64_t operandIndex,
             const std::vector<int64_t> operandTupleIndices, MlirContext ctx) {
            return cls(mlirMhloOutputOperandAliasGet(
                ctx, outputTupleIndices.size(), outputTupleIndices.data(),
                operandIndex, operandTupleIndices.size(),
                operandTupleIndices.data()));
          },
          py::arg("cls"), py::arg("output_tuple_indices"),
          py::arg("operand_index"), py::arg("operand_tuple_indices"),
          py::arg("ctx") = py::none(),
          "Creates a OutputOperandAlias attribute with the given tuple index.")
      .def_property_readonly(
          "output_tuple_indices",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self, mlirMhloOutputOperandAliasGetOutputTupleIndicesSize,
                mlirMhloOutputOperandAliasGetOutputTupleIndicesElem);
          })
      .def_property_readonly(
          "operand_index",
          [](MlirAttribute self) {
            return mlirMhloOutputOperandAliasGetOperandIndex(self);
          })
      .def_property_readonly("operand_tuple_indices", [](MlirAttribute self) {
        return attributePropertyVector(
            self, mlirMhloOutputOperandAliasGetOperandTupleIndicesSize,
            mlirMhloOutputOperandAliasGetOperandTupleIndicesElem);
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "ComparisonDirectionAttr", mlirMhloAttributeIsAComparisonDirectionAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &value, MlirContext ctx) {
            return cls(mlirMhloComparisonDirectionAttrGet(
                ctx, mlirStringRefCreate(value.c_str(), value.size())));
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none(),
          "Creates a ComparisonDirection attribute with the given value.")
      .def_property_readonly("value", [](MlirAttribute self) {
        return toPyString(mlirMhloComparisonDirectionAttrGetValue(self));
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "ComparisonTypeAttr", mlirMhloAttributeIsAComparisonTypeAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &value, MlirContext ctx) {
            return cls(mlirMhloComparisonTypeAttrGet(
                ctx, mlirStringRefCreate(value.c_str(), value.size())));
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none(),
          "Creates a ComparisonType attribute with the given value.")
      .def_property_readonly("value", [](MlirAttribute self) {
        return toPyString(mlirMhloComparisonTypeAttrGetValue(self));
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "PrecisionAttr", mlirMhloAttributeIsAPrecisionAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &value, MlirContext ctx) {
            return cls(mlirMhloPrecisionAttrGet(
                ctx, mlirStringRefCreate(value.c_str(), value.size())));
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none(),
          "Creates a Precision attribute with the given value.")
      .def_property_readonly("value", [](MlirAttribute self) {
        return toPyString(mlirMhloPrecisionAttrGetValue(self));
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "FftTypeAttr", mlirMhloAttributeIsAFftTypeAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &value, MlirContext ctx) {
            return cls(mlirMhloFftTypeAttrGet(
                ctx, mlirStringRefCreate(value.c_str(), value.size())));
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none(),
          "Creates a FftType attribute with the given value.")
      .def_property_readonly("value", [](MlirAttribute self) {
        return toPyString(mlirMhloFftTypeAttrGetValue(self));
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "DequantizeModeAttr", mlirMhloAttributeIsADequantizeModeAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &value, MlirContext ctx) {
            return cls(mlirMhloDequantizeModeAttrGet(
                ctx, mlirStringRefCreate(value.c_str(), value.size())));
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none(),
          "Creates a DequantizeMode attribute with the given value.")
      .def_property_readonly("value", [](MlirAttribute self) {
        return toPyString(mlirMhloDequantizeModeAttrGetValue(self));
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "TransposeAttr", mlirMhloAttributeIsATransposeAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &value, MlirContext ctx) {
            return cls(mlirMhloTransposeAttrGet(
                ctx, mlirStringRefCreate(value.c_str(), value.size())));
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none(),
          "Creates a Transpose attribute with the given value.")
      .def_property_readonly("value", [](MlirAttribute self) {
        return toPyString(mlirMhloTransposeAttrGetValue(self));
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "FusionKindAttr", mlirMhloAttributeIsAFusionKindAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &value, MlirContext ctx) {
            return cls(mlirMhloFusionKindAttrGet(
                ctx, mlirStringRefCreate(value.c_str(), value.size())));
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none(),
          "Creates a FusionKind attribute with the given value.")
      .def_property_readonly("value", [](MlirAttribute self) {
        return toPyString(mlirMhloFusionKindAttrGetValue(self));
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "RngDistributionAttr", mlirMhloAttributeIsARngDistributionAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &value, MlirContext ctx) {
            return cls(mlirMhloRngDistributionAttrGet(
                ctx, mlirStringRefCreate(value.c_str(), value.size())));
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none(),
          "Creates a RngDistribution attribute with the given value.")
      .def_property_readonly("value", [](MlirAttribute self) {
        auto value = mlirMhloRngDistributionAttrGetValue(self);
        return py::str(value.data, value.length);
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "RngAlgorithmAttr", mlirMhloAttributeIsARngAlgorithmAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &value, MlirContext ctx) {
            return cls(mlirMhloRngAlgorithmAttrGet(
                ctx, mlirStringRefCreate(value.c_str(), value.size())));
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none(),
          "Creates a RngAlgorithm attribute with the given value.")
      .def_property_readonly("value", [](MlirAttribute self) {
        auto value = mlirMhloRngAlgorithmAttrGetValue(self);
        return py::str(value.data, value.length);
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

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "TypeExtensions", mlirMhloAttributeIsTypeExtensions)
      .def_classmethod(
          "get",
          [](py::object cls, const std::vector<int64_t> &bounds,
             MlirContext ctx) {
            return cls(
                mlirMhloTypeExtensionsGet(ctx, bounds.size(), bounds.data()));
          },
          py::arg("cls"), py::arg("bounds"), py::arg("context") = py::none(),
          "Creates a TypeExtensions with the given bounds.")
      .def_property_readonly("bounds", [](MlirAttribute self) {
        return attributePropertyVector(self,
                                       mlirMhloTypeExtensionsGetBoundsSize,
                                       mlirMhloTypeExtensionsGetBoundsElem);
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "SparsityDescriptor", mlirMhloAttributeIsASparsityDescriptor)
      .def_classmethod(
          "get",
          [](py::object cls, const int64_t dimension, const int64_t n,
             const int64_t m, MlirContext ctx) {
            return cls(mlirMhloSparsityDescriptorGet(ctx, dimension, n, m));
          },
          py::arg("cls"), py::arg("dimension"), py::arg("n"), py::arg("m"),
          py::arg("context") = py::none(),
          "Creates a SparseDescriptor attribute with the given sparsity "
          "configurations.")
      .def_property_readonly(
          "dimension",
          [](MlirAttribute self) {
            return mlirMhloSparsityDescriptorGetDimension(self);
          })
      .def_property_readonly("n",
                             [](MlirAttribute self) {
                               return mlirMhloSparsityDescriptorGetN(self);
                             })
      .def_property_readonly("m", [](MlirAttribute self) {
        return mlirMhloSparsityDescriptorGetM(self);
      });
}
