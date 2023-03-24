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

// Declare passes used in xla_legalize_tf.

include "mlir/Pass/PassBase.td"

def LegalizeTF : Pass<"xla-legalize-tf", "mlir::func::FuncOp"> {
  let summary = "Legalize from TF dialect's or HLO dialect's control flow.";

  let description = [{
    Legalizes from TF dialect to HLO dialect. When allow_partial_conversion is
    false, emits an error if there is any operation that can't be legalized.
    When `tf2xla_fallback_device_type` is not `None`, also uses legalization
    patterns from TF2XLA fallback for provided device type (see
    legalize_tf_with_tf2xla.cc for details). By default, TF2XLA fallback is not
    used.
  }];
  let options = [
    Option<"allow_partial_conversion_", "allow-partial-conversion", "bool",
        /*default=*/"false",
        "Allow operations that can't be legalized.">,
    Option<"legalize_chlo_", "legalize-chlo", "bool", /*default=*/"true",
        "Legalizes intermediate chlo ops to hlo">,
    Option<"use_tf2xla_fallback_", "use-tf2xla-fallback", "bool",
        /*default=*/"false",
        "Use TF2XLA fallback for legalization">,
    Option<"device_type_", "device-type", "std::string",
        /*default=*/"\"INVALID_DEVICE_TYPE\"",
        "The device type used by TF2XLA fallback. Must be specified if "
        "use-tf2xla-fallback is true, otherwise not used">,
    Option<"prefer_tf2xla_", "prefer-tf2xla", "bool",
        /*default=*/"false",
        "Prioritize tf2xla fallback legalization over MLIR legalization "
        "patterns">,
    Option<"use_tf2xla_hlo_importer_", "use-tf2xla-hlo-importer",
       "bool", /*default=*/"false",
        "Use the experimental HLO to MHLO importer for per-op fallback calls "
        " from MLIR bridge to TF2XLA."
        "Users should not set this flag and ideally this goes away.">
  ];

  let constructor = "mlir::mhlo::createLegalizeTFPass()";
  let dependentDialects = ["arith::ArithDialect, chlo::ChloDialect",
                           "mhlo::MhloDialect",
                           "quant::QuantizationDialect",
                           "shape::ShapeDialect",
                           "func::FuncDialect",
                           "sparse_tensor::SparseTensorDialect"];
}

def LegalizeTFModulePass : Pass<"xla-fallback-legalize-tf-module-pass", "ModuleOp"> {
  let summary = "Legalize whitelisted Ops using TF2XLA fallback for ops that "
      "must also be able to create new functions.";

  let description = [{
    Legalizes whitelisted Ops from TF dialect to HLO dialect using TF2XLA
    fallback for ops that must be allowed to create new functions.
  }];
  let options = [
    Option<"device_type_", "device-type", "std::string",
        /*default=*/"\"INVALID_DEVICE_TYPE\"",
        "The device type used by TF2XLA fallback. Required.">,
  ];

  let constructor = "mlir::mhlo::createLegalizeTFModulePass()";
  let dependentDialects = ["arith::ArithDialect, chlo::ChloDialect",
                           "mhlo::MhloDialect",
                           "shape::ShapeDialect", "func::FuncDialect", "sparse_tensor::SparseTensorDialect"];
}

def ConvertMHLOQuantToInt : Pass<"convert-mhlo-quant-to-int", "mlir::func::FuncOp"> {
  let summary = "Convert from MHLO quantized ops to MHLO primitive ops.";

  let description = [{
    Convert from MHLO quantized ops with MHLO quant types to MHLO primitive ops
    like int ops.
  }];

  let constructor = "mlir::mhlo::createConvertMHLOQuantToIntPass()";
  let dependentDialects = ["chlo::ChloDialect", "mhlo::MhloDialect"];
}

def LegalizeTFNoFallback : Pass<"xla-legalize-tf-no-fallback", "mlir::func::FuncOp"> {
  let summary = "Legalize from TF dialect's or HLO dialect's control flow.";

  let description = [{
    Legalizes from TF dialect to HLO dialect. When allow_partial_conversion is
    false, emits an error if there is any operation that can't be legalized.
  }];
  let options = [
    Option<"allow_partial_conversion_", "allow-partial-conversion", "bool",
        /*default=*/"false",
        "Allow operations that can't be legalized.">
  ];

  let constructor = "mlir::mhlo::createLegalizeTFNoFallbackPass()";
  let dependentDialects = ["arith::ArithDialect", "chlo::ChloDialect",
                           "mhlo::MhloDialect",
                           "shape::ShapeDialect", "func::FuncDialect", "sparse_tensor::SparseTensorDialect"];
}

def LegalizeTfTypesPass : Pass<"xla-legalize-tf-types"> {
  let summary = "Replace TensorFlow types with types that are legal in the MHLO dialect";

  let description = [{
    The TF dialect uses some TF types that are illegal in the MHLO dialect and
    some generic types that are legal in MHLO. This pass legalizes TF types into
    types that are legal in MHLO. Rewrites here should run before TF to MHLO op
    legalizations are run.

    Specifically, this pass replaces each quantized integer type with the
    corresponding ordinary types. For example, `TF::Qint8Type` is replaced with
    `i8` everywhere it occurs. Types that are replaced are `TF::Qint8Type`,
    `TF::Qint16Type`, `TF::Qint32Type`, `TF::Quint8Type`, and `TF::Quint16Type`.
  }];

  let constructor = "::mlir::mhlo::CreateLegalizeTfTypesPass()";
}

def LegalizeTFCollective : Pass<"xla-legalize-tf-collective", "ModuleOp"> {
  let summary = "Legalize TF/XLA collective ops (TensorFlow dialect) to the HLO dialect";

  let constructor = "mlir::mhlo::CreateLegalizeTFCollectivePass()";
  let dependentDialects = ["mhlo::MhloDialect", "sparse_tensor::SparseTensorDialect"];
}

def VerifyTFXLALegalization : Pass<"tfxla-verify-legalization", "mlir::func::FuncOp"> {
  let summary = "Verifies that all TF ops have been legalized to XLA.";

  let description = [{"Ensures that all Tensorflow ops have been legalized to "
                    "XLA and reports an error about which op has not been"
                    "legalized. This pass does not transform any ops and is just"
                    " a verification pass to ensure invariants are true."}];

  let options = [
    Option<"legalize_chlo_", "legalize-chlo", "bool", /*default=*/"true",
        "Legalizes intermediate chlo ops to hlo">
    ];

  let constructor = "mlir::mhlo::CreateVerifyTFXLALegalizationPass()";
}

def TFXLADeviceSpecificTransforms : Pass<"tfxla-device-specific-transforms",
                                            "mlir::func::FuncOp"> {
  let summary = "Transforms ops that require device context into device independent TF Ops.";

  let description = [{"Transforms device specific ops into device independent"
                    "ops."}];

  let options = [
    Option<"device_type_", "device-type", "std::string",
        /*default=*/"\"INVALID_DEVICE_TYPE\"",
        "The device type being targeted.">,
  ];

  let constructor = "mlir::mhlo::CreateTFXLADeviceSpecificTransformsPass()";
}

def InfeedOpsXlaAdjustLayout : Pass<"infeed-ops-xla-adjust-layout",
                                    "mlir::func::FuncOp"> {
  let summary = "Adjusts Infeed ops layout for XLA.";

  let description = [{"Adjust layouts so infeed send & receive use the same "
                    "format."}];

  let constructor = "mlir::mhlo::CreateInfeedsOpsXlaAdjustLayoutPass()";

  let dependentDialects = ["mhlo::MhloDialect"];

}