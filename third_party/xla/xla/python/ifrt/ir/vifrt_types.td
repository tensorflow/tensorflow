/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_IR_VIFRT_TYPES_TD_
#define XLA_PYTHON_IFRT_IR_VIFRT_TYPES_TD_

include "mlir/IR/AttrTypeBase.td"
include "xla/python/ifrt/ir/vifrt_base.td"
include "xla/python/ifrt/ir/vifrt_dialect.td"

def Vifrt_VersionedTypeInterface : TypeInterface<"VifrtVersionedTypeInterface"> {
  let cppNamespace = "::xla::ifrt";
  let methods = [
    InterfaceMethod<
      "The min version of the VIFRT dialect an attribute is supported in.",
      "::xla::ifrt::Version", "getMinVersion">,
    InterfaceMethod<
      "The max version (inclusive) of the VIFRT dialect an attribute is supported in.",
      "::xla::ifrt::Version", "getMaxVersion">,
  ];
}

class Vifrt_TypeDef<string cpp_name, string name, string min_version, string max_version>
  : TypeDef<Vifrt_Dialect, cpp_name, [Vifrt_VersionedTypeInterface]> {
  let mnemonic = name;
  let extraClassDeclaration = [{
    ::xla::ifrt::Version getMinVersion() {
      return ::xla::ifrt::Version(}] # !subst(".", ", ", min_version) # [{);
    }
    ::xla::ifrt::Version getMaxVersion() {
      }] # !if(
        !eq(max_version, "current"),
        [{ return ::xla::ifrt::Version::getCurrentVersion(); }],
        [{ return ::xla::ifrt::Version("}] # !subst(".", ", ", max_version) # [{"); }]
      ) # [{
    }
  }];
}

def Vifrt_Array : Vifrt_TypeDef<"VifrtArrayV1", "array_v1", "0.1.0", "current"> {
  let parameters = (ins
    "::mlir::Type":$shape,
    "::mlir::Attribute":$sharding_attr,
    "::mlir::Attribute":$devices_attr,
    "::mlir::Attribute":$memory_kind_attr);

  let genVerifyDecl = 1;
  // The verifier does not check `shape` and `memory_kind_attr` because they
  // use builtin types `mlir::RankedTensor` and `mlir::StringAttr`.
  let extraClassDefinition = [{
    ::llvm::LogicalResult VifrtArrayV1Type::verify(
      ::llvm::function_ref<::mlir::InFlightDiagnostic ()> err_fn,
      ::mlir::Type shape,
      ::mlir::Attribute sharding_attr,
      ::mlir::Attribute devices_attr,
      ::mlir::Attribute memory_kind_attr
    ) {
      if (!isFromVifrt(sharding_attr) ||
          !isFromVifrt(devices_attr)) {
        return err_fn() << "expected VIFRT type or attribute";
      }
      return mlir::success();
    }
  }];
  let assemblyFormat = [{
    `<` $shape`,` $sharding_attr `,` $devices_attr `,` `memory_kind` `=`
        $memory_kind_attr `>`
  }];
}

def Vifrt_Control : Vifrt_TypeDef<"VifrtControlV1", "control_v1", "0.1.0", "current">;


#endif  // XLA_PYTHON_IFRT_IR_VIFRT_TYPES_TD_