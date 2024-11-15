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

#ifndef XLA_PYTHON_IFRT_IR_VIFRT_DIALECT_TD_
#define XLA_PYTHON_IFRT_IR_VIFRT_DIALECT_TD_

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/DialectBase.td"
include "xla/python/ifrt/ir/vifrt_interfaces.td"

//===---------------------------------------------------------------------------
// Dialect
//===---------------------------------------------------------------------------

def Vifrt_Dialect : Dialect {
  let name = "vifrt";
  let summary = "Versioned IFRT dialect";
  let cppNamespace = "::xla::ifrt";

  let description = [{
    A versioned copy of the IFRT IR dialect that is used for forward and
    backward compatible serialization/deserialization.

    Version log:
      0.1.0: Initial IFRT IR stability guarantees.
  }];

  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
  let usePropertiesForAttributes = 1;
}

//===---------------------------------------------------------------------------
// Attributes
//===---------------------------------------------------------------------------

class Vifrt_AttrDef<string name, string min_version, string max_version>
  : AttrDef<Vifrt_Dialect, name, [Vifrt_VersionedAttrInterface]> {
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

def Vifrt_DevicesAttrV1 : Vifrt_AttrDef<"VifrtDevicesV1", "0.1.0", "current"> {
  let mnemonic = "devices_v1";
  let parameters = (ins ArrayRefParameter<"int">:$ids);
  let assemblyFormat = "`[` $ids `]`";
}

def Vifrt_UnspecifiedShardingAttrV1
    : Vifrt_AttrDef<"VifrtUnspecifiedShardingV1", "0.1.0", "current"> {
  let mnemonic = "sharding_unspecified_v1";
  let parameters = (ins);
  let assemblyFormat = "";
}

def Vifrt_ShardingParameterV1 :
    AttrOrTypeParameter<"::xla::ifrt::ShardingParam", ""> {
  let parser = "::xla::ifrt::ShardingParam::ParseV1($_parser)";
  let printer = "::xla::ifrt::ShardingParam::PrintV1($_printer, $_self)";
}

def Vifrt_ShardingParamAttrV1 : Vifrt_AttrDef<"VifrtShardingParamV1", "0.1.0",
    "current"> {
  let mnemonic = "sharding_param_v1";
  let parameters = (ins Vifrt_ShardingParameterV1:$sharding);
  let assemblyFormat = "`<` $sharding `>`";
}

def Vifrt_IntervalAttrV1 : Vifrt_AttrDef<"VifrtIntervalV1", "0.1.0",
    "current"> {
  let mnemonic = "interval_v1";
  let parameters = (ins "int":$start, "int":$end, "int":$step);
  let assemblyFormat = "`[`$start `:` $end `:` $step`]`";
}

def Vifrt_MappingAttrV1 : Vifrt_AttrDef<"VifrtMappingV1", "0.1.0", "current"> {
  let mnemonic = "mapping_v1";
  let parameters = (ins
     Vifrt_IntervalAttrV1:$from_shards,
     Vifrt_IntervalAttrV1:$to_shards);
  let assemblyFormat = "`<` $from_shards `to` $to_shards `>`";
}

// Equivalent to `mlir::ArrayAttr`, but with VIFRT verification.
// This can be used to represent `Ifrt_MappingAttrArrayAttr`,
// `Ifrt_ArrayMappingAttrArrayAttr` and `Ifrt_IoAliasesAttr`, which are just
// arrays.
def Vifrt_GenericArrayAttrV1 : Vifrt_AttrDef<"VifrtGenericArrayAttrV1",
      "0.1.0", "current"> {
  let mnemonic = "generic_array_attr_v1";
  let parameters = (ins ArrayRefParameter<"mlir::Attribute">:$value);
  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    mlir::LogicalResult VifrtGenericArrayAttrV1Attr::verify(
        llvm::function_ref<mlir::InFlightDiagnostic ()> err_fn,
        llvm::ArrayRef<mlir::Attribute> value) {
      if (!allFromVifrt(value)) return err_fn() << "expected array of VIFRT attributes";
      return mlir::success();
    }
  }];
  let assemblyFormat = "`<` custom<AttributeArray>($value) `>`";
}

def Vifrt_ArrayMappingAttrV1 : Vifrt_AttrDef<"VifrtArrayMappingV1", "0.1.0",
    "current"> {
  let mnemonic = "array_mapping_v1";
  let parameters = (ins
    "int32_t":$in_array_index,
    "int32_t":$out_array_index,
    Vifrt_GenericArrayAttrV1:$mappings);
  let assemblyFormat = "`<`$in_array_index`,` $out_array_index`,` $mappings`>`";
}

//===---------------------------------------------------------------------------
// Types
//===---------------------------------------------------------------------------

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
    "::mlir::RankedTensorType":$shape,
    "::mlir::Attribute":$sharding_attr,
    Vifrt_DevicesAttrV1:$devices_attr,
    "::mlir::StringAttr":$memory_kind_attr);

  let assemblyFormat = [{
    `<` $shape`,` $sharding_attr `,` $devices_attr `,` `memory_kind` `=`
        $memory_kind_attr `>`
  }];
}

def Vifrt_Control : Vifrt_TypeDef<"VifrtControlV1", "control_v1", "0.1.0", "current">;

#endif  // XLA_PYTHON_IFRT_IR_VIFRT_DIALECT_TD_