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

#ifndef XLA_PYTHON_IFRT_IR_VIFRT_ATTRS_TD_
#define XLA_PYTHON_IFRT_IR_VIFRT_ATTRS_TD_

include "mlir/IR/AttrTypeBase.td"
include "xla/python/ifrt/ir/vifrt_base.td"
include "xla/python/ifrt/ir/vifrt_dialect.td"
include "xla/python/ifrt/ir/vifrt_types.td"

def Vifrt_VersionedAttrInterface : AttrInterface<"VifrtVersionedAttrInterface"> {
  let cppNamespace = "::xla::ifrt";
  let methods = [
    InterfaceMethod<
      "The min version of the VIFRT dialect an attribute is supported in.",
      "::xla::ifrt::Version", "getMinVersion">,
    InterfaceMethod<
      "The maxi version (inclusive) of the VIFRT dialect an attribute is supported in.",
      "::xla::ifrt::Version", "getMaxVersion">,
  ];
}

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

// TODO(icgog): Introduce versioned ShardingParamV1.
def Vifrt_ShardingParamAttrV1 : Vifrt_AttrDef<"VifrtShardingParamV1", "0.1.0",
    "current"> {
  let mnemonic = "sharding_param_v1";
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

def Vifrt_IoAliasesAttrV1 : Vifrt_AttrDef<"IfrtIoAliasesV1", "0.1.0", "current"> {
  let mnemonic = "io_aliases_v1";
  let parameters = (ins
    Vifrt_GenericArrayAttrV1:$io_aliases);
  let assemblyFormat = "`<` $io_aliases `>`";
}

// TODO(icgog): Introduce Vifrt_MappingAttrArrayAttrV1,
// Vifrt_ArrayMappingAttrArrayAttrV1.

#endif  // XLA_PYTHON_IFRT_IR_VIFRT_ATTRS_TD_