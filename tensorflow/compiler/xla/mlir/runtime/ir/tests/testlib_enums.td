/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TESTLIB_ENUMS
#define XLA_TESTLIB_ENUMS

include "mlir/IR/EnumAttr.td"

def TESTLIB_FOO : I32EnumAttrCase<"Foo", 0>;
def TESTLIB_BAR : I32EnumAttrCase<"Bar", 1>;
def TESTLIB_BAZ : I32EnumAttrCase<"Baz", 2>;

def TESTLIB_EnumType : I32EnumAttr<"EnumType",
    "Custom Call Testlib Enum Type.",
    [TESTLIB_FOO, TESTLIB_BAR, TESTLIB_BAZ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::xla::runtime";
}

def TESTLIB_EnumTypeAttr : EnumAttr<TestlibDialect, TESTLIB_EnumType,
                                    "enum_type"> {
  let assemblyFormat = "`<` $value `>`";
}

// Define another enum type to test enum conversion at the custom call boundary.
def TESTLIB_EnumType2 : I32EnumAttr<"EnumType2",
    "Another Custom Call Testlib Enum Type.",
    [TESTLIB_FOO, TESTLIB_BAR, TESTLIB_BAZ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::xla::runtime";
}

def TESTLIB_EnumType2Attr : EnumAttr<TestlibDialect, TESTLIB_EnumType2,
                                     "another_enum_type"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // XLA_TESTLIB_ENUMS
