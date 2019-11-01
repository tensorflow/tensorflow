//===-- enums.td - EnumsGen test definition file -----------*- tablegen -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

include "mlir/IR/OpBase.td"

def CaseA: StrEnumAttrCase<"CaseA">;
def CaseB: StrEnumAttrCase<"CaseB", 10>;

def StrEnum: StrEnumAttr<"StrEnum", "A test enum", [CaseA, CaseB]> {
  let cppNamespace = "Outer::Inner";
  let stringToSymbolFnName = "ConvertToEnum";
  let symbolToStringFnName = "ConvertToString";
}

def Case5: I32EnumAttrCase<"Case5", 5>;
def Case10: I32EnumAttrCase<"Case10", 10>;

def I32Enum: I32EnumAttr<"I32Enum", "A test enum", [Case5, Case10]>;

def Bit0 : BitEnumAttrCase<"None", 0x0000>;
def Bit1 : BitEnumAttrCase<"Bit1", 0x0001>;
def Bit3 : BitEnumAttrCase<"Bit3", 0x0004>;

def BitEnumWithNone : BitEnumAttr<"BitEnumWithNone", "A test enum",
                                  [Bit0, Bit1, Bit3]>;

def BitEnumWithoutNone : BitEnumAttr<"BitEnumWithoutNone", "A test enum",
                                     [Bit1, Bit3]>;
