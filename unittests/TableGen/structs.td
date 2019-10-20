//===-- structs.td - StructsGen test definition file -------*- tablegen -*-===//
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

def Test_Dialect : Dialect {
  let name = "test";
}

def Test_Struct : StructAttr<"TestStruct", Test_Dialect, [
                StructFieldAttr<"sample_integer", I32Attr>,
                StructFieldAttr<"sample_float", F32Attr>,
                StructFieldAttr<"sample_elements", ElementsAttr>] > {
  let description = "Structure for test data";
}
