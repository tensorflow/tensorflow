/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <stdio.h>
#include "tensorflow/core/framework/op.h"

REGISTER_OP("RestrictedTypeExample").Attr("t: {int32, float, bool}");

REGISTER_OP("NumberType").Attr("t: numbertype");

REGISTER_OP("EnumExample").Attr("e: {'apple', 'orange'}");

REGISTER_OP("MinIntExample").Attr("a: int >= 2");

REGISTER_OP("TypeListExample").Attr("a: list({int32, float}) >= 3");

REGISTER_OP("AttrDefaultExample").Attr("i: int = 0");

REGISTER_OP("AttrDefaultExampleForAllTypes")
    .Attr("s: string = 'foo'")
    .Attr("i: int = 0")
    .Attr("f: float = 1.0")
    .Attr("b: bool = true")
    .Attr("ty: type = DT_INT32")
    .Attr("sh: shape = { dim { size: 1 } dim { size: 2 } }")
    .Attr("te: tensor = { dtype: DT_INT32 int_val: 5 }")
    .Attr("l_empty: list(int) = []")
    .Attr("l_int: list(int) = [2, 3, 5, 7]");

int main(int argc, char* argv[]) {
  printf("All registered ops:\n%s\n",
         tensorflow::OpRegistry::Global()->DebugString(false).c_str());
  return 0;
}
