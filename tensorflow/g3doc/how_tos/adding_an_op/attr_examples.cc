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
