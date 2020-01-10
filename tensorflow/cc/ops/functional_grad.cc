#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

typedef FunctionDefHelper FDH;

Status MapAccumulateGrad(const AttrSlice& attrs, FunctionDef* ret) {
  const NameAttrList* func;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "f", &func));
  DataType T;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "T", &T));
  int k;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "K", &k));
  // The gradient function of f.
  //  f : (K*T, T, T) -> T
  //  g : (K*T, T, T, T) -> (K*T, T, T)
  auto grad = FDH::FunctionRef("SymbolicGradient",
                               {{"f", *func},
                                {"Tin", std::vector<DataType>(k + 3, T)},
                                {"Tout", std::vector<DataType>(k + 2, T)}});
  *ret = FDH::Define(
      // Arg defs
      {"theta: K*T", "x: T", "u: T", "dy: T"},
      // Ret val defs
      {"dtheta: K*T", "dx: T", "du: T"},
      // Attr defs
      {{"T: {float, double}"}},
      // nodes.
      {{{"y"},
        "MapAccumulate",
        {"theta", "x", "u"},
        {{"f", *func}, {"T", "$T"}, {"K", k}}},
       {{"dtheta", "dx", "du"},
        "MapAccumulateGrad",
        {"theta", "x", "u", "y", "dy"},
        {{"g", grad}, {"T", "$T"}, {"K", k}}}});
  return Status::OK();
}
REGISTER_OP_GRADIENT("MapAccumulate", MapAccumulateGrad);

}  // end namespace tensorflow
