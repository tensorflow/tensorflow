#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

typedef FunctionDefHelper FDH;

Status ReluGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"x: T", "dy: T"},
      // Ret val defs
      {"dx: T"},
      // Attr defs
      {{"T: {float, double}"}},
      // Nodes
      {
        {{"dx"}, "ReluGrad", {"dy", "x"}, {{"T", "$T"}}}
      });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("Relu", ReluGrad);

Status CrossEntropyGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
    // Arg defs
    {"features: T", "labels: T", "dcost_dloss: T", "donotcare: T"},
    // Ret val defs
    {"dcost_dfeatures: T", "dcost_dlabels: T"},
    // Attr defs
    {{"T: {float, double}"}},
    // Nodes
    {
      // _, dloss_dfeatures = CrossEntropy(features, labels)
      {{"donotcare_loss", "dloss_dfeatures"}, "CrossEntropy",
       {"features", "labels"}, {{"T", "$T"}}},
      // dcost_dloss is of shape [batch_size].
      // dcost_dloss_mat is of shape [batch_size, 1].
      FDH::Const("neg1", -1),
      {{"dcost_dloss_mat"}, "ExpandDims", {"dcost_dloss", "neg1"},
       {{"T", "$T"}}},
      // chain rule: dcost/dfeatures = dcost/dloss * dloss/dfeatures
      {{"dcost_dfeatures"}, "Mul", {"dcost_dloss_mat", "dloss_dfeatures"},
       {{"T", "$T"}}},
      {{"dcost_dlabels"}, "ZerosLike", {"labels"}, {{"T", "$T"}}},
    });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("CrossEntropy", CrossEntropyGrad);

}  // end namespace tensorflow
