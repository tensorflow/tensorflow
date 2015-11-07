#include "tensorflow/core/framework/graph_def_util.h"

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/public/status.h"

namespace tensorflow {

string SummarizeGraphDef(const GraphDef& graph_def) {
  string ret;
  for (const NodeDef& node : graph_def.node()) {
    strings::StrAppend(&ret, SummarizeNodeDef(node), ";\n");
  }
  return ret;
}

Status ValidateExternalGraphDefSyntax(const GraphDef& graph_def) {
  for (const NodeDef& node : graph_def.node()) {
    TF_RETURN_IF_ERROR(ValidateExternalNodeDefSyntax(node));
  }
  return Status::OK();
}

}  // namespace tensorflow
