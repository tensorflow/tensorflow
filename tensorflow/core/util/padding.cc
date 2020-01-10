#include "tensorflow/core/util/padding.h"

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

Status GetNodeAttr(const NodeDef& node_def, const string& attr_name,
                   Padding* value) {
  string str_value;
  TF_RETURN_IF_ERROR(GetNodeAttr(node_def, attr_name, &str_value));
  if (str_value == "SAME") {
    *value = SAME;
  } else if (str_value == "VALID") {
    *value = VALID;
  } else {
    return errors::NotFound(str_value, " is not an allowed padding type");
  }
  return Status::OK();
}

string GetPaddingAttrString() { return "padding: {'SAME', 'VALID'}"; }

}  // end namespace tensorflow
