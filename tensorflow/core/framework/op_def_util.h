// TODO(josh11b): Probably not needed for OpKernel authors, so doesn't
// need to be as publicly accessible as other files in framework/.

#ifndef TENSORFLOW_FRAMEWORK_OP_DEF_UTIL_H_
#define TENSORFLOW_FRAMEWORK_OP_DEF_UTIL_H_

#include <string>
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/public/status.h"

namespace tensorflow {

// Performs a consistency check across the fields of the op_def.
Status ValidateOpDef(const OpDef& op_def);

// Validates that attr_value satisfies the type and constraints from attr.
// REQUIRES: attr has already been validated.
Status ValidateAttrValue(const AttrValue& attr_value,
                         const OpDef::AttrDef& attr);

// The following search through op_def for an attr with the indicated name.
// Returns nullptr if no such attr is found.
const OpDef::AttrDef* FindAttr(StringPiece name, const OpDef& op_def);
OpDef::AttrDef* FindAttrMutable(StringPiece name, OpDef* op_def);

// Produce a human-readable version of an op_def that is more concise
// than a text-format proto.  Excludes descriptions.
string SummarizeOpDef(const OpDef& op_def);

// Returns an error if new_op is not backwards-compatible with (more
// accepting than) old_op.
// REQUIRES: old_op and new_op must pass validation.
Status OpDefCompatible(const OpDef& old_op, const OpDef& new_op);

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_OP_DEF_UTIL_H_
