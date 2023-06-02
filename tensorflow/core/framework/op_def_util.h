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

// TODO(josh11b): Probably not needed for OpKernel authors, so doesn't
// need to be as publicly accessible as other files in framework/.

#ifndef TENSORFLOW_CORE_FRAMEWORK_OP_DEF_UTIL_H_
#define TENSORFLOW_CORE_FRAMEWORK_OP_DEF_UTIL_H_

#include <string>

#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

// Performs a consistency check across the fields of the op_def.
Status ValidateOpDef(const OpDef& op_def);

// Check if an op is deprecated at the given GraphDef version.  If the op is
// deprecated at a future version, a warning will be logged.
Status CheckOpDeprecation(const OpDef& op_def, int graph_def_version);

// Validates that attr_value satisfies the type and constraints from attr.
// REQUIRES: attr has already been validated.
Status ValidateAttrValue(const AttrValue& attr_value,
                         const OpDef::AttrDef& attr);

// The following search through op_def for an attr with the indicated name.
// Returns nullptr if no such attr is found.
const OpDef::AttrDef* FindAttr(StringPiece name, const OpDef& op_def);
OpDef::AttrDef* FindAttrMutable(StringPiece name, OpDef* op_def);

// Searches op_def for input argument with the indicated name.
// Returns nullptr if no such attr is found.
const OpDef::ArgDef* FindInputArg(StringPiece name, const OpDef& op_def);

// Searches api_def for input argument with the indicated name.
// Returns nullptr if no such attr is found.
const ApiDef::Arg* FindInputArg(StringPiece name, const ApiDef& api_def);

// Produce a human-readable version of an op_def that is more concise
// than a text-format proto.  Excludes descriptions.
std::string SummarizeOpDef(const OpDef& op_def);

// Returns an error if new_op is not backwards-compatible with (more
// accepting than) old_op.
// REQUIRES: old_op and new_op must pass validation.
Status OpDefCompatible(const OpDef& old_op, const OpDef& new_op);

// Returns an error if any attr in penultimate_op that is not in old_op
// has a different default value in new_op.  In general it is not safe
// to change the default for an attr that has been added to an op.
Status OpDefAddedDefaultsUnchanged(const OpDef& old_op,
                                   const OpDef& penultimate_op,
                                   const OpDef& new_op);

// Returns an error if the default value for any attr is removed or modified
// in new_op compared to old_op.  Adding new default values is safe, and does
// not raise an error.
Status OpDefAttrDefaultsUnchanged(const OpDef& old_op, const OpDef& new_op);

// Remove all docs from *op_def / *op_list.
void RemoveDescriptionsFromOpDef(OpDef* op_def);
void RemoveDescriptionsFromOpList(OpList* op_list);

// Remove docs from *op_def but leave explanations of deprecations.
void RemoveNonDeprecationDescriptionsFromOpDef(OpDef* op_def);

// Returns true if `a1` is equal to `a2`.
// Equality includes all the fields.
bool AttrDefEqual(const OpDef::AttrDef& a1, const OpDef::AttrDef& a2);

// Returns hash of `a` that is consistent with AttrDefEqual.
uint64 AttrDefHash(const OpDef::AttrDef& a);

// Returns true if all AttrDefs in `a1` equal corresponding AttrDefs in
// `a2`. Correspondence is established by name.
bool RepeatedAttrDefEqual(const protobuf::RepeatedPtrField<OpDef::AttrDef>& a1,
                          const protobuf::RepeatedPtrField<OpDef::AttrDef>& a2);

// Returns hash of `a` that is consistent with RepeatedAttrDefEqual
uint64 RepeatedAttrDefHash(const protobuf::RepeatedPtrField<OpDef::AttrDef>& a);

// Returns true if `o1` is equal to `o2`.
// Equality includes all the fields. OpDef.attr field is treated as a set.
bool OpDefEqual(const OpDef& o1, const OpDef& o2);

// Returns hash of `o` that is consistent with AttrDefEqual.
uint64 OpDefHash(const OpDef& o);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_OP_DEF_UTIL_H_
