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

#ifndef TENSORFLOW_FRAMEWORK_NODE_DEF_UTIL_H_
#define TENSORFLOW_FRAMEWORK_NODE_DEF_UTIL_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

// Name of the attribute used to encode node colocation constraints.
//
// Nodes can be co-located on the same device. Desire for explicit co-location
// is described by list(string) attribute containing the name of colocation
// groups.
extern const char* const kColocationAttrName;

// String prefix applied to the operation name for colocation constraints.
extern const char* const kColocationGroupPrefix;

// Produce a human-readable version of a NodeDef that is more concise
// than a text-format proto.
string SummarizeNodeDef(const NodeDef& node_def);

typedef protobuf::Map<string, AttrValue> AttrValueMap;

// Adds an attr with name <name> and value <value> to *node_def.
// The type of the attr is based on the type of value.
template <class T>
void AddNodeAttr(StringPiece name, T&& value, NodeDef* node_def) {
  AttrValue attr_value;
  SetAttrValue(std::forward<T>(value), &attr_value);
  node_def->mutable_attr()->insert(
      AttrValueMap::value_type(name.ToString(), attr_value));
}

// Version to workaround C++'s "perfect" forwarding not being able to
// forward {...} initialization.
template <class T>
void AddNodeAttr(StringPiece name, std::initializer_list<T> value,
                 NodeDef* node_def) {
  AttrValue attr_value;
  SetAttrValue(value, &attr_value);
  node_def->mutable_attr()->insert(
      AttrValueMap::value_type(name.ToString(), attr_value));
}

// Adds an attr to an attr value map.
template <class T>
void AddAttr(StringPiece name, T&& value, AttrValueMap* map) {
  AttrValue attr_value;
  SetAttrValue(value, &attr_value);
  map->insert(AttrValueMap::value_type(name.ToString(), attr_value));
}

class AttrSlice {
 public:
  AttrSlice(const NodeDef& node_def);  // NOLINT(runtime/explicit)

  explicit AttrSlice(const AttrValueMap* a);

  // Returns the attr with attr_name if found.  Otherwise, returns
  // nullptr.
  const AttrValue* Find(StringPiece attr_name) const;

  // Returns the attr_value for attr_name if found. Otherwise, returns a
  // NotFound status.
  Status Find(StringPiece attr_name, const AttrValue** attr_value) const;

 private:
  const NodeDef* ndef_;
  const AttrValueMap* attrs_;
};

// Look up the attr with name attr_name and set *value to its value.  If no
// attr with attr_name is found in node_def, or the attr does not have
// a matching type, a non-ok status will be returned.
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   string* value);  // type: "string"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   int64* value);  // type: "int"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   int32* value);  // type: "int"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   float* value);  // type: "float"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   bool* value);  // type: "bool"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   DataType* value);  // type: "type"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   TensorShapeProto* value);  // type: "shape"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   TensorShape* value);  // type: "shape"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   PartialTensorShape* value);  // type: "shape"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   Tensor* value);  // type: "tensor"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   std::vector<string>* value);  // type "list(string)"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   std::vector<int64>* value);  // type "list(int)"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   std::vector<int32>* value);  // type "list(int)"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   std::vector<float>* value);  // type "list(float)"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   std::vector<bool>* value);  // type "list(bool)"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   std::vector<DataType>* value);  // type "list(type)"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   DataTypeVector* value);  // type "list(type)"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   std::vector<TensorShapeProto>* value);  // type "list(shape)"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   std::vector<TensorShape>* value);  // type "list(shape)"
Status GetNodeAttr(
    const AttrSlice& attrs, StringPiece attr_name,
    std::vector<PartialTensorShape>* value);  // type "list(shape)"
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   std::vector<Tensor>* value);  // type: "list(tensor)"

// This version avoids copying the TensorProto.
// REQUIRES: Must not use *value beyond the lifetime of node_def.
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   const TensorProto** value);  // type: "tensor"

// This version avoids copying the NameAttrList.
// REQUIRES: Must not use *value beyond the lifetime of node_def.
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   const NameAttrList** value);  // type: "func"

Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                   std::vector<NameAttrList>* value);  // type: "list(func)"

// Look up the attr with name attr_name and set *value to its value.  If no
// attr with attr_name is found in node_def, or the attr does not have
// a matching type, false is returned.
bool GetNodeAttrSimple(const AttrSlice& attrs, StringPiece attr_name,
                       string* value);  // type: "string"
bool GetNodeAttrSimple(const AttrSlice& attrs, StringPiece attr_name,
                       std::vector<string>* value);  // type: "string"

// Look up the attr with name attr_name and return a reference to its value.
// If no attr with attr_name is found in node_def, or the attr does not have
// a matching type, a reference to an empty string is returned.
// REQUIRES: Must not use the returned value beyond the lifetime of node_def.
const string& GetNodeAttrString(const AttrSlice& attrs, StringPiece attr_name);

// Computes the input and output types for a specific node.
// REQUIRES: ValidateOpDef(op_def).ok()
Status InOutTypesForNode(const NodeDef& node_def, const OpDef& op_def,
                         DataTypeVector* inputs, DataTypeVector* outputs);

// Validates that the NodeDef:
// * Defines all expected attrs from the OpDef.
// * All attrs satisfies constraints from the OpDef.
// * Has a signature matching SignatureForNode().
// etc.
Status ValidateNodeDef(const NodeDef& node_def, const OpDef& op_def);

// Computes the mapping from input/output argument name to the
// corresponding input/output index range.  For example,
// input "foo" corresponds to input indices
//   [ (*inputs)["foo"].first, (*inputs)["foo"].second ).
typedef std::unordered_map<string, std::pair<int, int>> NameRangeMap;
Status NameRangesForNode(const NodeDef& node_def, const OpDef& op_def,
                         NameRangeMap* inputs, NameRangeMap* outputs);

// Adds default values to *node_def for unspecified attrs from op_def.
void AddDefaultsToNodeDef(const OpDef& op_def, NodeDef* node_def);

// Validates the syntax of a NodeDef provided externally.
//
// The following is an EBNF-style syntax for NodeDef objects. Note that
// Node objects are actually specified as tensorflow::NodeDef protocol buffers,
// which contain many other fields that are not (currently) validated.
//
// Node         = NodeName, Inputs
// Inputs       = ( DataInput * ), ( ControlInput * )
// DataInput    = NodeName, ( ":", [1-9], [0-9] * ) ?
// ControlInput = "^", NodeName
// NodeName     = [A-Za-z0-9.], [A-Za-z0-9_./] *
Status ValidateExternalNodeDefSyntax(const NodeDef& node_def);

// Returns "status" with kernel's NodeDef attached as additional text
// in the error message.
Status AttachDef(const Status& status, const NodeDef& node_def);

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_NODE_DEF_UTIL_H_
