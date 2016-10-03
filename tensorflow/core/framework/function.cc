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

#include "tensorflow/core/framework/function.h"

#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/function.pb_text.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace tensorflow {

namespace {

// Extracts the actual type from "attr_values" based on its definition
// "arg_def".
//
// If "arg_def" is a N*T type, *is_type_list is set to false, and
// *dtypes is set to be a vector of size N and each element is T.
//
// If "arg_def" is a list(type), *is_type_list is set to true, and
// *dtypes is set to be a vector of types specified in attrs for
// arg_def.
//
// Otherwise (arg_def is a simple type T), *is_type_list is set to
// false, and *dtypes is set to a single element vector, whose only
// element is T.
Status ArgNumType(const InstantiateAttrValueMap& attrs,
                  const OpDef::ArgDef& arg_def, bool* is_type_list,
                  DataTypeVector* dtypes) {
  dtypes->clear();
  if (!arg_def.type_list_attr().empty()) {
    const AttrValue* v = gtl::FindOrNull(attrs, arg_def.type_list_attr());
    if (v == nullptr) {
      return errors::NotFound("type attr not found: ",
                              arg_def.type_list_attr());
    }
    *is_type_list = true;
    for (int i = 0; i < v->list().type_size(); ++i) {
      dtypes->push_back(v->list().type(i));
    }
    return Status::OK();
  }

  *is_type_list = false;
  int num = 1;
  if (!arg_def.number_attr().empty()) {
    const AttrValue* v = gtl::FindOrNull(attrs, arg_def.number_attr());
    if (v == nullptr) {
      return errors::NotFound("type attr not found: ", arg_def.type_attr());
    }
    num = v->i();
  }

  DataType dtype;
  if (arg_def.type() != DT_INVALID) {
    dtype = arg_def.type();
  } else if (arg_def.type_attr().empty()) {
    dtype = DT_INVALID;
  } else {
    const AttrValue* v = gtl::FindOrNull(attrs, arg_def.type_attr());
    if (v == nullptr) {
      return errors::NotFound("type attr not found: ", arg_def.type_attr());
    }
    dtype = v->type();
  }
  dtypes->resize(num, dtype);
  return Status::OK();
}

string Name(int node_index) { return strings::StrCat("n", node_index); }

string Name(int node_index, int output_index) {
  if (output_index == 0) {
    return Name(node_index);
  } else {
    return strings::StrCat("n", node_index, ":", output_index);
  }
}

string Dep(int node_index) { return strings::StrCat("^", Name(node_index)); }

template <typename T>
void AddAttr(const string& name, const T& val, NodeDef* ndef) {
  SetAttrValue(val, &((*ndef->mutable_attr())[name]));
}

Status ValidateSignatureWithAttrs(const OpDef& sig,
                                  const InstantiateAttrValueMap& attr_values) {
  // attr_values should specify all attrs defined in fdef.
  for (const auto& a : sig.attr()) {
    auto const iter = attr_values.find(a.name());
    if (iter == attr_values.end()) {
      return errors::NotFound("Attr ", a.name(), " is not found from ",
                              SummarizeOpDef(sig));
    }
    Status status = AttrValueHasType(iter->second, a.type());
    if (!status.ok()) {
      errors::AppendToMessage(&status, "for attr '", iter->first, "'");
      return status;
    }
  }

// TODO(josh11b): Enable this code once it works with function gradients.
// Right now the C++ function gradient code assumes it can pass
// all the attrs of the function to the gradient, and any attrs that
// the gradient doesn't care about will be ignored.
#if 0
  if (attr_values.size() != sig.attr_size()) {
    for (const auto& a : attr_values) {
      // TODO(josh11b): Possibly should ignore attrs that start with "_" here?
      bool found = false;
      for (const auto& s : sig.attr()) {
        if (a.first == s.name()) {
          found = true;
          break;
        }
      }
      if (!found) {
        return errors::NotFound("Attr ", a.first, " is not found in ",
                                SummarizeOpDef(sig));
      }
    }
  }
#endif

  return Status::OK();
}

// We build a small index for all names that can be used as a node's
// input arguments.
//
// If is_func_arg is true, the name is a function's argument.  In
// this case, the produced graph def has gdef.node[nid ... nid +
// dtype.size()).
//
// Otherwise, the name is a function body's node return value.  In
// this case, the produced graph def has one node gdef.node[nid] and
// the node's output index [idx ... idx + num) corresponds to the
// named outputs.
//
// In all cases, "dtype" specifies the data type.
struct NameInfoItem {
  bool is_func_arg;
  int nid;
  int idx;
  bool is_type_list;
  DataTypeVector dtypes;
};
typedef std::unordered_map<string, NameInfoItem> NameInfoIndex;

Status AddArgName(NameInfoIndex* name_info, const string& arg,
                  const NameInfoItem& item) {
  if (!name_info->insert({arg, item}).second) {
    return errors::InvalidArgument("Duplicated arg name.");
  }
  return Status::OK();
}

Status BuildInputArgIndex(const OpDef::ArgDef& arg_def,
                          const InstantiateAttrValueMap& attr_values,
                          NameInfoIndex* name_info,
                          InstantiationResult* result) {
  bool is_type_list;
  DataTypeVector dtypes;
  TF_RETURN_IF_ERROR(ArgNumType(attr_values, arg_def, &is_type_list, &dtypes));
  CHECK_GE(dtypes.size(), size_t{1});
  GraphDef* gdef = &result->gdef;
  int arg_index = gdef->node_size();
  TF_RETURN_IF_ERROR(AddArgName(name_info, arg_def.name(),
                                {true, arg_index, 0, is_type_list, dtypes}));
  // Creates dtypes.size() nodes in the gdef.
  for (size_t i = 0; i < dtypes.size(); ++i) {
    TF_RETURN_IF_ERROR(AddArgName(name_info,
                                  strings::StrCat(arg_def.name(), ":", i),
                                  {true, arg_index, 0, false, {dtypes[i]}}));
    DCHECK_EQ(arg_index, gdef->node_size());
    NodeDef* gnode = gdef->add_node();
    gnode->set_name(Name(arg_index));
    gnode->set_op("_Arg");
    AddAttr("T", dtypes[i], gnode);
    AddAttr("index", arg_index, gnode);
    result->arg_types.push_back(dtypes[i]);
    ++arg_index;
  }
  return Status::OK();
}

Status AddRetName(NameInfoIndex* name_info, const string& ret,
                  const NameInfoItem& item) {
  if (!name_info->insert({ret, item}).second) {
    return errors::InvalidArgument("Duplicated ret name.");
  }
  return Status::OK();
}

Status BuildNodeOutputIndex(const FunctionDef::Node& node,
                            const InstantiateAttrValueMap& attrs,
                            GetFunctionSignature get_function,
                            const int arg_index, NameInfoIndex* name_info) {
  const OpDef* node_sig = nullptr;
  TF_RETURN_IF_ERROR(get_function(node.op(), &node_sig));
  if (node_sig->output_arg_size() == 0) {
    // This node produces no output.
    if (node.ret_size() != 1) {
      return errors::InvalidArgument("Expect one ret name.");
    }
    return AddRetName(name_info, node.ret(0), {false, arg_index, 0, false, {}});
  }
  const int num_retval = node_sig->output_arg_size();
  if (num_retval != node.ret_size()) {
    return errors::InvalidArgument("Malformed function node (#ret): ",
                                   num_retval, " vs. ", node.ret_size());
  }
  int start = 0;
  bool is_type_list;
  DataTypeVector dtypes;
  for (int i = 0; i < num_retval; ++i) {
    TF_RETURN_IF_ERROR(
        ArgNumType(attrs, node_sig->output_arg(i), &is_type_list, &dtypes));
    TF_RETURN_IF_ERROR(
        AddRetName(name_info, node.ret(i),
                   {false, arg_index, start, is_type_list, dtypes}));
    for (int j = 0; j < static_cast<int>(dtypes.size()); ++j) {
      TF_RETURN_IF_ERROR(
          AddRetName(name_info, strings::StrCat(node.ret(i), ":", j),
                     {false, arg_index, start + j, false, {dtypes[j]}}));
    }
    start += dtypes.size();
  }
  return Status::OK();
}

Status BuildNodeOutputIndex(const NodeDef& node,
                            const InstantiateAttrValueMap& attrs,
                            GetFunctionSignature get_function,
                            const int arg_index, NameInfoIndex* name_info) {
  const OpDef* node_sig = nullptr;
  TF_RETURN_IF_ERROR(get_function(node.op(), &node_sig));
  if (node_sig->output_arg_size() == 0) {
    return AddRetName(name_info, node.name(), {false, arg_index, 0, false, {}});
  }
  const int num_retval = node_sig->output_arg_size();
  int start = 0;
  bool is_type_list;
  DataTypeVector dtypes;
  for (int i = 0; i < num_retval; ++i) {
    TF_RETURN_IF_ERROR(
        ArgNumType(attrs, node_sig->output_arg(i), &is_type_list, &dtypes));
    // Note that we rely on the backwards-compatibility test enforcing
    // that output_arg(*).name() doesn't change here.
    const string base_name =
        strings::StrCat(node.name(), ":", node_sig->output_arg(i).name());
    TF_RETURN_IF_ERROR(AddRetName(
        name_info, base_name, {false, arg_index, start, is_type_list, dtypes}));
    for (int j = 0; j < static_cast<int>(dtypes.size()); ++j) {
      TF_RETURN_IF_ERROR(
          AddRetName(name_info, strings::StrCat(base_name, ":", j),
                     {false, arg_index, start + j, false, {dtypes[j]}}));
    }
    start += dtypes.size();
  }
  return Status::OK();
}

Status InstantiateNode(const FunctionDef::Node& fnode,
                       const InstantiateAttrValueMap& attrs,
                       GetFunctionSignature get_function,
                       const NameInfoIndex& name_info, GraphDef* gdef) {
  const OpDef* fnode_sig = nullptr;
  TF_CHECK_OK(get_function(fnode.op(), &fnode_sig));
  NodeDef* gnode = gdef->add_node();
  gnode->set_name(Name(gdef->node_size() - 1));
  gnode->set_op(fnode.op());

  // Input
  const int num_args = fnode_sig->input_arg_size();
  bool is_type_list;
  DataTypeVector dtypes;
  int fnode_arg_index = 0;
  for (int i = 0; i < num_args; ++i) {
    TF_RETURN_IF_ERROR(
        ArgNumType(attrs, fnode_sig->input_arg(i), &is_type_list, &dtypes));
    if (!is_type_list) {
      const NameInfoItem* item =
          gtl::FindOrNull(name_info, fnode.arg(fnode_arg_index));
      if (item == nullptr) {
        return errors::InvalidArgument("arg[", i, "] is not found: ",
                                       ProtoShortDebugString(fnode));
      }
      if (dtypes != item->dtypes) {
        return errors::InvalidArgument("Invalid arg(", i,
                                       ") for function arg: ",
                                       DataTypeSliceString(dtypes), " vs. ",
                                       DataTypeSliceString(item->dtypes), ".");
      }
      for (size_t j = 0; j < dtypes.size(); ++j) {
        if (item->is_func_arg) {
          gnode->add_input(Name(item->nid + j));
        } else {
          gnode->add_input(Name(item->nid, item->idx + j));
        }
      }
      ++fnode_arg_index;
    } else {
      for (size_t j = 0; j < dtypes.size(); ++j) {
        const NameInfoItem* item =
            gtl::FindOrNull(name_info, fnode.arg(fnode_arg_index + j));
        if (item == nullptr) {
          return errors::InvalidArgument("arg[", i + j, "] is not found: ",
                                         ProtoShortDebugString(fnode));
        }
        if (item->dtypes.size() != 1 || (item->dtypes[0] != dtypes[j])) {
          return errors::InvalidArgument(
              "Invalid typelist arg(", i + j, ") for function arg: ",
              DataTypeSliceString(dtypes), " vs. ",
              DataTypeSliceString(item->dtypes), ".");
        }
        if (item->is_func_arg) {
          gnode->add_input(Name(item->nid));
        } else {
          gnode->add_input(Name(item->nid, item->idx));
        }
      }
      fnode_arg_index += dtypes.size();
    }
  }
  // Control deps.
  for (int i = 0; i < fnode.dep_size(); ++i) {
    const NameInfoItem* item = gtl::FindOrNull(name_info, fnode.dep(i));
    if (item == nullptr) {
      return errors::InvalidArgument("dep[", i, "] is not found.");
    }
    gnode->add_input(Dep(item->nid));
  }

  // Attrs.
  for (const auto& p : attrs) {
    (*gnode->mutable_attr())[p.first] = p.second;
  }

  return Status::OK();
}

Status InstantiateNode(const NodeDef& fnode,
                       const InstantiateAttrValueMap& attrs,
                       GetFunctionSignature get_function,
                       const NameInfoIndex& name_info, GraphDef* gdef) {
  const OpDef* fnode_sig = nullptr;
  TF_CHECK_OK(get_function(fnode.op(), &fnode_sig));
  NodeDef* gnode = gdef->add_node();
  gnode->set_name(Name(gdef->node_size() - 1));
  gnode->set_op(fnode.op());

  // Input
  const int num_args = fnode_sig->input_arg_size();
  bool is_type_list;  // ignored
  DataTypeVector dtypes;
  int fnode_arg_index = 0;
  for (int i = 0; i < num_args; ++i) {
    TF_RETURN_IF_ERROR(
        ArgNumType(attrs, fnode_sig->input_arg(i), &is_type_list, &dtypes));
    // Consume inputs (indexed by fnode_arg_index) until we have
    // matched each element of dtypes (indexed by j).
    for (size_t j = 0; j < dtypes.size(); ++fnode_arg_index) {
      if (fnode_arg_index >= fnode.input_size()) {
        // Should never happen if we computed dtypes correctly.
        return errors::InvalidArgument("Attempt to access beyond input size: ",
                                       fnode_arg_index, " >= ",
                                       fnode.input_size());
      }
      // Look up the next input.
      const string& input_name = fnode.input(fnode_arg_index);
      const NameInfoItem* item = gtl::FindOrNull(name_info, input_name);
      if (item == nullptr) {
        return errors::InvalidArgument("input ", input_name, " is not found: ",
                                       SummarizeNodeDef(fnode));
      }
      if (item->dtypes.size() > dtypes.size() - j) {
        return errors::InvalidArgument("Input ", input_name, " too long for ",
                                       fnode_sig->input_arg(i).name());
      }
      // Match up all the elements of this input (indexed by k) with
      // elements of dtypes (advancing j).
      for (int k = 0; k < item->dtypes.size(); ++k, ++j) {
        if (item->dtypes[k] != dtypes[j]) {
          return errors::InvalidArgument(
              "input ", fnode_sig->input_arg(i).name(), "[", j,
              "] expected type ", DataTypeString(dtypes[j]), " != ",
              DataTypeString(item->dtypes[k]), ", the type of ", input_name,
              "[", k, "]");
        }
        if (item->is_func_arg) {
          gnode->add_input(Name(item->nid + k));
        } else {
          gnode->add_input(Name(item->nid, item->idx + k));
        }
      }
    }
  }

  // Control deps.
  for (int i = fnode_arg_index; i < fnode.input_size(); ++i) {
    const string& input = fnode.input(i);
    if (input.empty() || input[0] != '^') {
      return errors::InvalidArgument("Expected input[", i, "] == '", input,
                                     "' to be a control input.");
    }
    const NameInfoItem* item = gtl::FindOrNull(name_info, input.substr(1));
    if (item == nullptr) {
      return errors::InvalidArgument("input[", i, "] == '", input,
                                     "', is not found.");
    }
    gnode->add_input(Dep(item->nid));
  }

  // Attrs.
  for (const auto& p : attrs) {
    (*gnode->mutable_attr())[p.first] = p.second;
  }

  return Status::OK();
}

// FunctionDef::Node version
Status AddReturnNode(const OpDef::ArgDef& ret_def,
                     const InstantiateAttrValueMap& attrs,
                     const NameInfoIndex& name_info, int* ret_index,
                     InstantiationResult* result) {
  bool is_type_list;
  DataTypeVector dtypes;
  TF_RETURN_IF_ERROR(ArgNumType(attrs, ret_def, &is_type_list, &dtypes));
  CHECK_GE(dtypes.size(), size_t{1});
  const NameInfoItem* item = gtl::FindOrNull(name_info, ret_def.name());
  if (item == nullptr) {
    return errors::InvalidArgument("ret is not found.");
  }
  if (dtypes != item->dtypes) {
    return errors::InvalidArgument("Invalid ret types ", ret_def.name(), " : ",
                                   DataTypeVectorString(dtypes), " vs. ",
                                   DataTypeVectorString(item->dtypes));
  }
  GraphDef* gdef = &result->gdef;
  for (size_t i = 0; i < dtypes.size(); ++i) {
    NodeDef* gnode = gdef->add_node();
    gnode->set_name(Name(gdef->node_size() - 1));
    gnode->set_op("_Retval");
    gnode->add_input(Name(item->nid, item->idx + i));
    AddAttr("T", dtypes[i], gnode);
    AddAttr("index", (*ret_index)++, gnode);
    result->ret_types.push_back(dtypes[i]);
  }
  return Status::OK();
}

// NodeDef version
Status AddReturnNode(const OpDef::ArgDef& ret_def,
                     const InstantiateAttrValueMap& attrs,
                     const ::tensorflow::protobuf::Map<string, string>& ret_map,
                     const NameInfoIndex& name_info, int* ret_index,
                     InstantiationResult* result) {
  auto ret_iter = ret_map.find(ret_def.name());
  if (ret_iter == ret_map.end()) {
    return errors::InvalidArgument("Return ", ret_def.name(), " missing.");
  }
  bool is_type_list;
  DataTypeVector dtypes;
  TF_RETURN_IF_ERROR(ArgNumType(attrs, ret_def, &is_type_list, &dtypes));
  CHECK_GE(dtypes.size(), size_t{1});
  const NameInfoItem* item = gtl::FindOrNull(name_info, ret_iter->second);
  if (item == nullptr) {
    return errors::InvalidArgument("Return ", ret_def.name(), " -> ",
                                   ret_iter->second, " is not found.");
  }
  if (dtypes != item->dtypes) {
    return errors::InvalidArgument("Invalid ret types ", ret_def.name(), " : ",
                                   DataTypeVectorString(dtypes), " vs. ",
                                   DataTypeVectorString(item->dtypes));
  }
  GraphDef* gdef = &result->gdef;
  for (size_t i = 0; i < dtypes.size(); ++i) {
    NodeDef* gnode = gdef->add_node();
    gnode->set_name(Name(gdef->node_size() - 1));
    gnode->set_op("_Retval");
    gnode->add_input(Name(item->nid, item->idx + i));
    AddAttr("T", dtypes[i], gnode);
    AddAttr("index", (*ret_index)++, gnode);
    result->ret_types.push_back(dtypes[i]);
  }
  return Status::OK();
}

// Various helpers Print(proto) to print relevant protos to ascii.
string Print(const OpDef::ArgDef& arg) {
  string out;
  strings::StrAppend(&out, arg.name(), ":");
  if (arg.is_ref()) strings::StrAppend(&out, "Ref(");
  if (!arg.number_attr().empty()) {
    strings::StrAppend(&out, arg.number_attr(), "*");
  }
  if (arg.type() != DT_INVALID) {
    strings::StrAppend(&out, DataTypeString(arg.type()));
  } else {
    strings::StrAppend(&out, arg.type_attr());
  }
  if (arg.is_ref()) strings::StrAppend(&out, ")");
  return out;
}

// TODO(josh11b): Merge this with SummarizeAttrValue().
string Print(const AttrValue& attr_value) {
  if (attr_value.value_case() == AttrValue::kType) {
    return DataTypeString(attr_value.type());
  } else if ((attr_value.value_case() == AttrValue::kList) &&
             (attr_value.list().type_size() > 0)) {
    string ret = "{";
    for (int i = 0; i < attr_value.list().type_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, DataTypeString(attr_value.list().type(i)));
    }
    strings::StrAppend(&ret, "}");
    return ret;
  } else if (attr_value.value_case() == AttrValue::kFunc) {
    if (attr_value.func().attr_size() == 0) {
      return attr_value.func().name();
    }
    std::vector<string> entries;
    for (auto p : attr_value.func().attr()) {
      entries.push_back(strings::StrCat(p.first, "=", Print(p.second)));
    }
    sort(entries.begin(), entries.end());
    return strings::StrCat(attr_value.func().name(), "[",
                           str_util::Join(entries, ", "), "]");
  }
  return SummarizeAttrValue(attr_value);
}

string Print(const FunctionDef::Node& node) {
  string out;
  for (int i = 0; i < node.ret_size(); ++i) {
    const auto& name = node.ret(i);
    if (i > 0) strings::StrAppend(&out, ", ");
    strings::StrAppend(&out, name);
  }
  strings::StrAppend(&out, " = ", node.op());
  if (node.attr_size() > 0) {
    std::vector<string> entries;
    for (auto p : node.attr()) {
      entries.push_back(strings::StrCat(p.first, "=", Print(p.second)));
    }
    sort(entries.begin(), entries.end());
    strings::StrAppend(&out, "[", str_util::Join(entries, ", "), "]");
  }
  strings::StrAppend(&out, "(");
  for (int i = 0; i < node.arg_size(); ++i) {
    if (i > 0) strings::StrAppend(&out, ", ");
    strings::StrAppend(&out, node.arg(i));
  }
  strings::StrAppend(&out, ")");
  if (node.dep_size() > 0) {
    strings::StrAppend(&out, " @ ");
    for (int i = 0; i < node.dep_size(); ++i) {
      if (i > 0) strings::StrAppend(&out, ", ");
      strings::StrAppend(&out, node.dep(i));
    }
  }
  return out;
}

// TODO(josh11b): Merge this with SummarizeNodeDef().
string Print(const NodeDef& n) {
  string out;
  strings::StrAppend(&out, n.name(), " = ", n.op());
  if (n.attr_size() > 0) {
    std::vector<string> entries;
    for (auto& a : n.attr()) {
      entries.push_back(strings::StrCat(a.first, "=", Print(a.second)));
    }
    sort(entries.begin(), entries.end());
    strings::StrAppend(&out, "[", str_util::Join(entries, ", "), "]");
  }
  strings::StrAppend(&out, "(");
  std::vector<StringPiece> dat;
  std::vector<string> dep;
  for (StringPiece s : n.input()) {
    if (s.Consume("^")) {
      dep.push_back(s.ToString());
    } else {
      dat.push_back(s);
    }
  }
  strings::StrAppend(&out, str_util::Join(dat, ", "), ")");
  if (!dep.empty()) {
    strings::StrAppend(&out, " @ ", str_util::Join(dep, ", "));
  }
  return out;
}

string Print(const FunctionDef& fdef) {
  string out;
  const OpDef& sig = fdef.signature();
  strings::StrAppend(&out, "\n", sig.name());
  if (sig.attr_size() > 0) {
    strings::StrAppend(&out, "[");
    for (int i = 0; i < sig.attr_size(); ++i) {
      const auto& a = sig.attr(i);
      if (i > 0) strings::StrAppend(&out, ", ");
      if (a.type() == "type") {
        strings::StrAppend(&out, a.name(), ":", Print(a.allowed_values()));
      } else {
        strings::StrAppend(&out, a.name(), ":", a.type());
      }
    }
    strings::StrAppend(&out, "]");
  }
  strings::StrAppend(&out, "(");
  for (int i = 0; i < sig.input_arg_size(); ++i) {
    if (i > 0) strings::StrAppend(&out, ", ");
    strings::StrAppend(&out, Print(sig.input_arg(i)));
  }
  strings::StrAppend(&out, ") -> (");
  for (int i = 0; i < sig.output_arg_size(); ++i) {
    if (i > 0) strings::StrAppend(&out, ", ");
    strings::StrAppend(&out, Print(sig.output_arg(i)));
  }
  strings::StrAppend(&out, ") {\n");
  if (fdef.node_def_size() > 0) {
    for (const auto& n : fdef.node_def()) {
      strings::StrAppend(&out, "  ", Print(n), "\n");
    }
    for (const auto& r : fdef.ret()) {
      strings::StrAppend(&out, "  return ", r.first, " = ", r.second, "\n");
    }
  } else {  // TODO(josh11b): Eventually remove this case.
    for (const auto& n : fdef.node()) {
      strings::StrAppend(&out, "  ", Print(n), "\n");
    }
  }
  strings::StrAppend(&out, "}\n");
  return out;
}

string Print(const GraphDef& gdef) {
  std::vector<const NodeDef*> arg;
  std::vector<const NodeDef*> ret;
  std::vector<const NodeDef*> body;
  for (const NodeDef& n : gdef.node()) {
    if (n.op() == "_Arg") {
      arg.push_back(&n);
    } else if (n.op() == "_Retval") {
      ret.push_back(&n);
    } else {
      body.push_back(&n);
    }
  }
  auto comp = [](const NodeDef* x, const NodeDef* y) {
    int xi;
    TF_CHECK_OK(GetNodeAttr(*x, "index", &xi));
    int yi;
    TF_CHECK_OK(GetNodeAttr(*y, "index", &yi));
    return xi < yi;
  };
  sort(arg.begin(), arg.end(), comp);
  sort(ret.begin(), ret.end(), comp);
  string out;
  strings::StrAppend(&out, "\n(");
  auto get_type = [](const NodeDef& n) {
    for (auto a : n.attr()) {
      if (a.first == "T") {
        return DataTypeString(a.second.type());
      }
    }
    return DataTypeString(DT_INVALID);
  };
  for (size_t i = 0; i < arg.size(); ++i) {
    const NodeDef* n = arg[i];
    if (i > 0) strings::StrAppend(&out, ", ");
    CHECK_EQ(2, n->attr_size());
    strings::StrAppend(&out, n->name(), ":", get_type(*n));
  }
  strings::StrAppend(&out, ") -> (");
  for (size_t i = 0; i < ret.size(); ++i) {
    const NodeDef* n = ret[i];
    if (i > 0) strings::StrAppend(&out, ", ");
    CHECK_EQ(2, n->attr_size());
    CHECK_EQ(1, n->input_size());
    strings::StrAppend(&out, n->input(0), ":", get_type(*n));
  }
  strings::StrAppend(&out, ") {\n");
  for (size_t i = 0; i < body.size(); ++i) {
    strings::StrAppend(&out, "  ", Print(*body[i]), "\n");
  }
  strings::StrAppend(&out, "}\n");
  return out;
}

Status AddDefaultAttrs(const string& op, GetFunctionSignature get_function,
                       InstantiateAttrValueMap* attrs) {
  const OpDef* op_def = nullptr;
  TF_RETURN_IF_ERROR(get_function(op, &op_def));
  AttrSlice attr_slice(attrs);
  for (const auto& attr_def : op_def->attr()) {
    if (attr_def.has_default_value() && !attr_slice.Find(attr_def.name())) {
      if (!attrs->insert({attr_def.name(), attr_def.default_value()}).second) {
        return errors::Internal("Somehow duplicated: ", attr_def.name());
      }
    }
  }
  return Status::OK();
}

}  // end namespace

Status InstantiateFunction(const FunctionDef& fdef,
                           const InstantiateAttrValueMap& attr_values,
                           GetFunctionSignature get_function,
                           InstantiationResult* result) {
  const OpDef& sig = fdef.signature();
  GraphDef* gdef = &result->gdef;
  gdef->Clear();

  TF_RETURN_IF_ERROR(ValidateSignatureWithAttrs(sig, attr_values));

  NameInfoIndex name_info;
  Status s;
  for (const OpDef::ArgDef& arg_def : sig.input_arg()) {
    s = BuildInputArgIndex(arg_def, attr_values, &name_info, result);
    if (!s.ok()) {
      errors::AppendToMessage(&s, "In ", Print(arg_def));
      return s;
    }
  }

  auto substitute = [&attr_values](const string& name, AttrValue* val) {
    auto iter = attr_values.find(name);
    if (iter == attr_values.end()) {
      return false;
    } else {
      *val = iter->second;
      return true;
    }
  };

  // Makes a copy of all attrs in fdef and substitutes placeholders.
  // After this step, every attr is bound to a concrete value.
  std::vector<InstantiateAttrValueMap> node_attrs;
  if (fdef.node_def_size() > 0) {
    node_attrs.resize(fdef.node_def_size());
    for (int i = 0; i < fdef.node_def_size(); ++i) {
      for (auto attr : fdef.node_def(i).attr()) {
        if (!SubstitutePlaceholders(substitute, &attr.second)) {
          return errors::InvalidArgument("Failed to bind all placeholders in ",
                                         SummarizeAttrValue(attr.second));
        }
        if (!node_attrs[i].insert(attr).second) {
          return errors::Internal("Somehow duplicated: ", attr.first);
        }
      }
      TF_RETURN_IF_ERROR(
          AddDefaultAttrs(fdef.node_def(i).op(), get_function, &node_attrs[i]));
    }

    for (int i = 0; i < fdef.node_def_size(); ++i) {
      s = BuildNodeOutputIndex(fdef.node_def(i), node_attrs[i], get_function,
                               gdef->node_size() + i, &name_info);
      if (!s.ok()) {
        errors::AppendToMessage(&s, "In ", SummarizeNodeDef(fdef.node_def(i)));
        return s;
      }
    }
    // Emits one gdef.node for each fdef.node_def.
    for (int i = 0; i < fdef.node_def_size(); ++i) {
      s = InstantiateNode(fdef.node_def(i), node_attrs[i], get_function,
                          name_info, gdef);
      if (!s.ok()) {
        errors::AppendToMessage(&s, "In ", SummarizeNodeDef(fdef.node_def(i)));
        return s;
      }
    }

    // Emits nodes for the function's return values.
    int ret_index = 0;
    for (const OpDef::ArgDef& ret_def : sig.output_arg()) {
      s = AddReturnNode(ret_def, attr_values, fdef.ret(), name_info, &ret_index,
                        result);
      if (!s.ok()) {
        errors::AppendToMessage(&s, "In function output ", Print(ret_def));
        return s;
      }
    }
  } else {  // TODO(josh11b): Eventually remove this case.
    node_attrs.resize(fdef.node_size());
    for (int i = 0; i < fdef.node_size(); ++i) {
      for (auto attr : fdef.node(i).attr()) {
        if (!SubstitutePlaceholders(substitute, &attr.second)) {
          return errors::InvalidArgument("Failed to bind all placeholders in ",
                                         SummarizeAttrValue(attr.second));
        }
        if (!node_attrs[i].insert(attr).second) {
          return errors::Internal("Somehow duplicated: ", attr.first);
        }
      }
      TF_RETURN_IF_ERROR(
          AddDefaultAttrs(fdef.node(i).op(), get_function, &node_attrs[i]));
    }

    for (int i = 0; i < fdef.node_size(); ++i) {
      s = BuildNodeOutputIndex(fdef.node(i), node_attrs[i], get_function,
                               gdef->node_size() + i, &name_info);
      if (!s.ok()) {
        errors::AppendToMessage(&s, "In ", Print(fdef.node(i)));
        return s;
      }
    }
    // Emits one gdef.node for each fdef.node.
    for (int i = 0; i < fdef.node_size(); ++i) {
      s = InstantiateNode(fdef.node(i), node_attrs[i], get_function, name_info,
                          gdef);
      if (!s.ok()) {
        errors::AppendToMessage(&s, "In ", Print(fdef.node(i)));
        return s;
      }
    }

    // Emits nodes for the function's return values.
    int ret_index = 0;
    for (const OpDef::ArgDef& ret_def : sig.output_arg()) {
      s = AddReturnNode(ret_def, attr_values, name_info, &ret_index, result);
      if (!s.ok()) {
        errors::AppendToMessage(&s, "In function output ", Print(ret_def));
        return s;
      }
    }
  }

  return Status::OK();
}

string DebugString(const FunctionDef& func_def) { return Print(func_def); }

string DebugString(const GraphDef& instantiated_func_def) {
  return Print(instantiated_func_def);
}

string DebugStringWhole(const GraphDef& gdef) {
  string ret;
  for (const auto& fdef : gdef.library().function()) {
    strings::StrAppend(&ret, Print(fdef));
  }
  strings::StrAppend(&ret, "\n");
  for (const auto& ndef : gdef.node()) {
    strings::StrAppend(&ret, Print(ndef), "\n");
  }
  return ret;
}

string Canonicalize(const string& funcname,
                    const InstantiateAttrValueMap& attrs) {
  std::vector<string> entries;
  entries.reserve(attrs.size());
  for (auto p : attrs) {
    entries.push_back(strings::StrCat(p.first, "=", Print(p.second)));
  }
  sort(entries.begin(), entries.end());
  return strings::StrCat(funcname, "[", str_util::Join(entries, ","), "]");
}

FunctionCallFrame::FunctionCallFrame(DataTypeSlice arg_types,
                                     DataTypeSlice ret_types)
    : arg_types_(arg_types.begin(), arg_types.end()),
      ret_types_(ret_types.begin(), ret_types.end()) {
  args_.resize(arg_types_.size());
  rets_.resize(ret_types_.size());
}

FunctionCallFrame::~FunctionCallFrame() {}

Status FunctionCallFrame::SetArgs(gtl::ArraySlice<Tensor> args) {
  // Input type checks.
  if (args.size() != arg_types_.size()) {
    return errors::InvalidArgument("Expects ", arg_types_.size(),
                                   " arguments, but ", args.size(),
                                   " is provided");
  }
  for (size_t i = 0; i < args.size(); ++i) {
    if (arg_types_[i] != args[i].dtype()) {
      return errors::InvalidArgument(
          "Expects arg[", i, "] to be ", DataTypeString(arg_types_[i]), " but ",
          DataTypeString(args[i].dtype()), " is provided");
    }
    args_[i] = args[i];
  }
  return Status::OK();
}

Status FunctionCallFrame::GetRetvals(std::vector<Tensor>* rets) const {
  rets->clear();
  rets->reserve(rets_.size());
  for (size_t i = 0; i < rets_.size(); ++i) {
    auto item = rets_[i];
    if (item.has_val) {
      rets->push_back(item.val);
    } else {
      return errors::Internal("Retval[", i, "] does not have value");
    }
  }
  return Status::OK();
}

Status FunctionCallFrame::GetArg(int index, Tensor* val) const {
  if (index < 0 || static_cast<size_t>(index) >= args_.size()) {
    return errors::InvalidArgument("GetArg ", index, " is not within [0, ",
                                   args_.size(), ")");
  }
  *val = args_[index];
  return Status::OK();
}

Status FunctionCallFrame::SetRetval(int index, const Tensor& val) {
  if (index < 0 || static_cast<size_t>(index) >= rets_.size()) {
    return errors::InvalidArgument("SetRetval ", index, " is not within [0, ",
                                   rets_.size(), ")");
  }
  if (val.dtype() != ret_types_[index]) {
    return errors::InvalidArgument(
        "Expects ret[", index, "] to be ", DataTypeString(ret_types_[index]),
        ", but ", DataTypeString(val.dtype()), " is provided.");
  }
  Retval* item = &rets_[index];
  if (!item->has_val) {
    item->has_val = true;
    item->val = val;
  } else {
    return errors::Internal("Retval[", index, "] has already been set.");
  }
  return Status::OK();
}

FunctionLibraryDefinition::FunctionLibraryDefinition(
    const FunctionLibraryDefinition& other)
    : default_registry_(other.default_registry_), func_grad_(other.func_grad_) {
  for (const auto& it : other.function_defs_) {
    TF_CHECK_OK(AddFunctionDef(it.second->fdef));
  }
}

FunctionLibraryDefinition::FunctionLibraryDefinition(
    const OpRegistryInterface* default_registry,
    const FunctionDefLibrary& def_lib)
    : default_registry_(default_registry),
      function_defs_(def_lib.function_size()) {
  for (const auto& fdef : def_lib.function()) {
    // The latter function definition wins.
    auto& ptr = function_defs_[fdef.signature().name()];
    ptr.reset(new FunctionDefAndOpRegistration(fdef));
  }
  for (const auto& grad : def_lib.gradient()) {
    func_grad_[grad.function_name()] = grad.gradient_func();
  }
}

FunctionLibraryDefinition::~FunctionLibraryDefinition() {}

const FunctionDef* FunctionLibraryDefinition::Find(const string& name) const {
  auto iter = function_defs_.find(name);
  if (iter == function_defs_.end()) {
    return nullptr;
  } else {
    return &iter->second->fdef;
  }
}

Status FunctionLibraryDefinition::AddFunctionDef(const FunctionDef& fdef) {
  auto& ptr = function_defs_[fdef.signature().name()];
  if (ptr != nullptr) {
    return errors::InvalidArgument("Function with name: ",
                                   fdef.signature().name(),
                                   " already exists in function library.");
  }
  ptr.reset(new FunctionDefAndOpRegistration(fdef));
  return Status::OK();
}

string FunctionLibraryDefinition::FindGradient(const string& func) const {
  return gtl::FindWithDefault(func_grad_, func, "");
}

Status FunctionLibraryDefinition::LookUp(
    const string& op, const OpRegistrationData** op_reg_data) const {
  auto iter = function_defs_.find(op);
  if (iter != function_defs_.end()) {
    *op_reg_data = &iter->second->op_registration_data;
    return Status::OK();
  }
  return default_registry_->LookUp(op, op_reg_data);
}

FunctionDefLibrary FunctionLibraryDefinition::ToProto() const {
  FunctionDefLibrary lib;
  for (const auto& f : function_defs_) {
    *lib.add_function() = f.second->fdef;
  }
  for (const auto& g : func_grad_) {
    GradientDef* gd = lib.add_gradient();
    gd->set_function_name(g.first);
    gd->set_gradient_func(g.second);
  }
  return lib;
}

Status InstantiateFunction(const FunctionDef& fdef,
                           InstantiateAttrValueSlice attr_values,
                           GetFunctionSignature get_function,
                           InstantiationResult* result) {
  InstantiateAttrValueMap m;
  for (const auto& aval : attr_values) {
    m.insert({aval.first, aval.second.proto});
  }
  return InstantiateFunction(fdef, m, get_function, result);
}

string Canonicalize(const string& funcname, InstantiateAttrValueSlice attrs) {
  InstantiateAttrValueMap m;
  for (const auto& aval : attrs) {
    m.insert({aval.first, aval.second.proto});
  }
  return Canonicalize(funcname, m);
}

Status FunctionLibraryRuntime::Instantiate(const string& function_name,
                                           InstantiateAttrValueSlice attrs,
                                           Handle* handle) {
  InstantiateAttrValueMap m;
  for (const auto& aval : attrs) {
    m.insert({aval.first, aval.second.proto});
  }
  return Instantiate(function_name, m, handle);
}

void FunctionDefHelper::AttrValueWrapper::InitFromString(StringPiece val) {
  if (val.size() >= 2 && val[0] == '$') {
    proto.set_placeholder(val.data() + 1, val.size() - 1);
  } else {
    SetAttrValue(val, &proto);
  }
}

FunctionDefHelper::AttrValueWrapper FunctionDefHelper::FunctionRef(
    const string& name,
    gtl::ArraySlice<std::pair<string, AttrValueWrapper>> attrs) {
  AttrValueWrapper ret;
  ret.proto.mutable_func()->set_name(name);
  for (const auto& a : attrs) {
    ret.proto.mutable_func()->mutable_attr()->insert({a.first, a.second.proto});
  }
  return ret;
}

FunctionDef::Node FunctionDefHelper::Node::ToProto() const {
  FunctionDef::Node n;
  for (const string& r : this->ret) {
    n.add_ret(r);
  }
  n.set_op(this->op);
  for (const string& a : arg) {
    n.add_arg(a);
  }
  for (const auto& a : this->attr) {
    n.mutable_attr()->insert({a.first, a.second.proto});
  }
  for (const string& d : dep) {
    n.add_dep(d);
  }
  return n;
}

NodeDef FunctionDefHelper::Node::ToNodeDef() const {
  NodeDef n;
  n.set_op(this->op);
  n.set_name(this->ret[0]);
  for (const string& a : arg) {
    n.add_input(a);
  }
  for (const auto& a : this->attr) {
    n.mutable_attr()->insert({a.first, a.second.proto});
  }
  for (const string& d : dep) {
    n.add_input(strings::StrCat("^", d));
  }
  return n;
}

/* static */
FunctionDef FunctionDefHelper::Create(
    const string& function_name, gtl::ArraySlice<string> in_def,
    gtl::ArraySlice<string> out_def, gtl::ArraySlice<string> attr_def,
    gtl::ArraySlice<Node> node_def,
    gtl::ArraySlice<std::pair<string, string>> ret_def) {
  FunctionDef fdef;

  // Signature
  OpDefBuilder b(function_name);
  for (const auto& i : in_def) b.Input(i);
  for (const auto& o : out_def) b.Output(o);
  for (const auto& a : attr_def) b.Attr(a);

  OpRegistrationData op_reg_data;
  TF_CHECK_OK(b.Finalize(&op_reg_data));
  fdef.mutable_signature()->Swap(&op_reg_data.op_def);

  // Function body
  for (const auto& n : node_def) {
    *(fdef.add_node_def()) = n.ToNodeDef();
  }

  // Returns
  for (const auto& r : ret_def) {
    fdef.mutable_ret()->insert({r.first, r.second});
  }
  return fdef;
}

/* static */
FunctionDef FunctionDefHelper::Define(const string& name,
                                      gtl::ArraySlice<string> arg_def,
                                      gtl::ArraySlice<string> ret_def,
                                      gtl::ArraySlice<string> attr_def,
                                      gtl::ArraySlice<Node> node_def) {
  FunctionDef fdef;
  OpDefBuilder b(name);
  for (const auto& a : arg_def) b.Input(a);
  for (const auto& r : ret_def) b.Output(r);
  for (const auto& a : attr_def) b.Attr(a);

  OpRegistrationData op_reg_data;
  TF_CHECK_OK(b.Finalize(&op_reg_data));
  fdef.mutable_signature()->Swap(&op_reg_data.op_def);
  for (const auto& n : node_def) {
    *(fdef.add_node()) = n.ToProto();
  }
  return fdef;
}

FunctionDef FunctionDefHelper::Define(gtl::ArraySlice<string> arg_def,
                                      gtl::ArraySlice<string> ret_def,
                                      gtl::ArraySlice<string> attr_def,
                                      gtl::ArraySlice<Node> node_def) {
  return Define("_", arg_def, ret_def, attr_def, node_def);
}

namespace gradient {

typedef std::unordered_map<string, Creator> OpGradFactory;

OpGradFactory* GetOpGradFactory() {
  static OpGradFactory* factory = new OpGradFactory;
  return factory;
}

bool RegisterOp(const string& op, Creator func) {
  CHECK(GetOpGradFactory()->insert({op, func}).second)
      << "Duplicated gradient for " << op;
  return true;
}

Status GetOpGradientCreator(const string& op, Creator* creator) {
  auto fac = GetOpGradFactory();
  auto iter = fac->find(op);
  if (iter == fac->end()) {
    return errors::NotFound("No gradient defined for op: ", op);
  }
  *creator = iter->second;
  return Status::OK();
}

}  // end namespace gradient

}  // end namespace tensorflow
