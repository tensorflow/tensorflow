#include "tensorflow/core/framework/function.h"

#include <unordered_set>

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace tensorflow {

REGISTER_OP("_Arg")
    .Output("output: T")
    .Attr("T: type")
    .Attr("index: int >= 0")
    .Doc(R"doc(
A graph node which represents an argument to a function.

output: The argument.
index: This argument is the index-th argument of the function.
)doc");

REGISTER_OP("_Retval")
    .Input("input: T")
    .Attr("T: type")
    .Attr("index: int >= 0")
    .Doc(R"doc(
A graph node which represents a return value of a function.

input: The return value.
index: This return value is the index-th return value of the function.
)doc");

REGISTER_OP("_ListToArray")
    .Input("input: Tin")
    .Output("output: N * T")
    .Attr("Tin: list(type)")
    .Attr("T: type")
    .Attr("N: int >= 1")
    .Doc(R"doc(
Converts a list of tensors to an array of tensors.
)doc");

REGISTER_OP("_ArrayToList")
    .Input("input: N * T")
    .Output("output: out_types")
    .Attr("T: type")
    .Attr("N: int >= 1")
    .Attr("out_types: list(type)")
    .Doc(R"doc(
Converts an array of tensors to a list of tensors.
)doc");

namespace {

// Extracts the actual type from "attr_values" based on its definition
// "arg_def".
Status ArgNumType(const InstantiateAttrValueMap& attrs,
                  const OpDef::ArgDef& arg_def, int* num, DataType* dtype) {
  if (!arg_def.type_list_attr().empty()) {
    return errors::Unimplemented("type_list is not supported.");
  }

  if (arg_def.number_attr().empty()) {
    *num = 1;
  } else {
    const AttrValue* v = gtl::FindOrNull(attrs, arg_def.number_attr());
    if (v == nullptr) {
      return errors::NotFound("type attr not found: ", arg_def.type_attr());
    }
    *num = v->i();
  }

  if (arg_def.type() != DT_INVALID) {
    *dtype = arg_def.type();
  } else if (arg_def.type_attr().empty()) {
    *dtype = DT_INVALID;
  } else {
    const AttrValue* v = gtl::FindOrNull(attrs, arg_def.type_attr());
    if (v == nullptr) {
      return errors::NotFound("type attr not found: ", arg_def.type_attr());
    }
    *dtype = v->type();
  }
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
    if (attr_values.find(a.name()) == attr_values.end()) {
      return errors::NotFound("Attr ", a.name(), " is not found.");
    }
  }

  for (const auto& p : attr_values) {
    if (HasPlaceHolder(p.second)) {
      return errors::InvalidArgument(p.first,
                                     " in attr_values is still a placeholder.");
    }
  }

  return Status::OK();
}

// We build a small index for all names that can be used as a node's
// input arguments.
//
// If is_func_arg is true, the name is a function's argument.  In
// this case, the produced graph def has gdef.node[nid ... nid +
// num).
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
  int num;
  DataType dtype;
};
typedef std::unordered_map<string, NameInfoItem> NameInfoIndex;

Status BuildInputArgIndex(const OpDef::ArgDef& arg_def,
                          const InstantiateAttrValueMap& attr_values,
                          NameInfoIndex* name_info,
                          InstantiationResult* result) {
  int num;
  DataType dtype;
  TF_RETURN_IF_ERROR(ArgNumType(attr_values, arg_def, &num, &dtype));
  CHECK_GE(num, 1);
  GraphDef* gdef = &result->gdef;
  int arg_index = gdef->node_size();
  if (!name_info->insert({arg_def.name(), {true, arg_index, 0, num, dtype}})
           .second) {
    return errors::InvalidArgument("Duplicated arg name.");
  }
  // Creates "num" nodes in the gdef.
  for (int i = 0; i < num; ++i) {
    DCHECK_EQ(arg_index, gdef->node_size());
    NodeDef* gnode = gdef->add_node();
    gnode->set_name(Name(arg_index));
    gnode->set_op("_Arg");
    AddAttr("T", dtype, gnode);
    AddAttr("index", arg_index, gnode);
    result->arg_types.push_back(dtype);
    ++arg_index;
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
    if (!name_info->insert({node.ret(0), {false, arg_index, 0, 0, DT_INVALID}})
             .second) {
      return errors::InvalidArgument("Duplicated ret name.");
    }
    return Status::OK();
  }

  // When the signature says the last return value is of list(type),
  // i.e., it's variadic, we need to consult
  // attrs[last_retval.type_list_attr] to determine for the last arg
  //   * the actual number of outputs;
  //   * the actual data type of outputs.
  const int num_retval = node_sig->output_arg_size();
  const OpDef::ArgDef& last_retval = node_sig->output_arg(num_retval - 1);
  const bool last_retval_is_typelist = !last_retval.type_list_attr().empty();
  if (!last_retval_is_typelist && (node.ret_size() != num_retval)) {
    return errors::InvalidArgument("Malformed function node (#ret).");
  }
  int start = 0;
  const int num_fixed_size_retval =
      last_retval_is_typelist ? num_retval - 1 : num_retval;
  for (int i = 0; i < num_fixed_size_retval; ++i) {
    int num;
    DataType dtype;
    TF_RETURN_IF_ERROR(
        ArgNumType(attrs, node_sig->output_arg(i), &num, &dtype));
    if (!name_info->insert({node.ret(i), {false, arg_index, start, num, dtype}})
             .second) {
      return errors::InvalidArgument("Duplicated ret name.");
    }
    start += num;
  }
  if (last_retval_is_typelist) {
    const AttrValue* typelist =
        gtl::FindOrNull(attrs, last_retval.type_list_attr());
    if (typelist == nullptr) {
      return errors::InvalidArgument("Missing attr ",
                                     last_retval.type_list_attr(), ".");
    }
    if (num_fixed_size_retval + typelist->list().type_size() !=
        node.ret_size()) {
      return errors::InvalidArgument("Wrong #ret: ", num_fixed_size_retval, " ",
                                     typelist->list().type_size(), " ",
                                     node.ret_size(), ".");
    }
    for (int i = 0; i < typelist->list().type_size(); ++i) {
      if (!name_info->insert({node.ret(i),
                              {false, arg_index, start, 1,
                               typelist->list().type(i)}})
               .second) {
        return errors::InvalidArgument("Duplicated ret name.");
      }
      ++start;
    }
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
  //
  // When the signature says the last argument is of list(type),
  // i.e., it's variadic, we need to consult
  // attrs[last_arg.type_list_attr] to determine for the last arg
  //   * the number of arguments;
  //   * the data types of arguments.
  const int num_arg = fnode_sig->input_arg_size();
  bool last_arg_is_typelist = false;
  if (num_arg > 0 &&
      !fnode_sig->input_arg(num_arg - 1).type_list_attr().empty()) {
    last_arg_is_typelist = true;
  }
  if (!last_arg_is_typelist && (fnode.arg_size() != num_arg)) {
    return errors::InvalidArgument("arg.size != sig.arg.size.");
  }
  const int num_fixed_size_args = last_arg_is_typelist ? num_arg - 1 : num_arg;
  for (int i = 0; i < num_fixed_size_args; ++i) {
    int num;
    DataType dtype;
    TF_RETURN_IF_ERROR(
        ArgNumType(attrs, fnode_sig->input_arg(i), &num, &dtype));
    const NameInfoItem* item = gtl::FindOrNull(name_info, fnode.arg(i));
    if (item == nullptr) {
      return errors::InvalidArgument("arg[", i, "] is not found: ",
                                     fnode.ShortDebugString());
    }
    if (num != item->num || dtype != item->dtype) {
      return errors::InvalidArgument("Invalid arg(", i, ") for function arg: ",
                                     " ", num, "/", dtype, " vs. ", item->num,
                                     "/", item->dtype, ".");
    }
    for (int j = 0; j < num; ++j) {
      if (item->is_func_arg) {
        gnode->add_input(Name(item->nid + j));
      } else {
        gnode->add_input(Name(item->nid, item->idx + j));
      }
    }
  }
  if (last_arg_is_typelist) {
    AttrValue typelist;
    for (int i = num_fixed_size_args; i < fnode.arg_size(); ++i) {
      const NameInfoItem* item = gtl::FindOrNull(name_info, fnode.arg(i));
      if (item == nullptr) {
        return errors::InvalidArgument("arg[", i, "] is not found.");
      }
      for (int j = 0; j < item->num; ++j) {
        if (item->is_func_arg) {
          gnode->add_input(Name(item->nid + j));
        } else {
          gnode->add_input(Name(item->nid, item->idx + j));
        }
        typelist.mutable_list()->add_type(item->dtype);
      }
    }

    // 'typelist' is inferred from the inputs' data types.
    const auto& last_arg = fnode_sig->input_arg(num_arg - 1);
    gnode->mutable_attr()->insert({last_arg.type_list_attr(), typelist});
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

Status AddReturnNode(const OpDef::ArgDef& ret_def,
                     const InstantiateAttrValueMap& attrs,
                     const NameInfoIndex& name_info, int* ret_index,
                     InstantiationResult* result) {
  int num;
  DataType dtype;
  TF_RETURN_IF_ERROR(ArgNumType(attrs, ret_def, &num, &dtype));
  CHECK_GE(num, 1);
  const NameInfoItem* item = gtl::FindOrNull(name_info, ret_def.name());
  if (item == nullptr) {
    return errors::InvalidArgument("ret is not found.");
  }
  if (num != item->num || dtype != item->dtype) {
    return errors::InvalidArgument("Invalid ret name.");
  }
  GraphDef* gdef = &result->gdef;
  for (int i = 0; i < num; ++i) {
    NodeDef* gnode = gdef->add_node();
    gnode->set_name(Name(gdef->node_size() - 1));
    gnode->set_op("_Retval");
    gnode->add_input(Name(item->nid, item->idx + i));
    AddAttr("T", dtype, gnode);
    AddAttr("index", (*ret_index)++, gnode);
    result->ret_types.push_back(dtype);
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
  for (const auto& n : fdef.node()) {
    strings::StrAppend(&out, "  ", Print(n), "\n");
  }
  strings::StrAppend(&out, "}\n");
  return out;
}

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

}  // end namespace

Status InstantiateFunction(const FunctionDef& fdef,
                           const InstantiateAttrValueMap& attr_values,
                           GetFunctionSignature get_function,
                           InstantiationResult* result) {
  const OpDef& sig = fdef.signature();
  GraphDef* gdef = &result->gdef;
  gdef->Clear();

  TF_RETURN_IF_ERROR(ValidateSignatureWithAttrs(sig, attr_values));

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
  node_attrs.resize(fdef.node_size());
  for (int i = 0; i < fdef.node_size(); ++i) {
    for (auto attr : fdef.node(i).attr()) {
      if (!SubstitutePlaceholders(substitute, &attr.second)) {
        return errors::InvalidArgument("Failed to bind all placeholders in ",
                                       SummarizeAttrValue(attr.second));
      }
      CHECK(node_attrs[i].insert(attr).second);
    }
  }

  NameInfoIndex name_info;
  Status s;
  for (const OpDef::ArgDef& arg_def : sig.input_arg()) {
    s = BuildInputArgIndex(arg_def, attr_values, &name_info, result);
    if (!s.ok()) {
      errors::AppendToMessage(&s, " In ", Print(arg_def));
      return s;
    }
  }
  for (int i = 0; i < fdef.node_size(); ++i) {
    s = BuildNodeOutputIndex(fdef.node(i), node_attrs[i], get_function,
                             gdef->node_size() + i, &name_info);
    if (!s.ok()) {
      errors::AppendToMessage(&s, " In ", Print(fdef.node(i)));
      return s;
    }
  }

  // Emits one gdef.node for each fdef.node.
  for (int i = 0; i < fdef.node_size(); ++i) {
    s = InstantiateNode(fdef.node(i), node_attrs[i], get_function, name_info,
                        gdef);
    if (!s.ok()) {
      errors::AppendToMessage(&s, " In ", Print(fdef.node(i)));
      return s;
    }
  }

  // Emits nodes for the function's return values.
  int ret_index = 0;
  for (const OpDef::ArgDef& ret_def : sig.output_arg()) {
    s = AddReturnNode(ret_def, attr_values, name_info, &ret_index, result);
    if (!s.ok()) {
      errors::AppendToMessage(&s, " In ", Print(ret_def));
      return s;
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
  for (auto fdef : gdef.library().function()) {
    strings::StrAppend(&ret, Print(fdef));
  }
  strings::StrAppend(&ret, "\n");
  for (auto ndef : gdef.node()) {
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
    return errors::OutOfRange("GetArg ", index, " is not within [0, ",
                              args_.size(), ")");
  }
  *val = args_[index];
  return Status::OK();
}

Status FunctionCallFrame::SetRetval(int index, const Tensor& val) {
  if (index < 0 || static_cast<size_t>(index) >= rets_.size()) {
    return errors::OutOfRange("SetRetval ", index, " is not within [0, ",
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
    const FunctionDefLibrary& def_lib)
    : function_defs_(def_lib.function_size()) {
  for (auto fdef : def_lib.function()) {
    // The latter function definition wins.
    function_defs_[fdef.signature().name()] = fdef;
  }
}

FunctionLibraryDefinition::~FunctionLibraryDefinition() {}

const FunctionDef* FunctionLibraryDefinition::Find(const string& name) const {
  auto iter = function_defs_.find(name);
  if (iter == function_defs_.end()) {
    return nullptr;
  } else {
    return &iter->second;
  }
}

const OpDef* FunctionLibraryDefinition::LookUp(const string& op,
                                               Status* status) const {
  auto fdef = Find(op);
  if (fdef != nullptr) {
    return &(fdef->signature());
  }
  return OpRegistry::Global()->LookUp(op, status);
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

/*  static */
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
  TF_CHECK_OK(b.Finalize(fdef.mutable_signature()));
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
