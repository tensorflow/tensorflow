/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/node_matchers.h"

#include <utility>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/graph/graph_node_util.h"

namespace tensorflow {
namespace testing {
namespace matchers {
namespace {

using impl::NodeMatcherProperties;
using impl::OutEdge;

string IndentAllButFirstLine(absl::string_view text) {
  std::vector<std::string> lines = absl::StrSplit(text, '\n');
  for (int i = 1; i < lines.size(); i++) {
    lines[i].insert(0, "  ");
  }
  return absl::StrJoin(lines, "\n");
}

template <typename T>
bool CompareTensor(const Tensor& actual, const Tensor& expected,
                   ::testing::MatchResultListener* listener) {
  if (actual.NumElements() != expected.NumElements()) {
    if (listener->IsInterested()) {
      *listener << "\nwas looking for tensor with " << expected.NumElements()
                << " elements, found tensor with " << actual.NumElements()
                << " elements";
      return false;
    }
  }

  for (int64 i = 0, e = actual.NumElements(); i < e; i++) {
    if (actual.flat<T>()(i) != expected.flat<T>()(i)) {
      *listener << "\nmismatch in constant tensor at index " << i
                << " expected = " << expected.flat<T>()(i)
                << " actual = " << actual.flat<T>()(i);
      return false;
    }
  }

  return true;
}

bool MatchAndExplainTensor(const Tensor& tensor, const Tensor& expected_tensor,
                           ::testing::MatchResultListener* listener) {
  if (tensor.dtype() != expected_tensor.dtype()) {
    if (listener->IsInterested()) {
      *listener << "\nexpected tensor of type "
                << DataType_Name(expected_tensor.dtype())
                << " but found one of type " << DataType_Name(tensor.dtype());
      return false;
    }
  }

  switch (tensor.dtype()) {
    case DT_HALF:
      return CompareTensor<Eigen::half>(tensor, expected_tensor, listener);
    case DT_FLOAT:
      return CompareTensor<float>(tensor, expected_tensor, listener);
    case DT_DOUBLE:
      return CompareTensor<double>(tensor, expected_tensor, listener);
    case DT_INT8:
      return CompareTensor<int8>(tensor, expected_tensor, listener);
    case DT_INT16:
      return CompareTensor<int16>(tensor, expected_tensor, listener);
    case DT_INT32:
      return CompareTensor<int32>(tensor, expected_tensor, listener);
    case DT_INT64:
      return CompareTensor<int64>(tensor, expected_tensor, listener);
    case DT_UINT8:
      return CompareTensor<uint8>(tensor, expected_tensor, listener);
    case DT_UINT16:
      return CompareTensor<uint16>(tensor, expected_tensor, listener);
    case DT_UINT32:
      return CompareTensor<uint32>(tensor, expected_tensor, listener);
    case DT_UINT64:
      return CompareTensor<uint64>(tensor, expected_tensor, listener);
    default:
      LOG(FATAL) << "Unsupported dtype "  // Crash ok: testonly.
                 << DataType_Name(tensor.dtype());
  }
}

struct NodeMatcher : public ::testing::MatcherInterface<const Node*> {
  bool MatchAndExplain(
      const Node* node,
      ::testing::MatchResultListener* listener) const override {
    if (op && node->type_string() != *op) {
      if (listener->IsInterested()) {
        *listener << "\nexpected op " << *op << " but found "
                  << node->type_string();
      }
      return false;
    }

    if (assigned_device && node->assigned_device_name() != *assigned_device) {
      if (listener->IsInterested()) {
        *listener << "\nexpected assigned_device " << *assigned_device
                  << " but found \"" << node->assigned_device_name() << "\"";
      }
      return false;
    }

    if (name && node->name() != *name) {
      if (listener->IsInterested()) {
        *listener << "\nexpected name " << *name << " but found "
                  << node->name();
      }
      return false;
    }

    if (constant_value) {
      const TensorProto* proto = nullptr;
      if (!TryGetNodeAttr(node->def(), "value", &proto)) {
        if (listener->IsInterested()) {
          *listener << "\ncould not find \"value\" attribute in node";
        }
        return false;
      }

      Tensor tensor(proto->dtype());
      if (!tensor.FromProto(*proto)) {
        if (listener->IsInterested()) {
          *listener << "\ncould not convert TensorProto in \"value\" attribute "
                       "to Tensor";
        }
        return false;
      }

      if (!MatchAndExplainTensor(/*tensor=*/tensor,
                                 /*expected_tensor=*/*constant_value,
                                 listener)) {
        return false;
      }
    }

    if (input_matchers) {
      if (input_matchers->size() != node->num_inputs()) {
        if (listener->IsInterested()) {
          *listener << "\nexpected " << input_matchers->size()
                    << " inputs but node has " << node->num_inputs();
        }
        return false;
      }

      for (int input_idx = 0, e = input_matchers->size(); input_idx < e;
           input_idx++) {
        if (!MatchAndExplainInput(node, input_idx, listener)) {
          return false;
        }
      }
    }

    std::vector<const Node*> control_deps;
    for (const Edge* e : node->in_edges()) {
      if (e->IsControlEdge()) {
        control_deps.push_back(e->src());
      }
    }

    ::testing::StringMatchResultListener inner_listener;
    if (control_dep_set &&
        !control_dep_set->MatchAndExplain(control_deps, &inner_listener)) {
      if (listener->IsInterested()) {
        string explanation = inner_listener.str();
        if (!explanation.empty()) {
          explanation = absl::StrCat(", ", explanation, ",");
        }
        *listener << "ctrl_deps" << explanation << " does not match expected: ";
        control_dep_set->DescribeTo(listener->stream());
      }
      return false;
    }

    const AttrValueMap attr_value_map = node->def().attr();
    for (const auto& attr_kv_pair : attrs) {
      auto it = attr_value_map.find(attr_kv_pair.first);
      if (it == attr_value_map.end()) {
        if (listener->IsInterested()) {
          *listener << "did not find attribute named \"" << attr_kv_pair.first
                    << "\" in node";
        }
        return false;
      }
      if (attr_kv_pair.second &&
          !AreAttrValuesEqual(it->second, *attr_kv_pair.second)) {
        if (listener->IsInterested()) {
          *listener << "attribute named " << attr_kv_pair.first
                    << " does not match value; expected: \""
                    << SummarizeAttrValue(*attr_kv_pair.second)
                    << "\", found: \"" << SummarizeAttrValue(it->second)
                    << "\"";
        }
        return false;
      }
    }

    return true;
  }

  void DescribeTo(::std::ostream* os) const override {
    std::vector<string> predicates;

    if (name) {
      predicates.push_back(absl::StrCat("name: ", *name));
    }

    if (op) {
      predicates.push_back(absl::StrCat("op: ", *op));
    }

    if (assigned_device) {
      predicates.push_back(absl::StrCat("assigned device: ", *assigned_device));
    }

    bool printed_something = !predicates.empty();

    *os << absl::StrJoin(predicates, ", ");

    if (constant_value) {
      printed_something = true;
      *os << "constant value: " << constant_value->DebugString();
    }

    if (input_matchers) {
      if (!input_matchers->empty()) {
        printed_something = true;
        *os << " with " << (input_matchers->size() == 1 ? "only " : "")
            << "input" << (input_matchers->size() == 1 ? "" : "s") << " ";
      }

      if (input_matchers->size() == 1) {
        ::std::stringstream ss;
        input_matchers->front().DescribeTo(&ss);
        printed_something = true;
        *os << "matching " << ss.str();
      } else {
        int edge_idx = 0;
        for (const ::testing::Matcher<OutEdge>& matcher : (*input_matchers)) {
          *os << "\n  [" << edge_idx << "] matching (";
          ::std::stringstream ss;
          matcher.DescribeTo(&ss);
          printed_something = true;
          *os << IndentAllButFirstLine(ss.str());
          *os << ")";
          edge_idx++;
        }
      }
    }

    if (control_dep_set) {
      printed_something = true;
      *os << " and control deps ";
      control_dep_set->DescribeTo(os);
    }

    if (!attrs.empty()) {
      printed_something = true;
      std::vector<string> attrs_str;
      absl::c_transform(
          attrs, std::back_inserter(attrs_str),
          [](const std::pair<string, absl::optional<AttrValue>>& attr_kv_pair) {
            return absl::StrCat(attr_kv_pair.first, "->",
                                attr_kv_pair.second
                                    ? SummarizeAttrValue(*attr_kv_pair.second)
                                    : "*");
          });
      *os << " and attr values matching [" << absl::StrJoin(attrs_str, ", ")
          << "]";
    }

    if (!printed_something) {
      *os << "is any node";
    }
  }

  bool MatchAndExplainInput(const Node* node, int input_idx,
                            ::testing::MatchResultListener* listener) const {
    const Edge* edge;
    if (!node->input_edge(input_idx, &edge).ok()) {
      if (listener->IsInterested()) {
        *listener << "\ncould not find incoming edge for input " << input_idx;
      }
      return false;
    }

    ::testing::StringMatchResultListener inner_listener;
    OutEdge input = {edge->src(), edge->src_output()};
    if ((*input_matchers)[input_idx].MatchAndExplain(input, &inner_listener)) {
      return true;
    }

    if (listener->IsInterested()) {
      *listener << "\ninput " << input_idx << " does not match expected:\n";
      (*input_matchers)[input_idx].DescribeTo(listener->stream());
      string explanation = inner_listener.str();
      if (!explanation.empty()) {
        *listener << ", " << explanation;
      }
    }
    return false;
  }

  absl::optional<string> op;
  absl::optional<string> name;
  absl::optional<string> assigned_device;
  absl::optional<Tensor> constant_value;
  absl::optional<std::vector<::testing::Matcher<OutEdge>>> input_matchers;
  absl::optional<::testing::Matcher<absl::Span<const Node* const>>>
      control_dep_set;
  std::map<string, absl::optional<AttrValue>> attrs;
};

// Matches a dst and dst_output on an input edge.  Today we only use this with
// dst_output=0 but we will eventually need to support multi-output operations.
class OutEdgeMatcher : public ::testing::MatcherInterface<OutEdge> {
 public:
  OutEdgeMatcher(::testing::Matcher<const Node*> src_matcher, int src_oidx)
      : src_matcher_(std::move(src_matcher)), src_oidx_(src_oidx) {}

  bool MatchAndExplain(
      OutEdge out_edge,
      ::testing::MatchResultListener* listener) const override {
    ::testing::StringMatchResultListener inner_listener;
    if (!src_matcher_.MatchAndExplain(out_edge.first, &inner_listener)) {
      if (listener->IsInterested()) {
        *listener << "\nsource does not match expected ";
        src_matcher_.DescribeTo(listener->stream());
        string explanation = inner_listener.str();
        if (!explanation.empty()) {
          *listener << "\n\t" << explanation;
        }
      }
      return false;
    }
    if (out_edge.second != src_oidx_) {
      if (listener->IsInterested()) {
        *listener << "\nexpected output slot to be " << src_oidx_
                  << " but found " << out_edge.second;
      }
      return false;
    }

    return true;
  }

  void DescribeTo(::std::ostream* os) const override {
    if (src_oidx_) {
      *os << "output slot: " << src_oidx_ << ", source: (";
    }

    src_matcher_.DescribeTo(os);

    if (src_oidx_) {
      *os << ")";
    }
  }

 private:
  ::testing::Matcher<const Node*> src_matcher_;
  int src_oidx_;
};
}  // namespace

::testing::Matcher<const Node*> impl::NodeWith(
    absl::Span<const NodeMatcherProperties> props) {
  NodeMatcher* matcher = new NodeMatcher();
  for (const NodeMatcherProperties& prop : props) {
    if (prop.name()) {
      DCHECK(!matcher->name);
      matcher->name = prop.name();
    }

    if (prop.op()) {
      DCHECK(!matcher->op);
      matcher->op = prop.op();
    }

    if (prop.constant_value()) {
      DCHECK(!matcher->constant_value);
      matcher->constant_value = prop.constant_value();
    }

    if (prop.assigned_device()) {
      DCHECK(!matcher->assigned_device);
      matcher->assigned_device = prop.assigned_device();
    }

    if (prop.inputs()) {
      DCHECK(!matcher->input_matchers);
      matcher->input_matchers = *prop.inputs();
    }

    if (prop.control_deps()) {
      DCHECK(!matcher->control_dep_set);
      matcher->control_dep_set =
          ::testing::UnorderedElementsAreArray(*prop.control_deps());
    }

    if (prop.attr()) {
      auto insert_result = matcher->attrs.insert(*prop.attr());
      DCHECK(insert_result.second);
    }
  }

  return ::testing::MakeMatcher(matcher);
}

impl::NodeMatcherProperties Name(string name) {
  impl::NodeMatcherProperties props;
  props.set_name(std::move(name));
  return props;
}

// Matches a node with op `op`.
impl::NodeMatcherProperties Op(string op) {
  impl::NodeMatcherProperties props;
  props.set_op(std::move(op));
  return props;
}

// Matches a node with assigned device `assigned_device`.
impl::NodeMatcherProperties AssignedDevice(string assigned_device) {
  impl::NodeMatcherProperties props;
  props.set_assigned_device(std::move(assigned_device));
  return props;
}

impl::NodeMatcherProperties impl::Inputs(
    absl::Span<const ::testing::Matcher<OutEdge>> inputs) {
  std::vector<::testing::Matcher<OutEdge>> inputs_vector;
  absl::c_copy(inputs, std::back_inserter(inputs_vector));

  impl::NodeMatcherProperties props;
  props.set_inputs(std::move(inputs_vector));
  return props;
}

impl::NodeMatcherProperties impl::CtrlDeps(
    absl::Span<const ::testing::Matcher<const Node*>> control_deps) {
  std::vector<::testing::Matcher<const Node*>> control_deps_vector;
  absl::c_copy(control_deps, std::back_inserter(control_deps_vector));

  impl::NodeMatcherProperties props;
  props.set_control_deps(std::move(control_deps_vector));
  return props;
}

std::pair<string, AttrValue> impl::AttrLiteralHelper(
    const std::pair<string, bool>& bool_attr) {
  AttrValue attr_value;
  attr_value.set_b(bool_attr.second);
  return {bool_attr.first, attr_value};
}

std::pair<string, AttrValue> impl::AttrLiteralHelper(
    const std::pair<string, absl::Span<const int>>& int_list_attr) {
  AttrValue attr_value;
  AttrValue::ListValue* list = attr_value.mutable_list();
  for (int i : int_list_attr.second) {
    list->add_i(i);
  }
  return {int_list_attr.first, attr_value};
}

std::pair<string, AttrValue> impl::AttrLiteralHelper(
    const std::pair<string, absl::Span<const string>>& string_list_attr) {
  AttrValue attr_value;
  AttrValue::ListValue* list = attr_value.mutable_list();
  for (const string& s : string_list_attr.second) {
    list->add_s(s);
  }
  return {string_list_attr.first, attr_value};
}

impl::NodeMatcherProperties impl::Attr(std::pair<string, AttrValue> attr) {
  impl::NodeMatcherProperties props;
  props.set_attr(std::move(attr));
  return props;
}

impl::NodeMatcherProperties impl::Attr(string name) {
  impl::NodeMatcherProperties props;
  props.set_attr({std::move(name), absl::nullopt});
  return props;
}

NodeMatcherProperties ConstantValue(
    const ::tensorflow::Input::Initializer& val) {
  TF_CHECK_OK(val.status);
  NodeMatcherProperties props;
  props.set_constant_value(val.tensor);
  return props;
}

::testing::Matcher<impl::OutEdge> Const(
    const ::tensorflow::Input::Initializer& val) {
  return Out(NodeWith(ConstantValue(val)));
}
::testing::Matcher<impl::OutEdge> Out(
    int oidx, ::testing::Matcher<const Node*> node_matcher) {
  return ::testing::MakeMatcher(new OutEdgeMatcher(node_matcher, oidx));
}
}  // namespace matchers

Node* FindNodeByName(Graph* g, absl::string_view name) {
  for (Node* n : g->nodes()) {
    if (n->name() == name) {
      return n;
    }
  }

  return nullptr;
}
}  // namespace testing

void PrintTo(const Node* n, ::std::ostream* os) { *os << SummarizeNode(*n); }
void PrintTo(Node* n, ::std::ostream* os) { *os << SummarizeNode(*n); }
}  // namespace tensorflow
