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

// Provides a set of matchers for tensorflow nodes.
//
// Example usage:
//
//  tensorflow::Node* node = ...;
//  EXPECT_THAT(node, NodeWith(Name("name"), Op("op"),
//                             Inputs(Out(3, NodeWith(Name("input"))))))
//
// Matchable node properties (the expressions that go inside NodeWith(...))
// are:
//
//  - Name(string): matches the node name exactly.  We will probably need to
//    have this take a string matcher soon in the future.
//
//  - Op(string): matches the op exactly.
//
//  - AssignedDevice(string): matches the assigned device exactly.
//
//  - Inputs(<ordered list>): matches the list of non-control inputs to the node
//    exactly (i.e. does not match a suffix or a prefix) where each element
//    matches an output of a node (see Out(idx, node) below).
//
//  - CtrlDeps(<unordered list>): matches the list of control dependences on the
//    node exactly but in any order.
//
//  - ConstantValue(tensorflow::Input::Initializer init): matches a Const node
//    with the constant value `init`.  Implies Op("Const").
//
//  - Attr(name, value): Matches a single attribute with name `name` and value
//    `value`.  Right now only boolean values are supported.
//
// Overlapping node properties may not be repeated in a single NodeWith(...)
// matcher.  E.g. NodeWith(Op("Foo"), Op("Bar")) will CHECK-fail.  Since
// ConstantValue implies Op("Const"), a single NodeWith matcher can't have both
// ConstantValue(...) and Op(...).  Multiple Attr() values can be combined as
// long as the attribute names are different.
//
// Out(idx, node) matches the `idx`'th output of a node that matches `node`.

#ifndef TENSORFLOW_COMPILER_JIT_NODE_MATCHERS_H_
#define TENSORFLOW_COMPILER_JIT_NODE_MATCHERS_H_

#include <array>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
namespace testing {
namespace matchers {

namespace impl {

using OutEdge = std::pair<const Node*, int>;

// -----------------------------------------------------------------------------
// Implementation details.

// Properties that we match on for a particular Node.  If a particular property
// is nullopt then any value for it is allowed.
class NodeMatcherProperties {
 public:
  using NodeSeqMatcher = std::vector<::testing::Matcher<const Node*>>;
  using InputSeqMatcher = std::vector<::testing::Matcher<OutEdge>>;
  using AttrKeyValuePair = std::pair<string, absl::optional<AttrValue>>;

  const absl::optional<string>& name() const { return name_; }
  const absl::optional<string>& op() const { return op_; }
  const absl::optional<string>& assigned_device() const {
    return assigned_device_;
  }
  const absl::optional<Tensor>& constant_value() const {
    return constant_value_;
  }
  const absl::optional<InputSeqMatcher>& inputs() const {
    return input_matchers_;
  }
  const absl::optional<NodeSeqMatcher>& control_deps() const {
    return control_deps_;
  }
  const absl::optional<AttrKeyValuePair>& attr() const { return attr_; }

  void set_name(string name) {
    DCHECK(IsEmpty());
    name_ = std::move(name);
  }

  void set_op(string op) {
    DCHECK(IsEmpty());
    op_ = std::move(op);
  }

  void set_assigned_device(string assigned_device) {
    DCHECK(IsEmpty());
    assigned_device_ = std::move(assigned_device);
  }

  void set_constant_value(Tensor constant_value) {
    DCHECK(IsEmpty());
    constant_value_ = std::move(constant_value);
    op_ = "Const";
  }

  void set_inputs(InputSeqMatcher inputs) {
    DCHECK(IsEmpty());
    input_matchers_ = std::move(inputs);
  }

  void set_control_deps(NodeSeqMatcher control_deps) {
    DCHECK(IsEmpty());
    control_deps_ = std::move(control_deps);
  }

  void set_attr(AttrKeyValuePair attr) {
    DCHECK(IsEmpty());
    attr_ = std::move(attr);
  }

  bool IsEmpty() const {
    return !name().has_value() && !op().has_value() && !inputs().has_value() &&
           !control_deps().has_value() && !attr().has_value();
  }

 private:
  absl::optional<string> name_;
  absl::optional<string> op_;
  absl::optional<string> assigned_device_;
  absl::optional<Tensor> constant_value_;
  absl::optional<InputSeqMatcher> input_matchers_;
  absl::optional<NodeSeqMatcher> control_deps_;
  absl::optional<AttrKeyValuePair> attr_;
};

::testing::Matcher<const Node*> NodeWith(
    absl::Span<const NodeMatcherProperties> props);

impl::NodeMatcherProperties Inputs(
    absl::Span<const ::testing::Matcher<OutEdge>> inputs);

impl::NodeMatcherProperties CtrlDeps(
    absl::Span<const ::testing::Matcher<const Node*>> control_deps);

impl::NodeMatcherProperties Attr(std::pair<string, AttrValue> attrs);
impl::NodeMatcherProperties Attr(string name);

std::pair<string, AttrValue> AttrLiteralHelper(
    const std::pair<string, bool>& bool_attr);

std::pair<string, AttrValue> AttrLiteralHelper(
    const std::pair<string, absl::Span<const int>>& int_list_attr);

std::pair<string, AttrValue> AttrLiteralHelper(
    const std::pair<string, absl::Span<const string>>& string_list_attr);
}  // namespace impl

// -----------------------------------------------------------------------------
// Public interface.

// Matches a node with name `name`.
impl::NodeMatcherProperties Name(string name);

// Matches a node with op `op`.
impl::NodeMatcherProperties Op(string op);

// Matches a node with assigned device `assigned_device`.
impl::NodeMatcherProperties AssignedDevice(string assigned_device);

// Matches a node with a boolean typed attribute named `name` and with value
// `value`.
template <typename ValueTy>
impl::NodeMatcherProperties Attr(const string& name, ValueTy value) {
  return impl::Attr({impl::AttrLiteralHelper({name, value})});
}

inline impl::NodeMatcherProperties Attr(const string& name) {
  return impl::Attr(name);
}

// Matches a node with inputs `inputs`.
//
// `inputs` are ordered; `inputs`[i] must match input i.
template <typename... Ts>
impl::NodeMatcherProperties Inputs(Ts... inputs) {
  return impl::Inputs({inputs...});
}

// Matches the `idx`'th output of a node that matches `node`.
::testing::Matcher<impl::OutEdge> Out(int oidx,
                                      ::testing::Matcher<const Node*> node);

// Matches the first output of a node that matches `node`.
inline ::testing::Matcher<impl::OutEdge> Out(
    ::testing::Matcher<const Node*> node) {
  return Out(0, node);
}

// Matches a node with control dependences `control_deps`.
//
// `control_deps` are unordered and will match the control deps of a node in any
// order.
template <typename... Ts>
impl::NodeMatcherProperties CtrlDeps(Ts... control_deps) {
  return impl::CtrlDeps({control_deps...});
}

// Matches a constant node with value `val`.
impl::NodeMatcherProperties ConstantValue(
    const ::tensorflow::Input::Initializer& val);

// The main gmock matcher.  See file comment for example usage.
template <typename... Ts>
::testing::Matcher<const Node*> NodeWith(Ts... args) {
  std::array<impl::NodeMatcherProperties, sizeof...(Ts)> array = {args...};
  return impl::NodeWith(array);
}

::testing::Matcher<impl::OutEdge> Const(
    const ::tensorflow::Input::Initializer& val);
}  // namespace matchers

// If `g` has a node named `name` returns it, otherwise returns null.
Node* FindNodeByName(Graph* g, absl::string_view name);
}  // namespace testing

void PrintTo(const Node* n, ::std::ostream* os);
void PrintTo(Node* n, ::std::ostream* os);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_NODE_MATCHERS_H_
