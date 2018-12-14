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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PATTERN_MATCHER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PATTERN_MATCHER_H_

#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/utility/utility.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

// A pattern matcher for HloInstructions, Shapes, and Layouts.
//
// The Match function's first argument must be HloInstruction*, Shape*, or
// Layout*. The second argument is a pattern that will be matched against the
// first argument, as described below.
//
// Patterns are constructed using the match::Op, match::Shape, or match::Layout
// functions. By default, the returned patterns will match any HloInstruction,
// Shape, or Layout, respectively. However the match can be made more specific
// by using the pattern's modifier methods, for example:
//
//   match::Op().WithOpcode(HloOpcode::kAdd).WithOperand(
//     0, match::Op().WithOpcode(HloOpcode::kConstant))
//
// This pattern will match Add instructions whose first operand is a constant.
//
// Each pattern type has the following modifiers, which are described where
// nontrivial.
//
//   Op():
//     - Is: is the given HloInstruction* (i.e. pointer equality)
//     - WithName
//     - WithOpcode
//     - WithoutOpcode: anything other than the given opcode
//     - WithShape: instr's shape matches the given pattern
//     - WithShapeEqualTo: instr's shape is equal to the given Shape
//     - WithShapeCompatibleTo: instr's shape is compatible with the given Shape
//     - WithNumOperands
//     - WithOperand: operand at the given index matches the given pattern
//     - IsConstant
//     - IsNonConstant
//     - IsConstantScalar/IsEffectiveConstantScalar: Optionally accepts a value,
//       e.g. IsConstantScalar() or IsConstantScalar(42).
//     - WithFusionKind
//     - WithTupleIndex: get-tuple-element operations with the given tuple index
//     - WithOneUse: Instruction is used as an operand exactly once.
//     - WithOneUser: Instruction is used by exactly one other instruction, but
//       is possibly used more than once as an operand (e.g. multiply(x,x)).
//
//   Shape():
//     - EqualTo
//     - CompatibleTo
//     - IsScalar/IsEffectiveScalar/IsArray/IsTuple
//     - IsDenseArray/IsSparseArray
//     - WithLayout: layout shape's layout matches the given pattern (e.g.
//       Layout().WithDenseFormat())
//     - WithLayoutEqualTo: shape's layout equals the argument (i.e. another
//       Layout, but not the result of Layout().foo())
//     - WithSubshape: shape is a tuple whose subshape matches the given pattern
//       (e.g. Shape().IsScalar()).
//     - WithSubshapeEqualTo: shape is a tuple with a subshape equal to the arg
//       (i.e. another Shape, but not the result of Shape().foo())
//     - WithElementType: shape is an array/scalar with the given elem type
//     - WithRank: shape is an array/scalar with the given rank
//
//  Layout():
//     - EqualTo
//     - WithDenseFormat/WithSparseFormat
//
// Op(), Shape(), and Layout() may be passed an argument of type
// HloInstruction**, Shape**, or Layout**, respectively, or const versions of
// these pointers. If the pattern is matched, the address of the matched value
// will be "captured" and stored at this location.
//
// For example:
//   HloInstruction* foo = ...;
//   HloInstruction* matched_operand;
//   CHECK(Match(foo,
//               match::Op().WithOperand(0, match::Op(&matched_operand))));
//
// Helpers are provided for most HLO instructions. These helpers can be called
// with no arguments, in which case they will match any instruction matching the
// opcode. They may also be called with matches for the operands and with an
// optional capture. (The capture must be the first argument.) Some examples of
// these helpers and their equivalents are provided below.

// Example nullary instruction:
//   Parameter()                    == Op().WithOpcode(HloOpcode::kParameter)
//   Parameter(&a)                  == Op(&a).WithOpcode(HloOpcode::kParameter)
//
// Example unary instruction:
//   Abs()                          == Op().WithOpcode(HloOpcode::kAbs)
//   Abs(Op(&a))                    == Op().WithOpcode(HloOpcode::kAbs)
//                                         .WithOperand(0, Op(&a)))
//   Abs(&a, Op(&b))                == Op(&a).WithOpcode(HloOpcode::kAbs)
//                                           .WithOperand(0, Op(&b))
//
// Commutative binary instructions have a special form that accepts either order
// of args, e.g.:
//
//   AddAnyOrder(Parameter(1), Abs()) ==
//     Op().WithOpcode(HloOpcode::kAdd)
//         .WithBinaryOperandsAnyOrder(Op().WithParameterNum(1), Abs());
//
//   MultiplyAnyOrder(&a, Parameter(), Abs())  // Captures the mul in `a`.
//
// The following additional helpers are provided.  In all cases, `&a` is
// optional.
//
//   ConstantScalar(&a)               == Op(&a).IsConstantScalar();
//   ConstantScalar(&a, v)            == Op(&a).IsConstantScalar(v);
//   ConstantEffectiveScalar(&a)      == Op(&a).IsConstantEffectiveScalar();
//   ConstantEffectiveScalar(&a, v)   == Op(&a).IsConstantEffectiveScalar(&a, v)
//   NonConstant(&a)                  == Op(&a).IsNonConstant()
//   GetTupleElement(&a, b, index)    == Op(&a).WithTupleIndex(index)
//                                             .WithOperand(0, b);
//   Parameter(&a, n)                 == Op(&a).WithParameterNum(n);

struct MatchOption {
  // If true, actually capture matched item into the user pointer.
  bool capture;

  // An explanation for why we failed to match is streamed here, if not-null.
  std::ostream* explain_os;
};

template <typename Value, typename Pattern>
bool Match(Value* value, const Pattern& pattern,
           MatchOption option = {/*.capture=*/true, /*.explain_os=*/nullptr}) {
  if (option.capture) {
    auto new_option = option;
    new_option.capture = false;
    if (!pattern.Match(value, new_option)) {
      return false;
    }
  }
  return pattern.Match(value, option);
}

namespace match {

namespace detail {

// Macro for streaming to option.explain_os if it's not null.
//
//   EXPLAIN << "value of foo(): " << foo()
//
#pragma push_macro("EXPLAIN")
#define EXPLAIN \
  if (option.explain_os) *option.explain_os

// kIndentInc is the additional number of spaces that we indent by when we
// increase the indent "by one".
enum {
  kIndentInc = 2,
};

// Writes a newline and then `indent` spaces.
//
// We follow an unintuitive convention in this file's pretty-printers: Indents
// are performed by the caller, not the callee.  For example, if you want to
// print
//
//   foo:
//    - bar
//
// you'd do:
//
//  Foo::DescribeTo(std::ostream* os, int64 indent) {
//    *os << "foo:";
//    Indent(os, indent)  // Create a newline at the *current* indent level.
//    *os << " - ";
//    bar.DescribeTo(os, indent + 3);  // + 3 because strlen(" * ") == 3.
//  }
//
//  Bar::DescribeTo(std::ostream* os, int64 indent) { *os << "bar"; }
//
// Notice that Bar::DescribeTo() does not call Indent; the indenting is
// performed by Foo.  This convention allows the caller to decide whether a
// matcher is preceded by a newline, which is important e.g. for the AllOf
// matcher.
//
// (Incidentally, indenting in Match's explanations is handled differently.
// Indents are a common case in DescribeTo [we're printing a whole tree], but
// they're a special case in Match [we're printing only a path through the tree
// that encounters a failing node]. Indents in Match only appear when we
// encounter a failing disjunction, so we just handle them as a special case
// there.)
inline void Indent(std::ostream* os, int64 indent) {
  *os << "\n";
  for (int64 i = 0; i < indent; ++i) {
    *os << " ";
  }
}

// SFINAE template that determines whether T declares a static member
// kIsTrivialMatcher.
//
// Trivial matchers get special treatment.  For example, when printing
// a conjunction of matchers, we don't print "and" after a trivial matcher. This
// yields e.g.
//    "a shape compatible with f32[1,2]"
// rather than
//    "a shape AND compatible with f32[1,2]"
template <typename T, typename Dummy = void>
struct IsTrivialMatcher {
  static constexpr bool value = false;
};
template <typename T>
struct IsTrivialMatcher<T,
                        typename std::enable_if<T::kIsTrivialMatcher>::type> {
  static constexpr bool value = true;
};

template <typename Item, typename... Patterns>
class AllOfPattern {
 public:
  explicit AllOfPattern(const Patterns&... patterns) : patterns_(patterns...) {}

  bool Match(const Item* item, MatchOption option) const {
    bool matched = MatchImpl(item, option, std::integral_constant<size_t, 0>());
    // This invariant is guaranteed by the top-level Match and AnyOf.
    DCHECK(matched || !option.capture);
    return matched;
  }

  bool Match(Item* item, MatchOption option) const {
    bool matched = MatchImpl(item, option, std::integral_constant<size_t, 0>());
    // This invariant is guaranteed by the top-level Match and AnyOf.
    DCHECK(matched || !option.capture);
    return matched;
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    DescribeToImpl(os, std::integral_constant<size_t, 0>(), indent);
  }

  // Accessor for patterns_.  Please don't use this outside of this file.
  const std::tuple<Patterns...>& patterns() const { return patterns_; }

 private:
  template <typename ItemType, size_t index>
  bool MatchImpl(ItemType* item, MatchOption option,
                 std::integral_constant<size_t, index>) const {
    // We don't need to do any EXPLAINing here; it's all correctly handled by
    // our sub-matchers (if any fail).
    return std::get<index>(patterns_).Match(item, option) &&
           MatchImpl(item, option, std::integral_constant<size_t, index + 1>());
  }

  template <typename ItemType>
  bool MatchImpl(ItemType* item, MatchOption option,
                 std::integral_constant<size_t, sizeof...(Patterns)>) const {
    return true;
  }

  // Pretty-printing a conjunction has some special cases to make it easy to
  // read in the simple (common) case.
  //
  // If sizeof...(Patterns) == 1, prints as e.g.
  //
  //   a shape
  //
  // If sizeof...(Patterns) == 2 and patterns_[0] is a trivial matcher (e.g. "a
  // shape") prints as
  //
  //   a shape compatible with f32[1,2]
  //
  // If sizeof...(Patterns) > 2 and patterns_[0] is a trivial matcher, prints as
  //
  //   a shape:
  //    * compatible with f32[1,2] AND
  //    * that represents a scalar
  //
  // Otherwise prints as:
  //
  //   all of:
  //    * foo AND
  //    * bar
  //
  template <size_t index>
  void DescribeToImpl(std::ostream* os, std::integral_constant<size_t, index>,
                      int64 indent) const {
    constexpr bool first_is_trivial =
        IsTrivialMatcher<typename std::remove_reference<decltype(
            std::get<0>(patterns_))>::type>::value;
    constexpr bool is_last = index == sizeof...(Patterns) - 1;
    const auto& submatcher = std::get<index>(patterns_);

    auto print_bulleted_item = [&] {
      *os << " * ";
      submatcher.DescribeTo(os, indent + 3);
      if (!is_last) {
        *os << " AND";
        Indent(os, indent);
      }
    };

    if (index == 0) {
      if (first_is_trivial || is_last) {
        submatcher.DescribeTo(os, indent + kIndentInc);
        if (sizeof...(Patterns) > 2) {
          *os << ":";
          Indent(os, indent);
        }
      } else {
        *os << "all of:";
        Indent(os, indent);
        print_bulleted_item();
      }
    } else if (first_is_trivial && index == 1 && sizeof...(Patterns) == 2) {
      *os << " ";
      submatcher.DescribeTo(os, indent);
    } else {
      print_bulleted_item();
    }
    DescribeToImpl(os, std::integral_constant<size_t, index + 1>(), indent);
  }

  void DescribeToImpl(std::ostream* os,
                      std::integral_constant<size_t, sizeof...(Patterns)>,
                      int64 indent) const {}

  std::tuple<Patterns...> patterns_;
};

}  // namespace detail

// Returns a pattern that represents the conjunction of all input patterns. All
// patterns need to match in order to have the AllOf pattern match.
template <typename Item, typename... Patterns>
detail::AllOfPattern<typename std::remove_const<Item>::type, Patterns...> AllOf(
    const Patterns&... patterns) {
  return detail::AllOfPattern<typename std::remove_const<Item>::type,
                              Patterns...>(patterns...);
}

// AllOf<AllOf<A, B...>, X, Y, ...> => AllOf<A, B, ..., X, Y, ...>.
//
// This transformation is necessary for good pretty-printing.
template <typename Item, typename... InnerPs, typename... OuterPs>
detail::AllOfPattern<typename std::remove_const<Item>::type, InnerPs...,
                     OuterPs...>
AllOf(const detail::AllOfPattern<Item, InnerPs...>& inner_p,
      const OuterPs&... outer_ps) {
  // Invoke constructor of AllOfPattern<Item, InnerPs..., OuterPs...>.
  auto make_all_of = [](const InnerPs&... inner_ps,
                        const OuterPs&... outer_ps) {
    return detail::AllOfPattern<typename std::remove_const<Item>::type,
                                InnerPs..., OuterPs...>(inner_ps...,
                                                        outer_ps...);
  };
  return absl::apply(make_all_of, std::tuple_cat(inner_p.patterns(),
                                                 std::make_tuple(outer_ps...)));
}

namespace detail {

template <typename LayoutType, typename Impl>
class LayoutPattern;

// The base LayoutPattern implementation. Matches only if the layout is not
// nullptr.
class LayoutPatternBaseImpl {
 public:
  bool Match(const ::xla::Layout* layout, MatchOption option) const {
    if (layout == nullptr) {
      EXPLAIN << "Layout is null";
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "a layout";
  }

  static constexpr bool kIsTrivialMatcher = true;
};

// A LayoutPattern implementation that matches only if the layout equals a
// Layout proto.
class LayoutPatternEqualImpl {
 public:
  explicit constexpr LayoutPatternEqualImpl(const ::xla::Layout* layout)
      : layout_(layout) {}

  bool Match(const ::xla::Layout* layout, MatchOption option) const {
    if (!LayoutUtil::Equal(*layout_, *layout)) {
      EXPLAIN << "Layout " << LayoutUtil::HumanString(*layout)
              << " is not equal to expected "
              << LayoutUtil::HumanString(*layout_);
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "equal to " << LayoutUtil::HumanString(*layout_);
  }

 private:
  const ::xla::Layout* layout_;
};

// A LayoutPattern implementation that matches only if the layout has a given
// format.
class LayoutPatternFormatImpl {
 public:
  explicit constexpr LayoutPatternFormatImpl(Format format) : format_(format) {}

  bool Match(const ::xla::Layout* layout, MatchOption option) const {
    if (layout->format() != format_) {
      EXPLAIN << "Layout has format " << Format_Name(layout->format())
              << " but expected " << Format_Name(format_);
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "with format " << Format_Name(format_);
  }

 private:
  Format format_;
};

// A pattern that matches Layouts.
template <typename LayoutType, typename Impl>
class LayoutPattern {
 private:
  template <typename NewImpl>
  auto AppendImpl(NewImpl new_impl) const
      -> LayoutPattern<LayoutType,
                       decltype(AllOf<Layout>(std::declval<Impl>(),
                                              std::move(new_impl)))> {
    auto new_allof = AllOf<Layout>(impl_, std::move(new_impl));
    return LayoutPattern<LayoutType, decltype(new_allof)>(std::move(new_allof),
                                                          matched_layout_);
  }

 public:
  explicit constexpr LayoutPattern(const Impl& impl,
                                   LayoutType** matched_layout)
      : impl_(impl), matched_layout_(matched_layout) {}

  // Returns true and captures the layout iff it matches the pattern.
  bool Match(const ::xla::Layout* layout, MatchOption option) const {
    if (impl_.Match(layout, option)) {
      if (option.capture && matched_layout_) {
        *matched_layout_ = layout;
      }
      return true;
    }
    return false;
  }

  // Returns true and captures the layout iff it matches the pattern.
  bool Match(::xla::Layout* layout, MatchOption option) const {
    if (impl_.Match(layout, option)) {
      if (option.capture && matched_layout_) {
        *matched_layout_ = layout;
      }
      return true;
    }
    return false;
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    impl_.DescribeTo(os, indent);
  }

  // Modifies the pattern to match only if the layout equals the given proto.
  // The layout must outlive the returned pattern.
  constexpr auto EqualTo(const ::xla::Layout* layout) const
      -> decltype(this->AppendImpl(LayoutPatternEqualImpl(layout))) {
    return AppendImpl(LayoutPatternEqualImpl(layout));
  }

  // Modifies the pattern to match only if the layout has a dense format.
  constexpr auto WithDenseFormat() const
      -> decltype(this->AppendImpl(LayoutPatternFormatImpl(DENSE))) {
    return AppendImpl(LayoutPatternFormatImpl(DENSE));
  }

  // Modifies the pattern to match only if the layout has a sparse format.
  constexpr auto WithSparseFormat() const
      -> decltype(this->AppendImpl(LayoutPatternFormatImpl(SPARSE))) {
    return AppendImpl(LayoutPatternFormatImpl(SPARSE));
  }

 private:
  Impl impl_;
  LayoutType** matched_layout_;
};

template <typename Item, typename... Patterns>
class AnyOfPattern {
 public:
  explicit AnyOfPattern(const Patterns&... patterns) : patterns_(patterns...) {}

  bool Match(const Item* item, MatchOption option) const {
    return MatchImpl(item, option);
  }

  bool Match(Item* item, MatchOption option) const {
    return MatchImpl(item, option);
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "any of:";
    Indent(os, indent);
    DescribeToImpl(os, std::integral_constant<size_t, 0>(), indent);
  }

 private:
  template <typename ItemType>
  bool MatchImpl(ItemType* item, MatchOption option) const {
    // If we're generating an explanation, buffer it until we know we failed.
    absl::optional<std::stringstream> explanation;
    MatchOption new_option = option;
    if (option.explain_os) {
      new_option.explain_os = &explanation.emplace();
    }
    bool rv = MatchRecursiveImpl(item, new_option,
                                 std::integral_constant<size_t, 0>());
    if (!rv && option.explain_os) {
      EXPLAIN << "None of the following matchers succeeded:";
      EXPLAIN << explanation->str();
    }
    return rv;
  }

  template <typename ItemType, size_t index>
  bool MatchRecursiveImpl(ItemType* item, MatchOption option,
                          std::integral_constant<size_t, index>) const {
    auto new_option = option;
    new_option.capture = false;

    absl::optional<std::stringstream> explanation;
    if (option.explain_os) {
      new_option.explain_os = &explanation.emplace();
    }

    // Try to match the sub-pattern without capturing behavior.
    if (std::get<index>(patterns_).Match(item, new_option)) {
      // Capture the branch.
      if (option.capture) {
        // TODO(timshen): Currently the behavior can be exponential. Optimize it
        // with memoization or recording the matched sub-pattern index, if it
        // takes too long to run.
        //
        // Specifically, the "memoization" approach is to create an empty
        // container with the key (pattern, instruction), and value as whether
        // matched or not.
        //
        // Alternatively, we may run the pattern matching with captures off, but
        // instead record a "trace" somewhere, indicating how exactly the
        // pattern matches the input. For example, the trace information for
        // AnyOf will be a runtime number indicate which sub-pattern is matched.
        // Then we run another pass to do captures only with the help of the
        // trace.
        bool matched = std::get<index>(patterns_).Match(item, option);
        DCHECK(matched);
      }
      return true;
    }
    if (option.explain_os) {
      EXPLAIN << "\nMatcher #" << index + 1;
      EXPLAIN << "\n - ";
      std::get<index>(patterns_).DescribeTo(option.explain_os, /*indent=*/3);
      EXPLAIN << "\nfailed with";
      EXPLAIN << "\n - ";
      EXPLAIN << absl::StrReplaceAll(explanation->str(), {{"\n", "\n   "}});
    }
    return MatchRecursiveImpl(item, option,
                              std::integral_constant<size_t, index + 1>());
  }

  template <typename ItemType>
  bool MatchRecursiveImpl(
      ItemType* item, MatchOption option,
      std::integral_constant<size_t, sizeof...(Patterns)>) const {
    return false;
  }

  template <size_t index>
  void DescribeToImpl(std::ostream* os, std::integral_constant<size_t, index>,
                      int64 indent) const {
    *os << " - ";
    std::get<index>(patterns_).DescribeTo(os, indent + 3);
    if (index != sizeof...(Patterns) - 1) {
      *os << " OR";
      Indent(os, indent);
    }
    DescribeToImpl(os, std::integral_constant<size_t, index + 1>(), indent);
  }

  void DescribeToImpl(std::ostream* os,
                      std::integral_constant<size_t, sizeof...(Patterns)>,
                      int64 indent) const {}

  std::tuple<Patterns...> patterns_;
};

}  // namespace detail

// Returns a pattern that represents the logical disjunction of the input
// patterns. The returned pattern matches from left to right, and stops on the
// first match.
template <typename Item, typename... Patterns>
detail::AnyOfPattern<typename std::remove_const<Item>::type, Patterns...> AnyOf(
    const Patterns&... patterns) {
  return detail::AnyOfPattern<typename std::remove_const<Item>::type,
                              Patterns...>(patterns...);
}

// Creates a layout pattern that will capture the matched layout in the
// argument.
inline constexpr detail::LayoutPattern<const ::xla::Layout,
                                       detail::LayoutPatternBaseImpl>
Layout(const ::xla::Layout** matched_layout = nullptr) {
  return detail::LayoutPattern<const ::xla::Layout,
                               detail::LayoutPatternBaseImpl>(
      detail::LayoutPatternBaseImpl(), matched_layout);
}

// Creates a layout pattern that will capture the matched layout in the
// argument.
inline constexpr detail::LayoutPattern<::xla::Layout,
                                       detail::LayoutPatternBaseImpl>
Layout(::xla::Layout** matched_layout) {
  return detail::LayoutPattern<::xla::Layout, detail::LayoutPatternBaseImpl>(
      detail::LayoutPatternBaseImpl(), matched_layout);
}

namespace detail {

template <typename ShapeType, typename Impl>
class ShapePattern;

// The base ShapePattern implementation. Matches only if the shape is not
// nullptr.
class ShapePatternBaseImpl {
 public:
  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    if (shape == nullptr) {
      EXPLAIN << "Shape is null";
    }
    return shape != nullptr;
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "a shape";
  }

  static constexpr bool kIsTrivialMatcher = true;
};

// A ShapePattern implementation that matches only if the shape equals a Shape
// proto.
class ShapePatternEqualImpl {
 public:
  explicit constexpr ShapePatternEqualImpl(const ::xla::Shape* shape)
      : shape_(shape) {}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    if (!ShapeUtil::Equal(*shape_, *shape)) {
      EXPLAIN << "Shape not equal to "
              << ShapeUtil::HumanStringWithLayout(*shape_);
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "equal to " << ShapeUtil::HumanStringWithLayout(*shape_);
  }

 private:
  const ::xla::Shape* shape_;
};

// A ShapePattern implementation that matches only if the shape is compatible to
// a Shape proto.
class ShapePatternCompatibleImpl {
 public:
  explicit constexpr ShapePatternCompatibleImpl(const ::xla::Shape* shape)
      : shape_(shape) {}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    if (!ShapeUtil::Compatible(*shape_, *shape)) {
      EXPLAIN << "Shape not compatible with "
              << ShapeUtil::HumanString(*shape_);
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "compatible with " << ShapeUtil::HumanString(*shape_);
  }

 private:
  const ::xla::Shape* shape_;
};

// A ShapePattern implementation that matches only if the shape has a given
// element type.
class ShapePatternElementTypeImpl {
 public:
  explicit constexpr ShapePatternElementTypeImpl(PrimitiveType element_type)
      : element_type_(element_type) {}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    if (shape->element_type() != element_type_) {
      EXPLAIN << "Shape does not have element type "
              << PrimitiveType_Name(element_type_);
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "with element type " << PrimitiveType_Name(element_type_);
  }

 private:
  PrimitiveType element_type_;
};

// A ShapePattern implementation that matches only if the shape is scalar.
class ShapePatternIsScalarImpl {
 public:
  explicit constexpr ShapePatternIsScalarImpl() {}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    if (!ShapeUtil::IsScalar(*shape)) {
      EXPLAIN << "Shape is not a scalar";
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "that represents a scalar";
  }
};

// A ShapePattern implementation that matches only if the shape is an array
class ShapePatternIsArrayImpl {
 public:
  explicit constexpr ShapePatternIsArrayImpl() {}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    if (!ShapeUtil::IsArray(*shape)) {
      EXPLAIN << "Shape is not an array";
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "that represents an array";
  }
};

// A ShapePattern implementation that matches only if the shape is a tuple.
class ShapePatternIsTupleImpl {
 public:
  explicit constexpr ShapePatternIsTupleImpl() {}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    if (!ShapeUtil::IsTuple(*shape)) {
      EXPLAIN << "Shape is not a tuple";
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "that represents a tuple";
  }
};

// A ShapePattern implementation that matches only if the shape is an effective
// scalar.
class ShapePatternEffectiveScalarImpl {
 public:
  explicit constexpr ShapePatternEffectiveScalarImpl() {}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    if (!ShapeUtil::IsEffectiveScalar(*shape)) {
      EXPLAIN << "Shape is not an effective scalar";
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "that is an effective scalar";
  }
};

// A ShapePattern implementation that matches only if the shape has a given
// rank.
class ShapePatternRankImpl {
 public:
  explicit constexpr ShapePatternRankImpl(int64 rank) : rank_(rank) {}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    if (ShapeUtil::Rank(*shape) != rank_) {
      if (rank_ == 0) {
        EXPLAIN << "Shape is not a scalar";
      } else {
        EXPLAIN << "Shape does not have rank " << rank_;
      }
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    if (rank_ == 0) {
      *os << "that is a scalar";
    } else {
      *os << "that has " << rank_ << " dimension" << (rank_ != 1 ? "s" : "");
    }
  }

 private:
  int64 rank_;
};

// A ShapePattern implementation that matches only if the shape has a layout
// that matches a given pattern.
template <typename LayoutType, typename LayoutImpl>
class ShapePatternLayoutImpl {
 public:
  explicit constexpr ShapePatternLayoutImpl(
      const LayoutPattern<LayoutType, LayoutImpl>& layout)
      : layout_(layout) {}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    return LayoutUtil::HasLayout(*shape) &&
           layout_.Match(&shape->layout(), option);
  }

  bool Match(Shape* shape, MatchOption option) const {
    if (!LayoutUtil::HasLayout(*shape)) {
      EXPLAIN << "Shape does not have a layout";
      return false;
    }
    if (!layout_.Match(shape->mutable_layout(), option)) {
      EXPLAIN << "\nin layout";
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "with";
    Indent(os, indent + kIndentInc);
    layout_.DescribeTo(os, indent + kIndentInc);
  }

 private:
  LayoutPattern<LayoutType, LayoutImpl> layout_;
};

// A ShapePattern implementation that matches only if the shape has a subshape
// that matches a given pattern.
template <typename SubshapeType, typename SubshapeImpl>
class ShapePatternSubshapeImpl {
 public:
  explicit ShapePatternSubshapeImpl(
      ShapeIndexView index,
      const ShapePattern<SubshapeType, SubshapeImpl>& subshape)
      : index_(index), subshape_(subshape) {}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    return MatchImpl(shape, option);
  }

  bool Match(::xla::Shape* shape, MatchOption option) const {
    return MatchImpl(shape, option);
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "with subshape at index " << index_.ToString() << " which is";
    Indent(os, indent + kIndentInc);
    subshape_.DescribeTo(os, indent + kIndentInc);
  }

 private:
  Shape* GetSubshape(Shape* shape) const {
    return ShapeUtil::GetMutableSubshape(shape, index_);
  }
  const Shape* GetSubshape(const Shape* shape) const {
    return &ShapeUtil::GetSubshape(*shape, index_);
  }

  template <typename ShapeType>
  bool MatchImpl(ShapeType* shape, MatchOption option) const {
    if (!ShapeUtil::IndexIsValid(*shape, index_)) {
      EXPLAIN << "No subshape at " << index_.ToString();
      return false;
    }
    if (!subshape_.Match(GetSubshape(shape), option)) {
      EXPLAIN << "\nin subshape at " << index_.ToString();
      return false;
    }
    return true;
  }

  ShapeIndexView index_;
  ShapePattern<SubshapeType, SubshapeImpl> subshape_;
};

// A pattern that matches Shapes.
template <typename ShapeType, typename Impl>
class ShapePattern {
 private:
  template <typename NewImpl>
  auto AppendImpl(NewImpl new_impl) const
      -> ShapePattern<ShapeType, decltype(AllOf<Shape>(std::declval<Impl>(),
                                                       std::move(new_impl)))> {
    auto new_all_of = AllOf<Shape>(impl_, std::move(new_impl));
    return ShapePattern<ShapeType, decltype(new_all_of)>(std::move(new_all_of),
                                                         matched_shape_);
  }

 public:
  explicit constexpr ShapePattern(const Impl& impl, ShapeType** matched_shape)
      : impl_(impl), matched_shape_(matched_shape) {}

  // Returns true and captures the shape iff it matches the pattern.
  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    if (impl_.Match(shape, option)) {
      if (option.capture && matched_shape_) {
        *matched_shape_ = shape;
      }
      return true;
    }
    if (shape) {
      EXPLAIN << "\nin "
              << (shape->has_layout() ? ShapeUtil::HumanStringWithLayout(*shape)
                                      : ShapeUtil::HumanString(*shape));
    }
    return false;
  }

  // Returns true and captures the shape iff it matches the pattern.
  bool Match(::xla::Shape* shape, MatchOption option) const {
    if (impl_.Match(shape, option)) {
      if (option.capture && matched_shape_) {
        *matched_shape_ = shape;
      }
      return true;
    }
    EXPLAIN << "\nin "
            << (shape->has_layout() ? ShapeUtil::HumanStringWithLayout(*shape)
                                    : ShapeUtil::HumanString(*shape));
    return false;
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    return impl_.DescribeTo(os, indent);
  }

  // Modifies the pattern to match only if the shape equals the given proto.
  // The layout must outlive the returned pattern.
  constexpr auto EqualTo(const ::xla::Shape* shape) const
      -> decltype(this->AppendImpl(ShapePatternEqualImpl(shape))) {
    return AppendImpl(ShapePatternEqualImpl(shape));
  }

  // Modifies the pattern to match only if the shape is compatible to the given
  // proto. The layout must outlive the returned pattern.
  constexpr auto CompatibleTo(const ::xla::Shape* shape) const
      -> decltype(this->AppendImpl(ShapePatternCompatibleImpl(shape))) {
    return AppendImpl(ShapePatternCompatibleImpl(shape));
  }

  // Modifies the pattern to match only if the shape has the given element type.
  constexpr auto WithElementType(PrimitiveType element_type) const
      -> decltype(this->AppendImpl(ShapePatternElementTypeImpl(element_type))) {
    return AppendImpl(ShapePatternElementTypeImpl(element_type));
  }

  // Modifies the pattern to match only if the shape is scalar.
  constexpr auto IsScalar() const
      -> decltype(this->AppendImpl(ShapePatternIsScalarImpl())) {
    return AppendImpl(ShapePatternIsScalarImpl());
  }

  // Modifies the pattern to match only if the shape is an array.
  constexpr auto IsArray() const
      -> decltype(this->AppendImpl(ShapePatternIsArrayImpl())) {
    return AppendImpl(ShapePatternIsArrayImpl());
  }

  // Modifies the pattern to match only if the shape is a tuple.
  constexpr auto IsTuple() const
      -> decltype(this->AppendImpl(ShapePatternIsTupleImpl())) {
    return AppendImpl(ShapePatternIsTupleImpl());
  }

  constexpr auto IsEffectiveScalar() const
      -> decltype(this->AppendImpl(ShapePatternEffectiveScalarImpl())) {
    return AppendImpl(ShapePatternEffectiveScalarImpl());
  }

  // Modifies the pattern to match only if the shape has the given rank.
  constexpr auto WithRank(int64 rank) const
      -> decltype(this->AppendImpl(ShapePatternRankImpl(rank))) {
    return AppendImpl(ShapePatternRankImpl(rank));
  }

  // Modifies the pattern to match only if the shape has a layout that matches
  // the given pattern.
  template <typename LayoutType, typename LayoutImpl>
  auto WithLayout(const LayoutPattern<LayoutType, LayoutImpl>& layout) const
      -> decltype(this->AppendImpl(
          ShapePatternLayoutImpl<LayoutType, LayoutImpl>(layout))) {
    return AppendImpl(ShapePatternLayoutImpl<LayoutType, LayoutImpl>(layout));
  }

  constexpr auto WithLayoutEqualTo(const ::xla::Layout* layout) const
      -> decltype(this->WithLayout(Layout().EqualTo(layout))) {
    return WithLayout(Layout().EqualTo(layout));
  }

  constexpr auto IsDenseArray() const
      -> decltype(this->WithLayout(Layout().WithDenseFormat())) {
    return WithLayout(Layout().WithDenseFormat());
  }

  constexpr auto IsSparseArray() const
      -> decltype(this->WithLayout(Layout().WithSparseFormat())) {
    return WithLayout(Layout().WithSparseFormat());
  }

  // Modifies the pattern to match only if the shape has a subshape that matches
  // the given pattern.
  template <typename SubshapeType, typename SubshapeImpl>
  auto WithSubshape(ShapeIndexView index,
                    const ShapePattern<SubshapeType, SubshapeImpl>& subshape)
      const -> decltype(this->AppendImpl(
          ShapePatternSubshapeImpl<SubshapeType, SubshapeImpl>(index,
                                                               subshape))) {
    return AppendImpl(
        ShapePatternSubshapeImpl<SubshapeType, SubshapeImpl>(index, subshape));
  }

  ShapePattern<ShapeType,
               AllOfPattern<Shape, Impl,
                            ShapePatternSubshapeImpl<
                                const ::xla::Shape,
                                AllOfPattern<::xla::Shape, ShapePatternBaseImpl,
                                             ShapePatternEqualImpl>>>>
  WithSubshapeEqualTo(ShapeIndexView index, const ::xla::Shape* shape) const {
    return WithSubshape(index,
                        ShapePattern<const ::xla::Shape, ShapePatternBaseImpl>(
                            ShapePatternBaseImpl(), nullptr)
                            .EqualTo(shape));
  }

  ShapePattern<ShapeType,
               AllOfPattern<Shape, Impl,
                            ShapePatternSubshapeImpl<
                                const ::xla::Shape,
                                AllOfPattern<::xla::Shape, ShapePatternBaseImpl,
                                             ShapePatternCompatibleImpl>>>>
  WithSubshapeCompatibleTo(ShapeIndexView index,
                           const ::xla::Shape* shape) const {
    return WithSubshape(index,
                        ShapePattern<const ::xla::Shape, ShapePatternBaseImpl>(
                            ShapePatternBaseImpl(), nullptr)
                            .CompatibleTo(shape));
  }

 private:
  Impl impl_;
  ShapeType** matched_shape_;
};

}  // namespace detail

// Creates a shape pattern that will capture the matched layout in the argument.
inline constexpr detail::ShapePattern<const ::xla::Shape,
                                      detail::ShapePatternBaseImpl>
Shape(const ::xla::Shape** matched_shape = nullptr) {
  return detail::ShapePattern<const ::xla::Shape, detail::ShapePatternBaseImpl>(
      detail::ShapePatternBaseImpl(), matched_shape);
}

// Creates a shape pattern that will capture the matched layout in the argument.
inline constexpr detail::ShapePattern<::xla::Shape,
                                      detail::ShapePatternBaseImpl>
Shape(::xla::Shape** matched_shape) {
  return detail::ShapePattern<::xla::Shape, detail::ShapePatternBaseImpl>(
      detail::ShapePatternBaseImpl(), matched_shape);
}

namespace detail {

// Overloads to get a const or non-const operand out of an instruction.
inline HloInstruction* HloOperand(HloInstruction* instr, int64 idx) {
  return instr->mutable_operand(idx);
}
inline const HloInstruction* HloOperand(const HloInstruction* instr,
                                        int64 idx) {
  return instr->operand(idx);
}

// Pretty-printer for HloInstruction.  Sort of like ToShortString, but with
// fewer %s and more shapes.
inline string InstToString(const HloInstruction* inst) {
  return inst->ToString(
      HloPrintOptions().set_print_metadata(false).set_print_percent(false));
}

template <typename HloInstructionType, typename Impl>
class HloInstructionPattern;

// The base HloInstructionPattern implementation. Matches only if the
// instruction is not nullptr.
class HloInstructionPatternBaseImpl {
 public:
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    if (inst == nullptr) {
      EXPLAIN << "HloInstruction* is null";
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "an HloInstruction";
  }

  static constexpr bool kIsTrivialMatcher = true;
};

// An HloInstructionPattern implementation that matches only if the instruction
// has a given name.
class HloInstructionPatternNameImpl {
 public:
  explicit HloInstructionPatternNameImpl(absl::string_view name)
      : name_(name) {}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    if (inst->name() != name_) {
      EXPLAIN << "HloInstruction not named \"" << name_ << "\"";
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "named \"" << name_ << "\"";
  }

 private:
  absl::string_view name_;
};

// An HloInstructionPattern implementation that matches only if the instruction
// equals a particular pointer.
class HloInstructionIsImpl {
 public:
  explicit HloInstructionIsImpl(const HloInstruction* inst) : inst_(inst) {}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    if (inst != inst_) {
      EXPLAIN << "HloInstruction " << inst << " is not " << inst_ << " ("
              << InstToString(inst_) << ")";
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "which is " << inst_ << " (" << InstToString(inst_) << ")";
  }

 private:
  const HloInstruction* inst_;
};

// An HloInstructionPattern implementation that matches only if the instruction
// has a given opcode.
class HloInstructionPatternOpcodeImpl {
 public:
  explicit constexpr HloInstructionPatternOpcodeImpl(HloOpcode opcode,
                                                     bool invert)
      : opcode_(opcode), invert_(invert) {}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    if (invert_ && inst->opcode() == opcode_) {
      EXPLAIN << "HloInstruction has opcode " << HloOpcodeString(opcode_)
              << ", expected anything else";
      return false;
    }
    if (!invert_ && inst->opcode() != opcode_) {
      EXPLAIN << "HloInstruction doesn't have opcode "
              << HloOpcodeString(opcode_);
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    if (!invert_) {
      *os << "with opcode " << HloOpcodeString(opcode_);
    } else {
      *os << "with any opcode other than " << HloOpcodeString(opcode_);
    }
  }

 private:
  HloOpcode opcode_;
  bool invert_;
};

// An HloInstructionPattern implementation that matches only if the instruction
// has the given number of operands.
class HloInstructionPatternNumOperandsImpl {
 public:
  explicit constexpr HloInstructionPatternNumOperandsImpl(int64 num_operands)
      : num_operands_(num_operands) {}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    if (inst->operand_count() != num_operands_) {
      EXPLAIN << "HloInstruction doesn't have " << num_operands_ << " operands";
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "with " << num_operands_ << " operand"
        << (num_operands_ != 1 ? "s" : "");
  }

 private:
  int64 num_operands_;
};

// An HloInstructionPattern implementation that matches only if the instruction
// has a shape that matches a given pattern.
template <typename ShapeType, typename ShapeImpl>
class HloInstructionPatternShapeImpl {
 public:
  explicit constexpr HloInstructionPatternShapeImpl(
      const ShapePattern<ShapeType, ShapeImpl>& shape)
      : shape_(shape) {}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    if (!shape_.Match(&inst->shape(), option)) {
      EXPLAIN << "\nin output shape";
      return false;
    }
    return true;
  }

  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    if (!shape_.Match(inst->mutable_shape(), option)) {
      EXPLAIN << "\nin output shape";
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "outputting";
    Indent(os, indent + kIndentInc);
    shape_.DescribeTo(os, indent + kIndentInc);
  }

 private:
  ShapePattern<ShapeType, ShapeImpl> shape_;
};

// An HloInstructionPattern implementation that matches only if the instruction
// has an operand that matches a given pattern.
template <typename OperandType, typename OperandImpl>
class HloInstructionPatternOperandImpl {
 public:
  explicit constexpr HloInstructionPatternOperandImpl(
      int64 operand_index,
      const HloInstructionPattern<OperandType, OperandImpl>& operand)
      : operand_index_(operand_index), operand_(operand) {}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }

  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "with operand " << operand_index_ << " which is:";
    Indent(os, indent + kIndentInc);
    operand_.DescribeTo(os, indent + kIndentInc);
  }

 private:
  template <typename HloInstructionType>
  bool MatchImpl(HloInstructionType* inst, MatchOption option) const {
    if (operand_index_ >= inst->operand_count()) {
      EXPLAIN << "desired operand index " << operand_index_
              << " is out of bounds";
      return false;
    }
    if (!operand_.Match(HloOperand(inst, operand_index_), option)) {
      EXPLAIN << "\nin operand " << operand_index_;
      return false;
    }
    return true;
  }

  int64 operand_index_;
  HloInstructionPattern<OperandType, OperandImpl> operand_;
};

// Matches a binary instruction whose operands come in any order.
template <typename OperandType1, typename OperandImpl1, typename OperandType2,
          typename OperandImpl2>
class HloInstructionPatternBinaryOperandsAnyOrderImpl {
 public:
  explicit constexpr HloInstructionPatternBinaryOperandsAnyOrderImpl(
      const HloInstructionPattern<OperandType1, OperandImpl1>& op1,
      const HloInstructionPattern<OperandType2, OperandImpl2>& op2)
      : op1_(op1), op2_(op2) {}

  bool Match(HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }

  bool Match(const HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "with two operands in either order:";
    Indent(os, indent);
    *os << " - ";
    op1_.DescribeTo(os, indent + 3);
    Indent(os, indent);
    *os << " - ";
    op2_.DescribeTo(os, indent + 3);
  }

 private:
  HloInstruction* operand(HloInstruction* inst, int64 idx) const {
    return inst->mutable_operand(idx);
  }
  const HloInstruction* operand(const HloInstruction* inst, int64 idx) const {
    return inst->operand(idx);
  }

  template <typename HloInstructionType>
  bool MatchImpl(HloInstructionType* inst, MatchOption option) const {
    // We could implement this using AnyOf and AllOf matchers, but the templates
    // get pretty difficult to debug, since any compile error herein becomes
    // not-an-error via SFINAE.  Also this way lets us give better messages on
    // failure.
    if (inst->operand_count() != 2) {
      EXPLAIN << "HloInstruction did not have two operands";
      return false;
    }

    // If we're not generating explanations, this is pretty simple.
    if (!option.explain_os) {
      auto try_match = [&](int64 idx1, int64 idx2) {
        MatchOption new_option = option;
        new_option.capture = false;
        if (op1_.Match(operand(inst, idx1), new_option) &&
            op2_.Match(operand(inst, idx2), new_option)) {
          if (option.capture) {
            bool matched = op1_.Match(operand(inst, idx1), option) &&
                           op2_.Match(operand(inst, idx2), option);
            DCHECK(matched);
          }
          return true;
        }
        return false;
      };
      return try_match(0, 1) || try_match(1, 0);
    }

    // If we are generating explanations, we have some work to do in order to
    // generate a helpful error.
    //
    // First, try all four operand/matcher combinations, recording the
    // failure explanations separately from option.explain_os. matches[i][j]
    // tells us if matcher_i matches operand j.
    bool matches[/*matcher*/ 2][/*operand*/ 2];
    std::stringstream explanations[/*matcher*/ 2][/*operand*/ 2];
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        MatchOption new_option = option;
        new_option.capture = false;
        new_option.explain_os = &explanations[i][j];
        matches[i][j] = i == 0 ? op1_.Match(operand(inst, j), new_option)
                               : op2_.Match(operand(inst, j), new_option);
      }
    }

    // Check if the match succeeded.
    for (int i = 0; i < 2; ++i) {
      if (matches[0][i] && matches[1][(i + 1) % 2]) {
        // Rerun the matches with capture enabled if necessary.
        if (option.capture) {
          auto* operand1 = operand(inst, i);
          auto* operand2 = operand(inst, (i + 1) % 2);
          bool matched =
              op1_.Match(operand1, option) && op2_.Match(operand2, option);
          DCHECK(matched);
        }
        return true;
      }
    }

    auto describe_matcher = [&](int matcher_idx) {
      EXPLAIN << "\n - ";
      if (matcher_idx == 0) {
        op1_.DescribeTo(option.explain_os, /*indent=*/3);
      } else {
        CHECK_EQ(matcher_idx, 1);
        op2_.DescribeTo(option.explain_os, /*indent=*/3);
      }
      for (int i = 0; i < 2; ++i) {
        if (matches[matcher_idx][/*operand*/ i]) {
          continue;
        }
        EXPLAIN << "\ndoes not match " << (i == 0 ? "LHS" : "RHS") << ":\n";
        EXPLAIN << " - ";
        EXPLAIN << absl::StrReplaceAll(
            explanations[matcher_idx][/*operand*/ i].str(), {{"\n", "\n   "}});
      }
    };

    // If we failed to match, one of the following is true:
    //  1. op1 (op2) matches neither LHS nor RHS, or
    //  2. op1 and op2 both match LHS (RHS), but neither matches RHS (LHS).
    // We print different explanations depending on which case we're in.

    // Case 1.
    bool wrote_explanation = false;
    for (int i = 0; !wrote_explanation && i < 2; ++i) {
      if (!matches[i][0] && !matches[i][1]) {
        EXPLAIN << "HloInstruction's operands (ignoring order) did not match "
                << (i == 0 ? "first" : "second") << " matcher.  Specifically,";
        describe_matcher(i);
        wrote_explanation = true;
      }
    }

    // Case 2.
    for (int i = 0; !wrote_explanation && i < 2; ++i) {
      if (matches[/*matcher*/ 0][/*operand*/ i] &&
          matches[/*matcher*/ 1][/*operand*/ i]) {
        CHECK(!matches[0][(i + 1) % 2]);
        CHECK(!matches[1][(i + 1) % 2]);
        CHECK(!wrote_explanation);
        EXPLAIN << "HloInstruction's " << (i == 1 ? "LHS" : "RHS")
                << " operand did not match either of the two matchers.  "
                   "Specifically,";
        describe_matcher(0);
        EXPLAIN << "\nand";
        describe_matcher(1);
        wrote_explanation = true;
      }
    }

    CHECK(wrote_explanation);
    return false;
  }

  HloInstructionPattern<OperandType1, OperandImpl1> op1_;
  HloInstructionPattern<OperandType2, OperandImpl2> op2_;
};

// An HloInstructionPattern implementation that matches only if the instruction
// is a fusion node with a particular kind.
class HloInstructionPatternFusionKindImpl {
 public:
  explicit constexpr HloInstructionPatternFusionKindImpl(
      ::xla::HloInstruction::FusionKind kind)
      : kind_(kind) {}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }

  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "with fusion kind " << ToString(kind_);
  }

 private:
  template <typename HloInstructionType>
  bool MatchImpl(HloInstructionType* inst, MatchOption option) const {
    if (inst->opcode() != HloOpcode::kFusion) {
      EXPLAIN << "HloInstruction does not have fusion kind " << ToString(kind_)
              << "; it's not a fusion";
      return false;
    }
    if (inst->fusion_kind() != kind_) {
      EXPLAIN << "HloInstruction does not have fusion kind " << ToString(kind_);
      return false;
    }
    return true;
  }

  ::xla::HloInstruction::FusionKind kind_;
};

// An HloInstructionPattern implementation that matches only if the instruction
// is a kGetTupleElement with a particular tuple index.
class HloInstructionPatternTupleIndexImpl {
 public:
  explicit constexpr HloInstructionPatternTupleIndexImpl(int64 tuple_index)
      : tuple_index_(tuple_index) {}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }

  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "which is a GTE with index " << tuple_index_;
  }

 private:
  template <typename HloInstructionType>
  bool MatchImpl(HloInstructionType* inst, MatchOption option) const {
    if (inst->opcode() != HloOpcode::kGetTupleElement) {
      EXPLAIN << "HloInstruction is not a GTE with index " << tuple_index_
              << "; it's not a GTE at all";
      return false;
    }
    if (inst->tuple_index() != tuple_index_) {
      EXPLAIN << "HloInstruction is not a GTE with index " << tuple_index_;
      return false;
    }
    return true;
  }

  int64 tuple_index_;
};

class HloInstructionPatternParameterNumImpl {
 public:
  explicit constexpr HloInstructionPatternParameterNumImpl(int64 parameter_num)
      : parameter_num_(parameter_num) {}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }

  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "which is parameter " << parameter_num_;
  }

 private:
  template <typename HloInstructionType>
  bool MatchImpl(HloInstructionType* inst, MatchOption option) const {
    if (inst->opcode() != HloOpcode::kParameter ||
        inst->parameter_number() != parameter_num_) {
      EXPLAIN << "HloInstruction is not parameter " << parameter_num_;
      return false;
    }
    return true;
  }

  int64 parameter_num_;
};

// Superclass that contains common code used by Op::WithOneUse() and
// Op::WithOneUser().
class HloInstructionPatternOneUseOrUserImpl {
 protected:
  bool MatchOneUser(const HloInstruction* inst, MatchOption option) const {
    if (inst->user_count() != 1) {
      EXPLAIN << "HloInstruction has " << inst->user_count()
              << " users, but expected exactly one.";
      if (inst->user_count() > 1) {
        EXPLAIN << "\nAll users:";
        for (const HloInstruction* user : inst->users()) {
          EXPLAIN << "\n - " << InstToString(user);
        }
      }
      return false;
    }
    return true;
  }
};

class HloInstructionPatternOneUseImpl
    : public HloInstructionPatternOneUseOrUserImpl {
 public:
  bool Match(const HloInstruction* inst, MatchOption option) const {
    if (!MatchOneUser(inst, option)) {
      return false;
    }

    int64 use_count = absl::c_count_if(
        inst->users()[0]->operands(),
        [&](const HloInstruction* operand) { return operand == inst; });
    if (use_count != 1) {
      EXPLAIN << "HloInstruction is used " << use_count
              << " times by its user, but is expected to be used just once: "
              << InstToString(inst->users()[0]);
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "which has exactly one use";
  }
};

class HloInstructionPatternOneUserImpl
    : public HloInstructionPatternOneUseOrUserImpl {
 public:
  bool Match(const HloInstruction* inst, MatchOption option) const {
    return MatchOneUser(inst, option);
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "which has exactly one user (but possibly is used multiple times by "
           "that instruction)";
  }
};

// Matches a constant scalar or effective scalar, optionally with a given value.
template <typename ScalarTy>
class HloConstantScalarImpl {
 public:
  explicit constexpr HloConstantScalarImpl(bool match_effective_scalar)
      : val_(absl::nullopt), match_effective_scalar_(match_effective_scalar) {}

  constexpr HloConstantScalarImpl(ScalarTy val, bool match_effective_scalar)
      : val_(val), match_effective_scalar_(match_effective_scalar) {}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }

  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    *os << "which is a constant "
        << (match_effective_scalar_ ? "effective " : "") << "scalar";
    if (val_.has_value()) {
      *os << " with value " << *val_;
    }
  }

 private:
  template <typename InstTy>
  bool MatchImpl(InstTy* inst, MatchOption option) const {
    const auto* const_inst = DynCast<HloConstantInstruction>(inst);
    if (!const_inst) {
      EXPLAIN << "HloInstruction is not a constant";
      return false;
    }
    if (match_effective_scalar_ &&
        !ShapeUtil::IsEffectiveScalar(inst->shape())) {
      EXPLAIN << "HloInstruction is not an effective scalar";
      return false;
    }
    if (!match_effective_scalar_ && !ShapeUtil::IsScalar(inst->shape())) {
      EXPLAIN << "HloInstruction is not a scalar";
      return false;
    }
    if (!val_.has_value()) {
      return true;
    }

    // Check that literal == static_cast<LitearlTy>(val) and
    // val == static_cast<ValTy>(literal).  This is sufficient to ensure that
    // the two constant scalars are actually "equal".
    auto val_literal = LiteralUtil::CreateR0(*val_);
    auto literal_r0_or = const_inst->literal().Reshape({});
    auto val_as_literal_ty_or =
        val_literal.Convert(const_inst->shape().element_type());
    if (!literal_r0_or.ok() || !val_as_literal_ty_or.ok()) {
      EXPLAIN << "could not construct relevant Literals (how did this happen?)";
      return false;
    }
    auto literal_r0 = std::move(literal_r0_or).ValueOrDie();
    auto val_as_literal_ty = std::move(val_as_literal_ty_or).ValueOrDie();
    auto literal_r0_as_val_ty_or =
        literal_r0.Convert(val_literal.shape().element_type());
    bool rv = literal_r0_as_val_ty_or.ok() &&  //
              literal_r0_as_val_ty_or.ValueOrDie() == val_literal &&
              literal_r0 == val_as_literal_ty;
    if (!rv) {
      EXPLAIN << "HloInstruction's constant value "
              << literal_r0.ToStringWithoutShape()
              << " did not match expected value " << *val_;
    }
    return rv;
  }

  absl::optional<ScalarTy> val_;
  bool match_effective_scalar_;
};

// A pattern that matches HloInstructions.
template <typename HloInstructionType, typename Impl>
class HloInstructionPattern {
 private:
  template <typename NewImpl>
  auto AppendImpl(NewImpl new_impl) const -> HloInstructionPattern<
      HloInstructionType, decltype(AllOf<HloInstruction>(
                              std::declval<Impl>(), std::move(new_impl)))> {
    auto new_allof = AllOf<HloInstruction>(impl_, std::move(new_impl));
    return HloInstructionPattern<HloInstructionType, decltype(new_allof)>(
        std::move(new_allof), matched_inst_);
  }

 public:
  explicit constexpr HloInstructionPattern(const Impl& impl,
                                           HloInstructionType** matched_inst)
      : impl_(impl), matched_inst_(matched_inst) {}

  // Returns true and captures the instruction iff it matches the pattern.
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    if (impl_.Match(inst, option)) {
      if (option.capture && matched_inst_) {
        *matched_inst_ = inst;
      }
      return true;
    }
    if (inst != nullptr) {
      EXPLAIN << "\nin " << InstToString(inst);
    }
    return false;
  }

  // Returns true and captures the instruction iff it matches the pattern.
  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    if (impl_.Match(inst, option)) {
      if (option.capture && matched_inst_) {
        *matched_inst_ = inst;
      }
      return true;
    }
    EXPLAIN << "\nin " << InstToString(inst);
    return false;
  }

  // Modifies the pattern to match only if the instruction has the given name.
  auto WithName(absl::string_view name) const
      -> decltype(this->AppendImpl(HloInstructionPatternNameImpl(name))) {
    return AppendImpl(HloInstructionPatternNameImpl(name));
  }

  // Modifies the pattern to match only if the instruction has the given opcode.
  auto WithOpcode(HloOpcode opcode) const
      -> decltype(this->AppendImpl(HloInstructionPatternOpcodeImpl(opcode,
                                                                   false))) {
    return AppendImpl(HloInstructionPatternOpcodeImpl(opcode, false));
  }

  auto WithNumOperands(int64 num_operands) const -> decltype(
      this->AppendImpl(HloInstructionPatternNumOperandsImpl(num_operands))) {
    return AppendImpl(HloInstructionPatternNumOperandsImpl(num_operands));
  }

  // Modifies the pattern to match only if the instruction does not have the
  // given opcode.
  auto WithoutOpcode(HloOpcode opcode) const
      -> decltype(this->AppendImpl(HloInstructionPatternOpcodeImpl(opcode,
                                                                   true))) {
    return AppendImpl(HloInstructionPatternOpcodeImpl(opcode, true));
  }

  constexpr auto Is(const HloInstruction* instr) const
      -> decltype(this->AppendImpl(HloInstructionIsImpl(instr))) {
    return AppendImpl(HloInstructionIsImpl(instr));
  }

  // Modifies the pattern to match only if the instruction is a constant.
  constexpr auto IsConstant() const
      -> decltype(this->WithOpcode(HloOpcode::kConstant)) {
    return WithOpcode(HloOpcode::kConstant);
  }

  constexpr auto IsConstantScalar() const -> decltype(this->AppendImpl(
      HloConstantScalarImpl</*Dummy*/ int>(/*match_effective_scalar=*/false))) {
    return AppendImpl(
        HloConstantScalarImpl</*Dummy*/ int>(/*match_effective_scalar=*/false));
  }

  // This does not check that T has the same type as the instruction, so e.g.
  // IsConstantScalar(1.0) may match a constant of shape int32[].
  template <typename ScalarTy>
  constexpr auto IsConstantScalar(const ScalarTy& val) const
      -> decltype(this->AppendImpl(HloConstantScalarImpl<ScalarTy>(
          val, /*match_effective_scalar=*/false))) {
    return AppendImpl(
        HloConstantScalarImpl<ScalarTy>(val, /*match_effective_scalar=*/false));
  }

  constexpr auto IsConstantEffectiveScalar() const -> decltype(this->AppendImpl(
      HloConstantScalarImpl</*Dummy*/ int>(/*match_effective_scalar=*/true))) {
    return AppendImpl(
        HloConstantScalarImpl</*Dummy*/ int>(/*match_effective_scalar=*/true));
  }

  template <typename ScalarTy>
  constexpr auto IsConstantEffectiveScalar(const ScalarTy& val) const
      -> decltype(this->AppendImpl(HloConstantScalarImpl<ScalarTy>(
          val, /*match_effective_scalar=*/true))) {
    return AppendImpl(
        HloConstantScalarImpl<ScalarTy>(val, /*match_effective_scalar=*/true));
  }

  // Modifies the pattern to match only if the instruction is not a constant.
  constexpr auto IsNonConstant() const
      -> decltype(this->WithoutOpcode(HloOpcode::kConstant)) {
    return WithoutOpcode(HloOpcode::kConstant);
  }

  // Modifies the pattern to match only if the instruction has a shape that
  // matches the given pattern.
  template <typename ShapeType, typename ShapeImpl>
  constexpr auto WithShape(const ShapePattern<ShapeType, ShapeImpl>& shape)
      const -> decltype(this->AppendImpl(
          HloInstructionPatternShapeImpl<ShapeType, ShapeImpl>(shape))) {
    return AppendImpl(
        HloInstructionPatternShapeImpl<ShapeType, ShapeImpl>(shape));
  }

  // Make this a templated function to work around gcc 4.9.4 template infinite
  // recursion bug.
  template <typename Dummy = void>
  constexpr auto WithShapeEqualTo(const ::xla::Shape* shape)
      -> decltype(this->WithShape(Shape().EqualTo(shape))) {
    return WithShape(Shape().EqualTo(shape));
  }

  // Make this a templated function to work around gcc 4.9.4 template infinite
  // recursion bug.
  template <typename Dummy = void>
  constexpr auto WithShapeCompatibleTo(const ::xla::Shape* shape)
      -> decltype(this->WithShape(Shape().CompatibleTo(shape))) {
    return WithShape(Shape().CompatibleTo(shape));
  }

  // Modifies the pattern to match only if the instruction has an operand that
  // matches the given pattern.
  template <typename OperandType, typename OperandImpl>
  constexpr auto WithOperand(
      int64 operand_index,
      const HloInstructionPattern<OperandType, OperandImpl>& operand) const
      -> decltype(this->AppendImpl(
          HloInstructionPatternOperandImpl<OperandType, OperandImpl>(
              operand_index, operand))) {
    return AppendImpl(
        HloInstructionPatternOperandImpl<OperandType, OperandImpl>(
            operand_index, operand));
  }

  template <typename OperandType1, typename OperandImpl1, typename OperandType2,
            typename OperandImpl2>
  constexpr auto WithBinaryOperandsAnyOrder(
      const HloInstructionPattern<OperandType1, OperandImpl1>& op1,
      const HloInstructionPattern<OperandType2, OperandImpl2>& op2) const
      -> decltype(this->AppendImpl(
          HloInstructionPatternBinaryOperandsAnyOrderImpl<
              OperandType1, OperandImpl1, OperandType2, OperandImpl2>(op1,
                                                                      op2))) {
    return AppendImpl(
        HloInstructionPatternBinaryOperandsAnyOrderImpl<
            OperandType1, OperandImpl1, OperandType2, OperandImpl2>(op1, op2));
  }

  // Modifies the pattern to match only if the instruction is a fusion node with
  // the given kind.
  constexpr auto WithFusionKind(HloInstruction::FusionKind kind) const
      -> decltype(this->AppendImpl(HloInstructionPatternFusionKindImpl(kind))) {
    return AppendImpl(HloInstructionPatternFusionKindImpl(kind));
  }

  // Modifies the pattern to match only if the instruction is a
  // get-tuple-element with the given tuple index.
  constexpr auto WithTupleIndex(int64 tuple_index) const -> decltype(
      this->AppendImpl(HloInstructionPatternTupleIndexImpl(tuple_index))) {
    return AppendImpl(HloInstructionPatternTupleIndexImpl(tuple_index));
  }

  // Modifies the pattern to match only if the instruction is a parameter
  // with the given parameter number.
  constexpr auto WithParameterNum(int64 parameter_num) const -> decltype(
      this->AppendImpl(HloInstructionPatternParameterNumImpl(parameter_num))) {
    return AppendImpl(HloInstructionPatternParameterNumImpl(parameter_num));
  }

  // Modifies the pattern to match if the instruction is used exactly once.
  // Does not match if the instruction is used twice by the same user (e.g.
  // multiply(x,x)).
  constexpr auto WithOneUse() const
      -> decltype(this->AppendImpl(HloInstructionPatternOneUseImpl())) {
    return AppendImpl(HloInstructionPatternOneUseImpl());
  }

  // Modifies the pattern to match if the instruction is used by exactly one
  // other instruction.  Will match if the instruction is used twice, so long as
  // it's by the same user (e.g.  multiply(x,x)).
  constexpr auto WithOneUser() const
      -> decltype(this->AppendImpl(HloInstructionPatternOneUserImpl())) {
    return AppendImpl(HloInstructionPatternOneUserImpl());
  }

  void DescribeTo(std::ostream* os, int64 indent = 0) const {
    impl_.DescribeTo(os, indent);
  }

 private:
  Impl impl_;
  HloInstructionType** matched_inst_;
};

}  // namespace detail

// Creates an instruction pattern that will capture the matched instruction in
// the argument.
inline constexpr detail::HloInstructionPattern<
    const ::xla::HloInstruction, detail::HloInstructionPatternBaseImpl>
Op(const ::xla::HloInstruction** matched_inst = nullptr) {
  return detail::HloInstructionPattern<const ::xla::HloInstruction,
                                       detail::HloInstructionPatternBaseImpl>(
      detail::HloInstructionPatternBaseImpl(), matched_inst);
}

// Creates an instruction pattern that will capture the matched instruction in
// the argument.
inline constexpr detail::HloInstructionPattern<
    ::xla::HloInstruction, detail::HloInstructionPatternBaseImpl>
Op(::xla::HloInstruction** matched_inst) {
  return detail::HloInstructionPattern<::xla::HloInstruction,
                                       detail::HloInstructionPatternBaseImpl>(
      detail::HloInstructionPatternBaseImpl(), matched_inst);
}

// Helpers for nullary instructions.
#define XLA_NULLOP_PATTERN(NAME)                                      \
  inline auto NAME()->decltype(Op().WithOpcode(HloOpcode::k##NAME)) { \
    return Op().WithOpcode(HloOpcode::k##NAME);                       \
  }                                                                   \
                                                                      \
  template <typename HloInstructionType>                              \
  inline auto NAME(HloInstructionType** matched_inst)                 \
      ->decltype(Op(matched_inst).WithOpcode(HloOpcode::k##NAME)) {   \
    return Op(matched_inst).WithOpcode(HloOpcode::k##NAME);           \
  }
XLA_NULLOP_PATTERN(Constant)
XLA_NULLOP_PATTERN(Parameter)
XLA_NULLOP_PATTERN(Iota)
XLA_NULLOP_PATTERN(Rng)
#undef XLA_NULLOP_PATTERN

// Helpers for unary instructions.
#define XLA_UNOP_PATTERN(NAME)                                        \
  inline auto NAME()->decltype(Op().WithOpcode(HloOpcode::k##NAME)) { \
    return Op().WithOpcode(HloOpcode::k##NAME);                       \
  }                                                                   \
                                                                      \
  template <typename Arg>                                             \
  inline auto NAME(Arg&& arg)->decltype(                              \
      Op().WithOpcode(HloOpcode::k##NAME)                             \
          .WithOperand(0, std::forward<Arg>(arg))) {                  \
    return Op()                                                       \
        .WithOpcode(HloOpcode::k##NAME)                               \
        .WithOperand(0, std::forward<Arg>(arg));                      \
  }                                                                   \
                                                                      \
  template <typename HloInstructionType, typename Arg>                \
  inline auto NAME(HloInstructionType** matched_inst, Arg&& arg)      \
      ->decltype(Op(matched_inst)                                     \
                     .WithOpcode(HloOpcode::k##NAME)                  \
                     .WithOperand(0, std::forward<Arg>(arg))) {       \
    return Op(matched_inst)                                           \
        .WithOpcode(HloOpcode::k##NAME)                               \
        .WithOperand(0, std::forward<Arg>(arg));                      \
  }
XLA_UNOP_PATTERN(Abs)
XLA_UNOP_PATTERN(RoundNearestAfz)
XLA_UNOP_PATTERN(Bitcast)
XLA_UNOP_PATTERN(Broadcast)
XLA_UNOP_PATTERN(Ceil)
XLA_UNOP_PATTERN(Convert)
XLA_UNOP_PATTERN(Copy)
XLA_UNOP_PATTERN(Cos)
XLA_UNOP_PATTERN(CrossReplicaSum)
XLA_UNOP_PATTERN(Exp)
XLA_UNOP_PATTERN(Fft)
XLA_UNOP_PATTERN(Floor)
XLA_UNOP_PATTERN(GetTupleElement)
XLA_UNOP_PATTERN(Imag)
XLA_UNOP_PATTERN(Infeed)
XLA_UNOP_PATTERN(IsFinite)
XLA_UNOP_PATTERN(Log)
XLA_UNOP_PATTERN(Not)
XLA_UNOP_PATTERN(Negate)
XLA_UNOP_PATTERN(Real)
XLA_UNOP_PATTERN(Recv)
XLA_UNOP_PATTERN(RecvDone)
XLA_UNOP_PATTERN(ReducePrecision)
XLA_UNOP_PATTERN(Reshape)
XLA_UNOP_PATTERN(Reverse)
XLA_UNOP_PATTERN(SendDone)
XLA_UNOP_PATTERN(Sign)
XLA_UNOP_PATTERN(Sin)
XLA_UNOP_PATTERN(Slice)
XLA_UNOP_PATTERN(Sort)
XLA_UNOP_PATTERN(Tanh)
XLA_UNOP_PATTERN(Transpose)
#undef XLA_UNOP_PATTERN

// Helpers for binary instructions.
#define XLA_BINOP_PATTERN(NAME)                                             \
  inline auto NAME()->decltype(Op().WithOpcode(HloOpcode::k##NAME)) {       \
    return Op().WithOpcode(HloOpcode::k##NAME);                             \
  }                                                                         \
                                                                            \
  template <typename Lhs, typename Rhs>                                     \
  inline auto NAME(Lhs&& lhs, Rhs&& rhs)                                    \
      ->decltype(Op().WithOpcode(HloOpcode::k##NAME)                        \
                     .WithOperand(0, std::forward<Lhs>(lhs))                \
                     .WithOperand(1, std::forward<Rhs>(rhs))) {             \
    return Op()                                                             \
        .WithOpcode(HloOpcode::k##NAME)                                     \
        .WithOperand(0, std::forward<Lhs>(lhs))                             \
        .WithOperand(1, std::forward<Rhs>(rhs));                            \
  }                                                                         \
                                                                            \
  template <typename HloInstructionType, typename Lhs, typename Rhs>        \
  inline auto NAME(HloInstructionType** matched_inst, Lhs&& lhs, Rhs&& rhs) \
      ->decltype(Op(matched_inst)                                           \
                     .WithOpcode(HloOpcode::k##NAME)                        \
                     .WithOperand(0, std::forward<Lhs>(lhs))                \
                     .WithOperand(1, std::forward<Rhs>(rhs))) {             \
    return Op(matched_inst)                                                 \
        .WithOpcode(HloOpcode::k##NAME)                                     \
        .WithOperand(0, std::forward<Lhs>(lhs))                             \
        .WithOperand(1, std::forward<Rhs>(rhs));                            \
  }

#define XLA_COMMUTATIVE_BINOP_PATTERN(NAME)                                 \
  XLA_BINOP_PATTERN(NAME)                                                   \
                                                                            \
  template <typename HloInstructionType, typename Lhs, typename Rhs>        \
  inline auto NAME##AnyOrder(HloInstructionType** matched_inst, Lhs&& lhs,  \
                             Rhs&& rhs)                                     \
      ->decltype(Op(matched_inst)                                           \
                     .WithOpcode(HloOpcode::k##NAME)                        \
                     .WithBinaryOperandsAnyOrder(std::forward<Lhs>(lhs),    \
                                                 std::forward<Rhs>(rhs))) { \
    return Op(matched_inst)                                                 \
        .WithOpcode(HloOpcode::k##NAME)                                     \
        .WithBinaryOperandsAnyOrder(std::forward<Lhs>(lhs),                 \
                                    std::forward<Rhs>(rhs));                \
  }                                                                         \
  template <typename Lhs, typename Rhs>                                     \
  inline auto NAME##AnyOrder(Lhs&& lhs, Rhs&& rhs)                          \
      ->decltype(NAME##AnyOrder<const HloInstruction>(                      \
          nullptr, std::forward<Lhs>(lhs), std::forward<Rhs>(rhs))) {       \
    return NAME##AnyOrder<const HloInstruction>(                            \
        nullptr, std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));           \
  }
XLA_COMMUTATIVE_BINOP_PATTERN(Add)
XLA_BINOP_PATTERN(Atan2)
XLA_BINOP_PATTERN(Divide)
XLA_BINOP_PATTERN(Complex)
XLA_BINOP_PATTERN(Convolution)
XLA_BINOP_PATTERN(Dot)
XLA_BINOP_PATTERN(DynamicSlice)
XLA_COMMUTATIVE_BINOP_PATTERN(Eq)
XLA_BINOP_PATTERN(Gather)
XLA_BINOP_PATTERN(Ge)
XLA_BINOP_PATTERN(Gt)
XLA_BINOP_PATTERN(Le)
XLA_BINOP_PATTERN(Lt)
XLA_COMMUTATIVE_BINOP_PATTERN(Maximum)
XLA_COMMUTATIVE_BINOP_PATTERN(Minimum)
XLA_COMMUTATIVE_BINOP_PATTERN(Multiply)
XLA_COMMUTATIVE_BINOP_PATTERN(Ne)
XLA_BINOP_PATTERN(Outfeed)
XLA_BINOP_PATTERN(Pad)
XLA_BINOP_PATTERN(Power)
XLA_BINOP_PATTERN(ReduceWindow)
XLA_BINOP_PATTERN(Remainder)
XLA_BINOP_PATTERN(Send)
XLA_BINOP_PATTERN(Subtract)
XLA_COMMUTATIVE_BINOP_PATTERN(And)
XLA_COMMUTATIVE_BINOP_PATTERN(Or)
XLA_BINOP_PATTERN(ShiftLeft)
XLA_BINOP_PATTERN(ShiftRightArithmetic)
XLA_BINOP_PATTERN(ShiftRightLogical)
#undef XLA_COMMUTATIVE_BINOP_PATTERN
#undef XLA_BINOP_PATTERN

// Helpers for ternary instructions.
#define XLA_TERNOP_PATTERN(NAME)                                       \
  inline auto NAME()->decltype(Op().WithOpcode(HloOpcode::k##NAME)) {  \
    return Op().WithOpcode(HloOpcode::k##NAME);                        \
  }                                                                    \
                                                                       \
  template <typename Arg0, typename Arg1, typename Arg2>               \
  inline auto NAME(Arg0&& arg0, Arg1&& arg1, Arg2&& arg2)              \
      ->decltype(Op().WithOpcode(HloOpcode::k##NAME)                   \
                     .WithOperand(0, std::forward<Arg0>(arg0))         \
                     .WithOperand(1, std::forward<Arg1>(arg1))         \
                     .WithOperand(2, std::forward<Arg2>(arg2))) {      \
    return Op()                                                        \
        .WithOpcode(HloOpcode::k##NAME)                                \
        .WithOperand(0, std::forward<Arg0>(arg0))                      \
        .WithOperand(1, std::forward<Arg1>(arg1))                      \
        .WithOperand(2, std::forward<Arg2>(arg2));                     \
  }                                                                    \
                                                                       \
  template <typename HloInstructionType, typename Arg0, typename Arg1, \
            typename Arg2>                                             \
  inline auto NAME(HloInstructionType** matched_inst, Arg0&& arg0,     \
                   Arg1&& arg1, Arg2&& arg2)                           \
      ->decltype(Op(matched_inst)                                      \
                     .WithOpcode(HloOpcode::k##NAME)                   \
                     .WithOperand(0, std::forward<Arg0>(arg0))         \
                     .WithOperand(1, std::forward<Arg1>(arg1))         \
                     .WithOperand(2, std::forward<Arg2>(arg2))) {      \
    return Op(matched_inst)                                            \
        .WithOpcode(HloOpcode::k##NAME)                                \
        .WithOperand(0, std::forward<Arg0>(arg0))                      \
        .WithOperand(1, std::forward<Arg1>(arg1))                      \
        .WithOperand(2, std::forward<Arg2>(arg2));                     \
  }
XLA_TERNOP_PATTERN(Clamp);
XLA_TERNOP_PATTERN(Scatter);
XLA_TERNOP_PATTERN(Select);
#undef XLA_TERNOP_PATTERN

namespace detail {
template <typename Matcher, typename FirstArg>
inline auto WithOperands(Matcher&& m, int64 operand_num, FirstArg&& first_arg)
    -> decltype(m.WithOperand(operand_num, std::forward<FirstArg>(first_arg))) {
  return m.WithOperand(operand_num, std::forward<FirstArg>(first_arg));
}

template <typename Matcher, typename FirstArg, typename... Args>
inline auto WithOperands(Matcher&& m, int64 operand_num, FirstArg&& first_arg,
                         Args&&... args)
    -> decltype(WithOperands(m.WithOperand(operand_num,
                                           std::forward<FirstArg>(first_arg)),
                             operand_num + 1, std::forward<Args>(args)...)) {
  return WithOperands(
      m.WithOperand(operand_num, std::forward<FirstArg>(first_arg)),
      operand_num + 1, std::forward<Args>(args)...);
}
}  // namespace detail

#define XLA_VARIADIC_OP_PATTERN(NAME)                                         \
  inline auto NAME()->decltype(Op().WithOpcode(HloOpcode::k##NAME)) {         \
    return Op().WithOpcode(HloOpcode::k##NAME);                               \
  }                                                                           \
                                                                              \
  template <typename... Args>                                                 \
  inline auto NAME(Args&&... args)                                            \
      ->decltype(detail::WithOperands(Op().WithOpcode(HloOpcode::k##NAME)     \
                                          .WithNumOperands(sizeof...(Args)),  \
                                      0, std::forward<Args>(args)...)) {      \
    return detail::WithOperands(                                              \
        Op().WithOpcode(HloOpcode::k##NAME).WithNumOperands(sizeof...(Args)), \
        /*operand_num=*/0, std::forward<Args>(args)...);                      \
  }                                                                           \
                                                                              \
  template <typename HloInstructionType, typename... Args>                    \
  inline auto NAME(HloInstructionType** matched_inst, Args&&... args)         \
      ->decltype(detail::WithOperands(Op(matched_inst)                        \
                                          .WithOpcode(HloOpcode::k##NAME)     \
                                          .WithNumOperands(sizeof...(Args)),  \
                                      0, std::forward<Args>(args)...)) {      \
    return detail::WithOperands(Op(matched_inst)                              \
                                    .WithOpcode(HloOpcode::k##NAME)           \
                                    .WithNumOperands(sizeof...(Args)),        \
                                /*operand_num=*/0,                            \
                                std::forward<Args>(args)...);                 \
  }

// We could implement all ops as "variadic" ops, but it would make the
// already-bad compile errors even worse.
XLA_VARIADIC_OP_PATTERN(AfterAll);
XLA_VARIADIC_OP_PATTERN(Concatenate);
XLA_VARIADIC_OP_PATTERN(CustomCall);
XLA_VARIADIC_OP_PATTERN(Map)
XLA_VARIADIC_OP_PATTERN(Reduce);
XLA_VARIADIC_OP_PATTERN(Tuple);

// Helpers for matching non-constant instructions.
inline auto NonConstant() -> decltype(Op().IsNonConstant()) {
  return Op().IsNonConstant();
}

template <typename HloInstructionType>
inline auto NonConstant(HloInstructionType** matched_inst)
    -> decltype(Op(matched_inst).IsNonConstant()) {
  return Op(matched_inst).IsNonConstant();
}

// Add overloads for GetTupleElement which take a int64 specifying which tuple
// element is selected.
template <typename Arg>
inline auto GetTupleElement(Arg&& arg, int64 tuple_index)
    -> decltype(Op().WithOpcode(HloOpcode::kGetTupleElement)
                    .WithOperand(0, std::forward<Arg>(arg))
                    .WithTupleIndex(tuple_index)) {
  return Op()
      .WithOpcode(HloOpcode::kGetTupleElement)
      .WithOperand(0, std::forward<Arg>(arg))
      .WithTupleIndex(tuple_index);
}

template <typename HloInstructionType, typename Arg>
inline auto GetTupleElement(HloInstructionType** matched_inst, Arg&& arg,
                            int64 tuple_index)
    -> decltype(Op(matched_inst)
                    .WithOpcode(HloOpcode::kGetTupleElement)
                    .WithOperand(0, std::forward<Arg>(arg))
                    .WithTupleIndex(tuple_index)) {
  return Op(matched_inst)
      .WithOpcode(HloOpcode::kGetTupleElement)
      .WithOperand(0, std::forward<Arg>(arg))
      .WithTupleIndex(tuple_index);
}

// Add overloads for Parameter which take an int64 specifying the parameter
// number.
inline auto Parameter(int64 parameter_num) -> decltype(
    Op().WithOpcode(HloOpcode::kParameter).WithParameterNum(parameter_num)) {
  return Op().WithOpcode(HloOpcode::kParameter).WithParameterNum(parameter_num);
}
template <typename HloInstructionType>
inline auto Parameter(HloInstructionType** matched_inst, int64 parameter_num)
    -> decltype(Op(matched_inst)
                    .WithOpcode(HloOpcode::kParameter)
                    .WithParameterNum(parameter_num)) {
  return Op(matched_inst)
      .WithOpcode(HloOpcode::kParameter)
      .WithParameterNum(parameter_num);
}

inline auto ConstantScalar() -> decltype(Op().IsConstantScalar()) {
  return Op().IsConstantScalar();
}

template <typename HloInstructionType>
inline auto ConstantScalar(HloInstructionType** matched_inst)
    -> decltype(Op(matched_inst).IsConstantScalar()) {
  return Op(matched_inst).IsConstantScalar();
}

template <typename ScalarTy>
inline auto ConstantScalar(ScalarTy val)
    -> decltype(Op().IsConstantScalar(val)) {
  return Op().IsConstantScalar(val);
}

template <typename HloInstructionType, typename ScalarTy>
inline auto ConstantScalar(HloInstructionType** matched_inst, ScalarTy val)
    -> decltype(Op(matched_inst).IsConstantScalar(val)) {
  return Op(matched_inst).IsConstantScalar(val);
}

inline auto ConstantEffectiveScalar() -> decltype(Op().IsConstantScalar()) {
  return Op().IsConstantEffectiveScalar();
}

template <typename HloInstructionType>
inline auto ConstantEffectiveScalar(HloInstructionType** matched_inst)
    -> decltype(Op(matched_inst).IsConstantScalar()) {
  return Op(matched_inst).IsConstantEffectiveScalar();
}

template <typename ScalarTy>
inline auto ConstantEffectiveScalar(ScalarTy val)
    -> decltype(Op().IsConstantEffectiveScalar(val)) {
  return Op().IsConstantEffectiveScalar(val);
}

template <typename HloInstructionType, typename ScalarTy>
inline auto ConstantEffectiveScalar(HloInstructionType** matched_inst,
                                    ScalarTy val)
    -> decltype(Op(matched_inst).IsConstantEffectiveScalar(val)) {
  return Op(matched_inst).IsConstantEffectiveScalar(val);
}

}  // namespace match

}  // namespace xla

#undef EXPLAIN
#pragma pop_macro("EXPLAIN")
#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_PATTERN_MATCHER_H_
