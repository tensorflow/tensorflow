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
// Each pattern type has the following modifiers:
//
//   Op():
//     - WithName: match operations with the given name
//     - WithOpcode: match operations with the given opcode
//     - WithShape: match operations whose shape matches the given pattern
//     - WithOperand: match operations whose operand matches the given pattern
//
//   Shape():
//     - EqualTo: matches shapes that are equal to the argument
//     - CompatibleTo: matches shapes that are compatible to the argument
//     - IsScalar/IsArray/IsTuple: matches scalar/array/tuple shapes
//     - IsDenseArray/IsSparseArray: matches arrays with dense/sparse format
//     - WithLayout: match shapes whose layout matches the given pattern
//     - WithLayoutEqualTo: matches shapes whose layouts equal the argument
//     - WithSubshape: matches tuple shapes whose subshape matches the given
//       pattern
//     - WithSubshapeEqualTo: matches shapes with a subshape equal the argument
//     - WithElementType: matches array/scalar shapes with the given element
//       type
//     - WithRank: matches array/scalar types with the given rank
//
//  Layout():
//     - EqualTo: matches layouts that are equal to the argument
//     - WithDenseFormat/WithSparseFormat: matches layouts with dense/sparse
//       format
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
// Helpers are provided for common nullary, unary, binary, and ternary
// instructions. These helpers can be called with no arguments, in which case
// they will match any instruction matching the opcode. They may also be called
// with matches for the operands and with an optional capture. (The capture must
// be the first argument.) Some examples of these helpers and their equivalents
// are provided below.
//
// Example nullary instruction:
//   Param()                        == Op().WithOpcode(HloOpcode::kParam)
//   Param(&a)                      == Op(&a).WithOpcode(HloOpcode::kParam)
//
// Example unary instruction:
//   Abs()                             == Op().WithOpcode(HloOpcode::kAbs)
//   Abs(Op(&a))                       == Op().WithOpcode(HloOpcode::kAbs)
//                                            .WithOperand(0, Op(&a)))
//   Abs(&a, Op(&b))                   == Op(&a).WithOpcode(HloOpcode::kAbs)
//                                              .WithOperand(0, Op(&b))
//
// Example binary instruction:
//   Add()                             == Op().WithOpcode(HloOpcode::kAdd)
//   Add(Op(&a), Op(&b))               == Op().WithOpcode(HloOpcode::kAdd)
//                                            .WithOperand(0, Op(&a))
//                                            .WithOperand(1, Op(&b))
//   Add(&a, Op(&b), Op(&c))           == Op(&a).WithOpcode(HloOpcode::kAdd)
//                                              .WithOperand(0, Op(&b))
//                                              .WithOperand(1, Op(&c))
//
// Example ternary instruction:
//   Clamp()                           == Op().WithOpcode(HloOpcode::kClamp)
//   Clamp(Op(&a), Op(&b), Op(&c))     == Op().WithOpcode(HloOpcode::kClamp)
//                                            .WithOperand(0, Op(&a))
//                                            .WithOperand(1, Op(&b))
//                                            .WithOperand(2, Op(&c))
//   Clamp(&a, Op(&b), Op(&c), Op(&d)) == Op(&a).WithOpcode(HloOpcode::kClamp)
//                                              .WithOperand(0, Op(&b))
//                                              .WithOperand(1, Op(&c))
//                                              .WithOperand(2, Op(&d))
//

struct MatchOption {
  // If true, actually capture matched item into the user pointer.
  bool capture;
};

template <typename Value, typename Pattern>
bool Match(Value* value, const Pattern& pattern,
           MatchOption option = {/*.capture=*/true}) {
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

 private:
  template <typename ItemType, size_t index>
  bool MatchImpl(ItemType* item, MatchOption option,
                 std::integral_constant<size_t, index>) const {
    return std::get<index>(patterns_).Match(item, option) &&
           MatchImpl(item, option, std::integral_constant<size_t, index + 1>());
  }

  template <typename ItemType>
  bool MatchImpl(ItemType* item, MatchOption option,
                 std::integral_constant<size_t, sizeof...(Patterns)>) const {
    return true;
  }

  std::tuple<Patterns...> patterns_;
};

}  // namespace detail

// Returns a pattern that represents the conjunction of all input patterns. All
// patterns need to match in order to have the AllOf pattern match.
//
// TODO(timshen): Currently AllOf is still nested, e.g. AllOf<AllOf<A>, B> is
// not AllOf<A, B>. We might want to flatten the AllOf type structure if the
// C++ compile error message gets annoying.
template <typename Item, typename... Patterns>
detail::AllOfPattern<typename std::remove_const<Item>::type, Patterns...> AllOf(
    const Patterns&... patterns) {
  return detail::AllOfPattern<typename std::remove_const<Item>::type,
                              Patterns...>(patterns...);
}

namespace detail {

template <typename LayoutType, typename Impl>
class LayoutPattern;

// The base LayoutPattern implementation. Matches only if the layout is not
// nullptr.
class LayoutPatternBaseImpl {
 public:
  bool Match(const ::xla::Layout* layout, MatchOption option) const {
    return layout != nullptr;
  }
};

// A LayoutPattern implementation that matches only if the layout equals a
// Layout proto.
class LayoutPatternEqualImpl {
 public:
  explicit constexpr LayoutPatternEqualImpl(const ::xla::Layout* layout)
      : layout_(layout) {}

  bool Match(const ::xla::Layout* layout, MatchOption option) const {
    return LayoutUtil::Equal(*layout_, *layout);
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
    return layout->format() == format_;
  }

 private:
  Format format_;
};

// A pattern that matches Layouts.
template <typename LayoutType, typename Impl>
class LayoutPattern {
 private:
  template <typename NewImpl>
  LayoutPattern<LayoutType, AllOfPattern<::xla::Layout, Impl, NewImpl>>
  AppendImpl(NewImpl new_impl) const {
    return LayoutPattern<LayoutType,
                         AllOfPattern<::xla::Layout, Impl, NewImpl>>(
        AllOf<Layout>(impl_, std::move(new_impl)), matched_layout_);
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
    return MatchImpl(item, option, std::integral_constant<size_t, 0>());
  }

  bool Match(Item* item, MatchOption option) const {
    return MatchImpl(item, option, std::integral_constant<size_t, 0>());
  }

 private:
  template <typename ItemType, size_t index>
  bool MatchImpl(ItemType* item, MatchOption option,
                 std::integral_constant<size_t, index>) const {
    auto new_option = option;
    new_option.capture = false;
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
        bool ret = std::get<index>(patterns_).Match(item, option);
        DCHECK(ret);
      }
      return true;
    }
    return MatchImpl(item, option, std::integral_constant<size_t, index + 1>());
  }

  template <typename ItemType>
  bool MatchImpl(ItemType* item, MatchOption option,
                 std::integral_constant<size_t, sizeof...(Patterns)>) const {
    return false;
  }

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
    return shape != nullptr;
  }
};

// A ShapePattern implementation that matches only if the shape equals a Shape
// proto.
class ShapePatternEqualImpl {
 public:
  explicit constexpr ShapePatternEqualImpl(const ::xla::Shape* shape)
      : shape_(shape) {}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    return ShapeUtil::Equal(*shape_, *shape);
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
    return ShapeUtil::Compatible(*shape_, *shape);
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
    return shape->element_type() == element_type_;
  }

 private:
  PrimitiveType element_type_;
};

// A ShapePattern implementation that matches only if the shape is scalar.
class ShapePatternIsScalarImpl {
 public:
  explicit constexpr ShapePatternIsScalarImpl() {}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    return ShapeUtil::IsScalar(*shape);
  }
};

// A ShapePattern implementation that matches only if the shape is an array
class ShapePatternIsArrayImpl {
 public:
  explicit constexpr ShapePatternIsArrayImpl() {}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    return ShapeUtil::IsArray(*shape);
  }
};

// A ShapePattern implementation that matches only if the shape is a tuple.
class ShapePatternIsTupleImpl {
 public:
  explicit constexpr ShapePatternIsTupleImpl() {}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    return ShapeUtil::IsTuple(*shape);
  }
};

// A ShapePattern implementation that matches only if the shape has a given
// rank.
class ShapePatternRankImpl {
 public:
  explicit constexpr ShapePatternRankImpl(int64 rank) : rank_(rank) {}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    return ShapeUtil::Rank(*shape) == rank_;
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
    return LayoutUtil::HasLayout(*shape) &&
           layout_.Match(shape->mutable_layout(), option);
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
    return ShapeUtil::IndexIsValid(*shape, index_) &&
           subshape_.Match(&ShapeUtil::GetSubshape(*shape, index_), option);
  }

  bool Match(::xla::Shape* shape, MatchOption option) const {
    return ShapeUtil::IndexIsValid(*shape, index_) &&
           subshape_.Match(ShapeUtil::GetMutableSubshape(shape, index_),
                           option);
  }

 private:
  ShapeIndexView index_;
  ShapePattern<SubshapeType, SubshapeImpl> subshape_;
};

// A pattern that matches Shapes.
template <typename ShapeType, typename Impl>
class ShapePattern {
 private:
  template <typename NewImpl>
  ShapePattern<ShapeType, AllOfPattern<::xla::Shape, Impl, NewImpl>> AppendImpl(
      NewImpl new_impl) const {
    return ShapePattern<ShapeType, AllOfPattern<::xla::Shape, Impl, NewImpl>>(
        AllOf<Shape>(impl_, std::move(new_impl)), matched_shape_);
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
    return false;
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

template <typename HloInstructionType, typename Impl>
class HloInstructionPattern;

// The base HloInstructionPattern implementation. Matches only if the
// instruction is not nullptr.
class HloInstructionPatternBaseImpl {
 public:
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    return inst != nullptr;
  }
};

// An HloInstructionPattern implementation that matches only if the instruction
// has a given name.
class HloInstructionPatternNameImpl {
 public:
  explicit HloInstructionPatternNameImpl(absl::string_view name)
      : name_(name) {}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    return inst->name() == name_;
  }

 private:
  absl::string_view name_;
};

// An HloInstructionPattern implementation that matches only if the instruction
// has a given opcode.
class HloInstructionPatternOpcodeImpl {
 public:
  explicit constexpr HloInstructionPatternOpcodeImpl(HloOpcode opcode,
                                                     bool invert)
      : opcode_(opcode), invert_(invert) {}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    return (invert_ ^ (inst->opcode() == opcode_));
  }

 private:
  HloOpcode opcode_;
  bool invert_;
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
    return shape_.Match(&inst->shape(), option);
  }

  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    return shape_.Match(inst->mutable_shape(), option);
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
    return operand_index_ < inst->operand_count() &&
           operand_.Match(inst->operand(operand_index_), option);
  }

  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    return operand_index_ < inst->operand_count() &&
           operand_.Match(inst->mutable_operand(operand_index_), option);
  }

 private:
  int64 operand_index_;
  HloInstructionPattern<OperandType, OperandImpl> operand_;
};

// An HloInstructionPattern implementation that matches only if the instruction
// is a fusion node with a particular kind.
class HloInstructionPatternFusionKindImpl {
 public:
  explicit constexpr HloInstructionPatternFusionKindImpl(
      ::xla::HloInstruction::FusionKind kind)
      : kind_(kind) {}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    return inst->opcode() == HloOpcode::kFusion && inst->fusion_kind() == kind_;
  }

  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    return inst->opcode() == HloOpcode::kFusion && inst->fusion_kind() == kind_;
  }

 private:
  ::xla::HloInstruction::FusionKind kind_;
};

// An HloInstructionPattern implementation that matches only if the instruction
// is a kGetTupleElement with a particular tuple index.
class HloInstructionPatternTupleIndexImpl {
 public:
  explicit constexpr HloInstructionPatternTupleIndexImpl(int64 tuple_index)
      : tuple_index_(tuple_index) {}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    return inst->opcode() == HloOpcode::kGetTupleElement &&
           inst->tuple_index() == tuple_index_;
  }

  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    return inst->opcode() == HloOpcode::kGetTupleElement &&
           inst->tuple_index() == tuple_index_;
  }

 private:
  int64 tuple_index_;
};

template <typename ItemType, typename Predicate>
class HloPredicatePatternImpl {
 public:
  explicit HloPredicatePatternImpl(Predicate pred) : pred_(std::move(pred)) {}

  bool Match(const ItemType* item, MatchOption option) const {
    return pred_(item);
  }

  bool Match(ItemType* item, MatchOption option) const { return pred_(item); }

 private:
  Predicate pred_;
};

struct PatternFriend;

// A pattern that matches HloInstructions.
template <typename HloInstructionType, typename Impl>
class HloInstructionPattern {
 private:
  template <typename NewImpl>
  HloInstructionPattern<HloInstructionType,
                        AllOfPattern<::xla::HloInstruction, Impl, NewImpl>>
  AppendImpl(NewImpl new_impl) const {
    return HloInstructionPattern<
        HloInstructionType, AllOfPattern<::xla::HloInstruction, Impl, NewImpl>>(
        AllOf<HloInstruction>(impl_, std::move(new_impl)), matched_inst_);
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

  // Modifies the pattern to match only if the instruction does not have the
  // given opcode.
  auto WithoutOpcode(HloOpcode opcode) const
      -> decltype(this->AppendImpl(HloInstructionPatternOpcodeImpl(opcode,
                                                                   true))) {
    return AppendImpl(HloInstructionPatternOpcodeImpl(opcode, true));
  }

  // Modifies the pattern to match only if the instruction is a constant.
  constexpr auto IsConstant() const
      -> decltype(this->WithOpcode(HloOpcode::kConstant)) {
    return WithOpcode(HloOpcode::kConstant);
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

 private:
  template <typename Predicate>
  constexpr auto WithPredicate(Predicate pred) const -> decltype(
      this->AppendImpl(HloPredicatePatternImpl<HloInstruction, Predicate>(
          std::move(pred)))) {
    return AppendImpl(
        HloPredicatePatternImpl<HloInstruction, Predicate>(std::move(pred)));
  }

  friend struct PatternFriend;

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
XLA_UNOP_PATTERN(Copy)
XLA_UNOP_PATTERN(Cos)
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
XLA_UNOP_PATTERN(Reduce)
XLA_UNOP_PATTERN(ReducePrecision)
XLA_UNOP_PATTERN(Reshape)
XLA_UNOP_PATTERN(Reverse)
XLA_UNOP_PATTERN(SendDone)
XLA_UNOP_PATTERN(Sign)
XLA_UNOP_PATTERN(Sin)
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
  template <typename Lhs, typename Rhs>                                     \
  inline auto NAME##AnyOrder(Lhs&& lhs, Rhs&& rhs)                          \
      ->decltype(AnyOf<HloInstruction>(NAME(lhs, rhs), NAME(rhs, lhs))) {   \
    return AnyOf<HloInstruction>(NAME(lhs, rhs), NAME(rhs, lhs));           \
  }                                                                         \
                                                                            \
  template <typename HloInstructionType, typename Lhs, typename Rhs>        \
  inline auto NAME##AnyOrder(HloInstructionType** matched_inst, Lhs&& lhs,  \
                             Rhs&& rhs)                                     \
      ->decltype(AnyOf<HloInstructionType>(NAME(matched_inst, lhs, rhs),    \
                                           NAME(matched_inst, rhs, lhs))) { \
    return AnyOf<HloInstructionType>(NAME(matched_inst, lhs, rhs),          \
                                     NAME(matched_inst, rhs, lhs));         \
  }
XLA_COMMUTATIVE_BINOP_PATTERN(Add)
XLA_BINOP_PATTERN(Atan2)
XLA_BINOP_PATTERN(Divide)
XLA_BINOP_PATTERN(Complex)
XLA_BINOP_PATTERN(Dot)
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
XLA_BINOP_PATTERN(Power)
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
XLA_TERNOP_PATTERN(Select);
#undef XLA_TERNOP_PATTERN

namespace detail {
struct PatternFriend {
  template <typename T>
  static auto ConstantScalar(T constant) -> decltype(
      Constant()
          .WithShape(match::Shape().IsScalar())
          .WithPredicate(
              std::declval<std::function<bool(const HloInstruction*)>>())) {
    std::function<bool(const HloInstruction*)> pred =
        [constant](const HloInstruction* instr) {
          const auto& literal = Cast<HloConstantInstruction>(instr)->literal();
          auto status_or_const = LiteralUtil::CreateR0(constant).Convert(
              literal.shape().element_type());
          return status_or_const.ok() &&
                 literal == status_or_const.ConsumeValueOrDie();
        };

    return Constant()
        .WithShape(match::Shape().IsScalar())
        .WithPredicate(std::move(pred));
  }
};
}  // namespace detail

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

template <typename T>
inline auto ConstantScalar(T constant)
    -> decltype(detail::PatternFriend::ConstantScalar(constant)) {
  return detail::PatternFriend::ConstantScalar(constant);
}

}  // namespace match

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_PATTERN_MATCHER_H_
