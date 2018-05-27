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

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"

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
//   Recv()                            == Op().WithOpcode(HloOpcode::kRecv)
//   Recv(&a)                          == Op(&a).WithOpcode(HloOpcode::kRecv)
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
template <typename Value, typename Pattern>
bool Match(Value* value, const Pattern& pattern) {
  return pattern.Match(value);
}

namespace match {

namespace detail {

template <typename LayoutType, typename Impl>
class LayoutPattern;

// The base LayoutPattern implementation. Matches only if the layout is not
// nullptr.
class LayoutPatternBaseImpl {
 public:
  bool Match(const ::xla::Layout* layout) const { return layout != nullptr; }
};

// A LayoutPattern implementation that matches only if the layout equals a
// Layout proto.
template <typename Previous>
class LayoutPatternEqualImpl {
 public:
  explicit constexpr LayoutPatternEqualImpl(const Previous& previous,
                                            const ::xla::Layout* layout)
      : previous_(previous), layout_(layout) {}

  bool Match(const ::xla::Layout* layout) const {
    return previous_.Match(layout) && LayoutUtil::Equal(*layout_, *layout);
  }

 private:
  Previous previous_;
  const ::xla::Layout* layout_;
};

// A LayoutPattern implementation that matches only if the layout has a given
// format.
template <typename Previous>
class LayoutPatternFormatImpl {
 public:
  explicit constexpr LayoutPatternFormatImpl(const Previous& previous,
                                             Format format)
      : previous_(previous), format_(format) {}

  bool Match(const ::xla::Layout* layout) const {
    return previous_.Match(layout) && layout->format() == format_;
  }

 private:
  Previous previous_;
  Format format_;
};

// A pattern that matches Layouts.
template <typename LayoutType, typename Impl>
class LayoutPattern {
 public:
  explicit constexpr LayoutPattern(const Impl& impl,
                                   LayoutType** matched_layout)
      : impl_(impl), matched_layout_(matched_layout) {}

  // Returns true and captures the layout iff it matches the pattern.
  bool Match(const ::xla::Layout* layout) const {
    if (impl_.Match(layout)) {
      if (matched_layout_) {
        *matched_layout_ = layout;
      }
      return true;
    }
    return false;
  }

  // Returns true and captures the layout iff it matches the pattern.
  bool Match(::xla::Layout* layout) const {
    if (impl_.Match(layout)) {
      if (matched_layout_) {
        *matched_layout_ = layout;
      }
      return true;
    }
    return false;
  }

  // Modifies the pattern to match only if the layout equals the given proto.
  // The layout must outlive the returned pattern.
  constexpr LayoutPattern<LayoutType, LayoutPatternEqualImpl<Impl>> EqualTo(
      const Layout* layout) const {
    return LayoutPattern<LayoutType, LayoutPatternEqualImpl<Impl>>(
        LayoutPatternEqualImpl<Impl>(impl_, layout), matched_layout_);
  }

  // Modifies the pattern to match only if the layout has a dense format.
  constexpr LayoutPattern<LayoutType, LayoutPatternFormatImpl<Impl>>
  WithDenseFormat() const {
    return LayoutPattern<LayoutType, LayoutPatternFormatImpl<Impl>>(
        LayoutPatternFormatImpl<Impl>(impl_, DENSE), matched_layout_);
  }

  // Modifies the pattern to match only if the layout has a sparse format.
  constexpr LayoutPattern<LayoutType, LayoutPatternFormatImpl<Impl>>
  WithSparseFormat() const {
    return LayoutPattern<LayoutType, LayoutPatternFormatImpl<Impl>>(
        LayoutPatternFormatImpl<Impl>(impl_, SPARSE), matched_layout_);
  }

 private:
  Impl impl_;
  LayoutType** matched_layout_;
};

}  // namespace detail

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
  bool Match(const ::xla::Shape* shape) const { return shape != nullptr; }
};

// A ShapePattern implementation that matches only if the shape equals a Shape
// proto.
template <typename Previous>
class ShapePatternEqualImpl {
 public:
  explicit constexpr ShapePatternEqualImpl(const Previous& previous,
                                           const ::xla::Shape* shape)
      : previous_(previous), shape_(shape) {}

  bool Match(const ::xla::Shape* shape) const {
    return previous_.Match(shape) && ShapeUtil::Equal(*shape_, *shape);
  }

 private:
  Previous previous_;
  const ::xla::Shape* shape_;
};

// A ShapePattern implementation that matches only if the shape is compatible to
// a Shape proto.
template <typename Previous>
class ShapePatternCompatibleImpl {
 public:
  explicit constexpr ShapePatternCompatibleImpl(const Previous& previous,
                                                const ::xla::Shape* shape)
      : previous_(previous), shape_(shape) {}

  bool Match(const ::xla::Shape* shape) const {
    return previous_.Match(shape) && ShapeUtil::Compatible(*shape_, *shape);
  }

 private:
  Previous previous_;
  const ::xla::Shape* shape_;
};

// A ShapePattern implementation that matches only if the shape has a given
// element type.
template <typename Previous>
class ShapePatternElementTypeImpl {
 public:
  explicit constexpr ShapePatternElementTypeImpl(const Previous& previous,
                                                 PrimitiveType element_type)
      : previous_(previous), element_type_(element_type) {}

  bool Match(const ::xla::Shape* shape) const {
    return previous_.Match(shape) && shape->element_type() == element_type_;
  }

 private:
  Previous previous_;
  PrimitiveType element_type_;
};

// A ShapePattern implementation that matches only if the shape is scalar.
template <typename Previous>
class ShapePatternIsScalarImpl {
 public:
  explicit constexpr ShapePatternIsScalarImpl(const Previous& previous)
      : previous_(previous) {}

  bool Match(const ::xla::Shape* shape) const {
    return previous_.Match(shape) && ShapeUtil::IsScalar(*shape);
  }

 private:
  Previous previous_;
};

// A ShapePattern implementation that matches only if the shape is an array
template <typename Previous>
class ShapePatternIsArrayImpl {
 public:
  explicit constexpr ShapePatternIsArrayImpl(const Previous& previous)
      : previous_(previous) {}

  bool Match(const ::xla::Shape* shape) const {
    return previous_.Match(shape) && ShapeUtil::IsArray(*shape);
  }

 private:
  Previous previous_;
};

// A ShapePattern implementation that matches only if the shape is a tuple.
template <typename Previous>
class ShapePatternIsTupleImpl {
 public:
  explicit constexpr ShapePatternIsTupleImpl(const Previous& previous)
      : previous_(previous) {}

  bool Match(const ::xla::Shape* shape) const {
    return previous_.Match(shape) && ShapeUtil::IsTuple(*shape);
  }

 private:
  Previous previous_;
};

// A ShapePattern implementation that matches only if the shape has a given
// rank.
template <typename Previous>
class ShapePatternRankImpl {
 public:
  explicit constexpr ShapePatternRankImpl(const Previous& previous, int64 rank)
      : previous_(previous), rank_(rank) {}

  bool Match(const ::xla::Shape* shape) const {
    return previous_.Match(shape) && ShapeUtil::Rank(*shape) == rank_;
  }

 private:
  Previous previous_;
  int64 rank_;
};

// A ShapePattern implementation that matches only if the shape has a layout
// that matches a given pattern.
template <typename Previous, typename LayoutType, typename LayoutImpl>
class ShapePatternLayoutImpl {
 public:
  explicit constexpr ShapePatternLayoutImpl(
      const Previous& previous,
      const LayoutPattern<LayoutType, LayoutImpl>& layout)
      : previous_(previous), layout_(layout) {}

  bool Match(const ::xla::Shape* shape) const {
    return previous_.Match(shape) && LayoutUtil::HasLayout(*shape) &&
           layout_.Match(&shape->layout());
  }

  bool Match(Shape* shape) const {
    return previous_.Match(shape) && LayoutUtil::HasLayout(*shape) &&
           layout_.Match(shape->mutable_layout());
  }

 private:
  Previous previous_;
  LayoutPattern<LayoutType, LayoutImpl> layout_;
};

// A ShapePattern implementation that matches only if the shape has a subshape
// that matches a given pattern.
template <typename Previous, typename SubshapeType, typename SubshapeImpl>
class ShapePatternSubshapeImpl {
 public:
  explicit ShapePatternSubshapeImpl(
      const Previous& previous, ShapeIndexView index,
      const ShapePattern<SubshapeType, SubshapeImpl>& subshape)
      : previous_(previous), index_(index), subshape_(subshape) {}

  bool Match(const ::xla::Shape* shape) const {
    return previous_.Match(shape) && ShapeUtil::IndexIsValid(*shape, index_) &&
           subshape_.Match(&ShapeUtil::GetSubshape(*shape, index_));
  }

  bool Match(::xla::Shape* shape) const {
    return previous_.Match(shape) && ShapeUtil::IndexIsValid(*shape, index_) &&
           subshape_.Match(ShapeUtil::GetMutableSubshape(shape, index_));
  }

 private:
  Previous previous_;
  ShapeIndexView index_;
  ShapePattern<SubshapeType, SubshapeImpl> subshape_;
};

// A pattern that matches Shapes.
template <typename ShapeType, typename Impl>
class ShapePattern {
 public:
  explicit constexpr ShapePattern(const Impl& impl, ShapeType** matched_shape)
      : impl_(impl), matched_shape_(matched_shape) {}

  // Returns true and captures the shape iff it matches the pattern.
  bool Match(const ::xla::Shape* shape) const {
    if (impl_.Match(shape)) {
      if (matched_shape_) {
        *matched_shape_ = shape;
      }
      return true;
    }
    return false;
  }

  // Returns true and captures the shape iff it matches the pattern.
  bool Match(::xla::Shape* shape) const {
    if (impl_.Match(shape)) {
      if (matched_shape_) {
        *matched_shape_ = shape;
      }
      return true;
    }
    return false;
  }

  // Modifies the pattern to match only if the shape equals the given proto.
  // The layout must outlive the returned pattern.
  constexpr ShapePattern<ShapeType, ShapePatternEqualImpl<Impl>> EqualTo(
      const ::xla::Shape* shape) const {
    return ShapePattern<ShapeType, ShapePatternEqualImpl<Impl>>(
        ShapePatternEqualImpl<Impl>(impl_, shape), matched_shape_);
  }

  // Modifies the pattern to match only if the shape is compatible to the given
  // proto. The layout must outlive the returned pattern.
  constexpr ShapePattern<ShapeType, ShapePatternCompatibleImpl<Impl>>
  CompatibleTo(const ::xla::Shape* shape) const {
    return ShapePattern<ShapeType, ShapePatternCompatibleImpl<Impl>>(
        ShapePatternCompatibleImpl<Impl>(impl_, shape), matched_shape_);
  }

  // Modifies the pattern to match only if the shape has the given element type.
  constexpr ShapePattern<ShapeType, ShapePatternElementTypeImpl<Impl>>
  WithElementType(PrimitiveType element_type) const {
    return ShapePattern<ShapeType, ShapePatternElementTypeImpl<Impl>>(
        ShapePatternElementTypeImpl<Impl>(impl_, element_type), matched_shape_);
  }

  // Modifies the pattern to match only if the shape is scalar.
  constexpr ShapePattern<ShapeType, ShapePatternIsScalarImpl<Impl>> IsScalar()
      const {
    return ShapePattern<ShapeType, ShapePatternIsScalarImpl<Impl>>(
        ShapePatternIsScalarImpl<Impl>(impl_), matched_shape_);
  }

  // Modifies the pattern to match only if the shape is an array.
  constexpr ShapePattern<ShapeType, ShapePatternIsArrayImpl<Impl>> IsArray()
      const {
    return ShapePattern<ShapeType, ShapePatternIsArrayImpl<Impl>>(
        ShapePatternIsArrayImpl<Impl>(impl_), matched_shape_);
  }

  // Modifies the pattern to match only if the shape is a tuple.
  constexpr ShapePattern<ShapeType, ShapePatternIsTupleImpl<Impl>> IsTuple()
      const {
    return ShapePattern<ShapeType, ShapePatternIsTupleImpl<Impl>>(
        ShapePatternIsTupleImpl<Impl>(impl_), matched_shape_);
  }

  // Modifies the pattern to match only if the shape has the given rank.
  constexpr ShapePattern<ShapeType, ShapePatternRankImpl<Impl>> WithRank(
      int64 rank) const {
    return ShapePattern<ShapeType, ShapePatternRankImpl<Impl>>(
        ShapePatternRankImpl<Impl>(impl_, rank), matched_shape_);
  }

  // Modifies the pattern to match only if the shape has a layout that matches
  // the given pattern.
  template <typename LayoutType, typename LayoutImpl>
  constexpr ShapePattern<ShapeType,
                         ShapePatternLayoutImpl<Impl, LayoutType, LayoutImpl>>
  WithLayout(const LayoutPattern<LayoutType, LayoutImpl>& layout) const {
    return ShapePattern<ShapeType,
                        ShapePatternLayoutImpl<Impl, LayoutType, LayoutImpl>>(
        ShapePatternLayoutImpl<Impl, LayoutType, LayoutImpl>(impl_, layout),
        matched_shape_);
  }

  constexpr ShapePattern<
      ShapeType,
      ShapePatternLayoutImpl<Impl, const ::xla::Layout,
                             LayoutPatternEqualImpl<LayoutPatternBaseImpl>>>
  WithLayoutEqualTo(const ::xla::Layout* layout) const {
    return WithLayout(Layout().EqualTo(layout));
  }

  constexpr ShapePattern<
      ShapeType,
      ShapePatternLayoutImpl<Impl, const ::xla::Layout,
                             LayoutPatternFormatImpl<LayoutPatternBaseImpl>>>
  IsDenseArray() const {
    return WithLayout(Layout().WithDenseFormat());
  }

  constexpr ShapePattern<
      ShapeType,
      ShapePatternLayoutImpl<Impl, const ::xla::Layout,
                             LayoutPatternFormatImpl<LayoutPatternBaseImpl>>>
  IsSparseArray() const {
    return WithLayout(Layout().WithSparseFormat());
  }

  // Modifies the pattern to match only if the shape has a subshape that matches
  // the given pattern.
  template <typename SubshapeType, typename SubshapeImpl>
  ShapePattern<ShapeType,
               ShapePatternSubshapeImpl<Impl, SubshapeType, SubshapeImpl>>
  WithSubshape(ShapeIndexView index,
               const ShapePattern<SubshapeType, SubshapeImpl>& subshape) const {
    return ShapePattern<
        ShapeType, ShapePatternSubshapeImpl<Impl, SubshapeType, SubshapeImpl>>(
        ShapePatternSubshapeImpl<Impl, SubshapeType, SubshapeImpl>(impl_, index,
                                                                   subshape),
        matched_shape_);
  }

  ShapePattern<ShapeType, ShapePatternSubshapeImpl<
                              Impl, const ::xla::Shape,
                              ShapePatternEqualImpl<ShapePatternBaseImpl>>>
  WithSubshapeEqualTo(ShapeIndexView index, const ::xla::Shape* shape) const {
    return WithSubshape(index,
                        ShapePattern<const ::xla::Shape, ShapePatternBaseImpl>(
                            ShapePatternBaseImpl(), nullptr)
                            .EqualTo(shape));
  }

  ShapePattern<ShapeType, ShapePatternSubshapeImpl<
                              Impl, const ::xla::Shape,
                              ShapePatternCompatibleImpl<ShapePatternBaseImpl>>>
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
  bool Match(const ::xla::HloInstruction* inst) const {
    return inst != nullptr;
  }
};

// An HloInstructionPattern implementation that matches only if the instruction
// has a given name.
template <typename Previous>
class HloInstructionPatternNameImpl {
 public:
  explicit HloInstructionPatternNameImpl(const Previous& previous,
                                         tensorflow::StringPiece name)
      : previous_(previous), name_(name) {}

  bool Match(const ::xla::HloInstruction* inst) const {
    return previous_.Match(inst) && inst->name() == name_;
  }

 private:
  Previous previous_;
  tensorflow::StringPiece name_;
};

// An HloInstructionPattern implementation that matches only if the instruction
// has a given opcode.
template <typename Previous>
class HloInstructionPatternOpcodeImpl {
 public:
  explicit constexpr HloInstructionPatternOpcodeImpl(const Previous& previous,
                                                     HloOpcode opcode,
                                                     bool invert)
      : previous_(previous), opcode_(opcode), invert_(invert) {}

  bool Match(const ::xla::HloInstruction* inst) const {
    return previous_.Match(inst) && (invert_ ^ (inst->opcode() == opcode_));
  }

 private:
  Previous previous_;
  HloOpcode opcode_;
  bool invert_;
};

// An HloInstructionPattern implementation that matches only if the instruction
// has a shape that matches a given pattern.
template <typename Previous, typename ShapeType, typename ShapeImpl>
class HloInstructionPatternShapeImpl {
 public:
  explicit constexpr HloInstructionPatternShapeImpl(
      const Previous& previous, const ShapePattern<ShapeType, ShapeImpl>& shape)
      : previous_(previous), shape_(shape) {}

  bool Match(const ::xla::HloInstruction* inst) const {
    return previous_.Match(inst) && shape_.Match(&inst->shape());
  }

  bool Match(::xla::HloInstruction* inst) const {
    return previous_.Match(inst) && shape_.Match(inst->mutable_shape());
  }

 private:
  Previous previous_;
  ShapePattern<ShapeType, ShapeImpl> shape_;
};

// An HloInstructionPattern implementation that matches only if the instruction
// has an operand that matches a given pattern.
template <typename Previous, typename OperandType, typename OperandImpl>
class HloInstructionPatternOperandImpl {
 public:
  explicit constexpr HloInstructionPatternOperandImpl(
      const Previous& previous, int64 operand_index,
      const HloInstructionPattern<OperandType, OperandImpl>& operand)
      : previous_(previous), operand_index_(operand_index), operand_(operand) {}

  bool Match(const ::xla::HloInstruction* inst) const {
    return previous_.Match(inst) && operand_index_ < inst->operand_count() &&
           operand_.Match(inst->operand(operand_index_));
  }

  bool Match(::xla::HloInstruction* inst) const {
    return previous_.Match(inst) && operand_index_ < inst->operand_count() &&
           operand_.Match(inst->mutable_operand(operand_index_));
  }

 private:
  Previous previous_;
  int64 operand_index_;
  HloInstructionPattern<OperandType, OperandImpl> operand_;
};

// An HloInstructionPattern implementation that matches only if the instruction
// is a fusion node with a particular kind.
template <typename Previous>
class HloInstructionPatternFusionKindImpl {
 public:
  explicit constexpr HloInstructionPatternFusionKindImpl(
      const Previous& previous, ::xla::HloInstruction::FusionKind kind)
      : previous_(previous), kind_(kind) {}

  bool Match(const ::xla::HloInstruction* inst) const {
    return previous_.Match(inst) && inst->opcode() == HloOpcode::kFusion &&
           inst->fusion_kind() == kind_;
  }

  bool Match(::xla::HloInstruction* inst) const {
    return previous_.Match(inst) && inst->opcode() == HloOpcode::kFusion &&
           inst->fusion_kind() == kind_;
  }

 private:
  Previous previous_;
  ::xla::HloInstruction::FusionKind kind_;
};

// A pattern that matches HloInstructions.
template <typename HloInstructionType, typename Impl>
class HloInstructionPattern {
 public:
  explicit constexpr HloInstructionPattern(const Impl& impl,
                                           HloInstructionType** matched_inst)
      : impl_(impl), matched_inst_(matched_inst) {}

  // Returns true and captures the instruction iff it matches the pattern.
  bool Match(const ::xla::HloInstruction* inst) const {
    if (impl_.Match(inst)) {
      if (matched_inst_) {
        *matched_inst_ = inst;
      }
      return true;
    }
    return false;
  }

  // Returns true and captures the instruction iff it matches the pattern.
  bool Match(::xla::HloInstruction* inst) const {
    if (impl_.Match(inst)) {
      if (matched_inst_) {
        *matched_inst_ = inst;
      }
      return true;
    }
    return false;
  }

  // Modifies the pattern to match only if the instruction has the given name.
  HloInstructionPattern<HloInstructionType, HloInstructionPatternNameImpl<Impl>>
  WithName(tensorflow::StringPiece name) const {
    return HloInstructionPattern<HloInstructionType,
                                 HloInstructionPatternNameImpl<Impl>>(
        HloInstructionPatternNameImpl<Impl>(impl_, name), matched_inst_);
  }

  // Modifies the pattern to match only if the instruction has the given opcode.
  constexpr HloInstructionPattern<HloInstructionType,
                                  HloInstructionPatternOpcodeImpl<Impl>>
  WithOpcode(HloOpcode opcode) const {
    return HloInstructionPattern<HloInstructionType,
                                 HloInstructionPatternOpcodeImpl<Impl>>(
        HloInstructionPatternOpcodeImpl<Impl>(impl_, opcode, false),
        matched_inst_);
  }

  // Modifies the pattern to match only if the instruction does not have the
  // given opcode.
  constexpr HloInstructionPattern<HloInstructionType,
                                  HloInstructionPatternOpcodeImpl<Impl>>
  WithoutOpcode(HloOpcode opcode) const {
    return HloInstructionPattern<HloInstructionType,
                                 HloInstructionPatternOpcodeImpl<Impl>>(
        HloInstructionPatternOpcodeImpl<Impl>(impl_, opcode, true),
        matched_inst_);
  }

  // Modifies the pattern to match only if the instruction is a constant.
  constexpr HloInstructionPattern<HloInstructionType,
                                  HloInstructionPatternOpcodeImpl<Impl>>
  IsConstant() const {
    return WithOpcode(HloOpcode::kConstant);
  }

  // Modifies the pattern to match only if the instruction is not a constant.
  constexpr HloInstructionPattern<HloInstructionType,
                                  HloInstructionPatternOpcodeImpl<Impl>>
  IsNonConstant() const {
    return WithoutOpcode(HloOpcode::kConstant);
  }

  // Modifies the pattern to match only if the instruction has a shape that
  // matches the given pattern.
  template <typename ShapeType, typename ShapeImpl>
  constexpr HloInstructionPattern<
      HloInstructionType,
      HloInstructionPatternShapeImpl<Impl, ShapeType, ShapeImpl>>
  WithShape(const ShapePattern<ShapeType, ShapeImpl>& shape) const {
    return HloInstructionPattern<
        HloInstructionType,
        HloInstructionPatternShapeImpl<Impl, ShapeType, ShapeImpl>>(
        HloInstructionPatternShapeImpl<Impl, ShapeType, ShapeImpl>(impl_,
                                                                   shape),
        matched_inst_);
  }

  // Modifies the pattern to match only if the instruction has an operand that
  // matches the given pattern.
  template <typename OperandType, typename OperandImpl>
  constexpr HloInstructionPattern<
      HloInstructionType,
      HloInstructionPatternOperandImpl<Impl, OperandType, OperandImpl>>
  WithOperand(
      int64 operand_index,
      const HloInstructionPattern<OperandType, OperandImpl>& operand) const {
    return HloInstructionPattern<
        HloInstructionType,
        HloInstructionPatternOperandImpl<Impl, OperandType, OperandImpl>>(
        HloInstructionPatternOperandImpl<Impl, OperandType, OperandImpl>(
            impl_, operand_index, operand),
        matched_inst_);
  }

  // Modifies the pattern to match only if the instruction is a fusion node with
  // the given kind.
  constexpr HloInstructionPattern<HloInstructionType,
                                  HloInstructionPatternFusionKindImpl<Impl>>
  WithFusionKind(HloInstruction::FusionKind kind) const {
    return HloInstructionPattern<HloInstructionType,
                                 HloInstructionPatternFusionKindImpl<Impl>>(
        HloInstructionPatternFusionKindImpl<Impl>(impl_, kind), matched_inst_);
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
XLA_NULLOP_PATTERN(Infeed)
XLA_NULLOP_PATTERN(Parameter)
XLA_NULLOP_PATTERN(Recv)
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
XLA_UNOP_PATTERN(Imag)
XLA_UNOP_PATTERN(IsFinite)
XLA_UNOP_PATTERN(Log)
XLA_UNOP_PATTERN(Not)
XLA_UNOP_PATTERN(Negate)
XLA_UNOP_PATTERN(Outfeed)
XLA_UNOP_PATTERN(Real)
XLA_UNOP_PATTERN(Reduce)
XLA_UNOP_PATTERN(ReducePrecision)
XLA_UNOP_PATTERN(Reshape)
XLA_UNOP_PATTERN(Reverse)
XLA_UNOP_PATTERN(Send)
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
XLA_BINOP_PATTERN(Add)
XLA_BINOP_PATTERN(Atan2)
XLA_BINOP_PATTERN(Divide)
XLA_BINOP_PATTERN(Complex)
XLA_BINOP_PATTERN(Dot)
XLA_BINOP_PATTERN(Eq)
XLA_BINOP_PATTERN(Gather)
XLA_BINOP_PATTERN(Ge)
XLA_BINOP_PATTERN(Gt)
XLA_BINOP_PATTERN(Le)
XLA_BINOP_PATTERN(Lt)
XLA_BINOP_PATTERN(Maximum)
XLA_BINOP_PATTERN(Minimum)
XLA_BINOP_PATTERN(Multiply)
XLA_BINOP_PATTERN(Ne)
XLA_BINOP_PATTERN(Power)
XLA_BINOP_PATTERN(Remainder)
XLA_BINOP_PATTERN(Subtract)
XLA_BINOP_PATTERN(And)
XLA_BINOP_PATTERN(Or)
XLA_BINOP_PATTERN(ShiftLeft)
XLA_BINOP_PATTERN(ShiftRightArithmetic)
XLA_BINOP_PATTERN(ShiftRightLogical)
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

// Helpers for matching non-constant instructions.
inline auto NonConstant() -> decltype(Op().IsNonConstant()) {
  return Op().IsNonConstant();
}

template <typename HloInstructionType>
inline auto NonConstant(HloInstructionType** matched_inst)
    -> decltype(Op(matched_inst).IsNonConstant()) {
  return Op(matched_inst).IsNonConstant();
}

}  // namespace match

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_PATTERN_MATCHER_H_
