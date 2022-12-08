/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_PYTHON_LIB_CORE_CUSTOM_FLOAT_H_
#define TENSORFLOW_PYTHON_LIB_CORE_CUSTOM_FLOAT_H_

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/tsl/python/lib/core/custom_float.h"

// NOLINTBEGIN(misc-unused-using-decls)
namespace tensorflow {
namespace custom_float_internal {
using tsl::custom_float_internal::BinaryUFunc;
using tsl::custom_float_internal::BinaryUFunc2;
using tsl::custom_float_internal::ByteSwap16;
using tsl::custom_float_internal::CastToCustomFloat;
using tsl::custom_float_internal::CastToFloat;
using tsl::custom_float_internal::CustomFloatTypeDescriptor;
using tsl::custom_float_internal::HashImpl;
using tsl::custom_float_internal::make_safe;
using tsl::custom_float_internal::NPyCast;
using tsl::custom_float_internal::NPyCustomFloat_ArgMaxFunc;
using tsl::custom_float_internal::NPyCustomFloat_ArgMinFunc;
using tsl::custom_float_internal::NPyCustomFloat_Compare;
using tsl::custom_float_internal::NPyCustomFloat_CompareFunc;
using tsl::custom_float_internal::NPyCustomFloat_CopySwap;
using tsl::custom_float_internal::NPyCustomFloat_CopySwapN;
using tsl::custom_float_internal::NPyCustomFloat_DotFunc;
using tsl::custom_float_internal::NPyCustomFloat_Fill;
using tsl::custom_float_internal::NPyCustomFloat_GetItem;
using tsl::custom_float_internal::NPyCustomFloat_NonZero;
using tsl::custom_float_internal::NPyCustomFloat_SetItem;
using tsl::custom_float_internal::PyCustomFloat;
using tsl::custom_float_internal::PyCustomFloat_Add;
using tsl::custom_float_internal::PyCustomFloat_Check;
using tsl::custom_float_internal::PyCustomFloat_CustomFloat;
using tsl::custom_float_internal::PyCustomFloat_Float;
using tsl::custom_float_internal::PyCustomFloat_FromT;
using tsl::custom_float_internal::PyCustomFloat_Hash;
using tsl::custom_float_internal::PyCustomFloat_Int;
using tsl::custom_float_internal::PyCustomFloat_Multiply;
using tsl::custom_float_internal::PyCustomFloat_Negative;
using tsl::custom_float_internal::PyCustomFloat_New;
using tsl::custom_float_internal::PyCustomFloat_Repr;
using tsl::custom_float_internal::PyCustomFloat_RichCompare;
using tsl::custom_float_internal::PyCustomFloat_Str;
using tsl::custom_float_internal::PyCustomFloat_Subtract;
using tsl::custom_float_internal::PyCustomFloat_TrueDivide;
using tsl::custom_float_internal::PyDecrefDeleter;
using tsl::custom_float_internal::PyLong_CheckNoOverflow;
using tsl::custom_float_internal::RegisterCasts;
using tsl::custom_float_internal::RegisterCustomFloatCast;
using tsl::custom_float_internal::RegisterUFunc;
using tsl::custom_float_internal::Safe_PyObjectPtr;
using tsl::custom_float_internal::SafeCastToCustomFloat;
using tsl::custom_float_internal::TypeDescriptor;
using tsl::custom_float_internal::UnaryUFunc;
using tsl::custom_float_internal::UnaryUFunc2;

namespace ufuncs {
using tsl::custom_float_internal::ufuncs::Abs;
using tsl::custom_float_internal::ufuncs::Add;
using tsl::custom_float_internal::ufuncs::Arccos;
using tsl::custom_float_internal::ufuncs::Arccosh;
using tsl::custom_float_internal::ufuncs::Arcsin;
using tsl::custom_float_internal::ufuncs::Arcsinh;
using tsl::custom_float_internal::ufuncs::Arctan;
using tsl::custom_float_internal::ufuncs::Arctan2;
using tsl::custom_float_internal::ufuncs::Arctanh;
using tsl::custom_float_internal::ufuncs::Cbrt;
using tsl::custom_float_internal::ufuncs::Ceil;
using tsl::custom_float_internal::ufuncs::Conjugate;
using tsl::custom_float_internal::ufuncs::CopySign;
using tsl::custom_float_internal::ufuncs::Cos;
using tsl::custom_float_internal::ufuncs::Cosh;
using tsl::custom_float_internal::ufuncs::Deg2rad;
using tsl::custom_float_internal::ufuncs::divmod;
using tsl::custom_float_internal::ufuncs::DivmodUFunc;
using tsl::custom_float_internal::ufuncs::Eq;
using tsl::custom_float_internal::ufuncs::Exp;
using tsl::custom_float_internal::ufuncs::Exp2;
using tsl::custom_float_internal::ufuncs::Expm1;
using tsl::custom_float_internal::ufuncs::Floor;
using tsl::custom_float_internal::ufuncs::FloorDivide;
using tsl::custom_float_internal::ufuncs::Fmax;
using tsl::custom_float_internal::ufuncs::Fmin;
using tsl::custom_float_internal::ufuncs::Fmod;
using tsl::custom_float_internal::ufuncs::Frexp;
using tsl::custom_float_internal::ufuncs::Ge;
using tsl::custom_float_internal::ufuncs::Gt;
using tsl::custom_float_internal::ufuncs::Heaviside;
using tsl::custom_float_internal::ufuncs::Hypot;
using tsl::custom_float_internal::ufuncs::IsFinite;
using tsl::custom_float_internal::ufuncs::IsInf;
using tsl::custom_float_internal::ufuncs::IsNan;
using tsl::custom_float_internal::ufuncs::Ldexp;
using tsl::custom_float_internal::ufuncs::Le;
using tsl::custom_float_internal::ufuncs::Log;
using tsl::custom_float_internal::ufuncs::Log10;
using tsl::custom_float_internal::ufuncs::Log1p;
using tsl::custom_float_internal::ufuncs::Log2;
using tsl::custom_float_internal::ufuncs::LogAddExp;
using tsl::custom_float_internal::ufuncs::LogAddExp2;
using tsl::custom_float_internal::ufuncs::LogicalAnd;
using tsl::custom_float_internal::ufuncs::LogicalNot;
using tsl::custom_float_internal::ufuncs::LogicalOr;
using tsl::custom_float_internal::ufuncs::LogicalXor;
using tsl::custom_float_internal::ufuncs::Lt;
using tsl::custom_float_internal::ufuncs::Maximum;
using tsl::custom_float_internal::ufuncs::Minimum;
using tsl::custom_float_internal::ufuncs::Modf;
using tsl::custom_float_internal::ufuncs::Multiply;
using tsl::custom_float_internal::ufuncs::Ne;
using tsl::custom_float_internal::ufuncs::Negative;
using tsl::custom_float_internal::ufuncs::NextAfter;
using tsl::custom_float_internal::ufuncs::Positive;
using tsl::custom_float_internal::ufuncs::Power;
using tsl::custom_float_internal::ufuncs::Rad2deg;
using tsl::custom_float_internal::ufuncs::Reciprocal;
using tsl::custom_float_internal::ufuncs::RegisterUFuncs;
using tsl::custom_float_internal::ufuncs::Remainder;
using tsl::custom_float_internal::ufuncs::Rint;
using tsl::custom_float_internal::ufuncs::Sign;
using tsl::custom_float_internal::ufuncs::SignBit;
using tsl::custom_float_internal::ufuncs::Sin;
using tsl::custom_float_internal::ufuncs::Sinh;
using tsl::custom_float_internal::ufuncs::Spacing;
using tsl::custom_float_internal::ufuncs::Sqrt;
using tsl::custom_float_internal::ufuncs::Square;
using tsl::custom_float_internal::ufuncs::Subtract;
using tsl::custom_float_internal::ufuncs::Tan;
using tsl::custom_float_internal::ufuncs::Tanh;
using tsl::custom_float_internal::ufuncs::TrueDivide;
using tsl::custom_float_internal::ufuncs::Trunc;
}  // namespace ufuncs

using tsl::custom_float_internal::RegisterNumpyDtype;
}  // namespace custom_float_internal
}  // namespace tensorflow
// NOLINTEND(misc-unused-using-decls)

#endif  // TENSORFLOW_PYTHON_LIB_CORE_CUSTOM_FLOAT_H_
