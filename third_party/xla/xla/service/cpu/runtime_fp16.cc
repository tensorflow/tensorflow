/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/cpu/runtime_fp16.h"

#include <cstring>

#include "absl/base/attributes.h"

namespace {

// Helper class that lets us access the underlying bit representation
// of a float without breaking C++ strict aliasing.
class AliasedFloatInt {
 public:
  static_assert(sizeof(float) == sizeof(uint32_t), "");

  static AliasedFloatInt FromFloat(float f) {
    AliasedFloatInt value;
    value.set_float(f);
    return value;
  }

  static AliasedFloatInt FromUInt(uint32_t u) {
    AliasedFloatInt value;
    value.set_uint(u);
    return value;
  }

  void set_float(float f) { memcpy(&value_, &f, sizeof(f)); }
  float as_float() const {
    float f;
    memcpy(&f, &value_, sizeof(f));
    return f;
  }

  void set_uint(uint32_t u) { value_ = u; }
  uint32_t as_uint() const { return value_; }

 private:
  uint32_t value_;
};
}  // namespace

// __gnu_f2h_ieee and __gnu_h2f_ieee are marked as weak symbols so if XLA is
// built with compiler-rt (that also defines these symbols) we don't get a
// duplicate definition linker error.  Making these symbols weak also ensures
// that the compiler-rt definitions "win", but that isn't essential.

// Algorithm copied from Eigen.
XlaF16ABIType ABSL_ATTRIBUTE_WEAK __gnu_f2h_ieee(float float_value) {
  AliasedFloatInt f = AliasedFloatInt::FromFloat(float_value);

  const AliasedFloatInt f32infty = AliasedFloatInt::FromUInt(255 << 23);
  const AliasedFloatInt f16max = AliasedFloatInt::FromUInt((127 + 16) << 23);
  const AliasedFloatInt denorm_magic =
      AliasedFloatInt::FromUInt(((127 - 15) + (23 - 10) + 1) << 23);
  unsigned int sign_mask = 0x80000000u;
  uint32_t o = static_cast<uint16_t>(0x0u);

  unsigned int sign = f.as_uint() & sign_mask;
  f.set_uint(f.as_uint() ^ sign);

  // NOTE all the integer compares in this function can be safely
  // compiled into signed compares since all operands are below
  // 0x80000000. Important if you want fast straight SSE2 code
  // (since there's no unsigned PCMPGTD).

  if (f.as_uint() >=
      f16max.as_uint()) {  // result is Inf or NaN (all exponent bits set)
    o = (f.as_uint() > f32infty.as_uint()) ? 0x7e00
                                           : 0x7c00;  // NaN->qNaN and Inf->Inf
  } else {                            // (De)normalized number or zero
    if (f.as_uint() < (113 << 23)) {  // resulting FP16 is subnormal or zero
      // use a magic value to align our 10 mantissa bits at the bottom of
      // the float. as long as FP addition is round-to-nearest-even this
      // just works.
      f.set_float(f.as_float() + denorm_magic.as_float());

      // and one integer subtract of the bias later, we have our final float!
      o = static_cast<uint16_t>(f.as_uint() - denorm_magic.as_uint());
    } else {
      unsigned int mant_odd =
          (f.as_uint() >> 13) & 1;  // resulting mantissa is odd

      // update exponent, rounding bias part 1
      f.set_uint(f.as_uint() + (static_cast<unsigned int>(15 - 127) << 23) +
                 0xfff);
      // rounding bias part 2
      f.set_uint(f.as_uint() + mant_odd);
      // take the bits!
      o = static_cast<uint16_t>(f.as_uint() >> 13);
    }
  }

  o |= static_cast<uint16_t>(sign >> 16);
  // The output can be a float type, bitcast it from uint16_t.
  auto ho = static_cast<uint16_t>(o);
  XlaF16ABIType ret = 0;
  std::memcpy(&ret, &ho, sizeof(ho));
  return ret;
}

// Algorithm copied from Eigen.
float ABSL_ATTRIBUTE_WEAK __gnu_h2f_ieee(XlaF16ABIType hf) {
  const AliasedFloatInt magic = AliasedFloatInt::FromUInt(113 << 23);
  const unsigned int shifted_exp = 0x7c00 << 13;  // exponent mask after shift
  AliasedFloatInt o;

  // The input can be a float type, bitcast it to uint16_t.
  uint16_t h;
  std::memcpy(&h, &hf, sizeof(h));
  o.set_uint((h & 0x7fff) << 13);                // exponent/mantissa bits
  unsigned int exp = shifted_exp & o.as_uint();  // just the exponent
  o.set_uint(o.as_uint() + ((127 - 15) << 23));  // exponent adjust

  // handle exponent special cases
  if (exp == shifted_exp) {                        // Inf/NaN?
    o.set_uint(o.as_uint() + ((128 - 16) << 23));  // extra exp adjust
  } else if (exp == 0) {                           // Zero/Denormal?
    o.set_uint(o.as_uint() + (1 << 23));           // extra exp adjust
    o.set_float(o.as_float() - magic.as_float());  // renormalize
  }

  o.set_uint(o.as_uint() | (h & 0x8000) << 16);  // sign bit
  return o.as_float();
}

XlaF16ABIType ABSL_ATTRIBUTE_WEAK __truncdfhf2(double d) {
  // This does a double rounding step, but it's precise enough for our use
  // cases.
  return __gnu_f2h_ieee(static_cast<float>(d));
}
