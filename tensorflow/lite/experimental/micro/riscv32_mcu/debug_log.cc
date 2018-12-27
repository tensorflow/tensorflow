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
// TODO(b/121324430): Add test for DebugLog fuctions
// TODO(b/121275099): Remove dependency on debug_log once the platform supports
// printf

#include <stdio.h>

namespace {

// All input buffers to the number conversion functions must be this long.
static const int kFastToBufferSize = 48;

// Reverses a zero-terminated string in-place.
char* ReverseStringInPlace(char* start, char* end) {
  char* p1 = start;
  char* p2 = end - 1;
  while (p1 < p2) {
    char tmp = *p1;
    *p1++ = *p2;
    *p2-- = tmp;
  }
  return start;
}

// Appends a string to a string, in-place. You need to pass in the maximum
// string length as the second argument.
char* StrCatStr(char* main, int main_max_length, char* to_append) {
  char* current = main;
  while (*current != 0) {
    ++current;
  }
  char* current_end = main + (main_max_length - 1);
  while ((*to_append != 0) && (current < current_end)) {
    *current = *to_append;
    ++current;
    ++to_append;
  }
  *current = 0;
  return current;
}

char* StrCpy(char* main, int main_max_length, const char* source) {
  char* current = main;
  char* current_end = main + (main_max_length - 1);
  while ((*source != 0) && (current < current_end)) {
    *current = *source;
    ++current;
    ++source;
  }
  *current = 0;
  return current;
}

// Populates the provided buffer with an ASCII representation of the number.
char* FastUInt32ToBufferLeft(uint32_t i, char* buffer, int base) {
  char* start = buffer;
  do {
    int32_t digit = i % base;
    char character;
    if (digit < 10) {
      character = '0' + digit;
    } else {
      character = 'a' + (digit - 10);
    }
    *buffer++ = character;
    i /= base;
  } while (i > 0);
  *buffer = 0;
  ReverseStringInPlace(start, buffer);
  return buffer;
}

// Populates the provided buffer with an ASCII representation of the number.
char* FastInt32ToBufferLeft(int32_t i, char* buffer) {
  uint32_t u = i;
  if (i < 0) {
    *buffer++ = '-';
    u = -u;
  }
  return FastUInt32ToBufferLeft(u, buffer, 10);
}

// Converts a number to a string and appends it to another.
char* StrCatInt32(char* main, int main_max_length, int32_t number) {
  char number_string[kFastToBufferSize];
  FastInt32ToBufferLeft(number, number_string);
  return StrCatStr(main, main_max_length, number_string);
}

// Converts a number to a string and appends it to another.
char* StrCatUInt32(char* main, int main_max_length, uint32_t number, int base) {
  char number_string[kFastToBufferSize];
  FastUInt32ToBufferLeft(number, number_string, base);
  return StrCatStr(main, main_max_length, number_string);
}

// Populates the provided buffer with ASCII representation of the float number.
// Avoids the use of any floating point instructions (since these aren't
// supported on many microcontrollers) and as a consequence prints values with
// power-of-two exponents.
char* FastFloatToBufferLeft(float i, char* buffer) {
  char* current = buffer;
  char* current_end = buffer + (kFastToBufferSize - 1);
  // Access the bit fields of the floating point value to avoid requiring any
  // float instructions. These constants are derived from IEEE 754.
  const uint32_t sign_mask = 0x80000000;
  const uint32_t exponent_mask = 0x7f800000;
  const int32_t exponent_shift = 23;
  const int32_t exponent_bias = 127;
  const uint32_t fraction_mask = 0x007fffff;
  const uint32_t u = *(uint32_t*)(&i);
  const int32_t exponent =
      ((u & exponent_mask) >> exponent_shift) - exponent_bias;
  const uint32_t fraction = (u & fraction_mask);
  // Expect ~0x2B1B9D3 for fraction.
  if (u & sign_mask) {
    *current = '-';
    current += 1;
  }
  *current = 0;
  // These are special cases for infinities and not-a-numbers.
  if (exponent == 128) {
    if (fraction == 0) {
      current = StrCatStr(current, (current_end - current), "Inf");
      return current;
    } else {
      current = StrCatStr(current, (current_end - current), "NaN");
      return current;
    }
  }
  // 0x007fffff represents 0.99... for the fraction, so to print the correct
  // decimal digits we need to scale our value before passing it to the
  // conversion function. This scale should be 10000000/8388608 = 1.1920928955.
  // We can approximate this using multipy-adds and right-shifts using the
  // values in this array.
  const int32_t scale_shifts_size = 13;
  const int8_t scale_shifts[13] = {3,  4,  8,  11, 13, 14, 17,
                                   18, 19, 20, 21, 22, 23};
  uint32_t scaled_fraction = fraction;
  for (int i = 0; i < scale_shifts_size; ++i) {
    scaled_fraction += (fraction >> scale_shifts[i]);
  }
  *current = '1';
  current += 1;
  *current = '.';
  current += 1;
  *current = 0;
  current = StrCatUInt32(current, (current_end - current), scaled_fraction, 10);
  current = StrCatStr(current, (current_end - current), "*2^");
  current = StrCatInt32(current, (current_end - current), exponent);
  return current;
}

}  // namespace

extern "C" void DebugLog(const char* s) { puts(s); }

extern "C" void DebugLogInt32(int32_t i) {
  char number_string[kFastToBufferSize];
  FastInt32ToBufferLeft(i, number_string);
  DebugLog(number_string);
}

extern "C" void DebugLogUInt32(uint32_t i) {
  char number_string[kFastToBufferSize];
  FastUInt32ToBufferLeft(i, number_string, 10);
  DebugLog(number_string);
}

extern "C" void DebugLogHex(uint32_t i) {
  char number_string[kFastToBufferSize];
  FastUInt32ToBufferLeft(i, number_string, 16);
  DebugLog(number_string);
}

extern "C" void DebugLogFloat(float i) {
  char number_string[kFastToBufferSize];
  FastFloatToBufferLeft(i, number_string);
  DebugLog(number_string);
}
