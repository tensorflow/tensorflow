#ifndef TENSORFLOW_FRAMEWORK_REGISTER_TYPES_H_
#define TENSORFLOW_FRAMEWORK_REGISTER_TYPES_H_
// This file is used by cuda code and must remain compilable by nvcc.

#include "tensorflow/core/platform/port.h"

// Macros to apply another macro to lists of supported types.  If you change
// the lists of types, please also update the list in types.cc.
//
// See example uses of these macros in core/ops.
//
//
// Each of these TF_CALL_XXX_TYPES(m) macros invokes the macro "m" multiple
// times by passing each invocation a data type supported by TensorFlow.
//
// The different variations pass different subsets of the types.
// TF_CALL_ALL_TYPES(m) applied "m" to all types supported by TensorFlow.
// The set of types depends on the compilation platform.
//.
// This can be used to register a different template instantiation of
// an OpKernel for different signatures, e.g.:
/*
   #define REGISTER_PARTITION(type)                                  \
     REGISTER_TF_OP_KERNEL("partition", DEVICE_CPU, #type ", int32", \
                           PartitionOp<type>);
   TF_CALL_ALL_TYPES(REGISTER_PARTITION)
   #undef REGISTER_PARTITION
*/

#ifndef __ANDROID__

// Call "m" for all number types that support the comparison operations "<" and
// ">".
#define TF_CALL_REAL_NUMBER_TYPES(m) \
  m(float);                          \
  m(double);                         \
  m(int64);                          \
  m(int32);                          \
  m(uint8);                          \
  m(int16);                          \
  m(int8)

#define TF_CALL_REAL_NUMBER_TYPES_NO_INT32(m) \
  m(float);                                   \
  m(double);                                  \
  m(int64);                                   \
  m(uint8);                                   \
  m(int16);                                   \
  m(int8)

// Call "m" for all number types, including complex64.
#define TF_CALL_NUMBER_TYPES(m) \
  TF_CALL_REAL_NUMBER_TYPES(m); \
  m(complex64)

#define TF_CALL_NUMBER_TYPES_NO_INT32(m) \
  TF_CALL_REAL_NUMBER_TYPES_NO_INT32(m); \
  m(complex64)

// Call "m" on all types.
#define TF_CALL_ALL_TYPES(m) \
  TF_CALL_NUMBER_TYPES(m);   \
  m(bool);                   \
  m(string)

// Call "m" on all types supported on GPU.
#define TF_CALL_GPU_NUMBER_TYPES(m) \
  m(float);                         \
  m(double)

#else  // __ANDROID__

#define TF_CALL_REAL_NUMBER_TYPES(m) \
  m(float);                          \
  m(int32)

#define TF_CALL_NUMBER_TYPES(m) TF_CALL_REAL_NUMBER_TYPES(m)

#define TF_CALL_REAL_NUMBER_TYPES_NO_INT32(m) m(float)

#define TF_CALL_NUMBER_TYPES_NO_INT32(m) TF_CALL_REAL_NUMBER_TYPES_NO_INT32(m)

#define TF_CALL_ALL_TYPES(m) TF_CALL_REAL_NUMBER_TYPES(m)

// Maybe we could put an empty macro here for Android?
#define TF_CALL_GPU_NUMBER_TYPES(m) m(float)

#endif  // __ANDROID__

#endif  // TENSORFLOW_FRAMEWORK_REGISTER_TYPES_H_
