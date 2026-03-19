#cython: boundscheck=False
#cython: wraparound=False
#cython: infer_types=True
import numpy as np
cimport numpy as np

from tensorflow.python.util import compat

def AppendBFloat16ArrayToTensorProto(
    tensor_proto, np.ndarray[np.uint16_t, ndim=1] nparray):
  cdef long i, n
  n = nparray.size
  for i in range(n):
    tensor_proto.half_val.append(nparray[i])


def AppendFloat8ArrayToTensorProto(
    tensor_proto, np.ndarray[np.uint8_t, ndim=1] nparray):
  tensor_proto.float8_val += nparray.tobytes()


def AppendFloat16ArrayToTensorProto(
    # For numpy, npy_half is a typedef for npy_uint16,
    # see: https://github.com/numpy/numpy/blob/master/doc/source/reference/c-api.coremath.rst#half-precision-functions
    # Because np.float16_t dosen't exist in cython, we use uint16_t here.
    # TODO: Use np.float16_t when cython supports it.
    tensor_proto, np.ndarray[np.uint16_t, ndim=1] nparray):
  cdef long i, n
  n = nparray.size
  for i in range(n):
    tensor_proto.half_val.append(nparray[i])


def AppendFloat32ArrayToTensorProto(
    tensor_proto, np.ndarray[np.float32_t, ndim=1] nparray):
  cdef long i, n
  n = nparray.size
  for i in range(n):
    tensor_proto.float_val.append(nparray[i])


def AppendFloat64ArrayToTensorProto(
    tensor_proto, np.ndarray[np.float64_t, ndim=1] nparray):
  cdef long i, n
  n = nparray.size
  for i in range(n):
    tensor_proto.double_val.append(nparray[i])


def AppendInt32ArrayToTensorProto(
    tensor_proto, np.ndarray[np.int32_t, ndim=1] nparray):
  cdef long i, n
  n = nparray.size
  for i in range(n):
    tensor_proto.int_val.append(nparray[i])

def AppendUInt32ArrayToTensorProto(
    tensor_proto, np.ndarray[np.uint32_t, ndim=1] nparray):
  cdef long i, n
  n = nparray.size
  for i in range(n):
    tensor_proto.uint32_val.append(nparray[i])

def AppendInt64ArrayToTensorProto(
    tensor_proto, np.ndarray[np.int64_t, ndim=1] nparray):
  cdef long i, n
  n = nparray.size
  for i in range(n):
    tensor_proto.int64_val.append(nparray[i])

def AppendUInt64ArrayToTensorProto(
    tensor_proto, np.ndarray[np.uint64_t, ndim=1] nparray):
  cdef long i, n
  n = nparray.size
  for i in range(n):
    tensor_proto.uint64_val.append(nparray[i])

def AppendUInt8ArrayToTensorProto(
    tensor_proto, np.ndarray[np.uint8_t, ndim=1] nparray):
  cdef long i, n
  n = nparray.size
  for i in range(n):
    tensor_proto.int_val.append(nparray[i])


def AppendUInt16ArrayToTensorProto(
    tensor_proto, np.ndarray[np.uint16_t, ndim=1] nparray):
  cdef long i, n
  n = nparray.size
  for i in range(n):
    tensor_proto.int_val.append(nparray[i])


def AppendInt16ArrayToTensorProto(
    tensor_proto, np.ndarray[np.int16_t, ndim=1] nparray):
  cdef long i, n
  n = nparray.size
  for i in range(n):
    tensor_proto.int_val.append(nparray[i])


def AppendInt8ArrayToTensorProto(
    tensor_proto, np.ndarray[np.int8_t, ndim=1] nparray):
  cdef long i, n
  n = nparray.size
  for i in range(n):
    tensor_proto.int_val.append(nparray[i])


def AppendComplex64ArrayToTensorProto(
    tensor_proto, np.ndarray[np.complex64_t, ndim=1] nparray):
  cdef long i, n
  n = nparray.size
  for i in range(n):
    tensor_proto.scomplex_val.append(nparray[i].real)
    tensor_proto.scomplex_val.append(nparray[i].imag)


def AppendComplex128ArrayToTensorProto(
    tensor_proto, np.ndarray[np.complex128_t, ndim=1] nparray):
  cdef long i, n
  n = nparray.size
  for i in range(n):
    tensor_proto.dcomplex_val.append(nparray[i].real)
    tensor_proto.dcomplex_val.append(nparray[i].imag)


def AppendObjectArrayToTensorProto(tensor_proto, np.ndarray nparray):
  cdef long i, n
  n = nparray.size
  for i in range(n):
    tensor_proto.string_val.append(compat.as_bytes(nparray[i]))


def AppendBoolArrayToTensorProto(tensor_proto, nparray):
  cdef long i, n
  n = nparray.size
  for i in range(n):
    tensor_proto.bool_val.append(nparray.item(i))
