Represents the type of the elements in a `Tensor`.

The following `DType` objects are defined:

* `tf.float16`: 16-bit half-precision floating-point.
* `tf.float32`: 32-bit single-precision floating-point.
* `tf.float64`: 64-bit double-precision floating-point.
* `tf.bfloat16`: 16-bit truncated floating-point.
* `tf.complex64`: 64-bit single-precision complex.
* `tf.complex128`: 128-bit double-precision complex.

* `tf.int8`: 8-bit signed integer.
* `tf.uint8`: 8-bit unsigned integer.
* `tf.uint16`: 16-bit unsigned integer.
* `tf.int16`: 16-bit signed integer.
* `tf.int32`: 32-bit signed integer.
* `tf.int64`: 64-bit signed integer.

* `tf.bool`: Boolean.

* `tf.string`: String.

* `tf.qint8`: Quantized 8-bit signed integer.
* `tf.quint8`: Quantized 8-bit unsigned integer.
* `tf.qint16`: Quantized 16-bit signed integer.
* `tf.quint16`: Quantized 16-bit unsigned integer.
* `tf.qint32`: Quantized 32-bit signed integer.

In addition, variants of these types with the `_ref` suffix are
defined for reference-typed tensors.

The `tf.as_dtype()` function converts numpy types and string type
names to a `DType` object.

- - -

#### `tf.DType.is_compatible_with(other)` {#DType.is_compatible_with}

Returns True if the `other` DType will be converted to this DType.

The conversion rules are as follows:

```
DType(T)       .is_compatible_with(DType(T))        == True
DType(T)       .is_compatible_with(DType(T).as_ref) == True
DType(T).as_ref.is_compatible_with(DType(T))        == False
DType(T).as_ref.is_compatible_with(DType(T).as_ref) == True
```

##### Args:


*  <b>`other`</b>: A `DType` (or object that may be converted to a `DType`).

##### Returns:

  True if a Tensor of the `other` `DType` will be implicitly converted to
  this `DType`.


- - -

#### `tf.DType.name` {#DType.name}

Returns the string name for this `DType`.


- - -

#### `tf.DType.base_dtype` {#DType.base_dtype}

Returns a non-reference `DType` based on this `DType`.


- - -

#### `tf.DType.real_dtype` {#DType.real_dtype}

Returns the dtype correspond to this dtype's real part.


- - -

#### `tf.DType.is_ref_dtype` {#DType.is_ref_dtype}

Returns `True` if this `DType` represents a reference type.


- - -

#### `tf.DType.as_ref` {#DType.as_ref}

Returns a reference `DType` based on this `DType`.


- - -

#### `tf.DType.is_floating` {#DType.is_floating}

Returns whether this is a (real) floating point type.


- - -

#### `tf.DType.is_complex` {#DType.is_complex}

Returns whether this is a complex floating point type.


- - -

#### `tf.DType.is_integer` {#DType.is_integer}

Returns whether this is a (non-quantized) integer type.


- - -

#### `tf.DType.is_quantized` {#DType.is_quantized}

Returns whether this is a quantized data type.


- - -

#### `tf.DType.is_unsigned` {#DType.is_unsigned}

Returns whether this type is unsigned.

Non-numeric, unordered, and quantized types are not considered unsigned, and
this function returns `False`.

##### Returns:

  Whether a `DType` is unsigned.



- - -

#### `tf.DType.as_numpy_dtype` {#DType.as_numpy_dtype}

Returns a `numpy.dtype` based on this `DType`.


- - -

#### `tf.DType.as_datatype_enum` {#DType.as_datatype_enum}

Returns a `types_pb2.DataType` enum value based on this `DType`.



#### Other Methods
- - -

#### `tf.DType.__init__(type_enum)` {#DType.__init__}

Creates a new `DataType`.

NOTE(mrry): In normal circumstances, you should not need to
construct a `DataType` object directly. Instead, use the
`tf.as_dtype()` function.

##### Args:


*  <b>`type_enum`</b>: A `types_pb2.DataType` enum value.

##### Raises:


*  <b>`TypeError`</b>: If `type_enum` is not a value `types_pb2.DataType`.


- - -

#### `tf.DType.max` {#DType.max}

Returns the maximum representable value in this data type.

##### Raises:


*  <b>`TypeError`</b>: if this is a non-numeric, unordered, or quantized type.


- - -

#### `tf.DType.min` {#DType.min}

Returns the minimum representable value in this data type.

##### Raises:


*  <b>`TypeError`</b>: if this is a non-numeric, unordered, or quantized type.


- - -

#### `tf.DType.size` {#DType.size}




