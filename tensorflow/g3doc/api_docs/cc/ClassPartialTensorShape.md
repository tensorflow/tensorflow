# `class tensorflow::PartialTensorShape`

Manages the partially known dimensions of a Tensor and their sizes.



###Member Details

#### `tensorflow::PartialTensorShape::PartialTensorShape()` {#tensorflow_PartialTensorShape_PartialTensorShape}

Construct an unknown ` PartialTensorShape `.



#### `tensorflow::PartialTensorShape::PartialTensorShape(gtl::ArraySlice< int64 > dim_sizes)` {#tensorflow_PartialTensorShape_PartialTensorShape}

Construct a ` PartialTensorShape ` from the provided sizes. REQUIRES: `dim_sizes[i] >= 0`



#### `tensorflow::PartialTensorShape::PartialTensorShape(std::initializer_list< int64 > dim_sizes)` {#tensorflow_PartialTensorShape_PartialTensorShape}





#### `tensorflow::PartialTensorShape::PartialTensorShape(const TensorShapeProto &proto)` {#tensorflow_PartialTensorShape_PartialTensorShape}

REQUIRES: `IsValid(proto)`



#### `PartialTensorShape tensorflow::PartialTensorShape::Concatenate(int64 size) const` {#PartialTensorShape_tensorflow_PartialTensorShape_Concatenate}



Add a dimension to the end ("inner-most"), returns a new PartialTensorShape . REQUIRES: `size >= -1`, where -1 means unknown

#### `PartialTensorShape tensorflow::PartialTensorShape::Concatenate(const PartialTensorShape &shape) const` {#PartialTensorShape_tensorflow_PartialTensorShape_Concatenate}



Appends all the dimensions from `shape`. Returns a new PartialTensorShape .

#### `Status tensorflow::PartialTensorShape::MergeWith(const PartialTensorShape &shape, PartialTensorShape *result) const` {#Status_tensorflow_PartialTensorShape_MergeWith}



Merges all the dimensions from `shape`. Returns `InvalidArgument` error if either `shape` has a different rank or if any of the dimensions are incompatible.

#### `int tensorflow::PartialTensorShape::dims() const` {#int_tensorflow_PartialTensorShape_dims}



Return the number of dimensions in the tensor. If the number of dimensions is unknown, return -1.

#### `bool tensorflow::PartialTensorShape::IsFullyDefined() const` {#bool_tensorflow_PartialTensorShape_IsFullyDefined}

Return true iff the rank and all of the dimensions are well defined.



#### `bool tensorflow::PartialTensorShape::IsCompatibleWith(const PartialTensorShape &shape) const` {#bool_tensorflow_PartialTensorShape_IsCompatibleWith}



Return true iff the ranks match, and if the dimensions all either match or one is unknown.

#### `bool tensorflow::PartialTensorShape::IsCompatibleWith(const TensorShape &shape) const` {#bool_tensorflow_PartialTensorShape_IsCompatibleWith}



Return true iff the dimensions of `shape` are compatible with `*this`.

#### `int64 tensorflow::PartialTensorShape::dim_size(int d) const` {#int64_tensorflow_PartialTensorShape_dim_size}

Returns the number of elements in dimension `d`. REQUIRES: `0 <= d < dims() `



#### `gtl::ArraySlice<int64> tensorflow::PartialTensorShape::dim_sizes() const` {#gtl_ArraySlice_int64_tensorflow_PartialTensorShape_dim_sizes}

Returns sizes of all dimensions.



#### `void tensorflow::PartialTensorShape::AsProto(TensorShapeProto *proto) const` {#void_tensorflow_PartialTensorShape_AsProto}

Fill `*proto` from `*this`.



#### `bool tensorflow::PartialTensorShape::AsTensorShape(TensorShape *tensor_shape) const` {#bool_tensorflow_PartialTensorShape_AsTensorShape}





#### `string tensorflow::PartialTensorShape::DebugString() const` {#string_tensorflow_PartialTensorShape_DebugString}

For error messages.



#### `bool tensorflow::PartialTensorShape::IsValid(const TensorShapeProto &proto)` {#bool_tensorflow_PartialTensorShape_IsValid}

Returns `true` iff `proto` is a valid partial tensor shape.



#### `Status tensorflow::PartialTensorShape::IsValidShape(const TensorShapeProto &proto)` {#Status_tensorflow_PartialTensorShape_IsValidShape}



Returns `OK` iff `proto` is a valid tensor shape, and a descriptive error status otherwise.

#### `string tensorflow::PartialTensorShape::DebugString(const TensorShapeProto &proto)` {#string_tensorflow_PartialTensorShape_DebugString}





#### `static Status tensorflow::PartialTensorShape::MakePartialShape(const int32 *dims, int n, PartialTensorShape *out)` {#static_Status_tensorflow_PartialTensorShape_MakePartialShape}

Returns a ` PartialTensorShape ` whose dimensions are `dims[0]`, `dims[1]`, ..., `dims[n-1]`. Values of -1 are considered "unknown".



#### `static Status tensorflow::PartialTensorShape::MakePartialShape(const int64 *dims, int n, PartialTensorShape *out)` {#static_Status_tensorflow_PartialTensorShape_MakePartialShape}




