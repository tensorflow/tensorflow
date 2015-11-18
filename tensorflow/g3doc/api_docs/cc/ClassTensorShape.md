# Class `tensorflow::TensorShape`

Manages the dimensions of a Tensor and their sizes.



##Member Summary

* [`tensorflow::TensorShape::TensorShape(gtl::ArraySlice< int64 > dim_sizes)`](#tensorflow_TensorShape_TensorShape)
  * Construct a ` TensorShape ` from the provided sizes. REQUIRES: `dim_sizes[i] >= 0`
* [`tensorflow::TensorShape::TensorShape(std::initializer_list< int64 > dim_sizes)`](#tensorflow_TensorShape_TensorShape)
* [`tensorflow::TensorShape::TensorShape(const TensorShapeProto &proto)`](#tensorflow_TensorShape_TensorShape)
  * REQUIRES: `IsValid(proto)`
* [`tensorflow::TensorShape::TensorShape()`](#tensorflow_TensorShape_TensorShape)
* [`void tensorflow::TensorShape::Clear()`](#void_tensorflow_TensorShape_Clear)
  * Clear a tensor shape.
* [`void tensorflow::TensorShape::AddDim(int64 size)`](#void_tensorflow_TensorShape_AddDim)
  * Add a dimension to the end ("inner-most"). REQUIRES: `size >= 0`
* [`void tensorflow::TensorShape::AppendShape(const TensorShape &shape)`](#void_tensorflow_TensorShape_AppendShape)
  * Appends all the dimensions from `shape`.
* [`void tensorflow::TensorShape::InsertDim(int d, int64 size)`](#void_tensorflow_TensorShape_InsertDim)
  * Insert a dimension somewhere in the ` TensorShape `. REQUIRES: `0 <= d <= dims() ` REQUIRES: `size >= 0`
* [`void tensorflow::TensorShape::set_dim(int d, int64 size)`](#void_tensorflow_TensorShape_set_dim)
  * Modifies the size of the dimension `d` to be `size` REQUIRES: `0 <= d < dims() ` REQUIRES: `size >= 0`
* [`void tensorflow::TensorShape::RemoveDim(int d)`](#void_tensorflow_TensorShape_RemoveDim)
  * Removes dimension `d` from the ` TensorShape `. REQUIRES: `0 <= d < dims() `
* [`int tensorflow::TensorShape::dims() const`](#int_tensorflow_TensorShape_dims)
  * Return the number of dimensions in the tensor.
* [`int64 tensorflow::TensorShape::dim_size(int d) const`](#int64_tensorflow_TensorShape_dim_size)
  * Returns the number of elements in dimension `d`. REQUIRES: `0 <= d < dims() `
* [`gtl::ArraySlice<int64> tensorflow::TensorShape::dim_sizes() const`](#gtl_ArraySlice_int64_tensorflow_TensorShape_dim_sizes)
  * Returns sizes of all dimensions.
* [`int64 tensorflow::TensorShape::num_elements() const`](#int64_tensorflow_TensorShape_num_elements)
  * Returns the number of elements in the tensor.
* [`bool tensorflow::TensorShape::IsSameSize(const TensorShape &b) const`](#bool_tensorflow_TensorShape_IsSameSize)
* [`bool tensorflow::TensorShape::operator==(const TensorShape &b) const`](#bool_tensorflow_TensorShape_operator_)
* [`void tensorflow::TensorShape::AsProto(TensorShapeProto *proto) const`](#void_tensorflow_TensorShape_AsProto)
  * Fill `*proto` from `*this`.
* [`Eigen::DSizes< Eigen::DenseIndex, NDIMS > tensorflow::TensorShape::AsEigenDSizes() const`](#Eigen_DSizes_Eigen_DenseIndex_NDIMS_tensorflow_TensorShape_AsEigenDSizes)
  * Fill `*dsizes` from `*this`.
* [`Eigen::DSizes< Eigen::DenseIndex, NDIMS > tensorflow::TensorShape::AsEigenDSizesWithPadding() const`](#Eigen_DSizes_Eigen_DenseIndex_NDIMS_tensorflow_TensorShape_AsEigenDSizesWithPadding)
* [`TensorShapeIter tensorflow::TensorShape::begin() const`](#TensorShapeIter_tensorflow_TensorShape_begin)
  * For iterating through the dimensions.
* [`TensorShapeIter tensorflow::TensorShape::end() const`](#TensorShapeIter_tensorflow_TensorShape_end)
* [`string tensorflow::TensorShape::DebugString() const`](#string_tensorflow_TensorShape_DebugString)
  * For error messages.
* [`string tensorflow::TensorShape::ShortDebugString() const`](#string_tensorflow_TensorShape_ShortDebugString)
* [`static bool tensorflow::TensorShape::IsValid(const TensorShapeProto &proto)`](#static_bool_tensorflow_TensorShape_IsValid)
  * Returns `true` iff `proto` is a valid tensor shape.

##Member Details

#### `tensorflow::TensorShape::TensorShape(gtl::ArraySlice< int64 > dim_sizes)` {#tensorflow_TensorShape_TensorShape}

Construct a ` TensorShape ` from the provided sizes. REQUIRES: `dim_sizes[i] >= 0`



#### `tensorflow::TensorShape::TensorShape(std::initializer_list< int64 > dim_sizes)` {#tensorflow_TensorShape_TensorShape}





#### `tensorflow::TensorShape::TensorShape(const TensorShapeProto &proto)` {#tensorflow_TensorShape_TensorShape}

REQUIRES: `IsValid(proto)`



#### `tensorflow::TensorShape::TensorShape()` {#tensorflow_TensorShape_TensorShape}



Create a tensor shape with no dimensions and one element, which you can then call ` AddDim() ` on.

#### `void tensorflow::TensorShape::Clear()` {#void_tensorflow_TensorShape_Clear}

Clear a tensor shape.



#### `void tensorflow::TensorShape::AddDim(int64 size)` {#void_tensorflow_TensorShape_AddDim}

Add a dimension to the end ("inner-most"). REQUIRES: `size >= 0`



#### `void tensorflow::TensorShape::AppendShape(const TensorShape &shape)` {#void_tensorflow_TensorShape_AppendShape}

Appends all the dimensions from `shape`.



#### `void tensorflow::TensorShape::InsertDim(int d, int64 size)` {#void_tensorflow_TensorShape_InsertDim}

Insert a dimension somewhere in the ` TensorShape `. REQUIRES: `0 <= d <= dims() ` REQUIRES: `size >= 0`



#### `void tensorflow::TensorShape::set_dim(int d, int64 size)` {#void_tensorflow_TensorShape_set_dim}

Modifies the size of the dimension `d` to be `size` REQUIRES: `0 <= d < dims() ` REQUIRES: `size >= 0`



#### `void tensorflow::TensorShape::RemoveDim(int d)` {#void_tensorflow_TensorShape_RemoveDim}

Removes dimension `d` from the ` TensorShape `. REQUIRES: `0 <= d < dims() `



#### `int tensorflow::TensorShape::dims() const` {#int_tensorflow_TensorShape_dims}

Return the number of dimensions in the tensor.



#### `int64 tensorflow::TensorShape::dim_size(int d) const` {#int64_tensorflow_TensorShape_dim_size}

Returns the number of elements in dimension `d`. REQUIRES: `0 <= d < dims() `



#### `gtl::ArraySlice<int64> tensorflow::TensorShape::dim_sizes() const` {#gtl_ArraySlice_int64_tensorflow_TensorShape_dim_sizes}

Returns sizes of all dimensions.



#### `int64 tensorflow::TensorShape::num_elements() const` {#int64_tensorflow_TensorShape_num_elements}

Returns the number of elements in the tensor.

We use `int64` and not `size_t` to be compatible with `Eigen::Tensor` which uses `ptrdiff_t`.

#### `bool tensorflow::TensorShape::IsSameSize(const TensorShape &b) const` {#bool_tensorflow_TensorShape_IsSameSize}



Returns true if `*this` and `b` have the same sizes. Ignores dimension names.

#### `bool tensorflow::TensorShape::operator==(const TensorShape &b) const` {#bool_tensorflow_TensorShape_operator_}





#### `void tensorflow::TensorShape::AsProto(TensorShapeProto *proto) const` {#void_tensorflow_TensorShape_AsProto}

Fill `*proto` from `*this`.



#### `Eigen::DSizes< Eigen::DenseIndex, NDIMS > tensorflow::TensorShape::AsEigenDSizes() const` {#Eigen_DSizes_Eigen_DenseIndex_NDIMS_tensorflow_TensorShape_AsEigenDSizes}

Fill `*dsizes` from `*this`.



#### `Eigen::DSizes< Eigen::DenseIndex, NDIMS > tensorflow::TensorShape::AsEigenDSizesWithPadding() const` {#Eigen_DSizes_Eigen_DenseIndex_NDIMS_tensorflow_TensorShape_AsEigenDSizesWithPadding}



Same as ` AsEigenDSizes() ` but allows for `NDIMS > dims() ` in which case we pad the rest of the sizes with 1.

#### `TensorShapeIter tensorflow::TensorShape::begin() const` {#TensorShapeIter_tensorflow_TensorShape_begin}

For iterating through the dimensions.



#### `TensorShapeIter tensorflow::TensorShape::end() const` {#TensorShapeIter_tensorflow_TensorShape_end}





#### `string tensorflow::TensorShape::DebugString() const` {#string_tensorflow_TensorShape_DebugString}

For error messages.



#### `string tensorflow::TensorShape::ShortDebugString() const` {#string_tensorflow_TensorShape_ShortDebugString}





#### `static bool tensorflow::TensorShape::IsValid(const TensorShapeProto &proto)` {#static_bool_tensorflow_TensorShape_IsValid}

Returns `true` iff `proto` is a valid tensor shape.


