# `class tensorflow::Tensor`

Represents an n-dimensional array of values.



###Member Details

#### `tensorflow::Tensor::Tensor()` {#tensorflow_Tensor_Tensor}

Creates a 1-dimensional, 0-element float tensor.

The returned Tensor is not a scalar (shape {}), but is instead an empty one-dimensional Tensor (shape {0}, NumElements() == 0). Since it has no elements, it does not need to be assigned a value and is initialized by default ( IsInitialized() is true). If this is undesirable, consider creating a one-element scalar which does require initialization:

```c++ Tensor(DT_FLOAT, TensorShape({}))

```

#### `tensorflow::Tensor::Tensor(DataType type, const TensorShape &shape)` {#tensorflow_Tensor_Tensor}

Creates a Tensor of the given `type` and `shape`. If LogMemory::IsEnabled() the allocation is logged as coming from an unknown kernel and step. Calling the Tensor constructor directly from within an Op is deprecated: use the OpKernelConstruction/OpKernelContext allocate_* methods to allocate a new tensor, which record the kernel and step.

The underlying buffer is allocated using a ` CPUAllocator `.

#### `tensorflow::Tensor::Tensor(Allocator *a, DataType type, const TensorShape &shape)` {#tensorflow_Tensor_Tensor}

Creates a tensor with the input `type` and `shape`, using the allocator `a` to allocate the underlying buffer. If LogMemory::IsEnabled() the allocation is logged as coming from an unknown kernel and step. Calling the Tensor constructor directly from within an Op is deprecated: use the OpKernelConstruction/OpKernelContext allocate_* methods to allocate a new tensor, which record the kernel and step.

`a` must outlive the lifetime of this Tensor .

#### `tensorflow::Tensor::Tensor(Allocator *a, DataType type, const TensorShape &shape, const AllocationAttributes &allocation_attr)` {#tensorflow_Tensor_Tensor}

Creates a tensor with the input `type` and `shape`, using the allocator `a` and the specified "allocation_attr" to allocate the underlying buffer. If the kernel and step are known allocation_attr.allocation_will_be_logged should be set to true and LogMemory::RecordTensorAllocation should be called after the tensor is constructed. Calling the Tensor constructor directly from within an Op is deprecated: use the OpKernelConstruction/OpKernelContext allocate_* methods to allocate a new tensor, which record the kernel and step.

`a` must outlive the lifetime of this Tensor .

#### `tensorflow::Tensor::Tensor(DataType type)` {#tensorflow_Tensor_Tensor}

Creates an empty Tensor of the given data type.

Like Tensor() , returns a 1-dimensional, 0-element Tensor with IsInitialized() returning True. See the Tensor() documentation for details.

#### `tensorflow::Tensor::Tensor(const Tensor &other)` {#tensorflow_Tensor_Tensor}





#### `tensorflow::Tensor::Tensor(Tensor &&other)` {#tensorflow_Tensor_Tensor}

Copy constructor.



#### `tensorflow::Tensor::~Tensor()` {#tensorflow_Tensor_Tensor}





#### `DataType tensorflow::Tensor::dtype() const` {#DataType_tensorflow_Tensor_dtype}

Returns the data type.



#### `const TensorShape& tensorflow::Tensor::shape() const` {#const_TensorShape_tensorflow_Tensor_shape}

Returns the shape of the tensor.



#### `int tensorflow::Tensor::dims() const` {#int_tensorflow_Tensor_dims}

Convenience accessor for the tensor shape.

For all shape accessors, see comments for relevant methods of ` TensorShape ` in ` tensor_shape.h `.

#### `int64 tensorflow::Tensor::dim_size(int d) const` {#int64_tensorflow_Tensor_dim_size}

Convenience accessor for the tensor shape.



#### `int64 tensorflow::Tensor::NumElements() const` {#int64_tensorflow_Tensor_NumElements}

Convenience accessor for the tensor shape.



#### `bool tensorflow::Tensor::IsSameSize(const Tensor &b) const` {#bool_tensorflow_Tensor_IsSameSize}





#### `bool tensorflow::Tensor::SharesBufferWith(const Tensor &b) const` {#bool_tensorflow_Tensor_SharesBufferWith}





#### `bool tensorflow::Tensor::IsInitialized() const` {#bool_tensorflow_Tensor_IsInitialized}

If necessary, has this Tensor been initialized?

Zero-element Tensors are always considered initialized, even if they have never been assigned to and do not have any memory allocated.

#### `size_t tensorflow::Tensor::TotalBytes() const` {#size_t_tensorflow_Tensor_TotalBytes}

Returns the estimated memory usage of this tensor.



#### `bool tensorflow::Tensor::IsAligned() const` {#bool_tensorflow_Tensor_IsAligned}

Returns true iff this tensor is aligned.



#### `Tensor& tensorflow::Tensor::operator=(const Tensor &other)` {#Tensor_tensorflow_Tensor_operator_}

Assign operator. This tensor shares other&apos;s underlying storage.



#### `Tensor & tensorflow::Tensor::operator=(Tensor &&other)` {#Tensor_tensorflow_Tensor_operator_}

Move operator. See move constructor for details.



#### `bool tensorflow::Tensor::CopyFrom(const Tensor &other, const TensorShape &shape) TF_MUST_USE_RESULT` {#bool_tensorflow_Tensor_CopyFrom}

Copy the other tensor into this tensor and reshape it.

This tensor shares other&apos;s underlying storage. Returns `true` iff `other.shape()` has the same number of elements of the given `shape`.

#### `Tensor tensorflow::Tensor::Slice(int64 dim0_start, int64 dim0_limit) const` {#Tensor_tensorflow_Tensor_Slice}

Slice this tensor along the 1st dimension.

I.e., the returned tensor satisfies returned[i, ...] == this[dim0_start + i, ...]. The returned tensor shares the underlying tensor buffer with this tensor.

NOTE: The returned tensor may not satisfies the same alignment requirement as this tensor depending on the shape. The caller must check the returned tensor&apos;s alignment before calling certain methods that have alignment requirement (e.g., ` flat() `, `tensor()`).

REQUIRES: ` dims() ` >= 1 REQUIRES: `0 <= dim0_start <= dim0_limit <= dim_size(0)`

#### `bool tensorflow::Tensor::FromProto(const TensorProto &other) TF_MUST_USE_RESULT` {#bool_tensorflow_Tensor_FromProto}

Parse `other` and construct the tensor.

Returns `true` iff the parsing succeeds. If the parsing fails, the state of `*this` is unchanged.

#### `bool tensorflow::Tensor::FromProto(Allocator *a, const TensorProto &other) TF_MUST_USE_RESULT` {#bool_tensorflow_Tensor_FromProto}





#### `void tensorflow::Tensor::AsProtoField(TensorProto *proto) const` {#void_tensorflow_Tensor_AsProtoField}

Fills in `proto` with `*this` tensor&apos;s content.

` AsProtoField() ` fills in the repeated field for `proto.dtype()`, while `AsProtoTensorContent()` encodes the content in `proto.tensor_content()` in a compact form.

#### `void tensorflow::Tensor::AsProtoTensorContent(TensorProto *proto) const` {#void_tensorflow_Tensor_AsProtoTensorContent}





#### `TTypes<T>::Vec tensorflow::Tensor::vec()` {#TTypes_T_Vec_tensorflow_Tensor_vec}

Return the tensor data as an `Eigen::Tensor` with the type and sizes of this ` Tensor `.

Use these methods when you know the data type and the number of dimensions of the Tensor and you want an `Eigen::Tensor` automatically sized to the ` Tensor ` sizes. The implementation check fails if either type or sizes mismatch.

Example:

```c++ typedef float T;
Tensor my_mat(...built with Shape{rows: 3, cols: 5}...);
auto mat = my_mat.matrix<T>();    // 2D Eigen::Tensor, 3 x 5.
auto mat = my_mat.tensor<T, 2>(); // 2D Eigen::Tensor, 3 x 5.
auto vec = my_mat.vec<T>();       // CHECK fails as my_mat is 2D.
auto vec = my_mat.tensor<T, 3>(); // CHECK fails as my_mat is 2D.
auto mat = my_mat.matrix<int32>();// CHECK fails as type mismatch.

```

#### `TTypes<T>::Matrix tensorflow::Tensor::matrix()` {#TTypes_T_Matrix_tensorflow_Tensor_matrix}





#### `TTypes< T, NDIMS >::Tensor tensorflow::Tensor::tensor()` {#TTypes_T_NDIMS_Tensor_tensorflow_Tensor_tensor}





#### `TTypes< T, NDIMS >::Tensor tensorflow::Tensor::bit_casted_tensor()` {#TTypes_T_NDIMS_Tensor_tensorflow_Tensor_bit_casted_tensor}

Return the tensor data to an `Eigen::Tensor` with the same size but a bitwise cast to the specified dtype `T`.

Using a bitcast is useful for move and copy operations. NOTE: this is the same as `tensor()` except a bitcast is allowed.

#### `TTypes<T>::Flat tensorflow::Tensor::flat()` {#TTypes_T_Flat_tensorflow_Tensor_flat}

Return the tensor data as an `Eigen::Tensor` of the data type and a specified shape.

These methods allow you to access the data with the dimensions and sizes of your choice. You do not need to know the number of dimensions of the Tensor to call them. However, they `CHECK` that the type matches and the dimensions requested creates an `Eigen::Tensor` with the same number of elements as the tensor.

Example:

```c++ typedef float T;
Tensor my_ten(...built with Shape{planes: 4, rows: 3, cols: 5}...);
// 1D Eigen::Tensor, size 60:
auto flat = my_ten.flat<T>();
// 2D Eigen::Tensor 12 x 5:
auto inner = my_ten.flat_inner_dims<T>();
// 2D Eigen::Tensor 4 x 15:
auto outer = my_ten.shaped<T, 2>({4, 15});
// CHECK fails, bad num elements:
auto outer = my_ten.shaped<T, 2>({4, 8});
// 3D Eigen::Tensor 6 x 5 x 2:
auto weird = my_ten.shaped<T, 3>({6, 5, 2});
// CHECK fails, type mismatch:
auto bad   = my_ten.flat<int32>();

```

#### `TTypes<T>::UnalignedFlat tensorflow::Tensor::unaligned_flat()` {#TTypes_T_UnalignedFlat_tensorflow_Tensor_unaligned_flat}





#### `TTypes< T, NDIMS >::Tensor tensorflow::Tensor::flat_inner_dims()` {#TTypes_T_NDIMS_Tensor_tensorflow_Tensor_flat_inner_dims}



Returns the data as an Eigen::Tensor with NDIMS dimensions, collapsing all Tensor dimensions but the last NDIMS-1 into the first dimension of the result. If NDIMS > dims() then leading dimensions of size 1 will be added to make the output rank NDIMS.

#### `TTypes< T, NDIMS >::Tensor tensorflow::Tensor::flat_outer_dims()` {#TTypes_T_NDIMS_Tensor_tensorflow_Tensor_flat_outer_dims}



Returns the data as an Eigen::Tensor with NDIMS dimensions, collapsing all Tensor dimensions but the first NDIMS-1 into the last dimension of the result. If NDIMS > dims() then trailing dimensions of size 1 will be added to make the output rank NDIMS.

#### `TTypes< T, NDIMS >::Tensor tensorflow::Tensor::shaped(gtl::ArraySlice< int64 > new_sizes)` {#TTypes_T_NDIMS_Tensor_tensorflow_Tensor_shaped}





#### `TTypes< T, NDIMS >::Tensor tensorflow::Tensor::bit_casted_shaped(gtl::ArraySlice< int64 > new_sizes)` {#TTypes_T_NDIMS_Tensor_tensorflow_Tensor_bit_casted_shaped}

Return the tensor data to an `Eigen::Tensor` with the new shape specified in `new_sizes` and cast to a new dtype `T`.

Using a bitcast is useful for move and copy operations. The allowed bitcast is the only difference from `shaped()`.

#### `TTypes< T, NDIMS >::UnalignedTensor tensorflow::Tensor::unaligned_shaped(gtl::ArraySlice< int64 > new_sizes)` {#TTypes_T_NDIMS_UnalignedTensor_tensorflow_Tensor_unaligned_shaped}





#### `TTypes< T >::Scalar tensorflow::Tensor::scalar()` {#TTypes_T_Scalar_tensorflow_Tensor_scalar}

Return the Tensor data as a `TensorMap` of fixed size 1: `TensorMap<TensorFixedSize<T, 1>>`.

Using ` scalar() ` allows the compiler to perform optimizations as the size of the tensor is known at compile time.

#### `TTypes<T>::ConstVec tensorflow::Tensor::vec() const` {#TTypes_T_ConstVec_tensorflow_Tensor_vec}

Const versions of all the methods above.



#### `TTypes<T>::ConstMatrix tensorflow::Tensor::matrix() const` {#TTypes_T_ConstMatrix_tensorflow_Tensor_matrix}





#### `TTypes< T, NDIMS >::ConstTensor tensorflow::Tensor::tensor() const` {#TTypes_T_NDIMS_ConstTensor_tensorflow_Tensor_tensor}





#### `TTypes< T, NDIMS >::ConstTensor tensorflow::Tensor::bit_casted_tensor() const` {#TTypes_T_NDIMS_ConstTensor_tensorflow_Tensor_bit_casted_tensor}

Return the tensor data to an `Eigen::Tensor` with the same size but a bitwise cast to the specified dtype `T`.

Using a bitcast is useful for move and copy operations. NOTE: this is the same as `tensor()` except a bitcast is allowed.

#### `TTypes<T>::ConstFlat tensorflow::Tensor::flat() const` {#TTypes_T_ConstFlat_tensorflow_Tensor_flat}





#### `TTypes<T>::UnalignedConstFlat tensorflow::Tensor::unaligned_flat() const` {#TTypes_T_UnalignedConstFlat_tensorflow_Tensor_unaligned_flat}





#### `TTypes< T, NDIMS >::ConstTensor tensorflow::Tensor::shaped(gtl::ArraySlice< int64 > new_sizes) const` {#TTypes_T_NDIMS_ConstTensor_tensorflow_Tensor_shaped}





#### `TTypes< T, NDIMS >::ConstTensor tensorflow::Tensor::bit_casted_shaped(gtl::ArraySlice< int64 > new_sizes) const` {#TTypes_T_NDIMS_ConstTensor_tensorflow_Tensor_bit_casted_shaped}

Return the tensor data to an `Eigen::Tensor` with the new shape specified in `new_sizes` and cast to a new dtype `T`.

Using a bitcast is useful for move and copy operations. The allowed bitcast is the only difference from `shaped()`.

#### `TTypes< T, NDIMS >::UnalignedConstTensor tensorflow::Tensor::unaligned_shaped(gtl::ArraySlice< int64 > new_sizes) const` {#TTypes_T_NDIMS_UnalignedConstTensor_tensorflow_Tensor_unaligned_shaped}





#### `TTypes< T >::ConstScalar tensorflow::Tensor::scalar() const` {#TTypes_T_ConstScalar_tensorflow_Tensor_scalar}





#### `TTypes< T, NDIMS >::ConstTensor tensorflow::Tensor::flat_inner_dims() const` {#TTypes_T_NDIMS_ConstTensor_tensorflow_Tensor_flat_inner_dims}





#### `TTypes< T, NDIMS >::ConstTensor tensorflow::Tensor::flat_outer_dims() const` {#TTypes_T_NDIMS_ConstTensor_tensorflow_Tensor_flat_outer_dims}





#### `string tensorflow::Tensor::SummarizeValue(int64 max_entries) const` {#string_tensorflow_Tensor_SummarizeValue}

Render the first `max_entries` values in `*this` into a string.



#### `string tensorflow::Tensor::DebugString() const` {#string_tensorflow_Tensor_DebugString}

A human-readable summary of the tensor suitable for debugging.



#### `void tensorflow::Tensor::FillDescription(TensorDescription *description) const` {#void_tensorflow_Tensor_FillDescription}



Fill in the `TensorDescription` proto with metadata about the tensor that is useful for monitoring and debugging.

#### `StringPiece tensorflow::Tensor::tensor_data() const` {#StringPiece_tensorflow_Tensor_tensor_data}

Returns a ` StringPiece ` mapping the current tensor&apos;s buffer.

The returned ` StringPiece ` may point to memory location on devices that the CPU cannot address directly.

NOTE: The underlying tensor buffer is refcounted, so the lifetime of the contents mapped by the ` StringPiece ` matches the lifetime of the buffer; callers should arrange to make sure the buffer does not get destroyed while the ` StringPiece ` is still used.

REQUIRES: `DataTypeCanUseMemcpy(dtype())`.

#### `void tensorflow::Tensor::UnsafeCopyFromInternal(const Tensor &, DataType dtype, const TensorShape &)` {#void_tensorflow_Tensor_UnsafeCopyFromInternal}



Copy the other tensor into this tensor and reshape it and reinterpret the buffer&apos;s datatype.

This tensor shares other&apos;s underlying storage.
