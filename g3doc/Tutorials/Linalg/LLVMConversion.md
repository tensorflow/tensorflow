# Conversion to the LLVM IR Dialect

This chapter in the tutorial uses the DialectConversion framework to implement
the conversion from the Linalg dialect to the [LLVM IR](../../Dialects/LLVM.md)
dialect. This framework is a part of a more general pattern rewriting
infrastructure available in MLIR. Its key feature is the ability to update
function signatures and function/block argument types, along with the
pattern-based operation rewriting patterns.

## Structure of a Dialect Conversion

The Dialect Conversion framework comprises three components:

1.  Type conversion function.
2.  (Optional) function signature conversion function.
3.  Operation conversion patterns.

The function signature conversion has a default implementation that performs
type conversion individually for each of the function arguments and results. A
custom implementation is required when function signature can change when
switching dialects, for example to include dialect-specific attributes or to
accommodate calling conventions.

## Linalg to LLVM IR Conversion

Let us illustrate how one can use the Dialect Conversion framework using Linalg
to LLVM IR Conversion and defining the three components listed above.

Instead of performing progressive lowering from Linalg to Standard dialect, we
will define the semantics of Linalg types and type-related operations in terms
of their LLVM IR counterparts.

### Linalg Types to LLVM IR Types

#### Range Type

The Linalg Range abstraction is a triple of size values representing `min`,
`max` and `step` of the space of iteration (address or loop). This easily maps
to the LLVM IR's structure type containing three integers: `{i64, i64, i64}`,
assuming that 64 bits are sufficient to hold a size.

In a conversion function, this can be implemented by checking if the input type
is indeed `linalg::RangeType` and constructing the corresponding LLVM IR dialect
type. The LLVM IR dialect types are merely LLVM IR types wrapped into an MLIR
object.

```c++
Type linalg::convertLinalgType(Type t) {
  // Obtain the MLIR context and the MLIR LLVM IR dialect.  The latter stores
  // the LLVM context that is necessary to construct types.
  auto *context = t.getContext();
  auto *dialect =
      static_cast<LLVM::LLVMDialect *>(context->getRegisteredDialect("llvm"));

  if (auto rangeTy = t.dyn_cast<linalg::RangeType>()) {
    // Create the LLVM IR i64 type.
    auto *int64Ty = llvm::Type::getInt64Ty(dialect->getLLVMContext());
    // Create the LLVM IR structure type containing three i64.
    auto *structTy = llvm::StructType::get(int64Ty, int64Ty, int64Ty);
    // Wrap the LLVM IR type into the MLIR LLVM IR dialect.
    return LLVM::LLVMType::get(context, structTy);
  }

  // Leave an unknown type as is.
  return t;
}
```

#### View Type

Converting a Linalg View type requires more careful consideration. First, a View
is a container type whose element must be converted as well. Second, it has a
defined semantics that its representation should respect. In particular, a View
is an abstraction around MLIR's standard `memref` type, which itself represents
a pointer with size information attached. Accessing an element through a view
means accessing the same buffer but with additional index arithmetics.

We start by re-postulating the features of the underlying data type, `memref`. A
`memref` is a contiguous array of data indexed by multiple values, typically
stored in a row-major format. We will base our representation of a View on a
notion of _stride_ as used in some machine learning frameworks. A _stride_ along
a given indexing dimension is the number of contiguous elements that separate
two elements with adjacent indices along the given dimension. For example, an
`2x6x4` array will have strides `6*4=24`, `4` and `1`. Strides immediately allow
us to capture the _step_ of the range used to construct the array: the step is
reflected as an additional multiplier in the stride, making View "step over"
elements.

A View will contain as many strides as it has dimensions. For rank-reducing
strides, this allows one to simply remove the stride of the dimension that is
not included in the view. For example, taking a view that projects away the
middle dimension from a `2x6x4` array will give one strides `24` and `1` over
the original buffer.

In addition to steps, ranges used to create a View can impose a lower and an
upper bound on the indices along each dimension. These indices are necessary for
two cases: (a) computing the indices in the original array given the indices in
the view and (b) verifying out-of-bounds accesses and overflows on loads and
stores. For the former purpose, we will introduce the _linearized offset_ below.
For the latter purpose, we will store the _size_ along the given dimension, i.e.
the difference between the maximum and the minimum value of the indices in the
range.

Finally, we need to account for rank-reducing views that fix the projected away
index at a specific value. This cannot be implemented as by keeping the `min`
value for all the projected away dimensions because it would make the
representation of Views obtained from `memref`'s of different ranks different,
defying the purpose of Views. Instead, we will keep only a single value
representing the (linearized) offset of the first contiguous element that can be
accessed. For example, if the second index in a `2x6x4` array was fixed to `3`
when producing a 2D view, the offset will be `3*4=12` elements. Adding the
strides to this offset will let one access the other elements in the view. Since
addresses are linearized anyway, and since we cannot have a rank-expanding view
by construction, it is sufficient to store a single linearized offset.

For the sake of simplicity, we will store the offset separate from the buffer
pointer. Combining the two can save the space required for storing the data but
make functionality like alias analysis more complex. Implementing such a
combination is left as an exercise to the reader.

Bringing all pieces together, we can define a view _descriptor_ that consists of
the following:

1.  the buffer pointer, `T*` where `T` is the result of converting the elemental
    type of the view;
1.  the linearized offset of the first accessed element;
1.  as many values as view rank, representing the size along each dimension;
1.  as many values as view rank, representing the step along each dimension in
    the index space of the original memref;

Using a hypothetical template syntax, the corresponding LLVM IR type would look
like `template <type T, i64 N> { T*, i64, i64[N], i64[N] }`.

Let's start implementing the conversion by extending the type conversion
function to include some primitive types, for example `f32`, `f64` and integers.

```c++
Type linalg::convertLinalgType(Type t) {
/*...*/
  // Construct an LLVM IR integer type of the same width as the MLIR type.
  if (auto intTy = t.dyn_cast<IntegerType>()) {
    int width = intTy.getWidth();
    auto *integerTy = llvm::IntegerType::get(dialect->getLLVMContext(), width);
    return LLVM::LLVMType::get(context, integerTy);
  }
  // Convert f32 to LLVM IR float.
  if (t.isF32()) {
    auto *floatTy = llvm::Type::getFloatTy(dialect->getLLVMContext());
    return LLVM::LLVMType::get(context, floatTy);
  }
  // Convret f64 to LLVM IR double.
  if (t.isF64()) {
    auto *doubleTy = llvm::Type::getDoubleTy(dialect->getLLVMContext());
    return LLVM::LLVMType::get(context, doubleTy);
  }
/*...*/
```

Once properly defined, the conversion of the view type to the view descriptor
type is straightforward:

```c++
/*...*/
  if (auto viewTy = t.dyn_cast<linalg::ViewType>()) {
    // Recursively call the type conversion for the element type, and extract
    // the LLVM IR type form the result.
    Type convertedElemTy = linalg::convertLinalgType(viewTy.getElementType());
    llvm::Type *convertedLLVMElemTy =
        convertedElemTy.cast<LLVM::LLVMType>().getUnderlyingType();
    llvm::PointerType *ptrToElemLLVMTy = convertedLLVMElemTy->getPointerTo();

    // Progressively construct the LLVM IR structure type.
    auto *int64Ty = llvm::Type::getInt64Ty(dialect->getLLVMContext());
    auto *arrayTy = llvm::ArrayType::get(int64Ty, viewTy.getRank());
    auto *structTy = llvm::StructType::get(
        ptrToElemLLVMTy, int64Ty, arrayTy, arrayTy);

    // Wrap the LLVM IR type into the MLIR type.
    return LLVM::LLVMType::get(context, structTy);
  }
/*...*/
```

### Function Signature Conversions

For the sake of simplicity, let's rely on the default implementation of the
function signature conversion that just converts the types.

Note that, in practice, LLVM IR does not support multi-result functions while
MLIR does, which would require changing the function signature and introducing
additional instructions during conversion. You can check how this is
[implemented](https://github.com/tensorflow/mlir/blob/master/lib/LLVMIR/Transforms/ConvertToLLVMDialect.cpp)
in the actual conversion to the LLVM IR dialect.

### Operation Conversions

Operations on the view abstractions are mostly defined by the definition of the
view descriptor: a `linalg.view` operation creates a view descriptor and fills
it in with the initial data; a `linalg.slice` operation creates a new descriptor
from an existing one, with modifications; `linalg.load` and `linalg.store` use
the view descriptor to compute the address of the element before accessing it.

#### `linalg.view`

The role of a `linalg.view` is to construct a view descriptor given a memref, or
rather, a memref descriptor as
[defined](../../ConversionToLLVMDialect.md#memref-model) by the conversion of
the standard dialect to the LLVM IR dialect. A memref descriptor is similar to a
view descriptor: it contains the buffer pointer and the list of _dynamic_ sizes
of the memref. Since memrefs are contiguous, there is no need to store the
offset, the min/max values or the strides. Their static (constant) dimensions
are available directly in the type signature.

An operation conversion is defined as special pattern by inheriting from
`mlir::ConversionPattern` and by reimplementing the matching and the rewriting
functions:

```c++
class ViewOpConversion : public ConversionPattern {
public:
  // A conversion constructor, may take arbtirary operands but must be able
  // to obtain an MLIRContext from them to call the parent constructor.
  explicit ViewOpConversion(MLIRContext *context);

  // A matching function takes an operation and checks whether the pattern is
  // applicable to it by inspecting its properties.
  PatternMatchResult match(Operation *op) const override;

  // A "rewriting" function that takes an original operation `op`, a list of
  // already rewritten opreands, and a function builder `rewriter`. It can use
  // the builder to construct new operations and ultimately create new values
  // that will replace those currently produced by the original operation.  It
  // needs to define as many value as the original operation, but their types
  // may be different.
  SmallVector<Value *, 4> rewrite(Operation *op, ArrayRef<Value *> operands,
                                  OpBuilder &rewriter) const override;
}
```

The `ConversionPattern` constructor takes, in addition to the context, the name
of the main operation to be matched and the "benefit" of a match. These operands
are intended to be useful for defining an optimization problem across multiple
possible conversions but are currently ignored by the conversion framework.

```c++
ViewOpConversion::ViewOpConversion(MLIRContext *context)
      : ConversionPattern(linalg::ViewOp::getOperationName(), 1, context) {}
```

The matching function can be used, for example, to apply different conversions
depending on the argument types or on the attributes of the operation. In our
case, it is applicable for any operation of the given type.

```c++
PatternMatchResult ViewOpConversion::match(Operation *op) const override {
  if (op->isa<linalg::ViewOp>())
    return matchSuccess();
  return matchFailure();
}
```

The actual conversion function may become quite involved. First, Let us go over
the components of a view descriptor and see how they can be constructed to
represent a _complete_ view of a `memref`, e.g. a view that covers all its
elements.

1.  The buffer pointer is copied as is from the memref descriptor.
1.  The linearized offset is always 0.
1.  The size is originally the size of a memref:
    -   static sizes are taken as constants given the values from the memref
        type signature;
    -   dynamic sizes are extracted from the memref descriptor.
1.  The stride along a dimension can be defined recursively as:
    -   the stride along the innermost dimension is always 1;
    -   the stride along any other dimension is the size of the next inner
        dimension times its stride.

When a view is not complete, we need to take into account the ranges supplied as
arguments to the `linalg.view` operation. In particular, the minimum and maximum
index for each dimension is extracted from the corresponding range. The
linearized offset is then computed as a sum of products of minimum indices along
each dimension with the strides of these dimensions.

If a single `index` is supplied instead of a range, i.e. if we have a
rank-reducing view, it will not have a dynamic representation in the view
descriptor. However, its value is used as the minimum value in the linearized
offset computation and the stride for this dimension participates in the
recursive definition, although it is not stored in the descriptor.

The full conversion function is
[available](https://github.com/tensorflow/mlir/blob/master/examples/Linalg1/lib/ConvertToLLVMDialect.cpp)
and accounts for all these details and minimizes the number of instructions it
produces. Let us consider some parts of this functions to understand how it
operates.

```c++
SmallVector<Value *, 4> ViewOpConversion::rewrite(
    Operation *op, ArrayRef<Value *> operands,
    OpBuilder &rewriter) const override {
  // Obtain the typed operation (we know we matched only one type).
  auto viewOp = op->cast<linalg::ViewOp>();

  // Extract the buffer pointer from the memref descriptor (first argument).
  Value *memrefDescriptor = operands[0];
  Value *bufferPtr;
  // The descriptor type varies depending on the memref type signature, so we
  // inspect the _original_ operand that has the memref type.
  auto memrefType = viewOp.getSupportingMemRef()->getType().cast<MemRefType>();

  // Build the "position" attribute, which correspond to the trailing constant
  // operands of the LLVM IR extractvalue instruction.  (Note: in MLIR,
  // compile-time operation parameters are passed in as attributes).  This is
  // an Array attribute holding Integer attributes.  In our case, it only
  // holds one value.  It will be used in insert/extact value below.
  auto attr = rewriter.getIntegerAttr(rewriter.getIntegerType(/*width=*/64),
                                     /*value=*/0);
  auto positionAttr = rewriter.getArrayAttr({attr});

  // Create the context object (RAII) in which we can use declarative builders.
  edsc::ScopedContext context(rewriter, op->getLoc());

  // For statically-shaped memrefs, the descriptor is just the pointer.
  if (memrefType.hasStaticShape()) {
    bufferPtr = memrefType;
  // For dynamically-shaped memrefs, it is a structure whose first element is
  // a pointer.  Extract it from the structure.
  } else {
    // Obtain an LLVM IR pointer type to the element, wrapped in MLIR type.
    Type wrappedElementTy =
        linalg::convertLinalgType(memrefType.getElementType());
    llvm::Type *elementTy =
        wrappedElementTy.cast<LLVM::LLVMType>().getUnderlyingType();
    llvm::Type *elementPtrTy = elementTy->getPointerTo();
    Type wrappedElementPtrTy = rewriter.getType<LLVM::LLVMType>(elementPtrTy);

    // Emit LLVM IR extractvalue to obtain the buffer pointer from the memref
    // descriptor.
    bufferPtr = intrinsics::extractvalue(wrappedElementPtrTy, memrefDescriptor,
                                         positionAttr);
  }

  // Convert the type of the view to get the type of its descriptor.
  Type viewDescriptorType = linalg::convertLinalgType(viewOp.getViewType());

  // Define the descriptor value using the "undef" operation, equivalent to LLVM
  // IR's "undef" value.  (Note: in MLIR, constants are not a special subclass
  // of a value; instead, they are produced by operations that take compile-time
  // constants as attributes and produce regular SSA values).
  Value *viewDescriptor = intrinsics::undef(viewDescriptorType);

  // Insert the buffer pointer into the descriptor using `insertvalue`.
  viewDescriptor = intrinsics::insertvalue(viewDescriptorType, viewDescriptor,
                                           bufferPtr, positionAttr);

  // *** the function continues with the remaining part of the descriptor *** //
}
```

Pay attention to the functions prefixed with `intrinsics`. They use the MLIR's
[declarative builders](DeclarativeBuilders.md) interface for better readability.
They can be rewritten using LLVM-like imperative IR builders. For example, the
`extractvalue` call becomes

```c++
  bufferPtr = rewriter.create<LLVM::ExtractValueOp>(
      op->getLoc(), wrappedElementPtrTy, memrefDescriptor, positionAttr);
```

#### `linalg.slice`

The slice operation creates a view from another view, changing only a single
dimension, so its conversion is significantly simpler. Practically, it creates a
new view descriptor and fills it given the old descriptor and the range
information supplied as argument. The minimum and maximum index values are
updated with those supplied in the range, the linearized offset is recomputed
with the new minimum, and the stride along the dimension is multiplied with the
step of the range. If an index is supplied instead of the range, the minimum,
maximum index and the stride corresponding to the slicing dimension are simply
omitted in the new descriptor while the linearized offset is recomputed using
the index as minimum value.

In order to avoid the proliferation of magic constants in insert/extractvalue
operations for the descriptor, we can define an auxiliary IR-emitting data
structure around it as follows.

```c++
struct ViewDescriptor {
  // Obtain the type of the descriptor.
  Type type() { return d->getType(); }

  // Obtain the pointer type to the element.
  Type elementPtrType() {
    llvm::Type *ty = type().cast<LLVM::LLVMType>().getUnderlyingType();
    llvm::StructType *structTy = cast<llvm::StructType>(ty);
    return builder.getType<LLVM::LLVMType>(structTy->getElementType(0));
  }

  // Construct the materialization of the index type (currently, i64).
  Type indexType() {
    auto *dialect = static_cast<LLVM::LLVMDialect *>(
        builder.getContext().getRegisteredDialect("llvm"));
    llvm::Type *ty = llvm::Type::getInt64Ty(dialect->getLLVMContext());
    return builder.getType<LLVM::LLVMType>(ty);
  }

  // Get the array attribute containing the given values as integer attributes.
  Attribute pos(ArrayRef<unsigned> position) {
    SmallVector<Attribute, 4> attrs;
    attrs.reserve(position.size());
    for (auto p : position)
      attrs.push_back(builder.getI64IntegerAttr(p));
    return builder.getArrayAttr(attrs);
  }

  // Emit instructions obtaining individual values from the decsriptor.
  Value *ptr() { return intrinsics::extractvalue(elementPtrType(), d, pos(0)); }
  Value *offset() { return intrinsics::extractvalue(indexType(), d, pos(1)); }
  Value *size(unsigned dim) {
     return intrinsics::extractvalue(indexType(), d, pos({2, dim}));
  }
  Value *stride(unsigned dim) {
     return intrinsics::extractvalue(indexType(), d, pos({3, dim}));
  }

  // Emit instructions inserting individual values in the descriptor.
  void setPtr(Value *v) {
    return intrinsics::insertvalue(type(), d, v, pos(0));
  }
  void setOffset(Value *v) {
    return intrinsics::insertvalue(type(), d, v, pos(1));
  }
  void setSize(unsigned dim, Value *v) {
    return intrinsics::insertvalue(type(), d, v, pos({2, dim}));
  }
  void setStride(unsigned dim, Value *v) {
    return intrinsics::insertvalue(type(), d, v, pos({3, dim}));
  }

  // The builder into which we emit code.
  OpBuilder &builder;

  // The actual descriptor.
  Value *d;
};
```

With such a descriptor, the conversion function resembles closely the conversion
rules described above:

```c++
SmallVector<Value *, 4> SliceOpConversion::rewrite(
    Operation *op, ArrayRef<Value *> operands,
    OpBuilder &rewriter) const override {
  // Obtain the typed operation (we know we matched only one type).
  auto sliceOp = op->cast<linalg::SliceOp>();

  // Create the context object (RAII) in which we can use declarative builders.
  // Bring all the builders into the namespace.
  using namespace intrinsics;
  edsc::ScopedContext context(rewriter, op->getLoc());

  auto newViewDescriptorType =
    linalg::convertLinalgType(sliceOp.getViewType());

  // Define the new descriptor and obtain the old descriptor.
  Value *newViewDescriptor = intrinsics::undef(newViewDescriptorType);
  Value *oldViewDescriptor = operands[0];

  // Create the code-emitting objects around the descriptors.  The range
  // descriptor is defined similarly to the view descriptor with additional
  // support on the value being either of linalg.range type or of index type.
  auto newDescriptor = ViewDescriptor{rewriter, newViewDescriptor};
  auto oldDescriptor = ViewDescriptor{rewriter, oldViewDescriptor};
  auto rangeDescriptor = RangeDescriptor{rewriter, operands[1]};

  // Properties of the slice.
  bool isRankDecreasing = sliceOp.isRankDecreasing();
  int dim = sliceOp.getSlicingDim();

  // Copy the buffer pointer.
  newDecsriptor.setPtr(oldDescriptor.ptr());

  // Recompute the offset.
  Value *min = rangeDescriptor.min();
  Value *stride = oldDescriptor.stride(dim);
  newDescriptor.setOffset(add(oldDescriptor.getOffset(), mul(min, stride)));

  // Copy the sizes and strides into the new descriptor, updating or dropping
  // the affected dimension.  If the `slice` is rank-decreasing, the resulting
  // view will no longer one of the dimensions, its size and stride become
  // unnecessary and can be dropped.  Otherwise, the size of the affected
  // updated to the size of the range and its stride is multiplied with the step
  // of the range.
  for (int i = 0, e = sliceOp.getRank(); i < e; ++i) {
    int originalPos = (isRankDecreasing && i >= dim) ? i + 1 : i;
    if (!isRankDecreasing && i == dim) {
      newDescriptor.setSize(
          i, sub(rangeDescriptor.max(), rangeDescriptor.min()));
      newDescriptor.setStride(
          i, mul(oldDescriptor.getStride(i), rangeDescriptor.step()));
    } else {
      newDescriptor.setSize(i, oldDescriptor.getSize(originalPos));
      newDescriptor.setStride(i, oldDescriptor.getStride(originalPos));
    }
  }

  return {newViewDescriptor};
}
```

Note that we used a `using namespace intrinsics` statement to make the
declarative builders for the LLVM IR dialect operations available without extra
qualification in order to make composed expressions even simpler. We also
omitted the matching function that is similar to that of the `linalg.view`.

#### `linalg.load` and `linalg.store`

Loads and stores through views are implemented in a similar fashion. Both need
to first compute the effective linearized address of the element in the
underlying buffer, and then emit either a load or a store operation on that
address. The linearization is straightforward given the presence of the offset
and strides in the descriptor: the total offset is the sum of the base offset
and the products between access subscripts with strides along the given
dimension.

The linearization part can be easily implemented using the code emitting object
for the view descriptor:

```c++
Value *obtainDataPtr(Location loc, int rank, Value *viewDescriptorVal,
                     ArrayRef<Value *> indices, OpBuilder &rewriter) {
  // Create the context object (RAII) in which we can use declarative builders.
  // Bring all the builders into the namespace.
  using namespace intrinsics;
  edsc::ScopedContext context(rewriter, loc);

  // Create the code emitting object for the descriptor.
  auto viewDescriptor = ViewDescriptor{rewriter, viewDescriptorVal};

  // Linearize subscripts as:
  //   base_offset + SUM_i index_i * stride_i.
  Value *offset = viewDescriptor.offset();
  for (int i = 0; i < rank; ++i) {
    Value *stride = viewDescriptor.getStride(i);
    offset = add(offset, mul(indices[i], stride));
  }

  // Emit a getelementptr instruction with the linearized offset from the buffer
  // pointer, producing a pointer to the accessed element.
  Value *elementPtr = gep(viewDescriptor.elementPtrType(), viewDescriptor.ptr(),
                          ArrayRef<Value *>{offset});
  return elementPtr;
}
```

Given this utility function template, it becomes easy to implement the actual
conversions for load and store operations.

```c++
// Load Operation Conversion.
SmallVector<Value *, 4> LoadOpConversion::rewrite(
    Operation *op, ArrayRef<Value *> operands,
    OpBuilder &rewriter) const override {
  // Obtain the typed operation (we know we matched only one type).
  auto loadOp = op->cast<linalg::LoadOp>();

  // Separate the descriptor operand from the index operands.
  Value *viewDescriptor = operands[0];
  ArrayRef<Value *> indices = operands.drop_front();

  // Call the auxiliary function to emit code computing the element pointer.
  Value *ptr = obtainDataPtr(op->getLoc(), loadOp->getRank(), viewDescriptor,
                             indices, rewriter);

  // Use declarative builders to load from the element pointer.
  edsc::ScopedContext edscContext(rewriter, op->getLoc());
  auto elementType = linalg::convertLinalgType(*op->getResultTypes().begin());
  Value *element = intrinsics::load(elementType, ptr);
  return {element};
}

// Store Operation Conversion
SmallVector<Value *, 4> StoreOpConversion::rewrite(
    Operation *op, ArrayRef<Value *> operands,
    OpBuilder &rewriter) const override {
  // Obtain the typed operation (we know we matched only one type).
  auto loadOp = op->cast<linalg::StoreOp>();

  // Separate the value and descriptor operands from the index operands.
  Value *data = operands[0];
  Value *viewDescriptor = operands[1];
  ArrayRef<Value *> indices = operands.drop_front(2);

  // Call the auxiliary function to emit code computing the element pointer.
  Value *ptr = obtainDataPtr(op->getLoc(), loadOp->getRank(), viewDescriptor,
                             indices, rewriter);

  // Use declarative builders to load from the element pointer.
  edsc::ScopedContext edscContext(rewriter, op->getLoc());
  Value *element = intrinsics::store(data, ptr);

  // "Store" does not produce any values.
  return {};
}
```

### Putting It All Together

Having defined the conversions for the types and the operations, we can now
proceed to invoking the dialect conversion framework that will transform entire
MLIR modules for us. The conversion class must inherit from `DialectConversion`
and override two pure virtual functions: one that initializes the list of
operation converters, and another one that is called to convert individual
types. Function signature conversion function can also be overridden but it has
a default implementation.

```c++
// Define a dialect conversion class.
class Lowering : public DialectConversion {
protected:
  // Produce a set of operation conversion patterns.  This is called once per
  // conversion.
  llvm::DenseSet<ConversionPattern *>
  initConverter(MLIRContext *context) override {
    allocator.Reset();
    // Call a helper function provided by MLIR to build a set of operation
    // conversion instances given a list of classes as template parameters.
    // These instances will be allocated within `allocator` and their lifetime
    // is managed by the Lowering class.
    return RewriteListBuilder<
        LoadOpConversion, SliceOpConversion, StoreOpConversion,
        ViewOpConversion>::build(allocator, context);
  }

  // Convert a type.  This function will be called for each function/region
  // argument or result type (unless there is a custom function signature
  // conversion) as well as for each block argument type.
  Type convertType(Type t) override { return linalg::convertLinalgType(t); }

  // The individual conversion patterns will live here.
  llvm::BumpPtrAllocator allocator;
};
```
