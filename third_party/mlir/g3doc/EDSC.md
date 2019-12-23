# Background: declarative builders API

The main purpose of the declarative builders API is to provide an intuitive way
of constructing MLIR programmatically. In the majority of cases, the IR we wish
to construct exhibits structured control-flow. Declarative builders provide an
API to make MLIR construction and manipulation very idiomatic, for the
structured control-flow case, in C++.

## ScopedContext

`mlir::edsc::ScopedContext` provides an implicit thread-local context,
supporting a simple declarative API with globally accessible builders. These
declarative builders are available within the lifetime of a `ScopedContext`.

## ValueHandle and IndexHandle

`mlir::edsc::ValueHandle` and `mlir::edsc::IndexHandle` provide typed
abstractions around an `mlir::Value`. These abstractions are "delayed", in the
sense that they allow separating declaration from definition. They may capture
IR snippets, as they are built, for programmatic manipulation. Intuitive
operators are provided to allow concise and idiomatic expressions.

```c++
ValueHandle zero = constant_index(0);
IndexHandle i, j, k;
```

## Intrinsics

`mlir::edsc::ValueBuilder` is a generic wrapper for the `mlir::Builder::create`
method that operates on `ValueHandle` objects and return a single ValueHandle.
For instructions that return no values or that return multiple values, the
`mlir::edsc::InstructionBuilder` can be used. Named intrinsics are provided as
syntactic sugar to further reduce boilerplate.

```c++
using load = ValueBuilder<LoadOp>;
using store = InstructionBuilder<StoreOp>;
```

## LoopBuilder and AffineLoopNestBuilder

`mlir::edsc::AffineLoopNestBuilder` provides an interface to allow writing
concise and structured loop nests.

```c++
  ScopedContext scope(f.get());
  ValueHandle i(indexType),
              j(indexType),
              lb(f->getArgument(0)),
              ub(f->getArgument(1));
  ValueHandle f7(constant_float(llvm::APFloat(7.0f), f32Type)),
              f13(constant_float(llvm::APFloat(13.0f), f32Type)),
              i7(constant_int(7, 32)),
              i13(constant_int(13, 32));
  AffineLoopNestBuilder(&i, lb, ub, 3)([&]{
      lb * index_t(3) + ub;
      lb + index_t(3);
      AffineLoopNestBuilder(&j, lb, ub, 2)([&]{
          ceilDiv(index_t(31) * floorDiv(i + j * index_t(3), index_t(32)),
                  index_t(32));
          ((f7 + f13) / f7) % f13 - f7 * f13;
          ((i7 + i13) / i7) % i13 - i7 * i13;
      });
  });
```

## IndexedValue

`mlir::edsc::IndexedValue` provides an index notation around load and store
operations on abstract data types by overloading the C++ assignment and
parenthesis operators. The relevant loads and stores are emitted as appropriate.

## Putting it all together

With declarative builders, it becomes fairly concise to build rank and
type-agnostic custom operations even though MLIR does not yet have generic
types. Here is what a definition of a general pointwise add looks in
Tablegen with declarative builders.

```c++
def AddOp : Op<"x.add">,
    Arguments<(ins Tensor:$A, Tensor:$B)>,
    Results<(outs Tensor: $C)> {
  code referenceImplementation = [{
    auto ivs = makeIndexHandles(view_A.rank());
    auto pivs = makePIndexHandles(ivs);
    IndexedValue A(arg_A), B(arg_B), C(arg_C);
    AffineLoopNestBuilder(pivs, view_A.getLbs(), view_A.getUbs(), view_A.getSteps())(
      [&]{
        C(ivs) = A(ivs) + B(ivs)
      });
  }];
}
```

Depending on the function signature on which this emitter is called, the
generated IR resembles the following, for a 4-D memref of `vector<4xi8>`:

```
// CHECK-LABEL: func @t1(%lhs: memref<3x4x5x6xvector<4xi8>>, %rhs: memref<3x4x5x6xvector<4xi8>>, %result: memref<3x4x5x6xvector<4xi8>>) -> () {
//       CHECK: affine.for {{.*}} = 0 to 3 {
//       CHECK:   affine.for {{.*}} = 0 to 4 {
//       CHECK:     affine.for {{.*}} = 0 to 5 {
//       CHECK:       affine.for {{.*}}= 0 to 6 {
//       CHECK:         {{.*}} = load %arg1[{{.*}}] : memref<3x4x5x6xvector<4xi8>>
//       CHECK:         {{.*}} = load %arg0[{{.*}}] : memref<3x4x5x6xvector<4xi8>>
//       CHECK:         {{.*}} = addi {{.*}} : vector<4xi8>
//       CHECK:         store {{.*}}, %arg2[{{.*}}] : memref<3x4x5x6xvector<4xi8>>
```

or the following, for a 0-D `memref<f32>`:

```
// CHECK-LABEL: func @t3(%lhs: memref<f32>, %rhs: memref<f32>, %result: memref<f32>) -> () {
//       CHECK: {{.*}} = load %arg1[] : memref<f32>
//       CHECK: {{.*}} = load %arg0[] : memref<f32>
//       CHECK: {{.*}} = addf {{.*}}, {{.*}} : f32
//       CHECK: store {{.*}}, %arg2[] : memref<f32>
```

Similar APIs are provided to emit the lower-level `loop.for` op with
`LoopNestBuilder`. See the `builder-api-test.cpp` test for more usage examples.

Since the implementation of declarative builders is in C++, it is also available
to program the IR with an embedded-DSL flavor directly integrated in MLIR. We
make use of these properties in the tutorial.

Spoiler: MLIR also provides Python bindings for these builders, and a
full-fledged Python machine learning DSL with automatic differentiation
targeting MLIR was built as an early research collaboration.

