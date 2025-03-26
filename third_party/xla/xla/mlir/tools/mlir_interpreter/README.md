# MLIR interpreter

MLIR interpreter is a development tool, mainly intended for debugging. As the
name implies, it executes MLIR programs. What makes it special is that it
implements dialects at all levels of abstraction and all standard data types
(tensor, memref, vector, scalar). One use case is to track down compiler bugs
by executing the IR after each one and looking for changed results.

Features:
- Tensors, memrefs, vectors, scalars
- Many dialects supported: from mhlo via linalg and scf all the way to arith.
- Easy to extend (look at dialects/affine.cc for an example).

## Creating a new dialect

To create a new dialect, follow the following steps:

1.  Create a `.cc` implementation file in `dialects/`. The name of the file
    should match your dialect name.
1.  Add a dependency to your dialect to the `dialects:dialects` target.
1.  Each op is implemented in a function.
    -   Op results:
        -   If your op has no result, the function should be `void`.
        -   If your op has a tensor, memref or vector result, the return type
            should be `InterpreterValue`.
        -   Scalars can be returned as their corresponding C++ type (e.g.
            `int64_t`, `float`)
        -   You can also return a `SmallVector` if your op has multiple results.
    -   Op arguments:
        -   Your function should take an `InterpreterState&`, an instance of
            your op and any arguments your op takes.
        -   Like above, arguments can either be taken as `InterpreterValue`s, as
            scalar C++ types or as `SmallVector` thereof (for variadic
            arguments).
    -   Error handling:
        -   If some of the inputs are invalid, use
            `InterpreterState::AddFailure` to report this.
1.  Register the ops: `REGISTER_MLIR_INTERPRETER_OP(YourOp)`.
1.  Add test cases in `dialects/tests`.
