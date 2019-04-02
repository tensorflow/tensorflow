# Chapter 1: Toy Tutorial Introduction

This tutorial runs through the implementation of a basic toy language on top of
MLIR. The goal of this tutorial is to introduce the concepts of MLIR, and
especially how *dialects* can help easily support language specific constructs
and transformations, while still offering an easy path to lower to LLVM or other
codegen infrastructure. This tutorial is based on the model of the
[LLVM Kaleidoscope Tutorial](https://llvm.org/docs/tutorial/LangImpl01.html).

This tutorial is divided in the following chapters:

-   [Chapter #1](Ch-1.md): Introduction to the Toy language, and the definition
    of its AST.
-   [Chapter #2](Ch-2.md): Traversing the AST to emit custom MLIR, introducing
    base MLIR concepts.
-   [Chapter #3](Ch-3.md): Defining and registering a dialect in MLIR, showing
    how we can start attaching semantics to our custom operations in MLIR.
-   [Chapter #4](Ch-4.md): High-level language-specific analysis and
    transformation, showcasing shape inference, generic function specialization,
    and basic optimizations.
-   [Chapter #5](Ch-5.md): Lowering to lower-level dialects. We'll convert our
    high level language specific semantics towards a generic linear-algebra
    oriented dialect for optimizations. Ultimately we will emit LLVM IR for code
    generation.
-   [Chapter #5](Ch-6.md): A REPL?
-   [Chapter #6](Ch-7.md): Custom backends? GPU using LLVM? TPU? XLA

## The Language

This tutorial will be illustrated with a toy language that we’ll call “Toy”
(naming is hard...). Toy is an array-based language that allows you to define
functions, some math computation, and print results.

Because we want to keep things simple, the codegen will be limited to arrays of
rank <= 2 and the only datatype in Toy is a 64-bit floating point type (aka
‘double’ in C parlance). As such, all values are implicitly double precision,
Values are immutable: every operation returns a newly allocated value, and
deallocation is automatically managed. But enough with the long description,
nothing is better than walking through an example to get a better understanding:

FIXME: update/modify matrix multiplication to use @ instead of *

```Toy {.toy}
def main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  # The shape is inferred from the supplied literal.
  var a = [[1, 2, 3], [4, 5, 6]];
  # b is identical to a, the literal array is implicitely reshaped: defining new
  # variables is the way to reshape arrays (element count must match).
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  # transpose() and print() are the only builtin, the following will transpose
  # b and perform a matrix multiplication before printing the result.
  print(a * transpose(b));
}
```

Type checking is statically performed through type inference, the language only
requires type declarations to specify array shapes when needed. Function are
generic: their parameters are unranked (in other word we know these are arrays
but we don't know how many dimensions or the size of the dimensions). They are
specialized for every newly discovered signature at call sites. Let's revisit
the previous example by adding a user-defined function:

```Toy {.toy}
# User defined generic function that operates on unknown shaped arguments
def multiply_transpose(a, b) {
  return a * transpose(b);
}

def main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  # This call will specialize `multiply_transpose` with <2, 3> for both
  # arguments and deduce a return type of <2, 2> in initialization of `c`.
  var c = multiply_transpose(a, b);
  # A second call to `multiply_transpose` with <2, 3> for both arguments will
  # reuse the previously specialized and inferred version and return `<2, 2>`
  var d = multiply_transpose(b, a);
  # A new call with `<2, 2>` for both dimension will trigger another
  # specialization of `multiply_transpose`.
  var e = multiply_transpose(c, d);
  # Finally, calling into `multiply_transpose` with incompatible shape will
  # trigger a shape inference error.
  var e = multiply_transpose(transpose(a), c);
}
```

## The AST

The AST is fairly straightforward from the above code, here is a dump of it:

```
Module:
  Function
    Proto 'multiply_transpose' @test/ast.toy:5:1'
    Args: [a, b]
    Block {
      Return
        BinOp: * @test/ast.toy:6:12
          var: a @test/ast.toy:6:10
          Call 'transpose' [ @test/ast.toy:6:14
            var: b @test/ast.toy:6:24
          ]
    } // Block
  Function
    Proto 'main' @test/ast.toy:9:1'
    Args: []
    Block {
      VarDecl a<2, 3> @test/ast.toy:11:3
        Literal: <2, 3>[<3>[1.000000e+00, 2.000000e+00, 3.000000e+00], <3>[4.000000e+00, 5.000000e+00, 6.000000e+00]] @test/ast.toy:11:17
      VarDecl b<2, 3> @test/ast.toy:12:3
        Literal: <6>[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00] @test/ast.toy:12:17
      VarDecl c<> @test/ast.toy:15:3
        Call 'multiply_transpose' [ @test/ast.toy:15:11
          var: a @test/ast.toy:15:30
          var: b @test/ast.toy:15:33
        ]
      VarDecl d<> @test/ast.toy:18:3
        Call 'multiply_transpose' [ @test/ast.toy:18:11
          var: b @test/ast.toy:18:30
          var: a @test/ast.toy:18:33
        ]
      VarDecl e<> @test/ast.toy:21:3
        Call 'multiply_transpose' [ @test/ast.toy:21:11
          var: b @test/ast.toy:21:30
          var: c @test/ast.toy:21:33
        ]
      VarDecl e<> @test/ast.toy:24:3
        Call 'multiply_transpose' [ @test/ast.toy:24:11
          Call 'transpose' [ @test/ast.toy:24:30
            var: a @test/ast.toy:24:40
          ]
          var: c @test/ast.toy:24:44
        ]
    } // Block
```

You can reproduce this result and play with the example in the `examples/Ch1/`
directory, try running `path/to/BUILD/bin/toyc test/ast.toy -emit=ast`.

The code for the lexer is fairly straighforward, it is all in a single header:
`examples/toy/Ch1/include/toy/Lexer.h`. The parser can be found in
`examples/toy/Ch1/include/toy/Parser.h`, it is a recursive descent parser. If
you are not familiar with such Lexer/Parser, these are very similar to the LLVM
Kaleidoscope equivalent that are detailed in the first two chapters of the
[Kaleidoscope Tutorial](https://llvm.org/docs/tutorial/LangImpl02.html#the-abstract-syntax-tree-ast).

The [next chapter](Ch-2.md) will demonstrate how to convert this AST into MLIR.
