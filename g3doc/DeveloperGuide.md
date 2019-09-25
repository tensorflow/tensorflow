# Developer Guide

This document attempts to describe a few developer policies used in MLIR (such
as coding standards used) as well as development approach (such as, testing
methods).

## Style guide

MLIR follows the [LLVM style](https://llvm.org/docs/CodingStandards.html) guide.
We also adhere to the following, that deviates from, or isn't specified in the
LLVM style guide:

*   Adopts [camelBack](https://llvm.org/docs/Proposals/VariableNames.html);
*   Except for IR units (Region, Block, and Operation), non-nullable output
    arguments are passed by non-const reference in general.
*   IR constructs are not designed for [const correctness](UsageOfConst.md).
*   Do *not* use recursive algorithms if the recursion can't be bounded
    statically: that is avoid recursion if there is a possible IR input that can
    trigger a stack overflow (for example traversing use-def chains in a
    recursive way). At the moment, we tolerate it for the two following cases:
    *   The nesting of the IR: we use recursion when traversing nested regions.
    *   Type nesting: recursion may be used for the nesting of composite types.

Please run clang-format on the files you modified with the `.clang-format`
configuration file available in the root directory. Check the clang-format
[documentation](https://clang.llvm.org/docs/ClangFormat.html) for more details
on integrating it with your development environment. In particular, if clang is
installed system-wide, running `git clang-format origin/master` will update the
files in the working directory with the relevant formatting changes; don't
forget to include those to the commit.

## Pass name and other command line options

To avoid collision between options provided by different dialects, the naming
convention is to prepend the dialect name to every dialect-specific passes and
options in general. Options that are specific to a pass should also be prefixed
with the pass name. For example, the affine dialect provides a loop tiling pass
that is registered on the command line as `-affine-tile`, and with a tile size
option that can be set with `-affine-tile-size`.

## Testing guidelines

See here for the [testing guide](TestingGuide.md).
