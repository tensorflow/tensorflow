# Developer Guide

This document attempts to describe a few developer policies used in MLIR (such
as coding standards used) as well as development approach (such as, testing
methods).

## Style guide

MLIR follows the [LLVM style](https://llvm.org/docs/CodingStandards.html) guide
except:

*   Adopts [camelBack](https://llvm.org/docs/Proposals/VariableNames.html);

## Pass name and other command line options

To avoid collision between options provided by different dialects, the naming
convention is to prepend the dialect name to every dialect-specific passes and
options in general. Also options that are specific to a pass should also be
prefixed with the pass name. For example, the affine dialect is providing a
loop tiling pass that will be registered on the command line as "-affine-tile",
and the tile size option can be set with "-affine-tile-size".
