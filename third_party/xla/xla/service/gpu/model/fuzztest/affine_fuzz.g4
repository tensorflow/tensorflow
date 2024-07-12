/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Grammar for generating random affine expressions. NOTE: this is not a
// complete grammar for affine expressions! We do not consider expressions that
// do not occur in the indexing maps used by MLIR codegen:
//   - ceildiv
//   - non-constant RHS for mod, floordiv and mul
// We also don't consider expressions with more than two dimensions or symbols.
// This gives us up to four variables in total, which should be enough.

grammar AFFINE_FUZZ;

affine: '(d0, d1)[s0, s1] -> (' expr ')';
floordiv: expr ' floordiv ' NONZERO;
mod: expr ' mod ' POSITIVE;
mul: expr ' * ' INTEGER;
sum: expr ' + ' expr;
expr: INTEGER | SYM | DIM | '(' floordiv ')' | '(' sum ')' | '(' mul ')' | '(' mod ')';

SYM : 's' [01];
DIM : 'd' [01];
ONETONINE : [1-9];
DIGITS : (DIGIT | DIGIT DIGIT)?;
DIGIT : '0' | ONETONINE;
NONZERO : '-'? ONETONINE DIGITS;
POSITIVE: ONETONINE DIGITS;
INTEGER: NONZERO | '0';
