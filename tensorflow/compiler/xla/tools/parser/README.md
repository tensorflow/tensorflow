# HloModule string syntax

TODO: Support subcomputations (for fusion, reduce, while, ...).

TODO: Support ops that require extra attributes, e.g. dimensions, strides.

```yacc
hlo_module
  : 'HloModule' name computation
  ;

computation
  : 'ENTRY' name param_list '->' shape instruction_list
  ;

instruction_list
  : '{' instruction_list1 '}'
  ;
instruction_list1
  : instruction
  | instruction_list1 instruction
  ;
instruction
  : name '=' shape opcode operands
  ;

operands
  : '(' operands1 ')'
  ;
operands1
  : /*empty*/
  | operand
  | operands1 ',' operand
  ;
operand
  : shape name
  ;

param_list
  : '(' param_list1 ')'
  ;
param_list1
  : /*empty*/
  | param
  | param_list1 ',' param
  ;
param
  : name shape
  ;

shape
  : shape_val_
  | '(' tuple_elements ')'
  ;
tuple_elements
  : /*empty*/
  | shape (',' shape)*
  ;

name
  : identifier ':'
  | '%' identifier
  ;

identifier
  : [a-zA-Z_][a-zA-Z0-9_.-]*
  ;

```
