# HloModule string syntax

TODO: Support all subcomputations (for fusion, reduce, ...).

TODO: Support all extra attributes, e.g. dimensions, strides.

```yacc
hlo_module
  : 'HloModule' name computations
  ;

computations
  : computation
  | computation computations
  ;

computation
  : 'ENTRY' name param_list '->' shape instruction_list
  | name param_list '->' shape instruction_list
  ;

instruction_list
  : '{' instruction_list1 '}'
  ;
instruction_list1
  : instruction
  | instruction_list1 instruction
  ;
instruction
  : 'ROOT' name '=' shape opcode operands extra_attributes
  | name '=' shape opcode operands extra_attributes
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

extra_attributes
  : /*empty*/
  | ',' extra_attribute
  | ',' extra_attribute extra_attributes
  ;
extra_attribute
  : attribute_name attribute_value
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

/* literal is in the right hand side of a constant instruction. */
literal
  : tuple
  | non_tuple
  ;
tuple
  : shape '(' literal_list ')'
  ;
literal_list
  : /*empty*/
  : literal
  | literal_list ',' literal
  ;
non_tuple
  : rank01
  | rank2345
  ;
rank2345
  : shape nested_array
  ;

```
