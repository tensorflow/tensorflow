" Vim syntax file
" Language:   mlir
" Maintainer: The MLIR team, http://github.com/tensorflow/mlir/
" Version:      $Revision$
" Adapted from the LLVM vim indent file

if version < 600
  syntax clear
elseif exists("b:current_syntax")
  finish
endif

syn case match

" Types.
syn keyword mlirType index f16 f32 f64
" Integer type.
syn match mlirType /\<i\d\+\>/
" Primitive types inside memref, tensor, or vector.
syn match mlirType /<.*x\s*\zs\(f16\|f32\|f64\|i\d\+\)/
syn match mlirType /\<memref\ze<.*>/
syn match mlirType /\<tensor\ze<.*>/
syn match mlirType /\<vector\ze<.*>/

" Operations.
" Core ops (not exhaustive yet).
" TODO: the list is not exhaustive.
syn keyword mlirOps alloc addf addi call call_indirect cmpi constant dealloc dma_start dma_wait dim extract_element getTensor load memref_cast mulf muli store select subf subi tensor_cast

" Affine ops.
syn match mlirOps /\<affine\.apply\>/
syn match mlirOps /\<affine\.dma_start\>/
syn match mlirOps /\<affine\.dma_wait\>/
syn match mlirOps /\<affine\.for\>/
syn match mlirOps /\<affine\.if\>/
syn match mlirOps /\<affine\.load\>/
syn match mlirOps /\<affine\.store\>/

" Keywords.
syn keyword mlirKeyword
      \ else
      \ func
      \ return
      \ step
      \ to

" Misc syntax.
syn match   mlirNoName /[%@!]\d\+\>/
syn match   mlirNumber /-\?\<\d\+\>/
syn match   mlirFloat  /-\?\<\d\+\.\d*\(e[+-]\d\+\)\?\>/
syn match   mlirFloat  /\<0x\x\+\>/
syn keyword mlirBoolean true false
syn match   mlirComment /\/\/.*$/
syn region  mlirString start=/"/ skip=/\\"/ end=/"/
syn match   mlirLabel /[-a-zA-Z$._][-a-zA-Z$._0-9]*:/
syn match   mlirIdentifier /[%@][-a-zA-Z$._][-a-zA-Z$._0-9]*/
syn match mlirMapSetOutline "#.*$"

" Syntax-highlight lit test commands and bug numbers.
syn match  mlirSpecialComment /\/\/\s*RUN:.*$/
syn match  mlirSpecialComment /\/\/\s*CHECK:.*$/
syn match  mlirSpecialComment "\v\/\/\s*CHECK-(NEXT|NOT|DAG|SAME|LABEL):.*$"
syn match  mlirSpecialComment /\/\/\s*expected-error.*$/
syn match  mlirSpecialComment /\/\/\s*expected-remark.*$/
syn match  mlirSpecialComment /;\s*XFAIL:.*$/
syn match  mlirSpecialComment /\/\/\s*PR\d*\s*$/
syn match  mlirSpecialComment /\/\/\s*REQUIRES:.*$/

if version >= 508 || !exists("did_c_syn_inits")
  if version < 508
    let did_c_syn_inits = 1
    command -nargs=+ HiLink hi link <args>
  else
    command -nargs=+ HiLink hi def link <args>
  endif

  HiLink mlirType Type
  HiLink mlirOps Statement
  HiLink mlirMapSetOutline PreProc
  HiLink mlirNumber Number
  HiLink mlirComment Comment
  HiLink mlirString String
  HiLink mlirLabel Label
  HiLink mlirKeyword Keyword
  HiLink mlirBoolean Boolean
  HiLink mlirFloat Float
  HiLink mlirNoName Identifier
  HiLink mlirConstant Constant
  HiLink mlirSpecialComment SpecialComment
  HiLink mlirIdentifier Identifier

  delcommand HiLink
endif

let b:current_syntax = "mlir"
