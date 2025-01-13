VERS_1.0 {

  /*
    Export abi-stable "vendor" implemented symbols.

    TODO: Add all vendor symbols. Also export qnn libc++ symbols
    (statically linked) as "protected" as needed.
  */

  global:

    /* Compiler Plugin */

    LiteRt*CompilerPlugin*;

    /* Compiled Result */

    LiteRt*CompiledResult*;

    /* Dispatch */

    LiteRtDispatch*;

  local:

    /* Hide everything else */

    *;
};
