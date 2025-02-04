VERS_1.0 {

  /*
    Export abi-stable "vendor" implemented symbols.

    TODO: Add all vendor symbols. Also export qnn libc++ symbols
    (statically linked) as "protected" as needed.
  */

  global:

    /* All LiteRt C APIs */
    LiteRt*;

  local:

    /* Hide everything else */

    *;
};
