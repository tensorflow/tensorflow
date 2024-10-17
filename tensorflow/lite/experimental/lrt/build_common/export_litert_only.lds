VERS_1.0 {

  /*
    Export abi-stable "vendor" implemented symbols.

    TODO: Add all vendor symbols. Also export qnn libc++ symbols
    (statically linked) as "protected" as needed.
  */

  global:

    /* Compiler Plugin */

    *LiteRtCompilerPlugin;
    *LiteRtPluginInit;
    *LiteRtPluginDestroy;
    *LiteRtPluginSocManufacturer;
    *LiteRtPluginNumSupportedSocModels;
    *LiteRtPluginGetSupportedSocModel;
    *LiteRtPluginPartitionModel;
    *LiteRtPluginCompile;

    /* Compiled Result */

    *LiteRtCompiledResult;
    *LiteRtCompiledResultDestroy;
    *LiteRtCompiledResultGetByteCode;
    *LiteRtCompiledResultGetCallInfo;
    *LiteRtCompiledResultGetNumCalls;

  local:

    /* Hide everything else */

    *;
};