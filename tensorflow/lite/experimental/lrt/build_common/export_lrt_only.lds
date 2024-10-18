VERS_1.0 {

  /*
    Export abi-stable "vendor" implemented symbols.

    TODO: Add all vendor symbols. Also export qnn libc++ symbols
    (statically linked) as "protected" as needed.
  */

  global:

    /* Compiler Plugin */

    *LrtCompilerPlugin;
    *LrtPluginInit;
    *LrtPluginDestroy;
    *LrtPluginSocManufacturer;
    *LrtPluginNumSupportedSocModels;
    *LrtPluginGetSupportedSocModel;
    *LrtPluginPartitionModel;
    *LrtPluginCompile;

    /* Compiled Result */

    *LrtCompiledResult;
    *LrtCompiledResultDestroy;
    *LrtCompiledResultGetByteCode;
    *LrtCompiledResultGetCallInfo;
    *LrtCompiledResultGetNumCalls;

  local:

    /* Hide everything else */

    *;
};