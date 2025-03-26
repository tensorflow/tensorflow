This temporary directory was created to store MHLO pass .cc and .h files. These
files have been migrated to StableHLO but are still used by inactive or
potentially outdated compilation paths. Once all MHLO passes have been migrated
to StableHLO, revisit this directory. At that point, we can replace the uses of
MHLO passes from this directory with the StableHLO passes.