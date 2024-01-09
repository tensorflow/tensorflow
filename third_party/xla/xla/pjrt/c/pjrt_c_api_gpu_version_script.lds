# The symbols listed in the "global" section of this file--and only those--
#    are discoverable by programs or frameworks that `dlopen()` libtpu.
# The linker will expose those symbols and no others to frameworks.
VERS_1.0 {
  global:
    extern "C" {
      GetPjrtApi;
    };

  local:
    *;
};
