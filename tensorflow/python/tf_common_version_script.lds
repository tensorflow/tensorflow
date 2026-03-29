VERS_1.0 {
  global:
    Wrapped_PyInit_*;
/*     the remaining global symbols are "only" needed to support custom op registration */
    extern "C++" {
      tensorflow::*;
      typeinfo?for?tensorflow::*;
      vtable?for?tensorflow::*;
    };
  local:
      *;
};
