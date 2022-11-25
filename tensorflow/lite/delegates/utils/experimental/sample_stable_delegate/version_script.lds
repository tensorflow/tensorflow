VERS_1.0 {
  # Export the stable delegate API.
  global:
    # Export stable delegate symbol.
    extern "C" {
      TFL_TheStableDelegate;
    };

    # Export operator new/delete.  These are defined as weak symbols and we need to keep these
    # symbols to ensure that if the main binary overrides these, we use the same version in the
    # stable delegate, since data that is allocated by stable delegate may be deallocated elsewhere.
    extern "C++" {
      # The syntax here is a bit awkward.  Here we want to match against a demangled symbol name
      # that contains spaces, but we also want to use glob-style pattern matching.
      # The linker script syntax doesn't allow spaces in symbol names unless quoted;
      # but if you do quote the symbol name, then it doesn't do glob-style pattern matching.
      # So we can't use quotes.  Instead we just use the wildcard character "?" to match space,
      # which is not ideal, since it would also match any other character.  But these patterns
      # are nevertheless sufficiently unique that they are unlikely to match any symbols other
      # than the overloaded global ::operator new and ::operator delete functions.
      operator?new*;
      operator?delete*;

      # Currently the demangler doesn't output the leading "::" for global symbols.
      # But just in case a future symbol demangler outputs the leading "::",
      # we also export those names too...
      ::operator?new*;
      ::operator?delete*;
    };

  # Hide everything else.
  local:
    *;
};
