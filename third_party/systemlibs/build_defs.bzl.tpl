# -*- Python -*-
"""Skylark macros for system libraries.
"""

SYSTEM_LIBS_ENABLED = %{syslibs_enabled}

SYSTEM_LIBS_LIST = [
%{syslibs_list}
]


def if_any_system_libs(a, b=[]):
  """Conditional which evaluates to 'a' if any system libraries are configured."""
  if SYSTEM_LIBS_ENABLED:
    return a
  else:
    return b


def if_system_lib(lib, a, b=[]):
  """Conditional which evaluates to 'a' if we're using the system version of lib"""

  if SYSTEM_LIBS_ENABLED and lib in SYSTEM_LIBS_LIST:
    return a
  else:
    return b


def if_not_system_lib(lib, a, b=[]):
  """Conditional which evaluates to 'a' if we're using the system version of lib"""

  return if_system_lib(lib, b, a)
