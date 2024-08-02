import os
import re


def __calc_import_path():
  module_name = os.path.basename(__file__)[:-3]
  outer_module_name = "" # template_val
  for var in ["PYWRAP_TARGET", "TEST_TARGET"]:
    path = __find_pywrap_module_by_target_label(os.environ.get(var))
    if path:
      return "%s.%s%s" % (path, outer_module_name, module_name)

  for var in ["RUNFILES_MANIFEST_FILE", "RUNFILES_DIR"]:
    path = __find_pywrap_module_by_runfiles_env(os.environ.get(var))
    if path:
      return "%s.%s%s" % (path, outer_module_name, module_name)

  raise RuntimeError("Could not detect original test/binary location")


def __find_pywrap_module_by_target_label(target_label):
  if target_label:
    return target_label.split("//", 1)[1].split(":")[0].replace("/", ".")
  return None


def __find_pywrap_module_by_runfiles_env(runfiles_env_var):
  pattern = re.compile(
      r"bazel-out/.*/bin/(?P<pkg>[\w/]*)/(?P<binary>\w+)(\.exe)?\.runfiles"
  )
  if runfiles_env_var:
    match = pattern.search(runfiles_env_var)
    return match.group("pkg").replace("/", ".")
  return None


def __update_globals(pywrap_m):
  if hasattr(pywrap_m, '__all__'):
    all_names = pywrap_m.__all__
  else:
    all_names = [name for name in dir(pywrap_m) if not name.startswith('_')]

  extra_names = [] # template_val
  all_names.extend(extra_names)
  globals().update({name: getattr(pywrap_m, name) for name in all_names})


__pywrap_m = __import__(__calc_import_path(), fromlist=["*"])
__update_globals(__pywrap_m)
