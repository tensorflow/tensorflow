from sys import modules
from types import ModuleType


def __update_globals(new_import_path, pywrap_m):
  all_names = pywrap_m.__all__ if hasattr(pywrap_m, '__all__') else dir(
      pywrap_m)
  modules[new_import_path] = pywrap_m
  for name in all_names:
    sub_pywrap = getattr(pywrap_m, name)
    if isinstance(sub_pywrap, ModuleType):
      sub_name = sub_pywrap.__name__[len(pywrap_m.__name__):]
      __update_globals(new_import_path + sub_name, sub_pywrap)


def __try_import():
  imports_paths = []  # template_val
  exceptions = []
  last_exception = None
  for import_path in imports_paths:
    try:
      pywrap_m = __import__(import_path, fromlist=["*"])
      __update_globals(__name__, pywrap_m)
      return
    except ImportError as e:
      exceptions.append(str(e))
      last_exception = e
      pass

  raise RuntimeError(f"""
Could not import original test/binary location, import paths tried: {imports_paths}. 
Previous exceptions: {exceptions}""", last_exception)


__try_import()
