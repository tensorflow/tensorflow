def __update_globals(pywrap_m):
  if hasattr(pywrap_m, '__all__'):
    all_names = pywrap_m.__all__
  else:
    all_names = [name for name in dir(pywrap_m) if not name.startswith('_')]

  extra_names = []  # template_val
  all_names.extend(extra_names)
  globals().update({name: getattr(pywrap_m, name) for name in all_names})


def __try_import():
  imports_paths = []  # template_val
  for import_path in imports_paths:
    try:
      pywrap_m = __import__(import_path, fromlist=["*"])
      __update_globals(pywrap_m)
      return
    except ImportError:
      # try another packge if there are any left
      pass

  raise RuntimeError(
    "Could not detect original test/binary location, import paths tried: %s" % imports_paths)

__try_import()
