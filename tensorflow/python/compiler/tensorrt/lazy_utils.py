# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import functools

@functools.lru_cache(None)
def LazyObj(base_cls):
  if not isinstance(base_cls, type):
    raise ValueError

  class _LazyObj(object):
    __base_cls__ = base_cls

    __bypass_getattribute__ = [
        "wrapped", "_wrapped", "_setup_fn", "__maybe_load__", "__setattr__",
        "__base_cls__", "_debug"
    ]

    __bypass_setattr__ = ["_setup_fn", "_wrapped", "_debug"]

    def __init__(self, setup_fn=None, debug=False):
      if setup_fn is None or not callable(setup_fn):
        raise ValueError

      self._debug = debug
      self._setup_fn = setup_fn
      self._wrapped = None

    @property
    def wrapped(self):
      if self._wrapped is None:
        if self._debug:
          print("Loading Obj")
        obj = self._setup_fn()
        if isinstance(obj, _LazyObj.__base_cls__):
          self._wrapped = obj
        else:
          self._wrapped = _LazyObj.__base_cls__(obj)
      return self._wrapped

    def __getattribute__(self, name):
      if name not in _LazyObj.__bypass_getattribute__:
        wrapped_obj = self.wrapped
        return wrapped_obj.__getattribute__(wrapped_obj, name)
      else:
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
      if name not in _LazyObj.__bypass_setattr__:
        self.wrapped.__setattr__(name, value)
      else:
        super().__setattr__(name, value)

    __class__ = getattr(base_cls, "__class__")
    __name__ = getattr(base_cls, "__name__")

  def new_method_proxy(func_name):
    def inner(self, *args, **kwargs):
      args = [
          arg if not isinstance(arg, _LazyObj) else arg.wrapped
          for arg in args
      ]
      kwargs = {
          k: v if not isinstance(v, _LazyObj) else v.wrapped
          for k, v in kwargs.items()
      }
      return self.wrapped.__getattribute__(func_name)(*args, **kwargs)

    return inner

  protected_methods = [
      # Re-implemented Methods
      "__class__", "__getattribute__", "__init__", "__setattr__", "__dict__",
      # Forbidden methods
      "__init_subclass__", "__new__", "__subclasshook__",
      # MetaClasses Specific methods: Targeting EnumMeta namely
      "__abstractmethods__", "__base__", "__bases__", "__basicsize__",
      "__dictoffset__", "__flags__", "__itemsize__", "__mro__", "__name__",
      "__qualname__", "__text_signature__", "__weakrefoffset__"
  ]

  for key in dir(base_cls):
    if key in protected_methods:
      continue

    fn = new_method_proxy(key)
    if not callable(getattr(base_cls, key)):
      fn = property(fn)

    setattr(_LazyObj, key, fn)

  setattr(_LazyObj, "__name__", getattr(base_cls, "__name__"))
  return _LazyObj
