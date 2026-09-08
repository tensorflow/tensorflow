# Copyright 2026 The OpenXLA Authors
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
"""Utilities for reading and parsing flags from LIBTPU_INIT_ARGS."""

from collections.abc import Callable
import os
import shlex
from typing import Any, TypeVar

from absl import flags
from absl import logging

T = TypeVar("T")


def _is_number(s: str) -> bool:
  """Checks if a string represents a valid number."""
  try:
    float(s)
    return True
  except ValueError:
    return False


def _parse_bool(val: str) -> bool:
  """Parses a string into a boolean."""
  val_lower = val.strip().lower()
  if val_lower in ("true", "1", "t", "yes", "y", ""):
    return True
  if val_lower in ("false", "0", "f", "no", "n"):
    return False
  raise ValueError(f"Cannot parse boolean value: {val!r}")


def _convert_value(val: str, flag_type: Callable[[str], T]) -> T | bool:
  """Converts a string flag value to the specified type."""
  if flag_type is bool:
    return _parse_bool(val)
  return flag_type(val)


def parse_libtpu_init_args(
    args_str: str | None = None,
    env_var: str = "LIBTPU_INIT_ARGS",
) -> dict[str, str]:
  """Parses flag arguments from a string or an environment variable.

  Tokens without a leading dash are ignored, unless they can be parsed as the
  value of the preceding flag.

  Args:
    args_str: Optional string containing flag arguments. If None, reads from the
      environment variable specified by `env_var`.
    env_var: Name of the environment variable to read from if `args_str` is
      None. Defaults to "LIBTPU_INIT_ARGS".

  Returns:
    A dictionary mapping flag names (without leading dashes) to their string
    values.
  """
  if args_str is None:
    args_str = os.environ.get(env_var, "")

  if not args_str or not args_str.strip():
    return {}

  try:
    tokens = shlex.split(args_str)
  except ValueError as e:
    logging.warning(
        "Failed to shlex.split flag string %r: %s. Falling back to split().",
        args_str,
        e,
    )
    tokens = args_str.split()

  parsed_flags: dict[str, str] = {}
  i = 0
  while i < len(tokens):
    token = tokens[i]
    if token.startswith("-"):
      flag_token = token.lstrip("-")
      if not flag_token:
        # Bare dash or double dash ('-' or '--')
        i += 1
        continue
      if "=" in flag_token:
        name, val = flag_token.split("=", 1)
        parsed_flags[name] = val
        i += 1
      elif i + 1 < len(tokens) and (
          not tokens[i + 1].startswith("-") or _is_number(tokens[i + 1])
      ):
        parsed_flags[flag_token] = tokens[i + 1]
        i += 2
      else:
        # Boolean or valueless flag
        parsed_flags[flag_token] = "true"
        i += 1
    else:
      # Token without leading dash that was not consumed as a value
      i += 1

  return parsed_flags


def get_flag_from_libtpu_init_args(
    flag_name: str,
    default: T = None,
    flag_type: Callable[[str], T] | None = None,
    args_str: str | None = None,
    env_var: str = "LIBTPU_INIT_ARGS",
) -> T | bool | str | None:
  """Retrieves a specific flag value from LIBTPU_INIT_ARGS.

  Args:
    flag_name: Name of the flag to look up (with or without leading dashes).
    default: Default value to return if the flag is not found.
    flag_type: Optional type or converter callable (e.g. int, bool, float).
    args_str: Optional string containing flag arguments. If None, reads from the
      environment variable specified by `env_var`.
    env_var: Name of the environment variable. Defaults to "LIBTPU_INIT_ARGS".

  Returns:
    The parsed flag value cast to `flag_type` (if given), or `default` if not
    found.
  """
  parsed = parse_libtpu_init_args(args_str=args_str, env_var=env_var)
  clean_name = flag_name.lstrip("-")
  norm_target = clean_name.replace("-", "_")

  raw_val: str | None = None
  if clean_name in parsed:
    raw_val = parsed[clean_name]
  elif norm_target in parsed:
    raw_val = parsed[norm_target]
  else:
    for k, v in parsed.items():
      if k.replace("-", "_") == norm_target:
        raw_val = v
        break

  if raw_val is None:
    return default

  if flag_type is not None:
    return _convert_value(raw_val, flag_type)
  return raw_val


def get_flag_value(
    flag_name: str,
    default: Any = None,
    flag_type: Callable[[str], Any] | None = None,
    flag_values: flags.FlagValues | None = None,
    env_var: str = "LIBTPU_INIT_ARGS",
) -> Any:
  """Gets a flag value from absl.flags.FLAGS or falls back to LIBTPU_INIT_ARGS.

  Args:
    flag_name: Name of the flag to look up (with or without leading dashes).
    default: Default value to return if flag is not found.
    flag_type: Optional type or callable to cast the value from the environment.
    flag_values: Optional absl.flags.FlagValues instance. Defaults to
      flags.FLAGS.
    env_var: Environment variable name to check as fallback. Defaults to
      "LIBTPU_INIT_ARGS".

  Returns:
    The flag value if found in flags or environment, else default.
  """
  clean_name = flag_name.lstrip("-")
  norm_name = clean_name.replace("-", "_")
  if flag_values is None:
    flag_values = flags.FLAGS

  if norm_name in flag_values:
    val = getattr(flag_values, norm_name)
    if val != -1 and val is not None:
      return val
  elif clean_name in flag_values:
    val = getattr(flag_values, clean_name)
    if val != -1 and val is not None:
      return val

  return get_flag_from_libtpu_init_args(
      flag_name=clean_name,
      default=default,
      flag_type=flag_type,
      env_var=env_var,
  )
