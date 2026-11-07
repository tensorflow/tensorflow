# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Helper utilities and configuration classes for TensorFlow configure.py."""

import os
import platform
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional


def is_windows() -> bool:
  """Returns True if the host operating system is Windows."""
  return platform.system() == 'Windows'


def is_cygwin() -> bool:
  """Returns True if running under Cygwin."""
  return platform.system().startswith('CYGWIN_NT')


def split_path_list(path_str: Optional[str]) -> List[str]:
  """Safely split PATH or PYTHONPATH across platforms.

  On Windows, paths contain ':' as drive letters (e.g. 'C:\\path1;D:\\path2')
  and elements are separated by ';'. On POSIX, elements are separated by ':'.
  Splitting by ':' on Windows corrupts drive letters. This helper correctly
  identifies the path separator.

  Args:
    path_str: Delimited path string from environment (e.g. PYTHONPATH).

  Returns:
    List of non-empty, stripped directory paths.
  """
  if not path_str:
    return []
  path_str = path_str.strip()
  if not path_str:
    return []

  # If os.pathsep (';' on Windows, ':' on POSIX) is present, use it.
  if os.pathsep in path_str:
    raw_parts = path_str.split(os.pathsep)
  elif is_windows() or is_cygwin():
    # Single Windows path like 'C:\Users\...' without delimiter
    raw_parts = [path_str]
  else:
    # POSIX single or colon-delimited path
    raw_parts = path_str.split(':')

  return [p.strip() for p in raw_parts if p.strip()]


def safe_decode(output: Any, encoding: str = 'utf-8') -> str:
  """Safely decode bytes or strings to a clean string with error replacement.

  Args:
    output: Raw bytes or string from subprocess output.
    encoding: Target character encoding (default utf-8).

  Returns:
    Decoded and stripped string.
  """
  if output is None:
    return ''
  if isinstance(output, str):
    return output.strip()
  return output.decode(encoding, errors='replace').strip()


class BazelrcConfig:
  """Manages entries in the generated .tf_configure.bazelrc file."""

  def __init__(self, filepath: Optional[str] = None):
    """Initializes BazelrcConfig.

    Args:
      filepath: Optional path to the .tf_configure.bazelrc file.
    """
    self.filepath = filepath
    self.lines: List[str] = []

  def set_filepath(self, filepath: str) -> None:
    """Updates the destination filepath."""
    self.filepath = filepath

  def add_line(self, line: str) -> None:
    """Adds a raw line to the bazelrc configuration."""
    self.lines.append(line)
    if self.filepath:
      with open(self.filepath, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

  def add_action_env(self, var_name: str, var_value: Any) -> None:
    """Adds a build --action_env VAR="VALUE" entry."""
    self.add_line(f'build --action_env {var_name}="{var_value}"')

  def add_repo_env(self, config_name: str, var_name: str, var_value: Any) -> None:
    """Adds a build:config --repo_env VAR="VALUE" entry."""
    self.add_line(f'build:{config_name} --repo_env {var_name}="{var_value}"')

  def reset(self) -> None:
    """Clears buffered lines and truncates destination file if set."""
    self.lines.clear()
    if self.filepath:
      parent = os.path.dirname(self.filepath)
      if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
      with open(self.filepath, 'w', encoding='utf-8') as f:
        pass

  def to_string(self) -> str:
    """Returns all configuration lines as a single string."""
    return '\n'.join(self.lines)


class PromptManager:
  """Handles interactive and non-interactive user prompts."""

  def __init__(
      self,
      non_interactive: bool = False,
      input_fn: Optional[Callable[[str], str]] = None,
      output_fn: Optional[Callable[..., None]] = None,
  ):
    """Initializes PromptManager.

    Args:
      non_interactive: If True, prompt queries immediately return empty string
        to allow default values to be automatically used.
      input_fn: Callable for reading user input (defaults to builtin input).
      output_fn: Callable for writing messages (defaults to builtin print).
    """
    self.non_interactive = non_interactive
    self.input_fn = input_fn or input
    self.output_fn = output_fn or print

  def get_input(self, question: str) -> str:
    """Prompts user or returns empty string if in non-interactive mode.

    Args:
      question: Query text to display to user.

    Returns:
      User input response string, or empty string.
    """
    if self.non_interactive:
      return ''
    try:
      return self.input_fn(question)
    except (EOFError, KeyboardInterrupt):
      return ''


def apply_cli_overrides(args: Any, environ_cp: Dict[str, str]) -> None:
  """Applies CLI flag overrides to the environment configuration dictionary.

  Args:
    args: Parsed argparse namespace.
    environ_cp: Mutable dictionary representing environment variables.
  """
  if getattr(args, 'python_bin_path', None):
    environ_cp['PYTHON_BIN_PATH'] = args.python_bin_path
  if getattr(args, 'python_lib_path', None):
    environ_cp['PYTHON_LIB_PATH'] = args.python_lib_path
  if getattr(args, 'cuda', None) is not None:
    environ_cp['TF_NEED_CUDA'] = '1' if args.cuda else '0'
  if getattr(args, 'rocm', None) is not None:
    environ_cp['TF_NEED_ROCM'] = '1' if args.rocm else '0'
  if getattr(args, 'clang', None) is not None:
    environ_cp['TF_NEED_CLANG'] = '1' if args.clang else '0'
  if getattr(args, 'xla', None) is not None:
    environ_cp['TF_ENABLE_XLA'] = '1' if args.xla else '0'
