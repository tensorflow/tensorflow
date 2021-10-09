# Lint as: python2, python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
#
# ==============================================================================
"""Checks if a set of configuration(s) is version and dependency compatible."""

import re
import sys

import six
from six.moves import range
import six.moves.configparser
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_inspect

PATH_TO_DIR = "tensorflow/tools/tensorflow_builder/compat_checker"


def _compare_versions(v1, v2):
  """Compare two versions and return information on which is smaller vs. larger.

  Args:
    v1: String that is a version to be compared against `v2`.
    v2: String that is a version to be compared against `v1`.

  Returns:
    Dict that stores larger version with key `larger` and smaller version with
      key `smaller`.
      e.g. {`larger`: `1.5.0`, `smaller`: `1.2.0`}

  Raises:
    RuntimeError: If asked to compare `inf` to `inf`.
  """
  # Throw error is asked to compare `inf` to `inf`.
  if v1 == "inf" and v2 == "inf":
    raise RuntimeError("Cannot compare `inf` to `inf`.")

  rtn_dict = {"smaller": None, "larger": None}
  v1_list = six.ensure_str(v1).split(".")
  v2_list = six.ensure_str(v2).split(".")
  # Take care of cases with infinity (arg=`inf`).
  if v1_list[0] == "inf":
    v1_list[0] = str(int(v2_list[0]) + 1)
  if v2_list[0] == "inf":
    v2_list[0] = str(int(v1_list[0]) + 1)

  # Determine which of the two lists are longer vs. shorter.
  v_long = v1_list if len(v1_list) >= len(v2_list) else v2_list
  v_short = v1_list if len(v1_list) < len(v2_list) else v2_list

  larger, smaller = None, None
  for i, ver in enumerate(v_short, start=0):
    if int(ver) > int(v_long[i]):
      larger = _list_to_string(v_short, ".")
      smaller = _list_to_string(v_long, ".")
    elif int(ver) < int(v_long[i]):
      larger = _list_to_string(v_long, ".")
      smaller = _list_to_string(v_short, ".")
    else:
      if i == len(v_short) - 1:
        if v_long[i + 1:] == ["0"]*(len(v_long) - 1 - i):
          larger = "equal"
          smaller = "equal"
        else:
          larger = _list_to_string(v_long, ".")
          smaller = _list_to_string(v_short, ".")
      else:
        # Go to next round.
        pass

    if larger:
      break

  rtn_dict["smaller"] = smaller
  rtn_dict["larger"] = larger

  return rtn_dict


def _list_to_string(l, s):
  """Concatenates list items into a single string separated by `s`.

  Args:
    l: List with items to be concatenated into a single string.
    s: String or char that will be concatenated in between each item.

  Returns:
    String that has all items in list `l` concatenated with `s` separator.
  """

  return s.join(l)


def _get_func_name():
  """Get the name of current function.

  Returns:
    String that is the name of current function.
  """
  return tf_inspect.stack()[1][3]


class ConfigCompatChecker(object):
  """Class that checks configuration versions and dependency compatibilities.

  `ConfigCompatChecker` checks a given set of configurations and their versions
  against supported versions and dependency rules defined in `.ini` config file.
  For project `TensorFlow Builder`, it functions as a sub-module for the builder
  service that validates requested build configurations from a client prior to
  initiating a TensorFlow build.
  """

  class _Reqs(object):
    """Class that stores specifications related to a single requirement.

    `_Reqs` represents a single version or dependency requirement specified in
    the `.ini` config file. It is meant ot be used inside `ConfigCompatChecker`
    to help organize and identify version and dependency compatibility for a
    given configuration (e.g. gcc version) required by the client.
    """

    def __init__(self, req, config, section):
      """Initializes a version or dependency requirement object.

      Args:
        req: List that contains individual supported versions or a single string
             that contains `range` definition.
               e.g. [`range(1.0, 2.0) include(3.0) exclude(1.5)`]
               e.g. [`1.0`, `3.0`, `7.1`]
        config: String that is the configuration name.
                  e.g. `platform`
        section: String that is the section name from the `.ini` config file
                 under which the requirement is defined.
                   e.g. `Required`, `Optional`, `Unsupported`, `Dependency`
      """
      # Req class variables.
      self.req = req
      self.exclude = None
      self.include = None
      self.range = [None, None]  # for [min, max]
      self.config = config
      self._req_type = ""  # e.g. `range` or `no_range`
      self._section = section
      self._initialized = None
      self._error_message = []

      # Parse and store requirement specifications.
      self.parse_single_req()

    @property
    def get_status(self):
      """Get status of `_Reqs` initialization.

      Returns:
        Tuple
          (Boolean indicating initialization status,
           List of error messages, if any)

      """

      return self._initialized, self._error_message

    def __str__(self):
      """Prints a requirement and its components.

      Returns:
        String that has concatenated information about a requirement.
      """
      info = {
          "section": self._section,
          "config": self.config,
          "req_type": self._req_type,
          "req": str(self.req),
          "range": str(self.range),
          "exclude": str(self.exclude),
          "include": str(self.include),
          "init": str(self._initialized)
      }
      req_str = "\n >>> _Reqs Instance <<<\n"
      req_str += "Section: {section}\n"
      req_str += "Configuration name: {config}\n"
      req_str += "Requirement type: {req_type}\n"
      req_str += "Requirement: {req}\n"
      req_str += "Range: {range}\n"
      req_str += "Exclude: {exclude}\n"
      req_str += "Include: {include}\n"
      req_str += "Initialized: {init}\n\n"

      return req_str.format(**info)

    def parse_single_req(self):
      """Parses a requirement and stores information.

      `self.req` _initialized in `__init__` is called for retrieving the
      requirement.

      A requirement can come in two forms:
        [1] String that includes `range` indicating range syntax for defining
            a requirement.
              e.g. `range(1.0, 2.0) include(3.0) exclude(1.5)`
        [2] List that includes individual supported versions or items.
              e.g. [`1.0`, `3.0`, `7.1`]

      For a list type requirement, it directly stores the list to
      `self.include`.

      Call `get_status` for checking the status of the parsing. This function
      sets `self._initialized` to `False` and immediately returns with an error
      message upon encountering a failure. It sets `self._initialized` to `True`
      and returns without an error message upon success.
      """
      # Regex expression for filtering requirement line. Please refer
      # to docstring above for more information.
      expr = r"(range\()?([\d\.\,\s]+)(\))?( )?(include\()?"
      expr += r"([\d\.\,\s]+)?(\))?( )?(exclude\()?([\d\.\,\s]+)?(\))?"

      # Check that arg `req` is not empty.
      if not self.req:
        err_msg = "[Error] Requirement is missing. "
        err_msg += "(section = %s, " % str(self._section)
        err_msg += "config = %s, req = %s)" % (str(self.config), str(self.req))
        logging.error(err_msg)
        self._initialized = False
        self._error_message.append(err_msg)

        return

      # For requirement given in format with `range`. For example:
      # python = [range(3.3, 3.7) include(2.7)] as opposed to
      # python = [2.7, 3.3, 3.4, 3.5, 3.6, 3.7]
      if "range" in self.req[0]:
        self._req_type = "range"
        match = re.match(expr, self.req[0])
        if not match:
          err_msg = "[Error] Encountered issue when parsing the requirement."
          err_msg += " (req = %s, match = %s)" % (str(self.req), str(match))
          logging.error(err_msg)
          self._initialized = False
          self._error_message.append(err_msg)

          return
        else:
          match_grp = match.groups()
          match_size = len(match_grp)
          for i, m in enumerate(match_grp[0:match_size-1], start=0):
            # Get next index. For example:
            # |    idx     |  next_idx  |
            # +------------+------------+
            # |  `range(`  | `1.1, 1.5` |
            # | `exclude(` | `1.1, 1.5` |
            # | `include(` | `1.1, 1.5` |
            next_match = match_grp[i + 1]

            if m not in ["", None, " ", ")"]:
              if "range" in m:
                # Check that the range definition contains only one comma.
                # If more than one comma, then there is format error with the
                # requirement config file.
                comma_count = next_match.count(",")
                if comma_count > 1 or comma_count == 0:
                  err_msg = "[Error] Found zero or more than one comma in range"
                  err_msg += " definition. (req = %s, " % str(self.req)
                  err_msg += "match = %s)" % str(next_match)
                  logging.error(err_msg)
                  self._initialized = False
                  self._error_message.append(err_msg)

                  return

                # Remove empty space in range and separate min, max by
                # comma. (e.g. `1.0, 2.0` => `1.0,2.0` => [`1.0`, `2.0`])
                min_max = next_match.replace(" ", "").split(",")

                # Explicitly define min and max values.
                # If min_max = ['', ''], then `range(, )` was provided as
                # req, which is equivalent to `include all versions`.
                if not min_max[0]:
                  min_max[0] = "0"

                if not min_max[1]:
                  min_max[1] = "inf"

                self.range = min_max
              if "exclude" in m:
                self.exclude = next_match.replace(" ", "").split(",")

              if "include" in m:
                self.include = next_match.replace(" ", "").split(",")

              self._initialized = True

      # For requirement given in format without a `range`. For example:
      # python = [2.7, 3.3, 3.4, 3.5, 3.6, 3.7] as opposed to
      # python = [range(3.3, 3.7) include(2.7)]
      else:
        self._req_type = "no_range"
        # Requirement (self.req) should be a list.
        if not isinstance(self.req, list):
          err_msg = "[Error] Requirement is not a list."
          err_msg += "(req = %s, " % str(self.req)
          err_msg += "type(req) = %s)" % str(type(self.req))
          logging.error(err_msg)
          self._initialized = False
          self._error_message.append(err_msg)
        else:
          self.include = self.req
          self._initialized = True

      return

  def __init__(self, usr_config, req_file):
    """Initializes a configuration compatibility checker.

    Args:
      usr_config: Dict of all configuration(s) whose version compatibilities are
                  to be checked against the rules defined in the `.ini` config
                  file.
      req_file: String that is the full name of the `.ini` config file.
                  e.g. `config.ini`
    """
    # ConfigCompatChecker class variables.
    self.usr_config = usr_config
    self.req_file = req_file
    self.warning_msg = []
    self.error_msg = []
    # Get and store requirements.
    reqs_all = self.get_all_reqs()
    self.required = reqs_all["required"]
    self.optional = reqs_all["optional"]
    self.unsupported = reqs_all["unsupported"]
    self.dependency = reqs_all["dependency"]

    self.successes = []
    self.failures = []

  def get_all_reqs(self):
    """Parses all compatibility specifications listed in the `.ini` config file.

    Reads and parses each and all compatibility specifications from the `.ini`
    config file by sections. It then populates appropriate dicts that represent
    each section (e.g. `self.required`) and returns a tuple of the populated
    dicts.

    Returns:
      Dict of dict
        { `required`: Dict of `Required` configs and supported versions,
          `optional`: Dict of `Optional` configs and supported versions,
          `unsupported`: Dict of `Unsupported` configs and supported versions,
          `dependency`: Dict of `Dependency` configs and supported versions }
    """
    # First check if file exists. Exit on failure.
    try:
      open(self.req_file, "rb")
    except IOError:
      msg = "[Error] Cannot read file '%s'." % self.req_file
      logging.error(msg)
      sys.exit(1)

    # Store status of parsing requirements. For local usage only.
    curr_status = True

    # Initialize config parser for parsing version requirements file.
    parser = six.moves.configparser.ConfigParser()
    parser.read(self.req_file)

    if not parser.sections():
      err_msg = "[Error] Empty config file. "
      err_msg += "(file = %s, " % str(self.req_file)
      err_msg += "parser sectons = %s)" % str(parser.sections())
      self.error_msg.append(err_msg)
      logging.error(err_msg)
      curr_status = False

    # Each dependency dict will have the following format.
    # _dict = {
    #   `<config_name>` : [_Reqs()],
    #   `<config_name>` : [_Reqs()]
    # }
    required_dict = {}
    optional_dict = {}
    unsupported_dict = {}
    dependency_dict = {}

    # Parse every config under each section defined in config file
    # and populate requirement dict(s).
    for section in parser.sections():
      all_configs = parser.options(section)
      for config in all_configs:
        spec = parser.get(section, config)
        # Separately manage each section:
        #   `Required`,
        #   `Optional`,
        #   `Unsupported`,
        #   `Dependency`
        # One of the sections is required.
        if section == "Dependency":
          dependency_dict[config] = []
          spec_split = spec.split(",\n")
          # First dependency item may only or not have `[` depending
          # on the indentation style in the config (.ini) file.
          # If it has `[`, then either skip or remove from string.
          if spec_split[0] == "[":
            spec_split = spec_split[1:]
          elif "[" in spec_split[0]:
            spec_split[0] = spec_split[0].replace("[", "")
          else:
            warn_msg = "[Warning] Config file format error: Missing `[`."
            warn_msg += "(section = %s, " % str(section)
            warn_msg += "config = %s)" % str(config)
            logging.warning(warn_msg)
            self.warning_msg.append(warn_msg)

          # Last dependency item may only or not have `]` depending
          # on the indentation style in the config (.ini) file.
          # If it has `[`, then either skip or remove from string.
          if spec_split[-1] == "]":
            spec_split = spec_split[:-1]
          elif "]" in spec_split[-1]:
            spec_split[-1] = spec_split[-1].replace("]", "")
          else:
            warn_msg = "[Warning] Config file format error: Missing `]`."
            warn_msg += "(section = %s, " % str(section)
            warn_msg += "config = %s)" % str(config)
            logging.warning(warn_msg)
            self.warning_msg.append(warn_msg)

          # Parse `spec_split` which is a list of all dependency rules
          # retrieved from the config file.
          # Create a _Reqs() instance for each rule and store it under
          # appropriate class dict (e.g. dependency_dict) with a proper
          # key.
          #
          # For dependency definition, it creates one _Reqs() instance each
          # for requirement and dependency. For example, it would create
          # a list in the following indexing sequence:
          #
          # [`config', <`config` _Reqs()>, `dep', <`dep` _Reqs()>]
          #
          # For example:
          # [`python`, _Reqs(), `tensorflow`, _Reqs()] for
          # `python 3.7 requires tensorflow 1.13`
          for rule in spec_split:
            # Filter out only the necessary information from `rule` string.
            spec_dict = self.filter_dependency(rule)
            # Create _Reqs() instance for each rule.
            cfg_name = spec_dict["cfg"]  # config name
            dep_name = spec_dict["cfgd"]  # dependency name
            cfg_req = self._Reqs(
                self.convert_to_list(spec_dict["cfg_spec"], " "),
                config=cfg_name,
                section=section
            )
            dep_req = self._Reqs(
                self.convert_to_list(spec_dict["cfgd_spec"], " "),
                config=dep_name,
                section=section
            )
            # Check status of _Reqs() initialization. If wrong formats are
            # detected from the config file, it would return `False` for
            # initialization status.
            # `<_Reqs>.get_status` returns [_initialized, _error_message]
            cfg_req_status = cfg_req.get_status
            dep_req_status = dep_req.get_status
            if not cfg_req_status[0] or not dep_req_status[0]:
              # `<_Reqs>.get_status()[1]` returns empty upon successful init.
              msg = "[Error] Failed to create _Reqs() instance for a "
              msg += "dependency item. (config = %s, " % str(cfg_name)
              msg += "dep = %s)" % str(dep_name)
              logging.error(msg)
              self.error_msg.append(cfg_req_status[1])
              self.error_msg.append(dep_req_status[1])
              curr_status = False
              break
            else:
              dependency_dict[config].append(
                  [cfg_name, cfg_req, dep_name, dep_req])

          # Break out of `if section == 'Dependency'` block.
          if not curr_status:
            break

        else:
          if section == "Required":
            add_to = required_dict
          elif section == "Optional":
            add_to = optional_dict
          elif section == "Unsupported":
            add_to = unsupported_dict
          else:
            msg = "[Error] Section name `%s` is not accepted." % str(section)
            msg += "Accepted section names are `Required`, `Optional`, "
            msg += "`Unsupported`, and `Dependency`."
            logging.error(msg)
            self.error_msg.append(msg)
            curr_status = False
            break

          # Need to make sure `req` argument for _Reqs() instance is always
          # a list. If not, convert to list.
          req_list = self.convert_to_list(self.filter_line(spec), " ")
          add_to[config] = self._Reqs(req_list, config=config, section=section)
        # Break out of `for config in all_configs` loop.
        if not curr_status:
          break

      # Break out of `for section in parser.sections()` loop.
      if not curr_status:
        break

    return_dict = {
        "required": required_dict,
        "optional": optional_dict,
        "unsupported": unsupported_dict,
        "dependency": dependency_dict
    }

    return return_dict

  def filter_dependency(self, line):
    """Filters dependency compatibility rules defined in the `.ini` config file.

    Dependency specifications are defined as the following:
      `<config> <config_version> requires <dependency> <dependency_version>`
    e.g.
      `python 3.7 requires tensorflow 1.13`
      `tensorflow range(1.0.0, 1.13.1) requires gcc range(4.8, )`

    Args:
      line: String that is a dependency specification defined under `Dependency`
            section in the `.ini` config file.

    Returns:
      Dict with configuration and its dependency information.
        e.g. {`cfg`: `python`,       # configuration name
              `cfg_spec`: `3.7`,     # configuration version
              `cfgd`: `tensorflow`,  # dependency name
              `cfgd_spec`: `4.8`}    # dependency version
    """
    line = line.strip("\n")
    expr = r"(?P<cfg>[\S]+) (?P<cfg_spec>range\([\d\.\,\s]+\)( )?"
    expr += r"(include\([\d\.\,\s]+\))?( )?(exclude\([\d\.\,\s]+\))?( )?"
    expr += r"|[\d\,\.\s]+) requires (?P<cfgd>[\S]+) (?P<cfgd_spec>range"
    expr += r"\([\d\.\,\s]+\)( )?(include\([\d\.\,\s]+\))?( )?"
    expr += r"(exclude\([\d\.\,\s]+\))?( )?|[\d\,\.\s]+)"
    r = re.match(expr, line.strip("\n"))

    return r.groupdict()

  def convert_to_list(self, item, separator):
    """Converts a string into a list with a separator.

    Args:
      item: String that needs to be separated into a list by a given separator.
            List item is also accepted but will take no effect.
      separator: String with which the `item` will be splited.

    Returns:
      List that is a splited version of a given input string.
        e.g. Input: `1.0, 2.0, 3.0` with `, ` separator
             Output: [1.0, 2.0, 3.0]
    """
    out = None
    if not isinstance(item, list):
      if "range" in item:
        # If arg `item` is a single string, then create a list with just
        # the item.
        out = [item]
      else:
        # arg `item` can come in as the following:
        # `1.0, 1.1, 1.2, 1.4`
        # if requirements were defined without the `range()` format.
        # In such a case, create a list separated by `separator` which is
        # an empty string (' ') in this case.
        out = item.split(separator)
        for i in range(len(out)):
          out[i] = out[i].replace(",", "")

    # arg `item` is a list already.
    else:
      out = [item]

    return out

  def filter_line(self, line):
    """Removes `[` or `]` from the input line.

    Args:
      line: String that is a compatibility specification line from the `.ini`
            config file.

    Returns:
      String that is a compatibility specification line without `[` and `]`.
    """
    filtered = []
    warn_msg = []

    splited = line.split("\n")

    # If arg `line` is empty, then requirement might be missing. Add
    # to warning as this issue will be caught in _Reqs() initialization.
    if not line and len(splited) < 1:
      warn_msg = "[Warning] Empty line detected while filtering lines."
      logging.warning(warn_msg)
      self.warning_msg.append(warn_msg)

    # In general, first line in requirement definition will include `[`
    # in the config file (.ini). Remove it.
    if splited[0] == "[":
      filtered = splited[1:]
    elif "[" in splited[0]:
      splited = splited[0].replace("[", "")
      filtered = splited
    # If `[` is missing, then it could be a formatting issue with
    # config file (.ini.). Add to warning.
    else:
      warn_msg = "[Warning] Format error. `[` could be missing in "
      warn_msg += "the config (.ini) file. (line = %s)" % str(line)
      logging.warning(warn_msg)
      self.warning_msg.append(warn_msg)

    # In general, last line in requirement definition will include `]`
    # in the config file (.ini). Remove it.
    if filtered[-1] == "]":
      filtered = filtered[:-1]
    elif "]" in filtered[-1]:
      filtered[-1] = six.ensure_str(filtered[-1]).replace("]", "")
    # If `]` is missing, then it could be a formatting issue with
    # config file (.ini.). Add to warning.
    else:
      warn_msg = "[Warning] Format error. `]` could be missing in "
      warn_msg += "the config (.ini) file. (line = %s)" % str(line)
      logging.warning(warn_msg)
      self.warning_msg.append(warn_msg)

    return filtered

  def in_range(self, ver, req):
    """Checks if a version satisfies a version and/or compatibility requirement.

    Args:
      ver: List whose first item is a config version that needs to be checked
           for support status and version compatibility.
             e.g. ver = [`1.0`]
      req: `_Reqs` class instance that represents a configuration version and
            compatibility specifications.

    Returns:
      Boolean output of checking if version `ver` meets the requirement
        stored in `req` (or a `_Reqs` requirements class instance).
    """
    # If `req.exclude` is not empty and `ver` is in `req.exclude`,
    # no need to proceed to next set of checks as it is explicitly
    # NOT supported.
    if req.exclude is not None:
      for v in ver:
        if v in req.exclude:
          return False

    # If `req.include` is not empty and `ver` is in `req.include`,
    # no need to proceed to next set of checks as it is supported and
    # NOT unsupported (`req.exclude`).
    include_checked = False
    if req.include is not None:
      for v in ver:
        if v in req.include:
          return True

      include_checked = True

    # If `req.range` is not empty, then `ver` is defined with a `range`
    # syntax. Check whether `ver` falls under the defined supported
    # range.
    if req.range != [None, None]:
      min_v = req.range[0]  # minimum supported version
      max_v = req.range[1]  # maximum supported version
      ver = ver[0]  # version to compare
      lg = _compare_versions(min_v, ver)["larger"]  # `ver` should be larger
      sm = _compare_versions(ver, max_v)["smaller"]  # `ver` should be smaller
      if lg in [ver, "equal"] and sm in [ver, "equal", "inf"]:
        return True
      else:
        err_msg = "[Error] Version is outside of supported range. "
        err_msg += "(config = %s, " % str(req.config)
        err_msg += "version = %s, " % str(ver)
        err_msg += "supported range = %s)" % str(req.range)
        logging.warning(err_msg)
        self.warning_msg.append(err_msg)
        return False

    else:
      err_msg = ""
      if include_checked:
        # user config is not supported as per exclude, include, range
        # specification.
        err_msg = "[Error] Version is outside of supported range. "
      else:
        # user config is not defined in exclude, include or range. config file
        # error.
        err_msg = "[Error] Missing specification. "

      err_msg += "(config = %s, " % str(req.config)
      err_msg += "version = %s, " % str(ver)
      err_msg += "supported range = %s)" % str(req.range)
      logging.warning(err_msg)
      self.warning_msg.append(err_msg)
      return False

  def _print(self, *args):
    """Prints compatibility check status and failure or warning messages.

    Prints to console without using `logging`.

    Args:
      *args: String(s) that is one of:
              [`failures`,       # all failures
               `successes`,      # all successes
               `failure_msgs`,   # failure message(s) recorded upon failure(s)
               `warning_msgs`]   # warning message(s) recorded upon warning(s)
    Raises:
      Exception: If *args not in:
                   [`failures`, `successes`, `failure_msgs`, `warning_msg`]
    """

    def _format(name, arr):
      """Prints compatibility check results with a format.

      Args:
        name: String that is the title representing list `arr`.
        arr: List of items to be printed in a certain format.
      """
      title = "### All Compatibility %s ###" % str(name)
      tlen = len(title)
      print("-"*tlen)
      print(title)
      print("-"*tlen)
      print(" Total # of %s: %s\n" % (str(name), str(len(arr))))
      if arr:
        for item in arr:
          detail = ""
          if isinstance(item[1], list):
            for itm in item[1]:
              detail += str(itm) + ", "
            detail = detail[:-2]
          else:
            detail = str(item[1])
          print("  %s ('%s')\n" % (str(item[0]), detail))
      else:
        print("  No %s" % name)
      print("\n")

    for p_item in args:
      if p_item == "failures":
        _format("Failures", self.failures)
      elif p_item == "successes":
        _format("Successes", self.successes)
      elif p_item == "failure_msgs":
        _format("Failure Messages", self.error_msg)
      elif p_item == "warning_msgs":
        _format("Warning Messages", self.warning_msg)
      else:
        raise Exception(
            "[Error] Wrong input provided for %s." % _get_func_name())

  def check_compatibility(self):
    """Checks version and dependency compatibility for a given configuration.

    `check_compatibility` immediately returns with `False` (or failure status)
    if any child process or checks fail. For error and warning messages, either
    print `self.(error_msg|warning_msg)` or call `_print` function.

    Returns:
      Boolean that is a status of the compatibility check result.
    """
    # Check if all `Required` configs are found in user configs.
    usr_keys = list(self.usr_config.keys())

    for k in six.iterkeys(self.usr_config):
      if k not in usr_keys:
        err_msg = "[Error] Required config not found in user config."
        err_msg += "(required = %s, " % str(k)
        err_msg += "user configs = %s)" % str(usr_keys)
        logging.error(err_msg)
        self.error_msg.append(err_msg)
        self.failures.append([k, err_msg])
        return False

    # Parse each user config and validate its compatibility.
    overall_status = True
    for config_name, spec in six.iteritems(self.usr_config):
      temp_status = True
      # Check under which section the user config is defined.
      in_required = config_name in list(self.required.keys())
      in_optional = config_name in list(self.optional.keys())
      in_unsupported = config_name in list(self.unsupported.keys())
      in_dependency = config_name in list(self.dependency.keys())

      # Add to warning if user config is not specified in the config file.
      if not (in_required or in_optional or in_unsupported or in_dependency):
        warn_msg = "[Error] User config not defined in config file."
        warn_msg += "(user config = %s)" % str(config_name)
        logging.warning(warn_msg)
        self.warning_msg.append(warn_msg)
        self.failures.append([config_name, warn_msg])
        temp_status = False
      else:
        if in_unsupported:
          if self.in_range(spec, self.unsupported[config_name]):
            err_msg = "[Error] User config is unsupported. It is "
            err_msg += "defined under 'Unsupported' section in the config file."
            err_msg += " (config = %s, spec = %s)" % (config_name, str(spec))
            logging.error(err_msg)
            self.error_msg.append(err_msg)
            self.failures.append([config_name, err_msg])
            temp_status = False

        if in_required:
          if not self.in_range(spec, self.required[config_name]):
            err_msg = "[Error] User config cannot be supported. It is not in "
            err_msg += "the supported range as defined in the 'Required' "
            err_msg += "section. (config = %s, " % config_name
            err_msg += "spec = %s)" % str(spec)
            logging.error(err_msg)
            self.error_msg.append(err_msg)
            self.failures.append([config_name, err_msg])
            temp_status = False

        if in_optional:
          if not self.in_range(spec, self.optional[config_name]):
            err_msg = "[Error] User config cannot be supported. It is not in "
            err_msg += "the supported range as defined in the 'Optional' "
            err_msg += "section. (config = %s, " % config_name
            err_msg += "spec = %s)" % str(spec)
            logging.error(err_msg)
            self.error_msg.append(err_msg)
            self.failures.append([config_name, err_msg])
            temp_status = False

        # If user config and version has a dependency, check both user
        # config + version and dependency config + version are supported.
        if in_dependency:
          # Get dependency information. The information gets retrieved in the
          # following format:
          #   [`config`, `config _Reqs()`, `dependency`, `dependency _Reqs()`]
          dep_list = self.dependency[config_name]
          if dep_list:
            for rule in dep_list:
              cfg = rule[0]  # config name
              cfg_req = rule[1]  # _Reqs() instance for config requirement
              dep = rule[2]  # dependency name
              dep_req = rule[3]  # _Reqs() instance for dependency requirement

              # Check if user config has a dependency in the following sequence:
              #   [1] Check user config and the config that has dependency
              #       are the same. (This is defined as `cfg_status`.)
              #   [2] Check if dependency is supported.
              try:
                cfg_name = self.usr_config[cfg]
                dep_name = self.usr_config[dep]

                cfg_status = self.in_range(cfg_name, cfg_req)
                dep_status = self.in_range(dep_name, dep_req)
                # If both status's are `True`, then user config meets dependency
                # spec.
                if cfg_status:
                  if not dep_status:
                    # throw error
                    err_msg = "[Error] User config has a dependency that cannot"
                    err_msg += " be supported. "
                    err_msg += "'%s' has a dependency on " % str(config_name)
                    err_msg += "'%s'." % str(dep)
                    logging.error(err_msg)
                    self.error_msg.append(err_msg)
                    self.failures.append([config_name, err_msg])
                    temp_status = False

              except KeyError:
                err_msg = "[Error] Dependency is missing from `Required`. "
                err_msg += "(config = %s, ""dep = %s)" % (cfg, dep)
                logging.error(err_msg)
                self.error_msg.append(err_msg)
                self.failures.append([config_name, err_msg])
                temp_status = False

      # At this point, all requirement related to the user config has been
      # checked and passed. Append to `successes` list.
      if temp_status:
        self.successes.append([config_name, spec])
      else:
        overall_status = False

    return overall_status
