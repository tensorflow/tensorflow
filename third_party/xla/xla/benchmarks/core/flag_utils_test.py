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
"""Unit tests for flags utility library."""

import os
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

from xla.benchmarks.core import flag_utils  # pylint: disable=g-direct-tensorflow-import


class FlagsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "empty_string",
          "",
          {},
      ),
      (
          "whitespace_string",
          "   ",
          {},
      ),
      (
          "key_value_equals",
          "--xla_tpu_dvfs_p_state=3 --xla_tpu_scoped_vmem_limit_kib=65536",
          {
              "xla_tpu_dvfs_p_state": "3",
              "xla_tpu_scoped_vmem_limit_kib": "65536",
          },
      ),
      (
          "space_separated",
          "--xla_tpu_dvfs_p_state 3 --other_flag hello",
          {
              "xla_tpu_dvfs_p_state": "3",
              "other_flag": "hello",
          },
      ),
      (
          "negative_numbers_equals",
          "--xla_tpu_dvfs_p_state=-1 --other_flag -10.5",
          {
              "xla_tpu_dvfs_p_state": "-1",
              "other_flag": "-10.5",
          },
      ),
      (
          "negative_numbers_space_separated",
          "--xla_tpu_dvfs_p_state -1",
          {
              "xla_tpu_dvfs_p_state": "-1",
          },
      ),
      (
          "boolean_and_nobool_flags",
          "--enable_feature --noenable_other",
          {
              "enable_feature": "true",
              "noenable_other": "true",
          },
      ),
      (
          "quotes_and_nested_equals",
          "--str_flag=\"hello world\" --key_eq=a=b=c --single_quoted='foo bar'",
          {
              "str_flag": "hello world",
              "key_eq": "a=b=c",
              "single_quoted": "foo bar",
          },
      ),
      (
          "repeated_flags_last_wins",
          "--xla_tpu_dvfs_p_state=1 --xla_tpu_dvfs_p_state=3",
          {
              "xla_tpu_dvfs_p_state": "3",
          },
      ),
      (
          "bare_dashes_ignored",
          "--flag1=1 -- - --flag2=2",
          {
              "flag1": "1",
              "flag2": "2",
          },
      ),
      (
          "single_dash_flags",
          "-foo=bar -baz 100",
          {
              "foo": "bar",
              "baz": "100",
          },
      ),
      (
          "flag_followed_by_flag",
          "--flag_a --flag_b=val",
          {
              "flag_a": "true",
              "flag_b": "val",
          },
      ),
      (
          "tokens_without_dash_ignored",
          "ignored_token --real_flag=123 another_ignored",
          {
              "real_flag": "123",
          },
      ),
      (
          "empty_value_equals",
          "--flag=",
          {
              "flag": "",
          },
      ),
  )
  def test_parse_libtpu_init_args(self, args_str, expected):
    self.assertEqual(flag_utils.parse_libtpu_init_args(args_str), expected)

  def test_parse_libtpu_init_args_default_empty_when_env_cleared(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      self.assertEqual(flag_utils.parse_libtpu_init_args(), {})

  def test_parse_libtpu_init_args_from_env_var(self):
    with mock.patch.dict(
        os.environ, {"LIBTPU_INIT_ARGS": "--xla_tpu_dvfs_p_state=5"}
    ):
      parsed = flag_utils.parse_libtpu_init_args()
      self.assertEqual(parsed, {"xla_tpu_dvfs_p_state": "5"})

  def test_parse_libtpu_init_args_custom_env_var(self):
    with mock.patch.dict(os.environ, {"CUSTOM_ENV": "--my_flag=custom_val"}):
      parsed = flag_utils.parse_libtpu_init_args(env_var="CUSTOM_ENV")
      self.assertEqual(parsed, {"my_flag": "custom_val"})

  def test_parse_libtpu_init_args_shlex_error_fallback(self):
    # Unclosed quote triggers ValueError in shlex.split, triggering split().
    args = '--flag1="unclosed quote --flag2=value'
    with mock.patch.object(flag_utils.logging, "warning") as mock_warning:
      parsed = flag_utils.parse_libtpu_init_args(args)
      mock_warning.assert_called_once()
    self.assertIn("flag2", parsed)
    self.assertEqual(parsed["flag2"], "value")

  @parameterized.named_parameters(
      (
          "int_exact_name",
          "xla_tpu_dvfs_p_state",
          "--xla_tpu_dvfs_p_state=3",
          int,
          None,
          3,
      ),
      (
          "int_with_leading_dashes",
          "--xla_tpu_dvfs_p_state",
          "--xla_tpu_dvfs_p_state=3",
          int,
          None,
          3,
      ),
      (
          "int_hyphenated_query_for_underscore_flag",
          "xla-tpu-dvfs-p-state",
          "--xla_tpu_dvfs_p_state=3",
          int,
          None,
          3,
      ),
      (
          "int_mixed_dashes_in_args_underscore_query",
          "xla_tpu_dvfs_p_state",
          "--xla-tpu_dvfs-p_state=3",
          int,
          None,
          3,
      ),
      (
          "int_mixed_dashes_in_args_hyphen_query",
          "--xla-tpu-dvfs-p-state",
          "--xla-tpu_dvfs-p_state=3",
          int,
          None,
          3,
      ),
      (
          "str_exact_name",
          "str_val",
          "--str_val=hello",
          str,
          None,
          "hello",
      ),
      (
          "str_without_flag_type",
          "str_flag",
          "--str_flag=123",
          None,
          None,
          "123",
      ),
      (
          "str_exact_clean_name",
          "custom_flag",
          "--custom_flag=hello",
          None,
          None,
          "hello",
      ),
      (
          "str_hyphen_lookup_for_underscore_flag",
          "custom-flag",
          "--custom_flag=hello",
          None,
          None,
          "hello",
      ),
      (
          "bool_value",
          "bool_val",
          "--bool_val=true",
          bool,
          None,
          True,
      ),
      (
          "float_value",
          "float_val",
          "--float_val=2.5",
          float,
          None,
          2.5,
      ),
      (
          "missing_flag_returns_none",
          "nonexistent_flag",
          "--xla_tpu_dvfs_p_state=3",
          None,
          None,
          None,
      ),
      (
          "missing_flag_returns_custom_default",
          "nonexistent_flag",
          "--xla_tpu_dvfs_p_state=3",
          int,
          42,
          42,
      ),
      (
          "empty_value_equals_returns_empty_string_not_default",
          "flag",
          "--flag=",
          None,
          "def",
          "",
      ),
  )
  def test_get_flag_from_libtpu_init_args(
      self, flag_name, args_str, flag_type, default, expected
  ):
    kwargs = {"args_str": args_str}
    if flag_type is not None:
      kwargs["flag_type"] = flag_type
    if default is not None:
      kwargs["default"] = default
    self.assertEqual(
        flag_utils.get_flag_from_libtpu_init_args(flag_name, **kwargs),
        expected,
    )

  @parameterized.named_parameters(
      ("true_lowercase", "true", True),
      ("one", "1", True),
      ("t", "t", True),
      ("yes", "yes", True),
      ("y", "y", True),
      ("false_lowercase", "false", False),
      ("zero", "0", False),
      ("f", "f", False),
      ("no", "no", False),
      ("n", "n", False),
  )
  def test_get_flag_from_libtpu_init_args_bool_variants(
      self, flag_val, expected
  ):
    self.assertEqual(
        flag_utils.get_flag_from_libtpu_init_args(
            "flag", args_str=f"--flag={flag_val}", flag_type=bool
        ),
        expected,
    )

  def test_get_flag_from_libtpu_init_args_invalid_bool(self):
    with self.assertRaises(ValueError):
      flag_utils.get_flag_from_libtpu_init_args(
          "flag", args_str="--flag=invalid", flag_type=bool
      )

  def test_get_flag_from_libtpu_init_args_custom_env_var(self):
    with mock.patch.dict(os.environ, {"MY_TPU_ARGS": "--some_flag=abc"}):
      val = flag_utils.get_flag_from_libtpu_init_args(
          "some_flag", env_var="MY_TPU_ARGS"
      )
      self.assertEqual(val, "abc")

  @parameterized.named_parameters(
      ("integer_positive", "123", True),
      ("integer_negative", "-123", True),
      ("float_positive", "123.45", True),
      ("float_negative", "-123.45", True),
      ("scientific_notation", "1e-5", True),
      ("alphabetic_string", "abc", False),
      ("empty_string", "", False),
      ("flag_token", "--flag", False),
  )
  def test_is_number(self, value, expected):
    self.assertEqual(flag_utils._is_number(value), expected)

  @parameterized.named_parameters(
      ("true_lowercase", "true", True),
      ("one", "1", True),
      ("t", "t", True),
      ("yes_lowercase", "yes", True),
      ("y_lowercase", "y", True),
      ("empty_string", "", True),
      ("true_padded_uppercase", "  TRUE  ", True),
      ("yes_capitalized", "Yes", True),
      ("false_lowercase", "false", False),
      ("zero", "0", False),
      ("f_lowercase", "f", False),
      ("no_lowercase", "no", False),
      ("n_lowercase", "n", False),
      ("false_padded_uppercase", "  FALSE  ", False),
      ("no_capitalized", "No", False),
  )
  def test_parse_bool(self, value, expected):
    self.assertEqual(flag_utils._parse_bool(value), expected)

  def test_parse_bool_invalid(self):
    with self.assertRaises(ValueError):
      flag_utils._parse_bool("maybe")

  @parameterized.named_parameters(
      ("bool_true", "true", bool, True),
      ("bool_false", "false", bool, False),
      ("int", "123", int, 123),
      ("float", "3.14", float, 3.14),
      ("str", "hello", str, "hello"),
  )
  def test_convert_value(self, value, flag_type, expected):
    self.assertEqual(flag_utils._convert_value(value, flag_type), expected)

  def test_get_flag_value_fallback(self):
    test_flag_values = flags.FlagValues()
    flags.DEFINE_integer(
        "test_p_state",
        -1,
        "Test p-state flag",
        flag_values=test_flag_values,
    )
    test_flag_values.mark_as_parsed()

    # Flag is in flag_values with default -1 -> falls back to env var
    with mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": "--test_p_state=3"}):
      val = flag_utils.get_flag_value(
          "test_p_state",
          default=None,
          flag_type=int,
          flag_values=test_flag_values,
      )
      self.assertEqual(val, 3)

    # Flag is set in flag_values to non -1 -> uses flag_values
    test_flag_values.test_p_state = 7
    with mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": "--test_p_state=3"}):
      val = flag_utils.get_flag_value(
          "test_p_state",
          default=None,
          flag_type=int,
          flag_values=test_flag_values,
      )
      self.assertEqual(val, 7)

    # Flag does not exist in flag_values -> uses env var
    with mock.patch.dict(
        os.environ, {"LIBTPU_INIT_ARGS": "--unregistered_flag=42"}
    ):
      val = flag_utils.get_flag_value(
          "unregistered_flag",
          default=None,
          flag_type=int,
          flag_values=test_flag_values,
      )
      self.assertEqual(val, 42)

    # Flag does not exist anywhere -> returns default
    with mock.patch.dict(os.environ, {}, clear=True):
      val = flag_utils.get_flag_value(
          "missing_flag",
          default=99,
          flag_type=int,
          flag_values=test_flag_values,
      )
      self.assertEqual(val, 99)

  def test_get_flag_value_zero_and_false(self):
    test_flag_values = flags.FlagValues()
    flags.DEFINE_integer(
        "zero_flag",
        0,
        "Zero flag",
        flag_values=test_flag_values,
    )
    flags.DEFINE_bool(
        "false_flag",
        False,
        "False flag",
        flag_values=test_flag_values,
    )
    test_flag_values.mark_as_parsed()

    # Zero and False values in flag_values should be respected and not fall back
    with mock.patch.dict(
        os.environ,
        {"LIBTPU_INIT_ARGS": "--zero_flag=5 --false_flag=true"},
    ):
      self.assertEqual(
          flag_utils.get_flag_value(
              "zero_flag", flag_type=int, flag_values=test_flag_values
          ),
          0,
      )
      self.assertFalse(
          flag_utils.get_flag_value(
              "false_flag", flag_type=bool, flag_values=test_flag_values
          )
      )

  @parameterized.named_parameters(
      ("flag_set_overrides_env", 7, "--xla_tpu_dvfs_p_state=3", 7),
      ("flag_default_falls_back_to_env", -1, "--xla_tpu_dvfs_p_state=3", 3),
      ("flag_default_no_env_returns_none", -1, None, None),
  )
  def test_get_flag_value_xla_tpu_dvfs_p_state_defined_in_flags(
      self, flag_val, env_arg_str, expected
  ):
    test_flag_values = flags.FlagValues()
    flags.DEFINE_integer(
        "xla_tpu_dvfs_p_state",
        -1,
        "DVFS P-state",
        flag_values=test_flag_values,
    )
    test_flag_values.mark_as_parsed()
    test_flag_values.xla_tpu_dvfs_p_state = flag_val

    env_dict = (
        {"LIBTPU_INIT_ARGS": env_arg_str} if env_arg_str is not None else {}
    )
    with mock.patch.dict(os.environ, env_dict, clear=(env_arg_str is None)):
      self.assertEqual(
          flag_utils.get_flag_value(
              "xla_tpu_dvfs_p_state",
              default=None,
              flag_type=int,
              flag_values=test_flag_values,
          ),
          expected,
      )

  @parameterized.named_parameters(
      ("found_in_env", "--xla_tpu_dvfs_p_state=5", 5),
      ("missing_in_env", None, None),
  )
  def test_get_flag_value_xla_tpu_dvfs_p_state_not_defined_in_flags(
      self, env_arg_str, expected
  ):
    test_flag_values = flags.FlagValues()
    test_flag_values.mark_as_parsed()

    env_dict = (
        {"LIBTPU_INIT_ARGS": env_arg_str} if env_arg_str is not None else {}
    )
    with mock.patch.dict(os.environ, env_dict, clear=(env_arg_str is None)):
      self.assertEqual(
          flag_utils.get_flag_value(
              "xla_tpu_dvfs_p_state",
              default=None,
              flag_type=int,
              flag_values=test_flag_values,
          ),
          expected,
      )

  @parameterized.named_parameters(
      ("hyphen_in_defined_name", "hyphen_flag", "hyphen-flag", "default_val"),
      ("leading_double_dash", "my_flag", "--my_flag", "default_val"),
      ("leading_dash_and_hyphen", "my_flag", "-my-flag", "default_val"),
  )
  def test_get_flag_value_name_variants_in_flag_values(
      self, defined_name, query_name, expected
  ):
    test_flag_values = flags.FlagValues()
    flags.DEFINE_string(
        defined_name,
        expected,
        "Test flag",
        flag_values=test_flag_values,
    )
    test_flag_values.mark_as_parsed()

    val = flag_utils.get_flag_value(
        query_name,
        default=None,
        flag_values=test_flag_values,
    )
    self.assertEqual(val, expected)

  def test_get_flag_value_custom_env_var_fallback(self):
    test_flag_values = flags.FlagValues()
    test_flag_values.mark_as_parsed()

    with mock.patch.dict(
        os.environ, {"MY_TPU_ARGS": "--custom_env_flag=success"}
    ):
      val = flag_utils.get_flag_value(
          "custom_env_flag",
          default=None,
          flag_values=test_flag_values,
          env_var="MY_TPU_ARGS",
      )
      self.assertEqual(val, "success")

  @parameterized.named_parameters(
      ("none_flag", "none_flag", 10),
      ("neg_flag", "neg_flag", 20),
  )
  def test_get_flag_value_fallback_when_flag_is_none_or_minus_one(
      self, flag_name, expected
  ):
    test_flag_values = flags.FlagValues()
    flags.DEFINE_integer(
        "none_flag", None, "None flag", flag_values=test_flag_values
    )
    flags.DEFINE_integer(
        "neg_flag", -1, "Neg flag", flag_values=test_flag_values
    )
    test_flag_values.mark_as_parsed()

    with mock.patch.dict(
        os.environ, {"LIBTPU_INIT_ARGS": "--none_flag=10 --neg_flag=20"}
    ):
      self.assertEqual(
          flag_utils.get_flag_value(
              flag_name, flag_type=int, flag_values=test_flag_values
          ),
          expected,
      )


if __name__ == "__main__":
  absltest.main()
