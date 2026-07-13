"""XLA restricted package helpers."""

load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

def xla_restricted_allow(target):
    """Sentinel for presubmit checks.

    Google's presubmit checks operate on deltas, so we opt to use a function
    with a distinct name specific to this feature so we can avoid false
    positives.

    This function returns a function that returns the target name. This is to
    prevent the user from bypassing presubmit checks by passing a list of
    strings to xla_restricted_verify instead of using this function.

    Args:
        target: The target name to allow, relative to the package.
    """

    def xla_restricted_allow_inner_fn():
        return target

    # We return a function so that the user cannot bypass presubmit checks by
    # passing a list of strings to verify.
    return xla_restricted_allow_inner_fn

def xla_restricted_verify(allowed_targets, test_suite_generator_functions = ["xla_test"]):
    """Verifies that all actual targets have been allowed.

    Args:
        allowed_targets: A list of functions returned by xla_restricted_allow().
        test_suite_generator_functions: A list of test suite generator functions
            to ignore non-test_suite expansions of.
    """

    allowed_names = []
    for allowed_target in allowed_targets:
        # Stringify the function to check that it was created by us and not the user.
        allowed_target_fn_str = str(allowed_target)
        if (
            not "xla_restricted_allow_inner_fn" in allowed_target_fn_str or
            not "restricted_package.bzl" in allowed_target_fn_str
        ):
            fail("Restricted package allowlist must be a list of functions returned by xla_restricted_allow().")

        target_name = allowed_target()
        allowed_names.append(Label(target_name).name)

    actual_targets = []
    for name, details in native.existing_rules().items():
        # Only include the top-level target for generator functions to reduce
        # noise in the allowlist. The top-level target has name equal to the
        # generator_name. It is typically a test_suite, but in some environments
        # it may be a cc_test.
        if (
            details.get("generator_function", "") in test_suite_generator_functions and
            name != details.get("generator_name", "")
        ):
            continue

        if (
            details.get("generator_function", "") == "generate_backend_suites" and
            details.get("kind", "") == "test_suite"
        ):
            continue

        actual_targets.append(name)

    allowed_names = sorted(allowed_names)
    actual_targets = sorted(actual_targets)

    if actual_targets != allowed_names:
        fail("Restricted package allowlist does not match actual targets.\ngot: {}\nexpected: {}".format(actual_targets, allowed_names))
