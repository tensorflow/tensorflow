load(
    "//third_party/py/rules_pywrap:pywrap.default.bzl",
    _pybind_extension = "pybind_extension",
    _pywrap_aware_cc_import = "pywrap_aware_cc_import",
    _pywrap_aware_filegroup = "pywrap_aware_filegroup",
    _pywrap_aware_genrule = "pywrap_aware_genrule",
    _pywrap_binaries = "pywrap_binaries",
    _pywrap_common_library = "pywrap_common_library",
    _pywrap_library = "pywrap_library",
    _stripped_cc_info = "stripped_cc_info",
    _use_pywrap_rules = "use_pywrap_rules",
)

pybind_extension = _pybind_extension
use_pywrap_rules = _use_pywrap_rules
pywrap_library = _pywrap_library
pywrap_common_library = _pywrap_common_library
stripped_cc_info = _stripped_cc_info
pywrap_aware_filegroup = _pywrap_aware_filegroup
pywrap_aware_genrule = _pywrap_aware_genrule
pywrap_aware_cc_import = _pywrap_aware_cc_import
pywrap_binaries = _pywrap_binaries
