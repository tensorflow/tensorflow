from subprocess import Popen, PIPE
import operator
import shlex
import os
import re

library = 'lib'

# It is intended that the user of this module will override warn() and error() as required
def warn(warn_string):
    print("WARNING: %s" % warn_string)

def error(error_string):
    raise Exception(error_string)

# Variables with these names are handled by xcommon, but not by this code. I intend to add support
# on an 'as required' basis. At least this way the user will be informed that something unusual may
# be happening.
unhandled_variables = [
    ('XCC_C_FLAGS.*', 'Remove this variable. Use XCC_FLAGS instead. If a file needs particular flags, use XCC_FLAGS_filename instead'),
    ('XCC_ASM_FLAGS.*', 'Remove this variable. Use XCC_FLAGS instead. If a file needs particular flags, use XCC_FLAGS_filename instead'),
    ('XCC_MAP_FLAGS.*', 'Remove this variable. Use XCC_FLAGS instead'),
    ('VERBOSE', 'Remove this variable.'),
    ('LIB_DIRS', 'This variable may be handled in the future depending on demand'),
    ('MODULE_LIBRARIES', 'This variable may be handled in the future depending on demand'),
    ('MODULE_XCC_XC_FLAGS', 'Remove this variable. Use MODULE_XCC_FLAGS instead. If a file needs particular flags, use XCC_FLAGS_filename instead'),
    ('MODULE_XCC_C_FLAGS', 'Remove this variable. Use MODULE_XCC_FLAGS instead. If a file needs particular flags, use XCC_FLAGS_filename instead'),
    ('MODULE_XCC_ASM_FLAGS', 'Remove this variable. Use MODULE_XCC_FLAGS instead. If a file needs particular flags, use XCC_FLAGS_filename instead'),
    ('LIBRARIES', 'This variable may be handled in the future depending on demand'),
    ('EXPORT_SOURCE_DIRS', 'Remove this variable. Use SOURCE_DIRS and EXCLUDE_FILES to control what goes into the library'),
    ('LIB_XCC_FLAGS.*', 'This variable may be handled in the future depending on demand')
    ]

def parse_makefile(node, makefile, xcc_flags=None):
    lines = []

    # Override the XMOS_MAKE_PATH such that it points to our benign makefile instead of the original xcommon one
    lines.append("XMOS_MAKE_PATH:=%s\n" % os.path.join(os.environ['XMOS_MAKE_PATH'], 'xwaf'))

    # Allow XCC_FLAGS to be exported by the app makefile into the module makefiles
    if xcc_flags:
        lines.append("XCC_FLAGS:=%s\n" % ' '.join(xcc_flags))

    lines.append("VARS_OLD := $(.VARIABLES)\n")

    lines.append(node.find_node(makefile).read())

    lines.append(r"$(foreach v, $(filter-out $(VARS_OLD) VARS_OLD,$(.VARIABLES)), $(info $(v) = $($(v))))")
    lines.append("\n")
    lines.append("dummy_target:\n")
    lines.append("\techo\n")

    mf_parseable = node.make_node(makefile + "_parseable")
    mf_parseable.write(''.join(lines))

    p = Popen(['xmake', '-f', mf_parseable.abspath()], stdout=PIPE)
    lines = p.communicate()[0]
    mf_parseable.delete()

    if not isinstance(lines, str):
        # Python 3 needs this
        lines = lines.decode('utf-8')
    lines = lines.split('\n')

    makefile_contents = {}
    for line in lines:
        # We use shlex.split(line) rather than line.split() because we want line contents:
        # this is "a test"
        # ...to become:
        # ['this', 'is', 'a test']
        l = shlex.split(line)

        if len(l) < 2:
            continue
        else:
            makefile_contents[l[0]] = l[2:]

    for variable in makefile_contents:
        for (unhandled_variable, advice) in unhandled_variables:
            if re.search(unhandled_variable, variable):
                error("Encountered the variable %r in %s. It is ignored. Advice: %s" % (variable, node.name, advice))

    return makefile_contents

def find_module(app, module_name):

    modules = []

    if 'PROJECT_ROOT' in app.env:
        cwd = app.find_node(app.env['PROJECT_ROOT'])
        # A value of 1 just searches cwd folder.
        remaining_attempts = 1
    else:
        cwd = app
        # A value of 4 searches from app/../../..
        remaining_attempts = 4

    while remaining_attempts and not modules:
        # ant_glob() is slow, so we cache the results
        if not hasattr(cwd, 'all_modules'):
            cwd.all_modules = [x.parent for x in cwd.ant_glob('**/*/module_build_info', excl='**/*/build build')]
        modules = [m for m in cwd.all_modules if m.name == module_name]
        cwd = cwd.parent
        remaining_attempts -= 1

    if len(modules) == 1:
        return modules[0]
    elif len(modules) > 1:
        error("Ambiguous search result, because module %r can be found in more than one location: %s" %
            (module_name, ' '.join([m.abspath() for m in modules])))
    else:
        error("Module %r not found" % module_name)

def _compare_version(v1, v2):
    """ Compare version strings of the form '1.2.3'
    :param v1: the first version string
    :param v2: the second version string
    :returns: -1 if v1 < v2, 0 if v1 == v2, and, 1 if v1 > v2
    """

    def cmp(x, y):
        """
        Replacement for built-in function cmp that was removed in Python 3
        Compare the two objects x and y and return an integer according to
        the outcome. The return value is negative if x < y, zero if x == y
        and strictly positive if x > y.
        """

        return (x > y) - (x < y)

    v1, v2 = (list(map(int, v.split('.'))) for v in (v1, v2))
    d = len(v2) - len(v1)
    return cmp(v1 + [0] * d, v2 + [0] * -d)

def checkModuleAgainstRequirement(module, requirement):

    def get_major_number(s):
        s = s.split('.')
        s = int(s[0])
        return s

    version = module.makefile_contents.get('VERSION', [None])[0]

    required_version, required_rule, requirer = requirement

    if required_version:
        if not version:
            error("'{}' requires a version number".format(module.name))

        comparison = _compare_version(version, required_version)
        ops = {
            '>=': operator.ge,
        }
        if not ops[required_rule](comparison, 0):
            error('Module {} requires version of {} to be {}{}. Actual verion: {}'.
                format(requirer.name, module.name, required_rule, required_version, version))

        if get_major_number(version) > get_major_number(required_version):
            warn(
                'Module {} requires version of {} to be {}{} and actual '
                'version has greater major version: {}. '
                'There could be API incompatibilities.'.format(
                    requirer.name, module.name, required_rule, required_version,
                    version))

def recurse_modules(app, node=None, use_source=True):

    if not node:
        node = app

    uses = node.uses

    if not hasattr(app, 'dep_modules'):
        app.dep_modules = {}

    for use in uses:
        if use not in app.dep_modules:
            # Haven't seen this module before. Create and add the module

            module = find_module(app, use)

            module.makefile_contents = parse_makefile(module,
                                                      'module_build_info',
                                                      app.xcc_flags)
            apply_module_xcc_flags(module, app.xcc_flags)
            apply_source_nodes(module, app.xcc_flags + module.xcc_flags)
            apply_export_include_dir_nodes(module)
            apply_include_dir_nodes(module)
            apply_library_name(module)
            apply_uses(module)

            app.dep_modules[use] = module
        else:
            # Seen this module before.
            module = app.dep_modules[use]

        requirement = uses[use]
        checkModuleAgainstRequirement(module, requirement)

        if module.library_name != '':
            module.is_library = True
        else:
            module.is_library = False

        # If the library can be found in its installed location, we use it (and therefore don't use source)
        if app.find_node(library + '/lib%s.a' % module.library_name) or not use_source:
            module.use_source = False
        else:
            module.use_source = getattr(module, 'use_source', True)

        recurse_modules(app, module, use_source=module.use_source)

def read_makefiles(app, config, makefile_contents=None):
    if makefile_contents is None:
        makefile_contents = parse_makefile(app, 'Makefile')

    app.makefile_contents = makefile_contents

    if 'XCOMMON_MAKEFILE' not in app.makefile_contents:
        error('Makefile does not include an xcommon makefile')

    xcommon_makefile = app.makefile_contents['XCOMMON_MAKEFILE'][0]

    if xcommon_makefile == "Makefile.common":
        pass
    elif xcommon_makefile == "Makefile.library":
        error('Makefile includes Makefile.library. This is not yet handled')
    else:
        error('Makefile includes an unrecognised Makefile')

    configs = {}
    for variable in app.makefile_contents:
        if variable.startswith('XCC_FLAGS_'):
            key = variable[len('XCC_FLAGS_'):]
            configs[key] = app.makefile_contents[variable]

    if config:
        try:
            XCC_FLAGS = configs[config]
        except:
            error('Config %r not found. Available values: %s' % (config, ' '.join(configs.keys())))
    else:
        if configs.keys():
            error('Config not supplied. Available values: %s' % ' '.join(configs.keys()))
        else:
            XCC_FLAGS = app.makefile_contents.get('XCC_FLAGS', [])

    app.xcc_flags = XCC_FLAGS

    apply_source_nodes(app, app.xcc_flags)
    apply_export_include_dir_nodes(app)
    apply_include_dir_nodes(app)
    apply_uses(app)

    recurse_modules(app)

def apply_library_name(module):
    makefile_contents = module.makefile_contents

    module.library_name = makefile_contents.get('LIBRARY', [''])[0]

def apply_uses(app_or_module):
    makefile_contents = app_or_module.makefile_contents

    # By allowing both options, we can use this function for apps and modules.
    if 'DEPENDENT_MODULES' in makefile_contents:
        raw_dependent_modules = makefile_contents['DEPENDENT_MODULES']
    elif 'USED_MODULES' in makefile_contents:
        raw_dependent_modules = makefile_contents['USED_MODULES']
    else:
        raw_dependent_modules = []

    dependency_regex = [
        '(?P<name>[a-zA-Z0-9_ ]*)', '\(?', '(?P<version_requirement>\D+)?',
        '(?P<version_number>\d+(?:\.\d+)*)?'
    ]
    dependency_regex = re.compile(''.join(dependency_regex))

    uses = {}
    for dep in raw_dependent_modules:
        info = dependency_regex.match(dep)
        name = info.group('name')
        version_number = info.group('version_number')
        version_requirement = info.group('version_requirement')
        uses[name] = (version_number, version_requirement, app_or_module)

    app_or_module.uses = uses

def get_dir_nodes(app_or_module, raw_dirs):
    """
    Given a waf node object and a list of strings, this function returns a list of
    waf node objects representing directories
    """

    dir_nodes = []
    for raw_dir in raw_dirs:
        if raw_dir.endswith('/*'):
            recurse = True
            rootdir = app_or_module.find_dir(raw_dir[:-2])
        elif raw_dir.endswith('*'):
            recurse = True
            rootdir = app_or_module
        else:
            recurse = False
            rootdir = app_or_module.find_dir(raw_dir)

        if not rootdir:
            error("Couldn't find directory %r in %s" % (raw_dir, app_or_module.abspath()))

        dir_nodes.append(rootdir)

        if recurse:
            dir_nodes.extend(rootdir.ant_glob('**/*', dir=True, src=False, excl='build'))

    return dir_nodes

def apply_source_nodes(app_or_module, existing_xcc_flags):
    makefile_contents = app_or_module.makefile_contents

    raw_source_dirs       = makefile_contents.get('SOURCE_DIRS', ['*'])
    exclude_files         = makefile_contents.get('EXCLUDE_FILES', [])
    exclude_files = ['**/' + x for x in exclude_files]

    app_or_module.source_dir_nodes = get_dir_nodes(app_or_module, raw_source_dirs)

    source_dirs = []
    for raw_source_dir in raw_source_dirs:
        if raw_source_dir.endswith('*'):
            raw_source_dir += '*' # Enable recursion

        source_dirs.append(raw_source_dir + '/*.xc')
        source_dirs.append(raw_source_dir + '/*.S')
        source_dirs.append(raw_source_dir + '/*.c')

    # Need to sort because ant_glob seems to return the files in different orders each time. The sort saves unnecessary recompilation
    source_nodes   = sorted(app_or_module.ant_glob(source_dirs, excl=exclude_files + ['build']), key = lambda node: node.name)

    # Source files can have special flags assigned on a per source file basis
    for node in source_nodes:
        variable = 'XCC_FLAGS_%s' % node.name
        if (variable in makefile_contents):
            XCC_FLAGS = makefile_contents[variable]
            if XCC_FLAGS[:len(existing_xcc_flags)] == existing_xcc_flags:
                # User has acknowledged that XCC_FLAGS (or MODULE_XCC_FLAGS in the case of a module) will be applied
                # We remove the unnecessary additional flags
                node.xccflags = XCC_FLAGS[len(existing_xcc_flags):]
            else:
                error(
                    "%r sets variable %r, but does not include ${XCC_FLAGS} or ${MODULE_XCC_FLAGS} (as appropriate) as the first part of the variable."
                    % (app_or_module.name, variable))

    app_or_module.source_nodes = source_nodes

def get_include_nodes(app_or_module, raw_include_dirs):
    includes = []
    for raw_include_dir in raw_include_dirs:
        if raw_include_dir.endswith('*'):
            include_dirs = app_or_module.ant_glob(raw_include_dir + '*/*.h', excl='build')
            include_dirs = [x.parent for x in include_dirs]
            includes.extend(include_dirs)
        else:
            includes.append(app_or_module.find_node(raw_include_dir))

    # Need to sort because Python set() function is unordered. The sort prevents unnecessary recompilation
    includes = sorted(list(set(includes)), key = lambda node: node.abspath())

    return includes

def apply_export_include_dir_nodes(app_or_module):
    makefile_contents = app_or_module.makefile_contents

    if 'EXPORT_INCLUDE_DIRS' in makefile_contents:
        raw_include_dirs = makefile_contents.get('EXPORT_INCLUDE_DIRS', ['*'])
    else:
        raw_include_dirs = makefile_contents.get('INCLUDE_DIRS', ['*'])

    app_or_module.export_include_dir_nodes = get_include_nodes(app_or_module, raw_include_dirs)

def apply_include_dir_nodes(app_or_module):
    makefile_contents = app_or_module.makefile_contents

    raw_include_dirs = makefile_contents.get('INCLUDE_DIRS', ['*'])

    app_or_module.include_dir_nodes = get_include_nodes(app_or_module, raw_include_dirs)

def apply_module_xcc_flags(module, existing_xcc_flags):
    makefile_contents = module.makefile_contents

    if 'MODULE_XCC_FLAGS' in makefile_contents:
        # The user has supplied MODULE_XCC_FLAGS. We want the user to have written:
        # MODULE_XCC_FLAGS = ${XCC_FLAGS} x y z
        # (or equivalent)
        # By writing this, the user is acknowledging that the MODULE_XCC_FLAGS are additive.
        #
        # If they have written:
        # MODULE_XCC_FLAGS = x y z
        # ...then this is bad, because the user might be knowingly or unknowingly trying to use
        # MODULE_XCC_FLAGS to _remove_ flags that have been set in XCC_FLAGS

        XCC_FLAGS = makefile_contents['MODULE_XCC_FLAGS']
        if XCC_FLAGS[:len(existing_xcc_flags)] == existing_xcc_flags:
            # User has put ${XCC_FLAGS} or equivalent at start of list
            # Remove these, because MODULE_XCC_FLAGS are additive anyway
            module.xcc_flags = XCC_FLAGS[len(existing_xcc_flags):]
        else:
            error(
                "Module %r sets variable MODULE_XCC_FLAGS, but does not include ${XCC_FLAGS} as first part of the variable."
                % module.name)
            module.xcc_flags = XCC_FLAGS
    else:
        # The user does not supply MODULE_XCC_FLAGS. We are happy that the user knew that XCC_FLAGS will be
        # applied.
        module.xcc_flags = []
