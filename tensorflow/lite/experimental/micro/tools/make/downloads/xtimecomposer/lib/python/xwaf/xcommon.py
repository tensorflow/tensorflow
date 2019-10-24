from waflib.Build import UninstallContext
from waflib.Configure import conf
from waflib import Errors, Logs
from waflib.Scripting import DistCheck

from subprocess import Popen, PIPE

import shutil
import os

from xwaf.utils import xcommon_utils as xcom
def warn(warning_string):
    Logs.warn("Warning: %s" % warning_string)

def error(error_string):
    raise Errors.WafError(error_string)

xcom.warn = warn
xcom.error = error

# Setting this value to xwaf.compiler_xcc completely removes PCA analysis from the build. It is useful
# to do this when you are debugging and you want to remove this complexity
COMPILER = 'xwaf.compiler_xcc_pca'

def options(opt):
    opt.load(COMPILER)

    opt.add_option('--config', action='store')
    opt.add_option('--debug', action='store_true')

def configure(conf):
    conf.load(COMPILER)

@conf
def xcommon_app(bld, app):
    if not bld.variant:
        target = bld.path.abspath() + '/bin/' + app.name + '.xe'
    else:
        target = bld.path.abspath() + '/bin/' + bld.variant + '.xe'

    bld.xc_program(
        name = app.name,
        source = app.source_nodes,
        includes = app.include_dir_nodes,
        export_includes = app.export_include_dir_nodes,
        use = app.uses,
        target = target,
        install_path = None)

@conf
def xcommon_module(bld, module):

    name = module.name
    export_includes = module.export_include_dir_nodes
    use = module.uses
    source = module.source_nodes

    # We override module.use_source if we are trying to uninstall AND there is some source to build with
    # after we have uninstalled! Essentially this is because we will be using source after we have
    # uninstalled.
    if module.use_source or (isinstance(bld, UninstallContext) and source):

        includes = module.include_dir_nodes
        cflags   = module.xcc_flags

        if module.is_library:
            bld.xc_stlib(
                name = name,
                source          = source,
                includes        = includes,
                export_includes = export_includes,
                use             = use,
                cflags          = cflags,
                target          = module.library_name,
                install_path    = bld.path.make_node(xcom.library))
        else:
            bld.xc_objects(
                source          = source,
                includes        = includes,
                export_includes = export_includes,
                use             = use,
                cflags          = cflags,
                target          = name)
    else:
        if module.is_library:
            bld.xc_read_stlib(
                name = name,
                target = module.library_name,
                lib_paths = [bld.path.find_node(xcom.library).abspath()],
                export_includes = export_includes,
                use             = use)
        else:
            bld.xc_read_includes(
                name = name,
                export_includes = export_includes,
                use             = use)

@conf
def do_xcommon(bld, makefile_contents=None):

    app = bld.path
    app.env = bld.env

    xcom.read_makefiles(app, bld.options.config, makefile_contents)

    # The XCC_FLAGS often contain settings for the preprocessor, compiler and linker. It makes sense
    # to split these settings up so that we don't provide settings to compile stages that will simply
    # ignore them.
    for flag in app.xcc_flags:
        if flag.startswith('-D'):
            bld.env.append_value('DEFINES', flag[2:])
        elif flag == '-report':
            bld.env.append_value('LDFLAGS', flag)
        else:
            # Unrecognised, so provide to both compiler and linker and let them sort it out.
            bld.env.append_value('CFLAGS', flag)
            bld.env.append_value('LDFLAGS', flag)

    target = app.makefile_contents['TARGET'][0]
    targets = []
    for n in app.source_dir_nodes:
        a = n.find_node('%s.xn' % target)
        if a:
            targets.append(a)

    if len(targets) == 1:
        target_string = ['-x', 'none', targets[0].abspath()]
    else:
        target_string = '-target=%s' % target

    bld.env.append_value('CFLAGS', target_string)
    bld.env.append_value('LDFLAGS', target_string)

    if bld.options.debug:
        bld.env.append_value('DEFINES', 'DEBUG')

    bld.env.append_value('INCLUDES', app.export_include_dir_nodes)

    optional_headers = []
    for module in app.dep_modules.values():
        optional_headers += module.makefile_contents.get('OPTIONAL_HEADERS', [])

    existing_optional_header_nodes = []
    for n in bld.env.INCLUDES:
        existing_optional_header_nodes += (n.ant_glob(optional_headers))

    existing_optional_headers = [x.name for x in existing_optional_header_nodes]

    existing_optional_header_defines = ['__{}_exists__=1'.format(h.replace('.', '_')) for h in existing_optional_headers]

    bld.env.append_value('DEFINES', existing_optional_header_defines)

    for module in app.dep_modules.values():
        bld.xcommon_module(module)

    bld.xcommon_app(app)

def dist(ctx):

    app = ctx.path

    xcom.read_makefiles(app, ctx.options.config)

    # We set the base path such that all the contents of the tarfile have 'positive'
    # directory depth.
    ctx.base_path = app.find_node('../../../')
    ctx.base_name = app.name

    global_excl = ctx.get_excl().split()

    files = []
    for module in app.dep_modules.values():
        if not module.use_source:
            excl = global_excl + '**/*.xc **/*.c **/*.S'.split()
        else:
            excl = global_excl

        files.extend(module.ant_glob('**/*', excl=excl))

    files.extend(app.ant_glob('**/*', excl=global_excl))

    ctx.files = files

    Logs.info("Creating archive at %s" % ctx.base_path.abspath())

# This function is copied almost directly from waflib.Build
def rm_empty_dirs(tgt):
    """
    Removes empty folders recursively when uninstalling.

    :param tgt: absolute path
    :type tgt: string
    """
    while tgt:
        tgt = os.path.dirname(tgt)
        try:
            os.rmdir(tgt)
        except OSError:
            break

class corrected_distcheck(DistCheck):
    """
    This class overrides the class DistCheck in waflib.Scripting. We do this because there appears to be some bugs
    in the check() function which are corrected here. I'm not certain if this is a bug because we are using the
    directories in a very 'creative' way that is not common for projects...
    """

    def check(self):
        """
        Creates the archive, uncompresses it and tries to build the project
        """
        import tempfile, tarfile

        # BUG FIX: We must obtain the path of the tarfile for reading in the SAME way as when the tarfile was opened
        # for writing (see code in waflib.Scripting in class Dist)
        arch_name = self.get_arch_name()
        node = self.base_path.find_node(arch_name)

        with tarfile.open(node.abspath()) as t:
            for x in t:
                t.extract(x)

        instdir = tempfile.mkdtemp('.inst', self.get_base_name())
        cmd = self.make_distcheck_cmd(instdir)

        try:
            # BUG FIX: Correspondlingly, we must chnage directory correctly to get to the original wscript file
            ret = Utils.subprocess.Popen(cmd, cwd=os.path.join(self.base_name, self.path.path_from(self.base_path))).wait()
        finally:
            # BUG FIX: Clean up whether it was a success or not
            shutil.rmtree(self.get_base_name())
            rm_empty_dirs(os.path.join(instdir, 'dummy_target'))

        if ret:
            raise Errors.WafError('distcheck failed with code %r' % ret)

        if os.path.exists(instdir):
            raise Errors.WafError('distcheck succeeded, but files were left in %s' % instdir)

def distcheck(ctx):
    dist(ctx)
