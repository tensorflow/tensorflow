import os
from waflib.Configure import conf
from waflib.TaskGen import extension, after_method, before_method, feature, task_gen
from waflib.Logs import warn
from waflib.Tools import c
from waflib import Node, Utils

def options(opt):
    pass

def configure(conf):
    conf.load('c')

    conf.env.AR='xmosar' # Seems to work with default ar too, but this is safer on Windows?
    conf.env.ARFLAGS = ['rcs']
    conf.env.CC='xcc'
    conf.env.LINK_CC='xcc'
    conf.env.CC_TGT_F            = ['-c', '-o']
    conf.env.CCLNK_TGT_F         = ['-o']
    conf.env.XPCA=os.path.join(os.environ['XMOS_TOOL_PATH'], 'libexec', 'xpca')
    conf.env.LIBPATH_ST = '-L%s'
    conf.env.STLIBPATH_ST = '-L%s'
    conf.env.LIB_ST = '-l%s'
    conf.env.STLIB_ST = '-l%s'
    conf.env.CPPPATH_ST = '-I%s'
    conf.env.DEFINES_ST = '-D%s'
    conf.env.cstlib_PATTERN = 'lib%s.a'

    conf.env.do_pca = False

###################
# Aliases
###################

@conf
def xc_program(self, *k, **kw):
    kw['features'] = 'c cprogram'
    return self(*k, **kw)

@conf
def xc_objects(self, *k, **kw):
    kw['features'] = 'c'
    return self(*k, **kw)

@conf
def xc_stlib(self, *k, **kw):
    kw['features'] = 'c cstlib'
    return self(*k, **kw)

@conf
def xc_read_stlib(self, *k, **kw):
    kw['features'] = 'fake_lib_with_target'
    kw['lib_type'] = 'stlib'
    return self(*k, **kw)

@conf
def xc_read_includes(self, *k, **kw):
    return self(*k, **kw)

###################
# Extensions
###################

@extension('.xc')
def xc_hook(self, xc_source_file_node):

    # Waf doesn't like the run_str unless cflags is always present.
    if not hasattr(xc_source_file_node, 'xccflags'):
        xc_source_file_node.xccflags = []

    if self.bld.env.do_pca:
        if 'cstlib' in self.features:
            t = self.create_compiled_task('source_compile_task', xc_source_file_node)
        else:
            t = self.create_analysed_task('xc_analyse_task', xc_source_file_node)
    else:
        t = self.create_compiled_task('source_compile_task', xc_source_file_node)

    return t

@extension('.S')
@extension('.c')
def source_hook_patched(self, source_file_node):

    # Waf doesn't like the run_str unless cflags is always present.
    if not hasattr(source_file_node, 'xccflags'):
        source_file_node.xccflags = []

    # source_file_node represents a .c or a .S source code file. We create a task to compile it.
    ct = self.create_compiled_task('source_compile_task', source_file_node)

    if self.bld.env.do_pca:
        if 'cstlib' in self.features:
            return ct
        else:
            # The output from compiling is an object file
            object_file_node = ct.outputs[0]

            at = self.create_analysed_task('object_analyse_task', object_file_node)

            return at
    else:
        return ct

###################
# TaskGen methods
###################

@feature('*')
@before_method('process_rule')
def remove_methods(self):
    unwanted_methods = 'set_full_paths_hpux set_macosx_deployment_target create_task_macapp create_task_macplist'.split()

    for x in unwanted_methods:
        try:
            self.meths.remove(x)
        except:
            pass
    
@feature('fake_lib_with_target')
def process_lib_with_target(self):
    """
    This function is based heavily 'process_lib' (in waflib/Tools/ccroot.py). The differences are
    * This looks for the library based on self.target, NOT self.name
    * This allows self.target to be different to self.name.
    """

    node = None

    name = self.env.cstlib_PATTERN % self.target
    for x in self.lib_paths + [self.path]:
        if not isinstance(x, Node.Node):
            x = self.bld.root.find_node(x) or self.path.find_node(x)
            if not x:
                continue

        node = x.find_node(name)
        if node:
            try:
                Utils.h_file(node.abspath())
            except EnvironmentError:
                raise ValueError('Could not read %r' % y)
            break
        else:
            continue
    else:
        raise Errors.WafError('could not find library %r' % name)
    self.link_task = self.create_task('fake_%s' % self.lib_type, [], [node])

###################
# Tasks
###################

def get_parent_task_gens(task_gen):
    parent_tgs = []
    prev_len = len(parent_tgs)

    parent_tgs.append(task_gen)

    while len(parent_tgs) != prev_len:
        prev_len = len(parent_tgs)

        for tg in task_gen.bld.get_all_task_gen():
            for parent_tg in parent_tgs:
                if parent_tg.name in getattr(tg, 'use', []) and tg not in parent_tgs:
                    parent_tgs.append(tg)

    # Remove the original task_gen, because we're only interested in its parents.
    return parent_tgs[1:]

def get_parent_export_includes(task_gen):
    parent_export_includes = []
    for parent_tg in get_parent_task_gens(task_gen):
        parent_export_includes += getattr(parent_tg, 'export_includes', [])

    return parent_export_includes

def check_deps(task):
    deps = task.generator.bld.node_deps[task.uid()]

    illegal_deps = []

    for dep in deps:
        if not dep.name.endswith('.h'):
            # Only apply check to header files that have have been found by original ant_glob

            # I don't particularly love this check for '.h'. I do it because the original search for INCLUDE_DIRS only looks for
            # dirs containing files ending in .h. So if someone includes another file with a different ending, we will raise a warning.
            continue

        if dep.parent in task.generator.includes:
            # This dependency is satisfied by this module or by a module
            # that this module uses.
            continue

        if dep.parent not in get_parent_export_includes(task.generator):
            illegal_deps.append(dep)

    if illegal_deps:
        warn_str = "WARNING: %s illegally includes (either directly or indirectly) the following headers:\n" % task.inputs[0].name 

        for illegal_dep in illegal_deps:
            warn_str += "\t%s" % illegal_dep.name
            for tg in task.generator.bld.get_all_task_gen():
                if illegal_dep.parent in getattr(tg, 'export_includes', []):
                    warn_str += " (from %s)" % tg.name
            warn_str += "\n"

        warn(warn_str)

class source_compile_task(c.c):
    run_str = ' '.join([
        '${CC}',
        '${ARCH_ST:ARCH}',
        '${CFLAGS}',
        '${tsk.inputs[0].xccflags}',
        '${FRAMEWORKPATH_ST:FRAMEWORKPATH}',
        '${CPPPATH_ST:INCPATHS}',
        '${DEFINES_ST:DEFINES}',
        '${CC_SRC_F}${SRC}',
        '${CC_TGT_F}${TGT[0].path_from(cwdx)}',     # We use CC_TGT_F because we are creating an object file
        '${CPPFLAGS}'])

    def post_run(self):
        check_deps(self)
        super(source_compile_task, self).post_run()
