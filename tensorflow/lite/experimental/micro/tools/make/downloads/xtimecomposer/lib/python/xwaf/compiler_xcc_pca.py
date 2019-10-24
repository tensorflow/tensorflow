import os
from waflib import Task, Logs
from waflib.Tools import c
from waflib.TaskGen import taskgen_method
from waflib.TaskGen import after_method, before_method, feature, task_gen

from xwaf.compiler_xcc import check_deps

def options(opt):
    opt.add_option('--disable_pca', action='store_true', default=False, help='Set to disable post compilation analysis')

def configure(conf):
    conf.load('xwaf.compiler_xcc')
    if conf.options.disable_pca:
        conf.env.do_pca = False
    else:
        conf.env.do_pca = True

###################
# TaskGen object methods
###################

@taskgen_method
def create_analysed_task(self, name, node):
    #out = '%s.%d.xml' % (node.name, self.idx)
    fname = node.name
    if fname.endswith('.o'):
        fname = fname[:-2]
    out = '%s.pca.xml' % fname
    task = self.create_task(name, node, node.parent.find_or_declare(out))
    try:
        self.analysed_tasks.append(task)
    except AttributeError:
        self.analysed_tasks = [task]
    return task

@taskgen_method
def create_postcompiled_task(self, name, src_node, precompile_node):
    task = self.create_compiled_task(name, src_node)
    task.inputs.append(precompile_node)
    return task

###################
# TaskGen methods
###################

def hacked_add_objects_from_tgen(self, tg):
    """
    Add the objects from the depending compiled tasks as link task inputs.

    Some objects are filtered: for instance, .pdb files are added
    to the compiled tasks but not to the link tasks (to avoid errors)
    PRIVATE INTERNAL USE ONLY
    """
    try:
        link_task = self.link_task
    except AttributeError:
        pass
    else:
        for tsk in getattr(tg, 'compiled_tasks', []):
            for x in tsk.outputs:
                if self.accept_node_to_link(x):
                    if x not in link_task.inputs:
                        link_task.inputs.append(x)

def hacked_process_use(self):
    """
    Process the ``use`` attribute which contains a list of task generator names::

        def build(bld):
            bld.shlib(source='a.c', target='lib1')
            bld.program(source='main.c', target='app', use='lib1')

    See :py:func:`waflib.Tools.ccroot.use_rec`.
    """

    use_not = self.tmp_use_not = set()
    self.tmp_use_seen = [] # we would like an ordered set
    use_prec = self.tmp_use_prec = {}
    self.uselib = self.to_list(getattr(self, 'uselib', []))
    names = self.to_list(getattr(self, 'use', []))

    for x in names:
        self.use_rec(x)

    for x in use_not:
        if x in use_prec:
            del use_prec[x]

    # topological sort
    out = self.tmp_use_sorted = []
    tmp = []
    for x in self.tmp_use_seen:
        for k in use_prec.values():
            if x in k:
                break
        else:
            tmp.append(x)

    while tmp:
        e = tmp.pop()
        out.append(e)
        try:
            nlst = use_prec[e]
        except KeyError:
            pass
        else:
            del use_prec[e]
            for x in nlst:
                for y in use_prec:
                    if x in use_prec[y]:
                        break
                else:
                    tmp.append(x)
    if use_prec:
        raise Errors.WafError('Cycle detected in the use processing %r' % use_prec)
    out.reverse()

    link_task = getattr(self, 'link_task', None)
    for x in out:
        y = self.bld.get_tgen_by_name(x)
        var = y.tmp_use_var
        if var and link_task:
            if self.env.SKIP_STLIB_LINK_DEPS and isinstance(link_task, stlink_task):
                # If the skip_stlib_link_deps feature is enabled then we should
                # avoid adding lib deps to the stlink_task instance.
                pass
            elif var == 'LIB' or y.tmp_use_stlib or x in names:
                self.env.append_value(var, [y.target[y.target.rfind(os.sep) + 1:]])
                self.link_task.dep_nodes.extend(y.link_task.outputs)
                tmp_path = y.link_task.outputs[0].parent.path_from(self.get_cwd())
                self.env.append_unique(var + 'PATH', [tmp_path])
        else:
            if y.tmp_use_objects:
                hacked_add_objects_from_tgen(self, y)
    
@feature('cprogram')
@after_method('process_source')
@before_method('apply_link')
def apply_pca_link(self):
    if not self.bld.env.do_pca:
        return

    # We now combine the precompile outputs into single xml file

    xmls = [t.outputs[0] for t in getattr(self, 'analysed_tasks', [])]

    target = self.path.find_or_declare('pca.xml')
    self.pca_link_task = self.create_task('pca_link_task', xmls, target)

    # The xc files associated with the precompile outputs still need compiling (but they are now able to make use of
    # the xml file). We create the compile tasks here

    for tsk in getattr(self, 'analysed_tasks', []):
        if tsk.inputs[0].name.endswith('.xc'):
            self.create_postcompiled_task('xc_compile_task', tsk.inputs[0], target)

    # We now have a pca link target, so we can create all the compiled tasks that were dependent on this:

    for use in self.bld.path.dep_modules.keys():
        tg = self.bld.get_tgen_by_name(use)

        for tsk in getattr(tg, 'analysed_tasks', []):
            self.pca_link_task.inputs.append(tsk.outputs[0])
            if tsk.inputs[0].name.endswith('.xc'):
                t = tg.create_postcompiled_task('xc_compile_task', tsk.inputs[0], target)

    # Previous calls to process_use for the modules have failed to add the objects arising from compilation of
    # the xc source code. We need to call process_use again to ensure that the objects are picked up
    for use in self.bld.path.dep_modules.keys():
        tg = self.bld.get_tgen_by_name(use)

        hacked_process_use(tg)

###################
# Tasks
###################

# This task is inspired by class link_task in waflib.Tools.ccroot...
class pca_link_task(Task.Task):
    run_str = "${XPCA} ${TGT} ${SRC}"

    color='YELLOW'

    weight = 3

    def keyword(self):
        return "PCA Linking"

    # Without this override, waf lists all the files being linked,
    # which can be a very long list
    def __str__(self):
        node = self.outputs[0]
        return node.path_from(node.ctx.launch_node())

class object_analyse_task(Task.Task):
    run_str = ' '.join([
        '${CC}',
        '${ARCH_ST:ARCH}',
        '-pre-compilation-analysis',
        # '${CFLAGS}',                              # We are not running the pre-processor, so it's not neccessary to send the CFLAGS...
        # '${tsk.inputs[0].xccflags}',              # ...or these...
        '${FRAMEWORKPATH_ST:FRAMEWORKPATH}',
        # '${CPPPATH_ST:INCPATHS}',                 # ...or these...
        # '${DEFINES_ST:DEFINES}',                  # ...or these.
        '${CC_SRC_F}${SRC}',
        '${CCLNK_TGT_F}${TGT[0].path_from(cwdx)}',  # We use CCLNK_TGT_F because we are not creating an object file
        '${CPPFLAGS}'])

    # Without this, waf reports "Compiling"
    def keyword(self):
        return "Analysing"

class xc_analyse_task(c.c):
    run_str = ' '.join([
        '${CC}',
        '${ARCH_ST:ARCH}',
        '-pre-compilation-analysis',
        '${CFLAGS}',
        '${tsk.inputs[0].xccflags}',
        '${FRAMEWORKPATH_ST:FRAMEWORKPATH}',
        '${CPPPATH_ST:INCPATHS}',
        '${DEFINES_ST:DEFINES}',
        '${CC_SRC_F}${SRC}',
        '${CCLNK_TGT_F}${TGT[0].path_from(cwdx)}',  # We use CCLNK_TGT_F because we are not creating an object file
        '${CPPFLAGS}'])

    def keyword(self):
        return "Analysing"

    def post_run(self):
        check_deps(self)
        super(xc_analyse_task, self).post_run()

class xc_compile_task(Task.Task):
    run_str = ' '.join([
        '${CC}',
        '${ARCH_ST:ARCH}',
        '${CFLAGS}',
        '${tsk.inputs[0].xccflags}',
        '-Xcompiler-xc', '-analysis',
        '-Xcompiler-xc', '${SRC[1].path_from(cwdx)}',
        '${FRAMEWORKPATH_ST:FRAMEWORKPATH}',
        '${CPPPATH_ST:INCPATHS}',
        '${DEFINES_ST:DEFINES}',
        '${CC_SRC_F}${SRC[0].path_from(cwdx)}',
        '${CC_TGT_F}${TGT[0].path_from(cwdx)}',     # We use CC_TGT_F because we are creating an object file
        '${CPPFLAGS}'])

    def keyword(self):
        return "Compiling"

    def __str__(self):
        node = self.inputs[0]
        return node.path_from(node.ctx.launch_node())
