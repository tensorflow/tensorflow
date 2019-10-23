#!/usr/bin/env python
# encoding: utf-8
# Anton Feldmann, 2012
# "Base for cabal"

from waflib import Task, Utils
from waflib.TaskGen import extension
from waflib.Utils import threading
from shutil import rmtree

lock = threading.Lock()
registering = False

def configure(self):
    self.find_program('cabal', var='CABAL')
    self.find_program('ghc-pkg', var='GHCPKG')
    pkgconfd = self.bldnode.abspath() + '/package.conf.d'
    self.env.PREFIX = self.bldnode.abspath() + '/dist'
    self.env.PKGCONFD = pkgconfd
    if self.root.find_node(pkgconfd + '/package.cache'):
        self.msg('Using existing package database', pkgconfd, color='CYAN')
    else:
        pkgdir = self.root.find_dir(pkgconfd)
        if pkgdir:
            self.msg('Deleting corrupt package database', pkgdir.abspath(), color ='RED')
            rmtree(pkgdir.abspath())
            pkgdir = None

        self.cmd_and_log(self.env.GHCPKG + ['init', pkgconfd])
        self.msg('Created package database', pkgconfd, color = 'YELLOW' if pkgdir else 'GREEN')

@extension('.cabal')
def process_cabal(self, node):
    out_dir_node = self.bld.root.find_dir(self.bld.out_dir)
    package_node = node.change_ext('.package')
    package_node = out_dir_node.find_or_declare(package_node.name)
    build_node   = node.parent.get_bld()
    build_path   = build_node.abspath()
    config_node  = build_node.find_or_declare('setup-config')
    inplace_node = build_node.find_or_declare('package.conf.inplace')

    config_task = self.create_task('cabal_configure', node)
    config_task.cwd = node.parent.abspath()
    config_task.depends_on = getattr(self, 'depends_on', '')
    config_task.build_path = build_path
    config_task.set_outputs(config_node)

    build_task = self.create_task('cabal_build', config_node)
    build_task.cwd = node.parent.abspath()
    build_task.build_path = build_path
    build_task.set_outputs(inplace_node)

    copy_task = self.create_task('cabal_copy', inplace_node)
    copy_task.cwd = node.parent.abspath()
    copy_task.depends_on = getattr(self, 'depends_on', '')
    copy_task.build_path = build_path

    last_task = copy_task
    task_list = [config_task, build_task, copy_task]

    if (getattr(self, 'register', False)):
        register_task = self.create_task('cabal_register', inplace_node)
        register_task.cwd = node.parent.abspath()
        register_task.set_run_after(copy_task)
        register_task.build_path = build_path

        pkgreg_task = self.create_task('ghcpkg_register', inplace_node)
        pkgreg_task.cwd = node.parent.abspath()
        pkgreg_task.set_run_after(register_task)
        pkgreg_task.build_path = build_path

        last_task = pkgreg_task
        task_list += [register_task, pkgreg_task]

    touch_task = self.create_task('cabal_touch', inplace_node)
    touch_task.set_run_after(last_task)
    touch_task.set_outputs(package_node)
    touch_task.build_path = build_path

    task_list += [touch_task]

    return task_list

def get_all_src_deps(node):
    hs_deps = node.ant_glob('**/*.hs')
    hsc_deps = node.ant_glob('**/*.hsc')
    lhs_deps = node.ant_glob('**/*.lhs')
    c_deps = node.ant_glob('**/*.c')
    cpp_deps = node.ant_glob('**/*.cpp')
    proto_deps = node.ant_glob('**/*.proto')
    return sum([hs_deps, hsc_deps, lhs_deps, c_deps, cpp_deps, proto_deps], [])

class Cabal(Task.Task):
    def scan(self):
        return (get_all_src_deps(self.generator.path), ())

class cabal_configure(Cabal):
    run_str = '${CABAL} configure -v0 --prefix=${PREFIX} --global --user --package-db=${PKGCONFD} --builddir=${tsk.build_path}'
    shell = True

    def scan(self):
        out_node = self.generator.bld.root.find_dir(self.generator.bld.out_dir)
        deps = [out_node.find_or_declare(dep).change_ext('.package') for dep in Utils.to_list(self.depends_on)]
        return (deps, ())

class cabal_build(Cabal):
    run_str = '${CABAL} build -v1 --builddir=${tsk.build_path}/'
    shell = True

class cabal_copy(Cabal):
    run_str = '${CABAL} copy -v0 --builddir=${tsk.build_path}'
    shell = True

class cabal_register(Cabal):
    run_str = '${CABAL} register -v0 --gen-pkg-config=${tsk.build_path}/pkg.config --builddir=${tsk.build_path}'
    shell = True

class ghcpkg_register(Cabal):
    run_str = '${GHCPKG} update -v0 --global --user --package-conf=${PKGCONFD} ${tsk.build_path}/pkg.config'
    shell = True

    def runnable_status(self):
        global lock, registering

        val = False 
        lock.acquire()
        val = registering
        lock.release()

        if val:
            return Task.ASK_LATER

        ret = Task.Task.runnable_status(self)
        if ret == Task.RUN_ME:
            lock.acquire()
            registering = True
            lock.release()

        return ret

    def post_run(self):
        global lock, registering

        lock.acquire()
        registering = False
        lock.release()

        return Task.Task.post_run(self)

class cabal_touch(Cabal):
    run_str = 'touch ${TGT}'

