"""Support for Sphinx documentation

This is a wrapper for sphinx-build program. Please note that sphinx-build supports only one output format which can
passed to build via sphinx_output_format attribute. The default output format is html.

Example wscript:

def configure(cnf):
    conf.load('sphinx')

def build(bld):
    bld(
        features='sphinx',
        sphinx_source='sources',  # path to source directory
        sphinx_options='-a -v',  # sphinx-build program additional options
        sphinx_output_format='man'  # output format of sphinx documentation
        )

"""

from waflib.Node import Node
from waflib import Utils
from waflib.Task import Task
from waflib.TaskGen import feature, after_method


def configure(cnf):
    """Check if sphinx-build program is available and loads gnu_dirs tool."""
    cnf.find_program('sphinx-build', var='SPHINX_BUILD', mandatory=False)
    cnf.load('gnu_dirs')


@feature('sphinx')
def build_sphinx(self):
    """Builds sphinx sources.
    """
    if not self.env.SPHINX_BUILD:
        self.bld.fatal('Program SPHINX_BUILD not defined.')
    if not getattr(self, 'sphinx_source', None):
        self.bld.fatal('Attribute sphinx_source not defined.')
    if not isinstance(self.sphinx_source, Node):
        self.sphinx_source = self.path.find_node(self.sphinx_source)
    if not self.sphinx_source:
        self.bld.fatal('Can\'t find sphinx_source: %r' % self.sphinx_source)

    Utils.def_attrs(self, sphinx_output_format='html')
    self.env.SPHINX_OUTPUT_FORMAT = self.sphinx_output_format
    self.env.SPHINX_OPTIONS = getattr(self, 'sphinx_options', [])

    for source_file in self.sphinx_source.ant_glob('**/*'):
        self.bld.add_manual_dependency(self.sphinx_source, source_file)

    sphinx_build_task = self.create_task('SphinxBuildingTask')
    sphinx_build_task.set_inputs(self.sphinx_source)
    sphinx_build_task.set_outputs(self.path.get_bld())

    # the sphinx-build results are in <build + output_format> directory
    sphinx_output_directory = self.path.get_bld().make_node(self.env.SPHINX_OUTPUT_FORMAT)
    sphinx_output_directory.mkdir()
    Utils.def_attrs(self, install_path=get_install_path(self))
    self.add_install_files(install_to=self.install_path,
                           install_from=sphinx_output_directory.ant_glob('**/*'),
                           cwd=sphinx_output_directory,
                           relative_trick=True)


def get_install_path(tg):
    if tg.env.SPHINX_OUTPUT_FORMAT == 'man':
        return tg.env.MANDIR
    elif tg.env.SPHINX_OUTPUT_FORMAT == 'info':
        return tg.env.INFODIR
    else:
        return tg.env.DOCDIR


class SphinxBuildingTask(Task):
    color = 'BOLD'
    run_str = '${SPHINX_BUILD} -M ${SPHINX_OUTPUT_FORMAT} ${SRC} ${TGT} ${SPHINX_OPTIONS}'

    def keyword(self):
        return 'Compiling (%s)' % self.env.SPHINX_OUTPUT_FORMAT
