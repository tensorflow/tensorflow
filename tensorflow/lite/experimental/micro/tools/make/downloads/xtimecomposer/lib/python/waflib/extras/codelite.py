#! /usr/bin/env python
# encoding: utf-8
# CodeLite Project
# Christian Klein (chrikle@berlios.de)
# Created: Jan 2012
# As templete for this file I used the msvs.py
# I hope this template will work proper

"""
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

"""
 

To add this tool to your project:
def options(conf):
        opt.load('codelite')

It can be a good idea to add the sync_exec tool too.

To generate solution files:
$ waf configure codelite

To customize the outputs, provide subclasses in your wscript files:

from waflib.extras import codelite
class vsnode_target(codelite.vsnode_target):
        def get_build_command(self, props):
                # likely to be required
                return "waf.bat build"
        def collect_source(self):
                # likely to be required
                ...
class codelite_bar(codelite.codelite_generator):
        def init(self):
                codelite.codelite_generator.init(self)
                self.vsnode_target = vsnode_target

The codelite class re-uses the same build() function for reading the targets (task generators),
you may therefore specify codelite settings on the context object:

def build(bld):
        bld.codelite_solution_name = 'foo.workspace'
        bld.waf_command = 'waf.bat'
        bld.projects_dir = bld.srcnode.make_node('')
        bld.projects_dir.mkdir()


ASSUMPTIONS:
* a project can be either a directory or a target, project files are written only for targets that have source files
* each project is a vcxproj file, therefore the project uuid needs only to be a hash of the absolute path
"""

import os, re, sys
import uuid # requires python 2.5
from waflib.Build import BuildContext
from waflib import Utils, TaskGen, Logs, Task, Context, Node, Options

HEADERS_GLOB = '**/(*.h|*.hpp|*.H|*.inl)'

PROJECT_TEMPLATE = r'''<?xml version="1.0" encoding="utf-8"?>
<CodeLite_Project Name="${project.name}" InternalType="Library">
  <Plugins>
    <Plugin Name="qmake">
      <![CDATA[00010001N0005Release000000000000]]>
    </Plugin>
  </Plugins>
  <Description/>
  <Dependencies/>
  <VirtualDirectory Name="src">
  ${for x in project.source}  
  ${if (project.get_key(x)=="sourcefile")}
  <File Name="${x.abspath()}"/>
  ${endif}
  ${endfor}  
  </VirtualDirectory>
  <VirtualDirectory Name="include">  
  ${for x in project.source}
  ${if (project.get_key(x)=="headerfile")}
  <File Name="${x.abspath()}"/>
  ${endif}
  ${endfor}
  </VirtualDirectory>  
  <Settings Type="Dynamic Library">
    <GlobalSettings>
      <Compiler Options="" C_Options="">
        <IncludePath Value="."/>
      </Compiler>
      <Linker Options="">
        <LibraryPath Value="."/>
      </Linker>
      <ResourceCompiler Options=""/>
    </GlobalSettings>
    <Configuration Name="Release" CompilerType="gnu gcc" ReleasegerType="GNU gdb Releaseger" Type="Dynamic Library" BuildCmpWithGlobalSettings="append" BuildLnkWithGlobalSettings="append" BuildResWithGlobalSettings="append">
      <Compiler Options="" C_Options="" Required="yes" PreCompiledHeader="" PCHInCommandLine="no" UseDifferentPCHFlags="no" PCHFlags="">
        <IncludePath Value="."/>
        <IncludePath Value="."/>
      </Compiler>
      <Linker Options="" Required="yes">
        <LibraryPath Value=""/>
      </Linker>
      <ResourceCompiler Options="" Required="no"/>
      <General OutputFile="${xml:project.build_properties[0].output_file}" IntermediateDirectory="" Command="" CommandArguments="" PauseExecWhenProcTerminates="yes"/>
      <Environment EnvVarSetName="&lt;Use Defaults&gt;" DbgSetName="&lt;Use Defaults&gt;">
        <![CDATA[]]>
      </Environment>
      <Releaseger IsRemote="no" RemoteHostName="" RemoteHostPort="" ReleasegerPath="">
        <PostConnectCommands/>
        <StartupCommands/>
      </Releaseger>
      <PreBuild/>
      <PostBuild/>
      <CustomBuild Enabled="yes">
        $b = project.build_properties[0]}
        <RebuildCommand>${xml:project.get_rebuild_command(project.build_properties[0])}</RebuildCommand>
        <CleanCommand>${xml:project.get_clean_command(project.build_properties[0])}</CleanCommand>
        <BuildCommand>${xml:project.get_build_command(project.build_properties[0])}</BuildCommand> 
        <Target Name="Install">${xml:project.get_install_command(project.build_properties[0])}</Target>
        <Target Name="Build and Install">${xml:project.get_build_and_install_command(project.build_properties[0])}</Target>        
        <Target Name="Build All">${xml:project.get_build_all_command(project.build_properties[0])}</Target>
        <Target Name="Rebuild All">${xml:project.get_rebuild_all_command(project.build_properties[0])}</Target>
        <Target Name="Clean All">${xml:project.get_clean_all_command(project.build_properties[0])}</Target>
        <Target Name="Build and Install All">${xml:project.get_build_and_install_all_command(project.build_properties[0])}</Target>
        <PreprocessFileCommand/>
        <SingleFileCommand/>
        <MakefileGenerationCommand/>
        <ThirdPartyToolName>None</ThirdPartyToolName>
        <WorkingDirectory/>
      </CustomBuild>
      <AdditionalRules>
        <CustomPostBuild/>
        <CustomPreBuild/>
      </AdditionalRules>
      <Completion>
        <ClangCmpFlags/>
        <ClangPP/>
        <SearchPaths/>
      </Completion>
    </Configuration>
    <Configuration Name="Release" CompilerType="gnu gcc" ReleasegerType="GNU gdb Releaseger" Type="" BuildCmpWithGlobalSettings="append" BuildLnkWithGlobalSettings="append" BuildResWithGlobalSettings="append">
      <Compiler Options="" C_Options="" Required="yes" PreCompiledHeader="" PCHInCommandLine="no" UseDifferentPCHFlags="no" PCHFlags="">
        <IncludePath Value="."/>
      </Compiler>
      <Linker Options="" Required="yes"/>
      <ResourceCompiler Options="" Required="no"/>
      <General OutputFile="" IntermediateDirectory="./Release" Command="" CommandArguments="" UseSeparateReleaseArgs="no" ReleaseArguments="" WorkingDirectory="$(IntermediateDirectory)" PauseExecWhenProcTerminates="yes"/>
      <Environment EnvVarSetName="&lt;Use Defaults&gt;" DbgSetName="&lt;Use Defaults&gt;">
        <![CDATA[
      
      
      
      ]]>
      </Environment>
      <Releaseger IsRemote="no" RemoteHostName="" RemoteHostPort="" ReleasegerPath="">
        <PostConnectCommands/>
        <StartupCommands/>
      </Releaseger>
      <PreBuild/>
      <PostBuild/>
      <CustomBuild Enabled="no">
        <RebuildCommand/>
        <CleanCommand/>
        <BuildCommand/>
        <PreprocessFileCommand/>
        <SingleFileCommand/>
        <MakefileGenerationCommand/>
        <ThirdPartyToolName/>
        <WorkingDirectory/>
      </CustomBuild>
      <AdditionalRules>
        <CustomPostBuild/>
        <CustomPreBuild/>
      </AdditionalRules>
      <Completion>
        <ClangCmpFlags/>
        <ClangPP/>
        <SearchPaths/>
      </Completion>
    </Configuration>
  </Settings>
</CodeLite_Project>'''




SOLUTION_TEMPLATE = '''<?xml version="1.0" encoding="utf-8"?>
<CodeLite_Workspace Name="${getattr(project, 'codelite_solution_name', None)[:-10]}" Database="./${getattr(project, 'codelite_solution_name', None)[:-10]}.tags">
${for p in project.all_projects}
  <Project Name = "${p.name}" Path = "${p.title}" Active="No"/>
${endfor}
  <BuildMatrix>
    <WorkspaceConfiguration Name="Release" Selected="yes">
${for p in project.all_projects}
      <Project Name="${p.name}" ConfigName="Release"/>        
${endfor}
    </WorkspaceConfiguration>        
  </BuildMatrix>
</CodeLite_Workspace>'''



COMPILE_TEMPLATE = '''def f(project):
        lst = []
        def xml_escape(value):
                return value.replace("&", "&amp;").replace('"', "&quot;").replace("'", "&apos;").replace("<", "&lt;").replace(">", "&gt;")

        %s

        #f = open('cmd.txt', 'w')
        #f.write(str(lst))
        #f.close()
        return ''.join(lst)
'''
reg_act = re.compile(r"(?P<backslash>\\)|(?P<dollar>\$\$)|(?P<subst>\$\{(?P<code>[^}]*?)\})", re.M)
def compile_template(line):
        """
        Compile a template expression into a python function (like jsps, but way shorter)
        """
        extr = []
        def repl(match):
                g = match.group
                if g('dollar'):
                        return "$"
                elif g('backslash'):
                        return "\\"
                elif g('subst'):
                        extr.append(g('code'))
                        return "<<|@|>>"
                return None

        line2 = reg_act.sub(repl, line)
        params = line2.split('<<|@|>>')
        assert(extr)


        indent = 0
        buf = []
        app = buf.append

        def app(txt):
                buf.append(indent * '\t' + txt)

        for x in range(len(extr)):
                if params[x]:
                        app("lst.append(%r)" % params[x])

                f = extr[x]
                if f.startswith(('if', 'for')):
                        app(f + ':')
                        indent += 1
                elif f.startswith('py:'):
                        app(f[3:])
                elif f.startswith(('endif', 'endfor')):
                        indent -= 1
                elif f.startswith(('else', 'elif')):
                        indent -= 1
                        app(f + ':')
                        indent += 1
                elif f.startswith('xml:'):
                        app('lst.append(xml_escape(%s))' % f[4:])
                else:
                        #app('lst.append((%s) or "cannot find %s")' % (f, f))
                        app('lst.append(%s)' % f)

        if extr:
                if params[-1]:
                        app("lst.append(%r)" % params[-1])

        fun = COMPILE_TEMPLATE % "\n\t".join(buf)
        #print(fun)
        return Task.funex(fun)


re_blank = re.compile('(\n|\r|\\s)*\n', re.M)
def rm_blank_lines(txt):
        txt = re_blank.sub('\r\n', txt)
        return txt

BOM = '\xef\xbb\xbf'
try:
        BOM = bytes(BOM, 'latin-1') # python 3
except (TypeError, NameError):
        pass

def stealth_write(self, data, flags='wb'):
        try:
                unicode
        except NameError:
                data = data.encode('utf-8') # python 3
        else:
                data = data.decode(sys.getfilesystemencoding(), 'replace')
                data = data.encode('utf-8')

        if self.name.endswith('.project'):
                data = BOM + data

        try:
                txt = self.read(flags='rb')
                if txt != data:
                        raise ValueError('must write')
        except (IOError, ValueError):
                self.write(data, flags=flags)
        else:
                Logs.debug('codelite: skipping %r', self)
Node.Node.stealth_write = stealth_write

re_quote = re.compile("[^a-zA-Z0-9-]")
def quote(s):
        return re_quote.sub("_", s)

def xml_escape(value):
        return value.replace("&", "&amp;").replace('"', "&quot;").replace("'", "&apos;").replace("<", "&lt;").replace(">", "&gt;")

def make_uuid(v, prefix = None):
        """
        simple utility function
        """
        if isinstance(v, dict):
                keys = list(v.keys())
                keys.sort()
                tmp = str([(k, v[k]) for k in keys])
        else:
                tmp = str(v)
        d = Utils.md5(tmp.encode()).hexdigest().upper()
        if prefix:
                d = '%s%s' % (prefix, d[8:])
        gid = uuid.UUID(d, version = 4)
        return str(gid).upper()

def diff(node, fromnode):
        # difference between two nodes, but with "(..)" instead of ".."
        c1 = node
        c2 = fromnode

        c1h = c1.height()
        c2h = c2.height()

        lst = []
        up = 0

        while c1h > c2h:
                lst.append(c1.name)
                c1 = c1.parent
                c1h -= 1

        while c2h > c1h:
                up += 1
                c2 = c2.parent
                c2h -= 1

        while id(c1) != id(c2):
                lst.append(c1.name)
                up += 1

                c1 = c1.parent
                c2 = c2.parent

        for i in range(up):
                lst.append('(..)')
        lst.reverse()
        return tuple(lst)

class build_property(object):
        pass

class vsnode(object):
        """
        Abstract class representing visual studio elements
        We assume that all visual studio nodes have a uuid and a parent
        """
        def __init__(self, ctx):
                self.ctx = ctx # codelite context
                self.name = '' # string, mandatory
                self.vspath = '' # path in visual studio (name for dirs, absolute path for projects)
                self.uuid = '' # string, mandatory
                self.parent = None # parent node for visual studio nesting

        def get_waf(self):
                """
                Override in subclasses...
                """
                return '%s/%s' % (self.ctx.srcnode.abspath(), getattr(self.ctx, 'waf_command', 'waf'))

        def ptype(self):
                """
                Return a special uuid for projects written in the solution file
                """
                pass

        def write(self):
                """
                Write the project file, by default, do nothing
                """
                pass

        def make_uuid(self, val):
                """
                Alias for creating uuid values easily (the templates cannot access global variables)
                """
                return make_uuid(val)

class vsnode_vsdir(vsnode):
        """
        Nodes representing visual studio folders (which do not match the filesystem tree!)
        """
        VS_GUID_SOLUTIONFOLDER = "2150E333-8FDC-42A3-9474-1A3956D46DE8"
        def __init__(self, ctx, uuid, name, vspath=''):
                vsnode.__init__(self, ctx)
                self.title = self.name = name
                self.uuid = uuid
                self.vspath = vspath or name

        def ptype(self):
                return self.VS_GUID_SOLUTIONFOLDER

class vsnode_project(vsnode):
        """
        Abstract class representing visual studio project elements
        A project is assumed to be writable, and has a node representing the file to write to
        """
        VS_GUID_VCPROJ = "8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942"
        def ptype(self):
                return self.VS_GUID_VCPROJ

        def __init__(self, ctx, node):
                vsnode.__init__(self, ctx)
                self.path = node
                self.uuid = make_uuid(node.abspath())
                self.name = node.name
                self.title = self.path.abspath()
                self.source = [] # list of node objects
                self.build_properties = [] # list of properties (nmake commands, output dir, etc)

        def dirs(self):
                """
                Get the list of parent folders of the source files (header files included)
                for writing the filters
                """
                lst = []
                def add(x):
                        if x.height() > self.tg.path.height() and x not in lst:
                                lst.append(x)
                                add(x.parent)
                for x in self.source:
                        add(x.parent)
                return lst

        def write(self):
                Logs.debug('codelite: creating %r', self.path)
                #print "self.name:",self.name

                # first write the project file
                template1 = compile_template(PROJECT_TEMPLATE)
                proj_str = template1(self)
                proj_str = rm_blank_lines(proj_str)
                self.path.stealth_write(proj_str)

                # then write the filter
                #template2 = compile_template(FILTER_TEMPLATE)
                #filter_str = template2(self)
                #filter_str = rm_blank_lines(filter_str)
                #tmp = self.path.parent.make_node(self.path.name + '.filters')
                #tmp.stealth_write(filter_str)

        def get_key(self, node):
                """
                required for writing the source files
                """
                name = node.name
                if name.endswith(('.cpp', '.c')):
                        return 'sourcefile'
                return 'headerfile'

        def collect_properties(self):
                """
                Returns a list of triplet (configuration, platform, output_directory)
                """
                ret = []
                for c in self.ctx.configurations:
                        for p in self.ctx.platforms:
                                x = build_property()
                                x.outdir = ''

                                x.configuration = c
                                x.platform = p

                                x.preprocessor_definitions = ''
                                x.includes_search_path = ''

                                # can specify "deploy_dir" too
                                ret.append(x)
                self.build_properties = ret

        def get_build_params(self, props):
                opt = ''
                return (self.get_waf(), opt)

        def get_build_command(self, props):
                return "%s build %s" % self.get_build_params(props)

        def get_clean_command(self, props):
                return "%s clean %s" % self.get_build_params(props)

        def get_rebuild_command(self, props):
                return "%s clean build %s" % self.get_build_params(props)
                
        def get_install_command(self, props):
                return "%s install %s" % self.get_build_params(props)
        def get_build_and_install_command(self, props):
                return "%s build install %s" % self.get_build_params(props)
                
        def get_build_and_install_all_command(self, props):
                return "%s build install" % self.get_build_params(props)[0]
                
        def get_clean_all_command(self, props):
                return "%s clean" % self.get_build_params(props)[0]
        
        def get_build_all_command(self, props):
                return "%s build" % self.get_build_params(props)[0]
                
        def get_rebuild_all_command(self, props):
                return "%s clean build" % self.get_build_params(props)[0]

        def get_filter_name(self, node):
                lst = diff(node, self.tg.path)
                return '\\'.join(lst) or '.'

class vsnode_alias(vsnode_project):
        def __init__(self, ctx, node, name):
                vsnode_project.__init__(self, ctx, node)
                self.name = name
                self.output_file = ''

class vsnode_build_all(vsnode_alias):
        """
        Fake target used to emulate the behaviour of "make all" (starting one process by target is slow)
        This is the only alias enabled by default
        """
        def __init__(self, ctx, node, name='build_all_projects'):
                vsnode_alias.__init__(self, ctx, node, name)
                self.is_active = True

class vsnode_install_all(vsnode_alias):
        """
        Fake target used to emulate the behaviour of "make install"
        """
        def __init__(self, ctx, node, name='install_all_projects'):
                vsnode_alias.__init__(self, ctx, node, name)

        def get_build_command(self, props):
                return "%s build install %s" % self.get_build_params(props)

        def get_clean_command(self, props):
                return "%s clean %s" % self.get_build_params(props)

        def get_rebuild_command(self, props):
                return "%s clean build install %s" % self.get_build_params(props)

class vsnode_project_view(vsnode_alias):
        """
        Fake target used to emulate a file system view
        """
        def __init__(self, ctx, node, name='project_view'):
                vsnode_alias.__init__(self, ctx, node, name)
                self.tg = self.ctx() # fake one, cannot remove
                self.exclude_files = Node.exclude_regs + '''
waf-2*
waf3-2*/**
.waf-2*
.waf3-2*/**
**/*.sdf
**/*.suo
**/*.ncb
**/%s
                ''' % Options.lockfile

        def collect_source(self):
                # this is likely to be slow
                self.source = self.ctx.srcnode.ant_glob('**', excl=self.exclude_files)

        def get_build_command(self, props):
                params = self.get_build_params(props) + (self.ctx.cmd,)
                return "%s %s %s" % params

        def get_clean_command(self, props):
                return ""

        def get_rebuild_command(self, props):
                return self.get_build_command(props)

class vsnode_target(vsnode_project):
        """
        CodeLite project representing a targets (programs, libraries, etc) and bound
        to a task generator
        """
        def __init__(self, ctx, tg):
                """
                A project is more or less equivalent to a file/folder
                """
                base = getattr(ctx, 'projects_dir', None) or tg.path
                node = base.make_node(quote(tg.name) + ctx.project_extension) # the project file as a Node
                vsnode_project.__init__(self, ctx, node)
                self.name = quote(tg.name)
                self.tg     = tg  # task generator

        def get_build_params(self, props):
                """
                Override the default to add the target name
                """
                opt = ''
                if getattr(self, 'tg', None):
                        opt += " --targets=%s" % self.tg.name
                return (self.get_waf(), opt)

        def collect_source(self):
                tg = self.tg
                source_files = tg.to_nodes(getattr(tg, 'source', []))
                include_dirs = Utils.to_list(getattr(tg, 'codelite_includes', []))
                include_files = []
                for x in include_dirs:
                        if isinstance(x, str):
                                x = tg.path.find_node(x)
                        if x:
                                lst = [y for y in x.ant_glob(HEADERS_GLOB, flat=False)]
                                include_files.extend(lst)

                # remove duplicates
                self.source.extend(list(set(source_files + include_files)))
                self.source.sort(key=lambda x: x.abspath())

        def collect_properties(self):
                """
                CodeLite projects are associated with platforms and configurations (for building especially)
                """
                super(vsnode_target, self).collect_properties()
                for x in self.build_properties:
                        x.outdir = self.path.parent.abspath()
                        x.preprocessor_definitions = ''
                        x.includes_search_path = ''

                        try:
                                tsk = self.tg.link_task
                        except AttributeError:
                                pass
                        else:                                
                                x.output_file = tsk.outputs[0].abspath()
                                x.preprocessor_definitions = ';'.join(tsk.env.DEFINES)
                                x.includes_search_path = ';'.join(self.tg.env.INCPATHS)

class codelite_generator(BuildContext):
        '''generates a CodeLite workspace'''
        cmd = 'codelite'
        fun = 'build'

        def init(self):
                """
                Some data that needs to be present
                """
                if not getattr(self, 'configurations', None):
                        self.configurations = ['Release'] # LocalRelease, RemoteDebug, etc
                if not getattr(self, 'platforms', None):
                        self.platforms = ['Win32']
                if not getattr(self, 'all_projects', None):
                        self.all_projects = []
                if not getattr(self, 'project_extension', None):
                        self.project_extension = '.project'
                if not getattr(self, 'projects_dir', None):
                        self.projects_dir = self.srcnode.make_node('')
                        self.projects_dir.mkdir()

                # bind the classes to the object, so that subclass can provide custom generators
                if not getattr(self, 'vsnode_vsdir', None):
                        self.vsnode_vsdir = vsnode_vsdir
                if not getattr(self, 'vsnode_target', None):
                        self.vsnode_target = vsnode_target
                if not getattr(self, 'vsnode_build_all', None):
                        self.vsnode_build_all = vsnode_build_all
                if not getattr(self, 'vsnode_install_all', None):
                        self.vsnode_install_all = vsnode_install_all
                if not getattr(self, 'vsnode_project_view', None):
                        self.vsnode_project_view = vsnode_project_view

                self.numver = '11.00'
                self.vsver  = '2010'

        def execute(self):
                """
                Entry point
                """
                self.restore()
                if not self.all_envs:
                        self.load_envs()
                self.recurse([self.run_dir])

                # user initialization
                self.init()

                # two phases for creating the solution
                self.collect_projects() # add project objects into "self.all_projects"
                self.write_files() # write the corresponding project and solution files

        def collect_projects(self):
                """
                Fill the list self.all_projects with project objects
                Fill the list of build targets
                """
                self.collect_targets()
                #self.add_aliases()
                #self.collect_dirs()
                default_project = getattr(self, 'default_project', None)
                def sortfun(x):
                        if x.name == default_project:
                                return ''
                        return getattr(x, 'path', None) and x.path.abspath() or x.name
                self.all_projects.sort(key=sortfun)

        def write_files(self):
                """
                Write the project and solution files from the data collected
                so far. It is unlikely that you will want to change this
                """
                for p in self.all_projects:
                        p.write()

                # and finally write the solution file
                node = self.get_solution_node()
                node.parent.mkdir()
                Logs.warn('Creating %r', node)
                #a = dir(self.root)
                #for b in a:
                #        print b
                #print self.group_names
                #print "Hallo2:   ",self.root.listdir()
                #print getattr(self, 'codelite_solution_name', None)
                template1 = compile_template(SOLUTION_TEMPLATE)
                sln_str = template1(self)
                sln_str = rm_blank_lines(sln_str)
                node.stealth_write(sln_str)

        def get_solution_node(self):
                """
                The solution filename is required when writing the .vcproj files
                return self.solution_node and if it does not exist, make one
                """
                try:
                        return self.solution_node
                except:
                        pass

                codelite_solution_name = getattr(self, 'codelite_solution_name', None)
                if not codelite_solution_name:
                        codelite_solution_name = getattr(Context.g_module, Context.APPNAME, 'project') + '.workspace'
                        setattr(self, 'codelite_solution_name', codelite_solution_name)
                if os.path.isabs(codelite_solution_name):
                        self.solution_node = self.root.make_node(codelite_solution_name)
                else:
                        self.solution_node = self.srcnode.make_node(codelite_solution_name)
                return self.solution_node

        def project_configurations(self):
                """
                Helper that returns all the pairs (config,platform)
                """
                ret = []
                for c in self.configurations:
                        for p in self.platforms:
                                ret.append((c, p))
                return ret

        def collect_targets(self):
                """
                Process the list of task generators
                """
                for g in self.groups:
                        for tg in g:
                                if not isinstance(tg, TaskGen.task_gen):
                                        continue

                                if not hasattr(tg, 'codelite_includes'):
                                        tg.codelite_includes = tg.to_list(getattr(tg, 'includes', [])) + tg.to_list(getattr(tg, 'export_includes', []))
                                tg.post()
                                if not getattr(tg, 'link_task', None):
                                        continue

                                p = self.vsnode_target(self, tg)
                                p.collect_source() # delegate this processing
                                p.collect_properties()                               
                                self.all_projects.append(p)

        def add_aliases(self):
                """
                Add a specific target that emulates the "make all" necessary for Visual studio when pressing F7
                We also add an alias for "make install" (disabled by default)
                """
                base = getattr(self, 'projects_dir', None) or self.tg.path

                node_project = base.make_node('build_all_projects' + self.project_extension) # Node
                p_build = self.vsnode_build_all(self, node_project)
                p_build.collect_properties()
                self.all_projects.append(p_build)

                node_project = base.make_node('install_all_projects' + self.project_extension) # Node
                p_install = self.vsnode_install_all(self, node_project)
                p_install.collect_properties()
                self.all_projects.append(p_install)

                node_project = base.make_node('project_view' + self.project_extension) # Node
                p_view = self.vsnode_project_view(self, node_project)
                p_view.collect_source()
                p_view.collect_properties()
                self.all_projects.append(p_view)

                n = self.vsnode_vsdir(self, make_uuid(self.srcnode.abspath() + 'build_aliases'), "build_aliases")
                p_build.parent = p_install.parent = p_view.parent = n
                self.all_projects.append(n)

        def collect_dirs(self):
                """
                Create the folder structure in the CodeLite project view
                """
                seen = {}
                def make_parents(proj):
                        # look at a project, try to make a parent
                        if getattr(proj, 'parent', None):
                                # aliases already have parents
                                return
                        x = proj.iter_path
                        if x in seen:
                                proj.parent = seen[x]
                                return

                        # There is not vsnode_vsdir for x.
                        # So create a project representing the folder "x"
                        n = proj.parent = seen[x] = self.vsnode_vsdir(self, make_uuid(x.abspath()), x.name)
                        n.iter_path = x.parent
                        self.all_projects.append(n)

                        # recurse up to the project directory
                        if x.height() > self.srcnode.height() + 1:
                                make_parents(n)

                for p in self.all_projects[:]: # iterate over a copy of all projects
                        if not getattr(p, 'tg', None):
                                # but only projects that have a task generator
                                continue

                        # make a folder for each task generator
                        p.iter_path = p.tg.path
                        make_parents(p)

