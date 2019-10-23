#! /usr/bin/env python
# per rosengren 2011

from os import environ, path
from waflib import TaskGen, Utils

def options(opt):
	grp = opt.add_option_group('Oracle ProC Options')
	grp.add_option('--oracle_home', action='store', default=environ.get('PROC_ORACLE'), help='Path to Oracle installation home (has bin/lib)')
	grp.add_option('--tns_admin', action='store', default=environ.get('TNS_ADMIN'), help='Directory containing server list (TNS_NAMES.ORA)')
	grp.add_option('--connection', action='store', default='dummy-user/dummy-password@dummy-server', help='Format: user/password@server')

def configure(cnf):
	env = cnf.env
	if not env.PROC_ORACLE:
		env.PROC_ORACLE = cnf.options.oracle_home
	if not env.PROC_TNS_ADMIN:
		env.PROC_TNS_ADMIN = cnf.options.tns_admin
	if not env.PROC_CONNECTION:
		env.PROC_CONNECTION = cnf.options.connection
	cnf.find_program('proc', var='PROC', path_list=env.PROC_ORACLE + path.sep + 'bin')

def proc(tsk):
	env = tsk.env
	gen = tsk.generator
	inc_nodes = gen.to_incnodes(Utils.to_list(getattr(gen,'includes',[])) + env['INCLUDES'])

	cmd = (
		[env.PROC] +
		['SQLCHECK=SEMANTICS'] +
		(['SYS_INCLUDE=(' + ','.join(env.PROC_INCLUDES) + ')']
			if env.PROC_INCLUDES else []) +
		['INCLUDE=(' + ','.join(
			[i.bldpath() for i in inc_nodes]
		) + ')'] +
		['userid=' + env.PROC_CONNECTION] +
		['INAME=' + tsk.inputs[0].bldpath()] +
		['ONAME=' + tsk.outputs[0].bldpath()]
	)
	exec_env = {
		'ORACLE_HOME': env.PROC_ORACLE,
		'LD_LIBRARY_PATH': env.PROC_ORACLE + path.sep + 'lib',
	}
	if env.PROC_TNS_ADMIN:
		exec_env['TNS_ADMIN'] = env.PROC_TNS_ADMIN
	return tsk.exec_command(cmd, env=exec_env)

TaskGen.declare_chain(
	name = 'proc',
	rule = proc,
	ext_in = '.pc',
	ext_out = '.c',
)

