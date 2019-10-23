#!/usr/bin/env python
# encoding: utf-8
# Ali Sabil, 2007

from waflib import TaskGen

TaskGen.declare_chain(
	name = 'gob2',
	rule = '${GOB2} -o ${TGT[0].bld_dir()} ${GOB2FLAGS} ${SRC}',
	ext_in = '.gob',
	ext_out = '.c'
)

def configure(conf):
	conf.find_program('gob2', var='GOB2')
	conf.env['GOB2FLAGS'] = ''

