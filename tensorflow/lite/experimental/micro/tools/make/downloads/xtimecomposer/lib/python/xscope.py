#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright XMOS Limited - 2018
"""

import os
from collections import defaultdict
import ctypes
import ctypes.util
import platform
import sys
import time

"""
 Function prototypes to match the c functions defined in xscope_endpoint.h
"""
PRINT_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.c_ulonglong,     # timestamp
    ctypes.c_uint,          # length
    ctypes.c_char_p)        # data

RECORD_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.c_uint,          # id
    ctypes.c_ulonglong,     # timestamp
    ctypes.c_uint,          # length
    ctypes.c_ulonglong,     # dataval
    ctypes.c_char_p)        # databytes

REGISTER_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.c_uint,          # id
    ctypes.c_uint,          # type
    ctypes.c_uint,          # r
    ctypes.c_uint,          # g
    ctypes.c_uint,          # b
    ctypes.c_char_p,        # name
    ctypes.c_char_p,        # unit
    ctypes.c_uint,          # data_type
    ctypes.c_char_p)        # data_name

STATS_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.c_uint,          # id
    ctypes.c_ulonglong)     # average

class Endpoint(object):
    """Python xSCOPE endpoint wrapper.

    Example:

        def my_callback(timestamp, probe, value):
            print '{} {} {}'.format(timestamp, probe, value)

        ep = Endpoint()

        try:
            if ep.connect('localhost', '12345'):
                print "Failed to connect"
            else:
                ep.consume(callback, 'my_probe")
                while(True):
                    pass

        except KeyboardInterrupt:
            ep.disconnect()
    """
    def __init__(self):
        self._probe_info = {}  # probe id to probe info lookup.
                               # probe_info includes name, units, data type, etc...
        self._consumers = defaultdict(set) # probe name -> callbacks lookup
                                           #NOTE: The consumers must be looked up by name and not id because
                                           #      they can be specified before the probe_info is defined

        tool_path = os.environ.get('XMOS_TOOL_PATH')
        ps = platform.system()
        if ps == 'Windows':
            lib_path = os.path.join(tool_path, 'lib', 'xscope_endpoint.dll')
        else:  # Darwin (aka MacOS) or Linux
            lib_path = os.path.join(tool_path, 'lib', 'xscope_endpoint.so')
        self.lib_xscope = ctypes.CDLL(lib_path)

        # create callbacks
        self._print_cb = self._print_callback_func()
        self.lib_xscope.xscope_ep_set_print_cb(self._print_cb)

        self._record_cb = self._record_callback_func()
        self.lib_xscope.xscope_ep_set_record_cb(self._record_cb)

        self._register_cb = self._register_callback_func()
        self.lib_xscope.xscope_ep_set_register_cb(self._register_cb)

        self._stats_cb = self._stats_callback_func()
        self.lib_xscope.xscope_ep_set_stats_cb(self._stats_cb)

    def _print_callback_func(self):
        def func(timestamp, length, data):
            self.on_print(timestamp, data[0:length])
        return PRINT_CALLBACK(func)

    def _register_callback_func(self):
        def func(id_, type_, r, g, b, name, unit, data_type, data_name):
            self._probe_info[id_] = {
                'type': type_,
                'name': name,
                'unit': unit,
                'data_type': data_type
            }
            self.on_register(id_, type_, name, unit, data_type)
        return REGISTER_CALLBACK(func)

    def _record_callback_func(self):
        def func(id_, timestamp, length, data_val, data_bytes):
            self.on_record(id_, timestamp, length, data_val, data_bytes)
        return RECORD_CALLBACK(func)

    def _stats_callback_func(self):
        def func(id_, average):
            #TODO
            pass
        return STATS_CALLBACK(func)

    def on_print(self, timestamp, data):
        """xScope printf handler.
           Override this to method to implement your own printing or to silence the printout.
        """
        print data.rstrip()

    def on_register(self, id_, type_, name, unit, data_type):
        """Server probe registration handler.
           Override this to method to implement your own registration or to silence the printout.
        """
        print 'Probe registered: id={}, type={}, name={}, unit={}, data_type={}'.format(id_, type_, name, unit, data_type)

    def on_record(self, id_, timestamp, length, data_val, data_bytes):
        """Server record handler.  Will dispatch to probe consumer callback.
           Override this to method to implement your own dispatcher.  However,
           that should rarely be necessary.
        """
        def notify_consumers(consumers, probe_name):
            for cb in consumers:
                cb(timestamp, probe_name, data_val)

        probe_info = self._probe_info[id_]
        probe_name =  probe_info['name']

        if probe_name in self._consumers:
            notify_consumers(self._consumers[probe_name], probe_name)
        if '*' in self._consumers:
            notify_consumers(self._consumers['*'], probe_name)

    def connect(self, hostname='localhost', port='10234'):
        """Connect to xSCOPE server

        Args:
            hostname (str): Hostname of running xSCOPE server.
            port (str): Port number of running xSCOPE server.

        Returns:
            0 for success
            1 for failure
        """
        return self.lib_xscope.xscope_ep_connect(hostname, port)

    def disconnect(self):
        """Disconnect from xSCOPE server
        """
        self.lib_xscope.xscope_ep_disconnect()

    def consume(self, callback, probe_name=None):
        """Consume a probe by name.

        Note:
            Set probe=None to consume all probes.

        Args:
            callback (callable): Callback function with signature (timestamp, probe, value).
            probe_name (str): Probe name
        """
        probe_name = probe_name or '*'
        self._consumers[probe_name].add(callback)

    def publish(self, data):
        """Publish message to endpoint.

        Args:
            data: Bytes to publish.

        Returns:
          0 for success
          1 for failure
        """
        return self.lib_xscope.xscope_ep_request_upload(ctypes.c_uint(len(data)+1), ctypes.c_char_p(data))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Python xSCOPE')
    parser.add_argument('--host', default='localhost', help='Hostname')
    parser.add_argument('--port', default='10234', help='Port')
    parser.add_argument('-c', '--consume', nargs='?', action='append', default=[], help='Probe names to consume (omit to consume all)')
    parser.add_argument('-p', '--publish', default=None, help='Message to publish')
    args = parser.parse_args()

    def test_callback(timestamp, probe_name, value):
        print '{} {} {}'.format(timestamp, probe_name, value)

    ep = Endpoint()
    try:
        if ep.connect(args.host, args.port):
            print "Failed to connect"
            sys.exit(1)

        if args.publish:
            ep.publish(args.publish)

        if args.consume:
            for probe in args.consume:
                ep.consume(test_callback, probe)
        else:
            ep.consume(test_callback)

        while(True):
            # Release the CPU
            time.sleep(1)
    except KeyboardInterrupt:
        ep.disconnect()
