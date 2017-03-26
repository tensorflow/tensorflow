import distutils.command.register as orig


class register(orig.register):
    __doc__ = orig.register.__doc__

    def run(self):
        # Make sure that we are using valid current name/version info
        self.run_command('egg_info')
        orig.register.run(self)
