import getpass
from distutils.command import upload as orig


class upload(orig.upload):
    """
    Override default upload behavior to obtain password
    in a variety of different ways.
    """

    def finalize_options(self):
        orig.upload.finalize_options(self)
        self.username = (
            self.username or
            getpass.getuser()
        )
        # Attempt to obtain password. Short circuit evaluation at the first
        # sign of success.
        self.password = (
            self.password or
            self._load_password_from_keyring() or
            self._prompt_for_password()
        )

    def _load_password_from_keyring(self):
        """
        Attempt to load password from keyring. Suppress Exceptions.
        """
        try:
            keyring = __import__('keyring')
            return keyring.get_password(self.repository, self.username)
        except Exception:
            pass

    def _prompt_for_password(self):
        """
        Prompt for a password on the tty. Suppress Exceptions.
        """
        try:
            return getpass.getpass()
        except (Exception, KeyboardInterrupt):
            pass
