from .. import tool

def test_keygen():
    def get_keyring():
        WheelKeys, keyring = tool.get_keyring()

        class WheelKeysTest(WheelKeys):
            def save(self):
                pass

        class keyringTest:
            @classmethod
            def get_keyring(cls):
                class keyringTest2:
                    pw = None
                    def set_password(self, a, b, c):
                        self.pw = c
                    def get_password(self, a, b):
                        return self.pw

                return keyringTest2()

        return WheelKeysTest, keyringTest

    tool.keygen(get_keyring=get_keyring)
