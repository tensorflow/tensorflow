import tempfile
import os.path
import unittest
import json

from wheel.signatures import keys

wheel_json = """
{
  "verifiers": [
    {
      "scope": "+", 
      "vk": "bp-bjK2fFgtA-8DhKKAAPm9-eAZcX_u03oBv2RlKOBc"
    }, 
    {
      "scope": "+", 
      "vk": "KAHZBfyqFW3OcFDbLSG4nPCjXxUPy72phP9I4Rn9MAo"
    },
    {
      "scope": "+", 
      "vk": "tmAYCrSfj8gtJ10v3VkvW7jOndKmQIYE12hgnFu3cvk"
    } 
  ], 
  "signers": [
    {
      "scope": "+", 
      "vk": "tmAYCrSfj8gtJ10v3VkvW7jOndKmQIYE12hgnFu3cvk"
    }, 
    {
      "scope": "+", 
      "vk": "KAHZBfyqFW3OcFDbLSG4nPCjXxUPy72phP9I4Rn9MAo"
    }
  ], 
  "schema": 1
}
"""

class TestWheelKeys(unittest.TestCase):
    def setUp(self):
        self.config = tempfile.NamedTemporaryFile(suffix='.json')
        self.config.close()
        
        self.config_path, self.config_filename = os.path.split(self.config.name) 
        def load(*args):
            return [self.config_path]
        def save(*args):
            return self.config_path
        keys.load_config_paths = load
        keys.save_config_path = save
        self.wk = keys.WheelKeys()
        self.wk.CONFIG_NAME = self.config_filename
        
    def tearDown(self):
        os.unlink(self.config.name)
        
    def test_load_save(self):
        self.wk.data = json.loads(wheel_json)
        
        self.wk.add_signer('+', '67890')
        self.wk.add_signer('scope', 'abcdefg')
        
        self.wk.trust('epocs', 'gfedcba')
        self.wk.trust('+', '12345')
        
        self.wk.save()
        
        del self.wk.data
        self.wk.load()
        
        signers = self.wk.signers('scope')
        self.assertTrue(signers[0] == ('scope', 'abcdefg'), self.wk.data['signers'])
        self.assertTrue(signers[1][0] == '+', self.wk.data['signers'])
        
        trusted = self.wk.trusted('epocs')
        self.assertTrue(trusted[0] == ('epocs', 'gfedcba'))
        self.assertTrue(trusted[1][0] == '+')
        
        self.wk.untrust('epocs', 'gfedcba')
        trusted = self.wk.trusted('epocs')
        self.assertTrue(('epocs', 'gfedcba') not in trusted)
        
    def test_load_save_incomplete(self):
        self.wk.data = json.loads(wheel_json)
        del self.wk.data['signers']
        self.wk.data['schema'] = self.wk.SCHEMA+1
        self.wk.save()
        try:
            self.wk.load()
        except ValueError:
            pass
        else:
            raise Exception("Expected ValueError")
        
        del self.wk.data['schema']
        self.wk.save()
        self.wk.load()
    
    
