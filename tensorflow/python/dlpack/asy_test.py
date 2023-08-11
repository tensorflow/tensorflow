import unittest


import asy


class TestAsync(unittest.TestCase):

    def test_main(self):
        number_of_networks = 10
        number_of_cpus = 4
        results = asy.main(number_of_networks, number_of_cpus)
        self.assertEqual(len(results), number_of_networks)


if __name__ == "__main__":
    unittest.main()
