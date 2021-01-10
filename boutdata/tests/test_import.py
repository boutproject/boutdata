import unittest

from boutdata.collect import collect


class TestImport(unittest.TestCase):
    def test_import(self):
        self.assertTrue(callable(collect))


if __name__ == "__main__":
    unittest.main()
