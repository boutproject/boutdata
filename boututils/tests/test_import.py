import unittest

from boututils.datafile import DataFile


class TestImport(unittest.TestCase):
    def test_import(self):
        d = DataFile()
        self.assertIsInstance(d, DataFile)


if __name__ == "__main__":
    unittest.main()
