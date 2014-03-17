import unittest

from guitarist.main import Window

class TestMain(unittest.TestCase):
    def test_main(self):
        window = Window(width=1366, height=768)
        window.start()
