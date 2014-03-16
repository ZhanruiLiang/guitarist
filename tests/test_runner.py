import unittest
import guitarist.runner as runner

class TestRunner(unittest.TestCase):
    def test_simple(self):
        class PlusRunner(runner.Runner):
            def run(self, progress, a, b):
                return a + b

            def on_finish(self, result):
                self.result = result

        r = PlusRunner(10, 20)
        r.start()
        r.join()
        assert r.result == 30

    def test_progress(self):
        class WriteProgressRunner(runner.Runner):
            def run(self, progress):
                progress.total = 100
                progress.current = 20
                progress.info = 'well done'

        r = WriteProgressRunner()
        r.start()
        r.join()
        assert r.progress.total == 100
        assert r.progress.current == 20
        assert r.progress.info == 'well done'

    def test_failing(self):
        class FailRunner(runner.Runner):
            def run(self, progress):
                assert 1 == 2
                return None

            def on_finish(self, result):
                self.result = result

            def on_fail(self, exception):
                self.result = exception

        r = FailRunner()
        r.start()
        r.join()
        assert isinstance(r.result, Exception)
