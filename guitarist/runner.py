import sys
from multiprocessing import Process, Value, Array, Pipe
from threading import Thread


class Progress:
    MAX_INFO_LENGTH = 100

    def __init__(self):
        self._total = Value('d', 0.)
        self._current = Value('d', 0.)
        self._info = Array('c', self.MAX_INFO_LENGTH)

    @property
    def total(self):
        return self._total.value

    @total.setter
    def total(self, total):
        self._total.value = total

    @property
    def current(self):
        return self._current.value

    @current.setter
    def current(self, current):
        self._current.value = current

    @property
    def info(self):
        try:
            info = self._info.value.decode('utf-8')
        except UnicodeDecodeError:
            info = '???'
        return info

    @info.setter
    def info(self, info):
        info = info.encode('utf-8')[:self.MAX_INFO_LENGTH]
        self._info.value = info


class Runner:
    def __init__(self, *args):
        self._args = args
        self.progress = Progress()

    def start(self):
        conn_recv, conn_send = Pipe()
        args = (conn_send, self.run, self.progress) + self._args
        self._proc = proc = Process(target=self._run_wrapper, args=args)
        self._thread = thread = Thread(target=self._wait, args=(conn_recv,))

        proc.start()
        thread.start()

    def _wait(self, result_pipe):
        proc = self._proc
        proc.join()
        if proc.exitcode == 0:
            self.on_finish(result_pipe.recv())
        else:
            self.on_fail(result_pipe.recv())

    def _run_wrapper(self, result_pipe, run, progress, *args):
        try:
            result = run(progress, *args)
        except Exception as e:
            result_pipe.send(e)
            raise
        else:
            result_pipe.send(result)

    def run(self, progress, *args):
        " A hook. It will be run on other process. "

    def stop(self, block=True):
        self._proc.terminate()

    def join(self):
        self._thread.join()

    def on_finish(self, result):
        " A hook which will be call on finish normally. "

    def on_fail(self, exception):
        print(exception, sys.stderr)
