import os

from raygllib import ui
import raygllib
import pysheetmusic

from .performer import Performer
from .arrange import FingeringArranger


def get_resouce_path(*sub_paths):
    return os.path.join(os.path.dirname(__file__), *sub_paths)

class ProgressBar(ui.Widget):
    def update(self, info, percent):
        pass

class TopBar(ui.Widget):
    pass

class PerformerViewer(ui.Widget):
    pass

class Window(ui.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.player = pysheetmusic.player.Player()
        self.sheet_viewer = pysheetmusic.viewer.SheetViewer()
        self.sheet_viewer.set_player(player)
        self.performer_viewer = PerformerViewer()
        self.top_bar = TopBar()
        scene = raygllib.model.load_scene(get_resouce_path('scene.dae'))
        self.performer = Performer(scene)
        self._sheet_parser = pysheetmusic.parse.MusicXMLParser()

    def set_runner(self, runner):
        if self._runner is not None:
            self._runner.stop(block=True)
        self._runner = runner
        runner.start()

    def load_sheet(self, path):
        self.set_runner(LoadSheetRunner(self._parser, path))

    def on_sheet_loaded(self, result):
        sheet, arranger = result
        self.sheet = sheet

    def play(self):
        pass

    def pause(self):
        pass

    def stop(self):
        pass


class LoadSheetRunner(Runner):
    def run(self, parser, path):
        sheet = parser.parse(path)
        pysheetmusic.tab.attach_tab(sheet)
        pysheetmusic.tab.attach_fingerings(sheet)
        arranger = FingeringArranger(sheet)
