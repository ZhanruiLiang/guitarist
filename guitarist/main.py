import os
import threading

from raygllib import ui
import raygllib
import raygllib.model
import raygllib.viewer
import raygllib.camera
import raygllib.config
raygllib.config.toonRenderEnable = False
import pysheetmusic
from pysheetmusic.player import PlayerState

from .performer import Performer
from .arrange import FingeringArranger
from .runner import Runner


def get_resouce_path(*sub_paths):
    return os.path.join(os.path.dirname(__file__), *sub_paths)

class ProgressBar(ui.Spin):
    def __init__(self):
        super().__init__(
            fixedSize=True, height=12, fontSize=10, maxValue=1., minValue=0., value=0.)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        pass

    def update_progress(self, progress):
        self.text = progress.info
        self.maxValue = max(1, progress.total)
        self.update_value(progress.current)

    def format_value(self, value):
        return '{}/{}({:.0f}%)'.format(
            int(self.value), int(self.maxValue), self.value / self.maxValue * 100)


class PlayProgressBar(ui.Spin):
    """
    public properties:
        text
    """
    def __init__(self):
        super().__init__(minValue=0, maxValue=1, value=0)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        pass

    def set_info(self, info):
        self.text = info

    def set_total_time(self, time):
        " time: Total time in seconds. "
        self.maxValue = time

    def set_current_time(self, time):
        " time: Current time in seconds. "
        self.update_value(time)

    @staticmethod
    def _format_time(time):
        time = int(time)
        minutes = time // 60
        seconds = time % 60
        return '{}:{:02d}'.format(minutes, seconds)

    def format_value(self, value):
        return '{}/{}'.format(
            self._format_time(value), self._format_time(self.maxValue))


class TopBar(ui.Panel):
    def __init__(self):
        super().__init__(
            layoutDirection=ui.LayoutDirection.HORIZONTAL,
            fixedSize=True, height=28,
        )
        path_bar = ui.PathInput(
            hint='Input MusicXML path',
            fixedSize=True, width=400,
        )
        button_width = 60
        play_button = ui.Button(text='Play', fixedSize=True, width=button_width)
        pause_button = ui.Button(text='Pause', fixedSize=True, width=button_width)
        stop_button = ui.Button(text='Stop', fixedSize=True, width=button_width)
        progress_bar = PlayProgressBar()
        self.children.extend([
            path_bar, play_button, pause_button, stop_button, progress_bar
        ])

        self.play_button = play_button
        self.pause_button = pause_button
        self.stop_button = stop_button
        self.path_bar = path_bar
        self.progress_bar = progress_bar


class PerformerViewer(ui.Widget):
    def __init__(self, scene):
        super().__init__()
        self.scene = scene
        self.canvas = raygllib.viewer.ViewerCanvas()
        self.children.append(self.canvas)
        self.canvas.scene = scene
        self.canvas.camera = raygllib.camera.Camera(
            pos = (2, -3.5, 1.7),
            up = (0, 0, 1),
            center = (0, 0, 0),
        )

    def update(self, dt):
        self.canvas.camera.update(dt)
        self.scene.update(dt)

class AppState:
    READY = 'ready'
    LOADING_SHEET = 'loading-sheet'
    PLAYING = 'playing'
    PAUSED = 'paused'
    STOPPED = 'STOPPED'

class Window(ui.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sheet_to_free = []
        self.sheet = None
        self.player = pysheetmusic.player.Player()
        self.sheet_viewer = pysheetmusic.viewer.SheetViewer()
        self.sheet_viewer.fixedSize = True
        self.sheet_viewer.width = 800
        self.sheet_viewer.set_player(self.player)
        self.top_bar = TopBar()
        self.progress_bar = ProgressBar()
        scene = raygllib.model.load_scene(get_resouce_path('scene', 'scene.dae'))
        self.performer_viewer = PerformerViewer(scene)
        self.performer = Performer(scene)
        self._sheet_parser = pysheetmusic.parse.MusicXMLParser()

        self.root.layoutDirection = ui.LayoutDirection.VERTICAL
        self.root.children = [
            self.top_bar,
            self.progress_bar,
            ui.Widget(layoutDirection=ui.LayoutDirection.HORIZONTAL, children=[
                self.sheet_viewer, self.performer_viewer,
            ])
        ]
        self._bind_functions()
        self._runner = None
        self.request_relayout()
        self._updateLock = threading.RLock()
        self.state = AppState.READY

    def _bind_functions(self):
        top_bar = self.top_bar
        top_bar.path_bar.connect_signal('open',
            lambda: self.load_sheet(os.path.expanduser(top_bar.path_bar.text)))
        top_bar.play_button.connect_signal('clicked', self.play)
        top_bar.pause_button.connect_signal('clicked', self.pause)
        top_bar.stop_button.connect_signal('clicked', self.stop)

    def set_runner(self, runner):
        if self._runner is not None:
            self._runner.stop(block=True)
        self._runner = runner
        runner.start()

    def load_sheet(self, path):
        window = self
        self.sheet_viewer.set_sheet_layout(None)
        self.stop()
        class LoadSheetRunner(Runner):
            def run(self, progress, parser, path):
                sheet = parser.parse(path)
                pysheetmusic.tab.attach_tab(sheet)
                pysheetmusic.tab.attach_fingerings(sheet)
                arranger = FingeringArranger(sheet)
                arranger.arrange(progress)
                layout = pysheetmusic.layout.LinearTabLayout(sheet)
                layout.layout()
                return sheet, arranger, layout

            def on_finish(self, result):
                sheet, arranger, layout = result
                with window._updateLock:
                    if window.sheet is not None:
                        window._sheet_to_free.append(window.sheet)
                    window.sheet = sheet
                    window.player.set_sheet(sheet)
                    window.sheet_viewer.set_sheet_layout(layout)
                    window.performer.set_sheet(sheet, arranger)
                window.play()
        with self._updateLock:
            self.set_runner(LoadSheetRunner(self._sheet_parser, path))
            self.state = AppState.LOADING_SHEET

    def play(self):
        if self.state == AppState.PLAYING:
            return
        with self._updateLock:
            self._runner = None
            self.player.play()
            self.top_bar.progress_bar.set_total_time(self.sheet.totalTime)
            self.top_bar.progress_bar.set_info('Playing')
            self.state = AppState.PLAYING

    def pause(self):
        if self.state != AppState.PLAYING:
            return
        with self._updateLock:
            self.player.pause()
            self.top_bar.progress_bar.set_info('Paused')
            self.state = AppState.PAUSED

    def stop(self):
        if self.state in (AppState.READY, AppState.LOADING_SHEET, AppState.STOPPED):
            return
        with self._updateLock:
            self.player.stop()
            self.top_bar.progress_bar.set_info('Stopped')
            self.state = AppState.STOPPED

    def update(self, dt):
        super().update(dt)
        with self._updateLock:
            for sheet in self._sheet_to_free:
                sheet.free()
            self._sheet_to_free.clear()
            self.performer_viewer.update(dt)
            if self.state == AppState.LOADING_SHEET:
                self.progress_bar.update_progress(self._runner.progress)
            elif self.state != AppState.READY:
                self.sheet_viewer.update(dt)
                current_time = self.player.get_current_time()
                self.top_bar.progress_bar.set_current_time(current_time)
                self.performer.sync_to_time(current_time)
                if self.player.state == PlayerState.STOPPED:
                    self.stop()

    def on_draw(self):
        with self._updateLock:
            super().on_draw()
