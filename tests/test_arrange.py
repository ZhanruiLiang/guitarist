import sys
import unittest

import pysheetmusic
from raygllib import ui

from guitarist.arrange import FingeringArranger
from guitarist.runner import Progress

SHEETS = [
    # 'Debug.mxl',
    'We_wish_you_a_Merry_Christmas.mxl',
    'Lute_Suite_No._1_in_E_Major_BWV_1006a_J.S._Bach.mxl',
    'Jeux_interdits.mxl',
    'K27_Domenico_Scarlatti.mxl',
    # 'Auld_Lang_Syne_guitar.mxl',
    # 'Allegretto_in_C_Major_for_Guitar_by_Carcassi_-_arr._by_Gerry_Busch.mxl',
    'Fernando_Sor_Op.32_Mazurka.mxl',
    'Fernando_Sor_Op.32_Galop.mxl',
    'Fernando_Sor_Op.32_Andante_Pastorale.mxl',
    'Fernando_Sor_Op.32_Andantino.mxl',
    'Allegro_by_Bernardo_Palma_V.mxl',
    'Chord_test.mxl',
    'Untitled_in_D_Major.mxl',
    'Divertimento_No._1.mxl',
    'Giuliani_-_Op.50_No.1.mxl',
    'Chrono_Cross_-_Quitting_the_Body.mxl',
    'Unter_dem_Lindenbaum.mxl',
    'Lagrima.mxl',
    'Guitar_Solo_No._116_in_A_Major.mxl',
    'Almain.mxl',
    'Somewhere_In_My_Memory.mxl',
    'Tango_Guitar_Solo_2.mxl',
    'Air.mxl',
    'Guitar_Solo_No._117_in_E_Minor.mxl',
    'Chrono_Cross_-_Frozen_Flame.mxl',
    'Guitar_Solo_No._118_-_Barcarolle_in_A_Minor.mxl',
    'Guitar_Solo_No._119_in_G_Major.mxl',
    'Guitar_Solo_No._15_in_E_Major.mxl',
    'Maria_Luisa_Mazurka_guitar_solo_the_original_composition.mxl',
    'Minuet_in_G_minor.mxl',
    'Pavane_No._6_for_Guitar_Luis_Milan.mxl',
    'People_Imprisoned_by_Destiny.mxl',
]

class Window(ui.Window):
    def __init__(self):
        self.viewer = pysheetmusic.viewer.SheetViewer()
        super().__init__(resizable=True)
        self.root.children.append(self.viewer)
        K = ui.key
        self.player = player = pysheetmusic.player.Player()
        self.viewer.set_player(player)

        self.add_shortcut(K.chain(K.Q), sys.exit)
        self.add_shortcut(K.chain(K.SHIFT, K.P), lambda: player.pause() or True)
        self.add_shortcut(K.chain(K.P), lambda: player.play() or True)

    def on_close(self):
        self.player.stop()
        super().on_close()

    def set_sheet(self, sheet):
        layout = pysheetmusic.layout.LinearTabLayout(sheet)
        layout.layout()
        self.viewer.set_sheet_layout(layout)
        self.player.set_sheet(sheet)
        self.player.play()

    def update(self, dt):
        super().update(dt)
        self.viewer.update(dt)


class TestArrange(unittest.TestCase):
    def test_arrange(self):
        # import crash_on_ipy
        parser = pysheetmusic.parse.MusicXMLParser()
        BASE = '/home/ray/python/pysheetmusic/tests/sheets/'
        for name in SHEETS[:]:
            sheet = parser.parse(BASE + name)
            pysheetmusic.tab.attach_tab(sheet)
            pysheetmusic.tab.attach_fingerings(sheet)
            arranger = FingeringArranger(sheet)
            progress = Progress()
            arranger.arrange(progress)

            window = Window()
            window.set_sheet(sheet)
            window.start()

if __name__ == '__main__':
    TestArrange().test_arrange()
