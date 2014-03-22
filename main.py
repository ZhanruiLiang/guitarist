import crash_on_ipy
from guitarist.main import Window

# window = Window(width=1366, height=768)
window = Window(width=1366, height=600)
# window.load_sheet('/home/ray/python/pysheetmusic/tests/sheets/We_wish_you_a_Merry_Christmas.mxl')
window.load_sheet('/home/ray/python/pysheetmusic/tests/sheets/Chord_test.mxl')
# window.load_sheet('/home/ray/python/pysheetmusic/tests/sheets/Debug.mxl')
window.start()
