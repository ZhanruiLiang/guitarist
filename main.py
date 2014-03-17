from guitarist.main import Window

import crash_on_ipy
window = Window(width=1366, height=768)
window.load_sheet('/home/ray/python/pysheetmusic/tests/sheets/Auld_Lang_Syne_guitar.mxl')
window.start()
