import sys
from fractions import Fraction
from itertools import combinations
from collections import namedtuple

import pysheetmusic
from raygllib.utils import timeit
from raygllib import ui

import pyximport; pyximport.install()
from ._arrange import CostCalculator
from ._arrange import MISS_PENALTY, get_fret_change_cost

__all__ = ['PlayState', 'FingeringArranger']

NoteEvent = namedtuple('NoteEvent', 'start end pitch effect note')

N_STRINGS = 6
N_FINGERS = 4
# FINGER_DISTANCES = [.06, .04, .05]
MAX_FINGER_DISTANCES = [.06, .10, .13]
MIN_FINGER_DISTANCES = [.008, .025, .025]
# MIN_FINGER_DISTANCES = [0, 0, 0]


class NoteEffect:
    REGULAR = 'regular'
    SLIDE = 'slide'
    HIT = 'hit'

def pitch_to_level(pitch):
    return 'C D EF G A B'.index(pitch.step) + int(pitch.alter) + pitch.octave * 12

def pitch_to_level2(step, octave):
    return 'C D EF G A B'.index(step) + octave * 12


class PlayState:
    __slots__ = ['bar', 'frets', 'strings', 'rings']

    def __init__(self):
        self.bar = False
        self.frets = [0] * 4
        self.strings = [-1] * 4
        self.rings = [-1] * N_STRINGS

    def froze(self):
        self.frets = tuple(self.frets)
        self.strings = tuple(self.strings)
        self.rings = tuple(self.rings)

    def match(self, fretboard, frame):
        matched = [None] * N_FINGERS
        pitchToEvent = {e.pitch: e for e in frame}
        for i in range(N_FINGERS):
            if self.bar and i == 0:
                continue
            if self.strings[i] != -1:
                pitch = fretboard.basePitches[self.strings[i]] + self.frets[i]
                matched[i] = ((self.frets[i], self.strings[i]), pitchToEvent.pop(pitch, None))
        if self.bar:
            matched[0] = []
            f = self.frets[0]
            for r in range(self.strings[0] + 1):
                try:
                    matched[0].append(((f, r), pitchToEvent.pop(fretboard.basePitches[r] + f)))
                except KeyError:
                    pass
        empty = []
        for r in range(N_STRINGS):
            if self.rings[r] >= 0:
                pitch = fretboard.basePitches[r]
                try:
                    empty.append((r, pitchToEvent.pop(pitch)))
                except KeyError:
                    pass
        missed = len(pitchToEvent)
        return matched, empty, missed

    def dump(self):
        try:
            s = '|---' * max(f for f, r in zip(self.frets, self.strings) if r >= 0)
        except ValueError:
            s = '|---'
        s = '-' + s
        t = [list(s) for i in range(N_STRINGS)]
        for i in range(N_FINGERS):
            r = self.strings[i] 
            if r >= 0:
                f = self.frets[i]
                t[r][4 * f - 1] = str(i + 1)
        if self.bar:
            i = 0
            f = self.frets[i]
            for r in range(self.strings[i]):
                t[r][4 * f - 1] = str(i + 1)
        for r in range(N_STRINGS):
            if self.rings[r] >= 0:
                t[r][0] = 'o'
        return str((self.frets, self.strings, self.rings)) + '\n' +\
            '\n'.join(map(''.join, t))

    def __hash__(self):
        return hash((self.bar, self.frets, self.strings, self.rings))


def get_note_events(sheet):
    noteEvents = []
    for timeStart, timeEnd, note in sheet.iter_note_sequence():
        if note.duration > 0:
            noteEvents.append(NoteEvent(
                timeStart, timeEnd, pitch_to_level(note.pitch), NoteEffect.REGULAR, note))
    noteEvents.sort(key=lambda x: x[:2])
    return noteEvents


def get_time_points(noteEvents):
    timePoints = []
    for event in noteEvents:
        if not timePoints or timePoints[-1] != event.start:
            timePoints.append(event.start)
    return timePoints


def get_frames(timePoints, noteEvents):
    """
    frames = {
        time0: [events at time0],
        time1: [events at time1],
        ...
    }
    """
    frames = {}
    activeEvents = []
    currentEventIdx = 0
    for time in timePoints:
        activeEvents = [event for event in activeEvents if event.end > time]
        while currentEventIdx < len(noteEvents) \
                and noteEvents[currentEventIdx].start == time:
            event = noteEvents[currentEventIdx]
            activeEvents.append(event)
            currentEventIdx += 1
        frame = {}
        for event in activeEvents:
            if event.pitch not in frame:
                frame[event.pitch] = event
            else:
                if event.end > frame[event.pitch].end:
                    frame[event.pitch] = event
        frames[time] = list(frame.values())

    return frames 

class StateCalculator:
    rates = []

    def __init__(self, fretboard):
        self.fretboard = fretboard
        self._cache = {}
        self._callCount = 0
        self._cacheHit = 0

    def report_stats(self):
        print('Total calls: {}'.format(self._callCount))
        print('Cache hit rate: {:.2f}'.format(self._cacheHit / self._callCount))

    # @profile
    def get_matched_states(self, frame):
        self._callCount += 1
        cacheKey = frozenset(e.pitch for e in frame)
        if cacheKey in self._cache:
            self._cacheHit += 1
            return self._cache[cacheKey].copy()

        collected = []
        iterCount = 0

        get_fret_distance = self.fretboard.get_fret_distance
        get_positions = self.fretboard.get_positions

        # @profile
        def collect(eventIdx):
            nonlocal iterCount
            while eventIdx < len(frame) and preHandled[eventIdx]:
                eventIdx += 1
            if eventIdx == len(frame):
                items = [
                    (f, -r) for i, (f, r) in enumerate(positions)
                    if f > 0 and not preHandled[i]
                ]
                items.sort()
                for fingers in combinations(range(1, N_FINGERS), len(items)):
                    state = PlayState()
                    state.frets[0] = indexFret
                    if indexString is not None:
                        state.strings[0] = indexString
                    finger1 = 0
                    for finger, (f, r) in zip(fingers, items):
                        state.frets[finger] = f
                        state.strings[finger] = -r
                        if f == state.frets[finger1] and not (
                                0 < state.strings[finger1] - state.strings[finger] <=
                                2 * (finger - finger1)):
                            break
                        if get_fret_distance(f, indexFret) > MAX_FINGER_DISTANCES[finger - 1]:
                            break
                        if bar and get_fret_distance(f, indexFret) < \
                                MIN_FINGER_DISTANCES[finger - 1]:
                            break
                        finger1 = finger
                    else:
                        state.rings[:] = stringUsed
                        state.bar = bar
                        state.froze()
                        collected.append(state)
                    iterCount += 1
                return
            event = frame[eventIdx]
            for (f, r) in get_positions(event.pitch):
                if stringUsed[r] == -1 and (f == 0 or f >= indexFret + bar and 
                        (indexString is not None or f - indexFret <= 3) and
                        get_fret_distance(f, indexFret) <= MAX_FINGER_DISTANCES[-1]):
                    # Assign this position to current event
                    stringUsed[r] = f
                    positions[eventIdx] = (f, r)
                    collect(eventIdx + 1)
                    stringUsed[r] = -1

        dropCount = 0
        frame0 = frame
        basePitches = self.fretboard.basePitches
        while not collected and dropCount < len(frame0):
            for frame in combinations(frame, len(frame0) - dropCount):
                stringUsed = [-1] * N_STRINGS
                preHandled = [False] * len(frame)
                positions = [None] * len(frame)
                bar = False
                indexFret = 0
                indexString = None
                for indexFret in range(1, self.fretboard.maxFret):
                    bar = False
                    # Use index finger, choose a event for it.
                    for i, event in enumerate(frame):
                        preHandled[i] = True
                        for r in range(N_STRINGS):
                            if basePitches[r] == event.pitch - indexFret:
                                stringUsed[r] = indexFret
                                positions[i] = (indexFret, r)
                                indexString = r
                                collect(0)
                                stringUsed[r] = -1
                        preHandled[i] = False
                    # Do not use index finger
                    indexString = None
                    collect(0)
                    # Bar
                    if indexFret <= 12 and len(frame) > 2:
                        bar = True
                        for r in range(2, N_STRINGS):
                            # Bar strings (0, 1, ..., r) using index finger
                            # Select all barred pitches
                            indexString = r
                            barred = {basePitches[r1] + indexFret: r1 for r1 in range(r + 1)}
                            for i, event in enumerate(frame):
                                if event.pitch in barred:
                                    preHandled[i] = True
                                    r1 = barred[event.pitch]
                                    positions[i] = (indexFret, r1)
                                    stringUsed[r1] = indexFret
                            collect(0)
                            stringUsed = [-1] * N_STRINGS
                            preHandled = [False] * len(frame)
            dropCount += 1
        dropCount -= 1
        if dropCount == 0:
            self._cache[cacheKey] = collected.copy()
        if dropCount:
            print('drop', dropCount)

        # if iterCount > 0:
        #     self.rates.append(len(collected) / iterCount)
        return collected


class FingeringArranger:
    """
    frames
    noteEvents
    time_points
    states
    fretboard
    """
    def __init__(self, sheet):
        self.sheet = sheet
        self.noteEvents = noteEvents = get_note_events(sheet)
        self.timePoints = timePoints = get_time_points(noteEvents)
        self.frames = get_frames(timePoints, noteEvents)
        self._make_fretboard()
        self.stateCalc = StateCalculator(self.fretboard)
        self.states = []
        # self.arrange()
        # self._dump_frames(timePoints, self.frames)
        # self.stateCalc.report_stats()

    @property
    def time_points(self):
        return self.timePoints

    # @profile
    # @timeit
    def arrange(self, progress):
        fb = self.fretboard
        timePoints = self.timePoints
        nTimePoints = len(timePoints)
        frames = self.frames
        # costs[i, j] = min cost of state statess[i][j] at time[i]
        costss = []
        choicess = []
        # statess[i] = states of frames[timePoints[i]]
        progress.info = 'Generate states...'
        progress.total = nTimePoints
        statess = []
        for i in range(nTimePoints):
            statess.append(self.stateCalc.get_matched_states(frames[timePoints[i]]))
            progress.current = i + 1
        # with open('debug-dump.txt', 'w') as outfile:
        #     for i, time in enumerate(timePoints):
        #         frame = frames[time]
        #         outfile.write('i: {}, time: {}\n'.format(i, time))
        #         for event in frame:
        #             outfile.write(
        #                 '  start:{}, end:{}, pitch:{}({}), \n    measure:{} note:{}\n'.format(
        #                     event.start, event.end, event.pitch, fb.get_positions(event.pitch),
        #                     event.note.measure.number, event.note,
        #                 )
        #             )
        #         outfile.write('  states:')
        #         for state in statess[i]:
        #             outfile.write(state.dump() + '\n=================\n')

        progress.info = 'Calculating best fingering...'
        progress.total = nTimePoints

        costCalc = CostCalculator(fb)
        i = 0
        costss.append([0] * len(statess[0]))
        for j, state in enumerate(statess[0]):
            matched, empty, missed = state.match(fb, frames[timePoints[0]])
            costss[0][j] += missed * MISS_PENALTY
        choicess.append([None] * len(statess[0]))
        costCalc.add_frame(frames[timePoints[0]], statess[0], timePoints[0])
        progress.current = 1

        for i in range(1, nTimePoints):
            # print(i, nTimePoints)
            states1 = statess[i - 1]
            states2 = statess[i]
            costs = [None] * len(states2)
            costss.append(costs)
            choices = [None] * len(states2)
            choicess.append(choices)
            t1, t2 = timePoints[i - 1], timePoints[i]
            dt = float(t2 - t1)
            costCalc.add_frame(frames[t2], states2, t2)
            tot = len(states2) * len(states1)
            cnt = 0
            j1Order = list(range(len(states1)))
            j1Order.sort(key=lambda j: costss[i - 1][j])
            # j1Order = j1Order[:5]

            for j2, state2 in enumerate(states2):
                minCost = 1e20
                choice = None
                fret0 = state2.frets[0]
                j1Order.sort(key=lambda j: abs(states1[j].frets[0] - fret0))
                for j1 in j1Order:
                    state1 = states1[j1]
                    lastCost = costss[i - 1][j1]
                    if lastCost + get_fret_change_cost(
                            state1.frets[0], state2.frets[0], dt) >= minCost:
                        break
                    if lastCost >= minCost:
                        continue
                    cnt += 1
                    cost = lastCost + costCalc.get_cost(j1, j2, minCost)
                    if minCost > cost:
                        minCost = cost
                        choice = j1
                assert choice is not None
                costs[j2] = minCost
                choices[j2] = choice
            # print(cnt / tot, cnt, tot)
            progress.current = i + 1
        # dump states
        # for i in range(nTimePoints):
        #     for j2, state2 in enumerate(statess[i]):
        #         print('  i: {}, j: {}, cost: {}, choice: {}\n{}'.format(
        #             i, j2, costss[i][j2], choicess[i][j2], state2.dump()))
        # exit(0)
        # import matplotlib.pyplot as plt
        # plt.plot([len(x) for x in statess])
        # plt.show()

        progress.info = 'Tracing best solution...'
        progress.total = nTimePoints
        j = min(range(len(statess[-1])), key=lambda j:costss[-1][j])
        i = nTimePoints - 1
        # print('minCost', costss[-1][j])
        # Trace back the solution
        states = []
        while i >= 0:
            # print('choice', 'i', i, 'j', j)
            state = statess[i][j]
            states.append(state)
            t = timePoints[i]
            matched, empty, missed = state.match(fb, frames[t])
            # print('=======', i, t)
            # print('matched', matched)
            # print('empty', empty)
            # print('missed', missed)
            # print(state.dump())
            # if missed:
            #     print('missed', missed, 'at', t, i)
            for finger in range(N_FINGERS):
                if not matched[finger]:
                    continue
                if finger == 0 and state.bar:
                    events = matched[finger]
                else:
                    events = [matched[finger]]
                for (f, r), event in events:
                    if event.start == t:
                        fingering = event.note.fingering
                        fingering.fret = f
                        fingering.string = r + 1
                        fingering.finger = finger + 1
            for r, event in empty:
                if event.start == t:
                    fingering = event.note.fingering
                    fingering.fret = 0
                    fingering.string = r + 1
            progress.current = nTimePoints - i

            j = choicess[i][j]
            i -= 1
        self.states =  states[::-1]
        progress.info = 'Arranged.'
        progress.total = progress.current = 1

        # import pickle
        # s = pickle.dumps((self.sheet, self))
        # print('picle size', len(s))

    def _make_fretboard(self):
        fretboard = Fretboard()
        for event in self.noteEvents:
            if fretboard.minPitch > event.pitch:
                fretboard = Fretboard(tuning='drop-d')
                break
        self.fretboard = fretboard

    def _dump_frames(self, timePoints, frames):
        # import matplotlib.pyplot as plt
        stateLens = []
        for time in timePoints[:]:
            frame = frames[time]
            states = self.stateCalc.get_matched_states(frame)
            stateLens.append(len(states))
            if len(states) >= 100:
                print('>>>>>>>>>>>>>>>>>>>')
                print(time, float(time), frame[0].note.measure)
                print('len(frame)', len(frame), 'len(states)', len(states))
                for event in frame:
                    pitch = event.pitch
                    step = (
                        'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'
                    )[pitch % 12]
                    octave = pitch // 12
                    print('   ', step, octave, event)
                for state in states:
                    print('========================')
                    print(state.dump())

        # plt.hist(stateLens)
        # plt.show()
        print('max len states', max(stateLens))


def dump_frame(frame):
    print(frame[0].note.measure)
    for event in frame:
        pitch = event.pitch
        step = (
            'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'
        )[pitch % 12]
        octave = pitch // 12
        print('   ', step, octave, event)

class Fretboard:
    TUNING = {
        'standard': [('E', 4), ('B', 3), ('G', 3), ('D', 3), ('A', 2), ('E', 2)],
        'drop-d': [('E', 4), ('B', 3), ('G', 3), ('D', 3), ('A', 2), ('D', 2)],
    }
    # Fretboard dimensions.
    TOP = .08
    BOTTOM = .10
    STRING_LENGTH = .90

    def __init__(self, maxFret=17, tuning='standard'):
        self.basePitches = [pitch_to_level2(*args) for args in self.TUNING[tuning]]
        self.minPitch = minPitch = self.basePitches[-1]
        self.maxPitch = maxPitch = self.basePitches[0] + maxFret
        self.maxFret = maxFret
        positions = self._positions = [[] for _ in range(maxPitch - minPitch + 1)]
        nStrings = len(self.basePitches)
        for r in range(nStrings):
            for f in range(maxFret + 1):
                pitch = self.basePitches[r] + f
                positions[pitch - minPitch].append((f, r))
        for ps in positions:
            ps.sort()
        self._make_bars()
        # print(self.basePitches)
        # print(self._fretPos, self.get_fret_distance(3, 7))

    def _make_bars(self):
        a = 2 ** (1. / 12)
        xs = [0] * (self.maxFret + 1)
        for i in range(1, len(xs)):
            xs[i] = ((a - 1) * self.STRING_LENGTH + xs[i - 1]) / a
        self._barPos = xs
        b = .8
        self._fretPos = [0] + \
            [xs[i] * (1 - b) + xs[i + 1] * b for i in range(len(xs) - 1)]

    def get_fret_distance(self, f1, f2):
        return abs(self._fretPos[f1] - self._fretPos[f2])

    def get_positions(self, pitch, minFret=0):
        " return: A list of (fret, string) tuples. "
        assert pitch >= self.minPitch
        ps = self._positions[pitch - self.minPitch]
        for i in range(len(ps)):
            if ps[i][0] >= minFret:
                return ps[i:]
        return []
