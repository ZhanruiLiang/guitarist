cdef:
    DEF N_STRINGS = 6
    DEF N_FINGERS = 4
    DEF MAX_STATES = 500
    DEF MAX_FRAME_SIZE = 10

    # Parameters about movement distance.
    float b1 = .02  # Change fret.
    float b2 = .04  # Change string.
    float b3 = 2.   # Change index fret.
    float b4 = 5.
    # Parameters about finger change cost.
    float a2 = 0.5  # release
    float a3 = 0.2  # keep
    float a4 = 1.  # press

    float c2 = a2 * 10 # bar release
    float c3 = a3 * 10 # bar keep
    float c4 = a4 * 10 # bar press

    float MIN_RELEASE_TIME = 0.05
    float MIN_PRESS_TIME = 0.1
    float MIN_JUMP_TIME = 0.1
    float HIGH_FRET_PENALTY = 1.
    float SAME_STRING_PENALTY = 10.
    DEF HIGH_FRET = 5

MISS_PENALTY = 10000.

cdef float sqr(float x):
    return x * x

ctypedef struct State:
    int frets[N_FINGERS]
    int strings[N_FINGERS]
    int rings[N_STRINGS]
    bint bar
    # Variables calculated by `match`.
    int missed
    int indexMatched[N_STRINGS + 1]
    int matched[N_FINGERS]  # Matched indices of the frame
    float endTime[N_FINGERS]
    # fingers[i] = f, means use finger f to resolve note i. -1 means missed,
    # -2 means empty string.
    int fingers[MAX_FRAME_SIZE]

cdef match_frame(State *s, fretboard, frame):
    cdef:
        int i, f, r, nIndexMatched, pitch
        dict pitchToEvent

    for i in range(N_FINGERS):
        s.matched[i] = -1
    for i in range(len(frame)):
        s.fingers[i] = -1
    pitchToEvent = {e.pitch: i for i, e in enumerate(frame)}
    for i in range(N_FINGERS):
        if s.bar and i == 0:
            continue
        if is_pressed(s, i):
            pitch = fretboard.basePitches[s.strings[i]] + s.frets[i]
            s.matched[i] = pitchToEvent.pop(pitch, -1)
            s.fingers[s.matched[i]] = i
    if s.bar:
        nIndexMatched = 0
        f = s.frets[0]
        for r in range(s.strings[0] + 1):
            pitch = fretboard.basePitches[r] + f
            if pitch in pitchToEvent:
                s.indexMatched[nIndexMatched] = pitchToEvent.pop(pitch)
                s.fingers[s.indexMatched[nIndexMatched]] = 0
                nIndexMatched += 1
        s.indexMatched[nIndexMatched] = -1
    for i in range(N_STRINGS):
        if s.rings[i] >= 0:
            noteId = pitchToEvent.pop(fretboard.basePitches[i], None)
            if noteId is not None:
                s.fingers[noteId] = -2
    for i in range(N_FINGERS):
        s.endTime[i] = _get_end_time(s, frame, i) if is_pressed(s, i) else -1.
    s.missed = len(pitchToEvent)

cpdef float get_fret_change_cost(int f1, int f2, float dt):
    if f1 == f2:
        return 0
    else:
        return b4 + b3 * abs(f1 - f2) / dt

cdef float _get_end_time(State *s, list frame, int finger):
    if not s.bar or finger > 0:
        assert s.matched[finger] >= 0
        return float(frame[s.matched[finger]].end)
    cdef float end = -1
    i = 0
    while i < N_STRINGS and s.indexMatched[i] != -1:
        end = max(end, float(frame[s.indexMatched[i]].end))
        i += 1
    return end

cdef is_pressed(State *s, int finger):
    return s.strings[finger] != -1

cdef class CostCalculator:
    cdef:
        State states[2][MAX_STATES]
        int nStates[2]
        int frameMatch[MAX_FRAME_SIZE]
        object fretboard, times, frames

    def __init__(self, fretboard):
        self.fretboard = fretboard
        self.nStates[0] = 0
        self.nStates[1] = 0
        self.times = [0, 0]
        self.frames = [None, None]

    def add_frame(self, list frame, list states, time):
        cdef:
            int i, j, n
            State *s
        self.nStates[0] = self.nStates[1]
        for i in range(self.nStates[1]):
            self.states[0][i] = self.states[1][i]
        self.times[0] = self.times[1]
        self.times[1] = time
        self.frames[0] = self.frames[1]
        self.frames[1] = frame
        n = self.nStates[1] = len(states)
        for i in range(n):
            state = states[i]
            s = &self.states[1][i]
            for j in range(N_FINGERS):
                s.frets[j] = state.frets[j]
                s.strings[j] = state.strings[j]
            for j in range(N_STRINGS):
                s.rings[j] = state.rings[j]
            s.bar = state.bar
            match_frame(s, self.fretboard, frame)

        lastFrame = self.frames[0]
        if lastFrame:
            for i, event in enumerate(frame):
                try:
                    j = lastFrame.index(event)
                except ValueError:
                    j = -1
                self.frameMatch[i] = j

    cdef _get_bar_cost(self, State *s1, State *s2, list frame1, list frame2, float t1, float t2):
        cdef:
            float endTime, keepTime, releaseTime, pressTime
            float cost = 0
        if s1.bar:
            if s2.bar and s1.strings[0] == s2.strings[0] and s1.frets[0] == s2.frets[0]:
                # Keep
                keepTime = (t2 - t1)
                cost += keepTime * c3
            else:
                # Release
                endTime = min(t2, s1.endTime[0])
                keepTime = endTime - t1
                if not is_pressed(s2, 0):
                    # Will not press after release
                    releaseTime = max(MIN_RELEASE_TIME, t2 - endTime)
                else:
                    # Will press after release
                    releaseTime = max(MIN_RELEASE_TIME, (t2 - endTime) / 2)
                cost += keepTime * c3 + c2 / releaseTime
        if s2.bar:
            if s1.bar:
                # keep
                pass
            elif is_pressed(s1, 0):
                # press
                endTime = min(t2, s1.endTime[0])
                pressTime = max((t2 - endTime) / 2, MIN_PRESS_TIME)
                cost += c4 / pressTime
            else:
                pressTime = t2 - t1
                cost += c4 / pressTime
        return cost

    cdef _get_finger_cost(self, State *s1, State *s2, list frame1, list frame2,
            float t1, float t2, int i):
        cdef:
            float endTime, keepTime, releaseTime, pressTime, jumpTime
            float cost = 0
        if is_pressed(s1, i):
            endTime = min(t2, s1.endTime[i])
        # Add normal keep or release cost
        if not (s1.bar and i == 0) and is_pressed(s1, i):
            if is_pressed(s2, i):
                if s1.strings[i] == s2.strings[i] and s1.frets[i] == s2.frets[i]:
                    # keep
                    keepTime = t2 - t1
                    cost += keepTime * a3
                else:
                    # will press after release
                    releaseTime = max(MIN_RELEASE_TIME, (t2 - endTime) / 2)
                    cost += a2 / releaseTime
            else:
                # will not press after release
                releaseTime = max(MIN_RELEASE_TIME, t2 - endTime)
                cost += a2 / releaseTime
        # Add normal press cost
        if not (s2.bar and i == 0) and is_pressed(s2, i):
            if s1.bar and i == 0:
                # press after release
                pressTime = max((t2 - endTime) / 2, MIN_PRESS_TIME)
                cost += a4 / pressTime
            elif is_pressed(s1, i):
                if s1.strings[i] == s2.strings[i] and s1.frets[i] == s2.frets[i]:
                    # keep
                    pass
                else:
                    # press after release
                    pressTime = max((t2 - endTime) / 2, MIN_PRESS_TIME)
                    cost += a4 / pressTime
            else:
                # free press
                pressTime = t2 - t1
                cost += a4 / pressTime
        # Add finger move cost
        if is_pressed(s1, i) and is_pressed(s2, i):
            jumpTime = max(MIN_JUMP_TIME, t2 - endTime)
            cost += (
                    b1 * sqr((s1.frets[i] - s1.frets[0]) - (s2.frets[i] - s2.frets[0])) + \
                    b2 * sqr(s1.strings[i] - s2.strings[i])
                ) / jumpTime
        return cost

    def get_cost(self, int stateIdx1, int stateIdx2, float minCost):
        cdef:
            float dt, totalCost, end1, keepTime, releaseTime, jumpTime
            int i, j
            int missed
            State *s1, *s2

        assert stateIdx1 < self.nStates[0]
        assert stateIdx2 < self.nStates[1]
        s1 = &self.states[0][stateIdx1]
        s2 = &self.states[1][stateIdx2]
        t1 = self.times[0]
        t2 = self.times[1]
        dt = float(t2 - t1)
        totalCost = 0
        frame1 = self.frames[0]
        frame2 = self.frames[1]

        missed = s2.missed
        for i in range(len(frame2)):
            if s2.fingers[i] == -1:
                continue
            j = self.frameMatch[i]
            if j == -1:
                continue
            if s2.fingers[i] != s1.fingers[j]:
                missed += 1
        totalCost += MISS_PENALTY * missed
        # Index fret change cost
        if s1.frets[0] != s2.frets[0]:
            totalCost += get_fret_change_cost(s1.frets[0], s2.frets[0], dt)
        if totalCost >= minCost:
            return totalCost
        # Add reusing same string cost
        for i in range(N_STRINGS):
            if s2.rings[i] >= 0 and s1.rings[i] >= 0 and s1.rings[i] != s2.rings[i]:
                totalCost += SAME_STRING_PENALTY
        if totalCost >= minCost:
            return totalCost
        for i in range(s2.bar, N_FINGERS):
            if totalCost >= minCost:
                return totalCost
            totalCost += self._get_finger_cost(s1, s2, frame1, frame2, t1, t2, i)
            if is_pressed(s2, i) and s2.frets[i] > HIGH_FRET:
                totalCost += (s2.frets[i] - HIGH_FRET) * HIGH_FRET_PENALTY
        totalCost += self._get_bar_cost(s1, s2, frame1, frame2, t1, t2)

        return totalCost
