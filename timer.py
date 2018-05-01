import time

class Timer(object):

    def __init__(self):
        self._start_time = -1
        self._lap = -1
        self.lap_duration = -1
        self.total_duration = -1
        self._lap_time = -1
        self.start()

    def start(self):
        self._start_time = time.time()
        self._lap_time = time.time()
        return self

    def lap(self):
        duration = time.time() - self._start_time
        self.lap_duration = duration
        self._lap_time = time.time()
        return self.lap_duration

    def stop(self):
        duration = time.time() - self._start_time
        self.total_duration = duration
        self._start_time = -1
        self._lap_time = -1
        return self.total_duration

    def pprint_lap(self, suffix=True):
        return Timer.pprint_time(self.lap(),suffix=suffix)

    def pprint_stop(self, suffix=True):
        return Timer.pprint_time(self.stop(),suffix=suffix)

    @staticmethod
    def pprint_time(duration, suffix=True):
        m, s = divmod(duration, 60)
        h, m = divmod(m, 60)
        res = "%d:%02d:%02d" % (h, m, s)
        if suffix:
            if h > 0:
                suffix = " hours"
            elif m > 0:
                suffix = " min"
            else:
                suffix = " sec"
            return res + suffix
        return res