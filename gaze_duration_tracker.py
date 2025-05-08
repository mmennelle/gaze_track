import time

class GazeDurationTracker:
    def __init__(self, max_idle_time=0.3):
        self.object_timers = {}  # {object_id: [start_time, last_seen_time]}
        self.max_idle_time = max_idle_time

    def update(self, object_id):
        now = time.time()
        if object_id not in self.object_timers:
            self.object_timers[object_id] = [now, now]
        else:
            start, last_seen = self.object_timers[object_id]
            if now - last_seen < self.max_idle_time:
                self.object_timers[object_id][1] = now
            else:
                # Reset if gaze was broken
                self.object_timers[object_id] = [now, now]

    def get_duration(self, object_id):
        now = time.time()
        if object_id not in self.object_timers:
            return 0.0
        start, last_seen = self.object_timers[object_id]
        if now - last_seen > self.max_idle_time:
            return 0.0
        return last_seen - start

    def reset(self, object_id):
        if object_id in self.object_timers:
            del self.object_timers[object_id]

    def reset_all(self):
        self.object_timers.clear()
