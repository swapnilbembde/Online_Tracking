import numpy as np
from collections import OrderedDict


class TrackNumbered(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class TrackState(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackNumbered.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0


    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        TrackState._count += 1
        return TrackState._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackNumbered.Lost

    def mark_removed(self):
        self.state = TrackNumbered.Removed

