import time
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class VerificationTracker:
    def __init__(self, max_disappeared: int = 5, window_representations: int = 10):
        self.window_representations = window_representations
        self.nextObjectID = 0
        self.disappeared = OrderedDict()
        self.rects = OrderedDict()
        self.representations = OrderedDict()
        self.representations_list = OrderedDict()
        self.maxDisappeared = max_disappeared
        self.name = OrderedDict()

    def register(self, rects, representation):
        self.rects[self.nextObjectID] = np.array(rects)
        self.representations[self.nextObjectID] = representation
        self.representations_list[self.nextObjectID] = [representation]
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, object_id):
        del self.disappeared[object_id]
        del self.rects[object_id]
        del self.representations[object_id]
        del self.representations_list[object_id]
        if self.rects:
            self.nextObjectID = max(self.rects.keys()) + 1
        else:
            self.nextObjectID = 0

    def update(self, rects, representations):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                if self.disappeared[object_id] == 0:
                    self.disappeared[object_id] = time.time()
                elif time.time() - self.disappeared[object_id] > self.maxDisappeared:
                    self.deregister(object_id)
            return self.rects, self.representations

        if len(self.rects) == 0:
            for i in range(len(rects)):
                self.register(rects[i], representations[i])
        else:
            objects_id = list(self.rects.keys())
            objects_representation = list(self.representations.values())

            d = dist.cdist(np.array(objects_representation), representations)

            rows = d.min(axis=1).argsort()
            cols = d.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = objects_id[row]
                self.rects[object_id] = rects[col]
                self.representations[object_id] = representations[col]
                self.representations_list[object_id].append(representations[col])
                if len(self.representations_list[object_id]) > self.window_representations:
                    self.representations_list[object_id].pop(0)
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, d.shape[0])).difference(used_rows)
            unused_cols = set(range(0, d.shape[1])).difference(used_cols)

            if d.shape[0] >= d.shape[1]:
                for row in unused_rows:
                    object_id = objects_id[row]
                    if self.disappeared[object_id] == 0:
                        self.disappeared[object_id] = time.time()
                    if time.time() - self.disappeared[object_id] > self.maxDisappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(rects[col], representations[col])

        return self.rects, self.representations

    def get_representations_list(self):
        return self.representations_list

    def set_name(self, name: str, object_id: int):
        self.name[object_id] = name

    def get_name(self, object_id: int):
        return self.name.get(object_id, None)


class CentroidTracker:
    def __init__(self, max_disappeared: int = 50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.rects = OrderedDict()
        self.maxDisappeared = max_disappeared

    def register(self, centroid, rects):
        self.objects[self.nextObjectID] = centroid
        self.rects[self.nextObjectID] = np.array(rects)
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.rects[object_id]

        if self.objects:
            self.nextObjectID = max(self.objects.keys()) + 1
        else:
            self.nextObjectID = 0

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.maxDisappeared:
                    self.deregister(object_id)
            return self.objects, self.rects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            c_x = int((startX + endX) / 2.0)
            c_y = int((startY + endY) / 2.0)
            input_centroids[i] = (c_x, c_y)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i], rects[i])

        else:
            objects_id = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            d = dist.cdist(np.array(object_centroids), input_centroids)
            rows = d.min(axis=1).argsort()
            cols = d.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = objects_id[row]
                self.objects[object_id] = input_centroids[col]
                self.rects[object_id] = rects[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, d.shape[0])).difference(used_rows)
            unused_cols = set(range(0, d.shape[1])).difference(used_cols)

            if d.shape[0] >= d.shape[1]:
                for row in unused_rows:
                    object_id = objects_id[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.maxDisappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col], rects[col])

        return self.objects, self.rects
