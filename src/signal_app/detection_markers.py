
import numpy as np


class DetectionMarkers:
    def __init__(self, axes):
        self.markers = axes.scatter(
            [],
            [],
            color="red",
            marker="x",
            label="Detections",
        )

    def update(self, x: np.ndarray, y: np.ndarray):
        if len(x) == 0:
            self.markers.set_offsets(np.empty((0, 2)))
        else:
            self.markers.set_offsets(
                np.column_stack((x, y))
            )
