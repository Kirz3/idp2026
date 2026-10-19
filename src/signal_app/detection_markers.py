
import numpy as np


class DetectionMarkers:
    def __init__(self, axes):
        self.markers = axes.scatter(
    # Create a scatter plot for detected peaks.
     # Circular markers make the detected points easy to see.
    [],
    [],
    s=55,
    facecolors="white",
    edgecolors="red",
    linewidths=1.8,
    marker="o",
    label="Detections",
    zorder=5,
)
        

    def update(self, x: np.ndarray, y: np.ndarray):
        # Update the marker positions when new detections are available.
        if len(x) == 0:
             # Clear old markers if no peaks were detected.
            self.markers.set_offsets(np.empty((0, 2)))
        else:
            self.markers.set_offsets(
                 # Combine x and y coordinates into (x, y) pairs.
                np.column_stack((x, y))
            )
