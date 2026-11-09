import numpy as np


class TumorMeasurement:

    def __init__(self):
        self.tumor_size = 0

    def calculate_tumor_size(self, detected_peaks: np.ndarray) -> int:
        """
        Calculate tumor measurement based on detected peaks.
        """
        self.tumor_size = len(detected_peaks)
        return self.tumor_size

    def calculate_tumor_size(self, detected_peaks: np.ndarray) -> int:
        """
        Calculate tumor measurement based on detected peaks.
        """
        tumor_size = len(detected_peaks)
        print("TumorMeasurement method called:", tumor_size)
        return tumor_size