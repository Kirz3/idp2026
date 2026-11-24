
import numpy as np


class TumorMeasurement:

    def __init__(
        self,
        sample_rate: int = 50_000,
        calibration_area_200ml: float | None = None
    ):
        self.sample_rate = sample_rate

        # Total peak area measured from a known 200 ml sample.
        self.calibration_area_200ml = calibration_area_200ml

        self.tumor_measurement = {}

    def calculate_tumor_size(
        self,
        signal: np.ndarray,
        detected_peaks: np.ndarray,
        peak_properties: dict
    ) -> dict:

        peak_count = len(detected_peaks)

        if peak_count == 0:
            self.tumor_measurement = {
                "peak_count": 0,
                "average_peak_height": 0.0,
                "average_peak_width_ms": 0.0,
                "total_peak_area": 0.0,
                "estimated_tumor_size_ml": 0.0
            }
            return self.tumor_measurement

        # Get peak heights and widths from the peak detector.
        peak_heights = peak_properties["prominences"]
        peak_widths_ms = peak_properties["widths_ms"]

        # Calculate average peak height.
        average_peak_height = float(np.mean(peak_heights))

        # Calculate average peak width in milliseconds.
        average_peak_width_ms = float(np.mean(peak_widths_ms))

        # Calculate the total area of detected peaks.
        total_peak_area = 0.0

        for i in range(peak_count):
            peak_index = detected_peaks[i]

            left_index = max(
                int(np.floor(peak_properties["left_ips"][i])),
                0
            )

            right_index = min(
                int(np.ceil(peak_properties["right_ips"][i])),
                len(signal) - 1
            )

            # Calculate the local baseline using peak prominence.
            baseline = signal[peak_index] - peak_heights[i]

            # Get the signal section belonging to this peak.
            peak_signal = signal[left_index:right_index + 1]

            # Calculate the area above the local baseline.
            corrected_signal = np.maximum(
                peak_signal - baseline,
                0
            )

            peak_area = np.trapezoid(
                corrected_signal,
                dx=1 / self.sample_rate
            )

            total_peak_area += peak_area

        # Estimate volume using the known 200 ml calibration sample.
        estimated_tumor_size_ml = 0.0

        if (
            self.calibration_area_200ml is not None
            and self.calibration_area_200ml > 0
        ):
            estimated_tumor_size_ml = (
                total_peak_area / self.calibration_area_200ml
            ) * 200

        self.tumor_measurement = {
            "peak_count": peak_count,
            "average_peak_height": average_peak_height,
            "average_peak_width_ms": average_peak_width_ms,
            "total_peak_area": float(total_peak_area),
            "estimated_tumor_size_ml": float(estimated_tumor_size_ml)
        }

        print("Tumor measurement:", self.tumor_measurement)

        return self.tumor_measurement
