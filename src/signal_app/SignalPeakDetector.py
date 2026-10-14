import numpy as np


class SignalPeakDetector:

    def __init__(self, noise_multiplier: float = 6.0):
        # Kerroin, jolla säädetään piikkien tunnistamisen herkkyyttä.
        # Change peak detection sensitivity
        self.noise_multiplier = noise_multiplier

    def detect_peaks(self, signal: np.ndarray) -> np.ndarray:
        # Muutetaan syöte NumPy-taulukoksi.
        # Change to numpy
        signal = np.asarray(signal, dtype=float)

        # Tarkistetaan, että signaalissa on vähintään kolme näytettä.
        # Check there is atleast 3 signals
        if len(signal) < 3:
            return np.array([], dtype=int)

        # Save
        peaks = []

        # Calculate diffrences
        differences = np.diff(signal)

        #  Calculate median
        median_difference = np.median(differences)

        # Arvioidaan kohinan tasoa erotusten poikkeamien avulla.
        # Estimate the noise level using the deviations of differences.
        deviations = np.abs(differences - median_difference)
        noise_level = np.median(deviations) / (0.6745 * np.sqrt(2))

        # Määritetään piikiltä vaadittava vähimmäiskorkeus ympäristöön nähden.
        # Determine the minimum height required for a peak relative to its surroundings
        threshold = self.noise_multiplier * noise_level

        # Käydään signaali läpi. Reunimmaisia näytteitä ei käsitellä,
        # koska jokaisella tarkasteltavalla näytteellä pitää olla naapurit.

        # Loop through the signal. Border samples are skipped,
        # be  cause each processed sample must have neighbors.
        for i in range(1, len(signal) - 1):

            # Tarkistetaan, onko näyte suurempi kuin molemmat naapurinsa.
            # Check if the sample is greater than both of its neighbors.
            is_local_maximum = (
                signal[i] > signal[i - 1]
                and signal[i] > signal[i + 1]
            )

            # Tarkistetaan, eroaako näyte riittävästi ympäristöstään.
            # Check if the sample differs enough from its surroundings.
            is_high_enough = (
                signal[i] - max(signal[i - 1], signal[i + 1])
                > threshold
            )

            # Tallennetaan indeksi, jos molemmat ehdot täyttyvät.
            # Save the index if both conditions are met.
            if is_local_maximum and is_high_enough:
                peaks.append(i)

        # Palautetaan piikkien indeksit kokonaislukuina.
        # Return the peak indices as integers.
        return np.asarray(peaks, dtype=int)