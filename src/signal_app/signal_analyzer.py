# SPDX-FileCopyrightText: 2023 Joni Hyttinen <joni.hyttinen@uef.fi>
#
# SPDX-License-Identifier: MIT
import csv
import logging
import time
from collections.abc import Iterator
from pathlib import Path
from signal_app.SignalPeakDetector import SignalPeakDetector

import numpy as np
from signal_app.signal_preprocessing import preprocess, good_samples
from PySide6.QtCore import Signal

logger = logging.getLogger(__name__)


class SignalAnalyzer:
     # Vastaa signaalitiedoston lukemisesta, käsittelystä ja käyttöliittymälle välittämisestä.
    """
    A "signal analyzer" that just generates an array of cosine function values.

    Instances of this class are intended to be run from a worker thread. Data
    is emitted through callbacks back to the parent thread.
    """

    def __init__(self):
        # Käsiteltävän signaalitiedoston polku.
        self.signal_path = None
        # Muutetaan tila sellaiseksi, että käsittelysilmukka lopettaa.
        self.running = False
        # Näytettävän signaali-ikkunan koko näytteiden lukumääränä.
        self.window_size = 600 * 50000  # Let's take 10 minutes of the signal
        # Signaali luetaan tiedostosta tämän kokoisina paloina.
        self.chunk_size = 500000  # ten second at a time
        # Nykyisen signaali-ikkunan aloituskohta.
        self.x = 0
        # Signaalin suurin käytettävissä oleva x-arvo.
        self.max_x = 0
        # Taulukko signaalin x-akselin arvoille.
        self.x_array = np.zeros((self.window_size,))
        # Taulukko signaalin y-arvoille; aluksi arvot ovat NaN-arvoja.
        self.y_array = np.zeros_like(self.x_array) * np.nan
        # Alustetaan y-akselin suurin arvo mahdollisimman pieneksi.
        self.y_max = -np.inf  # Use ridiculous values to force y-scale change
        # Alustetaan y-akselin pienin arvo mahdollisimman suureksi.
        self.y_min = np.inf
        # Alustetaan signaalipiikkien tunnistin.
        self.peak_detector = SignalPeakDetector(noise_multiplier=6.0)
        # Tallennetaan tunnistettujen piikkien indeksit.
        self.peak_indices = np.array([], dtype=int)
        # Variables used for navigating through the signal file
        #self.window_start = 0
        #self.window_size = 30_000_000
        #self.signal_length = 0


    def _read_signal(self, chunk_size: int = 1) -> Iterator[np.ndarray]:
        """
        Read the signal file in chunks of `chunk_size` rows.

        This method is a generator, and it should be called iteratively, for
        example in a `while` or a `for` loop. A generator method generates
        values as they are needed. Consequently, this method does not read
        the whole file at once, but only `count` rows that then yielded
        to the outer loop for processing. On the next loop iteration, the
        execution in this method continues after the yield statement.

        :param chunk_size: the number of rows in a chunk
        :return: a numpy array of shape (count, 2). The final array may be shorter.
        """
        if self.signal_path is None or not self.signal_path.is_file():
            # This could raise an exception, but then we would have to handle that somewhere.
            print(f"""signal path "{self.signal_path}" is not a valid file.""")
            return

       # Valitaan tiedoston lukutapa tiedostopäätteen perusteella.
        match self.signal_path.suffix.lower():
            case ".csv":
                with self.signal_path.open("r") as signal_file:
                    reader = csv.reader(signal_file)
                    # skip column headers ("adc1", "adc2")
                    next(reader)
                    data = np.zeros((chunk_size, 2))
                    i = 0
                    for i, row in enumerate(reader):
                        data[i % chunk_size] = np.array(row, np.uint16)
                        if i > 0 and i % chunk_size == 0:
                            yield data
                    if i % chunk_size != 0:
                        yield data[: i % chunk_size]
            case ".npy":
                # Open once to get array length.
                # The array in the file is a flat, single-dimensional array of n elements.
                signal = np.memmap(self.signal_path, dtype=np.int16, order="C")
                # Reopen the file with correct (n // 2, 2) shape.
                signal = np.memmap(
                    self.signal_path,
                    dtype=np.int16,
                    shape=(signal.shape[0] // 2, 2),
                    order="C",
                )
                self.max_x = signal.shape[0]
                # Throw out chunks of requested size
                for chunk_start in range(0, self.max_x, chunk_size):
                    yield signal[chunk_start : chunk_start + chunk_size, :].astype(
                        np.float64
                    )

     # Lukee signaalista tietyn x-koordinaattien välisen osan.
    def _read_signal_at(self, start_x, end_x) -> tuple[np.ndarray, np.ndarray]:
        logger.debug("Read signal from %s to %s", start_x, end_x)
        x_array = np.linspace(start_x, end_x - 1, end_x - start_x)
        if self.signal_path is None or not self.signal_path.is_file():
            logger.error("Signal path '%s' is not a valid file.", self.signal_path)
            return x_array, np.ones((end_x - start_x, 2)) * np.nan

        match self.signal_path.suffix.lower():
            case ".csv":
                logger.error("This function does not support CSV-files.")
                return x_array, np.ones((end_x - start_x, 2)) * np.nan

            case ".npy":
                # Open once to get array length.
                # The array in the file is a flat, single-dimensional array of n elements.
                signal = np.memmap(self.signal_path, dtype=np.int16, order="C")
                # Reopen the file with correct (n // 2, 2) shape.
                signal = np.memmap(
                    self.signal_path,
                    dtype=np.int16,
                    shape=(signal.shape[0] // 2, 2),
                    order="C",
                )

                # Update maximum x value
                self.max_x = signal.shape[0]

                if end_x < self.max_x:
                    # Return a slice from the signal array.
                    return x_array, signal[start_x:end_x, :].astype(np.float64)
                elif start_x < self.max_x:
                    # The signal array has less data than requested.
                    # Return an array of requested size with all data that is still available, rest of it are NaN.
                    y_array = np.ones((end_x - start_x, 2)) * np.nan
                    y_array[: self.max_x - start_x] = signal[start_x:, :]
                    return x_array, y_array
                else:
                    # The request is completely out of bounds
                    # Return an array of requested size, but fill it with NaNs.
                    return x_array, np.ones((end_x - start_x, 2)) * np.nan

            case _:
                # Ok, what did you do or didn't do to get here?
                logger.error("Completely invalid input file?!")
                return x_array, np.ones((end_x - start_x, 2)) * np.nan

    def _start(self, set_axis_y=None):
        """
        Tasks that should be done before starting the main loop of SignalAnalyzer.
        """
        self.running = True
        if set_axis_y:
            set_axis_y.emit(self.y_min, self.y_max)

    def stop(self):
        """
        Stops the main loop.
        """
        self.running = False

    def start(
        self,
        signal_path: Path,
        set_chart_axis_y: Signal | None = None,
        update_chart: Signal | None = None,
        progress_callback: Signal | None = None,
        update_markers: Signal | None = None,
    ):
        """
        Starts the SignalAnalyzer main loop.

        :param signal_path: the path to the signal data file.
        :param set_chart_axis_y: Signal to update chart y-axis.
        :param update_chart: Signal to update a chart.
        :param progress_callback: Signal to update progress meters.
        """

        self.signal_path = signal_path
        self._start(set_chart_axis_y)

        while self.running:
            self.x = -self.window_size

           # Käsitellään signaalia niin kauan kuin käsittely on käynnissä.
            end_time = time.time()
            for chunk in self._read_signal(self.chunk_size):
                # Record iteration start time
                start_time = time.time()
                logger.debug("Chunk read in %.2f ms", (start_time - end_time) * 1000)

                # Break out from the for-loop if `self.running` has changed
                if not self.running:
                    break

                chunk_size = chunk.shape[
                    0
                ]  # No guarantee that `chunk` is always of the requested size
                self.x_array = (
                    np.linspace(self.x, self.x + self.window_size - 1, self.window_size)
                    / 50000
                )
                self.y_array = np.roll(
                    self.y_array, -chunk_size
                )  # roll `self.y_array` back by `chunk_size`
                self.y_array[-chunk_size:] = chunk[
                    :, 1
                ]  # replace the last `chunk_size` entries with `chunk`
                self.x += chunk_size

                
                # Poimitaan signaalista vain kelvolliset arvot.
                valid_indices = np.flatnonzero(np.isfinite(self.y_array))
                valid_signal = self.y_array[valid_indices]

                # Tunnistetaan piikit Jennin algoritmilla.
                detected_indices = self.peak_detector.detect_peaks(valid_signal)
                print("Signaalin suurin arvo:", np.max(valid_signal))
                print("Tunnistettuja piikkejä:", len(detected_indices))

                # Muutetaan indeksit takaisin alkuperäisen signaalin indekseiksi.
                self.peak_indices = valid_indices[detected_indices]
                print("Piikkien indeksit:", self.peak_indices)
                print("Piikkien arvot:", self.y_array[self.peak_indices])
                print("Piikkien ajat:", self.x_array[self.peak_indices])

                
                # Update y-axis min/max if need be:
                self.y_min = (
                    y_min
                    if (y_min := np.nanmin(self.y_array)) < self.y_min
                    else self.y_min
                )
                self.y_max = (
                    y_max
                    if (y_max := np.nanmax(self.y_array)) > self.y_max
                    else self.y_max
                )

                # Uh oh, we quickly notice that plotting this many points is downright unfeasible. Let's reduce the
                # load for visualization by selecting only every 100th value in the arrays.
                # Poimitaan kuvaajaa varten vain joka sadas arvo, jotta piirtäminen olisi kevyempää.
                # HUOM tarviiko tätä kohtaa muokata
                
                  # these array slice views seem to go missing if not copied
                
                # Keep the regularly sampled points for efficient signal visualization.
                # Also include all detected peak indices so that peak locations are
                # not lost when the signal is downsampled for plotting.
                vis_indices = np.unique(
                    np.concatenate((
                        np.arange(0, len(self.y_array), 100),
                        self.peak_indices
                    ))
                )

                vis_x = self.x_array[vis_indices].copy()
                vis_y = self.y_array[vis_indices].copy()


                logger.debug(
                    "Computation done in %.2f ms", (time.time() - start_time) * 1000
                )

                # This signal will now block the thread until chart update is finished
                if set_chart_axis_y:
                    set_chart_axis_y.emit(self.y_min, self.y_max)

                # This signal will now block the thread until chart update is finished
                if update_chart:
                    update_chart.emit(vis_x, vis_y)

            # Lähetetään tunnistettujen piikkien sijainnit kuvaajalle.
            # Send the detected peak coordinates to the chart markers.
                if update_markers:
                    peak_x = self.x_array[self.peak_indices]
                    peak_y = self.y_array[self.peak_indices]
                    update_markers.emit(peak_x, peak_y)

                if progress_callback:
                    # This loop has no meaningful progress.
                    # Just report the current left-most x-coordinate.
                    progress_callback.emit(self.x)

                end_time = time.time()
                logger.debug(
                    "Finished iteration in %.2f ms", (end_time - start_time) * 1000
                )

            # Stop while-loop when the file ends
            self.running = False

    # Lataa tietyn signaali-ikkunan ja välittää sen kuvaajalle.
    def load_window(
        self,
        start_x: int,
        end_x: int,
        signal_path: Path,
        set_chart_axis_y: Signal | None = None,
        update_chart: Signal | None = None,
        progress_callback: Signal | None = None,
        update_markers: Signal | None = None,
    ):
        """Loads a signal window.
        :param start_x: the start of the window
        :param end_x: the end of the window
        :param signal_path: the path to the signal data file.
        :param set_chart_axis_y: Signal to update chart y-axis.
        :param update_chart: Signal to update a chart.
        :param progress_callback: Signal to update progress meters."""
        self.signal_path = signal_path
        if set_chart_axis_y:
            set_chart_axis_y.emit(self.y_min, self.y_max)

        self.x_array, self.y_array = self._read_signal_at(start_x, end_x)
        self.x_array = self.x_array / 50000
        self.y_array = self.y_array[:, 1]

        # PREPROCESSING: baseline removal (smoothing is off until the detector is updated)
        self.y_array = preprocess(self.y_array, smooth_ms=0)

        # DETECTION ALGORITHM(s)
        # Tunnistetaan signaalipiikit nykyisestä signaali-ikkunasta.
        # Regocnize signalpeaks from signal window 

        # Print the indices and coordinates of detected peaks
        # to help verify the detection results.

        self.peak_indices = self.peak_detector.detect_peaks(self.y_array)
        # Ignore peaks near the edges, where the filter is not reliable
        reliable = good_samples(self.y_array)
        self.peak_indices = self.peak_indices[reliable[self.peak_indices]]
        print("detected peaks:", len(self.peak_indices))
        print("peak indices:", self.peak_indices)
        print("peak cordinates:",
        list(zip(
        self.x_array[self.peak_indices],
        self.y_array[self.peak_indices]
    ))
)


        logger.debug(
            "Detected %d peaks in the current signal window.",
            len(self.peak_indices),
        )
        #Test comment
    

        # Tähän kohtaan voidaan lisätä signaalin tunnistusalgoritmit!!!
        # DETECTION ALGORITHMS

        # Update y-axis min/max if need be:
        self.y_min = (
            y_min if (y_min := np.nanmin(self.y_array)) < self.y_min else self.y_min
        )
        self.y_max = (
            y_max if (y_max := np.nanmax(self.y_array)) > self.y_max else self.y_max
        )

        vis_indices = np.unique(
        np.concatenate((
        np.arange(0, len(self.y_array), 100),
        self.peak_indices
    ))
)

        vis_x = self.x_array[vis_indices].copy()
        vis_y = self.y_array[vis_indices].copy()

        if set_chart_axis_y:
            set_chart_axis_y.emit(self.y_min, self.y_max)

        if update_chart:
            update_chart.emit(vis_x, vis_y)

                
        if update_markers:
            peak_x = self.x_array[self.peak_indices]
            peak_y = self.y_array[self.peak_indices]
            update_markers.emit(peak_x, peak_y)  
           


        # Tähän kohtaan voidaan lisätä tunnistuskohtia kuvaajaan!!!
        # ADD MARKERS TO CHART

    # Lataa parhaillaan valittuna olevan signaali-ikkunan uudelleen.
    def current_window(
        self,
        signal_path: Path,
        set_chart_axis_y: Signal | None = None,
        update_chart: Signal | None = None,
        progress_callback: Signal | None = None,
        update_markers: Signal | None = None,
    ):
        self.load_window(
            self.x,
            self.x + self.window_size,
            #self.window_start,
            #self.window_start + self.window_size,
            signal_path,
            set_chart_axis_y,
            update_chart,
            progress_callback,
            update_markers,
        )

        # ikkunoiden käsittely

    def previous_window(
        self,
        signal_path: Path,
        set_chart_axis_y: Signal | None = None,
        update_chart: Signal | None = None,
        progress_callback: Signal | None = None,
        update_markers: Signal | None = None,
    ):
        self.x = self.x - self.window_size if self.x > self.window_size else 0
        #self.window_start = max(0, self.window_start - self.window_size)
        self.load_window(
            self.x,
            self.x + self.window_size,
            #self.window_start,
            #self.window_start + self.window_size,
            signal_path,
            set_chart_axis_y,
            update_chart,
            progress_callback,
            update_markers,
        )

    def next_window(
        self,
        signal_path: Path,
        set_chart_axis_y: Signal | None = None,
        update_chart: Signal | None = None,
        progress_callback: Signal | None = None,
        update_markers: Signal | None = None,
    ):
        self.x = (
            self.x + self.window_size
            if self.max_x > self.x + self.window_size
            else self.x)
        #if self.has_next_window():
        #    self.window_start += self.window_size

        self.load_window(
            self.x,
            self.x + self.window_size,
            #self.window_start,
            #self.window_start + self.window_size,
            signal_path,
            set_chart_axis_y,
            update_chart,
            progress_callback,
            update_markers,
        )

    def has_previous_window(self):
        return self.x > 0
        #return self.window_start > 0

    def has_next_window(self):
        return self.x < self.max_x - self.window_size
        #return self.window_start + self.window_size < self.signal_length
