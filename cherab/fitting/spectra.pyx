
# Copyright 2014-2017 United Kingdom Atomic Energy Authority
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.

# External imports
import numpy as np
import matplotlib.pyplot as plt


cdef class BaseSpectra:

    cdef double[:] get_wavelengths(self):
        raise NotImplementedError()

    cdef double[:] get_spectral_samples(self):
        raise NotImplementedError()

    cdef double[:] get_errors(self):
        raise NotImplementedError()

    cdef void flush_spectral_windows(self):
        raise NotImplementedError()

    def set_active_window(self, window=None):
        """ Set the wavelength window.

        :param window: None or tuple, of either wavelength or index ranges.
        """

        if not window:
            self._px0 = 0
            self._px1 = -1
            self.flush_spectral_windows()
            return

        min_wvl, max_wvl = window

        if min_wvl < max_wvl:

            # Assume they are wavelength ranges
            if isinstance(min_wvl, float) and isinstance(max_wvl, float):
                self._px0 = self.nearest_wvl_index(min_wvl)
                self._px1 = self.nearest_wvl_index(max_wvl)
                self.flush_spectral_windows()

            # Assume inputs are pixel indicies
            else:
                self._px0 = min_wvl
                self._px1 = max_wvl
                self.flush_spectral_windows()

        elif min_wvl == 0 and max_wvl == -1:
            self._px0 = 0
            self._px1 = -1
            self.flush_spectral_windows()
        else:
            raise ValueError("Invalid syntax for setting active window, {}. Must be a None or a tuple of wavelength"
                             " indicies or index indicies.".format(window))

    def nearest_wvl_index(self, wavelength, in_active_window=False):
        """
        Get the index of the wavelength/sample/sigma arrays closest to the given wavelength argument.

        :param float wavelength: The wavelength value to lookup the index for.
        :param boolean in_active_window: If true, the index returned is relative to the active wavelength window.
        :return int: The integer index closest to the input wavelength.
        """

        if in_active_window:
            if not self.wavelengths[0] < wavelength < self.wavelengths[-1]:
                raise ValueError("Wavelength {} is not inside the active wavelength "
                                 "range of this spectra.".format(wavelength))
            return np.abs(np.array(self.wavelengths) - wavelength).argmin()

        else:
            if not self._wavelengths[0] < wavelength < self._wavelengths[-1]:
                raise ValueError("Wavelength {} is not inside the active wavelength "
                                 "range of this spectra.".format(wavelength))
            return np.abs(np.array(self._wavelengths) - wavelength).argmin()

    def max_pair_in_range(self, wvl_range=None):

        if wvl_range:
            min_wvl, max_wvl = wvl_range
            if not (self.wavelengths[0] <= min_wvl and max_wvl <= self.wavelengths[-1]):
                raise ValueError("Wavelength range ({}, {}) is not inside the active wavelength "
                                 "range of this spectra.".format(min_wvl, max_wvl))
            il, iu = (self.nearest_wvl_index(min_wvl, in_active_window=True), self.nearest_wvl_index(max_wvl, in_active_window=True))
            samples = self.samples[il:iu]
            wavelengths = self.wavelengths[il:iu]
        else:
            samples = self.samples
            wavelengths = self.wavelengths
        max_index = np.argmax(samples)

        return samples[max_index], wavelengths[max_index], max_index

    def plot(self):
        """ Plot spectra at current time slice"""
        plt.clf()
        plt.errorbar(self.wavelengths, self.samples, self.errors, label='data')
        plt.title(self.name)


cdef class Spectra(BaseSpectra):

    def __init__(self, wavelengths, spectral_samples, errors, double inst_sigma=0.0,
                 str name='', tuple active_window=None, tuple background_range=None):
        self.name = name
        self._wavelengths = np.array(wavelengths, np.float64)
        self._wavelength_window = self._wavelengths
        self._raw_wavelengths = np.array(wavelengths, np.float64)
        self._samples = np.array(spectral_samples, np.float64)
        self._samples_window = self._samples
        self._errors = np.array(errors, np.float64)
        self._errors_window = self._errors
        self.inst_sigma = inst_sigma
        self._wvl_offset = 0.0
        self.num_wvl_bins = len(wavelengths)

        # indicies for wavelength range
        self._px0 = 0
        self._px1 = -1
        self.set_active_window(active_window)
        self.background_range = background_range

    property wavelengths:
        def __get__(self):
            return self._wavelength_window

    cdef double[:] get_wavelengths(self):
        return self._wavelength_window

    property samples:
        def __get__(self):
            return self._samples_window

    cdef double[:] get_spectral_samples(self):
        return self._samples_window

    property errors:
        def __get__(self):
            return self._errors_window

    cdef double[:] get_errors(self):
        return self._errors_window

    property active_window:
        def __get__(self):
            return self._px0, self._px1

    # Apply a positive offset correction to the wavelength vector of this spectra.
    property wvl_offset:
        def __get__(self):
            return self._wvl_offset
        def __set__(self, offset):
            self._wavelengths += offset
            self._wvl_offset = offset
            self.flush_spectral_windows()

    cdef void flush_spectral_windows(self):
        self._wavelength_window = self._wavelengths[self._px0:self._px1]
        self._samples_window = self._samples[self._px0:self._px1]
        self._errors_window = self._errors[self._px0:self._px1]
        self.num_wvl_bins = self._wavelength_window.shape[0]


cdef class TimeSeriesSpectra(BaseSpectra):

    def __init__(self, wavelengths, spectral_samples, errors, times,
                 double inst_sigma=0.0, str name=None, tuple active_window=None, tuple background_range=None):

        if spectral_samples.shape != errors.shape:
            raise ValueError("The input spectra and their errors have incompatible shapes.")
        elif wavelengths.shape[0] != spectral_samples.shape[1] or spectral_samples.shape[1] != errors.shape[1]:
            raise ValueError("The input spectra and their errors have unequal length wavelength axes.")
        elif spectral_samples.shape[0] != errors.shape[0] or errors.shape[0] != times.shape[0]:
            raise ValueError("The input spectra and their errors have unequal length time axes.")

        self.name = name
        self._wavelengths = np.array(wavelengths, np.float64)
        self._wavelength_window = self._wavelengths
        self._raw_wavelengths = np.array(wavelengths, np.float64)
        self._samples = np.array(spectral_samples, np.float64)
        self._samples_window = self._samples[0, :]
        self._errors = np.array(errors, np.float64)
        self._errors_window = self._errors[0, :]
        self.inst_sigma = inst_sigma
        self._wvl_offset = 0.0
        self.num_wvl_bins = len(wavelengths)

        # indicies for wavelength range
        self._px0 = 0
        self._px1 = -1
        self.set_active_window(active_window)
        self.background_range = background_range

        self.times = np.array(times, np.float64)
        self._time_index = 0

    property wavelengths:
        def __get__(self):
            return self._wavelength_window

    cdef double[:] get_wavelengths(self):
        return self._wavelength_window

    property samples:
        def __get__(self):
            return self._samples_window

    cdef double[:] get_spectral_samples(self):
        return self._samples_window

    property errors:
        def __get__(self):
            return self._errors_window

    cdef double[:] get_errors(self):
        return self._errors_window

    property active_window:
        def __get__(self):
            return self._px0, self._px1

    # Apply a positive offset correction to the wavelength vector of this spectra.
    property wvl_offset:
        def __get__(self):
            return self._wvl_offset
        def __set__(self, offset):
            self._wavelengths += offset
            self._wvl_offset = offset
            self.flush_spectral_windows()

    cdef void flush_spectral_windows(self):
        self._wavelength_window = self._wavelengths[self._px0:self._px1]
        self._samples_window = self._samples[self._time_index, self._px0:self._px1]
        self._errors_window = self._errors[self._time_index, self._px0:self._px1]
        self.num_wvl_bins = self._wavelength_window.shape[0]

    property time:
        def __get__(self):
            return self.times[self._time_index]

    property time_index:
        def __get__(self):
            return self._time_index

    def move_time_curser_to(self, time):

        if not self.times[0] <= time <= self.times[-1]:
            raise ValueError("Time {} is outside of the time axis range.".format(time))

        self._time_index = self._get_index_closest_to_time(time)
        self.flush_spectral_windows()

    def move_to_next_time(self):
        self._time_index += 1
        self.flush_spectral_windows()

    def move_to_previous_time(self):
        self._time_index -= 1
        self.flush_spectral_windows()

    def _get_index_closest_to_time(self, time):
        return [np.abs(np.array(self.times) - time).argmin()][0]

