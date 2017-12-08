
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


cdef class BaseSpectra:

    cdef public int _px0, _px1, num_wvl_bins

    cdef double[:] get_wavelengths(self)

    cdef double[:] get_spectral_samples(self)

    cdef double[:] get_errors(self)

    cdef void flush_spectral_windows(self)


cdef class Spectra(BaseSpectra):

    cdef:
        public str name
        public double[:] _wavelengths, _wavelength_window, _raw_wavelengths
        public double[:] _samples, _samples_window, _errors, _errors_window
        public tuple background_range
        public double inst_sigma, _wvl_offset

    cdef double[:] get_wavelengths(self)

    cdef double[:] get_spectral_samples(self)

    cdef double[:] get_errors(self)

    cdef void flush_spectral_windows(self)


cdef class TimeSeriesSpectra(BaseSpectra):

    cdef:
        public str name
        public double[:] _wavelengths, _wavelength_window, _raw_wavelengths, _samples_window, _errors_window
        public double[:] times
        public double[:,:] _samples, _errors
        public tuple background_range
        public double inst_sigma, _wvl_offset
        public int _time_index

    cdef double[:] get_wavelengths(self)

    cdef double[:] get_spectral_samples(self)

    cdef double[:] get_errors(self)

    cdef void flush_spectral_windows(self)
