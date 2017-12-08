

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
