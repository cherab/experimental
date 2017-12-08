
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from cherab.fitting import Parameter
from cherab.fitting.fit_parameters cimport Parameter as ParamType, FreeParameter as FreeParamType


cdef class SpectralSource:

    cdef:
        public str name

    cdef double[:] evaluate(self, double[:] wavelengths, double[:] output)

    cdef update_parameter_values(self, list parameters)


cdef class Baseline(SpectralSource):

    cdef:
        public ParamType inty

    cdef update_parameter_values(self, list parameters)

    cdef double[:] evaluate(self, double[:] wavelengths, double[:] output)


cdef class LinearBaseline(SpectralSource):

    cdef:
        public double wvl_offset
        public ParamType inty
        public ParamType gradient

    cdef update_parameter_values(self, list parameters)

    cdef double[:] evaluate(self, double[:] wavelengths, double[:] output)


cdef class GaussianLine(SpectralSource):

    cdef:
        public ParamType inty, temp, wvl
        public double weight, inst_sigma

    cdef update_parameter_values(self, list parameters)

    cdef double[:] evaluate(self, double[:] wavelengths, double[:] output)


cdef class DopplerShiftedLine(SpectralSource):

    cdef:
        public ParamType inty, temp, vel
        public double natural_wavelength, cos_angle, weight, inst_sigma

    cdef update_parameter_values(self, list parameters)

    cdef double[:] evaluate(self, double[:] wavelengths, double[:] output)


cdef class SkewGaussianLine(SpectralSource):

    cdef:
        public double weight
        public ParamType inty, temp, wvl, alpha

    cdef update_parameter_values(self, list parameters)

    cdef double[:] evaluate(self, double[:] wavelengths, double[:] output)


cdef class GaussianFunction(SpectralSource):

    cdef public ParamType inty, sigma, centre_point

    cdef update_parameter_values(self, list parameters)

    cdef double[:] evaluate(self, double[:] wavelengths, double[:] output)

