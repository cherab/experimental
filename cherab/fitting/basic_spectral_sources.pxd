
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
        public ParamType inty, temp, vel, natural_wavelength
        public double cos_angle, weight, inst_sigma

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

