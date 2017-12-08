
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

cimport numpy as cnp

from cherab.fitting.spectra cimport BaseSpectra
from cherab.fitting.basic_spectral_sources cimport SpectralSource
from cherab.fitting.fit_parameters cimport FreeParameter

cdef class SingleSpectraModel:

    cdef:
        public list sources_to_fit
        public BaseSpectra spectra_to_fit
        public double dof, _logfact, _errfact
        public FreeParameter offset_parameter
        public int num_of_sources, ndata, _nd
        public double[:] _samples, _evaluation, _local_wvls, _xvec_reference, _xvec_working

    cpdef update_parameters(self, list parameters)

    cpdef double evaluate_chi2(self)

    cpdef double[:] evaluate_spectrum(self)

    cdef double loglikelihood(self)


cdef class MultiSpectraModel:

    cdef:
        public list spectral_models
        public list profiles
        public float dof

    cdef double evaluate_loglikelihood(self)

    cdef double evaluate_chi2(self)
