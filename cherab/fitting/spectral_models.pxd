
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
