
cimport numpy as np

from raysect.core.scenegraph.world cimport World
from raysect.core.math.point cimport Point3D
from raysect.core.math.vector cimport Vector3D
from raysect.optical.spectrum cimport Spectrum

# Internal Imports
from cherab.core cimport Beam
from cherab.fitting.basic_spectral_sources cimport SpectralSource


cdef class ACXAtPoint(SpectralSource):
    cdef:
        public Point3D origin
        public Vector3D direction
        public list nbi_beams
        public Spectrum spectrum

    cdef update_parameter_values(self, list parameters)

    cdef double[:] evaluate(self, double[:] wavelengths, double[:] output)


cdef class ACXAlongRay(SpectralSource):

    cdef:
        public Point3D origin
        public Vector3D direction
        public World world
        public double _min_wavelength
        public double _max_wavelength
        public double _num_samples

    cdef update_parameter_values(self, list parameters)

    cdef double[:] evaluate(self, double[:] wavelengths, double[:] output)
