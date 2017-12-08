
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
