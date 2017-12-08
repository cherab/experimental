
import numpy as np

from raysect.optical import Ray


cdef class ACXAtPoint(SpectralSource):

    def __init__(self, str name, Point3D origin, Vector3D direction, object pini, Spectrum spectrum):

        self.name = name
        self.origin = origin
        self.direction = direction
        self.nbi_beams = pini.components
        self.spectrum = spectrum

    property fit_parameters:
        def __get__(self):
            return []

    cdef update_parameter_values(self, list parameters):
        pass

    def __call__(self, double[:] wavelengths, double[:] output):
        """ Calculate the Gaussian line at wavelengths $f(/lambda)$

        :param wavelengths: A numpy array of wavelengths.
        :return:
        """
        return self.evaluate(wavelengths, output)

    # TODO - Is this in the wrong place???
    # Maybe we should go direct to the beam's material and ask for local emission. The emission function is not a
    # general property of materials.
    # TODO - this is very slow because using numpy arrays instead of spectrum objects.
    cdef double[:] evaluate(self, double[:] wavelengths, double[:] output):

        cdef:
            Beam beam
            Spectrum spectrum
            double[:] samples
            int i

        # Reset spectral samples
        spectrum = self.spectrum
        for i in range(spectrum.num_samples):
            spectrum.samples[i] = 0.0

        for beam in self.nbi_beams:
            spectrum = beam.emission_function(self.origin, self.direction, spectrum)

        samples = spectrum.to_photons()

        for i in range(spectrum.num_samples):
            output[i] = samples[i]

        return output


cdef class ACXAlongRay(SpectralSource):

    def __init__(self, str name, Point3D origin, Vector3D direction, World world, double min_wavelength = 375,
                 double max_wavelength = 785, int num_samples = 40):

        self.name = name
        self.origin = origin
        self.direction = direction
        self.world = world
        self._min_wavelength = min_wavelength
        self._max_wavelength = max_wavelength
        self._num_samples = num_samples

    property fit_parameters:
        def __get__(self):
            return []

    cdef update_parameter_values(self, list parameters):
        pass

    def __call__(self, double[:] wavelengths, double[:] output):
        """ Calculate the Gaussian line at wavelengths $f(/lambda)$

        :param wavelengths: A numpy array of wavelengths.
        :return:
        """
        return self.evaluate(wavelengths, output)

    # TODO - Is this in the wrong place???
    # Maybe we should go direct to the beam's material and ask for local emission. The emission function is not a
    # general property of materials.
    # TODO - this is very slow because using numpy arrays instead of spectrum objects.
    cdef double[:] evaluate(self, double[:] wavelengths, double[:] output):

        cdef:
            Spectrum sampled_spectrum
            double[:] samples
            int i

        ray = Ray(origin=self.origin, direction=self.direction, min_wavelength=self._min_wavelength,
                  max_wavelength=self._max_wavelength, num_samples=self._num_samples)
        sampled_spectrum = ray.trace(self.world)
        samples = sampled_spectrum.to_photons()

        for i in range(sampled_spectrum.num_samples):
            output[i] = samples[i]

        return output
