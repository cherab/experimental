
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

## cython: profile=True
# cython: cdivision=True
# cython: boundscheck=False

# external python imports
from scipy.constants import elementary_charge, speed_of_light, pi
from libc.math cimport sqrt, exp, erfc

cdef public double AMU = 1.66053892e-27
cdef public double ELEMENTARY_CHARGE = elementary_charge
cdef public double SPEED_OF_LIGHT = speed_of_light
cdef public double PI = pi
cdef public double DOF = 512.0 - 16.0


cdef class SpectralSource:

    property fit_parameters:
        def __get__(self):
            raise NotImplementedError()

    def __call__(self, double[:] wavelengths, double[:] output):
        """ Evaluate this SpectralSource for wavelength array lambda.

        :param ndarray wavelengths: A numpy array of wavelengths.
        :param ndarray output: A numpy array for the output samples.
        :return:
        """
        return self.evaluate(wavelengths, output)

    cdef double[:] evaluate(self, double[:] wavelengths, double[:] output):
        raise NotImplementedError("Virtual method __call__() not implemented. "
                                  "All derived SpectralSource classes must be callable.")

    def __repr__(self):
        parameters = [parameter for parameter in self.fit_parameters]
        rep_string = "Spectral source = {}/n".format(self.name)
        rep_string += ",".join(["{}={}".format(name, value) for name, value in parameters])
        return rep_string

    cdef update_parameter_values(self, list parameters):
        """ Update the FitParameter instances from a list of values
        :param parameters: The list object from which to de-serialise the data from.
        """
        raise NotImplementedError()


cdef class Baseline(SpectralSource):

    def __init__(self, str name, intensity):
        """ Create a Baseline instance
        :param str name: Name of this baseline source.
        :param ParamType intensity: Parameter representing the baseline instensity value.
        """

        self.name = name
        self.inty = intensity

    property fit_parameters:
        def __get__(self):
            return [self.inty]

    cdef update_parameter_values(self, list parameters):
        cdef FreeParamType param
        if self.inty.free:
            param = self.inty
            param.cset_normalised_value(parameters.pop(0))

    def __call__(self, double[:] wavelengths, double[:] output):
        """ f(lamda) = a, where a is a constant.

        :param wavelengths: array of wavelengths
        :return:
        """
        return self.evaluate(wavelengths, output)

    cdef double[:] evaluate(self, double[:] wavelengths, double[:] output):

        cdef int i, n = wavelengths.shape[0]
        cdef double value = self.inty.get_value()

        for i in range(n):
            output[i] = value
        return output


cdef class LinearBaseline(SpectralSource):

    def __init__(self, str name, wvl_offset, intensity, gradient):
        """ Create a Baseline instance
        :param str name: Name of this baseline source.
        :param float wvl_offset: Float value of reference wavelength around which baseline is calculated.
        :param ParamType intensity: Parameter representing the baseline instensity value.
        :param ParamType gradient: Parameter representing the baseline gradient value.
        """

        self.name = name
        self.wvl_offset = wvl_offset
        self.inty = intensity
        self.gradient = gradient

    property fit_parameters:
        def __get__(self):
            return [self.inty, self.gradient]

    cdef update_parameter_values(self, list parameters):
        cdef FreeParamType param
        if self.inty.free:
            param = self.inty
            param.cset_normalised_value(parameters.pop(0))
        if self.gradient.free:
            param = self.gradient
            param.cset_normalised_value(parameters.pop(0))

    cdef double[:] evaluate(self, double[:] wavelengths, double[:] output):

        cdef int i, n = wavelengths.shape[0]
        cdef double inty, gradient, wvl

        inty = self.inty.get_value()
        gradient = self.gradient.get_value()

        for i in range(n):
            output[i] = inty + gradient * (wavelengths[i] - self.wvl_offset)
        return output


cdef class GaussianLine(SpectralSource):

    def __init__(self, str name, intensity, temperature, wavelength, double weight,
                 bint intensity_from_peak=False, double inst_sigma=0.0):

        # Special case handling for case of data_peak
        if intensity_from_peak and intensity.free:
            temp = temperature.value
            wvl = wavelength.value
            inty = intensity.value
            sigma = sqrt(temp * ELEMENTARY_CHARGE / (weight * AMU)) * wvl / SPEED_OF_LIGHT
            sigma = sqrt(sigma*sigma + inst_sigma*inst_sigma)
            intensity.value = inty * (sigma * sqrt(2 * PI))

        self.name = name
        self.inty = intensity
        self.temp = temperature
        self.wvl = wavelength
        self.weight = weight
        self.inst_sigma = inst_sigma

    property fit_parameters:
        def __get__(self):
            return [self.inty, self.temp, self.wvl]

    cdef update_parameter_values(self, list parameters):
        cdef FreeParamType param
        if self.inty.free:
            param = self.inty
            param.cset_normalised_value(parameters.pop(0))
        if self.temp.free:
            param = self.temp
            param.cset_normalised_value(parameters.pop(0))
        if self.wvl.free:
            param = self.wvl
            param.cset_normalised_value(parameters.pop(0))

    cdef double[:] evaluate(self, double[:] wavelengths, double[:] output):

        cdef int i, n = wavelengths.shape[0], n_start, n_end
        cdef double temperature, centre_wvl, weight = self.weight
        cdef double intensity, sigma, i0, exponent, temp, inst_sigma, wvl_i, wvl_lower, wvl_higher

        intensity = self.inty.get_value()
        temperature = self.temp.get_value()
        centre_wvl = self.wvl.get_value()
        inst_sigma = self.inst_sigma

        # convert temperature to line width (sigma)
        sigma = sqrt(temperature * ELEMENTARY_CHARGE / (weight * AMU)) * centre_wvl / SPEED_OF_LIGHT
        sigma = sqrt(sigma*sigma + inst_sigma*inst_sigma)

        # Calculate 4 sigma wavelength bounds
        wvl_lower = centre_wvl - (sigma * 4)
        wvl_upper = centre_wvl + (sigma * 4)

        i0 = intensity/(sigma * sqrt(2 * PI))
        temp = 2*sigma*sigma
        # gaussian line
        for i in range(n):
            wvl_i = wavelengths[i]
            # only evaluate function if inside 4 sigma bounds
            if wvl_lower < wvl_i < wvl_upper:
                output[i] = i0 * exp(-(wvl_i - centre_wvl)**2 / temp)
            else:
                output[i] = 0.0
        return output


cdef class DopplerShiftedLine(SpectralSource):

    def __init__(self, str name, intensity, temperature, double natural_wavelength, velocity,
                 double cos_angle, double weight, bint intensity_from_peak=False, double inst_sigma=0.0):

        # Special case handling for case of data_peak
        if intensity_from_peak:
            inty = intensity.value
            temp = temperature.value
            vel = velocity.value
            wvl = natural_wavelength * (1 + vel * cos_angle / SPEED_OF_LIGHT)
            sigma = sqrt(temp * ELEMENTARY_CHARGE / (weight * AMU)) * wvl / SPEED_OF_LIGHT
            sigma = sqrt(sigma*sigma + inst_sigma*inst_sigma)
            intensity.value = inty * (sigma * sqrt(2 * PI))

        self.name = name
        self.natural_wavelength = natural_wavelength
        self.cos_angle = cos_angle
        self.weight = weight
        self.inst_sigma = inst_sigma

        # Parameters
        self.inty = intensity
        self.temp = temperature
        self.vel = velocity

    property fit_parameters:
        def __get__(self):
            return [self.inty, self.temp, self.vel]

    cdef update_parameter_values(self, list parameters):
        cdef FreeParamType param
        if self.inty.free:
            param = self.inty
            param.cset_normalised_value(parameters.pop(0))
        if self.temp.free:
            param = self.temp
            param.cset_normalised_value(parameters.pop(0))
        if self.vel.free:
            param = self.vel
            param.cset_normalised_value(parameters.pop(0))

    cdef double[:] evaluate(self, double[:] wavelengths, double[:] output):

        cdef int i, n = wavelengths.shape[0]
        cdef double temperature, centre_wvl, velocity, intensity, sigma, i0, exponent, temp, inst_sigma

        intensity = self.inty.get_value()
        temperature = self.temp.get_value()
        velocity = self.vel.get_value()
        inst_sigma = self.inst_sigma

        centre_wvl = self.natural_wavelength * (1 + velocity * self.cos_angle / SPEED_OF_LIGHT)

        # convert temperature to line width (sigma)
        sigma = sqrt(temperature * ELEMENTARY_CHARGE / (self.weight * AMU)) * centre_wvl / SPEED_OF_LIGHT
        sigma = sqrt(sigma*sigma + inst_sigma*inst_sigma)

        i0 = intensity/(sigma * sqrt(2 * PI))
        temp = 2*sigma*sigma
        # gaussian line
        for i in range(n):
            output[i] = i0 * exp(-(wavelengths[i] - centre_wvl)**2 / temp)
        return output


# TODO - This function may not be supported anymore due to asymetry in convolution.
cdef class SkewGaussianLine(SpectralSource):

    def __init__(self, str name, intensity, temperature, wavelength, alpha,
                 float weight, bint intensity_from_peak=False):

        # Special case handling for case of data_peak
        if intensity_from_peak:
            temp = temperature.value
            wvl = wavelength.value
            inty = intensity.value
            sigma = sqrt(temp * ELEMENTARY_CHARGE / (weight * AMU)) * wvl / SPEED_OF_LIGHT
            intensity.value = inty * (sigma * sqrt(2 * PI))

        self.name = name
        self.weight = weight

        # parameters
        self.inty = intensity
        self.temp = temperature
        self.wvl = wavelength
        self.alpha = alpha

    property fit_parameters:
        def __get__(self):
            return [self.inty, self.temp, self.wvl, self.alpha]

    cdef update_parameter_values(self, list parameters):
        cdef FreeParamType param
        if self.inty.free:
            param = self.inty
            param.cset_normalised_value(parameters.pop(0))
        if self.temp.free:
            param = self.temp
            param.cset_normalised_value(parameters.pop(0))
        if self.wvl.free:
            param = self.wvl
            param.cset_normalised_value(parameters.pop(0))
        if self.alpha.free:
            param = self.alpha
            param.cset_normalised_value(parameters.pop(0))

    cdef double[:] evaluate(self, double[:] wavelengths, double[:] output):

        cdef int i, n = wavelengths.shape[0]
        cdef double temperature, centre_wvl, intensity, sigma, alpha, i0, temp, temp2, weight

        weight = self.weight
        intensity = self.inty.get_value()
        temperature = self.temp.get_value()
        centre_wvl = self.wvl.get_value()
        alpha = self.alpha.get_value()

        # convert temperature to line width (sigma)
        sigma = sqrt(temperature * ELEMENTARY_CHARGE / (weight * AMU)) * centre_wvl / SPEED_OF_LIGHT
        i0 = intensity/(sigma * sqrt(2 * PI))
        temp = 2*sigma*sigma
        temp2 = -alpha / (sqrt(2) * sigma)

        # skew gaussian line
        for i in range(n):
            output[i] = i0 * exp(-(wavelengths[i] - centre_wvl)**2 / temp) * erfc(temp2 * (wavelengths[i] - centre_wvl))
        return output


cdef class GaussianFunction(SpectralSource):

    def __init__(self, str name, intensity, sigma, centre_point, bint intensity_from_peak=False):
        """
        Fits a Gaussian function, similar to Gaussian line but more general functional form.

        :param name:
        :param intensity:
        :param sigma:
        :param centre_point:
        :param intensity_from_peak:
        :return:
        """

        # Special case handling for case of data_peak
        if intensity_from_peak:
            sig = sigma.value
            cpt = centre_point.value
            inty = intensity.value
            intensity.value = inty * (sig * sqrt(2 * PI))

        self.name = name
        self.inty = intensity
        self.sigma = sigma
        self.centre_point = centre_point

    property fit_parameters:
        def __get__(self):
            return [self.inty, self.sigma, self.centre_point]

    cdef update_parameter_values(self, list parameters):
        cdef FreeParamType param
        if self.inty.free:
            param = self.inty
            param.cset_normalised_value(parameters.pop(0))
        if self.sigma.free:
            param = self.sigma
            param.cset_normalised_value(parameters.pop(0))
        if self.centre_point.free:
            param = self.centre_point
            param.cset_normalised_value(parameters.pop(0))

    cdef double[:] evaluate(self, double[:] wavelengths, double[:] output):

        cdef int i, n = wavelengths.shape[0]
        cdef double sigma, cpt, inty, i0, exponent, temp

        inty = self.inty.get_value()
        sigma = self.sigma.get_value()
        cpt = self.centre_point.get_value()

        i0 = inty/(sigma * sqrt(2 * PI))
        temp = 2*sigma*sigma
        # gaussian line
        for i in range(n):
            output[i] = i0 * exp(-(wavelengths[i] - cpt)**2 / temp)
        return output

