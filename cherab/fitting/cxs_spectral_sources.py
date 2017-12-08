
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

# External imports
from scipy import array, sqrt, exp, float64
from scipy.integrate import trapz
from scipy.constants import atomic_mass, electron_mass, elementary_charge, speed_of_light

# Internal Imports
from cherab import Species
from cherab.distribution import Maxwellian
from cherab.math import ToroidalVectorFunction3D
from cherab_extra.fitting import SpectralSource
from cherab_extra.fitting import Baseline, GaussianLine, InstrumentalFunction

amu = 1.66053892e-27


class CXSModel(SpectralSource):

    def __init__(self, plasma, profile_function):

        super().__init__()

        self.coordinates_profile = array([0., 0.3, 0.7, 1.0], dtype=float64)  # psi
        self._temperature_profile = array([6000, 4000, 2000, 400], dtype=float64)
        self._velocity_profile = array([160000, 60000, 60000, 20000], dtype=float64)
        self._electron_density = plasma.electron_distribution.density
        self._species_density = []
        for species in plasma.composition:
            self._species_density.append(species.distribution.density)

        # are parameters fixed during fitting?
        self.fixed_temperature = False
        self.fixed_velocity = False

        # range
        self.range_temperature = (0, 8000)
        self.range_velocity = (0, 200000)

        self.plasma = plasma
        self.profile_function = profile_function

        self._update_plasma()
        self._changed = True

    @property
    def temperature_profile(self):
        return self._temperature_profile

    @temperature_profile.setter
    def temperature_profile(self, value):
        if self.coordinates_profile is None:
            raise RuntimeError('Coordinates profile must be provided before!')
        if len(value) == len(self.coordinates_profile):
            self._temperature_profile = array(value, dtype=float64)
            self._changed = True
            self._update_plasma()

    @property
    def velocity_profile(self):
        return self._velocity_profile

    @velocity_profile.setter
    def velocity_profile(self, value):
        if self.coordinates_profile is None:
            raise RuntimeError('Coordinates profile must be provided before!')
        if len(value) == len(self.coordinates_profile):
            self._velocity_profile = array(value, dtype=float64)
            self._changed = True
            self._update_plasma()

    def __repr__(self):
        return "CHERAB Model"

    def _update_plasma(self):

        # set temperature and velocity profiles
        temperature_func = self.profile_function(self.coordinates_profile, self._temperature_profile)
        velocity_func = self.profile_function(self.coordinates_profile, self._velocity_profile)
        velocity_vect_func = ToroidalVectorFunction3D(velocity_func)

        new_electron_distrib = Maxwellian(self._electron_density,
                                          temperature_func,
                                          velocity_vect_func,
                                          electron_mass)
        self.plasma.electron_distribution = new_electron_distrib

        new_species = []
        composition = self.plasma.composition
        for i in range(len(composition)):
            species = composition[i]
            new_distrib = Maxwellian(self._species_density[i],
                                     temperature_func,
                                     velocity_vect_func,
                                     species.element.atomic_weight * atomic_mass)
            new_species.append(Species(species.element, species.ionisation, new_distrib))

        self.plasma.set_species(new_species)

    def serialise(self, parameters):
        """
        param parameters: The list object in which to serialise the data into.
        """

        if not self.fixed_temperature:
            for i in range(len(self.coordinates_profile)):
                parameters.append(self._normalise(self._temperature_profile[i], self.range_temperature))

        if not self.fixed_velocity:
            for i in range(len(self.coordinates_profile)):
                parameters.append(self._normalise(self._velocity_profile[i], self.range_velocity))

    def deserialise(self, parameters):
        """
        param parameters: The list object from which to de-serialise the data from.
        """
        self._changed = False

        if not self.fixed_temperature:
            old_temperature_profile = self._temperature_profile.copy()

            for i in range(len(self.coordinates_profile)):
                self._temperature_profile[i] = self._denormalise(parameters.pop(0), self.range_temperature)

            if (self._temperature_profile != old_temperature_profile).any():
                self._changed = True

        if not self.fixed_velocity:
            old_velocity_profile = self._velocity_profile.copy()

            for i in range(len(self.coordinates_profile)):
                self._velocity_profile[i] = self._denormalise(parameters.pop(0), self.range_velocity)

            if (self._velocity_profile != old_velocity_profile).any():
                self._changed = True

        if self._changed:
            self._update_plasma()


class CXSSpectrum(SpectralSource):

    def __init__(self, intensity, cherab_model, los, point):

        super().__init__()

        self._intensity = intensity

        # are parameters fixed during fitting?
        self.fixed_intensity = False

        # range
        self.range_intensity = (0, 1e18)

        self.cherab_model = cherab_model
        if los.root is cherab_model.plasma.root:
            self.los = los
            self.intersection_point = point
        else:
            raise ValueError('Plasma and LOS are not in the same scenegraph!')

        self._cache = {}
        self.cache_use = 0

    @property
    def intensity(self):
        return self._intensity

    @intensity.setter
    def intensity(self, value):
        self._intensity = value

    @property
    def temperature(self):
        return self.cherab_model.plasma.composition[0].distribution.effective_temperature(*self.intersection_point)

    @property
    def velocity(self):
        return self.cherab_model.plasma.composition[0].distribution.bulk_velocity(*self.intersection_point).length

    @property
    def wavelength(self):
        return self.los.spectrum.wavelengths[self.los.spectrum.samples.argmax()]

    @property
    def total_intensity(self):
        samples = self.los.spectrum.to_photons()
        return trapz(samples, self.los.spectrum.wavelengths)

    def __repr__(self):
        return "Spectral source using a CHERAB Model"

    def render(self, wavelengths):

        plasma_parameters = (tuple(self.cherab_model.temperature_profile), tuple(self.cherab_model.velocity_profile))

        if plasma_parameters in self._cache:
            self.cache_use += 1
            samples = self._cache[plasma_parameters]
        else:
            self.los.spectral_samples = len(wavelengths)
            half_delta_wvl = (wavelengths[1] - wavelengths[0]) / 2.
            self.los.min_wavelength = min(wavelengths) - half_delta_wvl
            self.los.max_wavelength = max(wavelengths) + half_delta_wvl
            self.los.observe()
            samples = self.los.spectrum.to_photons()
            self._cache[plasma_parameters] = samples

        if samples.max() != 0.:
            samples *= self._intensity / samples.max()

        return samples

    def serialise(self, parameters):
        """
        param parameters: The list object in which to serialise the data into.
        """

        if not self.fixed_intensity:
            parameters.append(self._normalise(self._intensity, self.range_intensity))

    def deserialise(self, parameters):
        """
        param parameters: The list object from which to de-serialise the data from.
        """

        if not self.fixed_intensity:
            self._intensity = self._denormalise(parameters.pop(0), self.range_intensity)


class DopplerShiftedGaussianLine(SpectralSource):

    def __init__(self, intensity, temperature, natural_wavelength, velocity, cos_angle, weight):

        super().__init__()

        self.intensity = intensity
        self.temperature = temperature
        self.natural_wavelength = natural_wavelength  # fixed
        self.velocity = velocity
        self.cos_angle = cos_angle  # fixed
        self.weight = weight

        # are parameters fixed during fitting?
        self.fixed_intensity = False
        self.fixed_temperature = False
        self.fixed_velocity = False

        # are we borrowing data from another GaussianLine object?
        self.intensity_source = None
        self.temperature_source = None
        self.velocity_source = None

        # range
        self.range_intensity = (0, 1e16)
        self.range_temperature = (0, 10000)
        self.range_velocity = (-500000, 500000)

    def __repr__(self):

        if self.intensity_source:
            intensity = self.intensity_source.intensity
        else:
            intensity = self.intensity

        if self.temperature_source:
            temperature = self.temperature_source.temperature
        else:
            temperature = self.temperature

        if self.velocity_source:
            wavelength = self.natural_wavelength * (1 + self.velocity_source.velocity * self.cos_angle / speed_of_light)
        else:
            wavelength = self.natural_wavelength * (1 + self.velocity * self.cos_angle / speed_of_light)

        return "I={}, T={}, w={}".format(intensity, temperature, wavelength)

    def render(self, wavelengths):

        if self.intensity_source:
            intensity = self.intensity_source.intensity
        else:
            intensity = self.intensity

        if self.temperature_source:
            temperature = self.temperature_source.temperature
        else:
            temperature = self.temperature

        if self.velocity_source:
            central_wavelength = self.natural_wavelength * (1 + self.velocity_source.velocity * self.cos_angle / speed_of_light)
        else:
            central_wavelength = self.natural_wavelength * (1 + self.velocity * self.cos_angle / speed_of_light)

        # convert temperature to line width (sigma)
        sigma = sqrt(temperature * elementary_charge / (self.weight * amu)) * central_wavelength / speed_of_light

        # gaussian line
        return intensity * exp(-(wavelengths - central_wavelength)**2 / (2*sigma**2))

    def position(self):

        if self.velocity_source:
            return self.natural_wavelength * (1 + self.velocity_source.velocity * self.cos_angle / speed_of_light)
        else:
            return self.natural_wavelength * (1 + self.velocity * self.cos_angle / speed_of_light)

    def serialise(self, parameters):
        """
        param parameters: The list object in which to serialise the data into.
        """

        if not (self.fixed_intensity or self.intensity_source):
            parameters.append(self._normalise(self.intensity, self.range_intensity))

        if not (self.fixed_temperature or self.temperature_source):
            parameters.append(self._normalise(self.temperature, self.range_temperature))

        if not (self.fixed_velocity or self.velocity_source):
            parameters.append(self._normalise(self.velocity, self.range_velocity))

    def deserialise(self, parameters):
        """
        param parameters: The list object from which to de-serialise the data from.
        """

        if not (self.fixed_intensity or self.intensity_source):
            self.intensity = self._denormalise(parameters.pop(0), self.range_intensity)

        if not (self.fixed_temperature or self.temperature_source):
            self.temperature = self._denormalise(parameters.pop(0), self.range_temperature)

        if not (self.fixed_velocity or self.velocity_source):
            self.velocity = self._denormalise(parameters.pop(0), self.range_velocity)


class LOSSpectralSource(SpectralSource):

    def __init__(self, measured_spectrum, inst_func_spectrum, cherab_model, los, point):

        self.measured_spectrum = measured_spectrum
        self.los_name = los.name

        self.inst_func = InstrumentalFunction(inst_func_spectrum, self.measured_spectrum.wavelengths)
        # self.inst_func.plot()

        # setup lines
        max_line_intensity = 2 * (self.measured_spectrum.samples.max() - self.measured_spectrum.samples.min())

        # baseline
        self.baseline = Baseline(measured_spectrum.samples[0])
        self.baseline.range = (self.measured_spectrum.samples.min(), self.measured_spectrum.samples.max())

        # active cx line using CHERAB
        self.active = CXSSpectrum(1e17, cherab_model, los, point)
        self.active.range_intensity = (0, max_line_intensity)

        # passive cx line
        self.passive = GaussianLine(2.7e15, 580, PASS_WVL, weight_c)
        self.passive.range_intensity = (0, max_line_intensity)
        self.passive.range_temperature = (50, 1500)
        # self.passive.fixed_intensity = True
        # self.passive.fixed_temperature = True
        # self.passive.fixed_wavelength = True

        # edge line (no edge line for this pulse)
        self.edge = GaussianLine(0, 60, 529.059, weight_c)
        self.edge.range_intensity = (0, 0)
        self.edge.range_temperature = (10, 150)
        self.edge.fixed_intensity = True
        self.edge.fixed_temperature = True
        self.edge.fixed_wavelength = True

        # setup nuisance lines
        self.nuisance = []

        # BeII
        line = GaussianLine(0.0, 100, 527.063, weight_be)
        line.range_intensity = (0, max_line_intensity)
        line.range_temperature = (20, 200)
        line.fixed_wavelength = True
        self.nuisance.append(line)

        # CIII
        line = GaussianLine(0.0, 100, 530.462, weight_c)
        line.range_intensity = (0, max_line_intensity)
        line.range_temperature = (40, 200)
        line.fixed_wavelength = True
        self.nuisance.append(line)

    def serialise(self, parameters):

        self.baseline.serialise(parameters)
        self.active.serialise(parameters)
        self.passive.serialise(parameters)
        self.edge.serialise(parameters)
        for line in self.nuisance:
            line.serialise(parameters)

    def deserialise(self, parameters):

        self.baseline.deserialise(parameters)
        self.active.deserialise(parameters)
        self.passive.deserialise(parameters)
        self.edge.deserialise(parameters)
        for line in self.nuisance:
            line.deserialise(parameters)

    def forward_model(self, wavelengths, components):

        samples = zeros(len(wavelengths))

        # generate spectrum
        for component in components:
            samples += component.render(wavelengths)

        # apply instrument function
        samples = self.inst_func.extrapol_convolve(samples)

        return samples

    def display(self):

        components = self.nuisance.copy()
        components.append(self.baseline)
        components.append(self.active)
        components.append(self.passive)
        components.append(self.edge)

        target = self.measured_spectrum

        plt.errorbar(target.wavelengths, target.samples, target.sigmas, color='black')
        fitted_spectrum = self.forward_model(target.wavelengths, components)
        plt.plot(target.wavelengths, fitted_spectrum, color='blue')

        colors = {self.active: 'red'}#, self.passive: 'purple', self.edge: 'orange'}
        for component in components:

            try:
                color = colors[component]
            except KeyError:
                color = 'orange'

            samples = component.render(target.wavelengths)
            samples = self.inst_func.extrapol_convolve(samples)

            plt.plot(target.wavelengths, samples, color=color)
            try:
                plt.plot([component.wavelength], [0.], '^', color=color)
            except AttributeError:
                pass

        # plt.plot([NAT_WVL, NAT_WVL], [0, max(fitted_spectrum)], '-r')

        # redraw the fit to see it well
        plt.plot(target.wavelengths, fitted_spectrum, color='blue')

        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Radiance (ph/s/m^2/str/nm)')
        # plt.legend(['Measured spectrum', 'Fitted spectrum'])

    def evaluate(self):

        # generate predictions
        components = self.nuisance.copy()
        components.append(self.baseline)
        components.append(self.active)
        components.append(self.passive)
        components.append(self.edge)
        predicted = self.forward_model(self.measured_spectrum.wavelengths, components)

        # calculate chi squared
        return sum(((predicted - self.measured_spectrum.samples) / self.measured_spectrum.sigmas)**2)


# TODO: this is should be an axisymmetricVector mapper function!
# cdef class ToroidalVectorFunction3D(VectorFunction3D):
#
#     def __init__(self, object norm_function):
#
#         super().__init__()
#         self._norm = autowrap_function3d(norm_function)
#
#     cdef Vector evaluate(self, double x, double y, double z):
#
#         cdef Vector direction = Vector(y, -x, 0.).normalise()
#
#         return self._norm(x, y, z) * direction