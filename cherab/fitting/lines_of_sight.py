
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
from collections import namedtuple
from math import cos, radians

import numpy as np
import matplotlib.pyplot as plt
from raysect.core.math import AffineMatrix3D
from raysect.core.scenegraph.observer import Observer
from raysect.core.scenegraph.world import World
from raysect.optical import Ray, Spectrum
from iminuit import Minuit
from iminuit.frontends import ConsoleFrontend


# Internal imports
from cherab.core.math.interpolators.interpolators1d import Interpolate1DCubic
from cherab.fitting import InvalidFit
from cherab.fitting import Baseline, GaussianLine, SingleSpectraModel, FreeParameter
from cherab.fitting.fit_strategy import find_wvl_estimate

doppler_shift = namedtuple("shift", "type sign")
RED_SHIFT = doppler_shift("red", 1)
BLUE_SHIFT = doppler_shift("blue", -1)
NO_SHIFT = doppler_shift("none", 1)


class LineOfSight(Observer):
    """
    A line of sight into the plasma (e.g. a single optical fibre).

    :param str name: Name/code for this sight line (e.g. fibre code D12).
    :param Spectra spectra: Time series spectra for this sight line.
    :param Point origin: The origin of the line of sight.
    :param Vector direction: The direction of this line of sight.
    :param float radius: Optional radius for this fibre. May not make sense to define a radius for this fibre.
    :param Point point_of_interest: Optional point of interest for this sight line. Used by users code. For example,
    might be pini intersection point in case of JET KS5 diagnostics.
    """

    # TODO - can psi_func be a default parameter? Maybe replaced with an equilibrium object?
    def __init__(self, name, spectra, origin, direction, psi_func, radius=None, point_of_interest=None,
                 active_volume=1.0, theta_los=None, spectral_samples=100, rays=1, pcx_theta=None):

        Observer.__init__(self, parent=None, transform=AffineMatrix3D(), name=name)

        self.name = name
        self.spectra = spectra
        self.origin = origin
        self.direction = direction
        self.radius = radius
        self.point_of_interest = point_of_interest  # TODO - Currently this is the pini 8.6 intersection, needs rethinking.

        # Calibration factor, based on line integration of line of sight.
        self.active_volume = active_volume

        self.psi_func = psi_func

        if theta_los:
            self.shift = RED_SHIFT if -90 <= theta_los <= 90 else BLUE_SHIFT
            self.theta_los = theta_los
            self._cos_theta_los = cos(radians(theta_los))
            if name == 'ks5c_D6' or name == 'ks5d_D6':
                print('shift => {}'.format(self.shift))
                print('theta los => {}'.format(self.theta_los))
                print('cos theta => {}'.format(self._cos_theta_los))
        else:
            self.shift = NO_SHIFT
            self.theta_los = None
            self._cos_theta_los = 1

        if pcx_theta:
            self.pcx_theta = pcx_theta
            self._cos_pcx_theta = cos(radians(pcx_theta))

        self._rays = rays
        self._spectral_samples = spectral_samples

        self._min_wavelength = 375.0  # nm
        self._max_wavelength = 740.0  # nm

        self._ray_max_depth = 15

        self.spectrum = None
        self.max_radiance = 0.
        self.auto_display = False

    @property
    def psi(self):
        return self.psi_func(*self.point_of_interest)

    @property
    def parent_losgroup(self):

        for group in LOSGroup._FibreGroups.values():
            for los in group:
                if los is self:
                    return group

        raise LOSGroupNotFound()

    def observe(self):
        """ Fire a single ray and fill the 'spectrum' attribute with the
        observed spectrum. If 'display' attribute is True, the spectrum is
        shown at the end.
        """

        if not isinstance(self.root, World):
            raise TypeError("Observer is not connected to a scene graph containing a World object.")

        if self._min_wavelength >= self._max_wavelength:
            raise RuntimeError("Min wavelength is superior to max wavelength!")

        world = self.root

        total_samples = self._rays * self._spectral_samples

        # generate rays
        rays = list()
        delta_wavelength = (self._max_wavelength - self._min_wavelength) / self._rays
        lower_wavelength = self._min_wavelength
        for index in range(self._rays):

            upper_wavelength = self._min_wavelength + delta_wavelength * (index + 1)

            rays.append(Ray(min_wavelength=lower_wavelength, max_wavelength=upper_wavelength,
                            num_samples=self._spectral_samples, max_depth=self._ray_max_depth))

            lower_wavelength = upper_wavelength

        self.spectrum = Spectrum(self._min_wavelength, self._max_wavelength, total_samples)

        lower_index = 0
        for index, ray in enumerate(rays):

            upper_index = self._spectral_samples * (index + 1)

            # convert ray parameters to world space
            ray.origin = self.origin
            ray.direction = self.direction

            # sample world
            sample = ray.trace(world)
            self.spectrum.samples[lower_index:upper_index] = sample.samples

            lower_index = upper_index

        # calculate max radiance for pretty displays:
        for radiance in self.spectrum.samples:
            self.max_radiance = max(self.max_radiance, radiance)


class LOSGroup:
    """
    A set of co-mounted optical fibres.

    This is like an optical head at JET, where a group of optical fibres are mounted together with a common set of
    optics, or "view" into the plasma.
    """

    _FibreGroups = {}

    def __init__(self, name):

        self.name = name
        self.los_dict = {}
        LOSGroup._FibreGroups[name] = self

    def __iter__(self):
        for losname, los in self.los_dict.items():
            yield(los)

    def __getitem__(self, item):
        return self.los_dict[item]

    def add_los(self, los):
        """
        Add a new Line of sight object to this FibreGroup.
        """

        if not isinstance(los, LineOfSight):
            raise ValueError("Argument los must be a LineOfSight instance, got {}".format(los))

        self.los_dict[los.name] = los

    def move_time_curser_to(self, time):

        for los in self.los_dict.values():
            los.spectra.move_time_curser_to(time)

    def check_errors(self, wvl_range=None):

        for los in self.los_dict.values():
            los.spectra.check_errors(wvl_range=wvl_range)

    # TODO - this code does not belong here

    def perform_wavelength_correction(self, natural_wvl, atomic_weight, wvl_range=1.0, num_fits=1, plot=False,
                                      print_fit=False, accept_invalid_fit=False):

        for los in self.los_dict.values():

            print("Fitting wvl references for los {}.".format(los.name))
            current_time = los.spectra.time
            spectra = los.spectra
            spectra.set_active_window((natural_wvl - wvl_range, natural_wvl + wvl_range))

            fit_found = False
            fit_count = 0
            offsets = np.zeros(num_fits)

            while spectra.time_index > 0 and fit_count < num_fits:

                max_samples = np.max(spectra.samples)
                min_samples = np.min(spectra.samples)
                max_intensity = 20 * max_samples
                min_intensity = 0

                # baseline
                baseline_inty = FreeParameter("Baseline_intensity", min_samples, valid_range=(0, 1e17))
                baseline = Baseline("Baseline", baseline_inty)

                # Calibration line
                # wvl_guess = spectra.wavelengths[spectra.samples.argmax()]
                wvl_guess = find_wvl_estimate(spectra)
                inty = FreeParameter("inty", max_samples, valid_range=(min_intensity, max_intensity))
                temp = FreeParameter("temp", 70, valid_range=(0, 500))
                wvl = FreeParameter("wvl", wvl_guess, valid_range=(natural_wvl - wvl_range, natural_wvl + wvl_range))
                line = GaussianLine("calibration_line", inty, temp, wvl, atomic_weight, intensity_from_peak=True)

                model = SingleSpectraModel([baseline, line], spectra)

                # Construct keyword arguments for minuit fitter
                forced_parameters, keyword_args = model.get_minuit_args()

                print_level = 1 if print_fit else 0

                # Start Minuit
                minuit = Minuit(model, forced_parameters=forced_parameters, frontend=ConsoleFrontend(), errordef=1,
                                print_level=print_level, **keyword_args)
                minuit.set_strategy(2)
                minuit.tol = 500
                fit_results, fit_params = minuit.migrad()

                if not fit_results['is_valid'] or fit_results['fval'] > 15:
                    fit_results, fit_params = minuit.migrad()
                    if not fit_results['is_valid']:
                        spectra.move_to_previous_time()
                        continue
                    else:
                        fit_found = True
                        offsets[fit_count] = natural_wvl - wvl.value
                        fit_count += 1
                        spectra.move_to_previous_time()
                        continue

                fit_found = True
                offsets[fit_count] = natural_wvl - wvl.value
                fit_count += 1
                spectra.move_to_previous_time()

            if not fit_found:
                if accept_invalid_fit:
                    print("Warning - invalid fit for los {}. Continuing without offset correction".format(los.name))
                else:
                    print(fit_results)
                    model.plot()
                    raise InvalidFit()

            if plot:
                plt.clf()
                model.plot()
                plt.show()
                input('waiting...')

            # Perform offset correction
            offset = offsets.sum()/fit_count
            los.spectra.wvl_offset = offset

            # Reset active window
            spectra.set_active_window(None)
            spectra.move_time_curser_to(current_time)

    def plot_wvl_references(self):

        plt.figure()
        for los in self:
            spectra = los.spectra
            plt.plot(spectra.wavelengths, spectra.samples, label=los.name)
        plt.legend()
        input("waiting...")
        plt.close('all')

    # temporary utility function
    def construct_psi_mappers(self, psi_func, r_axis=3.0, r_edge=3.82):
        """

        :param psi_func: A psi grid 3D function. Takes a set of 3D cartesian coordinates and returns the Psi value at
        that location.
        :param r_axis: Major radius R at location of the magnetic axis, Psi = 0.
        :param r_edge: Major radius R at location of the last closed flux surface on the outboard side, Psi = 1.
        :return: Two 1D interpolators that map between Psi <=> Major Radius.
        """
        psi = [0.0, 1.0]
        r = [r_axis, r_edge]
        for los in self.los_dict.values():
            # Get the Psi value at each los intersection
            psi.append(psi_func(los.radius, 0, 0))
            # Register the Major radius for each los.
            r.append(los.radius)

        psi_to_r = Interpolate1DCubic(psi, r)
        r_to_psi = Interpolate1DCubic(r, psi)
        return psi_to_r, r_to_psi


class LOSGroupNotFound(Exception):
    pass
