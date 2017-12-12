
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

import time
import ppf
import numpy as np
from iminuit import Minuit
from iminuit.frontends import ConsoleFrontend
import matplotlib.pyplot as plt

from cherab.fitting import InvalidFit


class NoFitParameter(Exception):
    pass


def minuit_fitter(multi_spectra_model, print_level=0):

    forced_parameters, keyword_args = multi_spectra_model.get_minuit_args()

    minuit = Minuit(multi_spectra_model, forced_parameters=forced_parameters, frontend=ConsoleFrontend(),
                    errordef=1, print_level=print_level, **keyword_args)
    minuit.set_strategy(2)
    minuit.tol = 500
    fit_results, fit_params = minuit.migrad()

    if not fit_results['is_valid']:
        multi_spectra_model.random_nudge_params()
        fit_results, fit_params = minuit.migrad()
        if not fit_results['is_valid']:
            raise InvalidFit

    for free_param in multi_spectra_model.free_parameters:
        name = free_param.name
        for param_results in fit_params:
            if param_results['name'] == name:
                free_param.set_normalised(param_results['value'])
                free_param.error = free_param.denormalise_error(param_results['error'])
                break
        else:
            raise NoFitParameter("Fit parameter '{}' could not be found in the fit results.".format(name))


class FitStrategy:
    """
    Provides the fit strategy for a 'single' fit. i.e. pairs of LOS or single LOS
    """

    def __init__(self, name, spectral_models, fitter):

        self.name = name
        self.spectral_models = spectral_models
        self.multi_spectra_model = None

        self._overall_fit_results = None  # Raw dict of overall fit results from Minuit.
        self._fit_parameter_results = None  # Raw dict of individual fit parameter results from Minuit.

        self.fitted_signals = []
        self.radii_count = 0

        self.fitter = fitter

    def fit_model(self, print_level=0, plot_level=0):

        if print_level > 0:
            print("Fitting process: {}".format(self.name))
        self.fitter(self.multi_spectra_model, print_level=print_level, plot_level=plot_level)

    def fit_results(self):
        raise NotImplementedError("This method must be implemented in the derived class.")

    def plot(self):
        raise NotImplementedError("This method must be implemented in the derived class.")


# Manages the fitting of a whole profile through smaller strategic steps, individual FitStrategy classes
class FitManager:

    def __init__(self, equilibrium, profiles):
        hrts_te, ks5_CVI_ti, ks5_CVI_inty, ks5_CVI_wvl = profiles
        self.hrts_profile = hrts_te
        self.ppf_profiles = (ks5_CVI_ti, ks5_CVI_inty, ks5_CVI_wvl)
        self._equilibrium = equilibrium
        self._process_queue = []
        self._spectral_models = []
        self.time_manager = None
        self.fitted_parameters = {}

    def add_fit_process(self, process):
        self._process_queue.append(process)
        self._spectral_models += process.spectral_models

    def total_fitted_radii(self):
        i = 0
        for process in self._process_queue:
            i += process.radii_count
        return i

    def fit(self, print_level=0, plot_level=0):

        self._initialise_results_structures()

        for i, t in enumerate(self.time_manager):

            # TODO - this sort of call needs to move to notifier/messaging design pattern
            if type(self.time_manager) is TimeManager:
                self._psin = self._equilibrium.time(t).psi_normalised
                for spectral_model in self._spectral_models:
                    spectral_model.spectra.move_time_curser_to(t)
                self.hrts_profile.move_time_curser_to(t)

            for j, process in enumerate(self._process_queue):

                if i == 0 or not (i % 7):  # force re-initialisation every 7 iterations
                    process.initialise_fit_params(i, j)

                if plot_level >= 3:
                    process.plot()
                    input("initial conditions plot...")
                    plt.close('all')
                if print_level > 0:
                    start = time.time()

                # Allow
                try:
                    process.fit_model(print_level=print_level, plot_level=plot_level)
                except InvalidFit:
                    process.initialise_fit_params(i, j)
                    process.multi_spectra_model.random_nudge_params()
                    try:
                        process.fit_model(print_level=print_level, plot_level=plot_level)
                    except InvalidFit:
                        process.initialise_fit_params(i, j)
                        process.multi_spectra_model.random_nudge_params()
                        try:
                            process.fit_model(print_level=print_level, plot_level=plot_level)
                        except InvalidFit:

                            if plot_level >= 1:
                                process.plot()
                                input("Final conditions at failure...")

                            self.fitted_parameters['FAIL'][i, j] = 1.0
                            print('Warning: fit {} failed.'.format(process.name))
                            continue

                if print_level > 0:
                    end = time.time()
                    print("Elapsed time = {:.2f} mins".format((end - start)/60))
                    chi2 = process.multi_spectra_model.chi2()
                    print("Chi^2 minimum => {:.4G}".format(chi2))

                # save fit results
                for key, value in process.fit_results():
                    self.fitted_parameters[key][i, j] = value

                if plot_level >= 2:
                    for key, value in process.fit_results():
                        print('key {}, value {:.4G}'.format(key, value))
                    plt.figure()
                    process.plot()
                    self.plot_ti_fit(i)
                    input("Final fitted conditions plot...")
                    plt.close('all')

            if plot_level >= 1:
                self.plot_current_fit(i)

            print("completed fit time {:.4G}s".format(t))

        ni, nj = self.fitted_parameters['FAIL'].shape
        print("Fitting complete!")
        print("Total fits attempted = {}".format(ni * nj))
        print("Total failed fits = {}".format(self.fitted_parameters['FAIL'].sum()))

    def _initialise_results_structures(self):
        ntimes = self.time_manager.total_times
        nradii = self.total_fitted_radii()

        for process in self._process_queue:
            for fitted_param in process.fitted_signals:
                try:
                    _ = self.fitted_parameters[fitted_param]
                except KeyError:
                    self.fitted_parameters[fitted_param] = np.zeros((ntimes, nradii))

        self.fitted_parameters['FAIL'] = np.zeros((ntimes, nradii))

    def write_ppf(self, pulse, uid, dda):

        # set PPF author
        ppf.ppfuid(uid, "w")
        ppf.ppfuid(uid, "r")

        # open new ppf
        time, date, ier = ppf.pdstd(pulse)
        ier = ppf.ppfopn(pulse, date, time, "CHERAB redblue CX")
        xaxis = self.fitted_parameters['rcor'][0, :]
        taxis = [t for t in self.time_manager]

        for dtype, value_array in self.fitted_parameters.items():

            nt, nx = value_array.shape

            irdat = ppf.ppfwri_irdat(nx, nt)
            ihdat = ppf.ppfwri_ihdat("none", "m", "s", "f", "f", "f", "Description (floats)")
            iwdat, ier = ppf.ppfwri(pulse, dda, dtype, irdat, ihdat, value_array, xaxis, taxis)

        seq, ier = ppf.ppfclo(pulse, "CHERAB", 1)
        print("Finished writing PPF for pulse={} seq={}".format(pulse, seq))

    def plot_current_fit(self, i):

        ks5_CVI_ti, ks5_CVI_inty, ks5_CVI_wvl = self.ppf_profiles

        # Plot results
        plt.figure()
        if ks5_CVI_ti is not None:
            plt.errorbar(ks5_CVI_ti.radii, ks5_CVI_ti.data, yerr=ks5_CVI_ti.edata, label=ks5_CVI_ti.name)
        plt.errorbar(self.fitted_parameters['rcor'][i, :], self.fitted_parameters['ti'][i, :],
                     yerr=[self.fitted_parameters['ti'][i, :]-self.fitted_parameters['tilo'][i, :],
                           self.fitted_parameters['tihi'][i, :]-self.fitted_parameters['ti'][i, :]], label='Cherab Ti')
        plt.errorbar(self.hrts_profile.radii, self.hrts_profile.data, yerr=self.hrts_profile.edata, label='HRTS')
        plt.xlim(2.8, 4.0)
        plt.ylim(0, 6500)
        plt.legend()

        plt.figure()
        if ks5_CVI_inty is not None:
            plt.errorbar(ks5_CVI_inty.radii, ks5_CVI_inty.data, yerr=ks5_CVI_inty.edata, label=ks5_CVI_inty.name)
        plt.errorbar(self.fitted_parameters['rcor'][i, :], self.fitted_parameters['i01'][i, :],
                     yerr=[self.fitted_parameters['i01'][i, :]-self.fitted_parameters['i01l'][i, :],
                           self.fitted_parameters['i01h'][i, :]-self.fitted_parameters['i01'][i, :]], label='Cherab Inty')
        plt.legend(loc=2)

        plt.figure()
        if ks5_CVI_wvl is not None:
            plt.errorbar(ks5_CVI_wvl.radii, ks5_CVI_wvl.data, yerr=ks5_CVI_wvl.edata, label=ks5_CVI_wvl.name)
        plt.errorbar(self.fitted_parameters['rcor'][i, :], self.fitted_parameters['w01'][i, :],
                     yerr=self.fitted_parameters['dw01'][i, :], label='Cherab wvl')
        plt.legend()
        input("waiting...")

    def plot_ti_fit(self, i):

        ks5_CVI_ti, ks5_CVI_inty, ks5_CVI_wvl = self.ppf_profiles

        plt.figure()
        if ks5_CVI_ti is not None:
            plt.errorbar(ks5_CVI_ti.radii, ks5_CVI_ti.data, yerr=ks5_CVI_ti.edata, label=ks5_CVI_ti.name)
        plt.errorbar(self.fitted_parameters['rcor'][i, :], self.fitted_parameters['ti'][i, :],
                     yerr=[self.fitted_parameters['ti'][i, :]-self.fitted_parameters['tilo'][i, :],
                           self.fitted_parameters['tihi'][i, :]-self.fitted_parameters['ti'][i, :]], label='Cherab Ti')
        plt.errorbar(self.hrts_profile.radii, self.hrts_profile.data, yerr=self.hrts_profile.edata, label='HRTS')
        plt.xlim(2.8, 4.0)
        plt.ylim(0, 6500)
        plt.legend()


# TODO - this needs to be a notifier class using the OOP messaging design pattern.
class TimeManager:

    def __init__(self, times, pini86, pini87, start, end):
        self.times = times
        self.pini86 = pini86
        self.pini87 = pini87
        self.start = start
        self.end = end

    def __iter__(self):
        for t in self.times:
            # If pini 8.6 or 8.7 is active at current time
            if (self.pini86(t) or self.pini87(t)) and self.start <= t <= self.end:
                yield t

    @property
    def total_times(self):
        i = 0
        for t in self:
            i += 1
        return i


class SingleTimeManager:

    def __init__(self, time_to_return):
        self.time = time_to_return

    def __iter__(self):
        yield self.time

    @property
    def total_times(self):
        return 1

