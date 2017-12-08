
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

# External inputs
import numpy as np
from iminuit import Minuit
from iminuit.frontends import ConsoleFrontend
import matplotlib.pyplot as plt

# Local inputs
from cherab.fitting.spectra import Spectra
from cherab.fitting import Baseline, GaussianFunction, FreeParameter, SingleSpectraModel, MultiSpectraModel


def fit_baseline(baseline, spectra, wvl_range, plot_fit=False):
    """ Fit the baseline of a measured spectra on its own.

    :param SpectralSource baseline: Baseline spectral source.
    :param spectra: Spectra to fit.
    :param wvl_range: Valid wavelength range for performing isolated baseline fit.
    :param plot_fit: Show the fit, defaults to false.
    """

    previous_wvl_window = spectra.active_window
    spectra.set_active_window(wvl_range)

    baseline.inty.value = np.mean(spectra.samples)
    baseline_model = SingleSpectraModel([baseline], spectra)
    multi_spectra_model = MultiSpectraModel([baseline_model], [])
    forced_parameters, keyword_args = multi_spectra_model.get_minuit_args()

    # Start minuit fit!!!
    minuit = Minuit(multi_spectra_model, forced_parameters=forced_parameters, frontend=ConsoleFrontend(), errordef=1,
                    print_level=0, **keyword_args)

    minuit.migrad()

    if plot_fit:
        plt.clf()
        baseline_model.plot()
        input("holding...")

    spectra.set_active_window(previous_wvl_window)


def find_wvl_estimate(spectra, wvl_range=None):
    """ Estimate the wvl location of the gaussian peak from measured spectra.

    Not as simple as just going for max value, tries to assess neighbouring samples.

    :param spectra: Measured spectra.
    :param wvl_range: Valid wavelength range in which to perform analysis. Defaults to active range.
    :return: wavelength estimate.
    """

    if wvl_range:
        previous_wvl_window = spectra.active_window
        spectra.set_active_window(wvl_range)

    peak_value, peak_wvl, peak_pixel = spectra.max_pair_in_range()
    # peak_pixel = spectra.nearest_wvl_index(peak_wvl, in_active_window=True)

    if spectra.samples[peak_pixel - 1] > spectra.samples[peak_pixel + 1]:
        if spectra.samples[peak_pixel - 2] > spectra.samples[peak_pixel - 1]:
            wvl_estimate = spectra.wavelengths[peak_pixel - 1]
        else:
            wvl_estimate = (spectra.wavelengths[peak_pixel] - spectra.wavelengths[peak_pixel-1]) / 2 + spectra.wavelengths[peak_pixel-1]
    else:
        if spectra.samples[peak_pixel + 2] > spectra.samples[peak_pixel + 1]:
            wvl_estimate = spectra.wavelengths[peak_pixel + 1]
        else:
            wvl_estimate = (spectra.wavelengths[peak_pixel+1] - spectra.wavelengths[peak_pixel]) / 2 + spectra.wavelengths[peak_pixel]

    if wvl_range:
        spectra.set_active_window(previous_wvl_window)

    return wvl_estimate


def fit_instrument_width(wavelengths, inst_func, print_fit=False, plot_fit=False):

    errors = np.ones(len(inst_func)) * 0.05 * max(inst_func)
    spec = Spectra(wavelengths, inst_func, errors, name="Instrument Spec")
    spec_min = np.min(spec.samples)
    spec_max = np.max(spec.samples)

    # baseline
    baseline_inty = FreeParameter("Baseline_intensity", spec_min, valid_range=(0, 1e17))
    baseline = Baseline("Baseline", baseline_inty)

    # Instrument line
    wvl_guess = find_wvl_estimate(spec)
    inty = FreeParameter("inty", spec_max, valid_range=(0, 20 * spec_max))
    sigma = FreeParameter("sigma", 0.06, valid_range=(0, 1))
    wvl = FreeParameter("wvl", wvl_guess, valid_range=(wavelengths[0], wavelengths[-1]))
    line = GaussianFunction("instrument_cal_line", inty, sigma, wvl, intensity_from_peak=True)

    model = SingleSpectraModel([baseline, line], spec)

    if plot_fit:
        plt.clf()
        model.plot()
        input('waiting...')

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
        for _, fit_param in model:
            if fit_param.free:
                value = fit_param.get_normalised_value()
                value += np.random.uniform(-1, 1, 1)[0] * value * 0.05
                fit_param.set_normalised_value(value)
        fit_results, fit_params = minuit.migrad()
        if not fit_results['is_valid']:
            for _, fit_param in model:
                if fit_param.free:
                    value = fit_param.get_normalised_value()
                    value += np.random.uniform(-1, 1, 1)[0] * value * 0.05
                    fit_param.set_normalised_value(value)
            fit_results, fit_params = minuit.migrad()
            if not fit_results['is_valid']:
                raise RuntimeError("Valid fit could not be found.")

    if plot_fit:
        plt.clf()
        model.plot()
        input('waiting...')

    return sigma.value


class InvalidFit(Exception):
    pass
