
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
# cython: boundscheck=False
# cython: cdivision=True

# External Imports
import numpy as np
from scipy import zeros
import matplotlib.pyplot as plt
from math import log


cdef class SingleSpectraModel:

    def __init__(self, list sources_to_fit, spectrum_to_fit, FreeParameter offset_parameter=None):

        self.sources_to_fit = sources_to_fit
        self.num_of_sources = len(sources_to_fit)
        self.spectra_to_fit = spectrum_to_fit
        self.offset_parameter = offset_parameter

        # load working arrays to avoid recreating empty arrays on every iteration
        # TODO - this will be broken if spectra changes length, add notifier???
        ndata = len(spectrum_to_fit.samples)
        self.ndata = ndata
        self._samples = zeros(ndata, np.float64)
        self._evaluation = zeros(ndata, dtype=np.float64)
        self._local_wvls = np.array(spectrum_to_fit.wavelengths, dtype=np.float64)

        # load constants for bayesian loglikelihood calculation

        self._errfact = np.log(spectrum_to_fit.errors).sum()
        self._logfact = - (ndata/2) * log(2*np.pi)

        cdef float free_params = 0.0, data_points

        # TODO - check this isn't a bug. May not be updated when parameter states change.
        # Count number of free params in model
        for source_function in sources_to_fit:
            for parameter in source_function.fit_parameters:
                if parameter.free:
                    free_params += 1.0

        # TODO - temporarily adding extra wavelength axis for shift. Should be removed or redesigned.
        # TODO - clean up and remove duplicates
        self._nd = len(self.spectra_to_fit.get_wavelengths())
        self._xvec_reference = np.array(self.spectra_to_fit.get_wavelengths())
        self._xvec_working = np.zeros(self._nd)
        if self.offset_parameter:
            free_params += 1.0

        # Get number of data points to fit
        data_points = len(spectrum_to_fit.samples)

        # Calculate degrees of freedom
        self.dof = data_points - free_params

    property free_parameters:
        def __get__(self):
            paramlist = []
            for source_function in self.sources_to_fit:
                for param in source_function.fit_parameters:
                    if param.free:
                        paramlist.append(param)

            if self.offset_parameter:
                paramlist.append(self.offset_parameter)

            return paramlist

    def __iter__(self):
        for source_function in self.sources_to_fit:
            for parameter in source_function.fit_parameters:
                yield(parameter.name, parameter)

        if self.offset_parameter:
            yield(self.offset_parameter.name, self.offset_parameter)

    def __call__(self, *parameters):
        """ Evaluate the model for a list of parameters.

        :param parameters:
        :return:
        """
        parameters = list(parameters)
        self.update_parameters(parameters)
        chi2 = self.evaluate_chi2() / self.dof
        return chi2

    cpdef update_parameters(self, list parameters):
        # Update the values in the model
        cdef SpectralSource source_function
        for i in range(self.num_of_sources):
            source_function = self.sources_to_fit[i]
            source_function.update_parameter_values(parameters)

        if self.offset_parameter:
            if self.offset_parameter.free:
                self.offset_parameter.set_normalised(parameters.pop(0))

    cpdef double evaluate_chi2(self):
        cdef double[:] wavelengths, data, error, temp_array
        cdef int i, j, nd
        cdef double chi2_value = 0.0, wvl_offset
        cdef SpectralSource source_function

        nd = self.spectra_to_fit.num_wvl_bins
        wavelengths = self.spectra_to_fit.get_wavelengths()
        data = self.spectra_to_fit.get_spectral_samples()
        error = self.spectra_to_fit.get_errors()

        if self.offset_parameter:
            wvl_offset = self.offset_parameter.value
            for i in range(nd):
                self._local_wvls[i] = wavelengths[i] + wvl_offset
        else:
            self._local_wvls = wavelengths

        # Cycle through each spectral source function and grab the latest values
        for i in range(nd):
            self._samples[i] = 0.0
        for i in range(self.num_of_sources):
            source_function = self.sources_to_fit[i]
            self._evaluation = source_function.evaluate(self._local_wvls, self._evaluation)
            for j in range(nd):
                self._samples[j] += self._evaluation[j]

        # if inst_func is not None:
        #     samples = inst_func.extrapol_convolve(samples)

        # calculate and return chi-squared
        for i in range(nd):
            chi2_value += (self._samples[i] - data[i])**2 / error[i]**2

        return chi2_value

    cpdef double[:] evaluate_spectrum(self):
        cdef double[:] wavelengths, data, error, samples, working_array, temp_array
        cdef double wvl_offset
        cdef int i, nd
        cdef SpectralSource source_function

        # nd = self.spectra_to_fit.num_wvl_bins
        nd = self._nd
        wavelengths = self.spectra_to_fit.get_wavelengths()
        data = self.spectra_to_fit.get_spectral_samples()
        error = self.spectra_to_fit.get_errors()

        if self.offset_parameter:
            wvl_offset = self.offset_parameter.get_value()
            for i in range(nd):
                self._xvec_working[i] = self._xvec_reference[i] + wvl_offset
        else:
            self._xvec_working = self._xvec_reference

        # Cycle through each spectral source function and grab the latest values
        for i in range(nd):
            self._samples[i] = 0.0
        for i in range(self.num_of_sources):
            source_function = self.sources_to_fit[i]
            self._evaluation = source_function.evaluate(self._xvec_working, self._evaluation)
            for j in range(nd):
                self._samples[j] += self._evaluation[j]

        # if inst_func is not None:
        #     samples = inst_func.extrapol_convolve(samples)

        return self._samples

    cdef double loglikelihood(self):

        cdef double[:] mvec, yvec, evec, cvec
        cdef double cval, chi2 = 0.0

        # evaluate model predictions vector from current parameter values
        mvec = self.evaluate_spectrum()

        yvec = self.spectra_to_fit.get_spectral_samples()
        evec = self.spectra_to_fit.get_errors()

        # evaluate chi squared
        for i in range(self._nd):
            cval = (yvec[i] - mvec[i]) / evec[i]
            chi2 += cval * cval

        # evaluate log likelihood for these particular parameter values
        # return self._logfact - self._errfact - 0.5*chi2
        return - 0.5*chi2

    def plot(self, wavelength=None):
        """ Plot current fit values """

        # Plot input spectra with errors
        spectrum = self.spectra_to_fit
        name = spectrum.name

        if self.offset_parameter:
            local_wvls = np.array(spectrum.wavelengths)
            local_wvls += self.offset_parameter.value
        else:
            local_wvls = np.array(spectrum.wavelengths)

        plt.errorbar(local_wvls, spectrum.samples, spectrum.errors, label=name+'_data')

        # Plot models with errors
        samples = zeros(local_wvls.shape[0], np.float64)
        working_array = zeros(local_wvls.shape[0], np.float64)
        for source_function in self.sources_to_fit:
            this_sample = source_function(local_wvls, working_array)
            # if inst_func is not None:
            #     this_sample = inst_func.extrapol_convolve(this_sample)
            samples += this_sample
            plt.plot(local_wvls, this_sample, label=name+'_'+source_function.name)
        plt.plot(local_wvls, samples, label=name+'_model')

    def get_minuit_args(self):
        """ Arguments for minuit. """

        forced_parameters = []
        keyword_args = {}

        for name, parameter in self:
            if parameter.free:
                forced_parameters.append(name)
                keyword_args[name] = parameter.get_normalised_value()
                keyword_args["limit_" + name] = (0.0, 1.0)
                keyword_args["error_" + name] = parameter.get_normalised_error()

        return forced_parameters, keyword_args

cdef class MultiSpectraModel:

    def __init__(self, list spectral_models, list profiles):

        cdef float combined_dof = 0.0

        self.spectral_models = spectral_models
        self.profiles = profiles

        # Get combined degrees of freedom
        for model in spectral_models:
            combined_dof += model.dof

        for profile in profiles:
            combined_dof += len(profile.free_parameters())

        self.dof = combined_dof

    def __call__(self, *parameters):
        """ Evaluate the model for a list of parameters.

        :param parameters:
        :return:
        """

        parameters = list(parameters)

        for profile in self.profiles:
            profile.update_parameter_values(parameters)

        for model in self.spectral_models:
            model.update_parameters(parameters)

        chi2 = 0.0
        for model in self.spectral_models:
            chi2 += model.evaluate_chi2()

        chi2 /= self.dof
        if np.isnan(chi2):
            chi2 = 1.0e6
        return chi2

    def __iter__(self):
        for model in self.spectral_models:
            yield(model)

    property free_parameters:
        def __get__(self):
            free_params = []
            for profile in self.profiles:
                free_params.extend(profile.free_parameters)
            for model in self.spectral_models:
                free_params.extend(model.free_parameters)
            return free_params

    cdef double evaluate_loglikelihood(self):
        cdef double loglike = 0, chi2
        cdef SingleSpectraModel model

        for model in self.spectral_models:
            chi2 = model.loglikelihood()
            loglike += chi2

        return loglike

    cdef double evaluate_chi2(self):
        cdef double chi2 = 0
        cdef SingleSpectraModel model

        for model in self.spectral_models:
            chi2 += model.evaluate_chi2()

        return chi2 / self.dof

    def chi2(self):
        return self.evaluate_chi2()

    def random_nudge_params(self):
        for model in self:
            for _, fit_param in model:
                if fit_param.free:
                    fit_param.propose_new_value()

    def get_minuit_args(self):

        forced_parameters = []
        keyword_args = {}

        for profile in self.profiles:
            for free_param in profile.free_parameters():
                name = free_param.name
                forced_parameters.append(name)
                # initial keyword value, i.e. x = 2
                keyword_args[name] = free_param.get_normalised_value()
                keyword_args["limit_" + name] = (0.0, 1.0)
                keyword_args["error_" + name] = free_param.get_normalised_value() * 0.1

        for single_model in self.spectral_models:
            for name, parameter in single_model:
                if parameter.free:
                    forced_parameters.append(name)
                    keyword_args[name] = parameter.get_normalised_value()
                    keyword_args["limit_" + name] = (0.0, 1.0)
                    keyword_args["error_" + name] = parameter.get_normalised_value() * 0.1

        return forced_parameters, keyword_args
