
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

import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt
import matplotlib.pyplot as plt

from raysect.core.math.random cimport uniform

from cherab.fitting.fit_parameters cimport FreeParameter
from cherab.fitting.spectral_models cimport MultiSpectraModel


cpdef mcmc_fit(MultiSpectraModel multi_spec_model, int n_points=20000, int n_thin=1, int n_burn=100, int n3=500,
               int max_cs=80000, double lambda_target=0.25, double kalf=0.95, int print_level=0, int plot_level=0):
    """
    Perform a Bayesian MCMC fit of the spectral model.

    :param MultiSpectraModel multi_spec_model:
    :param int n_points: Desired number of points used in calculation.
    :param int n_thin: Store MCMC iterations at a thinning interval of n_thin.
    :param int n_burn: Iterations used as burn in before storing results.
    :param int n3: number of independent parameter runs in sigma search.
    :param int max_cs: Max iterations of control system.
    :param double lambda_target: Target acceptance rate for markov chain.
    :param double kalf: empirical scaling factor for estimating single parameter acceptance rate.
    :return:
    """

    cdef:
        FreeParameter param, param_fixed
        list parameters, param_complete
        bint first_complete, complete
        int i, n_params, na, nr, i_burn, i_thin, i_points, cs_i, nap, ntap
        double best_likelihood, p1, p2, p_accept, r, tol2, alfm, accept1, meas_acc1
        double[:] last_position, best_values, p2_history
        double[:,:] param_history

    parameters = multi_spec_model.free_parameters
    n_params = len(parameters)
    param_history = np.zeros((n_params, n_points), dtype=np.float64)
    p2_history = np.zeros(n_points, dtype=np.float64)

    # load current settings as last position
    last_position = np.zeros(n_params, dtype=np.float64)
    best_values = np.zeros(n_params, dtype=np.float64)
    for i in range(n_params):
        last_position[i] = parameters[i].value
        best_values[i] = parameters[i].value

    best_likelihood = multi_spec_model.evaluate_loglikelihood()
    p1 = best_likelihood

    #################
    # BURN IN PHASE #
    #################

    i_burn = 1  # burn in counter
    while i_burn <= n_burn:

        # sample new coordinates in parameter space from proposal distributions
        for i in range(n_params):
            param = parameters[i]
            param.propose()

        # Evaluate unkown PDF/log-likelihood function at new coordinates
        p2 = multi_spec_model.evaluate_loglikelihood()
        # Ratio of likelihoods (p2/p1) gives acceptance probability.
        r = exp(p2 - p1)
        # Generate another random real for use in acceptance test
        p_accept = uniform()

        # If p_accept <= r then accept the move, otherwise stay in same place.
        if p_accept <= r:

            for i in range(n_params):
                param = parameters[i]
                last_position[i] = param.get_value()

            p1 = p2

            if p2 > best_likelihood:
                best_likelihood = p2
                best_values[:] = last_position[:]

        # If staying in the last position, need to reset the parameter values.
        else:

            for i in range(n_params):
                param = parameters[i]
                param.set_value(last_position[i])

        i_burn += 1

    #########################
    # Proposal sigma search #
    #########################

    alfm = 0.6426 + 0.1507 * exp(- n_params/15)  # scaling exponent for m=1 acceptance rate
    # empirical acceptance rate for single parameter, "acc(1)" in gregory.
    accept1 = lambda_target ** (1/(n_params**(kalf * alfm)))

    cs_i = 0  # total CS runs
    tol2 = 1.5 * sqrt(n3) / n3

    param_complete = [False for i in range(n_params)]
    first_complete = False
    complete = False

    while not complete:

        for j in range(n_params):

            if not first_complete and param_complete[j]:
                continue
            else:
                p_accept = uniform()
                if param_complete[j] and p_accept > 0.10:
                    continue

            param_fixed = parameters[j]

            nap = 0  # measured acceptance for this parameter, acc1
            ntap = 0  # total acceptance rate for all parameters, accept_all

            for k in range(n3):

                ###############################
                # First run with all parameters

                # sample new coordinates in parameter space from proposal distributions
                for i in range(n_params):
                    param = parameters[i]
                    param.propose()

                # Evaluate unkown PDF/log-likelihood function at new coordinates
                p2 = multi_spec_model.evaluate_loglikelihood()
                # Ratio of likelihoods (p2/p1) gives acceptance probability.
                r = exp(p2 - p1)
                # print('r {:.4f}'.format(r))
                # Generate another random real for use in acceptance test
                p_accept = uniform()

                # If p_accept <= r then accept the move, otherwise stay in same place.
                if p_accept <= r:
                    # print("p_accepted {:.4G}".format(p_accept))
                    ntap += 1
                    p1 = p2

                    for i in range(n_params):
                        param = parameters[i]
                        last_position[i] = param.get_value()

                    if p2 > best_likelihood:
                        best_likelihood = p2
                        best_values[:] = last_position[:]

                # If staying in the last position, need to reset the parameter values.
                else:
                    # print("p_rejected {:.4G}".format(p_accept))

                    for i in range(n_params):
                        param = parameters[i]
                        param.set_value(last_position[i])

                # print('--------------------------')

                ################################
                # Second run with all parameters

                # sample new coordinates for single parameter
                param_fixed.propose()

                # Evaluate unkown PDF/log-likelihood function at new coordinates
                p2 = multi_spec_model.evaluate_loglikelihood()
                # Ratio of likelihoods (p2/p1) gives acceptance probability.
                # print("p1 {:.7G} p2 {:.7G}".format(p1, p2))
                r = exp(p2 - p1)
                # print("r {:.4f}".format(r))
                # Generate another random real for use in acceptance test
                p_accept = uniform()
                # print('p_accept {:.4G}'.format(p_accept))

                # If p_accept <= r then accept the move, otherwise stay in same place.
                if p_accept <= r:
                    # print("p_accepted {:.4G}".format(p_accept))

                    nap += 1  # add to total acceptance for this parameter.
                    p1 = p2
                    # print('accepted, nap {}'.format(nap))

                    last_position[j] = param_fixed.get_value()

                    if p2 > best_likelihood:
                        best_likelihood = p2
                        best_values[:] = last_position[:]

                # If staying in the last position, need to reset the parameter values.
                else:
                    # print("p_rejected {:.4G}".format(p_accept))

                    # print('rejected, nap {}'.format(nap))
                    param_fixed.set_value(last_position[j])

                cs_i += 1

                # tr = uniform()
                # if tr <= 0.0025:
                #     plt.clf()
                #     for model in multi_spec_model.spectral_models:
                #         model.plot()
                #     input('waiting')

                # print('###########################')

            # Calculate new sigma based on measured sigma(1) and theoretical sigma(1)
            meas_acc1 = (<double> nap / <double> n3)  # measured acceptance(1)
            meas_acc_all = (<double> ntap / <double> n3)  # measured acceptance(all)
            delta = 0.01
            sigma1 = param_fixed.proposal_sigma
            sigma2 = sigma1 * sqrt(((meas_acc1 + delta)/accept1)*((1 - accept1)/(1 - meas_acc1 + delta)))
            if sigma2 > (param_fixed.vmax - param_fixed.vmin) * 0.1:
                param_fixed.proposal_sigma /= 10
                print("{} - Can't change parameter sigma to be larger than 10% of valid parameter range.".format(param_fixed.name))
            else:
                param_fixed.proposal_sigma = sigma2

            # mark this parameter as complete
            if -tol2 < meas_acc1 - accept1 < tol2:
                param_complete[j] = True
            else:
                param_complete[j] = False

            if print_level > 1:
                print('###########################')
                print('param {}'.format(param_fixed.name))
                print('nap: {}'.format(nap))
                print('measured accept1: {:.4G}'.format(meas_acc1))
                print('theoretical accept1: {:.4G}'.format(accept1))
                print('measured accept_all: {:.4G}'.format(meas_acc_all))
                print('current sigma1: {:.4G}'.format(sigma1))
                print('corrected sigma2: {:.4G}'.format(param_fixed.proposal_sigma))
                if nap == 0 or nap == n3:
                    print('failed value: {:.4G}'.format(param_fixed.get_value()))
                if param_complete[j]:
                    print('param {} complete with error {:.3G}%'.format(param_fixed.name, (meas_acc1 - accept1)*100))

            # input('waiting...')

        # input('waiting...')

        if cs_i >= max_cs:
            print('Max control system iterations reached. Moving on.')
            break
        elif all(param_complete) and first_complete:
            break
        elif all(param_complete):
            first_complete = True
            param_complete = [False for i in range(n_params)]

    #############################
    # MASTER MCMC SAMPLING LOOP #
    #############################

    # for i in range(n_params):
    #     param = parameters[i]
    #     print('name: {}, sigma => {:.4G}'.format(param.name, param.proposal_sigma))

    na = 0  # number accepted
    nr = 0  # number rejected
    i_thin = 1  # thin counter
    i_points = 0  # points counter
    param_history[:, 0] = last_position[:]

    while i_points < n_points:

        # Start loop for thinned values, skipped if n_thin = 1
        while i_thin != n_thin:

            # sample new coordinates in parameter space from proposal distributions
            for i in range(n_params):
                param = parameters[i]
                param.propose()

            p2 = multi_spec_model.evaluate_loglikelihood()  # Evaluate log-likelihood function at new coordinates
            r = exp(p2 - p1)  # Ratio of likelihoods (p2/p1) gives acceptance probability.
            p_accept = uniform()  # Generate another random real for use in acceptance test

            if p_accept <= r:  # If p_accept <= r then accept the move
                for i in range(n_params):
                    param = parameters[i]
                    last_position[i] = param.get_value()
                p1 = p2
                if p2 > best_likelihood:
                    best_likelihood = p2
                    best_values[:] = last_position[:]
            else:
                for i in range(n_params):
                    param = parameters[i]
                    param.set_value(last_position[i])

            i_thin += 1

        # sample new coordinates in parameter space from proposal distributions
        for i in range(n_params):
            param = parameters[i]
            param.propose()

        p2 = multi_spec_model.evaluate_loglikelihood()  # Evaluate log-likelihood function at new coordinates
        p2_history[i_points] = p2
        r = exp(p2 - p1)  # Ratio of likelihoods (p2/p1) gives acceptance probability.
        p_accept = uniform()  # Generate another random real for use in acceptance test

        # If p_accept <= r then accept the move, otherwise stay in same place.
        if p_accept <= r:

            for i in range(n_params):
                param = parameters[i]
                last_position[i] = param.get_value()

            p1 = p2

            param_history[:, i_points] = last_position[:]
            na += 1

            if p2 > best_likelihood:
                best_likelihood = p2
                best_values[:] = last_position[:]

        # If staying in the last position, need to reset the parameter values.
        else:

            for i in range(n_params):
                param = parameters[i]
                param.set_value(last_position[i])

            param_history[:, i_points] = last_position[:]
            nr += 1

        i_points += 1
        i_thin = 0

    for i in range(n_params):

        param = parameters[i]
        param_samples = param_history[i, :]

        # calculate stats
        peak = best_values[i]
        # calculate stats
        mean = np.mean(param_samples)
        maxai, minai = find_sigma_bounds(param_samples)

        # Save values
        param.set_value(mean)
        param.value_high = maxai
        param.value_low = minai

        if plot_level > 2:
            plt.figure()
            n, bin_edges, _ = plt.hist(param_samples, 40, normed=1, facecolor='green', alpha=0.75)
            plt.title("Sampled PDF of {}".format(param.name))
        if print_level > 0:
            print('param {} => peak: {:.6G}, mean: {:.6G}  upper: {:.6G}  lower: {:.6G}'.
                  format(param.name, peak, mean, maxai, minai))

    if plot_level > 3:
        # make histogram
        plt.figure()
        n, bin_edges, _ = plt.hist(p2_history, 40, normed=1, facecolor='green', alpha=0.75)
        plt.title("Distribution 2nd probability samples")

        for i in range(n_params):
            param = parameters[i]
            plt.figure()
            plt.plot(param_history[i, :], 'k.')
            plt.title('{} point history'.format(param.name))

    if print_level > 0:
        print("Number accepted => {:4G}".format(na))
        print("Number rejected => {:4G}".format(nr))
        print("Acceptance ratio => {:4G}".format(<double> na / <double> (n_points)))

    if plot_level > 2:
        plt.show()


def find_sigma_bounds(param_samples):

    nai = 50  # number of statistics bins

    # make histogram of samples
    hist, edges = np.histogram(param_samples, bins=nai)
    d_edge = edges[1] - edges[0]
    midpoints = []
    for i in range(len(edges) - 1):
        midpoints.append(edges[0] + i * d_edge)

    # obtain a list of bins sorted by probability
    yai = hist / (hist.sum() * d_edge)
    yai_sum = yai.sum()
    raw_bin_list = []
    for j in range(len(hist)):
        raw_bin_list.append((yai[j] / yai_sum, midpoints[j]))
    pbin = sorted(raw_bin_list, key=lambda x: x[0], reverse=True)

    # find upper and lower one sigma bounds
    minai = 1E99
    maxai = -1E99
    area = 0
    k = 0

    while area <= 0.683:

        area += pbin[k][0]
        if pbin[k][1] < minai:
            minai = pbin[k][1]
        if pbin[k][1] > maxai:
            maxai = pbin[k][1]

        k += 1

    return maxai, minai
