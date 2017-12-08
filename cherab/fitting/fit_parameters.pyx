
## cython: profile=True
# cython: cdivision=True

from numpy import pi as PI, abs
from libc.math cimport sqrt, exp, erfc
from copy import deepcopy

from raysect.core.math.random cimport normal


cdef class Parameter:
    """Base class for all parameters used in models that may or may not be fitable."""

    property value:
        def __get__(self):
            raise NotImplementedError()

    cdef double get_value(self):
        raise NotImplementedError()

    cdef void set_value(self, double value):
        raise NotImplementedError()


cdef class FreeParameter(Parameter):
    """A fit parameter that can be freely varied."""

    def __init__(self, str name, double value, error=None, sigma_prop=None, tuple valid_range=None):
        """ Initialise the FitParameter class.

        :param name: Parameter name (str)
        :param value: The actual starting value of the parameter (int/float, etc)
        :param error: An initial guess at the standard error
        :param sigma_prop: Proposal sigma
        :param valid_range: Tuple defining the upper and lower bounds.
        :return: FreeParameter instance
        """

        if not error:
            error = value/10.0

        self.name = name
        self._value = value
        self._previous_value = value
        self.initial_value = value
        self.error = error
        if sigma_prop:
            self.proposal_sigma = sigma_prop
        else:
            self.proposal_sigma = abs(error)
        self.vmin = valid_range[0]
        self.vmax = valid_range[1]
        self.value_high = 0.0
        self.value_low = 0.0
        self.free = True

    property value:
        def __get__(self):
            return self._value
        def __set__(self, value):
            self._value = value

    cdef double get_value(self):
        return self._value

    cdef void set_value(self, double value):
        self._value = value

    cdef double cget_normalised_value(self):
        return (self._value - self.vmin) / (self.vmax - self.vmin)

    cpdef double get_normalised_value(self):
        return self.cget_normalised_value()

    cdef void cset_normalised_value(self, double value):
        self._value = value * (self.vmax - self.vmin) + self.vmin

    cpdef set_normalised_value(self, double value):
        self.cset_normalised_value(value)

    cdef double cget_normalised_error(self):
        return self.error / (self.vmax - self.vmin)

    cpdef double get_normalised_error(self):
        return self.cget_normalised_error()

    cdef void cset_normalised_error(self, double value):
        self.error = value * (self.vmax - self.vmin)

    cpdef set_enormalised_error(self, double value):
        self.cset_normalised_error(value)

    cdef void reset_to_previous_value(self):
        self._value = self._previous_value

    cpdef propose_new_value(self):
        self.propose()

    cdef void propose(self):

        cdef double xj

        # Save previous value of this parameter
        self._previous_value = self._value

        xj = normal(self.get_value(), self.proposal_sigma)
        # reflect proposals of boundaries
        if xj < self.vmin:
            self._value = self.vmin + (self.vmin - xj)
        elif xj > self.vmax:
            self._value = self.vmax - (xj - self.vmax)
        else:
            self._value = xj

cdef class FixedParameter(Parameter):
    """A Fixed parameter that can takes a set value."""

    def __init__(self, str name, double value):
        """ Initialise the FitParameter class.

        :param name: Parameter name (str)
        :param value: The actual starting value of the parameter (int/float, etc)
        :return: FixedParameter instance
        """

        self.name = name
        self._value = value
        self.free = False

    property value:
        def __get__(self):
            return self._value

    cdef double get_value(self):
        return self._value

    cdef void set_value(self, double value):
        raise RuntimeError("Can't set a value on a Fixed Parameter")

    @staticmethod
    def from_free(parameter):
        if not isinstance(parameter, FreeParameter):
            raise ValueError("The parameter given as argument is not a free parameter.")
        return FixedParameter(parameter.name, parameter.value)


cdef class LinkedParameter(Parameter):
    """A Linked parameter that takes its value from another parameter."""

    def __init__(self, str name, Parameter source, double multiple=1.0):

        self.name = name
        self.source = source
        self.multiple = multiple
        self.free = False

    property value:
        def __get__(self):
            return self.source.value * self.multiple

    cdef double get_value(self):
        return self.source.get_value() * self.multiple

    cdef void set_value(self, double value):
        raise RuntimeError("Can't set a value on a Fixed Parameter")


cdef class FixedRatioParameter(Parameter):

    def __init__(self, str name, FreeParameter param1, FreeParameter param2, FreeParameter param3):
        """
        A Fixed parameter that sets its value by a ratio related to three other fit parameters.

        Imagine we have two other fit parameters that form a ratio, Param1/Param2 = r12.
        Now suppose we have this same line ratio appearing in another spectrum and we want to link them with the same ratio.
        The new parameters are Param3 and Param4, where Param3 is free, and Param4 will be fixed such that the ratio
        Param3/Param4 = r12

        Hence Param1/Param2 = Param3/Param4

        :param str name: Parameter name
        :param FreeParameter param1: parameter 1
        :param FreeParameter param2: parameter 2
        :param FreeParameter param3: parameter 3
        """

        self.name = name
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3
        self.free = False

    property value:
        def __get__(self):
            return (self.param3.get_value() * self.param2.get_value()) / self.param1.get_value()

    cdef double get_value(self):
        return (self.param3.get_value() * self.param2.get_value()) / self.param1.get_value()

    cdef void set_value(self, double value):
        raise RuntimeError("Can't set a value on a Fixed Parameter")


cdef class SampledParameter(Parameter):
    """A parameter that takes its value by sampling from an interpolator."""

    def __init__(self, str name, profile, float x):

        self.name = name
        self.profile = profile
        self.x = x
        self.free = False

    property value:
        def __get__(self):
            return self.profile(self.x)

    cdef double get_value(self):
        return self.profile.evaluate(self.x)

    cdef void set_value(self, double value):
        raise RuntimeError("Can't set a value on a Sampled Parameter")


cdef class FittedPlasmaProfile(Function1D):
    """An interpolated profile that holds a set of fit parameters."""

    def __init__(self, str name, list coordinates, list parameters):

        self.name = name
        self.psi_coordinates = coordinates
        self.parameters = parameters

        values = [param.value for param in parameters]

        self.profile = Interpolate1DCubic(coordinates, values, extrapolate=True)

    property free_parameters:
        def __get__(self):
            return [param for param in self.parameters if param.free]

    @staticmethod
    def from_raw_values(str name, list coordinates, list param_values, valid_range=None, bint zero_at_edge=False):

        cdef list coordinates_copy
        coordinates_copy = deepcopy(coordinates)

        if valid_range:
            params = [FreeParameter("{}_{}".format(name, i), v, valid_range=valid_range)
                      for i, v in enumerate(param_values)]
        else:
            params = [FreeParameter("{}_{}".format(name, i), v) for i, v in enumerate(param_values)]

        if zero_at_edge:
            coordinates_copy.append(1.0)
            params.append(FixedParameter("{}_edge".format(name), 0.0))

        return FittedPlasmaProfile(name, coordinates_copy, params)

    def set_active_parameters(self, radius_inner, radius_outer):

        for i, param in enumerate(self.parameters):
            try:
                if radius_inner <= self.psi_coordinates[i] <= radius_outer:
                    param.free = True
                else:
                    param.free = False
            except AttributeError:
                continue

    cpdef list free_parameters(self):
        cdef Parameter param
        return [param for param in self.parameters if param.free]

    cpdef update_parameter_values(self, list parameters):
        cdef Parameter param
        cdef bint changed = False

        for param in self.parameters:
            if param.free:
                param.set_value(parameters.pop(0))
                changed = True

        if changed:
            values = [param.value for param in self.parameters]
            self.profile = Interpolate1DCubic(self.psi_coordinates, values, extrapolate=True)

    cdef double evaluate(self, double psi) except? -1e999:
        return self.profile(psi)


cdef class RadialProfile(Function1D):
    """A stiff profile function for fitting.

    Follows the equation a(1-r^n)^m. Not scientifically real, but gives a good profile close to data from experience.
    """

    def __init__(self, str name, double n=1.0, double m=1.0, double a=1.0):
        self.name = name
        self.n = FreeParameter(name+"_profile_n", n)
        self.m = FreeParameter(name+"_profile_m", m)
        self.a = FreeParameter(name+"_profile_a", a)

    cpdef list free_parameters(self):
        cdef Parameter param
        return [param for param in [self.n, self.m, self.a] if param.free]


    cpdef update_parameter_values(self, list parameters):
        cdef Parameter param

        for param in [self.n, self.m, self.a]:
            if param.free:
                param.set_value(parameters.pop(0))

    cdef double evaluate(self, double r) except? -1e999:
        cdef double a, n, m

        a = self.a.get_value()
        n = self.n.get_value()
        m = self.m.get_value()

        return a * (1 - (r - 3.0)**n )**m


cdef class SkewGaussProfile(Function1D):
    """A skew gaussian profile function for fitting.

    Also known as the skew gaussian distribution. Gives a reasonable fit to data.
    """

    def __init__(self, str name, double a=1.0, double mu=1.0, double sigma=1.0, double alpha=0.0, double a0=0.0):
        self.name = name
        self.a = FreeParameter(name+"_profile_a", a)
        self.mu = FreeParameter(name+"_profile_mu", mu)
        self.sigma = FreeParameter(name+"_profile_sigma", sigma)
        self.alpha = FreeParameter(name+"_profile_alpha", alpha)
        self.a0 = FreeParameter(name+"_profile_a0", a0)

    cpdef list free_parameters(self):
        cdef Parameter param
        return [param for param in [self.a, self.mu, self.sigma, self.alpha, self.a0] if param.free]

    cpdef update_parameter_values(self, list parameters):
        cdef Parameter param

        for param in [self.a, self.mu, self.sigma, self.alpha, self.a0]:
            if param.free:
                param.set_value(parameters.pop(0))

    cdef double evaluate(self, double r) except? -1e999:
        cdef double a, sigma, mu, alpha, a0, sigt, i0

        # extract parameter values
        a = self.a.get_value()
        sigma = self.sigma.get_value()
        mu = self.mu.get_value()
        alpha = self.alpha.get_value()
        a0 = self.a0.get_value()

        sigt = sqrt(sigma) * mu

        i0 = a / (sigt * sqrt(2 * PI))

        return i0 * exp(-(r-mu)**2 / (2 * sigt**2)) * erfc(-alpha * (r-mu) / (sqrt(2) * sigt)) + a0
