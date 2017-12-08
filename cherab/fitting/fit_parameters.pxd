
from cherab.core.math.interpolators.interpolators1d cimport Interpolate1DCubic
from cherab.core.math.function cimport Function1D


cdef class Parameter:
    cdef:
        public bint free
        public str name

    cdef double get_value(self)

    cdef void set_value(self, double value)


cdef class FreeParameter(Parameter):

    cdef:
        double _value
        public double initial_value, error, vmin, vmax, value_high, value_low, _previous_value, proposal_sigma
        public tuple range

    cdef double get_value(self)

    cdef void set_value(self, double value)

    cdef double cget_normalised_value(self)

    cpdef double get_normalised_value(self)

    cdef void cset_normalised_value(self, double value)

    cpdef set_normalised_value(self, double value)

    cdef double cget_normalised_error(self)

    cpdef double get_normalised_error(self)

    cdef void cset_normalised_error(self, double value)

    cpdef set_enormalised_error(self, double value)

    cdef void reset_to_previous_value(self)

    cpdef propose_new_value(self)

    cdef void propose(self)


cdef class FixedParameter(Parameter):

    cdef:
        double _value

    cdef double get_value(self)

    cdef void set_value(self, double value)


cdef class LinkedParameter(Parameter):

    cdef:
        public double multiple
        public Parameter source

    cdef double get_value(self)

    cdef void set_value(self, double value)


cdef class FixedRatioParameter(Parameter):

    cdef public FreeParameter param1, param2, param3

    cdef double get_value(self)

    cdef void set_value(self, double value)


cdef class SampledParameter(Parameter):

    cdef:
        public double x
        public Function1D profile

    cdef double get_value(self)

    cdef void set_value(self, double value)


cdef class FittedPlasmaProfile(Function1D):

    cdef:
        public str name
        public list psi_coordinates, parameters
        public Interpolate1DCubic profile

    cpdef list free_parameters(self)

    cpdef update_parameter_values(self, list parameters)

    cdef double evaluate(self, double psi) except? -1e999


cdef class RadialProfile(Function1D):

    cdef:
        public str name
        public Parameter n
        public Parameter m
        public Parameter a

    cpdef list free_parameters(self)

    cpdef update_parameter_values(self, list parameters)

    cdef double evaluate(self, double r) except? -1e999

cdef class SkewGaussProfile(Function1D):

    cdef:
        public str name
        public Parameter a
        public Parameter mu
        public Parameter sigma
        public Parameter alpha
        public Parameter a0

    cpdef list free_parameters(self)

    cpdef update_parameter_values(self, list parameters)

    cdef double evaluate(self, double r) except? -1e999
