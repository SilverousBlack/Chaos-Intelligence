"""Chaos Intelligence Deterministic Functions Module

Defines the base Deterministic Function class, and other common deterministic functions.

Imports:
    base_utils.imports | *
Objects:
    BaseDeterministicFunction
    LorenzAttractor (BaseDeterministicFunction)
"""

from base_utils.imports import *

class BaseDeterministicFunction():
    """Base Class of Deterministic Function Classes.
    
    Deterministic functions must be sensitive to initial conditions, hence with a functor, these variables can be tracked.
    
    This object does nothing, aside from defining required contents of the functor. 
    """
    def __init__(self, **InitialConditions):
        """Creates a new Base Deterministic Function object.
        
        Args:
            InitialConditions (`dict[str, any]`, optional): Initial conditions to be saved in the object (saved in `.initial_conditions`). 
        """
        self.initial_conditions = {}
        for var in InitialConditions:
            self.initial_conditions[var] = InitialConditions[var]
    
    def __call__(self, **args):
        """See {}.call() for information.
        """.format(type(self).__name__)
        return self.call(**args)
    
    @abc.abstractmethod
    def call(self, TargetObject: object = object()):
        """Applies the functor on the input.
        
        Required functionality of all daughter classes, not implemented on base class.

        Raises:
            NotImplementedError: Functionality must be defined.
            
        Args:
            TargetObject (`object`, optional): Standard argument for daughter classes, object to be accessed in operations. Defaults to `object()`.
        """
        raise NotImplementedError("Functionality must be defined.")
    
    @abc.abstractmethod
    def reset(self):
        """Applies the functor on the input.
        
        Required functionality of all daughter classes, not implemented on base class.

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError("Functionality must be defined.")
    
    def __repr__(self):
        return "Deterministic Function [{}]: {}".format(type(self).__name__, self.initial_conditions)

class LorenzAttractor(BaseDeterministicFunction):
    def __init__(self,
                 beta: typing.SupportsFloat = 0,
                 rho: typing.SupportsFloat = 28,
                 sigma: typing.SupportsFloat = 10,
                 x_max: typing.SupportsFloat = random.randint(0, 10000),
                 y_max: typing.SupportsFloat = random.randint(0, 10000),
                 z_max: typing.SupportsFloat = random.randint(0, 10000)):
        """Creates a new Lorenz Attractor Instance.
        
        Possible values of x, y, z in calculation is defined as uniform distribution within [0, max].

        Args:
            beta (`typing.SupportsFloat`, optional): \u03D0 to be used in calculations. Defaults to `0`.
            rho (`typing.SupportsFloat`, optional): \u03C1 to be used in calculations. Defaults to `28`.
            sigma (`typing.SupportsFloat`, optional): \u03C3 to be used in calculations. Defaults to `10`.
            x_max (`typing.SupportsFloat`, optional): Maximum value of `x` in calculation. Defaults to random.randint(0, 10000).
            y_max (`typing.SupportsFloat`, optional): Maximum value of `y` in calculation. Defaults to random.randint(0, 10000).
            z_max (`typing.SupportsFloat`, optional): Maximum value of `z` in calculation. Defaults to random.randint(0, 10000).
        """
        super(LorenzAttractor, self).__init__(beta=beta, rho=rho, sigma=sigma,
                                              x_max=x_max, y_max=y_max, z_max=z_max)
    
    def __call__(self,
             target_object: object,
             EntropyFunction: typing.Callable = random.uniform,
             SingleOutput: bool = False):
        """[summary]

        Args:
            target_object (object): [description]
            EntropyFunction (typing.Callable, optional): [description]. Defaults to random.uniform.
            SingleOutput (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        index = EntropyFunction(0, 2)
        xx = EntropyFunction(0, self.initial_conditions["x_max"])
        yy = EntropyFunction(0, self.initial_conditions["y_max"])
        zz = EntropyFunction(0, self.initial_conditions["z_max"])
        x_dot = self.initial_conditions["sigma"] * (yy - xx)
        y_dot = self.initial_conditions["rho"] * xx - yy - xx * zz
        z_dot = xx * yy - self.initial_conditions["beta"] * zz
        if SingleOutput:
            return list([x_dot, y_dot, z_dot])[index]
        else:
            return tf.constant([x_dot, y_dot, z_dot])
    
    def reset(self,
              beta: typing.SupportsFloat = 0, 
              rho: typing.SupportsFloat = 28, 
              sigma: typing.SupportsFloat = 10,
              x_max: typing.SupportsFloat = random.randint(0, 10000), 
              y_max: typing.SupportsFloat = random.randint(0, 10000), 
              z_max: typing.SupportsFloat = random.randint(0, 10000)):
        self.initial_conditions["beta"] = beta
        self.initial_conditions["rho"] = rho
        self.initial_conditions["sigma"] = sigma
        self.initial_conditions["x_max"] = x_max
        self.initial_conditions["y_max"] = y_max
        self.initial_conditions["z_max"] = z_max

class CollatzConjecture(BaseDeterministicFunction):
    pass
    