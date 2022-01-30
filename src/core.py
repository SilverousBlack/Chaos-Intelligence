"""Chaos Intelligence Core Module.

This module defines core classes and functionalities of Chaos Intelligence.

Imports:
    base_utils.imports | *
Objects:
    CoreLayer (tensorflow.keras.layers.layer) | Base Class of Chaotic Layers
    CoreLayerNoBlindOverride (tensorflow.keras.layers.layer) | Base Class Variation of Chaotic Layers with Internal Overriding Disabled
    UniversalCoreLayer (tensorflow.keras.layers.layer) | Base Class Variation of Chaotic Layers with Optional Override Toggle
"""

from inspect import isclass
import random
import tensorflow as tf
from tensorflow.keras import layers, Model
import typing

class CoreLayer(layers.Layer):
    """The Base Class of Chaotic Layers. Follows Keras Functional API.
    
    Applies a different function by virtue of deterministic function.
    
    This Base Class serves as fundamental building block of all Chaotic Layers."""
    def __init__(self,
                 DeterministicFunction: typing.Callable = lambda s: 0,
                 RandomFunction: typing.Callable = lambda: random.random(),
                 LayerSizeUnits: int = 32,
                 SuperInitArgs: dict = {},
                 DetFuncArgs: dict = {},
                 RandFuncArgs: dict = {},
                 Functions: list = [lambda s, i: tf.reduce_sum(i)],
                 **OtherVars):
        """Creates a new Chaos Core Layer. A Chaos Layer can have any number functions that might be applied to input, by virtue of the Deterministic Function.

        Args:
            DeterministicFunction (`typing.Callable`, optional): Internally used deterministic function, return must be must be a numerical value. Defaults to `lambda:0`.
            RandomFunction (`typing.Callable`, optional): Internally used random function, for entropy, may be used externally by `self.use_rand()`. Defaults to `lambda:random.random()`.
            LayerSizeUnits (`int`, optional): Size or Number of Units in the Layer . Defaults to `32`.
            SuperInitArgs (`dict`, optional): Arguments to be passed to `super()` initializing  `tensorflow.keras.layers.Layer`. Defaults to `{}`.
            DetFuncArgs (`dict`, optional): Optional arguments to be passed to the deterministic function when called, arguments are recorded. Defaults to `{}`.
            RandFuncArgs (`dict`, optional): Optional arguments to be passed to the random function when called, arguments are recorded. Defaults to `{}`.
            Functions (`list`, optional): List of functions that might be applied, must be at least one (1) function. Defaults to `[lambda s, i: tf.reduce_sum(i)]`.
            OtherVars (`typing.Any`, optional): Arguments that will be stored in the object, may override existing stored. Defaults to `{}`.
            
        The result of the deterministic function maybe trivial, hence use the additional argument `interpreter_function=` to bind an interpreter.
        """
        if (len(Functions) < 1):
            raise ValueError("Function list must contain at least one (1) function.")
        super(CoreLayer, self).__init__(**SuperInitArgs)
        self.nature = "Core"
        self.deterministic_function = DeterministicFunction
        self.entropy_function = RandomFunction
        self.functions = Functions
        self.local_history = {"dfunc_args": DetFuncArgs, "rfunc_args": RandFuncArgs}
        self.units = LayerSizeUnits
        for var in OtherVars:
            setattr(self, var, OtherVars[var])
        
    def __repr__(self):
        return "Chaos {} Layer [{}]: {} unit(s), {} function(s)".format(self.nature, self.name, self.units, len(self.functions))
    
    def ensure_in_range(self, index: typing.SupportsFloat):
        if hasattr(self, "interpreter_function"):
            return self.interpreter_function(index)
        return int((index * len(self.functions)) % len(self.functions))
    
    def call(self,
             inputs:tf.Tensor):
        """Computes input tensor.
        
        Calculates the deterministic function given the saved arguments.
        Current input and calculated entropy, is saved at `local_history`

        Args:
            inputs (tf.Tensor): input tensor to be computed.

        Returns:
            tf.Tensor: computed tensor.
        """
        self.local_history["current_input"] = input
        self.local_history["current_entropy"] = self.use_rand()
        index = self.ensure_in_range(self.deterministic_function(self, **self.local_history["dfunc_args"]))
        self.local_history["current_targetfunc"] = index
        return self.functions[index](self, inputs)
        
    def build(self,
              input_shape: typing.Iterable,
              WeightArgs: dict = {},
              BiasArgs: dict = {}):
        """Builds chaos layer.

        Args:
            input_shape (typing.Iterable): iterable shape of inputs.
            WeightArgs (dict, optional): Arguments to be passed on `self.add_weight()` for `self.w`, with exception to be shape. Defaults to {}.
            BiasArgs (dict, optional): Arguments to be passed on `self.add_weight()` for `self.b`, with exception of the shape. Defaults to {}.
        """
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            **WeightArgs
        )
        self.b = self.add_weight(
            shape=(self.units,),
            **BiasArgs
        )
        
    def use_rand(self, NewArgs = None):
        if NewArgs is not None:
            self.local_history["rfunc_args"] = NewArgs
        return self.entropy_function(**self.local_history["rfunc_args"])if isinstance(self.local_history["rfunc_args"], dict) else self.entropy_function(self.local_history["rfunc_args"])
    
class CoreLayerNoBlindOverride(layers.Layer):
    """A Variation of Base Class of Chaotic Layers. Follows Keras Functional API.
    
    Applies a different function by virtue of deterministic function.
    No Blind Ovveride means that hidden overriding feature is disabled.
    
    This Base Class serves as fundamental building block of all Chaotic Layers."""
    def __init__(self,
                 DeterministicFunction: typing.Callable = lambda s: 0,
                 RandomFunction: typing.Callable = lambda: random.random(),
                 LayerSizeUnits: int = 32,
                 SuperInitArgs: dict = {},
                 DetFuncArgs: dict = {},
                 RandFuncArgs: dict = {},
                 Functions: list = [lambda s, i: tf.reduce_sum(i)],
                 **OtherVars):
        """Creates a new Chaos Core Layer. A Chaos Layer can have any number functions that might be applied to input, by virtue of the Deterministic Function.

        Args:
            DeterministicFunction (`typing.Callable`, optional): Internally used deterministic function, return must be must be a numerical value. Defaults to `lambda:0`.
            RandomFunction (`typing.Callable`, optional): Internally used random function, for entropy, may be used externally by `self.use_rand()`. Defaults to `lambda:random.random()`.
            LayerSizeUnits (`int`, optional): Size or Number of Units in the Layer . Defaults to `32`.
            SuperInitArgs (`dict`, optional): Arguments to be passed to `super()` initializing  `tensorflow.keras.layers.Layer`. Defaults to `{}`.
            DetFuncArgs (`dict`, optional): Optional arguments to be passed to the deterministic function when called, arguments are recorded. Defaults to `{}`.
            RandFuncArgs (`dict`, optional): Optional arguments to be passed to the random function when called, arguments are recorded. Defaults to `{}`.
            Functions (`list`, optional): List of functions that might be applied, must be at least one (1) function. Defaults to `[lambda s, i: tf.reduce_sum(i)]`.
            OtherVars (`typing.Any`, optional): Arguments that will be stored in the object, may override existing stored. Defaults to `{}`.
            
        The result of the deterministic function maybe trivial, hence use the additional argument `interpreter_function=` to bind an interpreter.
        """
        if (len(Functions) < 1):
            raise ValueError("Function list must contain at least one (1) function.")
        super(CoreLayer, self).__init__(**SuperInitArgs)
        self.nature = "Core"
        self.deterministic_function = DeterministicFunction
        self.entropy_function = RandomFunction
        self.functions = Functions
        self.local_history = {"dfunc_args": DetFuncArgs, "rfunc_args": RandFuncArgs}
        self.units = LayerSizeUnits
        for var in OtherVars:
            if hasattr(self, var):
                continue
            setattr(self, var, OtherVars[var])
        
    def __repr__(self):
        return "Chaos {} Layer [{}]: {} unit(s), {} function(s)".format(self.nature, self.name, self.units, len(self.functions))
    
    def ensure_in_range(self, index: typing.SupportsFloat):
        if hasattr(self, "interpreter_function"):
            return self.interpreter_function(index)
        return int((index * len(self.functions)) % len(self.functions))
    
    def call(self,
             inputs:tf.Tensor):
        """Computes input tensor.
        
        Calculates the deterministic function given the saved arguments.
        Current input and calculated entropy, is saved at `local_history`

        Args:
            inputs (tf.Tensor): input tensor to be computed.

        Returns:
            tf.Tensor: computed tensor.
        """
        self.local_history["current_input"] = input
        self.local_history["current_entropy"] = self.use_rand()
        index = self.ensure_in_range(self.deterministic_function(self, **self.local_history["dfunc_args"]))
        self.local_history["current_targetfunc"] = index
        return self.functions[index](self, inputs)
        
    def build(self,
              input_shape: typing.Iterable,
              WeightArgs: dict = {},
              BiasArgs: dict = {}):
        """Builds chaos layer.

        Args:
            input_shape (typing.Iterable): iterable shape of inputs.
            WeightArgs (dict, optional): Arguments to be passed on `self.add_weight()` for `self.w`, with exception to be shape. Defaults to {}.
            BiasArgs (dict, optional): Arguments to be passed on `self.add_weight()` for `self.b`, with exception of the shape. Defaults to {}.
        """
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            **WeightArgs
        )
        self.b = self.add_weight(
            shape=(self.units,),
            **BiasArgs
        )
        
    def use_rand(self, NewArgs = None):
        if NewArgs is not None:
            self.local_history["rfunc_args"] = NewArgs
        return self.entropy_function(**self.local_history["rfunc_args"]) if isinstance(self.local_history["rfunc_args"], dict) else self.entropy_function(self.local_history["rfunc_args"])
    
class UniversalCoreLayer(layers.Layer):
    """A Universal Variation of the Base Class of Chaotic Layers. Follows Keras Functional API.
    
    Applies a different function by virtue of deterministic function.
    
    This Base Class serves as fundamental building block of all Chaotic Layers."""
    def __init__(self,
                 Overwrite = False,
                 DeterministicFunction: typing.Callable = lambda s: 0,
                 RandomFunction: typing.Callable = lambda: random.random(),
                 LayerSizeUnits: int = 32,
                 SuperInitArgs: dict = {},
                 DetFuncArgs: dict = {},
                 RandFuncArgs: dict = {},
                 Functions: list = [lambda s, i: tf.reduce_sum(i) * s.b],
                 **OtherVars):
        """Creates a new Chaos Core Layer. A Chaos Layer can have any number functions that might be applied to input, by virtue of the Deterministic Function.

        Args:
            DeterministicFunction (`typing.Callable`, optional): Internally used deterministic function, return must be must be a numerical value. Defaults to `lambda:0`.
            RandomFunction (`typing.Callable`, optional): Internally used random function, for entropy, may be used externally by `self.use_rand()`. Defaults to `lambda:random.random()`.
            LayerSizeUnits (`int`, optional): Size or Number of Units in the Layer . Defaults to `32`.
            SuperInitArgs (`dict`, optional): Arguments to be passed to `super()` initializing  `tensorflow.keras.layers.Layer`. Defaults to `{}`.
            DetFuncArgs (`dict`, optional): Optional arguments to be passed to the deterministic function when called, arguments are recorded. Defaults to `{}`.
            RandFuncArgs (`dict`, optional): Optional arguments to be passed to the random function when called, arguments are recorded. Defaults to `{}`.
            Functions (`list`, optional): List of functions that might be applied, must be at least one (1) function. Defaults to `[lambda s, i: tf.reduce_sum(i)]`.
            OtherVars (`typing.Any`, optional): Arguments that will be stored in the object, may override existing stored. Defaults to `{}`.
            
        The result of the deterministic function maybe trivial, hence use the additional argument `interpreter_function=` to bind an interpreter.
        """
        if (len(Functions) < 1):
            raise ValueError("Function list must contain at least one (1) function.")
        super(CoreLayer, self).__init__(**SuperInitArgs)
        self.nature = "Core"
        self.deterministic_function = DeterministicFunction
        self.entropy_function = RandomFunction
        self.functions = Functions
        self.local_history = {"dfunc_args": DetFuncArgs, "rfunc_args": RandFuncArgs}
        self.units = LayerSizeUnits
        for var in OtherVars:
            if Overwrite:
                setattr(self, var, OtherVars[var])
        
    def __repr__(self):
        return "Chaos {} Layer [{}]: {} unit(s), {} function(s)".format(self.nature, self.name, self.units, len(self.functions))
    
    def ensure_in_range(self, index: typing.SupportsFloat):
        if hasattr(self, "interpreter_function"):
            return self.interpreter_function(index)
        return int((index * len(self.functions)) % len(self.functions))
    
    def call(self,
             inputs:tf.Tensor):
        """Computes input tensor.
        
        Calculates the deterministic function given the saved arguments.
        Current input and calculated entropy, is saved at `local_history`

        Args:
            inputs (tf.Tensor): input tensor to be computed.

        Returns:
            tf.Tensor: computed tensor.
        """
        self.local_history["current_input"] = input
        self.local_history["current_entropy"] = self.use_rand()
        index = self.ensure_in_range(self.deterministic_function(self, **self.local_history["dfunc_args"]))
        self.local_history["current_targetfunc"] = index
        return self.functions[index](self, inputs) if not isinstance(self.functions[index], layers.Layer) else self.functions[index](inputs)
        
    def build(self,
              input_shape: typing.Iterable,
              WeightArgs: dict = {},
              BiasArgs: dict = {}):
        """Builds chaos layer.

        Args:
            input_shape (typing.Iterable): iterable shape of inputs.
            WeightArgs (dict, optional): Arguments to be passed on `self.add_weight()` for `self.w`, with exception to be shape. Defaults to {}.
            BiasArgs (dict, optional): Arguments to be passed on `self.add_weight()` for `self.b`, with exception of the shape. Defaults to {}.
        """
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            **WeightArgs
        )
        self.b = self.add_weight(
            shape=(self.units,),
            **BiasArgs
        )
        
    def use_rand(self, NewArgs = None):
        if NewArgs is not None:
            self.local_history["rfunc_args"] = NewArgs
        return self.entropy_function(**self.local_history["rfunc_args"])if isinstance(self.local_history["rfunc_args"], dict) else self.entropy_function(self.local_history["rfunc_args"])
    
def _Core_test_standard_deterministic_function(TargetCallable: typing.Any):
    if not callable(TargetCallable):
        return False
    else:
        if (TargetCallable.__code__.co_argcount - (len(TargetCallable.__defaults__) if TargetCallable.__defaults__ is not None else 0)) > 2:
            return False
    return True
    
def _Core_test_standard_entropy_function(TargetCallable: typing.Any,
                                         Threshold: typing.SupportsFloat,
                                         TestCyle: int = 100,
                                         InitArgs = {},
                                         **EntropyArgs):
    if not callable(TargetCallable):
        return False
    bluf = TargetCallable(**InitArgs) if isclass(TargetCallable) else TargetCallable
    buffer = 0.0
    cycle = 0
    while cycle < TestCyle:
        buffer += abs(bluf(**EntropyArgs))
    return bool((buffer / (cycle + 1)) <= Threshold)

def _Core_test_standard_functionality(TargetCallable: typing.Any):
    pass
    