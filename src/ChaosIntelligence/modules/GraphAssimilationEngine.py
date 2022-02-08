from copy import deepcopy
import weakref
import matplotlib.pyplot
import numpy
import tensorflow as tf
from tensorflow.keras import layers, models
import typing

class GraphAssimilator():
    pass
class AssimilatorNode():
    def __init__(self, TargetInstance: GraphAssimilator):
        self.tracked_object = TargetInstance
        self.bound_object = None
    
    def imbue(self,
              TargetLayer: layers.Layer,
              index: typing.SupportsInt,
              name: typing.AnyStr):
        self.bound_object = TargetLayer
        while len(self.tracked_object.layers) <= index:
            self.tracked_object.layers.append({})
        setattr(TargetLayer, "assimilator", {})
        TargetLayer.assimilator["original"] = TargetLayer.call
        TargetLayer.assimilator["tracker"] = self
        def new_call(slf, inputs, *args, **kwargs):
            slf.assimilator["input"] = inputs
            slf.assimilator["other"] = args
            slf.assimilator["delta"]
            output = slf.assimilator["original"](inputs, *args, **kwargs)
            slf.assimilator["output"] = output
            return output
        setattr(TargetLayer, "call", new_call)
        self.tracked_object.layers[index] = {"name": name, "layer": TargetLayer}
    
    def revert(self):
        setattr(self.bound_object, "call", self.bound_object.assimilator["original"])
        delattr(self.bound_object, "assimilator")
        
    def retrieve(self):
        if self.bound_object is None:
            raise RuntimeError("")


class GraphAssimilator():
    """This object takes record of inputs and results of a data model to create an assimilation model.
    
    
    """
    def __init__(self):
        self.layers = []
