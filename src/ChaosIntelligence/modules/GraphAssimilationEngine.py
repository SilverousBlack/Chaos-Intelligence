from copy import deepcopy
import matplotlib.pyplot
import numpy
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import typing

class GraphAssimilator():
    pass

def __new_call__(self, inputs, *args, **kwargs):
    self.assimilator["input"] = inputs
    self.assimilator["other"] = (args, kwargs)
    output = self._old_call(inputs, *args, **kwargs)
    self.assimilator["output"] = output
    return output

class AssimilatorNode():
    def __init__(self, TargetInstance: GraphAssimilator):
        self.tracked_object = TargetInstance
        self.bound_object = None
        self._location = 0
    
    def imbue(self,
              TargetLayer: layers.Layer):
        if self.bound_object is not None:
            self.revert()
        self.bound_object = TargetLayer
        setattr(TargetLayer, "assimilator", {})
        boundmethod = TargetLayer.call.__get__(TargetLayer, TargetLayer.__class__)
        setattr(TargetLayer, "_old_call", boundmethod)
        TargetLayer.assimilator["tracker"] = self
        boundmethod = __new_call__.__get__(TargetLayer, TargetLayer.__class__)
        setattr(TargetLayer, "call", boundmethod)
        self._location = len(self.tracked_object.assimilators)
    
    def revert(self):
        if self.bound_object is None:
            raise RuntimeError("No bound object to revert")
        setattr(self.bound_object, "call", self.bound_object.assimilator["original"])
        delattr(self.bound_object, "assimilator")
        
    def retrieve(self):
        if self.bound_object is None:
            raise RuntimeError("No bound object to retrive data from.")
        inp: None
        outp: None
        if "input" in self.bound_object.assimilator.keys():
            inp = self.bound_object.assimilator["input"]
        else:
            raise RuntimeError("No input recorded to retrieve")
        if "output" in self.bound_object.assimilator.keys():
            outp = self.bound_object.assimilator["output"]
        else:
            raise RuntimeError("No output recorded to retrieve")
        return (self.bound_object.name, {"input": inp,
                                         "output": outp})

    def copy_imbue(self,
                   TargetLayer: layers.Layer,
                   index: typing.SupportsInt,
                   name: typing.AnyStr):
        internal = deepcopy(TargetLayer)
        self.imbue(internal, index, name)
        return internal

class GraphAssimilator():
    """This object takes record of inputs and results of a data model to create an assimilation model.
    
    
    """
    def __init__(self):
        self.layers = []
        self.assimilators = []
        self.bound_object = None
        
    def build(self,
              TargetModel: models.Model):
        if self.bound_object is not None:
            self.revert()
        self.bound_object = TargetModel
        for layer in TargetModel.layers:
            node = AssimilatorNode(self)
            if not isinstance(layer, layers.InputLayer):
                node.imbue(layer)
                self.layers.append(layer)
                self.assimilators.append(node)
            else:
                continue
    
    def retrieve(self):
        self.map = []
        for assimilator in self.assimilators:
            self.map.append(assimilator.retrieve())
        return self.map
    
    def revert(self):
        pass
