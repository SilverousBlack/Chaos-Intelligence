from copy import deepcopy
import matplotlib.pyplot
import numpy
import tensorflow as tf
from keras import layers, models
import typing

class GraphAssimilator():
    pass

def __new_call__(self, inputs, *args, **kwargs):
    if "input" not in self.assimilator.keys():
        self.assimilator["input"] = [inputs.numpy()] if self.assimilator["tracker"]._savemode == "collect" else inputs.numpy()
    else:
        if self.assimilator["tracker"]._savemode == "collect":
            self.assimilator["input"].append(inputs.numpy())
        else:
            self.assimilator["input"] = inputs.numpy()
    if "other" not in self.assimilator.keys():
        self.assimilator["other"] = [(args, kwargs)] if self.assimilator["tracker"]._savemode == "collect" else (args, kwargs)
    else:
        if self.assimilator["tracker"]._savemode == "collect":
            self.assimilator["other"].append((args, kwargs))
        else:
            self.assimilator["other"] = (args, kwargs)
    output = self._old_call(inputs, *args, **kwargs)
    if "output" not in self.assimilator.keys():
        self.assimilator["output"] = [output.numpy()] if self.assimilator["tracker"]._savemode == "collect" else output
    else:
        if self.assimilator["tracker"]._savemode == "collect":
            self.assimilator["output"].append(output.numpy())
        else:
            self.assimilator["output"] = output
    return output

class AssimilatorNode():
    def __init__(self,
                 TargetInstance: GraphAssimilator,
                 SaveMode: typing.AnyStr = "collect"):
        if SaveMode.lower() not in ["collect", "single"]:
            raise ValueError("Save modes are only \"collect\" and \"single\". Mode \"{}\" is not supported. Non-case-sensitive.".format(SaveMode))
        self.tracked_object = TargetInstance
        self.bound_object = None
        self._savemode = SaveMode
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
        setattr(self.bound_object, "call", self.bound_object._old_call)
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
        return (self.bound_object.name,
                self._location,
                {"input": inp, "output": outp})

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
              TargetModel: models.Model,
              SaveMode: typing.AnyStr):
        if self.bound_object is not None:
            self.revert()
        self.bound_object = TargetModel
        for layer in TargetModel.layers:
            node = AssimilatorNode(self, SaveMode)
            if not isinstance(layer, layers.InputLayer):
                node.imbue(layer)
                self.layers.append(layer)
                self.assimilators.append(node)
            else:
                continue
    
    def retrieve(self):
        self.raw_data_map = []
        for assimilator in self.assimilators:
            self.raw_data_map.append(assimilator.retrieve())
        return self.raw_data_map
    
    def add_interpreter(self,
                        InterpreterFunction: typing.Callable):
        if hasattr(self, "interpreter"):
            del self.interpreter
        self.interpreter = InterpreterFunction
        
    def remove_interpreter(self):
        if hasattr(self, "interpreter"):
            del self.interpreter
        
    def get_input_map(self,
                      raw=False):
        if not (hasattr(self, "raw_data_map") and raw):
            raise RuntimeError("No recorded raw data.")
        if not (hasattr(self, "procesed_data_map") and not raw):
            raise RuntimeError("No recorded processed data.")
        return self.raw_data_map if raw else self.processed_data_map
        
    def process_map(self):
        if not hasattr(self, "raw_data_map"):
            self.retrieve()
        self._layer_names = []
        self._input_data = []
        self._output_data = []
        self.processed_data_map = []
        for name, inp, outp in self.raw_data_map:
            self._layer_names.append(name)
            self._input_data.append(self.interpreter(inp) if hasattr(self, "interpreter") else inp)
            self._output_data.append(self.interpreter(outp) if hasattr(self, "interpreter") else outp)
            self.processed_data_map.append((self.interpreter(inp) if hasattr(self, "interpreter") else inp, self.interpreter(outp) if hasattr(self, "interpreter") else outp))
        return self.processed_data_map
    
    def revert(self):
        for assimilator in self.assimilators:
            assimilator.revert()
        self.assimilators = []
        self.layers = []
        self.bound_object = None
        
class GraphPlotter():
    pass
