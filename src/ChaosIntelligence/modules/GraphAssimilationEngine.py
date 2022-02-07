from copy import deepcopy
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
    
    def imbue(self,
              TargetLayer: layers.Layer,
              index: typing.SupportsInt,
              name: typing.AnyStr):
        pass
    pass

class GraphAssimilator():
    """This object takes record of inputs and results of a data model to create an assimilation model.
    
    
    """
    pass
