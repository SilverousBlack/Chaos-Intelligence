from copy import deepcopy
import matplotlib.pyplot
import numpy
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import typing

class GraphAssimilator():
    pass

def new_call(self, **kwargs):
    self.assimilator["other"] = kwargs
    output = self._old_call(**kwargs)
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
        TargetLayer._old_call = TargetLayer.call
        TargetLayer.assimilator["tracker"] = self
        boundmethod = new_call.__get__(TargetLayer, TargetLayer.__class__)
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
            node.imbue(layer)
            self.layers.append(layer)
            self.assimilators.append(node)
    
    def retrieve(self):
        self.map = []
        for assimilator in self.assimilators:
            self.map.append(assimilator.retrieve())
        return self.map
    
    def revert(self):
        pass
    
ass = GraphAssimilator()
inputs = tf.keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
ass.build(model)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data (these are NumPy arrays)
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
print("Fit model on training data")
history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=2,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
)
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(x_test[:3])
print("predictions shape:", predictions.shape)
ass.retrieve()
model
