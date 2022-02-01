# Chaos Intelligence Core Module

> This module defines core classes and functionalities of Chaos Intelligence.

**Table of Contents**
- [Chaos Intelligence Core Module](#chaos-intelligence-core-module)
  - [Summaries](#summaries)
    - [Imports](#imports)
    - [Objects](#objects)
    - [Functions](#functions)
  - [Concepts](#concepts)
    - [The *Core* Layer](#the-core-layer)
  - [Details](#details)
    - [`CoreLayer`](#corelayer)

## Summaries

### Imports

|       Module        | Specific Imports |
| :-----------------: | :--------------- |
|       `copy`        | `deepcopy`       |
|      `inspect`      | `isclass`        |
|      `random`       | --               |
| `tensorflow` (`tf`) | --               |
|         ^^          | `keras.layers`   |
|      `typing`       | --               |

### Objects

|                          Object                          | Description Summary                                                      |
| :------------------------------------------------------: | ------------------------------------------------------------------------ |
|        `CoreLayer` (`tensorflow.keras.layers.Layer`)         | Base Class of Chaotic Layers                                             |
| `CoreLayerNoBlindOverride` (`tensorflow.keras.layers.Layer`) | Base Class Variation of Chaotic Layers with Internal Overriding Disabled |
|    `UniversalCoreLayer` (`tensorflow.keras.layers.Layer`)    | Base Class Variation of Chaotic Layers with Optional Override Toggle     |

### Functions

|                  Function                  | Description                                                                              |
| :----------------------------------------: | ---------------------------------------------------------------------------------------- |
| *`_Core_test_standard_deterministic_function`* | Tests whether a callable object complies to deterministic function standards             |
|    *`_Core_test_standard_entropy_function`*    | Tests whether a callable object complies to entropy function standards                   |
|     *`_Core_test_standard_functionality`*      | Tests whether a callable object complies to chaos functionality standards                |
|      *`_Core_test_qualified_chaos_core`*       | Tests whether a name (class or object) is a qualified chaos core                         |
|               `MakeCoreLayer`                | Makes a new Chaos Core Layer variation from a given tensorflow.keras.layers.Layer object |

## Concepts

### The *Core* Layer

> A layer that is able to respond to a deterministic function's result



## Details

### `CoreLayer`
