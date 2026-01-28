# GLDF: Non-Homogeneous Causal Graph Discovery

This is the reference implementation of the framework described by [1]
for causal graph discovery on non-homogeneous data.

## Introduction

Algorithms for causal graph discovery (CD) on IID or stationary data are readily
available. However, real-world data rarely is IID or stationary.
This framework particularly aims to find ways in which a causal graph
*changes* over time, space or patterns in other extraneous information attached
to data. It does so by focusing on an approach that is local in the graph (gL)
and testing qualitative questions directly (D), hence the name gLD-framework
(GLDF).

Besides statistical efficiency, another major strength of this approach
is its modularity. This allows the present framework to directly
build on existing CD-algorithm implementations.
At the same time, almost arbitrary "patterns" generalizing persistence
in time or space can be leveraged to account for regime-structure.
The framework extensively builds on modified conditional independence tests
(CITs), where individual (or all) modifications can easily be customized
if the need arises.


## Getting Started

The best place to get started is probably the extensive [documentation]().
Further, there are tutorials in the form of jupyter-notebooks in the sub-directory "tutorials".
This package is designed to be easily extensible and to integrate well out-of-the box
with [tigramite](https://github.com/jakobrunge/tigramite).


## Requirements


Minimal:

*   python (version 3.10 or new recommended, tested on 3.13.5) and its standard-libary
*   numpy (version 2.0.0 or newer, tested on 2.3.2)
*   scipy (version 1.10.0 or newer, tested on 1.16.1)

Recommended (additionally):

*   matplotlib (version 3.7.0 or newer, tested on 3.10.5)
*   tigramite (version 5.2.0.0 or newer, tested on 5.2.8.2)
*   causal-learn (tested on 0.1.4.3)



## References

[1] M. Rabel, J. Runge.
   Context-Specific Causal Graph Discovery with Unobserved Contexts: Non-Stationarity, Regimes and Spatio-Temporal Patterns.
   *archive preprint* [arXiv:2511.21537](https://arxiv.org/abs/2511.21537), 2025.


## User Agreement

By downloading this software you agree with the following points:
This software is provided without any warranty or conditions of any kind. We assume no responsibility for errors or omissions in the results and interpretations following from application of this software.

You commit to cite above papers in your reports or publications.

   
## License

Copyright (C) 2025-2026 Martin Rabel

GNU General Public License v3.0

See the file LICENSE for full text.

This package is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version. This package is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.