Installation
============

From Source
-----------
Building from source requires:
 - `CMake <https://cmake.org/>`_ >= 3.15,
 - A `C++17 compiler <https://en.cppreference.com/w/cpp/compiler_support>`_,
 - `Python <https://www.python.org/>`_ and `NumPy <https://numpy.org/>`_ for the Python package,
 - A `SCIP <https://www.scipopt.org/>`_ installation.

.. code-block:: bash

   cmake -B build/
   cmake --build build/ --parallel
   python -m pip install build/python

Both the ``build/`` folder and ``ecole`` folder can safely be removed aftwerwards.

.. warning::

   SCIP is linked dynamically so it must remained installed when using Ecole.
