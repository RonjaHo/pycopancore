"""Cell entity type mixing class template.

TODO: adjust or fill in code and documentation wherever marked by "TODO:",
then remove these instructions
"""
# This file is part of pycopancore.
#
# Copyright (C) 2017 by COPAN team at Potsdam Institute for Climate
# Impact Research
#
# URL: <http://www.pik-potsdam.de/copan/software>
# License: MIT license

from .. import interface as I
from pycopancore import Event
import numpy as np

# from .... import master_data_model as D


class Cell (I.Cell):
    """Cell entity type mixin implementation class."""

    # standard methods:

    def __init__(self,
                 *,
                 eating_stock=100,
                 **kwargs):
        """Initialize an instance of Cell."""
        super().__init__(**kwargs)
        self.eating_stock = eating_stock


    # process-related methods:

    def snow_white_arrival(self):
        """Calculate snow white's arrival."""
        return np.random.exponential(18.)

    def snow_white_eating(self, unused_t):
        """Party hard."""
        self.eating_stock = self.eating_stock / 2.

    processes = [
        Event("snow_white",
              [I.Cell.eating_stock],
              ["time", snow_white_arrival, snow_white_eating]
              )
    ]
