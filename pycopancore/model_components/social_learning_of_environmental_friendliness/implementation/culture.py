"""Culture process taxon mixing class template.
"""

# This file is part of pycopancore.
#
# Copyright (C) 2017 by COPAN team at Potsdam Institute for Climate
# Impact Research
#
# URL: <http://www.pik-potsdam.de/copan/software>
# License: MIT license

from .... import Event
from .. import interface as I
from ...base import interface as B
from numpy.random import exponential, uniform


class Culture (I.Culture):
    """Culture process taxon mixin implementation class."""

    # process-related methods:

    def next_learning_time(self, t):
        """time of next awareness update"""
        res = t + exponential(1. / self.environmental_friendliness_learning_rate)
        return res

    def let_individuals_learn(self, t):
        """let some individuals learn environmental (un)friendliness"""
        print("terr.:",sum([c.terrestrial_carbon for w in self.worlds for c in w.cells]))
        for w in self.worlds:
            for i in w.individuals:
                if uniform() < self.environmental_friendliness_learning_fraction:
                    i.learn_environmental_friendliness(t)

    processes = [
                 Event("learn environmental (un)friendliness",
                       [B.Culture.worlds.individuals.is_environmentally_friendly],
                       ["time",
                        next_learning_time,
                        let_individuals_learn])                 
                 ]
