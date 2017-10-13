"""_abstract_dynamics_mixin class.

It sets the basic structure of dynamic mixins (culture, metabolism, nature).
"""
# This file is part of pycopancore.
#
# Copyright (C) 2016 by COPAN team at Potsdam Institute for Climate
# Impact Research
#
# URL: <http://www.pik-potsdam.de/copan/software>
# License: MIT license

from ..data_model import variable
from ..data_model import OrderedSet


class _AbstractProcessTaxonMixin(object):
    """Define Entity-unspecific abstract class.

    From this class all entity-specific abstract mixin classes are derived.
    """

    processes = []
    """All processes of this taxon"""
    model = None  # I can see why we need this, but I don't see it in use yet!
    """Model containing this taxon"""
    instances = None
    """List containing the unique instance of this taxon"""

    def __init__(self):
        """Initialize an _AbstractProcessTaxonMixin instance."""
        if self.__class__.instances:
            self.__class__.instances.append(self)
            print(self.__class__, 'Process Taxon is already initialized!')
        else:
            self.__class__.instances = [self]

    def delete(self):
        """Delete this Process Taxon from lists."""
        # Remove from list, if list is existent:
        if (self in self.__class__.instances
                and self.__class__.instances):
            self.__class__.instances.remove(self)
        # If list then has lenght == 0, set it to None again, so everything is
        # fresh again...
        if (len(self.__class__.instances) == 0
                and self.__class__.instances):
            self.__class__.instances = None
        # Delete for good:
        del(self)

    # the repr and the str methods were removed in the master/prototype_jobst1
    # Do we really don't want them anymore?
    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return repr(self)

    def set_value(self, var, value):
        """Dummy docstring"""
        # TODO: missing method docstring
        assert isinstance(var, variable.Variable), \
            "variable must be a Variable object"
        var.set_value(self, value)

    def assert_valid(self):
        """Make sure all variable values are valid.

        By calling assert_valid for all Variables

        """
        for v in self.variables:
            try:
                val = v.get_value(self)
            except:
                return
            v.assert_valid(val)
