"""Module for ReferenceVariable class."""

from . import Variable
from ..private import _DotConstruct

# TODO: complete logics, set other Variable attributes, validate etc.


class ReferenceVariable(Variable):
    """
    reference to another entity or process taxon
    """

    type = None
    """required type of referred entity or taxon
    (will be adjusted by model.configure to point to composite class
    instead of mixin class)"""

    def __init__(self,
                 name,
                 desc,
                 *,
                 type=object,
                 **kwargs):
        super().__init__(name, desc, **kwargs)
        self.type = type

    def __getattr__(self, name):
        """return a _DotConstruct representing a variable of the referenced class"""
        if name == "__qualname__":  # needed to make sphinx happy
            return "DUMMY"  # FIXME!
        return _DotConstruct(self, []).__getattr__(name)

    # validation:

    def _check_valid(self, v):
        """check validity of candidate value"""

        if v is None:
            if self.allow_none is False:
                return False, str(self) + " may not be None"
        else:
            if self.type is not None:
                if not isinstance(v, self.type):
                    return False, \
                        str(self) + " must be instance of " + str(self.type)

        return super()._check_valid(v)

    def __str__(self):
        return (self.owning_class.__name__ + "." + self.codename) \
                if self.owning_class \
                else self.name + "(uid=" + self._uid + ")"

    def __repr__(self):
        if self.owning_class:
            return self.owning_class.__name__ + "." + self.codename
        r = "read-only " if self.readonly else ""
        r += "extensive " if self.is_extensive else ""
        r += "intensive " if self.is_intensive else ""
        r += "reference variable '" + self.name + "'"
        if self.desc not in ("", None):
            r += " (" + self.desc + ")"
        if self.ref is not None:
            r += ", ref=" + self.ref
        if self.CF is not None:
            r += ", CF=" + self.CF
        if self.AMIP is not None:
            r += ", AMIP=" + self.AMIP
        if self.IAMC is not None:
            r += ", IAMC=" + self.IAMC
        if self.CETS is not None:
            r += ", CETS=" + self.CETS
        if self.symbol not in ("", None):
            r += ", symbol=" + self.symbol
        if self.allow_none is False:
            r += ", not None"
        if self.scale not in ("", None):
            r += ", scale=" + self.scale
        if self.type is not None:
            r += ", type=" + str(self.datatype)
        return r # + " (uid=" + str(self._uid) + ")"
