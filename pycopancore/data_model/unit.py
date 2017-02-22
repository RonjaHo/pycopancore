# TODO: doc strings

from functools import reduce
import operator

from .dimension import Dimension

class Unit():
    
    is_base = None
    """whether this is a base unit"""
    
    name = None
    """full name"""
    
    symbol = None
    """symbol"""
    
    desc = None
    """description"""
    
    factor = None
    """scalar factor in front of product of powers of base Units"""
    
    exponents = None
    """dict of base Unit: nonzero exponent"""
    
    @property
    def dimension(self):
        """corresponding Dimension"""
        if self.is_base: 
            return self._dimension
        else:
            nondim = Dimension(name="non-dimensional", desc="non-dimensional", exponents={}, default_unit = Unit(name="unity", symbol="", desc="number of unity", exponents={}))
#            nondim.default_unit = 
            return reduce(operator.mul, [unit.dimension**ex for unit, ex in self.exponents.items()], nondim)

    def __init__(self, is_base=True, name=None, symbol=None, desc=None, factor=1, exponents=None, dimension=None):
        assert factor > 0, "factor must be positive"
        self.factor = factor
        self.is_base = is_base
        if is_base:
            self.name = name
            self.symbol = symbol
            self.desc = name if desc is None else desc
            # don't use self as key before name has been assigned since name is used as hash:
            self.exponents = { self: 1 }
            if dimension is not None: 
                assert dimension.is_base, "dimension of base unit must be base dimension"
            self._dimension = dimension
        else:
            self.exponents = exponents
            # TODO: use words "per", "square", "cubic" and sort be descending exponents
            self.name = (str(self.factor) + " " if self.factor != 1 else "") \
                        + " ".join([unit.name + ("" if ex == 1 else "^" + str(ex) if ex >= 0 else "^(" + str(ex) + ")") 
                                    for unit, ex in exponents.items()]) if name is None else name
            self.symbol = (str(self.factor) + " " if self.factor != 1 else "") \
                        + " ".join([unit.symbol + ("" if ex == 1 else "^" + str(ex) if ex >= 0 else "^(" + str(ex) + ")") 
                                    for unit, ex in exponents.items()]) if symbol is None else symbol
            self.desc = "\n\n".join([unit.name + ": " + unit.desc for unit in exponents.keys()]) if desc is None else desc
            assert dimension is None, "dimension of non-base unit is derived automatically"

    def named(self, name, symbol=None, desc=None):
        return Unit(is_base=self.is_base, name=name, 
                    symbol=self.symbol if symbol is None else symbol, 
                    desc=self.desc if desc is None else desc, 
                    factor=self.factor, exponents=self.exponents)

    def __repr__(self):
        return self.name + " [" + self.symbol + "]"
    
    def __hash__(self):
        return hash(self.name) if self.is_base else None
    
    def __eq__(self, other):
        if self.is_base: 
            return other.is_base and other.dimension == self.dimension and other.name == self.name
        else:
            return self.factor == other.factor and self.exponents == other.exponents

    def __pow__(self, power):
        """exponentiation **"""
        return Unit(is_base = False, 
                    factor = self.factor**power,
                    exponents = { unit: ex*power for unit, ex in self.exponents.items() })
        
    def __mul__(self, other):
        """multiplication *"""
        pex = self.exponents.copy()
        if type(other) == Unit:
            oex = other.exponents
            for unit, ex in oex.items():
                if unit in pex:
                    pex[unit] += ex
                    if pex[unit] == 0: pex.pop(dim)
                else:
                    pex[unit] = ex
            return Unit(is_base = False, factor = self.factor * other.factor, exponents = pex)
        else:
            return Unit(is_base = False, factor = self.factor * other, exponents = pex)
        
    def __truediv__(self, other):
        """division /"""
        qex = self.exponents.copy()
        if type(other) == Unit:
            oex = other.exponents
            for dim, ex in oex.items():
                if dim in qex:
                    qex[dim] -= ex
                    if qex[dim] == 0: qex.pop(dim)
                else:
                    qex[dim] = -ex
            return Unit(is_base = False, factor = self.factor / other.factor, exponents = qex)
        else:
            return Unit(is_base = False, factor = self.factor / other, exponents = qex)

    def __rmul__(self, other):
        return self * other
    
    def __rtruediv__(self, other):
        return self**(-1) * other
        