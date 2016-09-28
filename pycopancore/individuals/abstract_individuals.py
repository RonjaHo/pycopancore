# This file is part of pycopancore.
#
# Copyright (C) 2016 by COPAN team at Potsdam Institute for Climate
# Impact Research
#
# URL: <http://www.pik-potsdam.de/copan/software>
# License: MIT license
# das habe ich veraendert


"""
Encapsulates states and dynamics of social micro agents.
"""

#
#  Imports
#


#
#  Define class MicroAgents
#

class Individuals(object):
    """
    Encapsulates states and dynamics of social micro agents.
    """

    #
    #  Definitions of internal methods
    #

    def __init__(self,
                 individual_identifier,
                 group_identifier,
                 cell_identifier,
                 ):
        """
        Initialize an instance of MicroAgents.
        The object_identifier variables are numbers to identify each object.
        The behavour_dict includes information about behaviour
        The connection_dict includes the identifers of connected individuals
        """

        self.individual_identifier = individual_identifier
        self.group_identifier = None
        self.cell_identifier = None

    def __str__(self):
        """
        Return a string representation of the object of class individual
        """
        return ('Individual with identifier % s, \
                group % s, \
                cell % s'
                ) % (
                self.individual_identifier,
                self.group_identifier,
                self.cell_identifier)

    def set_cell_identifier(self, cell_identifier):
        """
        A function to set the location or cell of an individual
        """
        self.cell_identifier = cell_identifier

    def set_group_identifier(self, group_identifier):
        """
        A function to set the group membership of an individual
        """
        self.group_identifier = group_identifier

    #
    #  Definitions of further methods
    #
