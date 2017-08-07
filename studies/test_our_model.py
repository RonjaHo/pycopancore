"""Skript to run Anderies carbon cycle."""

from time import time
from numpy import random
import numpy as np
import pycopancore.models.our_model as M
# import pycopancore.models.only_copan_global_like_carbon_cycle as M
from pycopancore import master_data_model as D
from pycopancore.runners import Runner

# import plotly.plotly as py
import plotly.offline as py
import plotly.graph_objs as go
from pylab import plot, gca, show



# first thing: set seed so that each execution must return same thing:
random.seed(1)

# parameters:

model = M.Model()

# instantiate process taxa:
nature = M.Nature()
# metabolism = M.Metabolism()

# generate entities and plug them together at random:
world = M.World(nature=nature, #metabolism=metabolism,
                  atmospheric_carbon = 0.2 * D.gigatonnes_carbon, #relative values
                  ocean_carbon = 0.6 * D.gigatonnes_carbon
                  )
society = M.Society(world=world)
cell = M.Cell(society=society)

# set initial values # only one cell so far
Sigma0 = 1.5e8 * D.square_kilometers
cell.land_area = Sigma0
# print(M.Cell.land_area.get_values(cells))

L0 = 0.2 * D.gigatonnes_carbon # 2480 is yr 2000
cell.terrestrial_carbon = L0
# print(M.Cell.terrestrial_carbon.get_values(cells))

G0 = 0.5 * D.gigatonnes_carbon   # 1125 is yr 2000
cell.fossil_carbon = G0
# print(M.Cell.fossil_carbon.get_values(cells))

# r = random.uniform(size=nsocs)
# P0 = 6e9 * D.people * r / sum(r)  # 500e9 is middle ages, 6e9 would be yr 2000
# M.Society.population.set_values(societies, P0)
# print(M.Society.population.get_values(societies))

# r = random.uniform(size=nsocs)
# in AWS paper: 1e12 (alternatively: 1e13):
# S0 = 1e13 * D.gigajoules * r / sum(r)
# M.Society.renewable_energy_knowledge.set_values(societies, S0)
# print(M.Society.renewable_energy_knowledge.get_values(societies))

# r = random.uniform(size=nsocs)
# K0 = sum(P0) * 1e4 * D.dollars/D.people * r / sum(r)  # ?
# M.Society.physical_capital.set_values(societies, K0)
# print(M.Society.physical_capital.get_values(societies))

# TODO: add noise to parameters

# from pycopancore.private._expressions import eval
# import pycopancore.model_components.base.interface as B
# import sympy as sp

runner = Runner(model=model)

start = time()
traj = runner.run(t_1=100, dt=.1)
print(time()-start, " seconds")

t = np.array(traj['t'])
print("max. time step", (t[1:]-t[:-1]).max())

data_ca = go.Scatter(
    x=t,
    y=traj[M.World.atmospheric_carbon][world],
    mode="lines",
    name="atmospheric carbon",
    line=dict(
        color="lightblue",
        width=4
    )
)
data_ct = go.Scatter(
    x=t,
    y=traj[M.World.terrestrial_carbon][world],
    mode="lines",
    name="terrestrial carbon",
    line=dict(
        color="green",
        width=4
    )
)
data_cm = go.Scatter(
    x=t,
    y=traj[M.World.ocean_carbon][world],
    mode="lines",
    name="maritime carbon",
    line=dict(
        color="blue",
        width=4
    )
)
data_cf = go.Scatter(
    x=t,
    y=traj[M.World.fossil_carbon][world],
    mode="lines",
    name="fossil carbon",
    line=dict(
        color="gray",
        width=4
    )
)
layout = dict(title = 'Our model (Anderies Carbon Cycle)',
              xaxis = dict(title = 'time [yr]'),
              yaxis = dict(title = 'Carbon [GtC]'),
              )

fig = dict(data=[data_ca, data_ct, data_cm, data_cf], layout=layout)
py.plot(fig, filename="our-model-result.html")
