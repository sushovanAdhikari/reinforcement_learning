import neurolab as nl
import numpy as np
import pylab as pl


'''
author: sushovan.adhikari
1. _input_values represent the game states(i.e game environment, one specific setting in a game equals to one state)
       - 1 represent agent/player
       - 10 represent reward/cheese
       - 0 represent walls
       - (-10) represent pit

2. target_values represent the corresponding action for each state in _input_values, len(_input_values) = len(target_values)
       - these values are retrived by using the trained q-table(using q-learning), q-table has the higest reward move for each state.
       - by putting player at different position in the board, and retrieving the higest reward move from the q-table.
'''


input = [[-10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10],
              [-10, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 10],
              [-10, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 10],
              [-10, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 10],
              [-10, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 10],
              [-10, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 10],
              [-10, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 10],
              [-10, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 10],
              [-10, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 10],
              [-10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 10],
              [ 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -10],
              [ 10, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -10],
              [ 10, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -10],
              [ 10, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -10],
              [ 10, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -10],
              [ 10, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -10],
              [ 10, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -10],
              [ 10, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -10],
              [ 10, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -10],
              [ 10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -10]]

target = [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0],[0]]

net = nl.net.newp([
                    [-10, 10], [-10, 10], [-10, 10], [-10, 10], [-10, 10], [-10, 10], [-10, 10], 
                    [-10, 10], [-10, 10], [-10, 10],[-10, 10], [-10, 10]],
                    1)

error = net.train(input, target, epochs=100, show=1, lr=0.01)
pl.plot(error)
pl.xlabel('Epochs')
pl.ylabel('Train error')
pl.grid()
pl.show()