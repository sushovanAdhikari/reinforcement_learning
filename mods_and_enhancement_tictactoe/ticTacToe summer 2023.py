# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 13:46:15 2023

Tic-Tac-Toe

I lost the version from earlier this summer
in a computer crash.

@author: patrick
"""

import random

rowWins = [[1, 1, 1, 0, 0, 0, 0, 0, 0], 
           [0, 0, 0, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1,]]

colWins = [[1, 0, 0, 1, 0, 0, 1, 0, 0],
           [0, 1, 0, 0, 1, 0, 0, 1, 0],
           [0, 0, 1, 0, 0, 1, 0, 0, 1]]

diagWins = [[1, 0, 0, 0, 1, 0, 0, 0, 1],
           [0, 0, 1, 0, 1, 0, 1, 0, 0]]

wins = rowWins+colWins+diagWins

def showBoard(board):
    count = 0
    print()
    print("Current board state:")
    print("-------------")
    
    for c in board:
        
        if (c == '#'):
            print('|', ' ', end=' ')
        else:
            print('|', c, end=' ')
            
        if ((count == 2) or (count == 5) or (count == 8)):
            print('|')
            
        count = count+1
        
    print("-------------")
            
        
        
def validateInput(spot, board):
    valid = True
    
    # If spot is in range, make sure the 
    # position is not occupied.
    if ((spot >= 0) and (spot <= 8)):
        if (board[spot] == '#'):
            pass
        else:
            print("Position is occupied.")
            valid = False
    else:
        print("Value entered not in range 0 to 8.")
        valid = False
    
    return valid



def didIWin(turn, board, wins):
    iWon = False
    
    for row in wins:
        count = 0
        matchCount = 0
        # Look to see if we have a possible win.
        for value in row:
            if ((value == 1) and (board[count] == turn)):
                matchCount = matchCount + 1
            count = count + 1
        
        # If there were 3 in row, there is a win. 
        if (matchCount == 3):
            iWon = True
    
    return iWon

def canIWin(turn, board):
    canWin, winSpot = canIWinRow(turn, board, rowWins)
    if (canWin == False):
        canWin, winSpot = canIWinCol(turn, board, colWins)
    if (canWin == False):
        canWin, winSpot = canIWinDiag(turn, board, diagWins)
        
    return canWin, winSpot

def canIBlock(turn, board):
    if (turn == 'x'):
        yourTurn = 'o'
    else:
        yourTurn = 'x'
        
    canBlock, blockSpot = canIWinRow(yourTurn, board, rowWins)
    
    if (canBlock == False):
        canBlock, blockSpot = canIWinCol(yourTurn, board, colWins)
    if (canBlock == False):
        canBlock, blockSpot = canIWinDiag(yourTurn, board, diagWins)
        
    return canBlock, blockSpot
    


def canIWinRow(turn, board, rWins):
    canWin = False
    winSpot = -1
    
    for row in rWins:
        count = 0
        matchCount = 0
        isGood = False
        for value in row:
            if ((value == 1) and (board[count] == '#')):
                winSpot = count
                isGood = True
            elif ((value == 1) and (board[count] == turn)):
                matchCount = matchCount + 1
            
            count = count + 1
                
        if ((matchCount == 2)and(isGood == True)):
            canWin = True
            break
        
    return canWin, winSpot

def canIWinCol(turn, board, cWins):
    canWin = False
    winSpot = -1
    for row in cWins:
        count = 0
        matchCount = 0
        isGood = False
        for value in row:
            if ((value == 1) and (board[count] == '#')):
                winSpot = count
                isGood = True
            elif ((value == 1) and (board[count] == turn)):
                matchCount = matchCount + 1
            
            count = count + 1
                
        if ((matchCount == 2)and(isGood == True)):
            canWin = True
            break
        
    return canWin, winSpot

def canIWinDiag(turn, board, dWins):
    canWin = False
    winSpot = -1
    for row in dWins:
        count = 0
        matchCount = 0
        isGood = False
        for value in row:
            if ((value == 1) and (board[count] == '#')):
                winSpot = count
                isGood = True
            elif ((value == 1) and (board[count] == turn)):
                matchCount = matchCount + 1
            
            count = count + 1
                
        if ((matchCount == 2)and(isGood == True)):
            canWin = True
            break
        
    return canWin, winSpot



def ticMe():
    print("Hello TV Land!")
    print("Welcome to Tic-Tac-Toe!")
    print()
    print("X goes first.")
    
    # Initialize the playing board.
    board = ['#', '#', '#', '#', '#', '#', '#', '#', '#' ]
    
    turn = 'x'
    count = 0
    playing  = True
    while(playing):
        print("It is ", turn, "'s turn.")
        print("Enter a position from 0 to 8.  Make sure the ")
        print("spot is not occupied.")
        
        # Get valid input.
        validInput = False
        while(validInput == False):
            spot = int(input())
            validInput = validateInput(spot, board)
        
        print("You entered ", spot)
        
        # Update playing board.
        board[spot] = turn
        
        # Detect a win.
        iWon = didIWin(turn, board, wins)
        
        # If a win is detected tell the user and end the game.
        if (iWon == True):
            print(turn, "Wins!")
            playing = False
        else:
            # Otherwise, change turns, and take of book keeping.
            # Change turns.
            if (turn == 'x'):
                turn = 'o'
            else:
                turn = 'x'
            
            # Increment count.
            count = count + 1
            
            # If no one wins, cat game, game over.
            if (count == 9):
                playing = False;
                print("The cat wins!")
        
        showBoard(board)
    return


def getOpenSpots(board):
    openSpots = []
    j = 0
    for myChar in board:
        if (myChar == '#'):
            openSpots.append(j)
        j = j + 1
        
    return openSpots


def ticMeCPUvsPerson(playRandom):
    print("Hello TV Land!")
    print("Welcome to Tic-Tac-Toe!")
    print()
    print("X goes first.")
    
    gettingInput = True
    while(gettingInput):
        print("Do you want to go first, 1 for yes, 0 for no.")
        myAnswer = int(input())
        if (myAnswer == 0):
            playersTurn = False
            gettingInput = False
        elif (myAnswer == 1):
            playersTurn = True
            gettingInput = False
        else:
            print("Input not in range")
            
    # Initialize the playing board.
    board = ['#', '#', '#', '#', '#', '#', '#', '#', '#' ]
    
    turn = 'x'
    count = 0
    playing  = True
    while(playing):
        print("It is ", turn, "'s turn.")
        if (playersTurn == True) :
            print("Enter a position from 0 to 8.  Make sure the ")
            print("spot is not occupied.")
            
            # Get valid input.
            validInput = False
            while(validInput == False):
                spot = int(input())
                validInput = validateInput(spot, board)
            
            print("You entered ", spot)
            playersTurn = False
        else:
            # Computers turn.
            print("computer's turn.")
            if (playRandom == True):
                # Get open positions.
                openSpots = getOpenSpots(board)
                # Select a positon.
                numSpots = len(openSpots)
                rIndex = random.randint(0, numSpots - 1)
                spot = openSpots[rIndex]
            else:
                print("turn = ", turn)
                
                # Check for a winning position.
                iCanWin, winSpot = canIWin(turn, board)
        
                if (iCanWin == True):
                    # Go with the winning spot if possible.
                    print("I see a winning move. winSpot = ", winSpot)
                    spot = winSpot
                else:
                    # Check to see computer can block a win.
                    
                    iCanBlock, blockSpot = canIBlock(turn, board)
                    if (iCanBlock == True):
                        print("I see a blocking move. block spot = ", blockSpot)
                        spot = blockSpot
                    else:
                        # Go with random spot.
                        print("I will choose a random position.")
                        # Get open positions.
                        openSpots = getOpenSpots(board)
                        # Select a positon.
                        numSpots = len(openSpots)
                        rIndex = random.randint(0, numSpots - 1)
                        spot = openSpots[rIndex]
                    
            # Set turn to players turn.
            playersTurn = True
            
        # Update playing board.
        board[spot] = turn
        
        # Detect a win.
        iWon = didIWin(turn, board, wins)
        
        # If a win is detected tell the user and end the game.
        if (iWon == True):
            print(turn, "Wins!")
            playing = False
        else:
            # Otherwise, change turns, and take of book keeping.
            # Change turns.
            if (turn == 'x'):
                turn = 'o'
            else:
                turn = 'x'
            
            # Increment count.
            count = count + 1
            
            # If no one wins, cat game, game over.
            if (count == 9):
                playing = False;
                print("The cat wins!")

        showBoard(board)
    return

    

# ticMe()
ticMeCPUvsPerson(False)
