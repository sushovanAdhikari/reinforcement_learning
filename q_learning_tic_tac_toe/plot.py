import matplotlib.pyplot as plt
def plot_win_count(game_win_count):
        players = list(game_win_count.keys())
        wins = list(game_win_count.values())

        plt.bar(players, wins, color=['blue', 'green', 'orange'])
        plt.xlabel('Players')
        plt.ylabel('Wins')
        plt.title('Game Win Count')
        plt.show()