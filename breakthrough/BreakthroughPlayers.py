import numpy as np
from breakthrough.BreakthroughGame import BreakthroughGame

def parse_movestr(movestr: str) -> ((int, int), (int, int)):
    return (
        7 - (ord(movestr[1]) - ord('1')), ord(movestr[0]) - ord('a'),
        7 - (ord(movestr[3]) - ord('1')), ord(movestr[2]) - ord('a')
    )

CRED = '\033[91m'
CEND = '\033[0m'


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanBreakthroughPlayer():
    def __init__(self, game):
        self.game = game
        self.prev = None

    def play(self, board):
        game = BreakthroughGame(8)
        if self.prev is None:
            self.prev = BreakthroughGame.display(game.getInitBoard(), output=False)
        new = BreakthroughGame.display(board, output=False)
        print(''.join(b if a == b else CRED + b + CEND for a, b in zip(self.prev, new)))
        while True:
            input_move = input("> ")
            parsed = parse_movestr(input_move)
            # Too lazy to figure out how to convert to action #s
            for action in range(game.getActionSize()):
                (try_next, _) = game.getNextState(board, 1, action)
                if (try_next[parsed[0]][parsed[1]] != board[parsed[0]][parsed[1]] and 
                    try_next[parsed[2]][parsed[3]] != board[parsed[2]][parsed[3]]):
                    self.prev = BreakthroughGame.display(try_next, output=False)
                    return action
