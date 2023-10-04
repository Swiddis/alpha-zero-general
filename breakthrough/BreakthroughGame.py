from Game import Game
import numpy as np

class BreakthroughGame(Game):
    def __init__(self, n = 8):
        assert n >= 4
        assert n % 2 == 0 # Restricted by action implementation
        self.n = n
    
    def getInitBoard(self):
        """
        Sticking with the conventions of Tic Tac Toe/Connect 4,
        We use an 8x8 board with 1 for white, -1 for black, and 0 for empty
        
        Home row for white is index n-1, so we see from white's perspective if
        we print the board with normal formatting.
        """
        board = np.zeros((self.n, self.n), dtype=np.int8)
        for row in (0, 1):
            board[row] -= 1
        for row in (self.n - 1, self.n - 2):
            board[row] += 1
        return board
        
    def getBoardSize(self):
        return (self.n, self.n)
    
    def getActionSize(self):
        """
        We separate the three kinds of pawn moves between forward and diagonals,
        and then count their destination squares.
        
        ```
        Forward   Right     Left
        1 1 1 1   0 1 1 1   1 1 1 0
        1 1 1 1   0 1 1 1   1 1 1 0
        1 1 1 1   0 1 1 1   1 1 1 0
        0 0 0 0   0 0 0 0   0 0 0 0
        n^2-n     n^2-2n+1  n^2-2n+1
        ```

        Adding them together we get: `3n^2 - 5n + 2 = (3n - 2)(n - 1)`

        Theoretically we could remove redundant final-row actions,
        but let's keep it simple.
        """
        return (3 * self.n - 2) * (self.n - 1)

    def getNextState(self, board, player, action):
        """
        Given the encoding from getActionSize, we still need to number individual moves.
        
        To make getSymmetries easy to implement, we want the numbering to encode horizontal symmetry.
        Let's join the forward/right/left boards from above into a 3x1 mega-board of size `(n-1)(3n-2)`:
        ```
        0 1 1 1 | 1 1 1 1 | 1 1 1 0
        0 1 1 1 | 1 1 1 1 | 1 1 1 0
        0 1 1 1 | 1 1 1 1 | 1 1 1 0
        0 0 0 0 | 0 0 0 0 | 0 0 0 0
        ```

        And number them such that each action is the horizontal inverse of `action_size - action - 1`,
        preserving contiguous regions of numbers with the move type:
        ```
        Region 1   R2      R3      R4
        00 01 02 | 09 10 : 19 20 | 27 28 29
        03 04 05 | 11 12 : 17 18 | 24 25 26
        06 07 08 | 13 14 : 15 16 | 21 22 23
        ```
        """
        board = np.copy(board)
        n = self.n # Saves some typing
        # Abuse symmetry on actions
        # TODO handle odd-sized boards
        left_act = min(action, (n - 1) * (3 * n - 2) - action - 1)

        if left_act < (n - 1) * (n - 1): # Region 1
            dest_row, src_col = (left_act // (n - 1), left_act % (n - 1))
            src_row, dest_col = (dest_row + 1, src_col + 1)
        else: # Region 2
            la_zeroed = left_act - (n - 1) * (n - 1)
            dest_row, dest_col = (la_zeroed // (n // 2), la_zeroed % (n // 2))
            src_row, src_col = (dest_row + 1, dest_col)

        if left_act < action: # Mirror horizontally
            src_col, dest_col = (n - src_col - 1, n - dest_col - 1)
        if player < 0: # Mirror vertically
            src_row, dest_row = (n - src_row - 1, n - dest_row - 1)
        
        board[src_row][src_col] = 0
        board[dest_row][dest_col] = player
        return (board, -player)

    def getValidMoves(self, board, player):
        """
        Reusing the encoding from getValidMoves, there are four regions we care about.
        We let numpy compute their validity in batches and flatten at the end.
        """
        n = self.n # Saves more typing
        board = self.getCanonicalForm(board, player) # Actions are also canonically symmetric
        
        # Region 1
        moves_right = np.logical_and(
            board[1:,:-1] > 0,
            board[:-1,1:] <= 0
        )
        # Region 2
        moves_forward1 = np.logical_and(
            board[1:,:n//2] > 0,
            board[:-1,:n//2] == 0
        )
        # Region 3
        moves_forward2 = np.logical_and(
            np.flipud(board[1:,n//2:]) > 0,
            np.flipud(board[:-1,n//2:]) == 0,
        )
        # Region 4
        moves_left = np.logical_and(
            np.flipud(board[1:,1:]) > 0,
            np.flipud(board[:-1,:-1]) <= 0
        )

        return np.concatenate((
            moves_right.flatten(),
            moves_forward1.flatten(),
            moves_forward2.flatten(),
            moves_left.flatten(),
        ), dtype=np.int8)

    def getGameEnded(self, board, player):
        w_win, b_win = 1 in board[0], -1 in board[self.n - 1]
        if w_win:
            return player
        if b_win:
            return -player
        return 0

    def getCanonicalForm(self, board, player):
        if player > 0:
            return board
        # We need to both flip colors and mirror the board horizontally
        return np.flipud(board * player)

    def getSymmetries(self, board, pi):
        # Due to our action encoding, we can safely flip board and pi left-right
        return [(board, pi), (np.fliplr(board), np.flip(pi))]

    def stringRepresentation(self, board):
        return ''.join(('.', 'W', 'B')[p] for p in board.flat)
    
    @staticmethod
    def display(board, output=True):
        result = ('\n' + '-'*(3*8+8) + '\n').join([
            ' ' + ' | '.join([f'{("_", "W", "B")[p]}' for p in row])
            for row in board
        ])
        if output:
            print(result)
        return result
