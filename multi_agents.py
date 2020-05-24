import numpy as np
import abc
import util
from game import Agent, Action
import math


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board
        max_tile = successor_game_state.max_tile
        score = successor_game_state.score


        return score + 100*largest_tiles_in_upper_row(board) + count_merges(board)


def largest_tiles_in_upper_row(board):
    """
    checks whether the upper row has the largest tile in descending order
    :param largest_tiles: an array of the largest values in the game. must be descending order
    :param board: the current board
    :return: the weighted score
    """
    largest_values = np.unique(board)
    largest_values = largest_values[::-1]
    largest_values = largest_values[:3]

    rating = 0
    for i in range(len(largest_values)):
        if largest_values[i] == 0 or largest_values[i] == 2 or largest_values[i] == 4:
            return -1

        if board[0, i] >= largest_values[i]:
            rating += 1
        else:
            rating -= 2

    # todo we have a problem when the board has the maximum repeated several times in the upper row
    return rating

def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth
        self.current_depth = 1

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        """*** YOUR CODE HERE ***"""
        #util.raiseNotDefined()

        v = -math.inf
        max_action = Action.STOP
        for action in game_state.get_legal_actions(0):
            min_max_value = self.min_value(game_state.generate_successor(0, action), 1)

            if v < min_max_value:
                v = min_max_value
                max_action = action

        return max_action

    def cutoff_test(self, current_depth):
        if current_depth == self.depth:
            return True

        return False


    def max_value(self, game_state, current_depth):
        """ returns a utility value based on the evaluation function for the max """

        legal_moves = game_state.get_legal_actions(0)

        if self.cutoff_test(current_depth) or len(legal_moves) == 0:
            return self.evaluation_function(game_state)

        v = -math.inf
        current_depth += 1
        # print("current depth: " + str(current_depth))
        for action in legal_moves:
            successor = game_state.generate_successor(0, action)
            v = max(v, self.min_value(successor, current_depth))

        return v

    def min_value(self, game_state, current_depth):

        legal_moves = game_state.get_legal_actions(1)

        if self.cutoff_test(current_depth) or len(legal_moves) == 0:
            return self.evaluation_function(game_state)

        v = math.inf
        for action in legal_moves:
            successor = game_state.generate_successor(1, action)
            v = min(v, self.max_value(successor, current_depth))

        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """*** YOUR CODE HERE ***"""

        v = -math.inf
        max_action = Action.STOP

        for action in game_state.get_legal_actions(0):
            a_b_val = self.min_value(game_state.generate_successor(0, action), -math.inf, math.inf, 1)
            if v < a_b_val:
                v = a_b_val
                max_action = action

        # print(v)
        return max_action

    def cutoff_test(self, current_depth):
        return current_depth == self.depth


    def max_value(self, game_state, alpha, beta, current_depth):

        legal_moves = game_state.get_legal_actions(0)

        if self.cutoff_test(current_depth) or len(legal_moves) == 0:
            return self.evaluation_function(game_state)

        v = -math.inf
        current_depth += 1

        for action in legal_moves:
            successor = game_state.generate_successor(0, action)
            v = max(v, self.min_value(successor, alpha, beta, current_depth))

            if v >= beta:
                return v
            else:
                alpha = max(alpha, v)

        return v

    def min_value(self, game_state, alpha, beta, current_depth):

        legal_moves = game_state.get_legal_actions(1)

        if self.cutoff_test(current_depth) or len(legal_moves) == 0:
            return self.evaluation_function(game_state)

        v = math.inf

        for action in legal_moves:
            successor = game_state.generate_successor(1, action)
            v = min(v, self.max_value(successor, alpha, beta, current_depth))

            if v <= alpha:
                return v
            else:
                beta = min(beta, v)

        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """

        v = -math.inf
        max_action = Action.STOP

        for action in game_state.get_legal_actions(0):
            expecti_val = self.min_value(game_state.generate_successor(0, action), 1)

            if v < expecti_val:
                v = expecti_val
                max_action = action

        return max_action

    def max_value(self, game_state, current_depth):

        legal_moves = game_state.get_legal_actions(0)
        if self.cutoff_test(current_depth) or len(legal_moves) == 0:
            return self.evaluation_function(game_state)

        v = -math.inf
        current_depth += 1

        for action in legal_moves:
            successor = game_state.generate_successor(0, action)
            v = max(v, self.min_value(successor, current_depth))

        return v

    def cutoff_test(self, current_depth):
        return current_depth == self.depth

    def min_value(self, game_state, current_depth):

        legal_actions = game_state.get_legal_actions(1)

        if self.cutoff_test(current_depth) or len(legal_actions) == 0:

            return self.evaluation_function(game_state)

        expectation = 0
        for action in legal_actions:
            successor = game_state.generate_successor(1, action)
            expectation += self.max_value(successor, current_depth)

        expectation /= len(legal_actions)
        return expectation

def count_merges(board):
    """
    calculates the value of all the available merges on the board
    :param board: an array whose merges we want to count.
    :return: the value of the avialable merges
    """

    merge_val = 0

    # count the left/right merges
    for row in range(4):
        for col in range(3):
            if board[row, col] == board[row, col + 1] and board[row, col] !=0:
                merge_val += board[row, col] + board[row, col + 1]


    # count the number of up/down merges
    for col in range(4):
        for row in range(3):
            if board[row, col] == board[row + 1, col] and board[row, col] != 0:
                merge_val += board[row, col] + board[row + 1, col]


    return merge_val


def monotonicity(board):
    """
    checks if the rows are ordered monotonically, from rght to left or from left to right. Or if the columns are
    ordered monotonically up and down or down and up.
    :param board: the current state of the board
    :return: the maximum number of rows or columns which are monotonically ordered, but not both or -1 if nothing is
    monotonically ordered.
    """

    #row monotonicity

    left_right_monotinicity = 4
    for row in range(4):
        for col in range(3):
            # if board[row, col] != 0 and board[row, col + 1] != 0:
                if not board[row, col] >= board[row, col + 1]:
                    left_right_monotinicity -= 1
                    break

    right_left_monotonicity = 4
    for row in range(4):
        for col in range(3,0,-1):
            # if board[row, col] != 0 and board[row, col -1] != 0:
                if not board[row, col] >= board[row, col - 1]:
                    right_left_monotonicity -= 1
                    break

    # column monotonicity

    down_column_monotonicity = 4
    for col in range(4):
        for row in range(3):
            # if board[row, col] != 0 and board[row + 1, col] != 0:
                if not board[row, col] >= board[row + 1, col]:
                    down_column_monotonicity -= 1
                    break

    up_column_monotonicity = 4
    for col in range(4):
        for row in range(3, 0, -1):
            # if board[row, col] != 0 and board[row - 1, col] != 0:
                if not board[row, col] >= board[row - 1, col]:
                    up_column_monotonicity -= 1
                    break

    monotonic = max(up_column_monotonicity, down_column_monotonicity, right_left_monotonicity, left_right_monotinicity)

    if monotonic == 0:
        return -1

    return monotonic




def tile_placement_ranking(current_game_state):
    """
    checks to see if the largest value is in one of the corners.
    :param current_game_state: the instance of the game state we want to analyze
    :return: 1 if the largest value is in a corner or -1 if it is not.
    """
    max_tile = current_game_state.max_tile
    board = current_game_state.board

    placement_ranking = 0
    if board[0, 0] == max_tile:
        return 1
    elif board[0,3] == max_tile:
        return 1
    elif board[3,0] == max_tile:
        return 1
    elif board[3,3] == max_tile:
        return 1

    return -1



def smoothness_score(board):
    """
    Ranks a board by the sum of the absolute differences between neighboring tiles.
    :param board: the current game board
    :return: return a number indicating the ranking of the smoothness of the board
    """
    smoothness = 0
    # rank the rows
    for row in range(4):
        for col in range(3):
            if board[row, col] != 0 and board[row, col + 1] != 0 and board[row, col] != board[row, col + 1]:
                smoothness += abs(board[row, col] - board[row, col + 1])

    for col in range(4):
        for row in range(3):
            if board[row, col] != 0 and board[row + 1, col] != 0 and board[row, col] != board[row + 1, col]:
                smoothness += abs(board[row, col] - board[row + 1, col])

    return -smoothness

def center_of_the_board(board):
    """
    ranks a board based on whether or not particularly large values are being held in the center. A state which is
    usually fairly problematic.
    :param board: a 4*4 array out of whose center we want to keep the larger values
    :return: the total penalty for having particularly large values in the center or 0 if there are no violations of
    this policy.
    """
    center_coordinates = [(1,1), (1,2), (2,1), (2,2)]
    center_values = [board[index[0], index[1]] for index in center_coordinates]

    largest_values = np.unique(board)

    violation = 0
    for val in largest_values:
        if val > 200 and val in center_values:
            violation += val

    return -violation


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: the evaluation calculates a variety of properties and adds them together in a weighted sum.
    merge score - the value of all the merges on the board. This gives weight to boards with more merges, the higher
    value the possible merges the better.

    empty squares - the number of empty squares, this is important so the algorithm doesnt favor boards that are too
    full and thereby get stuck.

    monotonicity - how monotonic the rows are from left to right or right to left or how monotonic the columns are
    going up and down or down and up. This means the evaluation function favors boards that are monotonic,
    which is a good strategy to play by.

    Smoothness - smoothness is a measure of the difference between neighboring tiles. The less smooth, the greater
    the differences between neighboring tiles the more points are deducted from the board.

    center_score - placing high valued tiles in the center of the board seems to generally be a cause of problems,
    it seems best to keep high valued tiles near to the edges.

    placement_score - an excellent strategy for winnning the game is to keep the highest valued tile in a corner,
    this allows the monotonicity to be fairly consistent and forces a certain organization on the tiles and the
    possible merges. So if the largest value is in one of the corners there is a bonus and if not points are deducted.

    All this is added together with weights chosen by instinct and alot of tinkering.
    """

    board = current_game_state.board
    merge_score = count_merges(board)
    monotonic_score = monotonicity(board)
    empty_squares = np.count_nonzero(board == 0)
    game_score = current_game_state.score
    smooth_score = smoothness_score(board)
    center_score = center_of_the_board(board)
    placement_score = tile_placement_ranking(current_game_state)

    return smooth_score + 8*empty_squares + 6*merge_score + 6*center_score + game_score + 500*monotonic_score + \
           500*placement_score


# Abbreviation
better = better_evaluation_function
