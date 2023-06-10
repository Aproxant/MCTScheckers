import random
import copy
import math

from collections import defaultdict

WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GOLD = (255, 215, 0)
HIGH = (160, 190, 255)

NORTHWEST = "northwest"
NORTHEAST = "northeast"
SOUTHWEST = "southwest"
SOUTHEAST = "southeast"


class Node:
    def __init__(self, state, parent=None, action=((0, 0), (0, 0), [])):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.score = 0

    def add_child(self, child_state, action):
        child_node = Node(child_state, self, action)
        self.children.append(child_node)

    def update(self, score):
        self.visits += 1
        self.score += score

    def fully_expanded(self):
        return len(self.children) == len(self.state.get_possible_moves())

    def select_child(self, exploration_constant, action_values, action_counts, player):
        max_score = float("-inf")
        selected_child = None

        def UCT():
            exploit_score = child.score / child.visits
            explore_score = math.sqrt(2 * math.log(self.visits) / child.visits)
            return exploit_score + exploration_constant * explore_score

        def UCB1_bias():
            str_action = str(child.action)
            W = 2
            bias_score = (action_values[str_action] / action_counts[str_action]) * W / (child.visits - child.score + 1)
            return UCT() + bias_score

        def UCB1_tuned():
            exploit_score = child.score / child.visits
            str_action = str(child.action)
            mean = action_values[str_action] / action_counts[str_action]
            variance = mean * (1 - mean)

            explore_score_var = math.sqrt(
                math.log(self.visits)
                / child.visits
                * min(1 / 4, variance + math.sqrt(2 * math.log(self.visits) / child.visits))
            )
            return exploit_score + exploration_constant * explore_score_var

        for child in self.children:
            if child.visits == 0:
                return child

            if player == 1:  # try to make blue better
                score = UCT()
            else:
                score = UCT()
            if score > max_score:
                max_score = score
                selected_child = child
        return selected_child


class Piece:
    def __init__(self, val, king=False):
        self.val = val
        self.king = king


class State:
    def __init__(self, game):
        self.board = [[None] * 8 for i in range(8)]
        for i in range(8):
            for j in range(8):
                if game.board.matrix[i][j].occupant != None:
                    if game.board.matrix[i][j].occupant.color == RED:
                        self.board[i][j] = Piece(-1)
                    else:
                        self.board[i][j] = Piece(1)

                    if game.board.matrix[i][j].occupant.king:
                        self.board[i][j].king = True
                else:
                    self.board[i][j] = Piece(0)

        if game.turn == RED:
            self.current_player = -1
        else:
            self.current_player = 1

    def get_possible_moves(self):
        moves = []
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if self.board[i][j].val == self.current_player:
                    for x in self.legal_moves(i, j):
                        if len(x[1]) != 0:
                            return [((i, j), x[0], x[1])]
                        moves.append(((i, j), x[0], x[1]))
        return moves

    def blind_legal_moves(self, x, y, hit=0):
        if self.board[x][y].val != 0 or hit == 1 or hit == -1:
            if (self.board[x][y].val == 1 or hit == 1) and self.board[x][y].king == False:
                blind_legal_moves = [(x - 1, y - 1, 0), (x + 1, y - 1, 0)]
            elif (self.board[x][y].val == -1 or hit == -1) and self.board[x][y].king == False:
                blind_legal_moves = [(x - 1, y + 1, 0), (x + 1, y + 1, 0)]
            else:
                blind_legal_moves = [(x - 1, y - 1, 1), (x + 1, y - 1, 1), (x - 1, y + 1, 1), (x + 1, y + 1, 1)]
        else:
            blind_legal_moves = []

        return blind_legal_moves

    def on_board(self, move):
        x = move[0]
        y = move[1]
        if x < 0 or y < 0 or x > 7 or y > 7:
            return False
        else:
            return True

    def legal_moves(self, x, y):
        blind_legal_moves = self.blind_legal_moves(x, y)
        legal_moves = []
        next_hop = True
        for move in blind_legal_moves:
            if self.on_board(move):
                if self.board[move[0]][move[1]].val == 0:
                    legal_moves.append((move, []))
                elif (
                    self.board[move[0]][move[1]].val != self.board[x][y].val
                    and self.on_board((move[0] + (move[0] - x), move[1] + (move[1] - y)))
                    and self.board[move[0] + (move[0] - x)][move[1] + (move[1] - y)].val == 0
                ):  # is this location filled by an enemy piece?
                    hit = (move[0], move[1])
                    legal_moves = [((move[0] + (move[0] - x), move[1] + (move[1] - y), move[2]), [hit])]
                    while next_hop:
                        next_hop = False
                        for i in self.blind_legal_moves(
                            legal_moves[-1][0][0], legal_moves[-1][0][1], self.board[x][y].val
                        ):
                            if self.on_board(i):
                                if i[0] == legal_moves[-1][1][-1][0] and i[1] == legal_moves[-1][1][-1][1]:
                                    continue
                                if (
                                    self.board[i[0]][i[1]].val != 0
                                    and self.board[i[0]][i[1]].val != self.board[x][y].val
                                    and self.on_board(
                                        (i[0] + (i[0] - legal_moves[-1][0][0]), i[1] + (i[1] - legal_moves[-1][0][1]))
                                    )
                                    and self.board[i[0] + (i[0] - legal_moves[-1][0][0])][
                                        i[1] + (i[1] - legal_moves[-1][0][1])
                                    ].val
                                    == 0
                                ):
                                    hit = (i[0], i[1])
                                    prevHit = copy.deepcopy(legal_moves[-1][1])
                                    prevHit.append(i)
                                    legal_moves.append(
                                        (
                                            (
                                                i[0] + (i[0] - legal_moves[-1][0][0]),
                                                i[1] + (i[1] - legal_moves[-1][0][1]),
                                                i[2],
                                            ),
                                            prevHit,
                                        )
                                    )
                                    next_hop = True

                    legal_moves = [legal_moves[-1]]

                    break

        return legal_moves

    def remove_piece(self, x, y):
        self.board[x][y].val = 0
        self.board[x][y].king = False

    def make_move(self, move):
        start, end = move[0], move[1]
        x, y = start
        end_x, end_y, isKing = end

        if len(move[2]) == 0:
            self.board[end_x][end_y].val = self.board[x][y].val
            self.board[end_x][end_y].king = self.board[x][y].king
            self.remove_piece(x, y)
        else:
            for i in move[2]:
                self.remove_piece(i[0], i[1])

            self.board[end_x][end_y].val = self.board[x][y].val
            self.board[end_x][end_y].king = self.board[x][y].king
            self.remove_piece(x, y)

        self.king(end_x, end_y)

        self.current_player = -self.current_player

    def printBoard(self):
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if self.board[j][i].val == -1:
                    print(2, end=" ")
                else:
                    print(self.board[j][i].val, end=" ")
            print()

    def getWinner(self):
        red = 0
        blue = 0
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if self.board[j][i].val == -1:
                    red += 1
                elif self.board[j][i].val == 1:
                    blue += 1

        if red > blue:
            return -1
        else:
            return 1

    def king(self, x, y):
        if self.board[x][y].val != 0:
            if (self.board[x][y].val == 1 and y == 0) or (self.board[x][y].val == -1 and y == 7):
                self.board[x][y].king = True

    def is_terminal(self):
        if len(self.get_possible_moves()) == 0:
            return True
        return False

    def check_for_endgame(self):
        for x in range(8):
            for y in range(8):
                if self.board[x][y].val != 0 and self.board[x][y].val == self.current_player:
                    if self.legal_moves(x, y) != []:
                        return False
        return True

    def evaluate_move(self, move):
        start, end = move[0], move[1]

        end_row, end_col, isKing = end

        evaluation = 0

        # Piece count

        piece_count = 0
        for x in range(8):
            for y in range(8):
                if self.board[x][y].val == self.current_player:
                    piece_count += 1

        # evaluation += piece_count

        # Evaluate king count
        """
        king_count = 0
        for x in range(8):
            for y in range(8):
                if self.board[x][y].val == self.current_player and self.board[x][y].king:
                    king_count += 1
        evaluation += king_count * 1.5
        """
        # Evaluate positional advantage
        position_scores = [
            [4, 0, 4, 0, 4, 0, 4, 0],
            [0, 3, 0, 3, 0, 3, 0, 4],
            [4, 0, 2, 0, 2, 0, 3, 0],
            [0, 2, 0, 1, 0, 2, 0, 2],
            [2, 0, 1, 0, 1, 0, 2, 0],
            [0, 2, 0, 2, 0, 1, 0, 2],
            [4, 0, 3, 0, 2, 0, 3, 0],
            [0, 4, 0, 4, 0, 4, 0, 4],
        ]

        # evaluation += position_scores[end_row][end_col]

        # Evaluate distance-based factor
        nearest_opponent_distance = float("inf")
        for x in range(8):
            for y in range(8):
                if self.board[x][y].val == -self.current_player:
                    distance = abs(x - end_row) + abs(y - end_col)
                    if distance < nearest_opponent_distance:
                        nearest_opponent_distance = distance

        distance_factor = 1 / (nearest_opponent_distance + 1)
        time_factor = 1 / piece_count  # Adjust the denominator to control scaling speed
        scaled_distance_factor = distance_factor * time_factor
        evaluation += scaled_distance_factor  # * 2

        # Evaluate distance-based factor from king

        if self.current_player == -1:
            distance_from_king_row = 7 - end_col
        else:
            distance_from_king_row = end_col

        time_factor = piece_count / 24  # Adjust the denominator to control scaling speed
        scaled_distance_factor = distance_from_king_row * time_factor
        evaluation += scaled_distance_factor  # * 2

        return evaluation

    def evaluate_state(self):
        position_scores = [
            [4, 0, 4, 0, 4, 0, 4, 0],
            [0, 3, 0, 3, 0, 3, 0, 4],
            [4, 0, 2, 0, 2, 0, 3, 0],
            [0, 2, 0, 1, 0, 2, 0, 2],
            [2, 0, 1, 0, 1, 0, 2, 0],
            [0, 2, 0, 2, 0, 1, 0, 2],
            [4, 0, 3, 0, 2, 0, 3, 0],
            [0, 4, 0, 4, 0, 4, 0, 4],
        ]
        evaluation = 0

        piece_count = [0, 0]  # Index 0 for player 1, index 1 for player 2
        king_count = [0, 0]
        position_score = [0, 0]
        advancement_score = [0, 0]

        for x in range(8):
            for y in range(8):
                piece = self.board[x][y]
                if piece.val != 0:  # If there is a piece on this square
                    player_index = 0 if piece.val == 1 else 1
                    piece_count[player_index] += 1
                    if piece.king:
                        king_count[player_index] += 1
                    else:
                        advancement_score[player_index] += x if player_index == 0 else 7 - x
                    position_score[player_index] += position_scores[x][y]

        # Compute scores for each player
        player_scores = [0, 0]
        for i in range(2):
            # Adjust piece count score
            piece_count_score = piece_count[i]  # Scale if necessary

            # Adjust king count score
            king_count_score = king_count[i] * 1.5  # Scale if necessary

            # Adjust positional score
            fposition_score = position_score[i] * 2  # Scale if necessary

            # Add all score components to the player's score
            player_scores[i] = piece_count_score + king_count_score + fposition_score + advancement_score[i]

        # Subtract the opponent's score from the current player's score
        evaluation = (
            player_scores[0] - player_scores[1] if self.current_player == 1 else player_scores[1] - player_scores[0]
        )

        return evaluation


class MCTS:
    def __init__(self, exploration_constant=0.4, simulation_count=1000):
        self.exploration_constant = exploration_constant
        self.simulation_count = simulation_count
        self.state_history = {}
        self.action_values = defaultdict(int)
        self.action_counts = defaultdict(int)
        self.action_variance = defaultdict(float)
        self.action_values_var = defaultdict(float)

    def search(self, state):
        root = Node(state)
        pla = root.state.current_player
        for i in range(self.simulation_count):
            node = self.selection(root, pla)
            score = self.simulation(copy.deepcopy(node.state), pla)
            self.backpropagate(node, score)
            # print(i)

        if len(root.children) == 0:
            return 0
        best_child = root.children[0]
        for child in root.children:
            if child.visits > best_child.visits:
                best_child = child

        return best_child.state, best_child.action

    def selection(self, node, pla):
        while not node.state.check_for_endgame():
            if not node.fully_expanded():
                self.expand(node)
                return node.select_child(self.exploration_constant, self.action_values, self.action_counts, pla)
            else:
                node = node.select_child(self.exploration_constant, self.action_values, self.action_counts, pla)
            # node.state.printBoard()
            # print()
        return node

    def expand(self, node):
        possible_moves = node.state.get_possible_moves()
        for move in possible_moves:
            new_state = copy.deepcopy(node.state)
            new_state.make_move(move)
            # new_state.printBoard()
            node.add_child(new_state, move)
        # random_child = random.choice(node.children)
        # return random_child

    def simulation(self, state, pla):
        actions_taken = []
        while not state.check_for_endgame():
            possible_moves = state.get_possible_moves()
            prioritized_moves = []

            for move in possible_moves:
                prioritized_moves.append(state.evaluate_move(move))

            move = random.choices(possible_moves, weights=prioritized_moves, k=1)[0]
            actions_taken.append(move)
            state.make_move(move)

        if pla == state.getWinner():
            return 1
        else:
            return 0

    def backpropagate(self, node, score):
        while node is not None:
            node.update(score)
            if node.action is not None:
                str_action = str(node.action)
                self.action_values[str_action] += score
                self.action_counts[str_action] += 1
            node = node.parent


def alphabeta(state, depth, alpha, beta, maximizing_player):
    if depth == 0 or state.is_terminal():
        return state.evaluate_state()

    if maximizing_player:
        max_eval = float("-inf")
        legal_moves = state.get_possible_moves()
        for move in legal_moves:
            child_state = copy.deepcopy(state)
            child_state.make_move(move)
            eval = alphabeta(child_state, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cutoff
        return max_eval
    else:
        min_eval = float("inf")
        legal_moves = state.get_possible_moves()
        for move in legal_moves:
            child_state = copy.deepcopy(state)
            child_state.make_move(move)
            eval = alphabeta(child_state, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval


def find_best_move(state, depth):
    best_move = None
    max_eval = float("-inf")
    legal_moves = state.get_possible_moves()

    for move in legal_moves:
        child_state = copy.deepcopy(state)
        child_state.make_move(move)
        eval = alphabeta(child_state, depth - 1, float("-inf"), float("inf"), False)
        if eval > max_eval:
            max_eval = eval
            print(max_eval)
            best_move = move
    state.make_move(best_move)
    return state
