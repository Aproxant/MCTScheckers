import random
import copy
import math

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
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = 0

    def add_child(self, child_state):
        child_node = Node(child_state, self)
        self.children.append(child_node)

    def update(self, score):
        self.visits += 1
        self.score += score

    def fully_expanded(self):
        return len(self.children) == len(self.state.get_possible_moves())

    def select_child(self, exploration_constant):
        max_score = float("-inf")
        selected_child = None
        for child in self.children:
            if child.visits == 0:
                return child
            if child.state.current_player == 1:
                score = -1 * child.score
            else:
                score = child.score
            exploit_score = score / child.visits
            explore_score = math.sqrt(2 * math.log(self.visits) / child.visits)
            score = exploit_score + exploration_constant * explore_score
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

    def blind_legal_moves(self, x, y):
        if self.board[x][y].val != 0:
            if self.board[x][y].val == 1 and self.board[x][y].king == False:
                blind_legal_moves = [(x - 1, y - 1), (x + 1, y - 1)]
            elif self.board[x][y].val == -1 and self.board[x][y].king == False:
                blind_legal_moves = [(x - 1, y + 1), (x + 1, y + 1)]
            else:
                blind_legal_moves = [(x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1), (x + 1, y + 1)]
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
                    legal_moves = [((move[0] + (move[0] - x), move[1] + (move[1] - y)), [hit])]
                    while next_hop:
                        next_hop = False
                        for i in self.blind_legal_moves(legal_moves[-1][0][0], legal_moves[-1][0][1]):
                            if i[0] == hit[0] and i[1] == hit[1]:
                                continue
                            if (
                                self.board[i[0]][i[1]].val != 0
                                and self.board[i[0]][i[1]].val
                                != self.board[legal_moves[-1][0][0]][legal_moves[-1][0][1]].val
                                and self.on_board(
                                    (i[0] + (i[0] - legal_moves[-1][0][0]), i[1] + (i[1] - legal_moves[-1][0][1]))
                                )
                                and self.board[i[0] + (i[0] - legal_moves[-1][0])][
                                    i[1] + (i[1] - legal_moves[-1][1])
                                ].val
                                == 0
                            ):
                                hit = (i[0], i[1])
                                legal_moves.append(
                                    (
                                        (i[0] + (i[0] - legal_moves[-1][0][0])),
                                        i[1] + (i[1] - legal_moves[-1][0][1]),
                                        [legal_moves[-1][1], hit],
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
        end_x, end_y = end

        # self.printBoard()
        # print()
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

        end_row, end_col = end

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

        evaluation += position_scores[end_row][end_col]

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


class MCTS:
    def __init__(self, exploration_constant=1.4, simulation_count=1000):
        self.exploration_constant = exploration_constant
        self.simulation_count = simulation_count

    def search(self, state):
        root = Node(state)

        for _ in range(self.simulation_count):
            node = self.selection(root)

            score = self.simulation(copy.deepcopy(node.state))
            self.backpropagate(node, score)

        best_child = root.children[0]
        for child in root.children:
            child.state.printBoard()
            print()
            if child.visits > best_child.visits:
                best_child = child

        return best_child.state

    def selection(self, node):
        while not node.state.check_for_endgame():
            if not node.fully_expanded():
                self.expand(node)
                return node.select_child(self.exploration_constant)
            else:
                node = node.select_child(self.exploration_constant)
            # node.state.printBoard()
            # print()
        return node

    def expand(self, node):
        possible_moves = node.state.get_possible_moves()
        for move in possible_moves:
            new_state = copy.deepcopy(node.state)
            new_state.make_move(move)
            # new_state.printBoard()
            node.add_child(new_state)
        # random_child = random.choice(node.children)
        # return random_child

    def simulation(self, state):
        while not state.check_for_endgame():
            possible_moves = state.get_possible_moves()
            prioritized_moves = []

            for move in possible_moves:
                prioritized_moves.append(state.evaluate_move(move))

            move = random.choices(possible_moves, weights=prioritized_moves, k=1)[0]

            state.make_move(move)

        if state.current_player == -1:
            return 1
        else:
            return -1

    def backpropagate(self, node, score):
        while node is not None:
            node.update(score)
            node = node.parent


"""
possible_moves = state.get_possible_moves()
            prioritized_moves = sorted(
                possible_moves, key=lambda move: state.evaluate_move(move), reverse=True
            )

            max_piece_count = float('-inf')
            best_move = None
            for move in prioritized_moves:
                temp_state = state.copy()
                temp_state.make_move(move)
                piece_count = temp_state.get_piece_count(self.current_player)
                if piece_count > max_piece_count:
                    max_piece_count = piece_count
                    best_move = move

            if best_move is not None:
                state.make_move(best_move)
            else:
                # If no move maintains a higher piece count, select a random move
                move = random.choice(possible_moves)
                state.make_move(move)
"""