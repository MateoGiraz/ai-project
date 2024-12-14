from agent import Agent
from board import Board
import numpy as np

class MiniMaxAgent(Agent):
    def __init__(self, player=1, max_depth=3):
        super().__init__(player)
        self.max_depth = max_depth
        self.opponent = 2 if player == 1 else 1

    def heuristic_utility(self, board):
        """
        Heurística personalizada para evaluar el estado del tablero.
        """
        proximity_to_center = self.evaluate_proximity_to_center(board)
        musketeer_mobility = self.evaluate_musketeer_mobility(board)
        enemy_mobility = self.evaluate_enemy_mobility(board)
        musketeer_distance = self.evaluate_musketeer_distance(board)
        enemy_proximity = self.evaluate_enemy_proximity_to_musketeers(board)
        alignment_penalty = self.evaluate_alignment_penalty(board)
        trap_penalty = self.evaluate_trap_penalty(board)
        
        return (
            #2 * proximity_to_center + 
            #1.5 * musketeer_mobility - 
            #1 * enemy_mobility - 
            #1.5 * musketeer_distance + 
            #2 * enemy_proximity - 
            5 * alignment_penalty - 
            10 * trap_penalty
        )

    def evaluate_proximity_to_center(self, board):
        """
        Evalúa qué tan cerca están los mosqueteros del centro del tablero.
        """
        center = (board.board_size[0] // 2, board.board_size[1] // 2)
        musketeer_positions = board.find_musketeer_positions()
        proximity = -sum(
            abs(pos[0] - center[0]) + abs(pos[1] - center[1])
            for pos in musketeer_positions
        )
        return proximity

    def evaluate_musketeer_mobility(self, board):
        """
        Evalúa la cantidad de movimientos válidos para los mosqueteros.
        """
        return len(board.get_musketeer_valid_movements())

    def evaluate_enemy_mobility(self, board):
        """
        Evalúa la cantidad de movimientos válidos para los enemigos.
        """
        return len(board.get_enemy_valid_movements())

    def evaluate_musketeer_distance(self, board):
        """
        Evalúa la distancia promedio entre los mosqueteros.
        """
        positions = board.find_musketeer_positions()
        total_distance = 0
        count = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                total_distance += abs(positions[i][0] - positions[j][0]) + abs(positions[i][1] - positions[j][1])
                count += 1
        return total_distance / (count if count > 0 else 1)

    def evaluate_enemy_proximity_to_musketeers(self, board):
        """
        Evalúa qué tan cerca están los enemigos de los mosqueteros.
        """
        musketeer_positions = board.find_musketeer_positions()
        enemy_positions = board.find_enemy_positions()
        proximity = 0
        for m_pos in musketeer_positions:
            for e_pos in enemy_positions:
                proximity += 1 / (abs(m_pos[0] - e_pos[0]) + abs(m_pos[1] - e_pos[1]) + 1)
        return proximity

    def evaluate_alignment_penalty(self, board):
        """
        Penaliza si los mosqueteros están alineados en una fila o columna.
        """
        musketeer_positions = board.find_musketeer_positions()
        rows = [pos[0] for pos in musketeer_positions]
        cols = [pos[1] for pos in musketeer_positions]

        if len(set(rows)) == 1 or len(set(cols)) == 1:
            return 1  # Penalización máxima si están alineados
        return 0

    def evaluate_trap_penalty(self, board):
        """
        Penaliza si un mosquetero pisa una trampa.
        """
        trap_position = board.find_trap_position()
        musketeer_positions = board.find_musketeer_positions()
        
        if trap_position and trap_position in musketeer_positions:
            return 1  # Penalización fuerte por pisar la trampa
        return 0

    def next_action(self, obs):
        """
        Devuelve la mejor acción posible según el algoritmo Minimax con poda alfa-beta.
        """
        _, best_action = self.minimax(obs, self.max_depth, self.player, float('-inf'), float('inf'))
        return best_action

    def minimax(self, board, depth, current_player, alpha, beta):
        """
        Algoritmo Minimax con poda alfa-beta.
        """
        is_end, winner = board.is_end(current_player)
        if is_end:
            return (100 if winner == self.player else -100), None
        
        if depth == 0:
            return self.heuristic_utility(board), None

        actions = board.get_possible_actions(current_player)
        best_action = None

        if current_player == self.player:  # Maximizar
            max_eval = float('-inf')
            for action in actions:
                board_copy = board.clone()
                board_copy.play(current_player, action)
                eval, _ = self.minimax(board_copy, depth - 1, self.opponent, alpha, beta)
                if eval > max_eval:
                    max_eval = eval
                    best_action = action
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_action
        else:  # Minimizar
            min_eval = float('inf')
            for action in actions:
                board_copy = board.clone()
                board_copy.play(current_player, action)
                eval, _ = self.minimax(board_copy, depth - 1, self.player, alpha, beta)
                if eval < min_eval:
                    min_eval = eval
                    best_action = action
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_action
