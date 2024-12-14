from agent import Agent
from board import Board
import numpy as np

class ExpectiMaxAgent(Agent):
    def __init__(self, player=1, 
                 max_depth=3, 
                 alignment_weight=1,
                 trap_weight=1, 
                 moves_weight=1):
        super().__init__(player)
        self.max_depth = max_depth
        self.opponent = 2

        # Pesos para las heurísticas
        self.alignment_weight = alignment_weight
        self.trap_weight = trap_weight
        self.moves_weight = moves_weight

    def heuristic_utility(self, board):
        """
        Heurística que evalúa el estado del tablero.
        """
        alignment_penalty = self.evaluate_alignment_penalty(board)
        trap_penalty = self.evaluate_trap_penalty(board)
        moves_score = self.evaluate_moves_score(board)

        return (
            self.alignment_weight * alignment_penalty + 
            self.trap_weight * trap_penalty +
            self.moves_weight * moves_score
        )

    def evaluate_alignment_penalty(self, board):
        """
        Penaliza si los mosqueteros están alineados en una fila o columna.
        """
        musketeer_positions = board.find_musketeer_positions()
        rows = [pos[0] for pos in musketeer_positions]
        cols = [pos[1] for pos in musketeer_positions]

        if len(set(rows)) == 1 or len(set(cols)) == 1:
            return -50
        return 0

    def evaluate_trap_penalty(self, board):
        """
        Penaliza si un mosquetero pisa una trampa.
        """
        trap_position = board.find_trap_position()
        musketeer_positions = board.find_musketeer_positions()
        
        if trap_position and trap_position in musketeer_positions:
            return -100  # Penalización fuerte por pisar la trampa
        return 0
    
    def evaluate_moves_score(self, board):
        """
        Puntaje basado en movimientos disponibles.
        """
        my_moves = len(board.get_possible_actions(self.player))
            
        # Queremos minimizar movimientos propios y del oponente
        return -1 * my_moves

    def next_action(self, obs):
        """
        Devuelve la mejor acción posible según el algoritmo Expectimax.
        """
        _, best_action = self.expectimax(obs, self.max_depth, self.player)
        return best_action

    def expectimax(self, board, depth, current_player):
        """
        Algoritmo Expectimax.
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
                eval, _ = self.expectimax(board_copy, depth - 1, self.opponent)
                if eval > max_eval:
                    max_eval = eval
                    best_action = action
            return max_eval, best_action
        else:
            total_eval = 0
            for action in actions:
                board_copy = board.clone()
                board_copy.play(current_player, action)
                eval, _ = self.expectimax(board_copy, depth - 1, self.player)
                total_eval += eval
            expected_value = total_eval / len(actions) if actions else 0
            return expected_value, None
