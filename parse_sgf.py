import os
import numpy as np
from sgfmill import sgf, boards
from tqdm import tqdm

def board_to_array(board: boards.Board):
    size = board.side
    arr = np.zeros((size, size), dtype=np.int8)
    for r in range(size):
        for c in range(size):
            stone = board.get(r, c)
            if stone == 'b':
                arr[r, c] = 1
            elif stone == 'w':
                arr[r, c] = -1
    return arr

def parse_sgf_to_numpy(filename: str) -> np.ndarray:
    with open(filename, 'rb') as f:
        sgf_game = sgf.Sgf_game.from_bytes(f.read())

    board_size = sgf_game.get_size()
    if board_size != 9:
        return None

    board = boards.Board(board_size)
    positions = []
    actions = []
    seq_positions = []  # New list for sequence positions
    recent_positions = [board_to_array(board)] * 8  # Keep track of recent positions (9 - 1 
    
    for node in sgf_game.get_main_sequence():
        if node.get_move():
            move = node.get_move()
            if move[0] is None or move[1] is None:
                continue
        color, (row, col) = move
        
        # Record the current board state in canonical form (invariant to color)
        turn = 1 if color == 'b' else -1
        raw_board = board_to_array(board)
        current_position = raw_board * turn
        positions.append(current_position)
        
        # Update recent positions and create sequence
        recent_positions.append(current_position)
        if len(recent_positions) > 9:
            recent_positions.pop(0)
        
        seq_positions.append(np.stack(recent_positions, axis=0))

        # Convert (row,col) to a single integer or pass
        if row is None or col is None:
            # pass move
            action_int = board_size * board_size
        else:
            action_int = row * board_size + col

        actions.append(action_int)

        # Now play the move
        board.play(row, col, color)

    if len(actions) == 0:
        return None, None

    actions = np.array(actions, dtype=np.int32)
    seq_positions = np.stack(seq_positions, axis=0)  # shape: (num_plays, 8, board_size, board_size)

    assert len(actions) == len(seq_positions)
    return actions, seq_positions


def parse_sgf_folder(sgf_folder: str):
    all_actions = []
    all_seq_positions = []  # New list for sequence positions
    
    # Recursively get all .sgf files
    sgf_files = []
    for root, _, files in os.walk(sgf_folder):
        for file in files:
            if file.endswith('.sgf'):
                sgf_files.append(os.path.join(root, file))
    
    # Wrap with tqdm for progress tracking
    for path in tqdm(sgf_files, desc="Parsing SGF files"):
        actions, seq_positions = parse_sgf_to_numpy(path)
        if actions is not None and seq_positions is not None:
            all_actions.append(actions)
            all_seq_positions.append(seq_positions)
    
    actions = np.concatenate(all_actions)
    seq_positions = np.concatenate(all_seq_positions)
    return actions, seq_positions

if __name__ == "__main__":
    actions, seq_positions = parse_sgf_folder("./data")
    print(f"Total number of positions: {len(actions)}")
    print(f"Actions shape: {actions.shape}")
    print(f"Sequence positions shape: {seq_positions.shape}")
    
    # Save positions and actions using numpy's savez
    np.savez("go9x9_data.npz", actions=actions, seq_positions=seq_positions)
