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

    for node in sgf_game.get_main_sequence():
        if node.get_move():
            move = node.get_move()
            if move[0] is None or move[1] is None:
                continue
        color, (row, col) = move
        
        # Record the current board state in canonical form (invariant to color)
        turn = 1 if color == 'b' else -1
        raw_board = board_to_array(board)
        positions.append(raw_board * turn)


        # Convert (row,col) to a single integer or pass
        if row is None or col is None:
            # pass move
            action_int = board_size * board_size
        else:
            action_int = row * board_size + col

        actions.append(action_int)

        # Now play the move
        board.play(row, col, color)

    if len(positions) == 0:
        return None, None

    positions = np.stack(positions, axis=0)  # shape: (num_plays, board_size, board_size)
    actions = np.array(actions, dtype=np.int32)  # shape: (num_plays,)

    # assert that length of positions and actions are the same
    assert len(positions) == len(actions)
    return positions, actions

def parse_sgf_folder(sgf_folder: str):
    all_positions = []
    all_actions = []
    
    # Recursively get all .sgf files
    sgf_files = []
    for root, _, files in os.walk(sgf_folder):
        for file in files:
            if file.endswith('.sgf'):
                sgf_files.append(os.path.join(root, file))
    
    # Wrap with tqdm for progress tracking
    for path in tqdm(sgf_files, desc="Parsing SGF files"):
        positions, actions = parse_sgf_to_numpy(path)
        if positions is not None and actions is not None:
            all_positions.append(positions)
            all_actions.append(actions)
    
    positions = np.concatenate(all_positions)
    actions = np.concatenate(all_actions)
    return positions, actions

if __name__ == "__main__":
    positions, actions = parse_sgf_folder("./data")
    print(f"Total number of positions: {len(positions)}")
    print(f"Positions shape: {positions.shape}")
    print(f"Actions shape: {actions.shape}")
    
    # Save positions and actions using numpy's savez
    np.savez("go9x9_data.npz", positions=positions, actions=actions)
