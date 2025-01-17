"""
Agent vs Agent evaluation script.

Evaluate two AlphaZero agents against each other by loading from checkpoints.
"""

import pickle
import jax
import fire
from play import PlayResults, agent_vs_agent_multiple_games
from utils import import_class

def evaluate(
    game_class: str = "games.go_game.GoBoard9x9",
    agent_class: str = "policies.resnet_policy.ResnetPolicyValueNet128",
    agent1_path: str = "./checkpoints/agent1.ckpt",
    agent2_path: str = "./checkpoints/agent2.ckpt",
    num_games: int = 100,
    num_simulations_per_move: int = 32,
    random_seed: int = 42,
):
    """Evaluate two agents against each other.
    
    Args:
        game_class: Path to game class implementation
        agent_class: Path to agent class implementation
        agent1_path: Path to first agent's checkpoint
        agent2_path: Path to second agent's checkpoint
        num_games: Number of games to play (will be doubled as agents swap positions)
        num_simulations_per_move: Number of MCTS simulations per move
        random_seed: Random seed for reproducibility
    """
    # Initialize environment and agents
    env = import_class(game_class)()
    agent1 = import_class(agent_class)(
        input_dims=env.observation().shape,
        num_actions=env.num_actions(),
    )
    agent2 = import_class(agent_class)(
        input_dims=env.observation().shape,
        num_actions=env.num_actions(),
    )

    # Load agent checkpoints
    print(f"Loading agent 1 from {agent1_path}")
    with open(agent1_path, "rb") as f:
        agent1 = agent1.load_state_dict(pickle.load(f)["agent"])

    print(f"Loading agent 2 from {agent2_path}")
    with open(agent2_path, "rb") as f:
        agent2 = agent2.load_state_dict(pickle.load(f)["agent"])

    rng_key = jax.random.PRNGKey(random_seed)
    rng_key1, rng_key2 = jax.random.split(rng_key)

    # Evaluate with agent1 as player 1
    print("\nPlaying games with Agent 1 as player 1...")
    result1: PlayResults = agent_vs_agent_multiple_games(
        agent1.eval(),
        agent2.eval(),
        env,
        rng_key1,
        num_games=num_games,
        num_simulations_per_move=num_simulations_per_move,
    )

    # Evaluate with agent2 as player 1
    print("\nPlaying games with Agent 2 as player 1...")
    result2: PlayResults = agent_vs_agent_multiple_games(
        agent2.eval(),
        agent1.eval(),
        env,
        rng_key2,
        num_games=num_games,
        num_simulations_per_move=num_simulations_per_move,
    )

    # Print results
    print("\nResults when Agent 1 plays as player 1:")
    print(f"Wins: {result1.win_count}, Draws: {result1.draw_count}, Losses: {result1.loss_count}")
    
    print("\nResults when Agent 2 plays as player 1:")
    print(f"Wins: {result2.win_count}, Draws: {result2.draw_count}, Losses: {result2.loss_count}")
    
    print("\nOverall results for Agent 1:")
    total_wins = result1.win_count + result2.loss_count
    total_draws = result1.draw_count + result2.draw_count
    total_losses = result1.loss_count + result2.win_count
    total_games = total_wins + total_draws + total_losses
    
    # Print and log results
    results_str = [
        f"\nEvaluation Results:",
        f"Agent 1: {agent1_path}",
        f"Agent 2: {agent2_path}",
        f"\nOverall results for Agent 1:",
        f"\tTotal games played: {total_games}",
        f"\tWins: {total_wins} ({total_wins/total_games*100:.1f}%)",
        f"\tDraws: {total_draws} ({total_draws/total_games*100:.1f}%)",
        f"\tLosses: {total_losses} ({total_losses/total_games*100:.1f}%)\n"
    ]

    # Print to console
    print("\n".join(results_str))

    # Append to file
    with open("evaluation_results.txt", "a") as f:
        f.write("\n".join(results_str))

if __name__ == "__main__":
    fire.Fire(evaluate) 