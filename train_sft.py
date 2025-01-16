import os
import pickle
import random
from functools import partial
from typing import Optional

import chex
import jax
import jax.numpy as jnp
import pax
import opax
import optax
import os
import pickle
import fire
import wandb
from tqdm import tqdm, trange
from typing import Dict
from utils import import_class


@pax.pure
def loss_fn(model: pax.Module, inputs):
    # Extract state from inputs dictionary
    state = inputs['seq_positions']
    action = inputs['actions']
    action_logits, _ = model(state, batched=True)
    # Calculate cross entropy loss using JAX native functions
    logits = jax.nn.log_softmax(action_logits)
    loss = -logits[jnp.arange(logits.shape[0]), action]
    return jnp.mean(loss)


def update_fn(model: pax.Module, optimizer: opax.GradientTransformation, inputs):
    loss, grads = jax.value_and_grad(loss_fn)(model, inputs)
    model, optimizer = opax.apply_gradients(model, optimizer, grads=grads)
    return model, optimizer, loss

# Create dummy data with correct shapes
# inputs = {
#     'state': jnp.ones((10, 64, 64, 3)),  # [batch_size, height, width, channels]
#     'action': jnp.zeros((10,), dtype=jnp.int32)  # [batch_size,] with integer labels
# }

# model = ResnetPolicyValueNet128((64, 64, 3), 1)
# optimizer = opax.adam(1e-3)
# # Initialize the optimizer state with the model parameters
# optimizer = optimizer.init(model.parameters())

# model, optimizer, loss = update_fn(model, optimizer, inputs)
# print(loss)
def load_dataset(dataset_path):
    """Load dataset from a file"""
    with jnp.load(dataset_path) as data:
        return {k: v for k, v in data.items()}

def create_batch_iterator(data, batch_size, key, num_devices=1):
    """Create an iterator that yields batches of data"""
    # Get dataset size
    dataset_size = len(data['seq_positions'])
    
    def shuffle_and_batch(key):
        # Number of complete batches
        steps_per_epoch = dataset_size // batch_size
        
        # Create permutation indices
        perm = jax.random.permutation(key, steps_per_epoch*batch_size)

        # Shuffle and batch the data
        batched_data = {}
        for k, v in data.items():
            if k == 'seq_positions':
                # Transpose seq_positions from (N, seq_len, H, W) to (N, H, W, seq_len)
                v = jnp.transpose(v, (0, 2, 3, 1))
            
            batched_data[k] = (
                v[perm,:].reshape((steps_per_epoch, batch_size, *v.shape[1:]))
                if len(v.shape) > 1 else
                v[perm].reshape((steps_per_epoch, batch_size))
            )
        
        return batched_data, steps_per_epoch

    return shuffle_and_batch

    
def train_step(net, optim, batch: Dict[str, chex.Array]):
    """A single training step."""
    loss, grads = jax.value_and_grad(loss_fn)(net, batch)
    net, optim = opax.apply_gradients(net, optim, grads)
    return net, optim, loss

def train(
    game_class: str = "games.go_game.GoBoard9x9",
    agent_class: str = "policies.resnet_policy.ResnetPolicyValueNet128",
    training_batch_size: int = 1024,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    ckpt_filename: str = "./sft_agent.ckpt",
    random_seed: int = 42,
    weight_decay: float = 1e-4,
    wandb_project: str = "a0-jax-sft",
    dataset_path: str = "./go9x9_data.npz",
):
    """Train an agent using supervised learning from demonstrations."""
    
    wandb.init(
        project=wandb_project,
        config={
            "game_class": game_class,
            "agent_class": agent_class,
            "training_batch_size": training_batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "random_seed": random_seed,
            "num_epochs": num_epochs,
        }
    )

    # Initialize environment and agent
    env = import_class(game_class)()
    agent = import_class(agent_class)(
        input_dims=env.observation().shape,
        num_actions=env.num_actions(),
    )

    # Initialize optimizer
    optim = opax.chain(
        opax.add_decayed_weights(weight_decay),
        opax.adam(learning_rate),
    ).init(agent.parameters())

    # Load checkpoint if exists
    start_epoch = 0
    if os.path.isfile(ckpt_filename):
        print("Loading weights at", ckpt_filename)
        with open(ckpt_filename, "rb") as f:
            dic = pickle.load(f)
            agent = agent.load_state_dict(dic["agent"])
            optim = optim.load_state_dict(dic["optim"])
            start_epoch = dic["epoch"] + 1
    
    # # Training loop
    # devices = jax.local_devices()
    num_devices = 1
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_path)
    
    # Create batch iterator
    batch_iterator_fn = create_batch_iterator(
        dataset, 
        training_batch_size, 
        jax.random.PRNGKey(random_seed),
        num_devices
    )

    for epoch in trange(start_epoch, num_epochs):        
        # Shuffle data for this epoch
        key = jax.random.PRNGKey(random_seed + epoch)
        batched_data, steps_per_epoch = batch_iterator_fn(key)
        
        agent = agent.train()
        losses = []
        for step in trange(steps_per_epoch, desc=f"Epoch {epoch}"):
            batch = {k: v[step] for k, v in batched_data.items()}
            agent, optim, loss = train_step(agent, optim, batch)
            losses.append(loss)
            wandb.log({"iteration_loss": loss, "step": step + epoch * steps_per_epoch})
            
        avg_loss = jnp.mean(jnp.array(losses))
        
        # Save checkpoint
        with open(ckpt_filename, "wb") as writer:
            dic = {
                "agent": jax.device_get(agent.state_dict()),
                "optim": jax.device_get(optim.state_dict()),
                "epoch": epoch,
            }
            pickle.dump(dic, writer)

        # Log metrics
        wandb.log({
            "epoch_loss": avg_loss,
            "learning_rate": optim[1][-1].learning_rate,
        })

        print(f"Epoch {epoch}: loss={avg_loss:.4f}, lr={optim[1][-1].learning_rate:.1e}")

    wandb.finish()
    print("Done!")

if __name__ == "__main__":
    if "COLAB_TPU_ADDR" in os.environ:
        jax.tools.colab_tpu.setup_tpu()
    print("Cores:", jax.local_devices())
    
    fire.Fire(train)