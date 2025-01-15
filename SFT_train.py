import chex
import jax
import jax.numpy as jnp
import pax
import opax
import optax


class ResidualBlock(pax.Module):
    """A residual block of conv layers

    y(x) = x + F(x),

    where:
        F = BatchNorm >> relu >> Conv1 >> BatchNorm >> relu >> Conv2
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.batchnorm1 = pax.BatchNorm2D(dim, True, True)
        self.batchnorm2 = pax.BatchNorm2D(dim, True, True)
        self.conv1 = pax.Conv2D(dim, dim, 3)
        self.conv2 = pax.Conv2D(dim, dim, 3)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        t = x
        t = jax.nn.relu(self.batchnorm1(t))
        t = self.conv1(t)
        t = jax.nn.relu(self.batchnorm2(t))
        t = self.conv2(t)
        return x + t


class ResnetPolicyValueNet(pax.Module):
    """Residual Conv Policy-Value network.

    Two-head network:
                        ┌─> action head
    input ─> Backbone  ─┤
                        └─> value head
    """

    def __init__(
        self, input_dims, num_actions: int, dim: int = 64, num_resblock: int = 5
    ) -> None:
        super().__init__()
        if len(input_dims) == 3:
            num_input_channels = input_dims[-1]
            input_dims = input_dims[:-1]
            self.has_channel_dim = True
        else:
            num_input_channels = 1
            self.has_channel_dim = False

        self.input_dims = input_dims
        self.num_actions = num_actions
        self.backbone = pax.Sequential(
            pax.Conv2D(num_input_channels, dim, 1), pax.BatchNorm2D(dim)
        )
        for _ in range(num_resblock):
            self.backbone >>= ResidualBlock(dim)
        self.action_head = pax.Sequential(
            pax.Conv2D(dim, dim, 1),
            pax.BatchNorm2D(dim, True, True),
            jax.nn.relu,
            pax.Conv2D(dim, self.num_actions, kernel_shape=input_dims, padding="VALID"),
        )
        self.value_head = pax.Sequential(
            pax.Conv2D(dim, dim, 1),
            pax.BatchNorm2D(dim, True, True),
            jax.nn.relu,
            pax.Conv2D(dim, dim, kernel_shape=input_dims, padding="VALID"),
            pax.BatchNorm2D(dim, True, True),
            jax.nn.relu,
            pax.Conv2D(dim, 1, kernel_shape=1, padding="VALID"),
            jnp.tanh,
        )

    def __call__(self, x: chex.Array, batched: bool = False):
        """Compute the action logits and value.

        Support both batched and unbatched states.
        """
        x = x.astype(jnp.float32)
        if not batched:
            x = x[None]  # add the batch dimension
        if not self.has_channel_dim:
            x = x[..., None]  # add channel dimension
        x = self.backbone(x)
        action_logits = self.action_head(x)
        value = self.value_head(x)
        if batched:
            return action_logits[:, 0, 0, :], value[:, 0, 0, 0]
        else:
            return action_logits[0, 0, 0, :], value[0, 0, 0, 0]


class ResnetPolicyValueNet128(ResnetPolicyValueNet):
    """Create a resnet of 128 channels, 5 blocks"""

    def __init__(
        self, input_dims, num_actions: int, dim: int = 128, num_resblock: int = 5
    ):
        super().__init__(input_dims, num_actions, dim, num_resblock)


@pax.pure
def loss_fn(model: pax.Module, inputs):
    # Extract state from inputs dictionary
    state = inputs['state']
    action = inputs['action']
    action_logits, value = model(state, batched=True)
    # Calculate cross entropy loss using JAX native functions
    logits = jax.nn.log_softmax(action_logits)
    loss = -logits[jnp.arange(logits.shape[0]), action]
    return jnp.mean(loss)


def update_fn(model: pax.Module, optimizer: opax.GradientTransformation, inputs):
    loss, grads = jax.value_and_grad(loss_fn)(model, inputs)
    model, optimizer = opax.apply_gradients(model, optimizer, grads=grads)
    return model, optimizer, loss

# Create dummy data with correct shapes
inputs = {
    'state': jnp.ones((10, 64, 64, 3)),  # [batch_size, height, width, channels]
    'action': jnp.zeros((10,), dtype=jnp.int32)  # [batch_size,] with integer labels
}

model = ResnetPolicyValueNet128((64, 64, 3), 1)
optimizer = opax.adam(1e-3)
# Initialize the optimizer state with the model parameters
optimizer = optimizer.init(model.parameters())

model, optimizer, loss = update_fn(model, optimizer, inputs)
print(loss)