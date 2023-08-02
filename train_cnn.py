import jax
import jax.numpy as jnp
from clu import metrics
from flax.training import train_state
from flax import struct
import optax
import tensorflow as tf
import matplotlib.pyplot as plt

from learn_jax.models import CNN
from learn_jax.data import get_datasets


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(module, rng, learning_rate, momentum):
    # the second argument: jnp.ones() is used to determine input_size and init the model
    # params accordingly. The value of it is not important while shape must be correct.
    params = module.init(rng, jnp.ones([1, 28, 28, 1]))["params"]
    tx = optax.sgd(learning_rate, momentum)
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=tx, metrics=Metrics.empty()
    )


@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["image"])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["label"]
        ).mean()
        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def compute_metrics(*, state, batch):
    logits = state.apply_fn({"params": state.params}, batch["image"])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["label"]
    ).mean()
    metric_updates = state.metrics.single_from_model_output(
        logits=logits, labels=batch["label"], loss=loss
    )
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


if __name__ == "__main__":
    num_epochs = 20
    batch_size = 32
    train_ds, test_ds = get_datasets(num_epochs, batch_size)
    tf.random.set_seed(0)
    init_rng = jax.random.PRNGKey(0)

    cnn = CNN()
    learning_rate = 0.01
    momentum = 0.9
    state = create_train_state(
        cnn, rng=init_rng, learning_rate=learning_rate, momentum=momentum
    )
    del init_rng
    steps_per_epoch = train_ds.cardinality().numpy() // num_epochs
    metrics_history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    for step, batch in enumerate(train_ds.as_numpy_iterator()):
        state = train_step(state, batch)
        state = compute_metrics(state=state, batch=batch)

        if (step + 1) % steps_per_epoch == 0:
            for metric, value in state.metrics.compute().items():
                metrics_history[f"train_{metric}"].append(value)
            state = state.replace(metrics=state.metrics.empty())

            for test_batch in test_ds.as_numpy_iterator():
                test_state = compute_metrics(state=state, batch=test_batch)
            for metric, value in test_state.metrics.compute().items():
                metrics_history[f"test_{metric}"].append(value)

            print(f"train epoch: {(step + 1) // steps_per_epoch}")
            print(f"loss: {metrics_history['train_loss'][-1]}")
            print(f"accuracy: {metrics_history['train_accuracy'][-1] * 100} %")
            print()
            print(f"test epoch: {(step+1) // steps_per_epoch}")
            print(f"loss: {metrics_history['test_loss'][-1]}")
            print(f"accuracy: {metrics_history['test_accuracy'][-1] * 100} %")
