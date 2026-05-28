import numpy as np
import pytest

from nn.optimizer import *


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def run_quadratic(optimizer, steps=300, theta_init=5.0, tol=1e-3):
    """
    Minimize f(θ) = 0.5 * θ²  →  gradient = θ  →  true minimum at θ = 0.
    Returns final theta (scalar) and loss history.
    """
    theta = np.array(float(theta_init))
    losses = []
    for _ in range(steps):
        losses.append(float(0.5 * theta ** 2))
        grad = theta          # df/dθ = θ
        theta = optimizer.optimize(theta, grad)
    return float(theta), losses


def run_quadratic_matrix(optimizer, shape=(4, 3), steps=300, tol=1e-3):
    """Same test but with a 2-D weight matrix — mirrors how Linear uses optimizers."""
    theta = np.random.randn(*shape) * 3.0
    for _ in range(steps):
        grad = theta
        theta = optimizer.optimize(theta, grad)
    return theta


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures — one fresh instance per test
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def gd():
    return GradientDescent(lr=0.05)

@pytest.fixture
def momentum():
    return Momentum(lr=0.05, gamma=0.9)

@pytest.fixture
def adagrad():
    return AdaGrad(lr=0.5, epsilon=1e-4)

@pytest.fixture
def rmsprop():
    return RMSProp(lr=0.05, gamma=0.9, epsilon=1e-4)

@pytest.fixture
def adam():
    # Standard hyperparameters
    return Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Convergence — scalar
# ═════════════════════════════════════════════════════════════════════════════

class TestConvergenceScalar:
    """Each optimizer must drive θ → 0 on f(θ) = 0.5 * θ²."""

    TOL = 1e-2

    def test_gradient_descent(self, gd):
        theta, _ = run_quadratic(gd)
        assert abs(theta) < self.TOL, f"GD: θ={theta:.4f}"

    def test_momentum(self, momentum):
        theta, _ = run_quadratic(momentum)
        assert abs(theta) < self.TOL, f"Momentum: θ={theta:.4f}"

    def test_adagrad(self, adagrad):
        theta, _ = run_quadratic(adagrad)
        assert abs(theta) < self.TOL, f"AdaGrad: θ={theta:.4f}"

    def test_rmsprop(self, rmsprop):
        theta, _ = run_quadratic(rmsprop)
        assert abs(theta) < self.TOL, f"RMSProp: θ={theta:.4f}"

    def test_adam(self, adam):
        theta, _ = run_quadratic(adam)
        assert abs(theta) < self.TOL, f"Adam: θ={theta:.4f}"


# ═════════════════════════════════════════════════════════════════════════════
# 2. Convergence — matrix input (mirrors Linear layer usage)
# ═════════════════════════════════════════════════════════════════════════════

class TestConvergenceMatrix:
    """Optimizers must handle 2-D numpy arrays without error and converge."""

    TOL = 1e-2

    def test_gradient_descent(self, gd):
        theta = run_quadratic_matrix(gd)
        assert np.all(np.abs(theta) < self.TOL), f"GD matrix max|θ|={np.max(np.abs(theta)):.4f}"

    def test_momentum(self, momentum):
        theta = run_quadratic_matrix(momentum)
        assert np.all(np.abs(theta) < self.TOL)

    def test_adagrad(self, adagrad):
        theta = run_quadratic_matrix(adagrad)
        assert np.all(np.abs(theta) < self.TOL)

    def test_rmsprop(self, rmsprop):
        theta = run_quadratic_matrix(rmsprop)
        assert np.all(np.abs(theta) < self.TOL)

    def test_adam(self, adam):
        theta = run_quadratic_matrix(adam)
        assert np.all(np.abs(theta) < self.TOL)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Loss is non-increasing (general trend check)
# ═════════════════════════════════════════════════════════════════════════════

class TestLossDecreasing:
    """
    Loss at step N should be lower than at step 0.
    Momentum is allowed to overshoot temporarily, so we only check
    the first vs last loss, not strict monotonicity.
    """

    def _assert_loss_decreased(self, opt, name):
        _, losses = run_quadratic(opt)
        assert losses[-1] < losses[0], (
            f"{name}: loss did not decrease. "
            f"start={losses[0]:.4f}, end={losses[-1]:.4f}"
        )

    def test_gradient_descent(self, gd):
        self._assert_loss_decreased(gd, "GradientDescent")

    def test_momentum(self, momentum):
        self._assert_loss_decreased(momentum, "Momentum")

    def test_adagrad(self, adagrad):
        self._assert_loss_decreased(adagrad, "AdaGrad")

    def test_rmsprop(self, rmsprop):
        self._assert_loss_decreased(rmsprop, "RMSProp")

    def test_adam(self, adam):
        self._assert_loss_decreased(adam, "Adam")


# ═════════════════════════════════════════════════════════════════════════════
# 4. Zero gradient → theta unchanged
# ═════════════════════════════════════════════════════════════════════════════

class TestZeroGradient:
    """If gradient is zero, optimizer must not move theta."""

    def _assert_no_update(self, opt, name):
        theta = np.array(3.0)
        grad  = np.array(0.0)
        theta_after = opt.optimize(theta.copy(), grad)
        assert abs(float(theta_after) - 3.0) < 1e-9, (
            f"{name}: theta moved on zero gradient. "
            f"before=3.0, after={theta_after:.6f}"
        )

    def test_gradient_descent(self, gd):
        self._assert_no_update(gd, "GradientDescent")

    def test_momentum(self, momentum):
        self._assert_no_update(momentum, "Momentum")

    def test_adagrad(self, adagrad):
        self._assert_no_update(adagrad, "AdaGrad")

    def test_rmsprop(self, rmsprop):
        self._assert_no_update(rmsprop, "RMSProp")

    def test_adam(self, adam):
        self._assert_no_update(adam, "Adam")


# ═════════════════════════════════════════════════════════════════════════════
# 5. Output shape preservation
# ═════════════════════════════════════════════════════════════════════════════

class TestOutputShape:
    """optimize() must return an array of the same shape as theta."""

    SHAPES = [(1,), (5,), (3, 4), (8, 8)]

    @pytest.mark.parametrize("shape", SHAPES)
    def test_gradient_descent(self, gd, shape):
        theta = np.ones(shape)
        assert gd.optimize(theta, theta).shape == shape

    @pytest.mark.parametrize("shape", SHAPES)
    def test_momentum(self, momentum, shape):
        theta = np.ones(shape)
        assert momentum.optimize(theta, theta).shape == shape

    @pytest.mark.parametrize("shape", SHAPES)
    def test_adagrad(self, adagrad, shape):
        theta = np.ones(shape)
        assert adagrad.optimize(theta, theta).shape == shape

    @pytest.mark.parametrize("shape", SHAPES)
    def test_rmsprop(self, rmsprop, shape):
        theta = np.ones(shape)
        assert rmsprop.optimize(theta, theta).shape == shape

    @pytest.mark.parametrize("shape", SHAPES)
    def test_adam(self, adam, shape):
        theta = np.ones(shape)
        assert adam.optimize(theta, theta).shape == shape


# ═════════════════════════════════════════════════════════════════════════════
# 6. OptimizerBuilder
# ═════════════════════════════════════════════════════════════════════════════

class TestOptimizerBuilder:

    VALID_NAMES = [
        "gradient descent",
        "Gradient Descent",
        "momentum",
        "adagrad",
        "rmsprop",
        "adam",
    ]

    EXPECTED_TYPES = {
        "gradient descent": GradientDescent,
        "momentum":         Momentum,
        "adagrad":          AdaGrad,
        "rmsprop":          RMSProp,
        "adam":             Adam,
    }

    @pytest.mark.parametrize("name", VALID_NAMES)
    def test_valid_names_return_optimizer(self, name):
        builder = OptimizerBuilder()
        opt = builder.build(name)
        assert opt is not None, f"builder.build('{name}') returned None"

    def test_returned_types(self):
        builder = OptimizerBuilder()
        for name, expected_cls in self.EXPECTED_TYPES.items():
            opt = builder.build(name)
            assert isinstance(opt, expected_cls), (
                f"builder.build('{name}') returned {type(opt).__name__}, "
                f"expected {expected_cls.__name__}"
            )

    def test_built_optimizer_is_callable(self):
        """Optimizer returned by builder must be usable immediately."""
        builder = OptimizerBuilder()
        opt = builder.build("adam")
        theta = np.array(3.0)
        result = opt.optimize(theta, theta)
        assert result is not None