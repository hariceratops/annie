import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import NewType

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

Array = NDArray[np.float64]
# z = W @ x + b, the input to the nonlinearity
PreActivation = NewType("PreActivation", Array)
# a = f(z), the output of the nonlinearity
Activated = NewType("Activated", Array)


@dataclass(frozen=True, slots=True)
class Activation:
  forward: Callable[[PreActivation], Activated]
  # Vector-Jacobian product: given dL/da, return dL/dz. Gets both z and a so
  # each derivative can use whichever is cheaper.
  backward: Callable[[PreActivation, Activated, Array], Array]


def _sigmoid(z: PreActivation) -> Activated:
  # expit is the overflow-safe 1 / (1 + e^-z)
  return Activated(expit(z))


def _sigmoid_backward(z: PreActivation, a: Activated, grad: Array) -> Array:
  # σ'(z) = σ(z)(1 - σ(z)) = a(1 - a)
  return grad * a * (1.0 - a)


def _tanh(z: PreActivation) -> Activated:
  return Activated(np.tanh(z))


def _tanh_backward(z: PreActivation, a: Activated, grad: Array) -> Array:
  # tanh'(z) = 1 - tanh²(z) = 1 - a²
  return grad * (1.0 - a * a)


def _relu(z: PreActivation) -> Activated:
  return Activated(np.maximum(z, 0.0))


def _relu_backward(z: PreActivation, a: Activated, grad: Array) -> Array:
  # 1 for z > 0, else 0 (0 at z = 0 by convention)
  return grad * (z > 0.0)


def leaky_relu(alpha: float = 0.01) -> Activation:
  def forward(z: PreActivation) -> Activated:
      return Activated(np.where(z > 0.0, z, alpha * z))

  def backward(z: PreActivation, a: Activated, grad: Array) -> Array:
      return grad * np.where(z > 0.0, 1.0, alpha)

  return Activation(forward, backward)


def elu(alpha: float = 1.0) -> Activation:
  def forward(z: PreActivation) -> Activated:
      # np.where evaluates both branches; clamping keeps expm1 from
      # overflowing on large positive z that the branch discards anyway
      return Activated(np.where(z > 0.0, z, alpha * np.expm1(np.minimum(z, 0.0))))

  def backward(z: PreActivation, a: Activated, grad: Array) -> Array:
      # for z <= 0: α·e^z = a + α
      return grad * np.where(z > 0.0, 1.0, a + alpha)

  return Activation(forward, backward)


def _softplus(z: PreActivation) -> Activated:
  # ln(1 + e^z) without overflow
  return Activated(np.logaddexp(0.0, z))


def _softplus_backward(z: PreActivation, a: Activated, grad: Array) -> Array:
  # softplus'(z) = σ(z)
  return grad * expit(z)


_GELU_C = math.sqrt(2.0 / math.pi)
_GELU_K = 0.044715


def _gelu_tanh_term(z: Array) -> Array:
  return np.tanh(_GELU_C * (z + _GELU_K * z**3))


def _gelu(z: PreActivation) -> Activated:
  # tanh approximation: 0.5·z·(1 + t), t = tanh(√(2/π)(z + 0.044715z³))
  return Activated(0.5 * z * (1.0 + _gelu_tanh_term(z)))


def _gelu_backward(z: PreActivation, a: Activated, grad: Array) -> Array:
  # product rule: 0.5(1 + t) + 0.5·z·(1 - t²)·√(2/π)(1 + 3·0.044715z²)
  t = _gelu_tanh_term(z)
  dt = (1.0 - t * t) * _GELU_C * (1.0 + 3.0 * _GELU_K * z * z)
  return grad * (0.5 * (1.0 + t) + 0.5 * z * dt)


def _linear(z: PreActivation) -> Activated:
  return Activated(z)


def _linear_backward(z: PreActivation, a: Activated, grad: Array) -> Array:
  return grad


def _softmax(z: PreActivation) -> Activated:
  # subtracting the row max leaves the result unchanged and prevents overflow
  e = np.exp(z - z.max(axis=-1, keepdims=True))
  return Activated(e / e.sum(axis=-1, keepdims=True))


def _softmax_backward(z: PreActivation, a: Activated, grad: Array) -> Array:
  # Jacobian is a_i(δ_ij - a_j); multiplied by grad without building it:
  # dL/dz_i = a_i(g_i - Σ_j g_j a_j)
  return a * (grad - np.sum(grad * a, axis=-1, keepdims=True))


SIGMOID = Activation(_sigmoid, _sigmoid_backward)
TANH = Activation(_tanh, _tanh_backward)
RELU = Activation(_relu, _relu_backward)
SOFTPLUS = Activation(_softplus, _softplus_backward)
GELU = Activation(_gelu, _gelu_backward)
LINEAR = Activation(_linear, _linear_backward)
SOFTMAX = Activation(_softmax, _softmax_backward)
