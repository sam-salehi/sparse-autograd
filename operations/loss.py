from .base import Operation
from tensor import Tensor
import numpy as np




class MSELoss(Operation):
    # works for both batches and single samples.
    @staticmethod
    def apply(actual: Tensor, pred: Tensor):
        op = MSELoss()
        assert pred.ndim == actual.ndim, f"Dimensions must match. Got actual: {actual.ndim}, pred: {pred.ndim}"
        assert pred.shape == actual.shape, f"Shapes must match. Got actual: {actual.shape}, pred: {pred.shape}"
        op._prev = (actual, pred)
        res = op._forward(actual.data, pred.data)
        return Tensor(res, op)

    def _forward(self, actual: np.ndarray, pred: np.ndarray) -> np.ndarray: 
        # MSE = (1/n) * Σ(y_true - y_pred)²
        # For batches: returns mean across all elements (both batch and feature dimensions)
        # Input shapes: (batch_size, ...) for both actual and pred
        return np.mean((actual - pred) ** 2)

    def _backward(self, grad_output: float):
        actual_prev = self._prev[0].data
        pred_prev = self._prev[1].data
        
        # For MSE loss:
        # dL/dy_true = (2/n) * (y_true - y_pred)
        # dL/dy_pred = -(2/n) * (y_true - y_pred)
        # where n is total number of elements (batch_size * feature_size)
        n = actual_prev.size
        diff = actual_prev - pred_prev
        
        # grad_output gets broadcasted for input shape
        grad_actual = grad_output * (2/n) * diff
        grad_pred = grad_output * -(2/n) * diff
        
        return grad_actual, grad_pred


class KLDiv(Operation):
    def __init__(self, sparsity):
        self.sparsity = sparsity  # desired sparsity (rho)
        self.rho_hat = None       # will be set in forward

    @staticmethod
    def apply(activations: Tensor, sparsity: float) -> Tensor:
        op = KLDiv(sparsity)
        op._prev = (activations,)
        res = op._forward(activations.data)
        return Tensor(res, op)

    def _forward(self, activations: np.ndarray) -> np.ndarray:
        # Compute mean activation per hidden unit (axis=0: batch dimension)
        self.rho_hat = np.mean(activations, axis=0)
        rho = self.sparsity
        rho_hat = self.rho_hat
        # Add small epsilon for numerical stability
        eps = 1e-8
        kl = rho * np.log((rho + eps) / (rho_hat + eps)) + \
             (1 - rho) * np.log((1 - rho + eps) / (1 - rho_hat + eps))
        return np.sum(kl)  # sum over all hidden units

    def _backward(self, grad_output: np.ndarray) -> np.ndarray:
        # grad_output is a scalar (from loss), so we broadcast
        rho = self.sparsity
        rho_hat = self.rho_hat
        # dKL/d(rho_hat)
        grad = (-rho / (rho_hat + 1e-8)) + ((1 - rho) / (1 - rho_hat + 1e-8))
        # Since rho_hat = mean(activations), d(rho_hat)/d(activations) = 1/N
        batch_size = self._prev[0].data.shape[0]
        grad = grad / batch_size  # distribute over batch
        # Broadcast grad to match activations shape
        grad_full = np.ones_like(self._prev[0].data) * grad  # shape: (batch, hidden)
        return grad_output * grad_full,  # return as tuple


