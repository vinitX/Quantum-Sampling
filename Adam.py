import numpy as np

class Adam:
    """
    Pure NumPy Adam optimizer with a TensorFlow-like API.

    Usage (like TF):
        opt = Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False)
        opt.apply_gradients([(grad, x), (g2, w2), ...])

    Notes:
    - Parameters (x, w2, …) must be np.ndarray and will be updated in-place.
    - Grads can be None (they're skipped), like TF.
    - epsilon is added INSIDE the denominator (TF/Keras behavior).
    - Defaults match tf.keras.optimizers.Adam in TF2.
    """

    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False):
        self.lr = float(learning_rate)
        self.b1 = float(beta_1)
        self.b2 = float(beta_2)
        self.eps = float(epsilon)
        self.amsgrad = bool(amsgrad)

        self.t = 0  # time step
        # per-parameter slots keyed by id(param)
        self._m = {}         # first moment
        self._v = {}         # second moment
        self._vhat_max = {}  # for AMSGrad

    def _ensure_slots(self, param):
        """Create m/v (and vhat_max) slots for this parameter if missing."""
        key = id(param)
        if key not in self._m:
            self._m[key] = np.zeros_like(param)
            self._v[key] = np.zeros_like(param)
            if self.amsgrad and key not in self._vhat_max:
                self._vhat_max[key] = np.zeros_like(param)
        elif self.amsgrad and key not in self._vhat_max:
            self._vhat_max[key] = np.zeros_like(param)

    def apply_gradients(self, grads_and_vars):
        """
        grads_and_vars: iterable of (grad, param) pairs, like TF's apply_gradients.
                        - grad: np.ndarray (same shape as param) or None
                        - param: np.ndarray updated in-place
        """
        # Increment time step once per call (like one "optimizer step")
        self.t += 1
        b1t_corr = 1.0 - (self.b1 ** self.t)
        b2t_corr = 1.0 - (self.b2 ** self.t)

        for grad, param in grads_and_vars:
            if grad is None:
                continue
            if not isinstance(param, np.ndarray):
                raise TypeError("Parameter must be a NumPy array.")
            if not isinstance(grad, np.ndarray):
                grad = np.asarray(grad, dtype=param.dtype)

            self._ensure_slots(param)
            key = id(param)
            m = self._m[key]
            v = self._v[key]

            # m_t = beta1 * m + (1 - beta1) * g
            m *= self.b1
            m += (1.0 - self.b1) * grad

            # v_t = beta2 * v + (1 - beta2) * g^2
            v *= self.b2
            v += (1.0 - self.b2) * (grad * grad)

            # Bias correction
            m_hat = m / b1t_corr
            v_hat = v / b2t_corr

            if self.amsgrad:
                vhat_max = self._vhat_max[key]
                np.maximum(vhat_max, v_hat, out=vhat_max)
                denom = np.sqrt(vhat_max) + self.eps
            else:
                denom = np.sqrt(v_hat) + self.eps

            # p -= lr * m_hat / (sqrt(v_hat) + eps)
            param -= self.lr * (m_hat / denom)

    # Optional helpers to mirror TF ergonomics
    def get_config(self):
        return {
            "learning_rate": self.lr,
            "beta_1": self.b1,
            "beta_2": self.b2,
            "epsilon": self.eps,
            "amsgrad": self.amsgrad,
        }

    def reset_state(self):
        self.t = 0
        self._m.clear()
        self._v.clear()
        self._vhat_max.clear()
