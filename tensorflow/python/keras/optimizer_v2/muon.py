import re
import tensorflow as tf

from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2 as Optimizer

class Muon(Optimizer):
    
    def __init__(
        self,
        learning_rate=0.001,
        adam_beta_1=0.9,
        adam_beta_2=0.999,
        epsilon=1e-7,
        weight_decay=0.1,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="muon",
        exclude_layers=None,
        exclude_embeddings=True,
        muon_a=3.4445,
        muon_b=-4.7750,
        muon_c=2.0315,
        adam_lr_ratio=0.1,
        momentum=0.95,
        ns_steps=6,
        nesterov=True,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.epsilon = epsilon
        self.muon_a = muon_a
        self.muon_b = muon_b
        self.muon_c = muon_c
        self.adam_lr_ratio = adam_lr_ratio
        self.momentum = momentum
        self.ns_steps = ns_steps
        self.nesterov = nesterov
        self.exclude_embeddings = exclude_embeddings
        self.exclude_layers = exclude_layers or []

    def _should_use_adamw(self, variable):
        if not (len(variable.shape) == 2 or len(variable.shape) == 3):
             return True

        var_id = variable.name
        if self.exclude_embeddings and "embedding" in var_id.lower():
            return True
        for keyword in self.exclude_layers:
            if re.search(keyword, var_id):
                return True
        return False

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
            if self._should_use_adamw(var):
                self.add_slot(var, "v")

    def _resource_apply_dense(self, grad, var):
        learning_rate = self._get_hyper("learning_rate")
        
        if self._should_use_adamw(var):
            self._adamw_update_step(grad, var, learning_rate * self.adam_lr_ratio)
        else:
            self._muon_update_step(grad, var, learning_rate)

    def _resource_apply_sparse(self, grad, var):
        dense_grad = tf.convert_to_tensor(grad)
        self._resource_apply_dense(dense_grad, var)

    def _muon_update_step(self, gradient, variable, lr):
        m = self.get_slot(variable, "m")

        m.assign(tf.math.add(tf.math.multiply(self.momentum, m), tf.math.multiply(tf.constant(1.0) - self.momentum, gradient)))

        shape = tf.shape(variable)
        if self.nesterov:
            g = tf.math.add(gradient, self.momentum * m)
        else:
            g = m

        ortho_result = self.zeropower_via_newtonschulz5(g, self.ns_steps)

        aspect_ratio_scale = tf.constant(1.0, dtype=ortho_result.dtype)
        if len(shape) >= 2 and shape[1] != 0:
             aspect_ratio_scale = tf.sqrt(tf.maximum(tf.constant(1.0, dtype=ortho_result.dtype),
                                                     tf.cast(shape[0], ortho_result.dtype) / tf.cast(shape[1], ortho_result.dtype)))

        variable.assign_sub(lr * ortho_result * aspect_ratio_scale)

    def _adamw_update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        gradient = tf.cast(gradient, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)

        adam_beta_1_power = tf.pow(
            tf.cast(self.adam_beta_1, variable.dtype), local_step
        )
        adam_beta_2_power = tf.pow(
            tf.cast(self.adam_beta_2, variable.dtype), local_step
        )

        m = self.get_slot(variable, "m")
        v = self.get_slot(variable, "v")

        alpha = lr * tf.sqrt(1 - adam_beta_2_power) / (1 - adam_beta_1_power)

        m.assign(tf.math.add(tf.math.multiply(self.adam_beta_1, m), tf.math.multiply(tf.constant(1.0) - self.adam_beta_1, gradient)))

        v.assign(tf.math.add(tf.math.multiply(self.adam_beta_2, v), tf.math.multiply(tf.constant(1.0) - self.adam_beta_2, tf.square(gradient))))

        variable.assign_sub(
            tf.divide(
                tf.multiply(m, alpha), tf.add(tf.sqrt(v), self.epsilon)
            )
        )

    def transpose_last_axis(self, X):
        shape = tf.shape(X)
        temp_order = list(range(len(shape)))
        if len(shape) < 2:
            return X
        temp_order[-2], temp_order[-1] = temp_order[-1], temp_order[-2]
        X = tf.transpose(X, temp_order)
        return X

    def zeropower_via_newtonschulz5(self, x, steps: int):
        """We apply the Newton-Schulz iteration to compute matrix G.

        We select a quintic iteration that maximizes the slope at zero. This
        approach helps minimize steps, even if the iteration doesn't fully
        converge across the interval. The result isn't exactly UV^T (from the
        SVD of G), but rather an approximation like US'V^T. Despite this
        approximation, model performance remains unaffected compared to using
        the exact UV^T from the SVD.
        """
        shape = tf.shape(x)
        assert len(shape) >= 2, f"Input must have at least 2 dimensions, got shape {shape}"

        a, b, c = self.muon_a, self.muon_b, self.muon_c
        
        original_transpose_needed = False
        if shape[-2] > shape[-1]:
            x = self.transpose_last_axis(x)
            original_transpose_needed = True

        norm = tf.norm(x, axis=(-2, -1), keepdims=True)
        x = x / (norm + self.epsilon)

        for _ in range(steps):
            temp_a = tf.matmul(x, self.transpose_last_axis(x))
            temp_b = b * temp_a + c * tf.matmul(temp_a, temp_a)
            x = a * x + tf.matmul(temp_b, x)

        if original_transpose_needed:
            x = self.transpose_last_axis(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "adam_beta_1": self.adam_beta_1,
                "adam_beta_2": self.adam_beta_2,
                "epsilon": self.epsilon,
                "exclude_layers": self.exclude_layers,
                "muon_a": self.muon_a,
                "muon_b": self.muon_b,
                "muon_c": self.muon_c,
                "adam_lr_ratio": self.adam_lr_ratio,
                "momentum": self.momentum,
                "ns_steps": self.ns_steps,
                "nesterov": self.nesterov,
                "exclude_embeddings": self.exclude_embeddings,
            }
        )
        return config