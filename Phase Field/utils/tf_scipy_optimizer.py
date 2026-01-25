import tensorflow.compat.v1 as tf
import numpy as np
import scipy.optimize
import sys

class ScipyOptimizerInterface(object):
    """
    A compatible replacement for tf.contrib.opt.ScipyOptimizerInterface.
    This version uses scipy.optimize.minimize directly.
    """

    def __init__(self, loss, var_list=None, equalitites=None, inequalities=None, var_to_bounds=None,
                 method='L-BFGS-B', options=None):
        """
        Initialize the optimizer.
        Args:
            loss: A scalar Tensor to be minimized.
            var_list: Optional list or tuple of Variable objects to update to minimize
                loss.  Defaults to the list of variables collected in the graph
                under the key GraphKeys.TRAINABLE_VARIABLES.
            equalitites: Optional list of equality constraints (not supported in this shim).
            inequalities: Optional list of inequality constraints (not supported in this shim).
            var_to_bounds: Optional dict where key is variable and value is a list of pairs
                of upper and lower bounds (not supported in this shim).
            method: The method to be used by scipy.optimize.minimize.
            options: A dictionary of solver options.
        """
        self._loss = loss
        
        if var_list is None:
            self._vars = tf.trainable_variables()
        else:
            self._vars = var_list

        self._method = method
        self._options = options if options is not None else {}
        
        # We need to flatten the variables to a single vector for scipy
        self._shapes = [v.get_shape().as_list() for v in self._vars]
        self._sizes = [np.prod(s) for s in self._shapes]
        
        # Check if we can compute gradients
        self._grads = tf.gradients(loss, self._vars)
        
        # Placeholders for assigning new weights
        self._placeholders = [tf.placeholder(v.dtype, shape=v.get_shape()) for v in self._vars]
        self._assign_ops = [v.assign(p) for v, p in zip(self._vars, self._placeholders)]

    def _pack(self, values):
        """Pack a list of arrays into a single flat array."""
        return np.hstack([v.flatten() for v in values])

    def _unpack(self, flat_values):
        """Unpack a flat array into a list of arrays with original shapes."""
        values = []
        offset = 0
        for i, shape in enumerate(self._shapes):
            size = self._sizes[i]
            val = flat_values[offset:offset + size].reshape(shape)
            values.append(val)
            offset += size
        return values

    def minimize(self, session=None, feed_dict=None, fetches=None, step_callback=None, loss_callback=None, **run_kwargs):
        """
        Minimize the loss.
        """
        if session is None:
            session = tf.get_default_session()
            
        if feed_dict is None:
            feed_dict = {}
            
        # Define the function to optimize
        def loss_and_grad(params):
            # 1. Update the variables in the graph with the current params from scipy
            unpacked_params = self._unpack(params)
            feed_dict_assign = dict(zip(self._placeholders, unpacked_params))
            session.run(self._assign_ops, feed_dict=feed_dict_assign)
            
            # 2. Compute loss and gradients
            # We must extend the feed_dict with the user provided feed_dict
            loss_val, grads_val = session.run([self._loss, self._grads], feed_dict=feed_dict)
            
            if loss_callback is not None:
                loss_callback(loss_val)
                
            # Pack gradients
            grad_flat = self._pack(grads_val)
            
            # Scipy expects float64 usually
            return loss_val.astype(np.float64), grad_flat.astype(np.float64)

        # Get initial values
        initial_vars = session.run(self._vars)
        initial_params = self._pack(initial_vars)
        
        # Run optimization
        result = scipy.optimize.minimize(
            fun=loss_and_grad,
            x0=initial_params,
            jac=True,
            method=self._method,
            options=self._options
        )
        
        # Assign final values
        final_params = result.x
        unpacked_final_params = self._unpack(final_params)
        feed_dict_assign = dict(zip(self._placeholders, unpacked_final_params))
        session.run(self._assign_ops, feed_dict=feed_dict_assign)

        if fetches is not None:
            fetch_vals = session.run(fetches, feed_dict=feed_dict)
            # print("Optimization finished. Final fetches:", fetch_vals)
