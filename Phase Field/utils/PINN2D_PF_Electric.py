"""
Electromechanical Phase-Field PINN with Piezoresistivity

Extended from CalculateUPhi to include:
- Electrical potential field (voltage)
- Piezoresistive conductivity (strain-dependent)
- Electrical degradation due to cracks
- Current conservation loss term

Based on: Quinteros et al., "Electromechanical phase-field fracture modelling 
of piezoresistive CNT-based composites", CMAME 407 (2023)
"""

import tensorflow.compat.v1 as tf
import numpy as np
import time
from utils.tf_scipy_optimizer import ScipyOptimizerInterface


class CalculateUPhiElectric:
    """
    Three-field DEM solver:
    - u, v: displacement fields
    - φ (phi): phase-field (damage)
    - ϕ (voltage): electrical potential
    
    Coupling:
    - Mechanical → Phase-field: strain energy drives damage
    - Phase-field → Mechanical: damage degrades stiffness (h1)
    - Mechanical → Electrical: piezoresistivity (strain changes conductivity)
    - Phase-field → Electrical: cracks block current (h2)
    """
    
    def __init__(self, model, NN_param):
        
        # ===================== MECHANICAL PARAMETERS =====================
        self.E = model['E']
        self.nu = model['nu']
        
        # Plane strain stiffness matrix
        self.c11 = self.E * (1 - self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
        self.c22 = self.c11
        self.c12 = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.c21 = self.c12
        self.c33 = self.E / (2 * (1 + self.nu))
        
        self.lamda = self.E * self.nu / ((1 - 2 * self.nu) * (1 + self.nu))
        self.mu = 0.5 * self.E / (1 + self.nu)
        
        # ===================== PHASE-FIELD PARAMETERS =====================
        self.cEnerg = model.get('Gc', 2.7)  # Critical energy release rate
        self.B = model.get('B', 92)
        self.l = model['l']  # Length scale
        
        # ===================== ELECTRICAL PARAMETERS =====================
        # Base electrical conductivity (S/m)
        self.sigma_0 = model.get('sigma_0', 1e-6)
        
        # Piezoresistivity coefficients (dimensionless)
        # These relate strain to relative resistivity change: Δρ/ρ₀ = Π:ε
        self.lambda_11 = model.get('lambda_11', 2.0)  # Longitudinal
        self.lambda_12 = model.get('lambda_12', 0.5)  # Transverse
        self.lambda_44 = (self.lambda_11 - self.lambda_12) / 2  # Shear
        
        # Electrical degradation function parameters (h2)
        self.k_elec = model.get('k_elec', 50.0)
        self.n_elec = model.get('n_elec', 6.0)
        
        # Applied voltage (for boundary conditions)
        self.V_applied = model.get('V_applied', 1.0)
        
        # Electrical loss weight
        self.alpha_elec = model.get('alpha_elec', 0.1)
        
        # Regularization for numerical stability
        self.eps_reg = 1e-7
        
        # ===================== DOMAIN BOUNDS =====================
        self.lb = model['lb']
        self.ub = model['ub']
        
        # ===================== NEURAL NETWORK =====================
        self.layers = NN_param['layers']  # Should be [2, 50, 50, 50, 4] for 4 outputs
        self.data_type = NN_param['data_type']
        
        # Cast all numerical constants to the correct dtype for TensorFlow
        self._cast_constants_to_dtype()
        self.weights, self.biases = self.initialize_NN(self.layers)
        
        # ===================== TF PLACEHOLDERS =====================
        self.x_f_tf = tf.placeholder(self.data_type)
        self.y_f_tf = tf.placeholder(self.data_type)
        self.wt_f_tf = tf.placeholder(self.data_type)
        self.hist_tf = tf.placeholder(self.data_type)
        self.vdelta_tf = tf.placeholder(self.data_type)
        
        # ===================== BUILD COMPUTATIONAL GRAPH =====================
        # Mechanical + Phase-field + Electrical energy
        self.energy_u_pred, self.energy_phi_pred, self.energy_elec_pred, self.hist_pred = \
            self.net_energy(self.x_f_tf, self.y_f_tf, self.hist_tf, self.vdelta_tf)
        
        # Field predictions
        self.u_pred, self.v_pred = self.net_uv(self.x_f_tf, self.y_f_tf, self.vdelta_tf)
        self.phi_pred = self.net_phi(self.x_f_tf, self.y_f_tf)
        self.voltage_pred = self.net_voltage(self.x_f_tf, self.y_f_tf)
        
        # Current density for postprocessing
        self.Jx_pred, self.Jy_pred = self.net_current_density(
            self.x_f_tf, self.y_f_tf, self.vdelta_tf)
        
        # Traction for mechanical postprocessing
        self.traction_pred = self.net_traction(self.x_f_tf, self.y_f_tf, self.vdelta_tf)
        self.f_u_pred, self.f_v_pred = self.net_f(self.x_f_tf, self.y_f_tf, self.vdelta_tf)
        
        # ===================== LOSS FUNCTION =====================
        self.loss_energy_u = tf.reduce_sum(self.energy_u_pred * self.wt_f_tf)
        self.loss_energy_phi = tf.reduce_sum(self.energy_phi_pred * self.wt_f_tf)
        self.loss_energy_elec = tf.reduce_sum(self.energy_elec_pred * self.wt_f_tf)
        
        # Total loss: mechanical + fracture + electrical
        self.loss = (self.loss_energy_u + 
                     self.loss_energy_phi + 
                     self.alpha_elec * self.loss_energy_elec)
        
        # ===================== OPTIMIZER =====================
        self.lbfgs_buffer = []
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(
            self.loss, var_list=[self.weights, self.biases])
        
        # ===================== TF SESSION =====================
        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False))
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver()

    # ======================== NEURAL NETWORK METHODS ========================
    
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=self.data_type), 
                           dtype=self.data_type)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], 
                          stddev=xavier_stddev, dtype=self.data_type), 
                          dtype=self.data_type)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def _cast_constants_to_dtype(self):
        """
        Cast all numerical constants to TensorFlow constants with correct dtype.
        This prevents float32/float64 mismatch errors.
        """
        # Electrical parameters as TF constants
        self.sigma_0_tf = tf.constant(self.sigma_0, dtype=self.data_type)
        self.lambda_11_tf = tf.constant(self.lambda_11, dtype=self.data_type)
        self.lambda_12_tf = tf.constant(self.lambda_12, dtype=self.data_type)
        self.lambda_44_tf = tf.constant(self.lambda_44, dtype=self.data_type)
        self.k_elec_tf = tf.constant(self.k_elec, dtype=self.data_type)
        self.n_elec_tf = tf.constant(self.n_elec, dtype=self.data_type)
        self.V_applied_tf = tf.constant(self.V_applied, dtype=self.data_type)
        self.eps_reg_tf = tf.constant(self.eps_reg, dtype=self.data_type)
        
        # Numerical constants
        self.one_tf = tf.constant(1.0, dtype=self.data_type)
        self.half_tf = tf.constant(0.5, dtype=self.data_type)
        self.two_tf = tf.constant(2.0, dtype=self.data_type)

    # ======================== MECHANICAL FIELD METHODS ========================
    
    def net_uv(self, x, y, vdelta):
        """Displacement field with exact BCs"""
        X = tf.concat([x, y], 1)
        output = self.neural_net(X, self.weights, self.biases)
        uNN = output[:, 0:1]
        vNN = output[:, 1:2]
        
        # Exact satisfaction of Dirichlet BCs
        # u = 0 at x=0 and x=1
        # v = 0 at y=0, v = vdelta at y=1
        u = (1 - x) * x * uNN
        v = y * (y - 1) * vNN + y * vdelta
        
        return u, v

    def net_phi(self, x, y):
        """Phase-field (damage)"""
        X = tf.concat([x, y], 1)
        output = self.neural_net(X, self.weights, self.biases)
        phi = output[:, 2:3]
        return phi

    # ======================== ELECTRICAL FIELD METHODS ========================
    
    def net_voltage(self, x, y):
        """
        Electrical potential field with exact BCs.
        BC: ϕ = 0 at y=0 (bottom electrode), ϕ = V_applied at y=1 (top electrode)
        """
        X = tf.concat([x, y], 1)
        output = self.neural_net(X, self.weights, self.biases)
        voltageNN = output[:, 3:4]  # 4th output
        
        # Exact BC satisfaction using distance functions
        # ϕ(y=0) = 0, ϕ(y=1) = V_applied
        # Linear interpolation + NN correction term
        voltage = y * self.V_applied_tf + y * (self.one_tf - y) * voltageNN
        
        return voltage
    
    def net_electric_field(self, x, y):
        """
        Electric field: E = -∇ϕ
        """
        voltage = self.net_voltage(x, y)
        E_x = -tf.gradients(voltage, x)[0]
        E_y = -tf.gradients(voltage, y)[0]
        return E_x, E_y
    
    def h1_degradation(self, phi):
        """
        Mechanical degradation function: h₁(φ) = (1-φ)²
        Degrades stiffness as damage increases.
        """
        return (self.one_tf - phi)**2 + self.eps_reg_tf
    
    def h2_degradation(self, phi):
        """
        Electrical conductivity degradation function (Eq. 44 from paper):
        h₂(φ,k,n) = (1 - exp(-k(1-φ)ⁿ)) / (1 - exp(-k))
        
        - φ = 0 (intact): h₂ ≈ 1 (full conductivity)
        - φ = 1 (cracked): h₂ ≈ 0 (no conductivity, current blocked)
        """
        numerator = self.one_tf - tf.exp(-self.k_elec_tf * (self.one_tf - phi)**self.n_elec_tf)
        denominator = self.one_tf - tf.exp(-self.k_elec_tf) + self.eps_reg_tf
        return numerator / denominator + self.eps_reg_tf
    
    def net_piezoresistivity(self, u_x, v_y, u_xy):
        """
        Compute strain-dependent conductivity using piezoresistivity tensor.
        
        Piezoresistivity relation (Eq. 27):
        Δρ₁₁/ρ₀ = λ₁₁*ε₁₁ + λ₁₂*ε₂₂
        Δρ₂₂/ρ₀ = λ₁₂*ε₁₁ + λ₁₁*ε₂₂
        
        Effective conductivity: σ_eff = σ₀ / (1 + Δρ/ρ₀)
        """
        # Strains
        eps_11 = u_x
        eps_22 = v_y
        eps_12 = self.half_tf * u_xy
        
        # Relative resistivity changes
        delta_rho_11 = self.lambda_11_tf * eps_11 + self.lambda_12_tf * eps_22
        delta_rho_22 = self.lambda_12_tf * eps_11 + self.lambda_11_tf * eps_22
        delta_rho_12 = self.lambda_44_tf * self.two_tf * eps_12
        
        # Effective conductivity (tensor components)
        # σ_eff,ij = σ₀ / (1 + Δρ_ij/ρ₀)
        sigma_eff_11 = self.sigma_0_tf / (self.one_tf + delta_rho_11 + self.eps_reg_tf)
        sigma_eff_22 = self.sigma_0_tf / (self.one_tf + delta_rho_22 + self.eps_reg_tf)
        sigma_eff_12 = self.sigma_0_tf / (self.one_tf + tf.abs(delta_rho_12) + self.eps_reg_tf)
        
        return sigma_eff_11, sigma_eff_22, sigma_eff_12
    
    def net_current_density(self, x, y, vdelta):
        """
        Current density: J = h₂(φ) * σ_eff(ε) * E
        
        This combines:
        1. Crack degradation (h₂): cracks block current
        2. Piezoresistivity (σ_eff): strain changes conductivity
        3. Electric field (E): voltage gradient drives current
        """
        # Get displacement and strain
        u, v = self.net_uv(x, y, vdelta)
        u_x = tf.gradients(u, x)[0]
        v_y = tf.gradients(v, y)[0]
        u_y = tf.gradients(u, y)[0]
        v_x = tf.gradients(v, x)[0]
        u_xy = u_y + v_x
        
        # Get phase-field (damage)
        phi = self.net_phi(x, y)
        
        # Get electric field
        E_x, E_y = self.net_electric_field(x, y)
        
        # Get degradation and piezoresistivity
        h2 = self.h2_degradation(phi)
        sigma_11, sigma_22, sigma_12 = self.net_piezoresistivity(u_x, v_y, u_xy)
        
        # Current density components: J = h₂ * σ_eff * E
        J_x = h2 * sigma_11 * E_x
        J_y = h2 * sigma_22 * E_y
        
        return J_x, J_y

    # ======================== HISTORY FUNCTION ========================
    
    def net_hist(self, x, y):
        """Initial crack history function"""
        shape = tf.shape(x)
        self.crackTip = 0.5
        init_hist = tf.zeros((shape[0], shape[1]), dtype=self.data_type)
        dist = tf.where(x > self.crackTip, 
                       tf.sqrt((x - 0.5)**2 + (y - 0.5)**2), 
                       tf.abs(y - 0.5))
        init_hist = tf.where(dist < 0.5 * self.l, 
                            self.B * self.cEnerg * 0.5 * (1 - (2 * dist / self.l)) / self.l, 
                            init_hist)
        return init_hist
    
    def net_update_hist(self, x, y, u_x, v_y, u_xy, hist):
        """Update strain history for irreversible damage"""
        init_hist = self.net_hist(x, y)
        
        # Computing tensile strain energy using spectral decomposition
        u_xy = 0.5 * u_xy
        M = tf.sqrt((u_x - v_y)**2 + 4 * (u_xy**2))
        lambda1 = 0.5 * (u_x + v_y) + 0.5 * M
        lambda2 = 0.5 * (u_x + v_y) - 0.5 * M
        
        eigSum = lambda1 + lambda2
        sEnergy_pos = (0.125 * self.lamda * (eigSum + tf.abs(eigSum))**2 + 
                      0.25 * self.mu * ((lambda1 + tf.abs(lambda1))**2 + 
                                        (lambda2 + tf.abs(lambda2))**2))
        
        hist_temp = tf.maximum(init_hist, sEnergy_pos)
        hist = tf.maximum(hist, hist_temp)
        
        return hist

    # ======================== ENERGY COMPUTATION ========================
    
    def net_energy(self, x, y, hist, vdelta):
        """
        Compute the three energy components for the loss function:
        1. Elastic strain energy (mechanical)
        2. Fracture energy (phase-field)
        3. Electrical residual energy (current conservation)
        """
        u, v = self.net_uv(x, y, vdelta)
        phi = self.net_phi(x, y)
        
        # Mechanical degradation
        g = self.h1_degradation(phi)
        
        # Phase-field gradients (for fracture energy)
        phi_x = tf.gradients(phi, x)[0]
        phi_y = tf.gradients(phi, y)[0]
        phi_xx = tf.gradients(phi_x, x)[0]
        phi_yy = tf.gradients(phi_y, y)[0]
        
        nabla = phi_x**2 + phi_y**2
        laplacian = (phi_xx + phi_yy)**2
        
        # Strain components
        u_x = tf.gradients(u, x)[0]
        v_y = tf.gradients(v, y)[0]
        u_y = tf.gradients(u, y)[0]
        v_x = tf.gradients(v, x)[0]
        u_xy = u_y + v_x
        
        # Update history function
        hist = self.net_update_hist(x, y, u_x, v_y, u_xy, hist)
        
        # Stress components
        sigmaX = self.c11 * u_x + self.c12 * v_y
        sigmaY = self.c21 * u_x + self.c22 * v_y
        tauXY = self.c33 * u_xy
        
        # ============ ELASTIC STRAIN ENERGY ============
        energy_u = 0.5 * g * (sigmaX * u_x + sigmaY * v_y + tauXY * u_xy)
        
        # ============ FRACTURE ENERGY (4th order) ============
        energy_phi = (0.5 * self.cEnerg * 
                     (phi**2 / self.l + self.l * nabla + 0.5 * self.l**3 * laplacian) + 
                     g * hist)
        
        # ============ ELECTRICAL ENERGY (Power Dissipation) ============
        # Variational formulation: minimize -∫J·E dV = ∫σ_eff|∇ϕ|² dV
        # This represents Joule heating/power dissipation
        # See: weak form of current conservation (Eq. 36 in paper)
        
        E_x, E_y = self.net_electric_field(x, y)
        
        # Get piezoresistive conductivity
        sigma_11, sigma_22, _ = self.net_piezoresistivity(u_x, v_y, u_xy)
        
        # Electrical degradation due to cracks
        h2 = self.h2_degradation(phi)
        
        # Power dissipation: P = J·E = h₂·σ·E·E = h₂·σ·|∇ϕ|²
        # We MINIMIZE this (current takes path of least resistance)
        energy_elec = h2 * (sigma_11 * E_x**2 + sigma_22 * E_y**2)
        
        return energy_u, energy_phi, energy_elec, hist

    # ======================== AUXILIARY METHODS ========================
    
    def net_f(self, x, y, vdelta):
        """Momentum residual for postprocessing"""
        u, v = self.net_uv(x, y, vdelta)
        phi = self.net_phi(x, y)
        
        g = self.h1_degradation(phi)
        
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_yx = tf.gradients(v_y, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_yy = tf.gradients(u_y, y)[0]
        v_x = tf.gradients(v, x)[0]
        v_xy = tf.gradients(v_x, y)[0]

        f_u = -g * (self.c11 * u_xx + self.c12 * v_yx + self.c33 * u_yy + self.c33 * v_xy)

        u_xy = tf.gradients(u_x, y)[0]
        v_yy = tf.gradients(v_y, y)[0]
        u_yx = tf.gradients(u_y, x)[0]
        v_xx = tf.gradients(v_x, x)[0]

        f_v = -g * (self.c21 * u_xy + self.c22 * v_yy + self.c33 * u_yx + self.c33 * v_xx)
        
        return f_u, f_v
    
    def net_traction(self, x, y, vdelta):
        """Compute traction on top boundary"""
        u, v = self.net_uv(x, y, vdelta)
        u_x = tf.gradients(u, x)[0]
        v_y = tf.gradients(v, y)[0]
        
        traction = self.c21 * u_x + self.c22 * v_y
        return traction

    # ======================== TRAINING ========================
    
    def callback(self, loss):
        self.lbfgs_buffer = np.append(self.lbfgs_buffer, loss)
        
    def train(self, X_f, v_delta, hist_f, nIter, nIterLBFGS):
        tf_dict = {
            self.x_f_tf: X_f[:, 0:1], 
            self.y_f_tf: X_f[:, 1:2], 
            self.wt_f_tf: X_f[:, 2:3],
            self.hist_tf: hist_f, 
            self.vdelta_tf: v_delta
        }

        start_time = time.time()
        self.loss_adam_buff = np.zeros(nIter)
        
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            self.loss_adam_buff[it] = loss_value
            
            if it % 100 == 0:
                elapsed = time.time() - start_time
                energy_u_val = self.sess.run(self.loss_energy_u, tf_dict)
                energy_phi_val = self.sess.run(self.loss_energy_phi, tf_dict)
                energy_elec_val = self.sess.run(self.loss_energy_elec, tf_dict)
                
                print('It: %d, Loss: %.3e, E_u: %.3e, E_phi: %.3e, E_elec: %.3e, Time: %.2f' %
                      (it, loss_value, energy_u_val, energy_phi_val, energy_elec_val, elapsed))
                start_time = time.time()
        
        # L-BFGS optimization
        self.optimizer = ScipyOptimizerInterface(
            self.loss,
            method='L-BFGS-B',
            options={
                'maxiter': nIterLBFGS,
                'maxfun': nIterLBFGS,
                'maxcor': 100,
                'maxls': 50,
                'ftol': 1.0 * np.finfo(float).eps
            }
        )
        
        self.optimizer.minimize(
            self.sess,
            feed_dict=tf_dict,
            fetches=[self.loss],
            loss_callback=self.callback
        )

    # ======================== PREDICTION ========================
    
    def predict(self, X_star, Hist_star, v_delta):
        tf_dict = {
            self.x_f_tf: X_star[:, 0:1], 
            self.y_f_tf: X_star[:, 1:2],
            self.hist_tf: Hist_star[:, 0:1], 
            self.vdelta_tf: v_delta
        }

        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        phi_star = self.sess.run(self.phi_pred, tf_dict)
        voltage_star = self.sess.run(self.voltage_pred, tf_dict)
        energy_u_star = self.sess.run(self.energy_u_pred, tf_dict)
        energy_phi_star = self.sess.run(self.energy_phi_pred, tf_dict)
        energy_elec_star = self.sess.run(self.energy_elec_pred, tf_dict)
        hist_star = self.sess.run(self.hist_pred, tf_dict)
        
        return u_star, v_star, phi_star, voltage_star, energy_u_star, energy_phi_star, energy_elec_star, hist_star
    
    def predict_current(self, X_star, v_delta):
        """Predict current density field"""
        tf_dict = {
            self.x_f_tf: X_star[:, 0:1], 
            self.y_f_tf: X_star[:, 1:2],
            self.vdelta_tf: v_delta
        }
        Jx_star = self.sess.run(self.Jx_pred, tf_dict)
        Jy_star = self.sess.run(self.Jy_pred, tf_dict)
        return Jx_star, Jy_star
    
    def predict_resistance(self, X_electrode_top, X_electrode_bottom, v_delta):
        """
        Compute total resistance between electrodes.
        R = V_applied / I_total
        """
        tf_dict = {
            self.x_f_tf: X_electrode_top[:, 0:1],
            self.y_f_tf: X_electrode_top[:, 1:2],
            self.vdelta_tf: v_delta
        }
        # Integrate current at top electrode
        Jy_top = self.sess.run(self.Jy_pred, tf_dict)
        I_total = np.sum(Jy_top)  # This should be weighted by integration weights
        
        R = self.V_applied / (np.abs(I_total) + 1e-10)
        return R
    
    def predict_phi(self, X_star):
        tf_dict = {
            self.x_f_tf: X_star[:, 0:1], 
            self.y_f_tf: X_star[:, 1:2]
        }
        phi_star = self.sess.run(self.phi_pred, tf_dict)
        return phi_star
    
    def predict_voltage(self, X_star):
        tf_dict = {
            self.x_f_tf: X_star[:, 0:1], 
            self.y_f_tf: X_star[:, 1:2]
        }
        voltage_star = self.sess.run(self.voltage_pred, tf_dict)
        return voltage_star
    
    def predict_f(self, X_star, v_delta):
        tf_dict = {
            self.x_f_tf: X_star[:, 0:1], 
            self.y_f_tf: X_star[:, 1:2], 
            self.vdelta_tf: v_delta
        }
        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        f_v_star = self.sess.run(self.f_v_pred, tf_dict)
        return f_u_star, f_v_star
    
    def predict_traction(self, X_star, v_delta):
        tf_dict = {
            self.x_f_tf: X_star[:, 0:1], 
            self.y_f_tf: X_star[:, 1:2],
            self.vdelta_tf: v_delta
        }
        trac_star = self.sess.run(self.traction_pred, tf_dict)
        return trac_star
    
    def getWeightsBiases(self):
        weights = self.sess.run(self.weights)
        biases = self.sess.run(self.biases)
        return weights, biases

    def save_model(self, path, step):
        self.saver.save(self.sess, path + 'model.ckpt', global_step=step)
