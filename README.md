# Deep Energy Method for Electromechanical Phase Field Fracture

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-1.x%20%2F%202.x%20compat-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A **Deep Energy Method (DEM)** implementation for simulating **electromechanical phase field fracture** with **piezoresistive self-sensing** capabilities. This code enables the study of fracture mechanics in smart materials that can detect damage through electrical resistance changes.

---

## 🎯 Key Features

### 1. Multiphysics Coupling
- **Mechanical**: Phase field fracture with adaptive mesh refinement
- **Electrical**: Piezoresistive response coupled to strain and damage
- **Self-Sensing**: Real-time damage detection via resistance monitoring

### 2. Deep Energy Method
- Neural network minimizes total potential energy
- Hybrid ADAM + L-BFGS optimization for fast, accurate convergence
- Exact boundary condition enforcement via distance functions

### 3. Adaptive Integration
- Automatic mesh refinement near damage zones
- Efficient quadrature with NURBS-based geometry

---

## 📐 Physics Model

Based on **Quinteros et al., CMAME 407 (2023)**: Electromechanical phase field for fracture with self-sensing.

**Total Energy Functional**:
$$\Pi = \int_\Omega \left[ g(\phi)\psi_e + \frac{G_c}{c_0}\left(\frac{\phi^2}{l} + l|\nabla\phi|^2\right) + \alpha\sigma(\phi,\varepsilon)\|\nabla V\|^2 \right] d\Omega$$

| Term | Description |
|------|-------------|
| $g(\phi)\psi_e$ | Degraded elastic strain energy |
| $G_c$ term | Fracture surface energy |
| $\sigma(\phi,\varepsilon)\|\nabla V\|^2$ | Electrical dissipation (Joule heating) |

**Piezoresistive Conductivity**:
$$\sigma = \sigma_0 (1 + \lambda_{11}\varepsilon_{11} + \lambda_{12}\varepsilon_{22}) \cdot f(\phi)$$

where $f(\phi)$ describes how damage reduces electrical conductivity.

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/BBahtiri/electro-fracture-mechanics-PINN.git
cd electro-fracture-mechanics-PINN

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run simulation
cd "DEM-PhaseField-github/Phase Field"
python Tensile_adaptive_4th_electric.py

# Generate visualization GIF from results
python visualize_electrofracture.py --data_dir TensionPlate_Electric
```

---

## 📁 Project Structure

```
DEM-PhaseField-github/
└── Phase Field/
    ├── Tensile_adaptive_4th_electric.py   # Main simulation script
    ├── visualize_electrofracture.py       # GIF visualization generator
    ├── TensionPlate_Electric/             # Results folder (demo data)
    └── utils/
        ├── BezExtr.py                     # NURBS geometry handling
        ├── PINN2D_PF_Electric.py          # Core DEM solver with electrical coupling
        ├── gridPlot2D.py                  # Plotting utilities
        └── tf_scipy_optimizer.py          # L-BFGS optimizer wrapper
```

---

## ⚙️ Configuration

Key parameters in `Tensile_adaptive_4th_electric.py`:

```python
# Mechanical properties
model['E'] = 500.0 * 1e2       # Young's modulus (MPa)
model['nu'] = 0.3              # Poisson's ratio
model['l'] = 0.005             # Phase-field length scale
model['Gc'] = 1.7              # Fracture energy (N/mm)

# Electrical properties (CNT-epoxy composite)
model['sigma_0'] = 1           # Base conductivity (S/m)
model['lambda_11'] = 2.0       # Piezoresistivity coeff (longitudinal)
model['lambda_12'] = 0.5       # Piezoresistivity coeff (transverse)
model['V_applied'] = 1.0       # Applied voltage (V)

# Geometry
model['num_holes'] = 3         # Random holes for stress concentration
model['hole_type'] = 'circle'  # Options: 'circle', 'rect', 'mixed'
```

---

## 📊 Output

The simulation generates:
- **Damage field** ($\phi$) evolution
- **Deformed shape** with displacement magnification
- **Voltage and current density** distributions
- **Resistance history** showing piezoresistive response
- **Energy evolution** (elastic, fracture, electrical)
- **Animated GIF** summarizing all fields

---

## 📚 References

1. **Goswami et al.** (2020) - [Adaptive fourth-order phase field analysis using deep energy minimization](https://www.sciencedirect.com/science/article/abs/pii/S0167844219306858), *Theoretical and Applied Fracture Mechanics*, 107, 102527 — **This implementation is based on this work**
2. **Quinteros et al.** (2023) - "Electromechanical phase field fracture model for self-sensing materials", *Computer Methods in Applied Mechanics and Engineering*, 407
3. **Raissi et al.** (2019) - "Physics-Informed Neural Networks: A deep learning framework for solving forward and inverse problems", *Journal of Computational Physics*

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

## 👤 Author

**Betim Bahtiri and Aamir Dean**
