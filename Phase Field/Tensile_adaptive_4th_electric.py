# Tensile Test with Piezoresistive Monitoring
# Implements the electromechanical phase field to study fracture with self-sensing
# Based on Quinteros et al., CMAME 407 (2023)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
import matplotlib.pyplot as plt
import time
import os
import scipy.io
import imageio  # For GIF creation
tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR)

from utils.gridPlot2D import scatterPlot
from utils.gridPlot2D import genGrid
from utils.gridPlot2D import plotDispStrainEnerg
from utils.gridPlot2D import plotPhiStrainEnerg
from utils.gridPlot2D import plotConvergence
from utils.gridPlot2D import createFolder
from utils.gridPlot2D import plot1dPhi
from utils.gridPlot2D import refineElemVertex
from utils.BezExtr import Geometry2D
from utils.PINN2D_PF_Electric import CalculateUPhiElectric

np.random.seed(1234)
tf.set_random_seed(1234)

# ==================== HOLE GEOMETRY FUNCTIONS ====================
def generate_random_holes(num_holes, L, W, hole_type='circle', seed=42):
    """
    Generate random non-overlapping holes.
    
    Args:
        num_holes: Number of holes to generate
        L, W: Domain dimensions
        hole_type: 'circle', 'rect', or 'mixed'
        seed: Random seed for reproducibility
    
    Returns:
        List of hole dictionaries
    """
    np.random.seed(seed)
    holes = []
    max_attempts = 100
    
    for i in range(num_holes):
        for attempt in range(max_attempts):
            # Determine hole type for this hole
            if hole_type == 'mixed':
                this_type = 'circle' if np.random.rand() > 0.5 else 'rect'
            else:
                this_type = hole_type
            
            if this_type == 'circle':
                r = np.random.uniform(0.01, 0.06)
                cx = np.random.uniform(r + 0.1, L - r - 0.1)
                cy = np.random.uniform(r + 0.15, W - r - 0.15)  # Keep away from BCs
                hole = {'type': 'circle', 'center': (cx, cy), 'radius': r}
            else:  # rect
                w = np.random.uniform(0.06, 0.14)
                h = np.random.uniform(0.04, 0.10)
                cx = np.random.uniform(w/2 + 0.1, L - w/2 - 0.1)
                cy = np.random.uniform(h/2 + 0.15, W - h/2 - 0.15)
                hole = {'type': 'rect', 'center': (cx, cy), 'width': w, 'height': h}
            
            if not check_hole_overlap(hole, holes):
                holes.append(hole)
                print(f"Hole {i+1}: {hole}")
                break
        else:
            print(f"Warning: Could not place hole {i+1} without overlap")
    
    return holes


def check_hole_overlap(new_hole, existing_holes, margin=0.03):
    """Check if new hole overlaps with existing ones."""
    for hole in existing_holes:
        dist = np.sqrt((new_hole['center'][0] - hole['center'][0])**2 + 
                       (new_hole['center'][1] - hole['center'][1])**2)
        min_dist = get_hole_radius(new_hole) + get_hole_radius(hole) + margin
        if dist < min_dist:
            return True
    return False


def get_hole_radius(hole):
    """Get effective radius for overlap checking."""
    if hole['type'] == 'circle':
        return hole['radius']
    else:
        return np.sqrt(hole['width']**2 + hole['height']**2) / 2


def point_in_hole(x, y, hole):
    """Check if point (x,y) is inside a hole."""
    cx, cy = hole['center']
    if hole['type'] == 'circle':
        return (x - cx)**2 + (y - cy)**2 < hole['radius']**2
    else:  # rect
        return (abs(x - cx) < hole['width']/2) and (abs(y - cy) < hole['height']/2)


def filter_points_outside_holes(X, holes):
    """Remove points that fall inside any hole."""
    if len(holes) == 0:
        return X
    mask = np.ones(len(X), dtype=bool)
    for i in range(len(X)):
        x, y = X[i, 0], X[i, 1]
        for hole in holes:
            if point_in_hole(x, y, hole):
                mask[i] = False
                break
    print(f"Filtered {np.sum(~mask)} points inside holes, {np.sum(mask)} remaining")
    return X[mask]


def plot_holes_on_scatter(ax, holes, color='red', alpha=0.5):
    """Add hole visualization to a scatter plot axis."""
    from matplotlib.patches import Circle, Rectangle
    for hole in holes:
        cx, cy = hole['center']
        if hole['type'] == 'circle':
            patch = Circle((cx, cy), hole['radius'], fill=True, 
                          facecolor=color, edgecolor='darkred', alpha=alpha, linewidth=2)
        else:
            w, h = hole['width'], hole['height']
            patch = Rectangle((cx - w/2, cy - h/2), w, h, fill=True,
                             facecolor=color, edgecolor='darkred', alpha=alpha, linewidth=2)
        ax.add_patch(patch)


def scatterPlotWithHoles(X_f, figHeight, figWidth, filename, holes):
    """Scatter plot of collocation points with holes visualized."""
    fig, ax = plt.subplots(figsize=(figWidth, figHeight))
    ax.scatter(X_f[:, 0], X_f[:, 1], s=0.5, c='blue', alpha=0.6)
    plot_holes_on_scatter(ax, holes, color='salmon', alpha=0.7)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.set_title(f'Collocation Points ({len(X_f)} pts, {len(holes)} holes)')
    plt.tight_layout()
    plt.savefig(filename + '.png', dpi=200)
    plt.close()

class Quadrilateral(Geometry2D):
    '''
    Class for defining a quadrilateral domain
    '''
    def __init__(self, quadDom):
        self.quadDom = quadDom
        
        self.x1, self.y1 = self.quadDom[0,:]
        self.x2, self.y2 = self.quadDom[1,:]
        self.x3, self.y3 = self.quadDom[2,:]
        self.x4, self.y4 = self.quadDom[3,:]
        
        geomData = dict()
        geomData['degree_u'] = 1
        geomData['degree_v'] = 1
        geomData['ctrlpts_size_u'] = 2
        geomData['ctrlpts_size_v'] = 2
        geomData['ctrlpts'] = np.array([[self.x1, self.y1, 0], [self.x2, self.y2, 0],
                        [self.x3, self.y3, 0], [self.x4, self.y4, 0]])
        geomData['weights'] = np.array([[1.0], [1.0], [1.0], [1.0]])
        geomData['knotvector_u'] = [0.0, 0.0, 1.0, 1.0]
        geomData['knotvector_v'] = [0.0, 0.0, 1.0, 1.0]
        super().__init__(geomData)
        

class PINN_PF_Electric(CalculateUPhiElectric):
    '''
    Extended class with boundary conditions for tension plate with electrodes
    '''
    def __init__(self, model, NN_param):
        super().__init__(model, NN_param)
        
    def net_uv(self, x, y, vdelta):
        """Displacement field with exact BCs"""
        X = tf.concat([x, y], 1)
        uvphi_volt = self.neural_net(X, self.weights, self.biases)
        uNN = uvphi_volt[:, 0:1]
        vNN = uvphi_volt[:, 1:2]
        
        # BCs: u=0 at x=0,1; v=0 at y=0, v=vdelta at y=1
        u = (1 - x) * x * uNN
        v = y * (y - 1) * vNN + y * vdelta
        
        return u, v
    
    def net_hist(self, x, y):
        """No initial crack - holes create stress concentrations instead"""
        shape = tf.shape(x)
        return tf.zeros((shape[0], shape[1]), dtype=tf.float64)


def plotVoltageField(nPred, xGrid, yGrid, voltage_pred, filename, figHeight, figWidth, holes=None):
    """Plot the electrical potential field using scatter (works with filtered grids)"""
    
    voltage_min = float(np.min(voltage_pred))
    voltage_max = float(np.max(voltage_pred))
    
    fig, ax = plt.subplots(figsize=(figWidth, figHeight))
    
    # Use scatter plot instead of contour for filtered grids
    sc = ax.scatter(xGrid.flatten(), yGrid.flatten(), c=voltage_pred.flatten(), 
                   cmap='coolwarm', s=2, vmin=voltage_min, vmax=voltage_max)
    
    # Add holes if provided
    if holes is not None:
        plot_holes_on_scatter(ax, holes, color='white', alpha=0.9)
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Voltage (V)', fontsize=14)
    ax.set_xlabel('$x$', fontweight='bold', fontsize=14)
    ax.set_ylabel('$y$', fontweight='bold', fontsize=14)
    ax.set_aspect('equal')
    ax.set_title('Electric Potential Field')
    plt.tight_layout()
    plt.savefig(filename + '.png', dpi=300, facecolor='w', edgecolor='w', 
                transparent=False, bbox_inches='tight')
    plt.close()


def plotFieldScatter(xGrid, yGrid, field, title, cbar_label, filename, 
                     figHeight, figWidth, holes=None, cmap='viridis'):
    """Generic scatter plot for any field on filtered grid"""
    
    field_min = float(np.min(field))
    field_max = float(np.max(field))
    
    fig, ax = plt.subplots(figsize=(figWidth, figHeight))
    
    sc = ax.scatter(xGrid.flatten(), yGrid.flatten(), c=field.flatten(), 
                   cmap=cmap, s=2, vmin=field_min, vmax=field_max)
    
    if holes is not None:
        plot_holes_on_scatter(ax, holes, color='white', alpha=0.9)
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(cbar_label, fontsize=14)
    ax.set_xlabel('$x$', fontweight='bold', fontsize=14)
    ax.set_ylabel('$y$', fontweight='bold', fontsize=14)
    ax.set_aspect('equal')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename + '.png', dpi=300)
    plt.close()


def plotCurrentDensity(X_grid, Jx, Jy, filename, figHeight, figWidth):
    """Plot current density magnitude and vectors"""
    fig, axes = plt.subplots(1, 2, figsize=(figWidth * 2, figHeight))
    
    Jx = np.array(Jx).flatten()
    Jy = np.array(Jy).flatten()
    J_mag = np.sqrt(Jx**2 + Jy**2)
    
    # Left: Current magnitude
    ax = axes[0]
    sc = ax.scatter(X_grid[:, 0], X_grid[:, 1], c=J_mag, 
                   cmap='hot', s=1, vmin=0)
    plt.colorbar(sc, ax=ax, label='|J| (A/m²)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Current Density Magnitude')
    ax.set_aspect('equal')
    
    # Right: Current vectors (quiver)
    ax2 = axes[1]
    # Subsample for quiver
    step = max(1, len(X_grid) // 400)
    ax2.scatter(X_grid[:, 0], X_grid[:, 1], c=J_mag, cmap='hot', s=0.5, alpha=0.3)
    Q = ax2.quiver(X_grid[::step, 0], X_grid[::step, 1], 
                   Jx[::step], Jy[::step], 
                   color='blue', alpha=0.7, scale=None)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Current Flow Direction')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(filename + '.pdf', format='pdf')
    plt.savefig(filename + '.png', dpi=200)
    plt.close()


def plotResistanceHistory(steps, resistance, filename, figHeight, figWidth):
    """Plot resistance vs load step"""
    fig, ax = plt.subplots(figsize=(figWidth, figHeight))
    
    R0 = resistance[0]
    deltaR_R0 = (resistance - R0) / R0
    
    ax.plot(steps, deltaR_R0 * 100, 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Load Step')
    ax.set_ylabel('ΔR/R₀ (%)')
    ax.set_title('Relative Resistance Change')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename + '.pdf', format='pdf')
    plt.savefig(filename + '.png', dpi=200)
    plt.close()



def plotGifFrame_OptionB(nPred, xGrid, yGrid, phi_pred, voltage_pred, energy_elec_pred, 
                         Jx, Jy, step_history, resistance_history, energy_history,
                         iStep, v_delta, figHeight, figWidth, model):
    """
    Enhanced Visualization with Professional Dark Theme - Scatter-based for filtered grids
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Circle, Rectangle
    
    # ==================== THEME SETTINGS ====================
    BG_COLOR = '#2d3748'
    PLOT_BG = '#3d4a5c'
    TEXT_COLOR = '#f7fafc'
    ACCENT_BLUE = '#63b3ed'     
    ACCENT_RED = '#fc8181'
    ACCENT_ACID = '#d6bcfa'
    GRID_COLOR = '#4a5568'
    
    holes = model.get('holes', [])
    
    with plt.style.context('dark_background'):
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'text.color': TEXT_COLOR,
            'axes.labelcolor': TEXT_COLOR,
            'figure.facecolor': BG_COLOR,
            'axes.facecolor': PLOT_BG,
            'axes.grid': False,
        })
        
        fig = plt.figure(figsize=(figWidth * 3.5, figHeight * 2.2), facecolor=BG_COLOR)
        gs = GridSpec(2, 6, figure=fig, hspace=0.35, wspace=0.4)
        
        phi_pred_clipped = np.clip(phi_pred, 0, 1)
        
        # Helper for scatter-based field plots
        def plot_scatter_panel(ax, data, cmap, title, label, vmin=None, vmax=None):
            if vmin is None:
                vmin = float(np.min(data))
            if vmax is None:
                vmax = float(np.max(data))
            sc = ax.scatter(xGrid.flatten(), yGrid.flatten(), c=data.flatten(),
                           cmap=cmap, s=3, vmin=vmin, vmax=vmax)
            # Draw holes
            for hole in holes:
                cx, cy = hole['center']
                if hole['type'] == 'circle':
                    patch = Circle((cx, cy), hole['radius'], fill=True, 
                                  facecolor=BG_COLOR, edgecolor='white', linewidth=1.5)
                else:
                    w, h = hole['width'], hole['height']
                    patch = Rectangle((cx - w/2, cy - h/2), w, h, fill=True,
                                     facecolor=BG_COLOR, edgecolor='white', linewidth=1.5)
                ax.add_patch(patch)
            cbar = plt.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(label, fontsize=12, color=TEXT_COLOR)
            cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
            ax.set_title(title, fontsize=14, fontweight='bold', color=TEXT_COLOR, pad=10)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
        
        # ========== Panel 1: Damage Field ==========
        ax1 = fig.add_subplot(gs[0, 0:2])
        plot_scatter_panel(ax1, phi_pred_clipped, 'gray', 'Damage Field', 'Damage $\\phi$', 0, 1)
        
        # ========== Panel 2: Voltage Field ==========
        ax2 = fig.add_subplot(gs[0, 2:4])
        plot_scatter_panel(ax2, voltage_pred, 'plasma', 'Electric Potential', 'Voltage (V)', 
                          0, model['V_applied'])
        
        # ========== Panel 3: Joule Heating ==========
        ax3 = fig.add_subplot(gs[0, 4:6])
        plot_scatter_panel(ax3, energy_elec_pred, 'inferno', 'Joule Heating', 'Power Density')
        
        # ========== Panel 4: Resistance History ==========
        ax4 = fig.add_subplot(gs[1, 0:3])
        ax4.set_facecolor(PLOT_BG)
        if len(resistance_history) > 0:
            R0 = resistance_history[0]
            if R0 > 0:
                deltaR = [(r - R0) / R0 * 100 for r in resistance_history]
            else:
                deltaR = resistance_history
            ax4.plot(step_history, deltaR, color=ACCENT_BLUE, marker='o', 
                    linewidth=2.5, markersize=6, label='$\\Delta R$')
            ax4.axvline(x=iStep, color=ACCENT_RED, linestyle='--', alpha=0.8)
            ax4.set_xlabel('Load Step', fontsize=12)
            ax4.set_ylabel('$\\Delta R/R_0$ (%)', fontsize=12)
            ax4.set_title('Piezoresistive Response', fontsize=14, fontweight='bold', pad=10)
        
        # ========== Panel 5: Energy Evolution ==========
        ax5 = fig.add_subplot(gs[1, 3:6])
        ax5.set_facecolor(PLOT_BG)
        if len(step_history) > 0:
            h = np.array(energy_history)
            ax5.semilogy(step_history, h[:, 0], color='#68d391', marker='o', label='Elastic', linewidth=2)
            ax5.semilogy(step_history, h[:, 1], color='#f6ad55', marker='^', label='Fracture', linewidth=2)
            ax5.semilogy(step_history, h[:, 2], color='#d53f8c', marker='s', label='Electrical', linewidth=2)
            ax5.axvline(x=iStep, color=ACCENT_RED, linestyle='--', alpha=0.8)
            ax5.set_xlabel('Load Step', fontsize=12)
            ax5.set_ylabel('Energy (Log Scale)', fontsize=12)
            ax5.set_title('System Energy Evolution', fontsize=14, fontweight='bold', pad=10)
            leg = ax5.legend(loc='upper left', frameon=True, facecolor=PLOT_BG, edgecolor=GRID_COLOR)
            for text in leg.get_texts():
                text.set_color(TEXT_COLOR)
            
        fig.suptitle(f'Electro-Fracture Mechanics: Step {iStep}', 
                     fontsize=20, fontweight='bold', color=TEXT_COLOR, y=0.98)
        
        filename = f'GIF_OptionB_{iStep:04d}'
        plt.savefig(filename + '.png', dpi=150, facecolor=BG_COLOR, edgecolor='none', bbox_inches='tight')
        plt.close()
        
        return filename + '.png'


if __name__ == "__main__":
    
    originalDir = os.getcwd()
    foldername = 'TensionPlate_Electric'    
    createFolder('./' + foldername + '/')
    os.chdir(os.path.join(originalDir, './' + foldername + '/'))
    
    figHeight = 5
    figWidth = 5
    nSteps = 35
    deltaV = 8e-4  # Displacement increment per step
    
    # ==================== MODEL PARAMETERS ====================
    model = dict()
    
    # Mechanical properties
    model['E'] = 500.0 * 1e2 # Young's modulus (MPa)
    model['nu'] = 0.3   # Poisson's ratio
    model['L'] = 1.0    # Plate length
    model['W'] = 1.0    # Plate width
    model['l'] = 0.005 # Phase-field length scale
    model['Gc'] = 1.7   # Fracture energy (N/mm)
    model['B'] = 92           # History function parameter
    
    # Electrical properties (CNT-epoxy composite)
    model['sigma_0'] = 1      # Base conductivity (S/m)
    model['lambda_11'] = 2.0     # Piezoresistivity coeff (longitudinal)
    model['lambda_12'] = 0.5     # Piezoresistivity coeff (transverse)
    model['k_elec'] = 50.0       # Electrical degradation k
    model['n_elec'] = 6.0        # Electrical degradation n
    model['V_applied'] = 1.0     # Applied voltage (V)
    model['alpha_elec'] = 0.1    # Electrical loss weight
    
    # Domain bounds
    model['lb'] = np.array([0.0, 0.0])
    model['ub'] = np.array([model['L'], model['W']])
    
    # ==================== HOLE CONFIGURATION ====================
    # Options: 'circle', 'rect', or 'mixed'
    model['num_holes'] = 3
    model['hole_type'] = 'circle'  # Change to 'rect' or 'mixed' as needed
    model['holes'] = generate_random_holes(
        model['num_holes'], model['L'], model['W'], 
        hole_type=model['hole_type'], seed=111
    )

    # ==================== NEURAL NETWORK ====================s
    NN_param = dict()
    # 4 outputs: u, v, φ, ϕ (displacement, displacement, damage, voltage)
    NN_param['layers'] = [2, 50, 50, 50, 4]
    NN_param['data_type'] = tf.float64
    
    # ==================== GEOMETRY ====================
    domainCorners = np.array([[0, 0], [model['W'], 0.], 
                              [0, model['L']], [model['W'], model['L']]])
    myQuad = Quadrilateral(domainCorners)

    numElemU = 40
    numElemV = 40
    numGauss = 4
    maxLevel = 4
    maxInnerIter = 3
    phiRefThresh = 0.1

    vertex = myQuad.genElemList(numElemU, numElemV)
    xPhys, yPhys, wgtsPhys = myQuad.getElemIntPts(vertex, numGauss)
    X_f = np.concatenate((xPhys, yPhys, wgtsPhys), axis=1)
    
    # Filter out points inside holes
    X_f = filter_points_outside_holes(X_f, model['holes'])
    hist_f = np.transpose(np.array([np.zeros((X_f.shape[0]), dtype=np.float64)]))
    
    filename = 'Training_scatter'
    scatterPlotWithHoles(X_f, figHeight, figWidth, filename, model['holes'])
    
    # Prediction mesh - uniform grid (no crack-centered refinement)
    nPredX, nPredY = 100, 100  # Uniform grid resolution
    xGrid_1d = np.linspace(0, model['L'], nPredX)
    yGrid_1d = np.linspace(0, model['W'], nPredY)
    xGrid_2d, yGrid_2d = np.meshgrid(xGrid_1d, yGrid_1d)
    xGrid = xGrid_2d.flatten()[:, np.newaxis]
    yGrid = yGrid_2d.flatten()[:, np.newaxis]
    Grid = np.concatenate([xGrid, yGrid], axis=1)
    hist_grid = np.zeros((len(Grid), 1), dtype=np.float64)
    
    # Add extra refinement around holes
    for hole in model['holes']:
        cx, cy = hole['center']
        if hole['type'] == 'circle':
            r = hole['radius']
        else:
            r = max(hole['width'], hole['height']) / 2
        
        # Add dense points in ring around hole
        n_angular = 30
        n_radial = 8
        for ri in range(n_radial):
            r_ring = r * (1.0 + 0.15 * (ri + 1))  # Rings from 1.15r to 2.2r
            for ai in range(n_angular):
                theta = 2 * np.pi * ai / n_angular
                px = cx + r_ring * np.cos(theta)
                py = cy + r_ring * np.sin(theta)
                if 0 < px < model['L'] and 0 < py < model['W']:
                    Grid = np.vstack([Grid, [px, py]])
                    xGrid = np.vstack([xGrid, [[px]]])
                    yGrid = np.vstack([yGrid, [[py]]])
                    hist_grid = np.vstack([hist_grid, [[0.0]]])
    
    print(f"Prediction grid before hole filtering: {len(Grid)} points")
    
    # Filter prediction grid to exclude holes
    mask_grid = np.ones(len(Grid), dtype=bool)
    for i in range(len(Grid)):
        for hole in model['holes']:
            if point_in_hole(Grid[i, 0], Grid[i, 1], hole):
                mask_grid[i] = False
                break
    Grid = Grid[mask_grid]
    xGrid = xGrid[mask_grid]
    yGrid = yGrid[mask_grid]
    hist_grid = hist_grid[mask_grid]
    print(f"Prediction grid after hole filtering: {len(Grid)} points")
    
    filename = 'Prediction_scatter'
    scatterPlotWithHoles(Grid, figHeight, figWidth, filename, model['holes'])

    phi_pred_old = hist_grid
    
    # ==================== CREATE MODEL ====================
    modelNN = PINN_PF_Electric(model, NN_param)    
    num_train_its = 500
    num_lbfgs_its = 500
    
    # Storage for resistance history
    resistance_history = []
    step_history = []
    energy_history = []  # For Option B: [Eu, Ephi, Eelec]
    simulation_data = []  # Store data for GIF visualization
    
    for iStep in range(0, nSteps):
        
        v_delta = deltaV * iStep        
        keepTraining = 1
        miter = 0
        
        start_time = time.time()

        while (keepTraining > 0) and (miter < maxInnerIter):
            
            miter = miter + 1
            print('Inner Iteration: %d' % (miter))
            
            modelNN.train(X_f, v_delta, hist_f, num_train_its, num_lbfgs_its)
            
            # Predict and update history
            u_pred, v_pred, phi_f, voltage_f, _, _, _, hist_f = modelNN.predict(
                X_f[:, 0:2], hist_f, v_delta)
            f_u, f_v = modelNN.predict_f(X_f[:, 0:2], v_delta)
            
            res_err = np.sqrt(f_u**2 + f_v**2)
            numElem = len(vertex)
            errElem = np.zeros(numElem)
            for iElem in range(numElem):
                ptIndStart = iElem * numGauss**2
                ptIndEnd = (iElem + 1) * numGauss**2
                errElem[iElem] = np.sum(res_err[ptIndStart:ptIndEnd] * 
                                       X_f[ptIndStart:ptIndEnd, 2])
            
            # Element refinement (same as original)
            N = 10
            ntop = int(np.round(numElem * N / 100))
            sort_err_ind = np.argsort(-errElem, axis=0)
            elem_refine_residual = np.squeeze(sort_err_ind[0:ntop])
            
            level = vertex[elem_refine_residual, 4]
            index_ref_residual = np.where(level < maxLevel)[0]
            elem_refine_residual = elem_refine_residual[index_ref_residual]
            
            index = np.where(phi_f >= phiRefThresh)
            elem_refine = np.floor(index[0] / numGauss**2).astype(int)
            elem_refine = np.unique(elem_refine)
            
            level = vertex[elem_refine, 4]
            index_ref_phi = np.where(level < maxLevel)[0]
            elem_refine = elem_refine[index_ref_phi]
            
            index_ref = np.union1d(elem_refine, elem_refine_residual)
            
            if (len(elem_refine) > 0) and (iStep == 0):
                print("Number of elements refining: ", len(index_ref))
                
                vertex = refineElemVertex(vertex, index_ref)
                xPhys, yPhys, wgtsPhys = myQuad.getElemIntPts(vertex, numGauss)
                X_f = np.concatenate((xPhys, yPhys, wgtsPhys), axis=1)
                hist_f = np.transpose(np.array([np.zeros((X_f.shape[0]), dtype=np.float64)]))
                
                u_pred, v_pred, phi_pred, voltage_pred, elas_energy_pred, frac_energy_pred, energy_elec_pred, hist_grid = \
                    modelNN.predict(Grid, hist_grid, v_delta)
                _, _, _, _, _, _, _, hist_f = modelNN.predict(X_f[:, 0:2], hist_f, v_delta)
                
                filename = 'Refine_scatter_' + str(iStep) + '_Miter_' + str(miter)
                scatterPlot(X_f, figHeight, figWidth, filename)
            else:
                keepTraining = 0
        
        elapsed = time.time() - start_time
        print('Training time: %.4f' % (elapsed))
        
        # ==================== PREDICTIONS ====================
        # ==================== PREDICTIONS ====================
        u_pred, v_pred, phi_pred, voltage_pred, elas_energy_pred, frac_energy_pred, energy_elec_pred, hist_grid = \
            modelNN.predict(Grid, hist_grid, v_delta)
        
        # Predict current density
        Jx_pred, Jy_pred = modelNN.predict_current(Grid, v_delta)
        
        phi_pred = np.maximum(phi_pred, phi_pred_old)
        phi_pred_old = phi_pred
        
        # ==================== COMPUTE RESISTANCE ====================
        # Get top electrode points (y close to 1)
        top_mask = Grid[:, 1] > 0.95
        if np.any(top_mask):
            R = modelNN.predict_resistance(Grid[top_mask], Grid[~top_mask], v_delta)
        else:
            R = model['V_applied'] / (np.abs(np.sum(Jy_pred)) + 1e-10)
        
        resistance_history.append(R)
        step_history.append(iStep)
        print(f'Step {iStep}: Resistance R = {R:.4e} Ohm')
        
        # Calculate Total Energies for History Plot (Option B)
        # We need to integrate over the domain. The exact way is to use the integration points X_f during training.
        # The model just trained on X_f, so we can evaluate the loss terms (which are integrals).
        tf_dict_energy = {
            modelNN.x_f_tf: X_f[:, 0:1], 
            modelNN.y_f_tf: X_f[:, 1:2], 
            modelNN.wt_f_tf: X_f[:, 2:3],
            modelNN.hist_tf: hist_f, 
            modelNN.vdelta_tf: v_delta
        }
        total_eu = modelNN.sess.run(modelNN.loss_energy_u, tf_dict_energy)
        total_ephi = modelNN.sess.run(modelNN.loss_energy_phi, tf_dict_energy)
        total_eelec = modelNN.sess.run(modelNN.loss_energy_elec, tf_dict_energy)
        energy_history.append([total_eu, total_ephi, total_eelec])
        
        # ==================== SAVE PLOTS ====================
        # Use scatter-based plots for filtered grids with holes
        filename = f'Step_{iStep:04d}_Phi'
        plotFieldScatter(xGrid, yGrid, phi_pred, 'Damage Field $\\phi$', '$\\phi$',
                        filename, figHeight, figWidth, holes=model['holes'], cmap='gray')
        
        filename = f'Step_{iStep:04d}_FracEnergy'
        plotFieldScatter(xGrid, yGrid, frac_energy_pred, 'Fracture Energy', 'Energy',
                        filename, figHeight, figWidth, holes=model['holes'], cmap='hot')
        
        filename_disp = f'Step_{iStep:04d}_Disp_u'
        plotFieldScatter(xGrid, yGrid, u_pred, 'Displacement u', 'u',
                        filename_disp, figHeight, figWidth, holes=model['holes'], cmap='RdBu_r')
        
        filename_disp = f'Step_{iStep:04d}_Disp_v'
        plotFieldScatter(xGrid, yGrid, v_pred, 'Displacement v', 'v',
                        filename_disp, figHeight, figWidth, holes=model['holes'], cmap='RdBu_r')
        
        filename_elas = f'Step_{iStep:04d}_ElasEnergy'
        plotFieldScatter(xGrid, yGrid, elas_energy_pred, 'Elastic Energy', 'Energy',
                        filename_elas, figHeight, figWidth, holes=model['holes'], cmap='viridis')
        
        filename_voltage = f'Step_{iStep:04d}_Voltage'
        plotVoltageField(None, xGrid, yGrid, voltage_pred, filename_voltage, 
                        figHeight, figWidth, holes=model['holes'])
        
        filename_current = f'Step_{iStep:04d}_Current'
        plotCurrentDensity(Grid, Jx_pred.flatten(), Jy_pred.flatten(), 
                          filename_current, figHeight, figWidth)
        
        adam_buff = modelNN.loss_adam_buff
        lbfgs_buff = modelNN.lbfgs_buffer
        plotConvergence(num_train_its, adam_buff, lbfgs_buff, iStep, figHeight, figWidth)
        
        # Save model
        modelNN.save_model('./' + foldername + '/', iStep)
        
        # ==================== STORE DATA FOR GIF ====================
        step_data = {
            'step': int(iStep),
            'v_delta': float(v_delta),
            'phi_pred': phi_pred.flatten().tolist(),
            'voltage_pred': voltage_pred.flatten().tolist(),
            'energy_elec_pred': energy_elec_pred.flatten().tolist(),
            'Jx_pred': Jx_pred.flatten().tolist(),
            'Jy_pred': Jy_pred.flatten().tolist(),
            'u_pred': u_pred.flatten().tolist(),
            'v_pred': v_pred.flatten().tolist(),
            'resistance': float(R),
            'energy_u': float(total_eu),
            'energy_phi': float(total_ephi),
            'energy_elec': float(total_eelec),
        }
        simulation_data.append(step_data)
        
        print('Completed ' + str(iStep + 1) + ' of ' + str(nSteps) + '.')    
    
    # ==================== FINAL PLOTS ====================
    plotResistanceHistory(np.array(step_history), np.array(resistance_history),
                         'Resistance_History', figHeight, figWidth)
    
    # Save resistance data
    scipy.io.savemat('resistance_history.mat', {
        'steps': step_history,
        'resistance': resistance_history,
        'deltaV': deltaV
    })
    
    # ==================== SAVE SIMULATION DATA FOR GIF ====================
    import json
    
    # Prepare complete data for visualization script
    all_data = {
        'steps': simulation_data,
        'xGrid': xGrid.flatten().tolist(),
        'yGrid': yGrid.flatten().tolist(),
        'holes': model['holes'],
        'V_applied': model['V_applied'],
        'model_params': {
            'E': model['E'],
            'nu': model['nu'],
            'L': model['L'],
            'W': model['W'],
            'l': model['l'],
            'Gc': model['Gc'],
        }
    }
    
    data_file = 'simulation_data.json'
    with open(data_file, 'w') as f:
        json.dump(all_data, f)
    print(f'Simulation data saved to {data_file}')
    print(f'To generate GIF, run: python visualize_electrofracture.py --data_dir {foldername}')
    
    os.chdir(originalDir)
    print('Results saved in', foldername)
