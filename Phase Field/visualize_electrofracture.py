#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GIF Visualization for Electro-Fracture Phase Field Simulation.
Loads saved simulation data and generates animated GIFs.

Layout:
- Top row: Damage field (phi) | Deformed shape with crack opening
- Middle row: Voltage | Current density with arrows | Power dissipation
- Bottom row: Resistance evolution

Shows both the phase field damage AND the physical crack opening via displacement.

Usage:
    python visualize_electrofracture.py --data_dir TensionPlate_Electric
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle
from scipy.interpolate import make_interp_spline


def load_simulation_data(data_dir):
    """Load simulation data from JSON file."""
    data_file = os.path.join(data_dir, 'simulation_data.json')
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded simulation data from {data_file}")
    print(f"  Steps: {len(data['steps'])}")
    print(f"  Grid points: {len(data['xGrid'])}")
    print(f"  Holes: {len(data['holes'])}")
    
    # Check if displacement data is available
    if len(data['steps']) > 0 and 'u_pred' in data['steps'][0]:
        print(f"  Displacement data: Available")
    else:
        print(f"  Displacement data: NOT AVAILABLE (deformed view disabled)")
    
    return data


# Crack detection settings
CRACK_DETECTION_FRAME = 28  # Frame where first crack is detected via piezoresistivity
PAUSE_DURATION_SECONDS = 4  # How long to pause at crack detection frame
END_PAUSE_SECONDS = 2  # How long to pause at the last frame

def generate_electrofracture_gif(data, output_file='Multiphysics_PINN.gif', fps=2, dpi=150,
                                   disp_scale=50.0, save_frames=False):
    """Generate animated GIF from simulation data.
    
    Args:
        data: Simulation data dictionary
        output_file: Output GIF filename
        fps: Frames per second
        dpi: DPI for output
        disp_scale: Scale factor for displacement visualization (larger = more exaggerated)
        save_frames: If True, save each frame as a separate PNG file
    """
    
    # Extract data
    steps = data['steps']
    xGrid = np.array(data['xGrid'])
    yGrid = np.array(data['yGrid'])
    holes = data['holes']
    
    n_original_frames = len(steps)
    
    # Create frame sequence with pause at crack detection and at end
    pause_frames = int(PAUSE_DURATION_SECONDS * fps)  # Number of duplicate frames for pause
    end_pause_frames = int(END_PAUSE_SECONDS * fps)  # Number of duplicate frames for end pause
    frame_sequence = []
    for i in range(n_original_frames):
        frame_sequence.append(i)
        if i == CRACK_DETECTION_FRAME and CRACK_DETECTION_FRAME < n_original_frames:
            # Add duplicate frames for pause effect
            frame_sequence.extend([i] * pause_frames)
    
    # Add pause at the last frame
    if n_original_frames > 0:
        frame_sequence.extend([n_original_frames - 1] * end_pause_frames)
    
    n_frames = len(frame_sequence)
    print(f"Generating GIF with {n_original_frames} unique frames ({n_frames} total with pauses)...")
    print(f"Crack detection alert at frame {CRACK_DETECTION_FRAME} (pausing for {PAUSE_DURATION_SECONDS}s)")
    print(f"End frame pause: {END_PAUSE_SECONDS}s")
    print(f"Displacement scale factor: {disp_scale}x")
    
    # Check if displacement data is available
    has_displacement = len(steps) > 0 and 'u_pred' in steps[0] and 'v_pred' in steps[0]
    
    # ==================== THEME SETTINGS ====================
    BG_COLOR = '#2d3748'
    PLOT_BG = '#3d4a5c'
    TEXT_COLOR = '#f7fafc'
    ACCENT_BLUE = '#63b3ed'
    ACCENT_RED = '#fc8181'
    ACCENT_PURPLE = '#d6bcfa'
    ACCENT_GREEN = '#68d391'
    GRID_COLOR = '#4a5568'
    
    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 14,
        'text.color': TEXT_COLOR,
        'axes.labelcolor': TEXT_COLOR,
        'figure.facecolor': BG_COLOR,
        'axes.facecolor': PLOT_BG,
        'axes.grid': False,
    })
    
    # Create figure with new layout:
    # Row 0: Phi (damage) | Deformed mesh with crack opening (equal size)
    # Row 1: Voltage | Current | Power
    # Row 2: Resistance evolution - spans all columns
    fig = plt.figure(figsize=(18, 16), facecolor=BG_COLOR)
    gs = GridSpec(3, 6, figure=fig, height_ratios=[1.2, 1, 0.6], hspace=0.3, wspace=0.25)
    
    # Helper to draw holes (with optional displacement)
    def draw_holes(ax, holes, facecolor=BG_COLOR, u_disp=None, v_disp=None, scale=1.0):
        for hole in holes:
            cx, cy = hole['center']
            # Apply average displacement in hole region if provided
            if u_disp is not None or v_disp is not None:
                # Find points near hole center and use average displacement
                dist = np.sqrt((xGrid - cx)**2 + (yGrid - cy)**2)
                r = hole['radius'] if hole['type'] == 'circle' else max(hole['width'], hole['height'])/2
                near_mask = dist < r * 2
                if np.any(near_mask):
                    if u_disp is not None:
                        cx += np.mean(u_disp[near_mask]) * scale
                    if v_disp is not None:
                        cy += np.mean(v_disp[near_mask]) * scale
            
            if hole['type'] == 'circle':
                patch = Circle((cx, cy), hole['radius'], fill=True,
                              facecolor=facecolor, edgecolor='white', linewidth=1.5)
            else:
                w, h = hole['width'], hole['height']
                patch = Rectangle((cx - w/2, cy - h/2), w, h, fill=True,
                                 facecolor=facecolor, edgecolor='white', linewidth=1.5)
            ax.add_patch(patch)
    
    # Get color limits from final frame
    voltage_vmax = data.get('V_applied', 1.0)
    
    # Create axes - top row has 2 equal panels, middle row has 3 panels
    ax_phi = fig.add_subplot(gs[0, 0:3])       # Damage - left half
    ax_deformed = fig.add_subplot(gs[0, 3:6])  # Deformed mesh - right half (equal size)
    ax_voltage = fig.add_subplot(gs[1, 0:2])   # Voltage
    ax_current = fig.add_subplot(gs[1, 2:4])   # Current density
    ax_power = fig.add_subplot(gs[1, 4:6])     # Power dissipation
    ax_resist = fig.add_subplot(gs[2, :])      # Resistance - full width
    
    # Initial scatter plots
    sc_phi = ax_phi.scatter(xGrid, yGrid, c=np.zeros(len(xGrid)), cmap='hot', s=8, vmin=0, vmax=1)
    sc_deformed = ax_deformed.scatter(xGrid, yGrid, c=np.zeros(len(xGrid)), cmap='coolwarm', s=6)
    sc_voltage = ax_voltage.scatter(xGrid, yGrid, c=np.zeros(len(xGrid)), cmap='plasma', s=4, vmin=0, vmax=voltage_vmax)
    sc_current = ax_current.scatter(xGrid, yGrid, c=np.zeros(len(xGrid)), cmap='Blues', s=4)
    sc_power = ax_power.scatter(xGrid, yGrid, c=np.zeros(len(xGrid)), cmap='inferno', s=4)
    
    # Quiver for current (will be updated each frame)
    step_quiver = max(1, len(xGrid) // 200)
    quiver = ax_current.quiver(xGrid[::step_quiver], yGrid[::step_quiver],
                                np.zeros(len(xGrid[::step_quiver])), np.zeros(len(yGrid[::step_quiver])),
                                color=ACCENT_PURPLE, alpha=0.9, scale=None, width=0.008)
    
    # Colorbars
    cbar_phi = plt.colorbar(sc_phi, ax=ax_phi, pad=0.02, shrink=0.8)
    cbar_phi.set_label('Damage $\\phi$', fontsize=14, color=TEXT_COLOR)
    
    cbar_def = plt.colorbar(sc_deformed, ax=ax_deformed, pad=0.02, shrink=0.8)
    cbar_def.set_label('Displacement $v$', fontsize=14, color=TEXT_COLOR)
    
    cbar_volt = plt.colorbar(sc_voltage, ax=ax_voltage, pad=0.02)
    cbar_volt.set_label('Voltage (V)', fontsize=12, color=TEXT_COLOR)
    
    cbar_curr = plt.colorbar(sc_current, ax=ax_current, pad=0.02)
    cbar_curr.set_label('|J| (A/m²)', fontsize=12, color=TEXT_COLOR)
    
    cbar_pow = plt.colorbar(sc_power, ax=ax_power, pad=0.02)
    cbar_pow.set_label('Power', fontsize=12, color=TEXT_COLOR)
    
    # Pre-compute resistance spline for all frames at once
    all_steps = np.array([s['step'] for s in steps])
    all_resistance = np.array([s['resistance'] for s in steps])
    R0 = all_resistance[0] if len(all_resistance) > 0 else 1.0
    if R0 > 0:
        all_deltaR = (all_resistance - R0) / R0 * 100
    else:
        all_deltaR = all_resistance
    # Clamp to >= 0
    all_deltaR_clamped = np.maximum(all_deltaR, 0)
    
    # Create smooth interpolation over all frames
    n_smooth = 100 * n_frames  # Fine resolution for smooth animation
    if len(all_steps) >= 4:
        x_smooth_full = np.linspace(all_steps[0], all_steps[-1], n_smooth)
        spline_full = make_interp_spline(all_steps, all_deltaR_clamped, k=3)
        y_smooth_full = np.maximum(spline_full(x_smooth_full), 0)
    else:
        x_smooth_full = all_steps
        y_smooth_full = all_deltaR_clamped
    
    # Pre-compute y-axis limits
    y_max_global = np.max(all_deltaR_clamped) if len(all_deltaR_clamped) > 0 else 1
    y_margin_global = max(y_max_global * 0.2, 0.1)
    
    # Get resistance value at crack detection frame for annotation
    crack_detection_resistance = all_deltaR_clamped[CRACK_DETECTION_FRAME] if CRACK_DETECTION_FRAME < len(all_deltaR_clamped) else 0
    
    def animate(frame_idx):
        # Map animation frame to actual data frame (handles duplicates for pause)
        actual_frame = frame_sequence[frame_idx]
        is_crack_detection_frame = (actual_frame == CRACK_DETECTION_FRAME)
        
        step_data = steps[actual_frame]
        iStep = step_data['step']
        
        # Get field data
        phi = np.array(step_data['phi_pred'])
        voltage = np.array(step_data['voltage_pred'])
        energy_elec = np.array(step_data['energy_elec_pred'])
        Jx = np.array(step_data.get('Jx_pred', np.zeros(len(xGrid))))
        Jy = np.array(step_data.get('Jy_pred', np.zeros(len(xGrid))))
        J_mag = np.sqrt(Jx**2 + Jy**2)
        
        # Get displacement data if available
        u_disp = np.array(step_data.get('u_pred', np.zeros(len(xGrid))))
        v_disp = np.array(step_data.get('v_pred', np.zeros(len(xGrid))))
        disp_mag = np.sqrt(u_disp**2 + v_disp**2)
        
        # Update scatter data
        sc_phi.set_array(np.clip(phi, 0, 1))
        sc_voltage.set_array(voltage)
        sc_current.set_array(J_mag)
        sc_current.set_clim(0, np.max(J_mag) if np.max(J_mag) > 0 else 1)
        sc_power.set_array(energy_elec)
        sc_power.set_clim(np.min(energy_elec), np.max(energy_elec))
        
        # Update deformed mesh - show crack opening with scaled displacement (v direction only)
        if has_displacement and np.max(np.abs(v_disp)) > 0:
            # Deformed coordinates - only vertical displacement for clearer crack opening view
            x_def = xGrid  # Keep x constant
            y_def = yGrid + v_disp * disp_scale
            
            # Update scatter positions and colors
            sc_deformed.set_offsets(np.column_stack([x_def, y_def]))
            sc_deformed.set_array(v_disp)  # Color by vertical displacement
            v_max = np.max(np.abs(v_disp))
            sc_deformed.set_clim(-v_max, v_max)
            
            # Update deformed mesh holes (only v direction)
            for patch in ax_deformed.patches[:]:
                patch.remove()
            draw_holes(ax_deformed, holes, u_disp=None, v_disp=v_disp, scale=disp_scale)
            
            # Fixed x-axis limits (initial coordinates), dynamic y-axis
            x_margin = 0.1
            y_margin = 0.15
            ax_deformed.set_xlim(np.min(xGrid) - x_margin, np.max(xGrid) + x_margin)
            ax_deformed.set_ylim(np.min(y_def) - y_margin, np.max(y_def) + y_margin)
        else:
            sc_deformed.set_array(np.zeros(len(xGrid)))
        
        ax_deformed.set_aspect('equal')
        ax_deformed.set_xticks([])
        ax_deformed.set_yticks([])
        
        # Update quiver arrows
        quiver.set_UVC(Jx[::step_quiver], Jy[::step_quiver])
        
        # Draw holes on each field axis
        for ax in [ax_phi, ax_voltage, ax_current, ax_power]:
            for patch in ax.patches[:]:
                patch.remove()
            draw_holes(ax, holes)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Set titles
        ax_phi.set_title('Crack Propagation (Damage Field $\\phi$)', fontsize=16, 
                        fontweight='bold', color=TEXT_COLOR, pad=10)
        ax_deformed.set_title(f'Deformed Shape ({disp_scale:.0f}× scale)', fontsize=16, 
                             fontweight='bold', color=TEXT_COLOR, pad=10)
        ax_voltage.set_title('Electric Potential', fontsize=14, fontweight='bold', color=TEXT_COLOR)
        ax_current.set_title('Current Density', fontsize=14, fontweight='bold', color=TEXT_COLOR)
        ax_power.set_title('Heat Dissipation', fontsize=14, fontweight='bold', color=TEXT_COLOR)
        
        # Resistance history plot - use pre-computed spline
        ax_resist.clear()
        ax_resist.set_facecolor(PLOT_BG)
        
        current_step = all_steps[actual_frame]
        
        # Slice pre-computed smooth curve up to current frame
        mask_smooth = x_smooth_full <= current_step
        x_plot = x_smooth_full[mask_smooth]
        y_plot = y_smooth_full[mask_smooth]
        
        if len(x_plot) > 0:
            # Plot smooth curve up to current step
            ax_resist.plot(x_plot, y_plot, color=ACCENT_BLUE,
                          linewidth=2.5, label='$\\Delta R/R_0$')
            # Plot original data points as markers (up to current frame)
            ax_resist.scatter(all_steps[:actual_frame+1], all_deltaR_clamped[:actual_frame+1], 
                             color=ACCENT_BLUE, s=50, zorder=5, edgecolor='white', linewidth=1)
            ax_resist.fill_between(x_plot, y_plot, alpha=0.3, color=ACCENT_BLUE)
        
        ax_resist.axvline(x=current_step, color=ACCENT_RED, linestyle='--', 
                         alpha=0.8, linewidth=2, label='Current Step')
        
        # Add crack detection marker and alert
        if actual_frame >= CRACK_DETECTION_FRAME and CRACK_DETECTION_FRAME < len(all_steps):
            crack_step = all_steps[CRACK_DETECTION_FRAME]
            
            # Vertical line at crack detection
            ax_resist.axvline(x=crack_step, color='#fbbf24', linestyle='-', 
                             alpha=0.9, linewidth=2.5, zorder=4)
            
            # Marker at the detection point
            ax_resist.scatter([crack_step], [crack_detection_resistance], 
                             color='#fbbf24', s=150, zorder=10, marker='*',
                             edgecolor='white', linewidth=1.5)
            
            # Alert annotation box
            alert_text = '⚠️ CRACK DETECTED\nEarly Warning!'
            bbox_props = dict(boxstyle='round,pad=0.5', facecolor='#fbbf24', 
                            edgecolor='white', linewidth=2, alpha=0.95)
            
            # Position annotation above the detection point
            ax_resist.annotate(alert_text, 
                              xy=(crack_step, crack_detection_resistance),
                              xytext=(crack_step + 5, crack_detection_resistance + y_max_global * 0.3),
                              fontsize=12, fontweight='bold', color='#1a1a2e',
                              ha='center', va='bottom',
                              bbox=bbox_props,
                              arrowprops=dict(arrowstyle='->', color='#fbbf24', lw=2))
            
            # Add "First Peak" label
            ax_resist.text(crack_step, -y_margin_global * 0.5, 'First Peak', 
                          fontsize=10, ha='center', color='#fbbf24', fontweight='bold')
        
        # Flash effect when at crack detection frame
        if is_crack_detection_frame:
            # Add glowing border effect to the resistance plot
            for spine in ax_resist.spines.values():
                spine.set_edgecolor('#fbbf24')
                spine.set_linewidth(3)
        else:
            for spine in ax_resist.spines.values():
                spine.set_edgecolor(GRID_COLOR)
                spine.set_linewidth(1)
        
        # Fixed axis limits for consistent scaling
        ax_resist.set_xlim(-0.5, all_steps[-1] + 0.5)
        ax_resist.set_ylim(-y_margin_global, y_max_global + y_margin_global)
        
        ax_resist.set_xlabel('Load Step', fontsize=14)
        ax_resist.set_ylabel('$\\Delta R/R_0$ (%)', fontsize=14)
        ax_resist.set_title('Piezoresistive Response (Structural Health Monitoring)', fontsize=16, fontweight='bold', 
                           color=TEXT_COLOR, pad=10)
        ax_resist.legend(loc='upper left', fontsize=11, frameon=True, 
                        facecolor=PLOT_BG, edgecolor=GRID_COLOR)
        
        # Main title
        fig.suptitle(f'Multiphysics PINN: Electro-Fracture Mechanics  •  Step {iStep}',
                     fontsize=22, fontweight='bold', color=TEXT_COLOR, y=0.995)
        
        # Coupled equations subtitle (single math mode block)
        equations = r'$\nabla \cdot \mathbf{\sigma}(\mathbf{u}, \phi) = 0 \;\longleftrightarrow\; \nabla \cdot [\sigma_e(\phi) \nabla V] = 0 \;|\; \sigma_e = \sigma_0 (1-\phi)^2$'
        fig.text(0.5, 0.965, equations, ha='center', va='top', 
                 fontsize=20, color='white', style='italic')
        
        return [sc_phi, sc_deformed, sc_voltage, sc_current, sc_power]
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                   interval=1000/fps, blit=False)
    
    # Save as GIF
    print(f"Saving GIF to {output_file}...")
    anim.save(output_file, writer='pillow', fps=fps, dpi=dpi)
    
    # Save individual frames if requested
    if save_frames:
        frames_dir = os.path.join(os.path.dirname(output_file), 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Saving individual frames to {frames_dir}...")
        for frame_idx in range(n_frames):
            animate(frame_idx)
            frame_file = os.path.join(frames_dir, f'frame_{frame_idx:04d}.png')
            fig.savefig(frame_file, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor='none')
            print(f"  Saved {frame_file}")
        print(f"All {n_frames} frames saved.")
    
    plt.close()
    
    print(f"GIF saved: {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate GIF from simulation data')
    parser.add_argument('--data_dir', type=str, default='TensionPlate_Electric',
                        help='Directory containing simulation_data.json')
    parser.add_argument('--output', type=str, default='Multiphysics_PINN.gif',
                        help='Output GIF filename')
    parser.add_argument('--fps', type=int, default=3,
                        help='Frames per second')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI of output')
    parser.add_argument('--disp_scale', type=float, default=5.0,
                        help='Scale factor for displacement visualization (larger = more exaggerated)')
    parser.add_argument('--save_frames', action='store_true', default=True,
                        help='Save individual frames as PNG files (default: True)')
    
    args = parser.parse_args()
    
    # Load data
    data = load_simulation_data(args.data_dir)
    
    # Generate GIF
    output_path = os.path.join(args.data_dir, args.output)
    generate_electrofracture_gif(data, output_path, fps=args.fps, dpi=args.dpi, 
                                 disp_scale=args.disp_scale, save_frames=args.save_frames)
