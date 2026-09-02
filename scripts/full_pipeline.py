# =====================================================
# full_pipeline_test.py
# Complete end-to-end test of the reconstruction → registration → deviation analysis pipeline
# =====================================================

import os
import time
from joint_segment_torch import nii2mesh
from gesm_match_adv import neural_GESM
from vis_distance import visualize_registration_deviation, trimesh_to_open3d_mesh
import trimesh
# ========================= CONFIG =========================
# Change these paths to your actual files
NII_FILE       = "538_2134_MCP_2.nii"          # ← your raw NIfTI
TEMPLATE_UP    = "mean_mcp2.ply"                   # ← standard/normal template (up surface)
OUTPUT_PREFIX  = "538_2134_MCP_2"

# Optional: you can also process the down surface the same way
# ==========================================================

print("=== STAGE 1: Joint Surface Reconstruction ===")
start = time.time()

nii2mesh(
    nii_name=NII_FILE,
    prefix_name=OUTPUT_PREFIX,
    if_return=False          # set True if you want to get the trimesh objects back
)

print(f"Reconstruction finished in {time.time()-start:.1f} seconds\n")

# Reconstructed files
patient_up = f"{OUTPUT_PREFIX}_up.ply"
patient_down = f"{OUTPUT_PREFIX}_down.ply"

print("=== STAGE 2: Neural GESM Non-rigid Registration ===")
start = time.time()

registered_up = f"reg_{OUTPUT_PREFIX}_up.ply"

neural_GESM(
    source_name=patient_up,
    target_name=TEMPLATE_UP,
    save_name=registered_up
)

print(f"Registration finished in {time.time()-start:.1f} seconds\n")

print("=== STAGE 3: Deviation Analysis & Visualization ===")
start = time.time()

# Drop-in for the original vis_distance.visualize_registration_deviation.
# Body is the full abnormal_gui.py pipeline:
#   adaptive DBSCAN, Löwner-John ellipsoid (A + B), per-cluster Open3D GUI.
# You can pass either file paths (str) or trimesh / Open3D mesh objects.
visualize_registration_deviation(
    mesh_A_filename=patient_up,      # registered patient mesh
    mesh_B_filename=registered_up        # template (ground-truth shape)
)

print(f"Visualization & quantification finished in {time.time()-start:.1f} seconds")

print("\n" + "="*60)
print("FULL PIPELINE COMPLETED SUCCESSFULLY!")
print("Check the Open3D windows for interactive visualization.")
print("Console output contains:")
print("   • Automatic percentile recommendation")
print("   • Adaptive DBSCAN + convex-hull cluster volumes (mm³)")
print("   • Löwner-John ellipsoid volume & aspect ratio (A and B)")
print("   • Max/mean deviation statistics")
print("   • Per-cluster Open3D GUI views on mesh_B")
print("="*60)
