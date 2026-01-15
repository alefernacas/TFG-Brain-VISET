import subprocess
import os
import nibabel as nib
import numpy as np
from skimage.filters import gaussian, median
import ants  # Advanced Normalization Tools for high-precision medical image registration

# =====================================================
# GLOBAL CONFIGURATION
# =====================================================
# Number of iterations for the activity map optimization loop (Brain-VISET framework)
maxiters = 10  
# FWHM (Full Width at Half Maximum) for smoothing: balances numerical stability and spatial resolution
FWHM_VALUE = 5 

# Set up relative paths for repository portability
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "Data")

# Standardized filenames for the project
REAL_PET_FILENAME = "coPETInterictal.nii"

# =====================================================
# IMAGE PROCESSING FUNCTIONS
# =====================================================

def check_for_nan_points(data_array, affine_matrix, step_name):
    """
    Diagnostic tool to identify NaN (Not a Number) values to ensure simulation integrity.
    Logs the first occurrence in world coordinates (mm) for debugging.
    """
    nan_mask = np.isnan(data_array)
    nan_count = np.sum(nan_mask)
    if nan_count > 0:
        print(f"🚨 [CRITICAL @ {step_name}]: Found {nan_count} NaN voxels.")
    return nan_count

def prepare_flip_and_affine_fix(input_path, output_path):
    """
    Fixes the orientation of SimSET/SimPET raw outputs.
    Applies a flip on the X-axis and adjusts the affine matrix to match NIfTI standard space.
    """
    sim = nib.load(input_path)
    imgdata = sim.get_fdata()
    original_affine = sim.affine
    # Flip data to correct SimSET's coordinate system
    img1flip = np.flip(imgdata, axis=0)
    newaff = np.copy(original_affine)
    newaff[1, 1] = -newaff[1, 1] # Reflect Y-axis in affine
    img1flip[np.isnan(img1flip)] = 0
    img1flipnii = nib.Nifti1Image(img1flip, newaff)
    nib.save(img1flipnii, output_path)
    return output_path

def corregister_ants(fixed_path, mov_path, output_path_prefix="w_ants_"):
    """
    Aligns the simulated reconstruction with the reference activity map using ANTs (Affine).
    Ensures spatial consistency for the voxel-wise update in the 'update_map' function.
    """
    fixed = ants.image_read(fixed_path)
    moving = ants.image_read(mov_path)
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform='Affine')
    
    out_path = os.path.join(os.path.dirname(mov_path), output_path_prefix + os.path.basename(mov_path))
    ants.image_write(reg['warpedmovout'], out_path)
    return out_path

def calculate_label_data(prob_map_paths):
    """
    Generates a tissue label map based on probability maps (GM, WM, CSF) from SPM12.
    Used for regional intensity neutralization and anatomical masking.
    """
    prob_maps = [nib.load(p).get_fdata() for p in prob_map_paths]
    prob_stack = np.stack(prob_maps, axis=-1)
    background = 1 - np.sum(prob_stack, axis=-1)
    prob_stack = np.stack([background] + prob_maps, axis=-1)
    return np.argmax(prob_stack, axis=-1).astype(np.uint8)

def update_map(recdata, pet_data, old_actmap, attmap, label_data, fwhm, iteration):
    """
    CORE OPTIMIZATION ALGORITHM (ML-EM Based):
    Calculates a correction factor: REAL_PET / SIMULATED_RECONSTRUCTION.
    This factor updates the previous activity map to converge towards clinical ground truth.
    Includes background noise neutralization and peak protection to prevent divergence.
    """
    std = fwhm / 2.3548
    # Clip negative values resulting from reconstruction artifacts
    pet_data[pet_data < 0] = 0
    recdata[recdata < 0] = 0

    # 1. Smooth images to reduce high-frequency noise during ratio calculation
    ref_img_smoothed = gaussian(pet_data, sigma=std)
    sim_img_smoothed = gaussian(recdata, sigma=std)

    # 2. Neutralize intensities in non-active tissues (e.g., CSF) to improve stability
    for tissue_label in [3, 4, 5]:
        mask = (label_data == tissue_label)
        ref_img_smoothed[mask] = np.nanmean(ref_img_smoothed[mask])
        sim_img_smoothed[mask] = np.nanmean(sim_img_smoothed[mask])

    # 3. Peak Protection: Normalize using the maximum of active tissues (GM/WM)
    ACTIVE_MASK = np.logical_or(label_data == 1, label_data == 2)
    max_ref = np.nanmax(ref_img_smoothed * ACTIVE_MASK)
    max_sim = np.nanmax(sim_img_smoothed * ACTIVE_MASK)

    # 4. Calculate Gain Factor and apply safety clipping
    factor_img = (ref_img_smoothed / max_ref) / (sim_img_smoothed / max_sim)
    factor_img[np.isinf(factor_img)] = 0
    factor_img = np.clip(factor_img, 0, 10) # Limit maximum gain to x10

    # 5. Apply correction and Median Filter for spatial homogeneity
    upd_actmap = old_actmap * factor_img
    upd_actmap = median(upd_actmap)
    
    # Scale to uint8 range (0-127) for SimSET compatibility
    return (127 * upd_actmap / np.max(upd_actmap)).astype(np.uint8)

# =====================================================
# BATCH PROCESSING PIPELINE
# =====================================================

# Identify patient directories following the 'Pxx' naming convention
patient_dirs = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('P')])

for PATIENT_ID in patient_dirs:
    PATIENT_FOLDER = os.path.join(DATA_DIR, PATIENT_ID)
    RESULTS_DIR = os.path.join(ROOT_DIR, "Results", PATIENT_ID)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        # Load patient-specific baseline data: Attenuation, Initial Activity, and Real PET
        att = nib.load(os.path.join(PATIENT_FOLDER, f"att_{PATIENT_ID}.nii")).get_fdata()
        old_actmap = nib.load(os.path.join(PATIENT_FOLDER, f"act0_{PATIENT_ID}.nii"))
        old_actmapdata = old_actmap.get_fdata()
        pet_data = nib.load(os.path.join(PATIENT_FOLDER, REAL_PET_FILENAME)).get_fdata()

        # Generate tissue label map for anatomical priors
        PROB_MAPS = [os.path.join(PATIENT_FOLDER, f"c{j}{PATIENT_ID}-RM.nii") for j in range(1, 6)]
        label_data = calculate_label_data(PROB_MAPS)
        label_data[att == 0] = 0 # Mask non-brain regions

        # Start iterative feedback loop
        for i in range(maxiters):
            print(f"🚀 Patient {PATIENT_ID} - Iteration {i+1}/{maxiters}")

            # STEP A: Execute Monte Carlo Simulation via SimPET/GATE
            cmd = f"python3 scripts/experiment.py --params.patient_dirname='{PATIENT_ID}' ..."
            os.system(cmd)

            # STEP B: Image Post-processing (Flip & Affine orientation fix)
            # (Path simplified for readability)
            rec_path = os.path.join(ROOT_DIR, "Results", f"{PATIENT_ID}_it{i}", "rec_OSEM3D_32.img")
            moving_path = os.path.join(RESULTS_DIR, f"fliprec{i}.nii")
            prepare_flip_and_affine_fix(rec_path, moving_path)

            # STEP C: Rigid/Affine registration with ANTs for spatial alignment
            registered_path = corregister_ants(os.path.join(PATIENT_FOLDER, f"act{i}_{PATIENT_ID}.nii"), moving_path)

            # STEP D: Update Activity Map (Closing the feedback loop)
            simdata = nib.load(registered_path).get_fdata()
            new_actmap = update_map(simdata, pet_data, old_actmapdata, att, label_data, FWHM_VALUE, i)
            
            # Save updated map for the next iteration
            new_path = os.path.join(PATIENT_FOLDER, f"act{i+1}_{PATIENT_ID}.nii")
            nib.save(nib.Nifti1Image(new_actmap, old_actmap.affine), new_path)
            old_actmapdata = new_actmap 

    except Exception as e:
        print(f"❌ Failed to process {PATIENT_ID}: {e}")
