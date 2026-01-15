import os
import numpy as np
import nibabel as nib
import scipy
from scipy.ndimage import median
from totalsegmentator.python_api import totalsegmentator

# ==============================================================================
# ANATOMICAL MASKING (TotalSegmentator)
# ==============================================================================

def generate_body_mask(mri_path, out_dir):
    """
    Automated Body Mask Generation using TotalSegmentator AI.
    
    This function ensures that only anatomical regions are processed, 
    eliminating background noise and artifacts outside the patient's body.
    """
    mask_path = os.path.join(out_dir, "body_extremities.nii.gz")

    if os.path.exists(mask_path):
        print("🟢 Using existing TotalSegmentator body mask.")
        return nib.load(mask_path).get_fdata()

    print("⚠️ Body mask not found. Initializing TotalSegmentator (task: body_mr)...")

    try:
        # Running TotalSegmentator Python API to segment body boundaries in MRI
        totalsegmentator(
            mri_path,
            out_dir,
            task="body_mr",
            fast=False
        )
    except Exception as e:
        raise RuntimeError(f"❌ TotalSegmentator failed: {e}")

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"❌ TotalSegmentator mask generation failed: {mask_path}")

    print("✅ Body mask generated successfully.")
    return nib.load(mask_path).get_fdata()


# ==============================================================================
# TISSUE LABELING & MORPHOLOGICAL CLEANUP
# ==============================================================================

def _calculate_label_data(atlas, MRIdata, prob_maps):
    """
    Calculates a voxel-wise tissue label map (Ground Truth Segment).
    
    Args:
        atlas: Hammers probability atlas.
        MRIdata: Corrected MRI data.
        prob_maps: List of tissue probability maps (GM, WM, CSF, etc. from SPM12).
    
    Returns:
        label_data: Integrated map where each voxel is assigned a tissue type.
    """
    # 1. Probabilistic integration: Compute background probability
    background = 1 - sum(prob_maps)
    prob_stack = np.stack([background] + prob_maps, axis=-1)

    # 2. Maximum Likelihood Assignment: Determine tissue type per voxel
    label_data = np.argmax(prob_stack, axis=-1)

    # 3. Structural Masking: Zero-out areas outside the scanned MRI volume
    label_data[MRIdata == 0] = 0

    # 4. Morphological Cleanup: Median filter to remove small segmentation artifacts
    brainmask = (label_data != 0).astype(np.uint8)
    brainmask_med = scipy.ndimage.median_filter(brainmask, size=3)
    brainmask[brainmask_med == 0] = 0

    # 5. Connected Component Analysis: Retain only the largest anatomical structure
    structure = np.ones((3, 3, 3))
    labelled, n = scipy.ndimage.label(brainmask, structure)
    sizes = np.bincount(labelled.ravel())

    if len(sizes) > 1:
        # Filter out isolated noise clusters; keep the main body/brain component
        brainmask = (labelled == (sizes[1:].argmax() + 1))
    else:
        brainmask = np.zeros_like(brainmask)

    label_data[brainmask == 0] = 0
    return label_data


# ==============================================================================
# ATTENUATION & ACTIVITY MAP GENERATION
# ==============================================================================

def initial_mapator(atlas, PETdata, label_data):
    """
    Generates Attenuation (ATT) and Activity (ACT) maps for SimPET input.
    
    - ATT Map: Defines physical density (Bone=3, Soft Tissue=4).
    - ACT Map: Defines regional tracer distribution based on clinical PET uptake.
    """
    IMG_SIZE = np.shape(atlas)
    Nreg = 83  # Total regions defined in the Hammers atlas

    # --- ATTENUATION MAP GENERATION (uint8) ---
    attmap = np.zeros(IMG_SIZE, np.uint8)
    # Assign specific attenuation coefficients to segmented tissues
    attmap[label_data == 4] = 3  # Dense structures / Bone
    attmap[(label_data != 0) & (label_data != 4)] = 4  # Soft tissue / Brain matter

    # --- ACTIVITY MAP GENERATION ---
    actmap = np.zeros(IMG_SIZE)  

    # Define tissue-specific atlases
    GMmasked_atlas = atlas * (label_data == 1)
    WMmasked_atlas = atlas * (label_data == 2)

    # Calculate regional mean PET uptake for Grey Matter and White Matter
    for i in range(1, Nreg + 1):
        gm_vals = PETdata[GMmasked_atlas == i]
        wm_vals = PETdata[WMmasked_atlas == i]

        if gm_vals.size > 0:
            actmap[GMmasked_atlas == i] = np.mean(gm_vals)
        if wm_vals.size > 0:
            actmap[WMmasked_atlas == i] = np.mean(wm_vals)

    # Calculate mean uptake for specific secondary tissue labels
    for t in [3, 4, 5]:
        vals = PETdata[label_data == t]
        if vals.size > 0:
            actmap[label_data == t] = np.mean(vals)

    # IMPORTANT: Linear scaling to 0-127 range and conversion to uint8
    # This ensures compatibility with Monte Carlo Simulators (SimSET/SimPET)
    actmap = (127 * actmap / np.max(actmap)).astype(np.uint8)

    # Assign residual low-level activity to ensure no zero-vacuum artifacts in simulation
    actmap[(actmap == 0) & (attmap != 0)] = 5

    return attmap, actmap


# ==============================================================================
# MAIN MULTI-PATIENT PIPELINE
# ==============================================================================

if __name__ == "__main__":
    # Define local project paths
    root_dir = "/Users/alfer/Documents/Física/9 semestre/TFG/Data"
    patients = sorted([p for p in os.listdir(root_dir) if p.startswith("P")])

    print("\n" + "="*40)
    print("  PET-SIM PRE-PROCESSING: MULTI-PATIENT RUN  ")
    print("="*40 + "\n")

    for patient_id in patients:
        print(f"🧩 Processing Cohort: {patient_id}")
        base_dir = os.path.join(root_dir, patient_id)

        try:
            # --- 1. MRI Loading and NaN Sanitization ---
            mri_file = os.path.join(base_dir, f"{patient_id}-RM.nii")
            if not os.path.exists(mri_file): continue
            
            MRI = nib.load(mri_file)
            # Ensure no NaN values interfere with mathematical calculations
            MRIdata = np.nan_to_num(MRI.get_fdata())

            # --- 2. Body Segmentation (AI-powered) ---
            body_mask = generate_body_mask(mri_file, base_dir).astype(np.uint8)
            MRIdata *= body_mask

            # --- 3. Clinical PET Loading ---
            pet_file = os.path.join(base_dir, "coPETInterictal.nii")
            if not os.path.exists(pet_file): continue
            PETdata = np.nan_to_num(nib.load(pet_file).get_fdata()) * body_mask

            # --- 4. Tissue Probability Maps (SPM12) ---
            prob_maps = []
            for i in range(1, 6):
                pm_file = os.path.join(base_dir, f"c{i}{patient_id}-RM.nii")
                prob_maps.append(nib.load(pm_file).get_fdata() * body_mask)

            # --- 5. Anatomical Atlas Loading ---
            atlas_file = os.path.join(base_dir, "rhammers.nii")
            atlas = nib.load(atlas_file).get_fdata() * body_mask

            # --- 6. Label Calculation & Map Generation ---
            label_data = _calculate_label_data(atlas, MRIdata, prob_maps)
            attmap, act0map = initial_mapator(atlas, PETdata, label_data)

            # --- 7. Export Results with explicit uint8 encoding for SimPET compatibility ---
            # Save Attenuation Map
            nib.save(nib.Nifti1Image(attmap, MRI.affine, dtype=np.uint8),
                     os.path.join(base_dir, f"att_{patient_id}.nii"))

            # Save Initial Activity Map (act0)
            nib.save(nib.Nifti1Image(act0map, MRI.affine, dtype=np.uint8),
                     os.path.join(base_dir, f"act0_{patient_id}.nii"))

            print(f"✅ Patient {patient_id}: att and act0 maps generated successfully.")

        except Exception as e:
            print(f"❌ Error in patient {patient_id}: {str(e)}")

    print("\n🎯 Processing pipeline completed.")
