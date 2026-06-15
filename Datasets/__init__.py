import os
from Datasets import (
    mvtec, visa, btad, mpdd, ksdd2, mvtecad2,
    tn3k, clinicdb, colondb, isic, br35h, brainmri, headct, kvasirseg,
)

# Set AVA_DINO_DATA_ROOT in your environment, or edit DATA_ROOT below.
INDUSTRIAL_ROOT = os.environ.get("AVA_DINO_INDUSTRIAL_ROOT", "/data/industrial")
MEDICAL_ROOT    = os.environ.get("AVA_DINO_MEDICAL_ROOT",    "/data/medical")

DATASET_REGISTRY = {
    "mvtec":     (mvtec.Dataset,     mvtec.DatasetSplit,     f"{INDUSTRIAL_ROOT}/MVTecAD"),
    "visa":      (visa.Dataset,      visa.DatasetSplit,      f"{INDUSTRIAL_ROOT}/VisA_20220922"),
    "btad":      (btad.Dataset,      btad.DatasetSplit,      f"{INDUSTRIAL_ROOT}/BTech_Dataset_transformed"),
    "mpdd":      (mpdd.Dataset,      mpdd.DatasetSplit,      f"{INDUSTRIAL_ROOT}/MPDD"),
    "mvtecad2":  (mvtecad2.Dataset,  mvtecad2.DatasetSplit,  f"{INDUSTRIAL_ROOT}/mvtec_ad_2"),
    "tn3k":      (tn3k.Dataset,      tn3k.DatasetSplit,      f"{MEDICAL_ROOT}/TN3K"),
    "clinicdb":  (clinicdb.Dataset,  clinicdb.DatasetSplit,  f"{MEDICAL_ROOT}/CVC-ClinicDB"),
    "colondb":   (colondb.Dataset,   colondb.DatasetSplit,   f"{MEDICAL_ROOT}/CVC-ColonDB"),
    "isic":      (isic.Dataset,      isic.DatasetSplit,      f"{MEDICAL_ROOT}/ISIC"),
    "ksdd2":     (ksdd2.Dataset,     ksdd2.DatasetSplit,     f"{INDUSTRIAL_ROOT}/KSDD2"),
    "br35h":     (br35h.Dataset,     br35h.DatasetSplit,     f"{MEDICAL_ROOT}/Br35H"),
    "brainmri":  (brainmri.Dataset,  brainmri.DatasetSplit,  f"{MEDICAL_ROOT}/brainmri"),
    "headct":    (headct.Dataset,    headct.DatasetSplit,    f"{MEDICAL_ROOT}/head_ct"),
    "kvasirseg": (kvasirseg.Dataset, kvasirseg.DatasetSplit, f"{MEDICAL_ROOT}/Kvasir-SEG"),
}

DATASET_CLASSES = {
    "mvtec":     {'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
                  'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'},
    "visa":      {'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                  'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'},
    "btad":      {'01', '02', '03'},
    "mpdd":      {'bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes'},
    "mvtecad2":  {'fabric', 'can', 'vial', 'rice', 'walnuts', 'fruit_jelly', 'wallplugs', 'sheet_metal'},
    "tn3k":      {'01'},
    "clinicdb":  {'01'},
    "colondb":   {'01'},
    "isic":      {'01'},
    "ksdd2":     {'01'},
    "br35h":     {'01'},
    "brainmri":  {'01'},
    "headct":    {'01'},
    "kvasirseg": {'01'},
}
