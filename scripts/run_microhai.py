#!/usr/bin/env python3
"""One-shot micro-HAI run: NIfTI -> cortical PLY -> registration -> abnormality PLY -> label NIfTI.

Put this file in the same directory as joint_segment_torch.py, gesm_match_adv.py,
export_abnormality_ply.py and ply_to_nii.py.

    python run_microhai.py 538_2134_MCP_2.nii

MCP_2 uses mean_mcp2.ply, MCP_3 uses mean_mcp3.ply. The Open3D GUI in
full_pipeline.py is skipped; abnormality clusters are written to disk and
mapped back onto the source volume. Each cluster gets its own label.

Labels
------
1  proximal cortical surface (*_up.ply)
2  distal cortical surface (*_down.ply)
3+ one label per abnormality cluster, in the order written
"""

from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from export_abnormality_ply import export_from_meshes
from gesm_match_adv import neural_GESM
from joint_segment_torch import nii2mesh
from ply_to_nii import map_ply_to_label


TEMPLATES = {
    "2": "mean_mcp2.ply",
    "3": "mean_mcp3.ply",
}


def stem_of(nii_path: str) -> str:
    name = os.path.basename(nii_path)
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return os.path.splitext(name)[0]


def mcp_index(name: str) -> str:
    # Underscore is a word character, so \b fails on MCP_2_o and MCP2_BL.
    match = re.search(r"MCP[_-]?(2|3)(?!\d)", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(
            "Cannot tell MCP_2 from MCP_3 in '{}'. "
            "Expected a name like 538_2134_MCP_2.nii, or pass --template.".format(name)
        )
    return match.group(1)



def resolve_template(nii_path: str, template: str | None, template_dir: str) -> tuple[str, str]:
    prefix = stem_of(nii_path)
    if template:
        path = template if os.path.isabs(template) else os.path.join(template_dir, template)
        which = "override"
    else:
        which = mcp_index(prefix)
        path = os.path.join(template_dir, TEMPLATES[which])
    if not os.path.isfile(path):
        raise FileNotFoundError("Template not found: {}".format(path))
    return which, path


def run(
    nii_path: str,
    template: str | None = None,
    template_dir: str = ".",
    out_dir: str | None = None,
    percentile: float = 75.0,
    min_points: int = 20,
    point_radius: float = 1.0,
    visualize: bool = False,
) -> str:
    nii_path = os.path.abspath(nii_path)
    if not os.path.isfile(nii_path):
        raise FileNotFoundError(nii_path)
    prefix = stem_of(nii_path)
    work = os.path.abspath(out_dir or os.path.dirname(nii_path) or ".")
    os.makedirs(work, exist_ok=True)
    which, template_path = resolve_template(nii_path, template, os.path.abspath(template_dir))

    patient_up = os.path.join(work, prefix + "_up.ply")
    patient_down = os.path.join(work, prefix + "_down.ply")
    registered_up = os.path.join(work, "reg_" + prefix + "_up.ply")
    abnormality_dir = os.path.join(work, "abnormality_ply")
    label_path = os.path.join(work, prefix + "_microhai_labels.nii.gz")

    print("=== MCP{}  template {} ===".format(which, template_path))
    print("=== 1/4 cortical surface ===")
    cwd = os.getcwd()
    os.chdir(work)
    try:
        nii2mesh(nii_name=nii_path, prefix_name=prefix, if_return=False)
    finally:
        os.chdir(cwd)
    if not os.path.isfile(patient_up) or not os.path.isfile(patient_down):
        raise FileNotFoundError("nii2mesh did not write {} and {}".format(patient_up, patient_down))

    print("=== 2/4 neural GESM registration ===")
    print(registered_up)
    neural_GESM(source_name=template_path, target_name=patient_up, save_name=registered_up)
    if not os.path.isfile(registered_up):
        raise FileNotFoundError("neural_GESM did not write {}".format(registered_up))

    if visualize:
        from vis_distance import visualize_registration_deviation

        print("=== optional deviation GUI ===")
        visualize_registration_deviation(patient_up, registered_up)

    print("=== 3/4 abnormality clusters ===")
    abnormality = export_from_meshes(
        mesh_a=patient_up,
        mesh_b=registered_up,
        out_dir=abnormality_dir,
        prefix=prefix + "_up",
        percentile=percentile,
        min_points=min_points,
    )

    print("=== 4/4 map PLY back to NIfTI ===")
    map_ply_to_label(
        nii_path,
        [patient_up, patient_down],
        abnormality,
        label_path,
        space="pipeline",
        pad=5.0,
        point_radius=point_radius,
        label_each=True,
    )
    print("label 1: {}".format(patient_up))
    print("label 2: {}".format(patient_down))
    for i, path in enumerate(abnormality):
        print("label {}: {}".format(3 + i, path))
    print("wrote {}".format(label_path))
    return label_path


def main() -> int:
    p = argparse.ArgumentParser(description="Run reconstruction, registration, abnormality export and NIfTI label mapping.")
    p.add_argument("nii", help="MCP volume, e.g. 538_2134_MCP_2.nii")
    p.add_argument("--template", default=None, help="Override template PLY. Default follows MCP_2 / MCP_3 in the file name.")
    p.add_argument("--template-dir", default=".", help="Directory containing mean_mcp2.ply and mean_mcp3.ply")
    p.add_argument("--out-dir", default=None, help="Output directory. Default: the NIfTI's directory")
    p.add_argument("--percentile", type=float, default=75.0)
    p.add_argument("--min-points", type=int, default=20)
    p.add_argument("--point-radius", type=float, default=1.0, help="Abnormality splat radius in voxels")
    p.add_argument("--visualize", action="store_true", help="Also open the vis_distance GUI before export")
    args = p.parse_args()
    run(
        args.nii,
        template=args.template,
        template_dir=args.template_dir,
        out_dir=args.out_dir,
        percentile=args.percentile,
        min_points=args.min_points,
        point_radius=args.point_radius,
        visualize=args.visualize,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
