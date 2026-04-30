## Criticality Notes

These scripts are manual analysis templates.

Files:
- `compute_original_qreshape_mass_indicator.py`: build raw `Q_reshape^mass`, action, drift, and product curves
- `compare_potential_related_indicators.py`: compare multiple potential-related indicators from the same run and export `action` / `potential` columns for downstream criticality plots

How to use:
1. Open the target script.
2. Edit the manual defaults near the top:
   - run name
   - seed
   - checkpoint selector
   - device
   - output label / slug
3. Run the script directly.

Suggested criticality views:
- original `Q_reshape^mass`
- action curve
- drift-related companion curve when needed
- downstream additive criticality with `criticality = alpha * action + beta * potential`

Current public naming:
- `action`: normalized action component
- `potential`: normalized `Q_div_abs_mass` component
- final figure assembly lives in `downstream/`, not in YAML

What is not bundled here:
- figures
- saved manifests
- previous downstream outputs

This repository only keeps the code path. Recreate results by pointing the script to your own run directory.
