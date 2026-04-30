## Downstream Notes

This folder contains manual downstream plotting and summary scripts.

Files:
- `run_downstream.py`: convenience wrapper to execute the full downstream chain
- `figure_b_continuous_dynamics.py`: draw continuous dynamics from `piuot/export_trajectory_points.py` output
- `analyze_manifold_physics_fates.py`: physics-fate panel and related summaries
- `build_focus_bundle.py`: compact focus bundle for a selected run
- `build_perturbation_dynamic_fraction.py`: perturbation-driven dynamic cell-type fractions
- `build_perturbation_manifest.py`: collect perturbation outputs into a lightweight manifest
- `export_potential_landscape_points.py`: export the `*_potential_landscape_points.csv` required by Figure c
- `figure_c_umap_potential_landscape.py`: draw the UMAP potential landscape from `*_potential_landscape_points.csv`
- `build_figure_d_benchmark_metrics.py`: build Figure D benchmark metric panels from `per_time_metrics.csv`
- `build_action_potential_criticality.py`: build action/potential criticality components and `alpha * action + beta * potential` curves
- `build_additive_criticality_board.py`: build an additive criticality board from selected component panels
- `figure_e_warning_umap.py`: draw the warning UMAP panel from a warning-score CSV
- `figure_g_gene_perturbation_screen.py`: draw Figure G from perturbation-screen CSV files
- `GITHUB_FIGURE_SEQUENCE.md`: compact four-figure layout for GitHub or paper assembly

How to use:
1. Open `run_downstream.py` or the target script.
2. Edit the manual defaults near the top:
   - run name
   - data path
   - embedding key
   - checkpoint
   - output label / slug
   - state / fate keys
   - device
3. Run the script directly.

Trajectory CSV:
- `figure_b_continuous_dynamics.py` is only the renderer. It expects the combined CSV written by `piuot/export_trajectory_points.py`.
- `run_downstream.py` runs this export automatically before drawing Figure b.

Potential landscape CSV:
- `figure_c_umap_potential_landscape.py` is only the renderer. It expects a points CSV with two plot coordinates, a time column, and `surface_z`.
- Generate that CSV from a trained PIUOT run with `export_potential_landscape_points.py`.
- The full downstream wrapper now does this automatically before calling `figure_c_umap_potential_landscape.py`.

Example:
```bash
python downstream/export_potential_landscape_points.py \
  --run-name <run_name> \
  --seed 0 \
  --checkpoint auto \
  --data-path <input.h5ad> \
  --embedding-key <obsm_key> \
  --time-obs-key <time_column> \
  --output-prefix <label>

python downstream/figure_c_umap_potential_landscape.py \
  --points-csv downstream/output/<label>_potential_landscape/<label>_potential_landscape_points.csv \
  --x-column plot_x \
  --y-column plot_y \
  --z-column surface_z \
  --time-column time_obs
```

Do not patch `config.pt` by hand to add fields such as `run_name`, `x_dim`, `k_dims`, `layers`, or `sigma_type`. Those errors usually mean an old plotting script is being used with the current PIUOT output. Use `export_potential_landscape_points.py` instead; it reads the current run directory and fills compatible defaults from the checkpoint and h5ad metadata when possible.

Latest validated downstream workflow:
- use this folder as a manual downstream workflow template
- choose your own run, embedding, labels, and checkpoint
- this folder is intended for rebuilding manuscript figures from your current run rather than shipping fixed result bundles

Common figure sequence:
- `Figure b`: continuous dynamics from reconstructed trajectory points
- `Figure c`: UMAP potential landscape with CSV-time point coloring
- `Figure D`: benchmark metric comparison panels
- additive criticality board
- `Figure e`: warning UMAP landscape
- `Figure G`: gene perturbation fate screen from Day2 `z=+10` overexpression

Figure D benchmark metrics:
- input is a `per_time_metrics.csv` table with at least `model`, `time`, `w1`, `w2_sq`, and `mmd_rbf`
- the script drops the earliest time point by default, matching the future-time benchmark convention
- it writes two folders: `complete/` with metric names, model names, and numeric y ticks; `clean/` with no text for manual PPT or Illustrator layout
- the default manuscript ticks are `w1=1,2,4`, `w2_sq=1,4,16`, and `mmd_rbf=0.02,0.1,0.5`

Example:
```bash
python downstream/build_figure_d_benchmark_metrics.py \
  --per-time-csv <compare_output>/per_time_metrics.csv \
  --model-order PIUOT-official-gae15,DeepRUOT,PISDE,TrajectoryNet,PRESCIENT \
  --output-prefix ipsc_figure_d
```

Current criticality convention:
- `action`: the normalized action component
- `potential`: the normalized `Q_div_abs_mass` potential-related component
- `criticality`: `alpha * action + beta * potential`
- the downstream scripts own these plotting choices; YAML files only configure training and data loading

Design rule:
- keep this folder as a code-only manual workflow
- do not rely on YAML here
- do not commit generated figures or manifests into this GitHub release
