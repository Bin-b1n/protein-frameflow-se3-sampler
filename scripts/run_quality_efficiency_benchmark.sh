#!/usr/bin/env bash
set -euo pipefail

# Run FrameFlow sampler benchmarks and write metrics/runtime files.
#
# Example:
#   SAMPLES_PER_LENGTH=20 \
#   TIMESTEPS="5 10 20 50 100" \
#   LENGTH_SUBSET="[50,60,70,80,90,100]" \
#   METHODS="euler heun ab2" \
#   bash scripts/run_quality_efficiency_benchmark.sh

METHODS="${METHODS:-euler heun ab2}"
TIMESTEPS="${TIMESTEPS:-5 10 20 50 100}"
LENGTH_SUBSET="${LENGTH_SUBSET:-[50,60,70,80,90,100]}"
SAMPLES_PER_LENGTH="${SAMPLES_PER_LENGTH:-20}"
NUM_GPUS="${NUM_GPUS:-1}"
BENCH_ROOT="${BENCH_ROOT:-inference_outputs/weights/pdb/published/unconditional}"
CORRECTOR_WEIGHT="${CORRECTOR_WEIGHT:-1.0}"
MAX_CORRECTOR_NORM_RATIO="${MAX_CORRECTOR_NORM_RATIO:-0.5}"
MAX_MULTISTEP_CORRECTION_RATIO="${MAX_MULTISTEP_CORRECTION_RATIO:-0.1}"
MULTISTEP_TRANS_WEIGHT="${MULTISTEP_TRANS_WEIGHT:-0.0}"
MULTISTEP_ROT_WEIGHT="${MULTISTEP_ROT_WEIGHT:-0.5}"
AB2_GEOMETRY_GUARD="${AB2_GEOMETRY_GUARD:-True}"
AB2_GUARD_TOLERANCE="${AB2_GUARD_TOLERANCE:-0.0}"
RUN_TAG="${RUN_TAG:-n${SAMPLES_PER_LENGTH}}"

for method in ${METHODS}; do
  for steps in ${TIMESTEPS}; do
    run_name="${method}_${steps}_${RUN_TAG}"
    run_dir="${BENCH_ROOT}/${run_name}"
    echo "==> Running ${run_name}"

    start_seconds="${SECONDS}"
    python -W ignore experiments/inference_se3_flows.py \
      -cn inference_unconditional \
      inference.num_gpus="${NUM_GPUS}" \
      "inference.interpolant.sampling.method=${method}" \
      "inference.interpolant.sampling.num_timesteps=${steps}" \
      "inference.interpolant.sampling.corrector_weight=${CORRECTOR_WEIGHT}" \
      "inference.interpolant.sampling.max_corrector_norm_ratio=${MAX_CORRECTOR_NORM_RATIO}" \
      "inference.interpolant.sampling.max_multistep_correction_ratio=${MAX_MULTISTEP_CORRECTION_RATIO}" \
      "inference.interpolant.sampling.multistep_trans_weight=${MULTISTEP_TRANS_WEIGHT}" \
      "inference.interpolant.sampling.multistep_rot_weight=${MULTISTEP_ROT_WEIGHT}" \
      "inference.interpolant.sampling.ab2_geometry_guard=${AB2_GEOMETRY_GUARD}" \
      "inference.interpolant.sampling.ab2_guard_tolerance=${AB2_GUARD_TOLERANCE}" \
      "inference.samples.length_subset=${LENGTH_SUBSET}" \
      "inference.samples.samples_per_length=${SAMPLES_PER_LENGTH}" \
      "inference.inference_subdir=${run_name}"
    runtime_seconds="$((SECONDS - start_seconds))"

    if [ ! -d "${run_dir}" ]; then
      echo "Expected output directory not found: ${run_dir}" >&2
      exit 1
    fi
    printf "%s\n" "${runtime_seconds}" > "${run_dir}/runtime_seconds.txt"

    echo "==> Evaluating ${run_name}"
    python analysis/evaluate_samples.py \
      --root "${run_dir}" \
      --out "${run_dir}/metrics.csv"
  done
done

echo "==> Plotting quality-efficiency curves"
python analysis/plot_quality_efficiency.py \
  --root "${BENCH_ROOT}" \
  --tag "${RUN_TAG}" \
  --out-dir "${BENCH_ROOT}/quality_efficiency_${RUN_TAG}" \
  --x runtime
