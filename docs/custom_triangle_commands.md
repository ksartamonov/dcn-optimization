# Triangle Topology (custom_topology.json)

Run from the repository root `netcalc_dc_release`.

## 1) Quick route samples (PNG + JSON)
```bash
python3 scripts/route_samples.py \
  --config configs/config_custom_triangle.yaml \
  --output triangle_routes \
  --load-factor 0.3 \
  --flows 3
```

## 1b) Quick route samples with manual flows
```bash
python3 scripts/route_samples.py \
  --config configs/config_custom_triangle_manual.yaml \
  --output triangle_routes_manual \
  --load-factor 1.0 \
  --flows 6
```

## 2) Print example routes to terminal
```bash
python3 scripts/inspect_routes.py \
  --config configs/config_custom_triangle.yaml \
  --load-factor 0.3 \
  --flows-limit 3
```

## 2b) Print example routes (manual flows)
```bash
python3 scripts/inspect_routes.py \
  --config configs/config_custom_triangle_manual.yaml \
  --load-factor 1.0 \
  --flows-limit 6
```

## 3) One quick experiment run (metrics)
```bash
python3 scripts/suite.py \
  -o triangle_quick \
  --include custom_custom_triangle \
  --repetitions 1 \
  --loads 0.3 \
  --k-paths 3 \
  --no-plots
```

## 4) Timing run (time_suite)
```bash
python3 scripts/time_suite.py \
  -o triangle_time \
  --include custom_custom_triangle \
  --repetitions 1 \
  --workers 2 \
  --loads 0.3 \
  --k-paths 3
```

## 4b) Timing run with manual flows (config)
```bash
python3 scripts/time_suite.py \
  -o triangle_time_manual \
  --config configs/config_custom_triangle_manual.yaml \
  --repetitions 1 \
  --workers 2
```
