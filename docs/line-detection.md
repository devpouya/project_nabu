---
layout: default
title: Line Detection
---

# Line Detection for Cuneiform Tablets

Line segmentation is a critical preprocessing step for cuneiform tablet analysis. Because cuneiform text is traditionally inscribed along horizontal lines, identifying these lines reduces a complex 2D search problem into simpler 1D searches along detected paths.

## Overview

The line detection system uses a two-stage approach:

```
Tablet Image → CNN Segmentation → Hough Transform → Line Hypotheses → Text Lines
```

1. **Stage 1: Pixel-wise Classification (CNN)**
   - Neural network classifies each pixel as "line" or "non-line"
   - Produces probability maps highlighting text regions

2. **Stage 2: Geometric Line Extraction (Hough)**
   - Hough transform on skeletonized CNN output
   - Groups and refines line hypotheses
   - Outputs final line coordinates

## Architecture

### LineNet CNN

An AlexNet-style architecture adapted for grayscale tablet images:

```
Input (1×227×227)
    │
    ▼
Conv(1→64, 11×11, s=4) → ReLU → MaxPool(3×3, s=2) → LRN
    │
    ▼
Conv(64→256, 5×5) → ReLU → MaxPool(3×3, s=2) → LRN
    │
    ▼
Conv(256→384, 3×3) → BatchNorm → ReLU
    │
    ▼
Conv(384→384, 3×3) → BatchNorm → ReLU
    │
    ▼
Conv(384→256, 3×3) → BatchNorm → ReLU → MaxPool(3×3, s=2)
    │
    ▼
FC(9216→512) → ReLU → Dropout
    │
    ▼
FC(512→2) → Softmax
    │
    ▼
Output: [P(non-line), P(line)]
```

### LineNetFCN (Fully Convolutional)

For inference, the classifier is converted to a fully convolutional network:

```
FC Layers → Conv Layers (net surgery)
FC(9216→512) → Conv(256→512, 6×6)
FC(512→2)    → Conv(512→2, 1×1)
```

This allows processing images of arbitrary size and produces dense pixel-wise predictions.

## Training

### Dataset Preparation

The training dataset requires line annotations in the following format:

```csv
image_id,line_idx,x,y
P010054,0,100,50
P010054,0,150,52
P010054,0,200,51
P010054,1,100,120
...
```

Each row represents a point along a text line. Multiple points per line trace the line path.

### Synthetic Annotation Generation

When manual annotations aren't available, synthetic annotations can be generated using edge detection:

```bash
python scripts/train_line_segmentation.py --generate-annotations
```

This uses:
1. Sobel edge detection for horizontal features
2. Morphological operations to connect line regions
3. Hough transform to extract line paths
4. Automatic coordinate sampling along detected lines

### Training Command

```bash
# Basic training
python scripts/train_line_segmentation.py \
    --train \
    --epochs 100 \
    --batch-size 128

# With experiment tracking
python scripts/train_line_segmentation.py \
    --train \
    --epochs 100 \
    --batch-size 128 \
    --tracker wandb \
    --run-name line_detection_v1

# Full pipeline (generate + train)
python scripts/train_line_segmentation.py \
    --generate-annotations \
    --train \
    --epochs 100
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 0.01 | Initial learning rate |
| `--crop-size` | 227 | Input patch size |
| `--tracker` | tensorboard | Experiment tracking (tensorboard/wandb/all/none) |

### Data Augmentation

Training uses the following augmentations:
- Random rotation (±8°)
- Random resized crop (scale 0.5-1.0)
- Center crop with margin

## Inference

### Running Detection

```bash
# Single image
python scripts/run_line_segmentation.py \
    --model-path outputs/line_training/weights/linenet_fcn.pth \
    --image data/external/cuneiml/images/P010054.jpg

# Batch processing
python scripts/run_line_segmentation.py \
    --model-path outputs/line_training/weights/linenet_fcn.pth \
    --batch

# Using edge detection (no trained model)
python scripts/run_line_segmentation.py \
    --use-edges \
    --image path/to/tablet.jpg
```

### Python API

```python
from nabu.detection import LineDetector, LineNetFCN
from PIL import Image

# Load model
detector = LineDetector(
    model_path="outputs/line_training/weights/linenet_fcn.pth",
    device="cuda"
)

# Run detection
image = Image.open("tablet.jpg")
result = detector.detect(image, num_lines=15)

# Access results
print(f"Detected {len(result.line_hypos)} lines")
print(f"Line segments: {len(result.line_segs)}")
print(f"Interline distance: {result.dist_interline_median:.1f}px")

# Get line coordinates
for idx, row in result.line_hypos.iterrows():
    angle, dist = row['angle'], row['dist']
    print(f"Line {idx}: angle={angle:.2f}°, dist={dist:.1f}px")
```

## Post-Processing Pipeline

### Step 1: Skeletonization

The CNN output (probability map) is thresholded and skeletonized:

```python
from skimage.morphology import skeletonize

# Threshold CNN output
binary_mask = (cnn_output > 0.7).astype(np.uint8)

# Reduce to single-pixel lines
skeleton = skeletonize(binary_mask)
```

### Step 2: Hough Transform

Classic Hough transform focused on horizontal lines (83-97°):

```python
from skimage.transform import hough_line, hough_line_peaks

# Focused angle range for cuneiform
theta = np.linspace(np.deg2rad(83), np.deg2rad(97), 50)

# Compute Hough transform
h, theta, d = hough_line(skeleton, theta=theta)

# Find peaks
accums, angles, dists = hough_line_peaks(
    h, theta, d,
    min_distance=1,
    min_angle=14,
    num_peaks=num_lines * 2
)
```

### Step 3: Line Grouping

Lines that intersect within the image center are grouped:

```python
# Check intersection in center region
interval = [width * 0.125, width * 0.875]

# Group intersecting lines
for l1, l2 in combinations(lines, 2):
    if do_intersect_in_interval(l1, l2, interval):
        merge_lines(l1, l2)
```

### Step 4: Proximity-Based Merging

Nearby parallel lines are merged using area method:

```python
# Compute area between line segments
area = shoelace_formula(segment_polygon)

# Merge if area is small relative to interline distance
if area < interline_distance * interval_length / 2:
    merge_lines(l1, l2)
```

### Step 5: Line Segmentation

Probabilistic Hough extracts individual segments:

```python
from skimage.transform import probabilistic_hough_line

segments = probabilistic_hough_line(
    skeleton,
    threshold=6,
    line_length=8,
    line_gap=6,
    theta=theta_range
)
```

### Step 6: Watershed Segmentation

Segments are assigned to lines using watershed:

```python
from skimage.segmentation import watershed
from scipy import ndimage as ndi

# Distance transform
distance = ndi.distance_transform_edt(binary_mask)

# Create markers from line segments
markers = create_markers(segments, line_labels)

# Watershed segmentation
segm_labels = watershed(-distance, markers, mask=binary_mask)
```

## Output Format

### LineDetectionResult

```python
@dataclass
class LineDetectionResult:
    line_hypos: pd.DataFrame    # Line hypotheses (angle, dist, label)
    line_segs: List             # Detected line segments
    segm_labels: np.ndarray     # Pixel-wise line labels
    skeleton: np.ndarray        # Skeletonized detection map
    binary_mask: np.ndarray     # Binary CNN output
    dist_interline_median: float  # Median distance between lines
    hough_accumulator: np.ndarray  # Hough transform accumulator
```

### Line Hypotheses DataFrame

| Column | Description |
|--------|-------------|
| `accum` | Hough accumulator value (confidence) |
| `angle` | Line angle in radians |
| `dist` | Perpendicular distance from origin |
| `label` | Group label (merged lines share label) |

## Visualization

The detection script produces a 4-panel visualization:

```
┌─────────────────────┬─────────────────────┐
│   Original Image    │    Binary Mask      │
│   + Detected Lines  │   (CNN output)      │
├─────────────────────┼─────────────────────┤
│     Skeleton        │   Segmentation      │
│   + Line Segments   │   (labeled lines)   │
└─────────────────────┴─────────────────────┘
```

## Hyperparameters

### CNN Training

| Parameter | Value | Notes |
|-----------|-------|-------|
| `line_height` | 32 | Pixels from line center considered "on line" |
| `sample_radius` | 96 | Maximum sampling distance from lines |
| `soft_bg_frac` | 0.1 | Fraction of easy negative samples |
| `crop_size` | 227×227 | Input patch size |

### Hough Post-Processing

| Parameter | Value | Notes |
|-----------|-------|-------|
| `theta_range` | 83°-97° | Focused on horizontal lines |
| `num_peaks_factor` | 1.9 | Multiplier for expected peaks |
| `min_distance` | 1 | Minimum peak separation |
| `min_angle` | 14° | Minimum angle between peaks |
| `line_length` | 8 | Minimum segment length |
| `line_gap` | 6 | Maximum gap in segments |

## GCP Training

For training on Google Cloud with GPU:

```bash
# Upload to GCP
./gcp/upload_to_gcp.sh nabu-training us-central1-a

# SSH and run training
gcloud compute ssh nabu-training --zone=us-central1-a
cd ~/project_nabu
./gcp/train.sh 100 128 0.01 wandb line_detection_v1

# Download results
./gcp/download_results.sh nabu-training us-central1-a
```

## Performance

### Detection Quality

| Metric | Value | Notes |
|--------|-------|-------|
| Line Detection Rate | ~85% | On CuneiML test set |
| False Positive Rate | ~10% | Extra lines detected |
| Position Accuracy | ±5px | Average line position error |

### Speed

| Operation | Time | Hardware |
|-----------|------|----------|
| CNN Inference | ~50ms | NVIDIA T4 |
| Hough Post-processing | ~100ms | CPU |
| Full Pipeline | ~200ms | Per image |

## Limitations

1. **Damaged Tablets**: Broken or worn areas may cause false detections
2. **Non-Horizontal Text**: Angled or curved lines not well handled
3. **Dense Text**: Very close lines may be merged
4. **Image Quality**: Low resolution or poor lighting affects accuracy

## Future Improvements

- [ ] Multi-scale detection for varying line densities
- [ ] Curved line support using polynomial fitting
- [ ] Confidence scoring for each detected line
- [ ] Integration with sign detection for end-to-end OCR
- [ ] Attention mechanisms for context-aware detection
