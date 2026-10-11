# Evaluation Protocol

All official numbers follow one strict order per head, per view:

1. Forward in FP32 (autocast BF16 allowed on CUDA — identical to training
   monitoring; final scores are computed from `.float()` logits).
2. If the view was horizontally flipped, unflip the **logits**.
3. Resize logits bilinearly back to the padded input size
   (NYUv2: 480x648; SUN RGB-D: 552x744).
4. Crop to the physical image (NYUv2: width 640; SUN RGB-D: the recorded
   resized size, then a second bilinear restore to the native GT size).
5. Softmax → per-view **probabilities**.
6. Average probabilities over all views (MS5 = scales {0.5, 0.75, 1.0, 1.25,
   1.5} x {orig, hflip}; NYU view sizes round to multiples of 24).
7. Fuse heads: `P = 0.5 * P_A + 0.5 * P_B`, then argmax.

mIoU comes from ONE confusion matrix accumulated over the whole split (never
a mean of per-image IoUs). Do not softmax before resizing, do not average
logits across views or heads, and do not swap the order.

Multi-scale NYU view sizes: 240x336, 360x480, 480x648, 600x816, 720x960.

Known numerical note: identical weights evaluated with different batch sizes
or hardware can differ in the last digits (~0.005 mIoU points observed
historically). Cross-implementation acceptance compares same-device logits
first, then summary metrics.
