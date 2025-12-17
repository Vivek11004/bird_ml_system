import numpy as np

def compute_weight_with_confidence(bboxes):
    if len(bboxes) == 0:
        return 0.0, 0.0

    areas = []
    for x1, y1, x2, y2 in bboxes:
        areas.append((x2 - x1) * (y2 - y1))

    mean_area = np.mean(areas)
    std_area = np.std(areas)

    weight_index = mean_area / 10000  # normalized
    confidence = 1 / (1 + std_area)

    return float(weight_index), float(confidence)
