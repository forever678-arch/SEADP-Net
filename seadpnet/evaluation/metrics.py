"""Dynamic confusion matrix and explicit absent-class handling."""
import torch


class SemanticMetrics:
    def __init__(self, num_classes, class_names=None, ignore_index=255):
        self.n = int(num_classes)
        self.names = class_names or [str(index) for index in range(self.n)]
        if len(self.names) != self.n or len(set(self.names)) != self.n:
            raise ValueError("Class names must be unique and match num_classes")
        self.ignore = int(ignore_index)
        self.matrix = torch.zeros(self.n, self.n, dtype=torch.int64)

    def update(self, prediction, target):
        prediction, target = prediction.detach().cpu().long(), target.detach().cpu().long()
        if prediction.shape != target.shape:
            raise ValueError("Prediction and native GT shapes differ")
        keep = target != self.ignore
        p, t = prediction[keep], target[keep]
        if not bool(((t >= 0) & (t < self.n) & (p >= 0) & (p < self.n)).all()):
            raise ValueError("Invalid metric labels")
        self.matrix += torch.bincount(self.n*t+p, minlength=self.n*self.n).reshape(self.n, self.n)

    def compute(self):
        m = self.matrix.double()
        tp, gt, pred = m.diag(), m.sum(1), m.sum(0)
        union = gt + pred - tp
        iou, accuracy = tp/union.clamp_min(1), tp/gt.clamp_min(1)
        present = union > 0
        miou = float(iou[present].mean()) if present.any() else 0.
        return {"miou": miou, "miou_percent": 100*miou,
                "pixel_accuracy": float(tp.sum()/gt.sum().clamp_min(1)),
                "mean_class_accuracy": float(accuracy[gt > 0].mean()) if (gt > 0).any() else 0.,
                "classes_in_union": int(present.sum()), "num_classes": self.n,
                "absent_classes": [self.names[i] for i in range(self.n) if not present[i]],
                "miou_mean_rule": "mean over classes with confusion-matrix union > 0",
                "per_class_iou": {name: float(iou[i]) if present[i] else None for i, name in enumerate(self.names)},
                "per_class_pixel_accuracy": {name: float(accuracy[i]) if gt[i] else None for i, name in enumerate(self.names)},
                "confusion_matrix": self.matrix.tolist()}
