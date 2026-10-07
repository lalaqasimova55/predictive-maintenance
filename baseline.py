"""Median RUL baseline and common prediction adapter."""


class MedianRULBaseline:
    """Predict the training median RUL for every requested prefix."""

    def __init__(self):
        self.median_rul = None

    def fit(self, y):
        values = sorted(float(v) for v in y)
        if not values:
            raise ValueError("Training targets cannot be empty.")
        mid = len(values) // 2
        self.median_rul = values[mid] if len(values) % 2 else (
            values[mid - 1] + values[mid]
        ) / 2.0
        return self

    def predict(self, prefixes):
        if self.median_rul is None:
            raise RuntimeError("Fit the baseline before prediction.")
        return [max(0.0, self.median_rul) for _ in prefixes]
