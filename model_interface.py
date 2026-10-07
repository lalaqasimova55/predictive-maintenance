"""Shared model prediction contract."""


class PredictionAdapter:
    """Expose a common predict(prefixes) interface."""

    def __init__(self, model):
        self.model = model

    def predict(self, prefixes):
        if not hasattr(self.model, "predict"):
            raise TypeError("Wrapped model must expose predict().")
        predictions = self.model.predict(prefixes)
        return [max(0.0, float(value)) for value in predictions]
