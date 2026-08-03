"""Custom exceptions for pydrlpdid."""

class DRLPDIDError(Exception):
    """Base package exception."""

class PanelValidationError(DRLPDIDError, ValueError):
    """Invalid panel structure or calendar."""

class TreatmentValidationError(DRLPDIDError, ValueError):
    """Treatment path and cohort information are inconsistent."""

class SupportError(DRLPDIDError, ValueError):
    """The requested estimand lacks clean-control support."""

class NuisanceConvergenceError(DRLPDIDError, RuntimeError):
    """A nuisance estimator did not converge."""

class JacobianError(DRLPDIDError, RuntimeError):
    """The stacked estimating-equation Jacobian is singular or ill-conditioned."""

class InferenceError(DRLPDIDError, RuntimeError):
    """Inference could not be constructed coherently."""

class ExperimentalFeatureError(DRLPDIDError, RuntimeError):
    """An experimental feature was requested without explicit opt-in."""
