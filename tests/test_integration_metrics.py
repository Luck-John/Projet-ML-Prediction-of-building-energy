import joblib
import os


def _find_metrics(obj):
    if isinstance(obj, dict):
        if 'mae' in obj:
            return obj
        for v in obj.values():
            found = _find_metrics(v)
            if found is not None:
                return found
    return None


def test_refactored_metrics_close_to_notebook():
    path = os.path.join('artifacts', 'compare_report.joblib')
    assert os.path.exists(path), f"Missing compare report at {path}"
    report = joblib.load(path)

    nb_metrics = _find_metrics(report.get('notebook_metrics', report))
    ref_metrics = _find_metrics(report.get('refactor_metrics', report))

    assert nb_metrics is not None and ref_metrics is not None, "Could not find metrics in report"

    mae_nb = float(nb_metrics['mae'])
    mae_ref = float(ref_metrics['mae'])

    # allow up to 5% relative difference
    rel_diff = abs(mae_ref - mae_nb) / mae_nb
    assert rel_diff < 0.05, f"MAE relative difference too large: {rel_diff:.3f}"
