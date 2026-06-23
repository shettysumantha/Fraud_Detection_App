import joblib
from pathlib import Path
p = Path('models/fraud_pipeline.joblib')
print('exists', p.exists())
if p.exists():
    obj = joblib.load(p)
    print('type', type(obj))
    try:
        keys = list(obj.keys())
        print('keys', keys)
    except Exception:
        print('not a dict; repr:', repr(obj)[:200])

