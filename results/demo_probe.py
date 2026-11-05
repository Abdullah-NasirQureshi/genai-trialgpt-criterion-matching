import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import app
note = """0. A 67-year-old woman presents with progressive shortness of breath over two weeks.
1. She has a 15-year history of type 2 diabetes treated with insulin.
2. Echocardiography shows an ejection fraction of 30%.
3. She takes lisinopril and furosemide.
4. She has never smoked."""
inc = "Adults with heart failure and reduced ejection fraction\nTreated with an ACE inhibitor or ARB"
exc = "Pregnancy\nActive malignancy\nEstimated GFR below 30 mL/min"
df, msg, status, rep = app.screen(note, "Demo trial", inc, exc, "KV cache", 4)
print(df[["Type", "Criterion", "Decision", "Evidence IDs"]].to_string())
print(df["Raw output"].tolist())
print(msg.splitlines()[0])
