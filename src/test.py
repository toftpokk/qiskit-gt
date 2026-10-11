import traceback
from ucimlrepo import fetch_ucirepo

try:
    fetch_ucirepo(id=45)
    print("OK")
except Exception:
    traceback.print_exc()