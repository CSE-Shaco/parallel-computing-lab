"""Check every policy against a separate Python integer reference."""
import json
import subprocess
import sys
from pathlib import Path
exe = str(Path(sys.argv[1]).resolve())
MASK = (1 << 64) - 1

def task(index, iterations):
    x = index + 1
    for _ in range(iterations):
        x ^= x >> 12
        x ^= (x << 25) & MASK
        x ^= x >> 27
        x = (x * 2685821657736338717) & MASK
    return x

cases = 0
for n in [1, 7, 37]:
    for layout in ['uniform', 'tail']:
        costs = [13 if layout == 'tail' and i >= n - max(1, n//10) else 3 for i in range(n)]
        expected = sum(task(i, c) for i, c in enumerate(costs)) & MASK
        for policy in ['static', 'dynamic', 'guided', 'auto']:
            for threads in [1, 4, 8]:
                for chunk in ([0] if policy == 'auto' else [1, 10]):
                    row = json.loads(subprocess.check_output([exe, policy, str(threads), str(chunk), str(n), '3','13',layout], text=True))
                    assert row['verified'] and row['checksum'] == expected, row
                    assert sum(w['items'] for w in row['workers']) == n
                    assert sum(w['iterations'] for w in row['workers']) == sum(costs)
                    assert 1 <= row['actual_threads'] <= threads
                    cases += 1
for args in [('bogus','2','1','7','3','13','tail'), ('auto','2','1','7','3','13','tail'),
             ('static','0','1','7','3','13','tail'), ('static','2','0','7','3','13','tail'),
             ('static','2','1','0','3','13','tail'), ('static','2','1','7','3','13','bogus')]:
    r = subprocess.run([exe, *args], capture_output=True, text=True)
    assert r.returncode != 0, args
print(f'PASS: {cases} OpenMP reference/distribution cases and 6 invalid-input cases')
