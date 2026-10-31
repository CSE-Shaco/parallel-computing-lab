"""Independent integer oracle plus cross-thread and invalid-input checks."""
import json
import subprocess
import sys
from pathlib import Path

exe = str(Path(sys.argv[1]).resolve())
MASK = (1 << 64) - 1

def mix(x):
    x = (x + 0x9e3779b97f4a7c15) & MASK
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & MASK
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & MASK
    return x ^ (x >> 31)

def reference(n, seed):
    return sum((mix((seed + 2*i) & MASK) >> 38)**2 +
               (mix((seed + 2*i + 1) & MASK) >> 38)**2 <= 1 << 52
               for i in range(n))

cases = 0
for n, seed in [(1, 0), (7, 42), (10003, 123), (257, MASK)]:
    expected = reference(n, seed)
    for mode in ['serial', 'local', 'mutex']:
        for t in ([1] if mode == 'serial' else [1, 2, 4, 8, 16, 32]):
            row = json.loads(subprocess.check_output([exe, mode, str(t), str(n), str(seed)], text=True))
            assert row['count'] == expected, row
            assert 0 <= row['count'] <= n and row['ms'] >= 0
            cases += 1
for args in [('local','0','5','1'), ('local','2','0','1'), ('serial','2','5','1'),
             ('race','2','5','1'), ('local','-1','5','1'), ('local','999999999999','5','1'),
             ('local','2','12junk','1'), ('local','2','5','-1')]:
    r = subprocess.run([exe, *args], capture_output=True, text=True)
    assert r.returncode != 0, args
print(f'PASS: {cases} exact Monte Carlo comparisons and 8 invalid-input cases')
