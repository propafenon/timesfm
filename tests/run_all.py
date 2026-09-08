"""Run every test module in this directory. Exit non-zero on the first failure."""
import glob, os, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
failed = []
for path in sorted(glob.glob(os.path.join(here, 'test_*.py'))):
    name = os.path.basename(path)
    result = subprocess.run([sys.executable, path], capture_output=True, text=True)
    tail = (result.stdout.strip().splitlines() or ["(no output)"])[-1]
    print(f"{name:<20} {'PASS' if result.returncode == 0 else 'FAIL'}  {tail}")
    if result.returncode != 0:
        failed.append(name)
        print(result.stdout[-2000:], result.stderr[-2000:], sep="\n")

print(f"\n{len(failed)} failed" if failed else "\nall suites passed")
sys.exit(1 if failed else 0)
