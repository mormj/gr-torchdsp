import subprocess
import os

for dir in os.listdir("models/triton"):
    p = os.path.join("models/triton", dir)
    print("Making model in {}".format(p))
    if (not os.path.exists(p)):
        subprocess.call(
            "mkdir 1",
            cwd=p,
            shell=True
        )
    subprocess.call(
        "python3 make_model.py",
        cwd=p,
        shell=True
    )
