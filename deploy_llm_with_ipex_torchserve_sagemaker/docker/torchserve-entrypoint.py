from __future__ import absolute_import

import shlex
import subprocess
import sys


if sys.argv[1] == "serve":
    from sagemaker_pytorch_serving_container import serving

    serving.main()
else:
    subprocess.check_call(shlex.split(" ".join(sys.argv[1:])))

# prevent docker exit
subprocess.call(["tail", "-f", "/dev/null"])
