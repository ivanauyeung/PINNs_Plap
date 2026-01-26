#PhysicsNeMo on p-Laplacian problem

#### Docker Container

The recommended PhysicsNeMo Docker image can be pulled from the NVIDIA Container Registry
`sudo pull nvcr.io/nvidia/physicsnemo/physicsnemo:25.06`

and to run the container, run:
`sudo docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -p 8888:8888 -it --rm nvcr.io/nvidia/physicsnemo/physicsnemo:25.06 bash`

The container launches jupyter lab and runs on port 8888
`jupyter-lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root`

Then, open the jupyter lab in browser: http://localhost:8888

#### PyPI

The recommended method for installing the latest version of PhysicsNeMo is using PyPI:

`pip install nvidia-physicsnemo`

`pip install nvidia-physicsnemo-sym --no-build-isolation`
