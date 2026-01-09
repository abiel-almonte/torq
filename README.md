`torq` **- Graph compiler for deterministic inference. (WIP)**

Build complex systems from simple pieces:
```python
import torq as tq

# Compose a multi-camera fusion pipeline
system = tq.Sequential(
    tq.Concurrent(
        tq.Sequential(rgb_camera, rgb_preprocess, rgb_model),
        tq.Sequential(depth_camera, depth_preprocess, depth_model)
    ),
    fusion_model,
    console_writer
)

# Build DAG, optimize lazily
system = tq.compile(system)
system.run(iters=-1)
```

Teach torq how to call your own objects:
```python
from visionrt import Camera

tq.register(cls=Camera, adapter=lambda x: next(x.stream()))

system = tq.Sequential(
    Camera("/dev/video0"),
    ...
)
```
