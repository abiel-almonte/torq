# Designing torq

## Staying small and nimble

TODO

## What are branches?

They are a means of assigning resources without being wasteful. 

Initialy, each node was being assigned with a potential CUDA steam prematurely, even before knowing which hardware the pipe depended on.

Establishing a principle that nodes *own* resources, like threads or streams. Resulting in the delegation of resources without the consideration of the system as a whole.

Branches offer this missing context. Think of them as **global sequential paths that span across the entire system.**

Now when we assign resources to a branch, any node on the branch has access to those resources without contention because branches are sequential.

When we create a system similar to the one in the `README`:
```python
system = tq.Sequential(
    tq.Concurrent(
       tq.Sequential(...), # branch 0
       tq.Sequential(...)  # branch 1
    ),
    sequential_fn1  # branch 0
)
```
We get the following branch structure:
``` 
             /——————— branch 0 ———————\
branch 0 ———|                          |——— branch 0
             \——————— branch 1 ———————/
```
So the first `Sequential(...)` block  and `sequential_fn1` both have access to the resources assigned to `branch 0` despite them being in different pipeline blocks.

After the `Concurrent` join, branch 1 terminates and its resources can be reused.

**Branches are a concept that keep the assignment of resources predictable and efficient.**


## What are pipes and why aren't they IR?
The is the biggest tension in `torq`: 
What are pipes, and why are they not considered intermediate representation (IR)?

### Short answer:

For IR to be IR, they must be nodes, and nodes exist in a graph.

Pipes do not exist in a graph. Therefore, pipes are not nodes, and by transitive property, not IR. 

### Long answer:

Pipelines/Pipes exist for one reason. They are declarative abstractions that communicate exactly *how* the system runs without burdening the user with assigning internal semantics to their entire system. 

In other words, the core abstractions define the system. While nodes enable optimization on the *defined* system, because they are context-aware—they know their own dependencies.

The pipes live in isolation—they only know how to run themselves, and rightly so. 

The user should only be concerned with how each individual piece of their system runs.

**How they are interconnected is the job for the compiler - `torq`.**

### but, the following questions arise:
If pipes are not IR, why does `torq`’s lowering process require pipe "materialization"? Why is materialization coupled in the compiler frontend if it’s not lowering? Why not just call it lowering? etc.

This is where the distinction between materialization and lowering becomes critical.

I can respond with “lowering is performed on a defined structure, and materialization is the process of defining such structure”. But, if thats the case, we are back where we started: *why is it required in lowering?* So that rationale is quite circular. 

So the question of whether pipes should be IR naturally resurfaces.

`torq`’s lowering needs to observe the behavior of these pipes to lower them into the appropriate nodes.

Each pipe is doing its job, *running*. The pipes do not know they are being lowered.

Materialization is just an artifact of an earlier, equally important design decision: **don't bother the user with internal semantics.**

If the boundary between materialization and lowering is ever dissolved, a pipe becomes just another abstraction of a node. Exposing `torq`’s job of “interconnecting” back to the user.

---

**The pipe abstraction is not IR. `torq` delegates the responsibility of defining the system to the user and understanding it to the compiler.**

## Why this matters

Lets take a look **solely** at how similar purposed software handle their **user interface**, namely NVIDIA's `DeepStream`.

An [object detection pipeline](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/master/apps/deepstream-test1/deepstream_test_1.py)  requires manual element creation,

```python
source = Gst.ElementFactory.make("filesrc", "file-source")
h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
...
```
manual pipeline management,
```python
pipeline.add(source)
pipeline.add(h264parser)
pipeline.add(decoder)
...
```
and manual graph wiring.
```python
srcpad.link(sinkpad)
streammux.link(pgie)
pgie.link(nvvidconv)
nvvidconv.link(nvosd)
nvosd.link(sink)
...
```
`DeepStream`'s approach contrasts with our design philosophy. And as we expect, it exposes a user interface where developers must assemble and connect individual nodes.

Whereas `torq` delegates interconnection to the compiler, where it belongs:

```python
system = tq.Sequential(
    video_source, h264_parser, decoder,
    inference_model,
    ...
    sink
)

tq.compile(system).run()
```

