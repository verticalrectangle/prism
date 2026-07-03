#!/usr/bin/env python3
"""
Patch a wav2vec2 ONNX to expose hidden_states alongside logits.

The Xenova/wav2vec2-base-960h ONNX only exports "logits" (CTC head output).
This script traces back through the CTC projection (Add + MatMul) following
the non-initializer data path to find the 768-dim encoder output, then adds
it as a named graph output "hidden_states" via an Identity node.

Usage:
  python3 tools/patch_wav2vec2_onnx.py <input.onnx> <output.onnx>
"""

import sys
import onnx
from onnx import helper, TensorProto, shape_inference

if len(sys.argv) != 3:
    sys.exit(f"Usage: {sys.argv[0]} <input.onnx> <output.onnx>")

src, dst = sys.argv[1], sys.argv[2]

model = onnx.load(src)
model = shape_inference.infer_shapes(model)
graph = model.graph

if any(o.name == "hidden_states" for o in graph.output):
    print("hidden_states already present — copying as-is.")
    onnx.save(model, dst)
    sys.exit(0)

initializers = {i.name for i in graph.initializer}
nodes_by_output = {o: node for node in graph.node for o in node.output}

def non_init_input(node):
    """Return the first input of node that is not an initializer (constant)."""
    for inp in node.input:
        if inp and inp not in initializers:
            return inp
    return None

# Walk back from the logits output through the CTC projection to the encoder output.
# Xenova graph: logits ← Add(bias, MatMul(hidden, weight)) ← encoder_output
logits_name = graph.output[0].name
tensor_name = logits_name
for _ in range(8):  # bounded walk — encoder output is always within a few hops
    node = nodes_by_output.get(tensor_name)
    if node is None:
        break
    nxt = non_init_input(node)
    if nxt is None:
        break
    # Stop when the producing node is NOT part of the lm_head
    # (heuristic: node name no longer contains "lm_head")
    if "lm_head" not in (node.name or ""):
        break
    tensor_name = nxt

if tensor_name == logits_name:
    sys.exit("Could not trace encoder output — graph structure not recognised.")

print(f"Encoder output tensor: {tensor_name!r}")

graph.node.append(helper.make_node(
    "Identity",
    inputs=[tensor_name],
    outputs=["hidden_states"],
    name="expose_hidden_states"))

graph.output.append(
    helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, [1, None, 768]))

onnx.checker.check_model(model)
onnx.save(model, dst)
print(f"Saved {dst}  outputs: {[o.name for o in model.graph.output]}")
