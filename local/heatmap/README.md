# E2E Heatmap Calculations

This module includes code to calculate and visualize post-hoc attention maps on CNNs, that is, "hetmaps" showing where in the input video or image the network was "paying attention".

## Note on Choosing a Label

Most implemented methods have some variant of the following:

```python
if is_binary:
    if target_value < 0.5:
        output = -output
else:
    output = output[..., class_idx]

output.backward()
```

* If the classifier is *multi-class*, this is straightforward: we choose the score for the target class and *only backpropagate that*.
* If the classifier is *binary*, this is messy, because *we only have one score!*

Luckily, a bit of math comparing the logistic function $l(z) = 1 / (1 + e^{-z})$ and the softmax function $s(\mathbf{z}) = e^\mathbf{z} / ||e^\mathbf{z}||_1$ for two classes $\mathbf{z} = (z_0, z_1)$ lets us show that

$$ l(z') = s(\mathbf{z}) \quad \Leftrightarrow \quad z' = z_1 - z_0, $$

and we have one degree of freedom. Choosing $z_0 = -z_1$ for convenience,

$$ z_0 = -\frac{z'}{2}, \quad z_1 = \frac{z'}{2}. $$

Up to scaling, this corresponds to our code.

It is possible that we would obtain more significant class differences if we used a multi-class head, since we are essentially backpropagating the same value. However, sign matters in the modified backpasses used by Grad-CAM and guided backpropagation, so not all information is lost using a single score.

## Note on Backprop Methods

Most implemented methods use some variant of the backpropagation algorithm to calculate their attention maps. In our PyTorch implementation, this all relies on *module hooks*:

```python
def forward_hook(module: nn.Module, inputs: Tuple[Tensor, ...], output: Tensor) -> Tensor | None:
    ...

target_module.register_forward_hook(forward_hook)
```

*Forward hooks* are called on the *forward* pass of the module, i.e., when one calls `target_module(data)`. Under normal conditions, the module computes `output = module(*inputs)`. If the hook returns a tensor, the returned value is used to substitute `output` in the forward chain. Since the module is free to have more than one input or output, one should be careful with the exact nature of `input` and `output`.

```python
def backward_hook(module: nn.Module, input_gradients: Tuple[Tensor, ...], output_gradients: Tuple[Tensor, ...]) -> Tuple[Tensor, ...] | None:
    ...

target_module.register_backward_hook(backward_hook)
```
*Backward hooks* are called on the *backward* pass of the module, i.e., when one calls `final_output.backward()`. Under normal conditions, the module computes `input_gradients = module.backward(*output_gradients)`. If the hook returns a tuple of tensors, the returned values are used to substitute `input_gradients` in the backward chain.

* [Here is an interesting article](https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/) discussing how this works and why the author is opposed to using *module* hooks, instead proposing to use *tensor* hooks.
* Note that `Module.register_backward_hook()` is officially deprecated in favor of `Module.register_full_backward_hook()`, but *the new variant fails if there are in-place operations in the network*. This makes our current implementations fail on out-of-the-box `torchvision` networks because of activations being performed in-place. It could be solved by substituting all `ReLU` modules with a modified `ReLU` class, though that seems heavy-handed.
