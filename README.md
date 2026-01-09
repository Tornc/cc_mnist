# Computercraft ML library adapted from C

All of the ML-related code is based on [**Magicalbat**](https://www.youtube.com/@Magicalbat)'s youtube video. Please go watch it, he's absolutely cracked.

[![coding a machine learning library in c from scratch](https://img.youtube.com/vi/hL_n_GljC0I/0.jpg)](https://www.youtube.com/watch?v=hL_n_GljC0I)

Corresponding [**GitHub repository**](https://github.com/Magicalbat/videos/tree/main/machine-learning).

> [!WARNING]
> Compared to the C version, the matrix math is hilariously slow. Use of **CraftOS-PC Accelerated (i.e. LuaJIT)** is highly advised.

The specific MNIST dataset used is from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/data.

### TODO
- [ ] Finish the damn thing
- [ ] Save state to disk at every epoch
- [ ] Test the performance of pregenerating hard-coded matrix math functions for specific sizes and save it to disk to re-use
- [ ] Make it work with CraftOS/inside Minecraft (yield-spam)