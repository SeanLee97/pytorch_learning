由relu函数可值最小值为0
故
对于numpy
h_relu = np.maximum(h, 0)

对于tensor
h_relu = torch.clamp(min=0)
