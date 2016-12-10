#for a neural network object nw ...
nw = 

figure(1); hold(True)    
delta = 0.05; levels = 100
a = arange(-1,2,delta)
b = arange(-1,2,delta)
A, B = meshgrid(a, b)
values = np.zeros(A.shape)

for i in range(len(a)):
    for j in range(len(b)):
        values[i,j] = nw.sim( np.mat([a[i],b[j]]) )[0,0]
contour(A, B, values, levels=[.5], colors=['k'], linestyles='dashed')
contourf(A, B, values, levels=linspace(values.min(),values.max(),levels), cmap=cm.RdBu)
