function surf_test()
x = 2:0.2:4;
y = 1:0.2:3;
[X,Y] = meshgrid(x,y);
Z = (X-3).^2 - (Y-2).^2;
surf(X,Y,Z)
