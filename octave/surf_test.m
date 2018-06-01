function surf_test()
x = 2:0.2:40;
y = 1:0.2:30;
[X,Y] = meshgrid(x,y);
Z = (X).^3 - (Y).^3;
surf(X,Y,Z)
