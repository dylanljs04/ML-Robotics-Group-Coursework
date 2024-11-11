
m = lw{2,1}*max(0,iw{1,1}.*x+b{1,1});

sim = lw{2,1}*max(0,iw{1,1}.*x+b{1,1})+b{2,1};
plot(x,sim)