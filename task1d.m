clear
x = -1:0.05:1;
y = 0.8*x.^3 + 0.3*x.^2-0.4*x;

net = newff(minmax(x),[5,1],{'poslin','purelin'},"traingdx");
net.trainparam.show=50;
net.trainparam.lr=0.001;
net.trainparam.epochs=5000;
net.trainparam.goal=1e-6;

[net,tr] = train(net,x,y);
iw = net.iw;
lw = net.lw;
b = net.b;

xtest = -0.95:0.05:0.95;
net_output = sim(net,xtest);

simulated_values = lw{2,1}*max(0,iw{1,1}*x+b{1,1})+b{2,1};

figure(1)
plot(xtest,net_output)
hold on
plot(x,y)
plot(x,simulated_values,"-k")

hold off

figure(2)
relus = transpose(lw{2,1}).*max(0,iw{1,1}*x+b{1,1})+b{2,1};
plot(x,relus(1,:),"-r")
hold on
plot(x,relus(2,:),"-g")
plot(x,relus(3,:),"-b")
plot(x,relus(4,:),"-y")
plot(x,relus(5,:),"-k")
legend("relu 1", "relu 2", "relu 3", "relu 4", "relu 5")
hold off