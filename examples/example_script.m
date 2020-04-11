%% load example data
clear
clc
load('example_data.mat')
%% fit ITC data with 2-piece CBS function
% each row is a choice between option 1 (Amt at Delay) vs option 2 (20 now).
Amount1 = ITCdat.Amt1;
Delay1 = ITCdat.Delay1;
Amount2 = 20;
Delay2 = 0;
Choice = ITCdat.Choice;

% fit the model
out = CBS_ITC(Choice,Amount1,Delay1,Amount2,Delay2,2);

% plot the choices (x = Delay, y = relative amount : 20 / delayed amount)
figure
plot(Delay1(Choice==1),20./Amount1(Choice==1),'bo')
hold on
plot(Delay1(Choice==0),20./Amount1(Choice==0),'ro')

% plot the fitted CBS
x = 0:180;
plot(x,CBSfunc(out.xpos,out.ypos,x));

%% fit Risky choice data with 2-piece CBS function
% each row is a choice between option 1 (Amt with prob) vs option 2 (20 for 100%).
Amount1 = RCdat.Amt1;
Prob1 = RCdat.Prob1;
Amount2 = 20;
Prob2 = 1;
Choice = RCdat.Choice;

% fit the model
out = CBS_RC(Choice,Amount1,Prob1,Amount2,Prob2,2);

% plot the choices (x = Probability, y = relative amount : 20 / delayed amount)
figure
plot(Prob1(Choice==1),20./Amount1(Choice==1),'bo')
hold on
plot(Prob1(Choice==0),20./Amount1(Choice==0),'ro')

% plot the fitted CBS
x = 0:.01:1;
plot(x,CBSfunc(out.xpos,out.ypos,x));