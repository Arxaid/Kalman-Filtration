% This file is part of the Statistical Dynamics coursework.
%
% Copyright (c) 2018 Vladislav Sosedov.

clear all
close all
clc

global dt T Tend                % Global time
dt = 1;
Tend = 1000;
T = 0:dt:Tend;

% Input params

Xb(:,1) = [0;0];                % Starting boat coordinates
Xh(:,1) = [15000;2500];         % Starting helicopter coordinates

M_Vw = [2; 7];                  % Wind disturbance (mathematical expectation)
Sigma_Vw = sqrt(1);             % Wind disturbance (sigma)
M_Vs = [0.95; 0.7];             % Waves disturbance (mathematical expectation)
Sigma_Vs = sqrt(0.25);          % Waves disturbance (sigma)
D_xi = Sigma_Vs^2 + Sigma_Vw^2; % Disturbance matrix

Sigma_Y = sqrt(1);              % Measurement (sigma)
D_eta = [[Sigma_Y^2 0];         % Cov matrix
         [0 Sigma_Y^2]];
D_eta_inv = D_eta^(-1);         % Inverse cov matrix

H = [[1 0 0 0];                 % H matrix
     [0 1 0 0]];
A = [[0 0 1 0];                 % Dynamical matrix
     [0 0 0 1];
     [0 0 0 0];
     [0 0 0 0]];
L = [[0 0];                     % Disturbance L matrix
     [0 0];
     [1 0];
     [0 1]];
F = eye(4,4) + dt*A;            % Fundamental matrix
V = [[dt 0];                    % Control matrix
     [0 dt];
     [0 0];
     [0 0]];

Xp = [0; 0; 0; 0];              % A priori estimations
Pp = [[1e3 0 0 0]; 
      [0 1e3 0 0]; 
      [0 0 1e3 0]; 
      [0 0 0 1e3]];

% Main Kalman filter loop

for i = 1:length(T)
    
    Vs(:,i) = M_Vs + [Sigma_Vs*randn(); Sigma_Vs*randn()];      % True wind disturbance model
    Vw(:,i) = M_Vw + [Sigma_Vw*randn(); Sigma_Vw*randn()];      % True waves disturbance model
    
    Y = Xb(:,i) - Xh(:,i) + [Sigma_Y*randn(); Sigma_Y*randn()]; % Measurement model
 
    P(:,:,i) = (Pp^(-1) + H'*(D_eta^(-1))*H)^(-1);              % Correction via filter
    X(:,i) = Xp + P(:,:,i) * H'*(D_eta^(-1))*(Y - H*Xp);
    
    if (i == length(T))
        break;
    end;
    
    Vh = [X(1,i); X(2,i)]/(2*dt) + [X(3,i);X(4,i)];             % Helicopter control velocity 
    if ((sqrt(Vh(1)^2 + Vh(2)^2)) > 250)                        % Max velocity check
        Vh = Vh * 250/(sqrt(Vh(1)^2 + Vh(2)^2));
    end;
    
    Xb(:,i+1) = Xb(:,i) + Vs(:,i)*dt;                           % True boat model
    Xh(:,i+1) = Xh(:,i) + (Vh + Vw(:,i))*dt;                    % True helicopter model

    Pp = F*P(:,:,i)*F' + L*[[D_xi 0];[0 D_xi]]*L';              % Next iteration predict
    Xp = F*X(:,i) + V*(-Vh);
    
end;

% True boat and helicopter trajectory

figure('Name', 'True boat and helicopter traectory')
plot (Xb(1,:), Xb(2,:), Xh(1,:), Xh(2,:))
xlabel('X Coord, m')
ylabel('Y Coord, m')
legend('Boat','Helicopter')

% X1 region

X1_true = Xb(1,:) - Xh(1,:);
X1 = reshape(X(1,:), length(X(1,:)), 1);

figure('Name', '[X1] True value and measurement')
plot (T, X1_true, T, X1'), grid
xlabel('t, sec')
ylabel('Xbx - Xhx, m')
legend('True value X1','Measurement X1')

P1 = reshape(P(1,1,:), length(P(1,1,:)), 1);
S1 = 3*sqrt(P1);

figure('Name', '[X1] Delta and tube')
plot (T, X1_true - X1', T, S1, T, -S1), grid
xlabel('t, sec')
ylabel('delta1, m')
legend('Delta X1','Tube X1')

% X2 region

X2_true = Xb(2,:) - Xh(2,:);
X2 = reshape(X(2,:), length(X(2,:)), 1);

figure('Name', '[X2] True value and measurement')
plot (T, X2_true, T, X2'), grid
xlabel('t, sec')
ylabel('Xby - Xhy, m')
legend('True value X2','Measurement X2')

P2 = reshape(P(2,2,:), length(P(2,2,:)), 1);
S2 = 3*sqrt(P2);

figure('Name', '[X2] Delta and tube')
plot (T, X2_true - X2', T, S2, T, -S2), grid
xlabel('t, sec')
ylabel('delta2, m')
legend('Delta X2','Tube X2')

% X3 region

X3_true = Vs(1,:) - Vw(1,:);
X3 = reshape(X(3,:), length(X(3,:)), 1);

figure('Name', '[X3] True value and measurement')
plot (T, X3_true, T, X3'),grid
xlabel('t, sec')
ylabel('Vsx - Vwx, m/sec')
legend('True value X3','Measurement X3')

P3 = reshape(P(3,3,:), length(P(3,3,:)), 1);
S3 = 3*sqrt(P3);

figure('Name', '[X3] Delta and tube')
plot (T, X3_true - X3', T, S3, T, -S3), grid
xlabel('t, sec')
ylabel('delta3, m/sec')
legend('Delta X3','Tube X3')

% X4 region

X4_true = Vs(2,:) - Vw(2,:);
X4 = reshape(X(4,:), length(X(4,:)), 1);

figure('Name', '[X4] True value and measurement')
plot (T, X4_true, T, X4'), grid
xlabel('t, sec')
ylabel('Vsy - Vwy, m/sec')
legend('True value X4','Measurement X4')

P4 = reshape(P(4,4,:), length(P(4,4,:)), 1);
S4 = 3*sqrt(P4);

figure('Name', '[X4] Delta and tube')
plot (T, X4_true - X4', T, S4, T, -S4), grid
xlabel('t, sec')
ylabel('delta4, m/sec')
legend('Delta X4','Tube X4')

% P matrix evolution

figure('Name', '[P1] P matrix evolution')
plot (reshape(P(1,1,:), length(P(1,1,:)), 1), 'black')
xlabel('t, s')
ylabel('P1, (m)^2')

figure('Name', '[P2] P matrix evolution')
plot (reshape(P(2,2,:), length(P(2,2,:)), 1), 'black')
xlabel('t, c')
ylabel('P2, (m)^2')

figure('Name', '[P3] P matrix evolution')
plot (reshape(P(3,3,:), length(P(3,3,:)), 1), 'black')
xlabel('t, c')
ylabel('P3, (m/sec)^2')

figure('Name', '[P4] P matrix evolution')
plot (reshape(P(4,4,:), length(P(4,4,:)), 1), 'black')
xlabel('t, c')
ylabel('P4, (m/sec)^2')
