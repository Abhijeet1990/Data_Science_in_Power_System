% read the first 5000 columns of original voltage data (in pu)
Volt_Data_pu = csvread('NASPI-2014-Workshop-Oscillation-Case1.csv',4,6,[4,6,5000,10]);

% plot original voltage data (in pu)
figure(1);
plot(Volt_Data_pu,'DisplayName','Volt_Data_pu');
grid on;

%---------- start of running PCA through SVD ----------%
Volt_Data_No_Mean_pu = detrend(Volt_Data_pu, 'constant'); % remove column means from per-unit voltage values

X = Volt_Data_No_Mean_pu';  % get X matrix

m = size(X,1);  % get number of measurement channels
n = size(X,2);  % get number of snapshots

[U,Sigma_X,V] = svd((1/sqrt(n-1)) * X);  % run SVD & get variances for X (original measurements)

P = U'; % get principle components (each row of P)

Y = P * X;  % get Y (projection of original measurements onto new basis P)

Sigma_Y = Sigma_X * Sigma_X';  % get variances for Y (projection of original measurements onto new basis P)
%----------- end of running PCA through SVD -----------%

% get projection of original measurements onto new basis P
Volt_Data_pu_Proj = Y';   % correspond to 'score' output of pca()

% plot projection of original measurements onto new basis P
figure(3);
plot(Volt_Data_pu_Proj,'DisplayName','Volt_Data_pu_Proj');
grid on;

% get principle components as each column of PC
PC = P';  % correspond to 'coeff' output of pca()

% plot projection of eigen direction on space of pc1 and pc2
figure(4);
for iter = 1:1:m
    plot([0,PC(iter,1)],[0,PC(iter,2)],'-x');
    hold on;
end;
grid on;
hold off;
% plot([0,PC(1,1)],[0,PC(1,2)],[0,PC(2,1)],[0,PC(2,2)],[0,PC(3,1)],[0,PC(3,2)],[0,PC(4,1)],[0,PC(4,2)],[0,PC(5,1)],[0,PC(5,2)]);

% get variances of projected measurements
Var_Y = diag(Sigma_Y);  % correspond to 'latent' output of pca()

% plot variances of projected measurements
figure(5);
bar(Var_Y);
grid on;

% get percentage of individual variances of projected measurements
Var_Individual_Percent = Var_Y/sum(Var_Y) * 100;

% plot percentage of individual variances of projected measurements
figure(6);
bar(Var_Individual_Percent);
grid on;

% get percentage of cumulative percentage of variances of projected measurements 
Var_Cumulative_Percent = zeros(m,1);  % correspond to 'explained' output of pca()
for iter = 1:1:length(Var_Y)
    Var_Cumulative_Percent(iter,1) = sum(Var_Individual_Percent(1:iter));
end;

% plot percentage of cumulative percentage of variances of projected measurements 
figure(7);
plot([0:1:m]',[0;Var_Cumulative_Percent],'-x');
grid on;

% plot 2D scatter plot on space of PC1 and PC2
figure(8);
a = 20;
c = linspace(1,10,length(Volt_Data_pu_Proj(:,1)));
scatter(Volt_Data_pu_Proj(:,1),Volt_Data_pu_Proj(:,2),a,c);
grid on;

% plot 3D scatter plot on space of PC1, PC2, and PC3
figure(9);
a = 20;
c = linspace(1,10,length(Volt_Data_pu_Proj(:,1)));
scatter3(Volt_Data_pu_Proj(:,1),Volt_Data_pu_Proj(:,2),Volt_Data_pu_Proj(:,3),a,c);

[coeff,score,latent,tsquared,explained,mu] = pca(Volt_Data_pu);

% figure(10);
% plot([0,coeff(1,1)],[0,coeff(1,2)],[0,coeff(2,1)],[0,coeff(2,2)],[0,coeff(3,1)],[0,coeff(3,2)],[0,coeff(4,1)],[0,coeff(4,2)],[0,coeff(5,1)],[0,coeff(5,2)]);