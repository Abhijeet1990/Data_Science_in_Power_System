%% set up parameters and inputs for ARMA, AR, MA models
a = [1 0.5];  % AR coeffs
b = [1 0.4 0.3];  % MA coeffs
T = 20000;  % sampling time

x_all = randn(T,1);  % generate gaussian white noise
x = x_all(1:T/2);
x_fore = x_all(T/2+1:end);

%% create time series using ARMA model
y_arma = filter(b,a,x);  % output of linear filter (mean = 0)
[c_arma, lags_arma]=xcov(y_arma,'biased');

% plot y_arma
figure;
plot(y_arma);
xlabel('t');
ylabel('y');
title('ARMA(1,2)');

%% create time series using AR model
y_ar = filter(1,a,x);
[c_ar, lags_ar]=xcov(y_ar,'biased');

% plot y_ar
figure;
plot(y_ar);
xlabel('t');
ylabel('y');
title('AR(1)');

%% create time series using MA model
y_ma = filter(b,1,x);
[c_ma, lags_ma]=xcov(y_ma,'biased');

% plot y_ma
figure;
plot(y_ma);
xlabel('t');
ylabel('y');
title('MA(1)');

%% identify AR, MA, ARMA models (p & q values)
% plot autocorr_arma and parcorr_arma
figure;
subplot(2,1,1);
autocorr(y_arma);
title('Sample Auto-Correlation of ARMA(1,2)');
subplot(2,1,2);
parcorr(y_arma);
title('Sample Partial-Correlation of ARMA(1,2)');

% plot autocorr_ar
figure;
subplot(2,1,1);
autocorr(y_ar);
title('Sample Auto-Correlation of AR(1)');
subplot(2,1,2);
parcorr(y_ar);
title('Sample Partial-Correlation of AR(1)');

% plot autocorr_ma
figure;
subplot(2,1,1);
autocorr(y_ma);
title('Sample Auto-Correlation of MA(2)');
subplot(2,1,2);
parcorr(y_ma);
title('Sample Partial-Correlation of MA(2)');

%% estimate ARMA model parameters (AR coeffs & MA coeffs)
p = 1;  % # of AR coeffs
q = 2;  % # of MA coeffs

% ARMA(1,2)
ARIMA_Model_1_2 = arima(1,0,2);
[ARIMA_Model_Fitted_1_2,EstParamCov_1_2,logL_1_2,info_1_2] = estimate(ARIMA_Model_1_2,y_arma);

% ARMA(5,5)
ARIMA_Model_5_5 = arima(5,0,5);
[ARIMA_Model_Fitted_5_5,EstParamCov_5_5,logL_5_5,info_5_5] = estimate(ARIMA_Model_5_5,y_arma);

%% compare ARMA(1,2) and ARMA(5,5)

% estimated ARMA(1,2)
a_1_2 = [1, -cell2mat(ARIMA_Model_Fitted_1_2.AR)];
b_1_2 = [1, cell2mat(ARIMA_Model_Fitted_1_2.MA)];

y_1_2_fore = filter(b_1_2,a_1_2,x_fore) + ARIMA_Model_Fitted_1_2.Constant;  % output of linear filter (mean = 0)
y_1_2 = filter(b_1_2,a_1_2,x) + ARIMA_Model_Fitted_1_2.Constant;  % output of linear filter (mean = 0)

% estimated ARMA(5,5)
a_5_5 = [1, -cell2mat(ARIMA_Model_Fitted_5_5.AR)];
b_5_5 = [1, cell2mat(ARIMA_Model_Fitted_5_5.MA)];

y_5_5_fore = filter(b_5_5,a_5_5,x_fore) + ARIMA_Model_Fitted_5_5.Constant;  % output of linear filter (mean = 0)
y_5_5 = filter(b_5_5,a_5_5,x) + ARIMA_Model_Fitted_5_5.Constant;  % output of linear filter (mean = 0)

% original ARMA(1,2)
y_fore = filter(b,a,x_fore);  % output of linear filter (mean = 0)

% plot fitting errors
err_1_2_fit = (y_1_2-y_arma)./y_arma;
err_5_5_fit= (y_5_5-y_arma)./y_arma;

figure;
plot(err_1_2_fit,'b');
title('Fitting Errors of Estimated ARMA(1,2) Model');
figure;
plot(err_5_5_fit,'r');
title('Fitting Errors of Estimated ARMA(5,5) Model');

ave_err_1_2_fit = sum(abs((y_1_2-y_arma)./y_arma))/(T/2)
ave_err_5_5_fit = sum(abs((y_5_5-y_arma)./y_arma))/(T/2)

% plot forecasting errors
err_1_2 = (y_1_2_fore-y_fore)./y_fore;
err_5_5 = (y_5_5_fore-y_fore)./y_fore;

figure;
plot(err_1_2,'b');
title('Forecasting Errors of Estimated ARMA(1,2) Model');
figure;
plot(err_5_5,'r');
title('Forecasting Errors of Estimated ARMA(5,5) Model');

ave_err_1_2 = sum(abs((y_1_2_fore-y_fore)./y_fore))/(T/2)
ave_err_5_5 = sum(abs((y_5_5_fore-y_fore)./y_fore))/(T/2)