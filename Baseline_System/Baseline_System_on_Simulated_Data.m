clear all;
close all;
clc;

% *************************************************************************
% Simulate Signal and Noise
fs = 8000; % Sampling frequency
t = 0:1/fs:2; % Time vector (2 seconds)
signal = sin(2 * pi * 10 * t)'; % Simulated clean signal (10 Hz sine wave)
noise = 0.5 * randn(size(t))'; % Simulated Gaussian noise
noisy_signal = signal + noise; % Combine signal and noise

% Calculate initial SNR
initial_SNR = 10 * log10(sum(signal.^2) / sum(noise.^2));
fprintf('Initial SNR: %.2f dB\n', initial_SNR);

% Filter Parameters
N = length(noisy_signal); % Signal length
M = 16; % Filter order

% *************************************************************************
% LMS Filter
mu_LMS = 0.01; % Step size for LMS
w_LMS = zeros(M, 1); % Initialize filter weights
padded_signal = [zeros(M-1, 1); noisy_signal]; % Pad noisy signal
output_LMS = zeros(N, 1); % Filter output

for n = 1:N
    u_vect = padded_signal(n:n+M-1); % Current input vector
    e = signal(n) - w_LMS' * u_vect; % Error signal
    w_LMS = w_LMS + mu_LMS * e * u_vect; % Update weights
    output_LMS(n) = w_LMS' * u_vect; % Filtered output
end

% *************************************************************************
% NLMS Filter
mu_NLMS = 1; % Step size for NLMS
w_NLMS = zeros(M, 1); % Initialize filter weights
output_NLMS = zeros(N, 1); % Filter output
Eps = 1e-6; % Stability constant

for n = 1:N
    u_vect = padded_signal(n:n+M-1); % Current input vector
    mu_adapt = mu_NLMS / (Eps + norm(u_vect)^2); % Adaptive step size
    e = signal(n) - w_NLMS' * u_vect; % Error signal
    w_NLMS = w_NLMS + mu_adapt * e * u_vect; % Update weights
    output_NLMS(n) = w_NLMS' * u_vect; % Filtered output
end

% *************************************************************************
% RLS Filter
lambda = 0.99; % Forgetting factor
delta = 1e-2; % Initialization constant
P = (1 / delta) * eye(M); % Initialize inverse correlation matrix
w_RLS = zeros(M, 1); % Initialize filter weights
output_RLS = zeros(N, 1); % Filter output

for n = 1:N
    u_vect = padded_signal(n:n+M-1); % Current input vector
    gain_k = P * u_vect / (lambda + u_vect' * P * u_vect); % Gain vector
    e = signal(n) - w_RLS' * u_vect; % Error signal
    w_RLS = w_RLS + gain_k * e; % Update weights
    P = (P - gain_k * u_vect' * P) / lambda; % Update P matrix
    output_RLS(n) = w_RLS' * u_vect; % Filtered output
end

% *************************************************************************
% Calculate SNR Improvement
filtered_SNR_LMS = 10 * log10(sum(signal.^2) / sum((signal - output_LMS).^2));
filtered_SNR_NLMS = 10 * log10(sum(signal.^2) / sum((signal - output_NLMS).^2));
filtered_SNR_RLS = 10 * log10(sum(signal.^2) / sum((signal - output_RLS).^2));

fprintf('SNR after LMS Filter: %.2f dB\n', filtered_SNR_LMS);
fprintf('SNR after NLMS Filter: %.2f dB\n', filtered_SNR_NLMS);
fprintf('SNR after RLS Filter: %.2f dB\n', filtered_SNR_RLS);

% *************************************************************************
% Visualization of Results
figure;
subplot(5, 1, 1);
plot(signal);
title('Clean Signal');
xlabel('Sample Number');
ylabel('Amplitude');
xlim([0 16000]);

subplot(5, 1, 2);
plot(noisy_signal);
title('Noisy Signal');
xlabel('Sample Number');
ylabel('Amplitude');
xlim([0 16000]);

subplot(5, 1, 3);
plot(output_LMS);
title('Filtered Signal (LMS)');
xlabel('Sample Number');
ylabel('Amplitude');
xlim([0 16000]);

subplot(5, 1, 4);
plot(output_NLMS);
title('Filtered Signal (NLMS)');
xlabel('Sample Number');
ylabel('Amplitude');
xlim([0 16000]);

subplot(5, 1, 5);
plot(output_RLS);
title('Filtered Signal (RLS)');
xlabel('Sample Number');
ylabel('Amplitude');
xlim([0 16000]);

% *************************************************************************
% Normalize filtered signals
output_LMS = output_LMS / max(abs(output_LMS));
output_NLMS = output_NLMS / max(abs(output_NLMS));
output_RLS = output_RLS / max(abs(output_RLS));

% Save Filtered Signals
audiowrite('Filtered_LMS.wav', output_LMS, fs);
audiowrite('Filtered_NLMS.wav', output_NLMS, fs);
audiowrite('Filtered_RLS.wav', output_RLS, fs);

fprintf('Filtered signals saved as audio files.\n');

% Crop the figure and save as PDF
tightfig();
saveas(gcf, 'filtered_signals_comparison.pdf');

function tightfig()
    % Tighten the figure by removing excess whitespace
    set(gcf, 'Units', 'Inches');
    pos = get(gcf, 'Position');
    set(gcf, 'PaperUnits', 'Inches');
    set(gcf, 'PaperSize', [pos(3) pos(4)]);
    set(gcf, 'PaperPosition', [0 0 pos(3) pos(4)]);
end

%%
clear all;
close all;
clc;

% *************************************************************************
% Simulate Signal and Noise
fs = 8000; % Sampling frequency
t = 0:1/fs:2; % Time vector (2 seconds)
signal = sin(2 * pi * 10 * t)'; % Simulated clean signal (10 Hz sine wave)

% Generate noise at -20 dB SNR
signal_power = mean(signal.^2);
SNR_dB = -20; % Desired SNR in dB
SNR_linear = 10^(SNR_dB/10);
noise_power = signal_power / SNR_linear;
noise = sqrt(noise_power) * randn(size(t))';

noisy_signal = signal + noise; % Combine signal and noise

% Calculate initial SNR
initial_SNR = 10 * log10(sum(signal.^2) / sum(noise.^2));
fprintf('Initial SNR: %.2f dB\n', initial_SNR);

% Filter Parameters
N = length(noisy_signal); % Signal length
M = 12; % Filter order (adjusted to match first code)

% *************************************************************************
% LMS Filter
mu_LMS = 0.001; % Step size for LMS (adjusted from first code)
w_LMS = randn(M, 1); % Initialize filter weights
padded_signal = [zeros(M-1, 1); noisy_signal]; % Pad noisy signal
output_LMS = zeros(N, 1); % Filter output

for n = 1:N
    u_vect = padded_signal(n:n+M-1); % Current input vector
    e = signal(n) - w_LMS' * u_vect; % Error signal
    w_LMS = w_LMS + mu_LMS * e * u_vect; % Update weights
    output_LMS(n) = w_LMS' * u_vect; % Filtered output
end

% *************************************************************************
% NLMS Filter
mu_NLMS = 1; % Step size for NLMS (kept from first code)
w_NLMS = randn(M, 1); % Initialize filter weights
output_NLMS = zeros(N, 1); % Filter output
Eps = 0.0001; % Stability constant (adjusted from first code)

for n = 1:N
    u_vect = padded_signal(n:n+M-1); % Current input vector
    mu_adapt = mu_NLMS / (Eps + norm(u_vect)^2); % Adaptive step size
    e = signal(n) - w_NLMS' * u_vect; % Error signal
    w_NLMS = w_NLMS + mu_adapt * e * u_vect; % Update weights
    output_NLMS(n) = w_NLMS' * u_vect; % Filtered output
end

% *************************************************************************
% RLS Filter
lambda = 1 - 1 / (0.1 * M); % Forgetting factor (adjusted from first code)
delta = 0.01; % Initialization constant (adjusted from first code)
P = (1 / delta) * eye(M); % Initialize inverse correlation matrix
w_RLS = randn(M, 1); % Initialize filter weights
padded_signal = [sqrt(delta) * randn(M-1, 1); noisy_signal]; % Pad noisy signal
output_RLS = zeros(N, 1); % Filter output

for n = 1:N
    u_vect = padded_signal(n:n+M-1); % Current input vector
    PI = P * u_vect; % Intermediate calculation
    gain_k = PI / (lambda + u_vect' * PI); % Gain
    e = signal(n) - w_RLS' * u_vect; % Error signal
    w_RLS = w_RLS + gain_k * e; % Update weights
    P = P / lambda - gain_k * (u_vect' * P) / lambda; % Update P matrix
    output_RLS(n) = w_RLS' * u_vect; % Filtered output
end

% *************************************************************************
% Calculate SNR Improvement
filtered_SNR_LMS = 10 * log10(sum(signal.^2) / sum((signal - output_LMS).^2));
filtered_SNR_NLMS = 10 * log10(sum(signal.^2) / sum((signal - output_NLMS).^2));
filtered_SNR_RLS = 10 * log10(sum(signal.^2) / sum((signal - output_RLS).^2));

fprintf('SNR after LMS Filter: %.2f dB\n', filtered_SNR_LMS);
fprintf('SNR after NLMS Filter: %.2f dB\n', filtered_SNR_NLMS);
fprintf('SNR after RLS Filter: %.2f dB\n', filtered_SNR_RLS);

% *************************************************************************
% Visualization of Results
figure;
subplot(5, 1, 1);
plot(signal);
title('Clean Signal');
xlabel('Sample Number');
ylabel('Amplitude');
xlim([0 16000]);

subplot(5, 1, 2);
plot(noisy_signal);
title('Noisy Signal');
xlabel('Sample Number');
ylabel('Amplitude');
xlim([0 16000]);

subplot(5, 1, 3);
plot(output_LMS);
title('Filtered Signal (LMS)');
xlabel('Sample Number');
ylabel('Amplitude');
xlim([0 16000]);

subplot(5, 1, 4);
plot(output_NLMS);
title('Filtered Signal (NLMS)');
xlabel('Sample Number');
ylabel('Amplitude');
xlim([0 16000]);

subplot(5, 1, 5);
plot(output_RLS);
title('Filtered Signal (RLS)');
xlabel('Sample Number');
ylabel('Amplitude');
xlim([0 16000]);

% *************************************************************************
% Normalize filtered signals
output_LMS = output_LMS / max(abs(output_LMS));
output_NLMS = output_NLMS / max(abs(output_NLMS));
output_RLS = output_RLS / max(abs(output_RLS));

% Save Filtered Signals
audiowrite('Filtered_LMS.wav', output_LMS, fs);
audiowrite('Filtered_NLMS.wav', output_NLMS, fs);
audiowrite('Filtered_RLS.wav', output_RLS, fs);

fprintf('Filtered signals saved as audio files.\n');

% Crop the figure and save as PDF
tightfig();
saveas(gcf, 'filtered_signals_comparison.pdf');
