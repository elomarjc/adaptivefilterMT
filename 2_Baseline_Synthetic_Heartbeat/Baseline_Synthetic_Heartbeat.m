%% Baseline Lub-Dub Synthetic Heartbeat
clear all;
close all;
clc;

% Input signal
[clean, fs] = audioread('clean_synthetic_heartbeat.wav'); 

% Time vector
t = 0:1/fs:(length(clean)-1)/fs;

% Ensure the heartbeat signal matches the length of the time vector
clean = clean(1:length(t));

% Generate noise at -20 dB SNR
signal_power = mean(clean.^2);   % Compute signal power
SNR_dB = -20;                     % Desired SNR in dB
SNR_linear = 10^(SNR_dB/10);   
noise_power = signal_power / SNR_linear; % Compute noise power
noise = sqrt(noise_power) * randn(size(t))'; % Generate Gaussian noise with the computed power

primary = clean + noise; % Combine signal and noise

% Calculate initial SNR
initial_SNR = 10 * log10(sum(clean.^2) / sum((clean - noise).^2));
fprintf('Initial SNR: %.2f dB\n', initial_SNR);

% Filter Parameters
N = length(primary); % Signal length
M = 12; % Filter order (adjusted to match first code)

% Pad noisy signal
padded_signal = [zeros(M-1, 1); primary]; 

%% LMS Filter
mu_LMS = 0.06; % Step size for LMS
w_LMS = zeros(M, 1); % Initialize filter weights
output_LMS = zeros(N, 1); % Filter output

for n = 1:N
    u_vect = padded_signal(n:n+M-1); % Current input vector
    e = noise(n) - w_LMS' * u_vect; % Error signal
    w_LMS = w_LMS + mu_LMS * e * u_vect; % Update weights
    output_LMS(n) = w_LMS' * u_vect; % Filtered output
end

filtered_signal_LMS = primary - output_LMS; 

%% NLMS Filter
mu_NLMS = 1; % Step size for NLMS (kept from first code)
w_NLMS = zeros(M, 1); % Initialize filter weights
output_NLMS = zeros(N, 1); % Filter output
Eps = 0.0001; % Stability constant (adjusted from first code)

for n = 1:N
    u_vect = padded_signal(n:n+M-1); % Current input vector
    mu_adapt = mu_NLMS / (Eps + norm(u_vect)^2); % Adaptive step size
    e = noise(n) - w_NLMS' * u_vect; % Error signal
    w_NLMS = w_NLMS + mu_adapt * e * u_vect; % Update weights
    output_NLMS(n) = w_NLMS' * u_vect; % Filtered output
end

filtered_signal_NLMS = primary - output_NLMS; 

%% RLS Filter
lambda = 1 - 1 / (0.1 * M); % Forgetting factor (adjusted from first code)
delta = 0.01; % Initialization constant (adjusted from first code)
P = (1 / delta) * eye(M); % Initialize inverse correlation matrix
w_RLS = zeros(M, 1); % Initialize filter weights
padded_signal = [sqrt(delta) * randn(M-1, 1); primary]; % Pad noisy signal
output_RLS = zeros(N, 1); % Filter output

for n = 1:N
    u_vect = padded_signal(n:n+M-1); % Current input vector
    PI = P * u_vect; % Intermediate calculation
    gain_k = PI / (lambda + u_vect' * PI); % Gain
    e = noise(n) - w_RLS' * u_vect; % Error signal
    w_RLS = w_RLS + gain_k * e; % Update weights
    P = P / lambda - gain_k * (u_vect' * P) / lambda; % Update P matrix
    output_RLS(n) = w_RLS' * u_vect; % Filtered output
end

filtered_signal_RLS = primary - output_RLS; 

%% Calculate SNR Improvement
filtered_SNR_LMS = 10 * log10(sum(clean.^2) / sum((clean - filtered_signal_LMS).^2));
filtered_SNR_NLMS = 10 * log10(sum(clean.^2) / sum((clean - filtered_signal_NLMS).^2));
filtered_SNR_RLS = 10 * log10(sum(clean.^2) / sum((clean - filtered_signal_RLS).^2));

fprintf('SNR after LMS Filter: %.2f dB\n', filtered_SNR_LMS);
fprintf('SNR after NLMS Filter: %.2f dB\n', filtered_SNR_NLMS);
fprintf('SNR after RLS Filter: %.2f dB\n', filtered_SNR_RLS);

%% Visualization of Results
figure;
subplot(5, 1, 1);
plot(clean);
title('Clean Signal');
xlabel('Sample Number');
ylabel('Amplitude');
ylim([-1.1 1.1]);
xlim([0 length(t)]);

subplot(5, 1, 2);
plot(primary);
title('Noisy Signal');
xlabel('Sample Number');
ylabel('Amplitude');
ylim([-5 5]);
xlim([0 length(t)]);

subplot(5, 1, 3);
plot(filtered_signal_LMS);
title('Filtered Signal (LMS)');
xlabel('Sample Number');
ylabel('Amplitude');
ylim([-1.1 1.1]);
xlim([0 length(t)]);

subplot(5, 1, 4);
plot(filtered_signal_NLMS);
title('Filtered Signal (NLMS)');
xlabel('Sample Number');
ylabel('Amplitude');
ylim([-1.1 1.1]);
xlim([0 length(t)]);

subplot(5, 1, 5);
plot(filtered_signal_RLS);
title('Filtered Signal (RLS)');
xlabel('Sample Number');
ylabel('Amplitude');
ylim([-1.1 1.1]);
xlim([0 length(t)]);

%% Export filtered signals

% Normalize filtered signals    (To prevent "Warning: Data clipped when writing file") 
filtered_signal_LMS = filtered_signal_LMS / max(abs(filtered_signal_LMS));
filtered_signal_NLMS = filtered_signal_NLMS / max(abs(filtered_signal_NLMS));
filtered_signal_RLS = filtered_signal_RLS / max(abs(filtered_signal_RLS));
primary = primary / max(abs(primary));

% Save Filtered Signals
audiowrite('Filtered_LMS.wav', filtered_signal_LMS, fs);
audiowrite('Filtered_NLMS.wav', filtered_signal_NLMS, fs);
audiowrite('Filtered_RLS.wav', filtered_signal_RLS, fs);
audiowrite('noisy_synthetic_heartbeat.wav', primary, fs);

fprintf('Filtered signals saved as audio files.\n\n');

% Tuning Parementers for tables
fprintf('Sampling rate: %f Hz\n', fs);
fprintf('Filter order (LMS, NLMS, RLS): %f \n', M);
fprintf('Step size (LMS): %f \n', mu_LMS);
fprintf('Step size (NLMS): %f \n', mu_NLMS);
fprintf('Forgetting factor (RLS): %f \n', lambda);

% Crop the figure and save as PDF
tightfig();
saveas(gcf, 'Synthetic_heartbeat.pdf');

% Function for Tightening Figure Layout
function tightfig()
    set(gcf, 'Units', 'Inches');
    pos = get(gcf, 'Position');
    set(gcf, 'PaperUnits', 'Inches');
    set(gcf, 'PaperSize', [pos(3) pos(4)]);
    set(gcf, 'PaperPosition', [0 0 pos(3) pos(4)]);
end