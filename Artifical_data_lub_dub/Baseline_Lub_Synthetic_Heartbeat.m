%% Baseline Lub-Dub Synthetic Heartbeat
clear all;
close all;
clc;

% *************************************************************************
% Simulate heart sound
%% Parameters for Heartbeat Sound
fs = 1000; % Sampling frequency (in Hz)
duration = 15; % Duration of the signal (in seconds)
t = 0:1/fs:duration; % Time vector
heartbeat_bpm = 60; % Heartbeat rate (60 beats per minute)
heartbeat_period = 60 / heartbeat_bpm; % Time between beats in seconds

% Define the "lub-dub" structure of a single heartbeat
lub_duration = 0.1; % Duration of "lub" sound (in seconds)
dub_duration = 0.15; % Duration of "dub" sound (in seconds)
pause_duration = heartbeat_period - (lub_duration + dub_duration); % Pause between heartbeats

% Create the "lub-dub" pattern using Gaussian pulses
t_lub = 0:1/fs:lub_duration; % Time for "lub" sound
t_dub = 0:1/fs:dub_duration; % Time for "dub" sound
lub_sound = gauspuls(t_lub, 150, 0.5); % Simulated "lub" sound with Gaussian pulse
dub_sound = gauspuls(t_dub, 80, 0.5);  % Simulated "dub" sound with lower frequency Gaussian pulse

% Create one full heartbeat cycle
heartbeat_cycle = [lub_sound, ...
                   zeros(1, round(pause_duration*fs/2)), ...
                   dub_sound, ...
                   zeros(1, round(pause_duration*fs/2))];

% Calculate how many cycles fit in the signal duration
num_heartbeats = floor(duration / heartbeat_period);

% Replicate the heartbeat cycle to fill the total duration
signal = repmat(heartbeat_cycle, 1, num_heartbeats);

% Ensure the heartbeat signal matches the length of the time vector
signal = signal(1:length(t));

% Normalize the heartbeat signal
signal = signal(:); % Convert to row vector

% Save the synthetic heartbeat signal to a .wav file (optional)
audiowrite('clean_synthetic_heartbeat.wav', signal, fs);

%% Generate noise at -20 dB SNR
signal_power = mean(signal.^2);   % Compute signal power
SNR_dB = -20;                     % Desired SNR in dB
SNR_linear = 10^(SNR_dB/10);   
noise_power = signal_power / SNR_linear; % Compute noise power
noise = sqrt(noise_power) * randn(size(t))'; % Generate Gaussian noise with the computed power

noisy_signal = signal + noise; % Combine signal and noise

% Calculate initial SNR
initial_SNR = 10 * log10(sum(signal.^2) / sum(noise.^2));
fprintf('Initial SNR: %.2f dB\n', initial_SNR);

% Filter Parameters
N = length(noisy_signal); % Signal length
M = 12; % Filter order (adjusted to match first code)

% Pad noisy signal
padded_signal = [zeros(M-1, 1); noisy_signal]; 
%% *************************************************************************
% LMS Filter
mu_LMS = 0.06; % Step size for LMS (adjusted from first code)
w_LMS = zeros(M, 1); % Initialize filter weights
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
w_NLMS = zeros(M, 1); % Initialize filter weights
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
lambda = 0.1; % Forgetting factor (adjusted from first code)
delta = 0.01; % Initialization constant (adjusted from first code)
P = (1 / delta) * eye(M); % Initialize inverse correlation matrix
w_RLS = zeros(M, 1); % Initialize filter weights
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
% Ensure their lengths match signal:
output_LMS = output_LMS(1:length(signal));
output_NLMS = output_NLMS(1:length(signal));
output_RLS = output_RLS(1:length(signal));

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
ylim([-1.1 1.1]);
xlim([0 15000]);

subplot(5, 1, 2);
plot(noisy_signal);
title('Noisy Signal');
xlabel('Sample Number');
ylabel('Amplitude');
ylim([-5 5]);
xlim([0 15000]);

subplot(5, 1, 3);
plot(output_LMS);
title('Filtered Signal (LMS)');
xlabel('Sample Number');
ylabel('Amplitude');
ylim([-1.1 1.1]);
xlim([0 15000]);

subplot(5, 1, 4);
plot(output_NLMS);
title('Filtered Signal (NLMS)');
xlabel('Sample Number');
ylabel('Amplitude');
ylim([-1.1 1.1]);
xlim([0 15000]);

subplot(5, 1, 5);
plot(output_RLS);
title('Filtered Signal (RLS)');
xlabel('Sample Number');
ylabel('Amplitude');
ylim([-1.1 1.1]);
xlim([0 15000]);

% *************************************************************************

% Normalize filtered signals                             (To prevent "Warning: Data clipped when writing file") 
output_LMS = output_LMS / max(abs(output_LMS));
output_NLMS = output_NLMS / max(abs(output_NLMS));
output_RLS = output_RLS / max(abs(output_RLS));
noisy_signal = noisy_signal / max(abs(noisy_signal));

% Save Filtered Signals
audiowrite('Filtered_LMS.wav', output_LMS, fs);
audiowrite('Filtered_NLMS.wav', output_NLMS, fs);
audiowrite('Filtered_RLS.wav', output_RLS, fs);
audiowrite('noisy_synthetic_heartbeat.wav', noisy_signal, fs);

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


% sound(signal, fs) %play clean audio 
% sound(noise, fs) %play noise 
% sound(noisy_signal, fs) %play clean + noisy audio 

%% Function for Tightening Figure Layout
function tightfig()
    set(gcf, 'Units', 'Inches');
    pos = get(gcf, 'Position');
    set(gcf, 'PaperUnits', 'Inches');
    set(gcf, 'PaperSize', [pos(3) pos(4)]);
    set(gcf, 'PaperPosition', [0 0 pos(3) pos(4)]);
end